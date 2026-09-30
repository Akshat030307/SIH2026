"""Learned deinterleaver (PDF §Deinterleaving): representation learning + density clustering.

A small Transformer encoder reads windows of consecutive PDWs and maps each
pulse to a unit-norm embedding. Pulses from the same emitter should land close
together even when the emitter hops frequency or its RF/PW/AOA overlap with a
neighbour's.
  * Inputs: scaled RF/PW/AOA/amplitude plus multi-scale Fourier features of
    the time of arrival. Attention can then pick up PRI periodicity (pulses
    that sit a whole number of PRIs apart share phase).
  * Training: supervised contrastive loss (Khosla et al. 2020) on labelled
    windows from the simulator (or the Turing dataset).
  * Inference: embed overlapping windows, concatenate the embeddings with
    the scaled static features, and run HDBSCAN.

Train:  python -m smartscan.deinterleave.learned train
Eval:   python -m smartscan.deinterleave.learned eval
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from smartscan.deinterleave.dataset import N_FREQ, WINDOW, cached_training_set, pulse_inputs
from smartscan.sim.world import ROOT
from smartscan.util import md_table

CKPT = ROOT / "checkpoints" / "deinterleaver.pt"
PERIODS = np.geomspace(20e-6, 20e-3, N_FREQ).astype(np.float32)
TWO_PI_OVER_PERIODS = (2 * np.pi / PERIODS).astype(np.float32)


class PulseEncoder(nn.Module):
    def __init__(self, d_in: int = 5 + 2 * N_FREQ, d: int = 128, layers: int = 3, heads: int = 4, d_out: int = 32):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.05, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, d_out)

    def forward(self, x, pad_mask=None):
        h = self.enc(self.inp(x), src_key_padding_mask=pad_mask)
        return F.normalize(self.out(h), dim=-1)


def supcon_loss(z: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss per window. z: (B, L, D); y: (B, L); mask: (B, L) valid."""
    sim = z @ z.transpose(1, 2) / tau  # (B, L, L)
    L = z.shape[1]
    eye = torch.eye(L, dtype=torch.bool, device=z.device)[None]
    valid = mask[:, :, None] & mask[:, None, :] & ~eye
    pos = (y[:, :, None] == y[:, None, :]) & valid
    sim = sim.masked_fill(~valid, -1e9)
    logp = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
    npos = pos.sum(-1)
    loss = -(logp * pos).sum(-1) / npos.clamp(min=1)
    keep = mask & (npos > 0)
    return loss[keep].mean()


def train(epochs: int = 12, n_seeds: int = 120, batch: int = 32, lr: float = 3e-4, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t = time.time()
    X, Y, M = cached_training_set(n_seeds)
    print(f"{len(X)} windows ready in {time.time() - t:.0f}s; training on {device}")
    model = PulseEncoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(X) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    rng = np.random.default_rng(1)
    for ep in range(epochs):
        perm = rng.permutation(len(X))
        tot = 0.0
        for k in range(len(X) // batch):
            b = perm[k * batch:(k + 1) * batch]
            xb, yb, mb = (torch.from_numpy(a[b]).to(device) for a in (X, Y, M))
            loss = supcon_loss(model(xb, ~mb), yb, mb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"epoch {ep + 1}/{epochs} loss {tot / (len(X) // batch):.4f}")
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT)
    print(f"saved {CKPT}")
    return model


def load(device: str | None = None) -> PulseEncoder:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = PulseEncoder()
    m.load_state_dict(torch.load(CKPT, map_location=device))
    return m.to(device).eval()


@torch.inference_mode()
def embed(
    model: PulseEncoder,
    pdws: np.ndarray,
    window: int = WINDOW,
    stride: int = 192,
    batch_size: int = 64,
) -> np.ndarray:
    n = len(pdws)
    d_out = model.out.out_features
    if n == 0:
        return np.zeros((0, d_out), dtype=np.float32)

    dev = next(model.parameters()).device

    if n < window:
        inp = torch.from_numpy(pulse_inputs(pdws))[None].to(dev, non_blocking=True)
        z = model(inp)[0].cpu().numpy()
        norm = np.linalg.norm(z, axis=1, keepdims=True)
        return z / np.maximum(norm, 1e-9)

    starts = list(range(0, max(n - window, 0) + 1, stride))
    if starts[-1] + window < n:
        starts.append(n - window)

    if dev.type == "cuda":
        a_rad = torch.deg2rad(torch.from_numpy(pdws["aoa"].astype(np.float32)).to(dev, non_blocking=True))
        rf = torch.from_numpy(pdws["rf"].astype(np.float32)).to(dev, non_blocking=True)
        pw = torch.from_numpy(pdws["pw"].astype(np.float32)).to(dev, non_blocking=True)
        amp = torch.from_numpy(pdws["amp"].astype(np.float32)).to(dev, non_blocking=True)
        toa = torch.from_numpy(pdws["toa"].astype(np.float32)).to(dev, non_blocking=True)

        static = torch.stack([
            (rf - 10e9) / 5e9,
            torch.log10(torch.clamp(pw, min=1e-9) * 1e6),
            torch.sin(a_rad),
            torch.cos(a_rad),
            (amp - 30.0) / 15.0,
        ], dim=1)
        two_pi_periods = torch.from_numpy(TWO_PI_OVER_PERIODS).to(dev, non_blocking=True)

        acc = torch.zeros((n, d_out), dtype=torch.float32, device=dev)
        cnt = torch.zeros((n, 1), dtype=torch.float32, device=dev)

        for i in range(0, len(starts), batch_size):
            b_starts = starts[i : i + batch_size]
            stat_b = torch.stack([static[s : s + window] for s in b_starts])
            toa_b = torch.stack([toa[s : s + window] - toa[s] for s in b_starts])
            ph_b = toa_b.unsqueeze(-1) * two_pi_periods
            bx = torch.cat([stat_b, torch.sin(ph_b), torch.cos(ph_b)], dim=-1)
            z_b = model(bx)
            for b_idx, s in enumerate(b_starts):
                acc[s : s + window] += z_b[b_idx]
                cnt[s : s + window] += 1.0

        z = acc / torch.clamp(cnt, min=1.0)
        norm = torch.linalg.norm(z, dim=1, keepdim=True)
        z = z / torch.clamp(norm, min=1e-9)
        return z.cpu().numpy()
    else:
        a = np.deg2rad(pdws["aoa"])
        static = np.stack([
            (pdws["rf"] - 10e9) / 5e9,
            np.log10(np.maximum(pdws["pw"], 1e-9) * 1e6),
            np.sin(a),
            np.cos(a),
            (pdws["amp"] - 30.0) / 15.0,
        ], axis=1).astype(np.float32)
        toa = pdws["toa"]

        acc = np.zeros((n, d_out), dtype=np.float32)
        cnt = np.zeros(n, dtype=np.float32)

        for i in range(0, len(starts), batch_size):
            b_starts = starts[i : i + batch_size]
            B = len(b_starts)
            batch_x = np.empty((B, window, 5 + 2 * N_FREQ), dtype=np.float32)
            for b_idx, s in enumerate(b_starts):
                batch_x[b_idx, :, :5] = static[s : s + window]
                ph = (toa[s : s + window] - toa[s])[:, None] * TWO_PI_OVER_PERIODS[None, :]
                batch_x[b_idx, :, 5 : 5 + N_FREQ] = np.sin(ph)
                batch_x[b_idx, :, 5 + N_FREQ :] = np.cos(ph)

            bx_t = torch.from_numpy(batch_x).to(dev)
            z_b = model(bx_t).cpu().numpy()
            for b_idx, s in enumerate(b_starts):
                acc[s : s + window] += z_b[b_idx]
                cnt[s : s + window] += 1.0

        z = acc / np.maximum(cnt, 1.0)[:, None]
        norm = np.linalg.norm(z, axis=1, keepdims=True)
        return z / np.maximum(norm, 1e-9)


def learned_deinterleave(
    model,
    pdws: np.ndarray,
    min_cluster_size: int = 30,
    w_emb: float = 10.0,
    w_static: float = 0.20,
    core_dist_n_jobs: int = -1,
) -> np.ndarray:
    """HDBSCAN on [w_emb·embedding, w_static·scaled static features]."""
    import hdbscan

    from smartscan.deinterleave.classical import _scaled

    if len(pdws) < min_cluster_size:
        return np.zeros(len(pdws), dtype=int)

    z = embed(model, pdws)
    feats = np.concatenate([z * w_emb, w_static * _scaled(pdws) / 4.0], axis=1)
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        core_dist_n_jobs=core_dist_n_jobs,
    ).fit_predict(feats)


def evaluate(seeds=range(1000, 1010), scenarios=("S7_colocated", "S2_dense")) -> dict:
    import pandas as pd

    from smartscan.deinterleave.classical import clustering_scores, dbscan_deinterleave, hdbscan_deinterleave
    from smartscan.deinterleave.data import busiest_channels, simulate_stream

    model = load()
    rows = []
    for sc in scenarios:
        for s in seeds:
            p = simulate_stream(sc, seed=s, duration=3.0, channels=busiest_channels(sc, s, k=6))
            for name, fn in [("dbscan", dbscan_deinterleave), ("hdbscan", hdbscan_deinterleave),
                             ("learned", lambda q: learned_deinterleave(model, q))]:
                t = time.time()
                lab = fn(p)
                rows.append(dict(scenario=sc, seed=s, method=name, sec=time.time() - t,
                                 **clustering_scores(p["emitter"], lab)))
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "deinterleave"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    summ = df.groupby(["scenario", "method"])[["ari", "ami", "v_measure", "homogeneity", "completeness", "n_true",
                                                "n_pred", "sec"]].mean().round(3)
    (out / "summary.md").write_text("# Deinterleaving (held-out seeds)\n\n" + md_table(summ) + "\n")
    print(summ)
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=120)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.cmd == "train":
        train(a.epochs, a.seeds)
    else:
        evaluate()


if __name__ == "__main__":
    main()
