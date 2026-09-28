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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from smartscan.sim.world import ROOT
from smartscan.util import md_table

CKPT = ROOT / "checkpoints" / "deinterleaver.pt"
WINDOW = 256
N_FREQ = 16


def pulse_inputs(pdws: np.ndarray) -> np.ndarray:
    """(N, 5 + 2·N_FREQ) float32 inputs. Time features are relative to the first pulse."""
    a = np.deg2rad(pdws["aoa"])
    toa = pdws["toa"] - pdws["toa"][0]
    # periods from 20 µs to ~20 ms, log-spaced
    periods = np.geomspace(20e-6, 20e-3, N_FREQ)
    ph = 2 * np.pi * toa[:, None] / periods[None, :]
    static = np.stack([
        (pdws["rf"] - 10e9) / 5e9,
        np.log10(np.maximum(pdws["pw"], 1e-9) * 1e6),
        np.sin(a), np.cos(a),
        (pdws["amp"] - 30.0) / 15.0,
    ], axis=1)
    return np.concatenate([static, np.sin(ph), np.cos(ph)], axis=1).astype(np.float32)


class PulseEncoder(nn.Module):
    def __init__(self, d_in: int = 5 + 2 * N_FREQ, d: int = 128, layers: int = 3, heads: int = 4, d_out: int = 32):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.05, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
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


# ------------------------------------------------------------------ data
def make_windows(pdws: np.ndarray, rng, n: int, window: int = WINDOW):
    x = pulse_inputs(pdws)
    y = pdws["emitter"].astype(np.int64)
    out = []
    for _ in range(n):
        if len(x) <= window:
            s = 0
        else:
            s = int(rng.integers(0, len(x) - window))
        xs, ys = x[s:s + window].copy(), y[s:s + window]
        xs[:, 5:] = pulse_inputs(pdws[s:s + window])[:, 5:]  # time features relative to the window start
        out.append((xs, ys))
    return out


def build_training_set(seeds, scenarios=("S7_colocated", "S2_dense"), duration=1.5, per_stream=40):
    from smartscan.deinterleave.data import busiest_channels, simulate_stream

    rng = np.random.default_rng(0)
    data = []
    for sc in scenarios:
        for s in seeds:
            ch = busiest_channels(sc, s, k=6)
            p = simulate_stream(sc, seed=s, duration=duration, channels=ch, t0=float(rng.uniform(0, 30)))
            if len(p) < 64:
                continue
            data += make_windows(p, rng, per_stream)
    return data


def collate(batch, device):
    L = max(len(x) for x, _ in batch)
    B = len(batch)
    X = np.zeros((B, L, batch[0][0].shape[1]), np.float32)
    Y = np.full((B, L), -1, np.int64)
    M = np.zeros((B, L), bool)
    for i, (x, y) in enumerate(batch):
        X[i, :len(x)], Y[i, :len(y)], M[i, :len(x)] = x, y, True
    return (torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device), torch.from_numpy(M).to(device))


def train(epochs: int = 12, n_seeds: int = 120, batch: int = 32, lr: float = 3e-4, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t = time.time()
    data = build_training_set(range(n_seeds))
    print(f"{len(data)} windows built in {time.time() - t:.0f}s; training on {device}")
    model = PulseEncoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(data) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    rng = np.random.default_rng(1)
    for ep in range(epochs):
        perm = rng.permutation(len(data))
        tot = 0.0
        for k in range(len(data) // batch):
            X, Y, M = collate([data[i] for i in perm[k * batch:(k + 1) * batch]], device)
            loss = supcon_loss(model(X, ~M), Y, M)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"epoch {ep + 1}/{epochs} loss {tot / (len(data) // batch):.4f}")
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT)
    print(f"saved {CKPT}")
    return model


def load(device: str | None = None) -> PulseEncoder:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = PulseEncoder()
    m.load_state_dict(torch.load(CKPT, map_location=device))
    return m.to(device).eval()


@torch.no_grad()
def embed(model: PulseEncoder, pdws: np.ndarray, window: int = WINDOW, stride: int = 192) -> np.ndarray:
    dev = next(model.parameters()).device
    n = len(pdws)
    acc = np.zeros((n, model.out.out_features), np.float32)
    cnt = np.zeros(n, np.float32)
    starts = list(range(0, max(n - window, 0) + 1, stride))
    if starts[-1] + window < n:
        starts.append(n - window)
    for s in starts:
        seg = pdws[s:s + window]
        z = model(torch.from_numpy(pulse_inputs(seg))[None].to(dev))[0].cpu().numpy()
        acc[s:s + len(seg)] += z
        cnt[s:s + len(seg)] += 1
    z = acc / np.maximum(cnt, 1)[:, None]
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)


def learned_deinterleave(model, pdws: np.ndarray, min_cluster_size: int = 15, w_static: float = 0.5) -> np.ndarray:
    import hdbscan

    from smartscan.deinterleave.classical import _scaled

    z = embed(model, pdws)
    feats = np.concatenate([z * 4.0, w_static * _scaled(pdws) / 4.0], axis=1)
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5).fit_predict(feats)


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
    a = ap.parse_args(argv)
    if a.cmd == "train":
        train(a.epochs, a.seeds)
    else:
        evaluate()


if __name__ == "__main__":
    main()
