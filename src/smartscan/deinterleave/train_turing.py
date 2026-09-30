"""Train / Fine-tune the PulseEncoder Transformer on the Turing Synthetic Radar Dataset.

Extracts windows from the 30 train_scan files and optimizes the model using
Supervised Contrastive Loss (supcon_loss).

Saves to:
  checkpoints/deinterleaver_turing.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from smartscan.deinterleave.dataset import WINDOW, make_windows
from smartscan.deinterleave.learned import PulseEncoder, supcon_loss
from smartscan.deinterleave.turing import load as load_tsrd, local_files
from smartscan.sim.world import ROOT

CKPT_OUT = ROOT / "checkpoints" / "deinterleaver_turing.pt"
CKPT_INIT = ROOT / "checkpoints" / "deinterleaver.pt"


def build_turing_dataset(split: str = "train_scan", n_windows_per_file: int = 50, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = local_files(split)
    if not files:
        raise FileNotFoundError(f"No files found in data/turing/{split}")

    rng = np.random.default_rng(seed)
    all_windows = []
    print(f"Extracting {n_windows_per_file} windows per file across {len(files)} files in {split}...")
    t0 = time.time()

    for idx, path in enumerate(files):
        pdws = load_tsrd(path)
        if len(pdws) < WINDOW:
            continue
        windows = make_windows(pdws, rng, n=n_windows_per_file, window=WINDOW)
        all_windows.extend(windows)

    X = np.stack([w[0] for w in all_windows], axis=0).astype(np.float32)
    Y = np.stack([w[1] for w in all_windows], axis=0).astype(np.int64)
    M = np.ones((len(X), WINDOW), dtype=bool)

    print(f"Prepared {len(X)} windows from {split} in {time.time() - t0:.1f}s (Shape: {X.shape})")
    return X, Y, M


def train_turing(
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 6e-5,
    n_windows_per_file: int = 60,
    val_windows_per_file: int = 10,
    from_pretrained: bool = True,
    seed: int | None = None,
    device: str | None = None,
) -> PulseEncoder:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_seed = seed if seed is not None else int(time.time()) % 10000
    print(f"\n==================================================")
    print(f"  Training PulseEncoder on Turing Radar Dataset   ")
    print(f"  Device: {device} | Epochs: {epochs} | Batch: {batch_size} | Seed: {run_seed}")
    print(f"==================================================")

    # 1. Training set
    X_train, Y_train, M_train = build_turing_dataset(
        split="train_scan", n_windows_per_file=n_windows_per_file, seed=run_seed
    )

    # 2. Validation set from val_scan
    try:
        X_val, Y_val, M_val = build_turing_dataset(
            split="val_scan", n_windows_per_file=val_windows_per_file, seed=123
        )
    except Exception as e:
        print(f"Note: Could not build validation set from val_scan ({e}); using subset of train")
        val_n = min(1000, len(X_train) // 10)
        X_val, Y_val, M_val = X_train[-val_n:], Y_train[-val_n:], M_train[-val_n:]
        X_train, Y_train, M_train = X_train[:-val_n], Y_train[:-val_n], M_train[:-val_n]

    model = PulseEncoder().to(device)
    init_path = CKPT_OUT if (CKPT_OUT.exists() and from_pretrained) else CKPT_INIT
    if init_path.exists() and from_pretrained:
        print(f"Initializing weights from: {init_path.name}")
        model.load_state_dict(torch.load(init_path, map_location=device))
    else:
        print("Initializing weights from scratch")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_train_batches = len(X_train) // batch_size
    n_val_batches = max(1, len(X_val) // batch_size)
    total_steps = epochs * n_train_batches
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=total_steps)
    rng = np.random.default_rng(42)

    best_val_loss = float("inf")
    history = []
    t_start = time.time()

    print(f"\n{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Time':^8} | {'Status'}")
    print("-" * 55)

    for ep in range(epochs):
        ep_t0 = time.time()
        perm = rng.permutation(len(X_train))
        train_loss = 0.0
        model.train()

        for k in range(n_train_batches):
            b_idx = perm[k * batch_size : (k + 1) * batch_size]
            xb = torch.from_numpy(X_train[b_idx]).to(device)
            yb = torch.from_numpy(Y_train[b_idx]).to(device)
            mb = torch.from_numpy(M_train[b_idx]).to(device)

            z = model(xb, ~mb)
            loss = supcon_loss(z, yb, mb)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / n_train_batches

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for k in range(n_val_batches):
                s = k * batch_size
                xb = torch.from_numpy(X_val[s : s + batch_size]).to(device)
                yb = torch.from_numpy(Y_val[s : s + batch_size]).to(device)
                mb = torch.from_numpy(M_val[s : s + batch_size]).to(device)
                z = model(xb, ~mb)
                val_loss += supcon_loss(z, yb, mb).item()

        avg_val_loss = val_loss / n_val_batches
        ep_time = time.time() - ep_t0

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CKPT_OUT)
            status_str = f"★ Best ({best_val_loss:.4f})"
        else:
            status_str = ""

        history.append({
            "epoch": ep + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "time_s": round(ep_time, 2),
        })

        print(f"{ep + 1:^7d} | {avg_train_loss:^12.4f} | {avg_val_loss:^10.4f} | {ep_time:^6.1f}s | {status_str}")

    total_time = time.time() - t_start
    print("-" * 55)
    print(f"Training completed in {total_time:.1f}s | Best Val Loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {CKPT_OUT}")
    return model


def main():
    ap = argparse.ArgumentParser(description="Train/fine-tune PulseEncoder on TSRD")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--windows-per-file", type=int, default=100)
    ap.add_argument("--scratch", action="store_true", help="Train from scratch rather than fine-tune")
    args = ap.parse_args()

    train_turing(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_windows_per_file=args.windows_per_file,
        from_pretrained=not args.scratch,
    )


if __name__ == "__main__":
    main()
