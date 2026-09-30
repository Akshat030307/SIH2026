"""Benchmark classical and learned deinterleavers on Turing Synthetic Radar Dataset (TSRD).

Evaluates:
  * DBSCAN
  * HDBSCAN
  * Learned Transformer (sim-pretrained: checkpoints/deinterleaver.pt)
  * Learned Transformer (Turing-fine-tuned: checkpoints/deinterleaver_turing.pt, if present)

Outputs:
  * results/deinterleave/turing_results.csv
  * results/deinterleave/turing_summary.md
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from smartscan.deinterleave.classical import clustering_scores, dbscan_deinterleave, hdbscan_deinterleave
from smartscan.deinterleave.learned import PulseEncoder, learned_deinterleave, load as load_sim_model
from smartscan.deinterleave.turing import load as load_tsrd, local_files
from smartscan.sim.world import ROOT
from smartscan.util import md_table


def evaluate_tsrd(
    split: str = "test_scan",
    max_pulses: int | None = 5000,
    n_files: int | None = 25,
    stride: int = 1,
    device: str | None = None,
) -> pd.DataFrame:
    files = local_files(split)
    if not files:
        raise FileNotFoundError(f"No .h5 files found in data/turing/{split}")

    if n_files is not None and 0 < n_files < len(files):
        indices = np.linspace(0, len(files) - 1, n_files, dtype=int)
        files = [files[i] for i in indices]

    out_dir = ROOT / "results" / "deinterleave"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    models: dict[str, PulseEncoder] = {}
    try:
        models["learned_sim"] = load_sim_model(device=device)
    except Exception as e:
        print(f"Warning: could not load sim model: {e}")

    turing_ckpt = ROOT / "checkpoints" / "deinterleaver_turing.pt"
    if turing_ckpt.exists():
        try:
            import torch
            m = PulseEncoder()
            m.load_state_dict(torch.load(turing_ckpt, map_location=device or "cpu"))
            models["learned_turing"] = m.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
        except Exception as e:
            print(f"Warning: could not load turing-trained model: {e}")

    rows = []
    print(f"Evaluating {len(files)} files from {split} (max_pulses={max_pulses})...")

    for f_idx, path in enumerate(files):
        pdws = load_tsrd(path)
        if max_pulses is not None and len(pdws) > max_pulses:
            pdws = pdws[:max_pulses:stride]

        n_true = len(np.unique(pdws["emitter"]))
        print(f"\n[{f_idx + 1}/{len(files)}] {path.name}: {len(pdws)} pulses, {n_true} true emitters")

        # 1. DBSCAN
        t0 = time.time()
        lab_db = dbscan_deinterleave(pdws)
        dt_db = time.time() - t0
        scores_db = clustering_scores(pdws["emitter"], lab_db)
        rows.append(dict(file=path.name, method="dbscan", pulses=len(pdws), sec=dt_db, **scores_db))
        print(f"  dbscan:         ARI={scores_db['ari']:.3f} AMI={scores_db['ami']:.3f} V={scores_db['v_measure']:.3f} "
              f"Pred={scores_db['n_pred']} ({dt_db * 1e3:.1f}ms)")

        # 2. HDBSCAN
        t0 = time.time()
        lab_hdb = hdbscan_deinterleave(pdws)
        dt_hdb = time.time() - t0
        scores_hdb = clustering_scores(pdws["emitter"], lab_hdb)
        rows.append(dict(file=path.name, method="hdbscan", pulses=len(pdws), sec=dt_hdb, **scores_hdb))
        print(f"  hdbscan:        ARI={scores_hdb['ari']:.3f} AMI={scores_hdb['ami']:.3f} V={scores_hdb['v_measure']:.3f} "
              f"Pred={scores_hdb['n_pred']} ({dt_hdb * 1e3:.1f}ms)")

        # 3. Learned models
        for m_name, model in models.items():
            t0 = time.time()
            lab_lrn = learned_deinterleave(model, pdws)
            dt_lrn = time.time() - t0
            scores_lrn = clustering_scores(pdws["emitter"], lab_lrn)
            rows.append(dict(file=path.name, method=m_name, pulses=len(pdws), sec=dt_lrn, **scores_lrn))
            print(f"  {m_name:15s} ARI={scores_lrn['ari']:.3f} AMI={scores_lrn['ami']:.3f} V={scores_lrn['v_measure']:.3f} "
                  f"Pred={scores_lrn['n_pred']} ({dt_lrn * 1e3:.1f}ms)")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "turing_results.csv", index=False)

    # Compute summary aggregated by method
    metric_cols = ["ari", "ami", "v_measure", "homogeneity", "completeness", "n_true", "n_pred", "noise_frac", "sec"]
    summary = df.groupby("method")[metric_cols].mean().round(3)

    summary_md = (
        f"# Turing Synthetic Radar Dataset Benchmark ({split})\n\n"
        f"Evaluated on {len(files)} pulse train recordings (subset of {max_pulses or 'all'} pulses each).\n\n"
        + md_table(summary)
        + "\n"
    )
    (out_dir / "turing_summary.md").write_text(summary_md)
    print("\n" + "=" * 60)
    print(f"Summary on {split}:")
    print(summary)
    print(f"\nSaved results to:\n  {out_dir / 'turing_results.csv'}\n  {out_dir / 'turing_summary.md'}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Benchmark deinterleaving on TSRD")
    ap.add_argument("--split", default="test_scan", choices=["test_scan", "train_scan", "val_scan"])
    ap.add_argument("--n-files", type=int, default=25, help="Number of files to evaluate (0 for all)")
    ap.add_argument("--max-pulses", type=int, default=5000, help="Max pulses per file (None for full)")
    ap.add_argument("--full", action="store_true", help="Run on full pulse train")
    args = ap.parse_args()

    max_p = None if args.full else args.max_pulses
    n_f = None if args.n_files == 0 else args.n_files
    evaluate_tsrd(split=args.split, max_pulses=max_p, n_files=n_f)


if __name__ == "__main__":
    main()
