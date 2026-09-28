"""Pick the D3QN checkpoint to ship, using validation seeds only.

    python -m smartscan.rl.select checkpoints/a.pt checkpoints/b.pt ... [--seeds 5] [--install]

Each candidate is scored on every scenario × validation seeds 500..500+n (disjoint
from training seeds 0–99 and test seeds 100+). With --install, the winner is copied
to checkpoints/d3qn_best.pt.
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from smartscan.rl.env import DEFAULT_SCENARIOS
from smartscan.sim.world import ROOT


def _job(args):
    import torch

    torch.set_num_threads(1)
    from smartscan.rl.policy import D3QNScheduler
    from smartscan.runner import run_episode
    from smartscan.sim.world import build_world

    ckpt, sc, seed = args
    m, _, _ = run_episode(build_world(sc, seed=seed), D3QNScheduler(ckpt=ckpt, device="cpu"), seed=seed)
    return {"ckpt": str(ckpt), "scenario": sc, "seed": seed, "pd": m["pd"], "pd_weighted": m["pd_weighted"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--install", action="store_true")
    a = ap.parse_args(argv)
    jobs = [(c, sc, s) for c in a.ckpts for sc in DEFAULT_SCENARIOS for s in range(500, 500 + a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        df = pd.DataFrame(list(ex.map(_job, jobs)))
    summ = df.groupby("ckpt")[["pd", "pd_weighted"]].agg(["mean", "std"]).sort_values(("pd_weighted", "mean"),
                                                                                       ascending=False)
    print(summ.round(4))
    out = ROOT / "results" / "d3qn_selection.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    best = summ.index[0]
    print(f"best on validation: {best}  (pd_weighted {summ.iloc[0][('pd_weighted', 'mean')]:.4f})")
    if a.install and Path(best).resolve() != (ROOT / "checkpoints" / "d3qn_best.pt").resolve():
        shutil.copy(best, ROOT / "checkpoints" / "d3qn_best.pt")
        print("installed as checkpoints/d3qn_best.pt")
    return best


if __name__ == "__main__":
    main()
