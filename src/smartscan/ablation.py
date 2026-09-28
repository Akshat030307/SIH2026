"""Component ablation of the smart scheduler.

    python -m smartscan.ablation --seeds 5
"""

from __future__ import annotations

import argparse

from smartscan.eval import run_benchmark
from smartscan.schedulers.registry import ABLATIONS
from smartscan.sim.world import ROOT, list_scenarios
from smartscan.util import md_table

WHAT = {
    "smart": "full scheduler",
    "smart-no_lock": "no period lock (no predicted-beam dwells)",
    "smart-no_acquire": "no acquisition revisits (locks form only by chance)",
    "smart-no_lock_no_acquire": "tracker only: bandit exploration",
    "smart-sweep_explore": "exploration by linear sweep instead of D-UCB (no jitter)",
    "smart-random_explore": "exploration uniformly at random instead of D-UCB",
    "smart-no_jitter": "D-UCB exploration without dwell jitter",
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args(argv)
    names = ["smart", *ABLATIONS]
    df = run_benchmark(list_scenarios(), names, range(100, 100 + a.seeds), a.workers)
    out = ROOT / "results" / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    piv = df.pivot_table(index="scheduler", columns="scenario", values="pd_weighted", aggfunc="mean")
    piv["mean"] = piv.mean(axis=1)
    piv = piv.loc[names]
    piv.insert(0, "variant", [WHAT[n] for n in piv.index])
    ttfi = df.groupby("scheduler")["ttfi_censored_mean"].mean().loc[names]
    piv["TTFI [s]"] = ttfi
    md = ("# Smart-scheduler ablation (threat-weighted Pd, held-out seeds)\n\n"
          + md_table(piv.round(3)) + "\n")
    (out / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
