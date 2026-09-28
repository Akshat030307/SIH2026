"""Benchmark runner: scenario × scheduler × seed → results table and plots.

    python -m smartscan.eval --scenarios all --schedulers sweep,random,round_robin,bandit,smart --seeds 5
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from smartscan.runner import run_episode
from smartscan.schedulers.registry import make_scheduler
from smartscan.sim.world import ROOT, build_world, list_scenarios

HEADLINE = [
    ("pd", "Pd", "{:.3f}"),
    ("pd_weighted", "Pd (threat-wtd)", "{:.3f}"),
    ("intercept_rate", "Intercept rate [emitters/s]", "{:.3f}"),
    ("ttfi_censored_mean", "TTFI [s]", "{:.2f}"),
    ("intercept_time_error_ms", "Intercept time err [ms]", "{:.1f}"),
    ("pfa", "Pfa", "{:.3f}"),
    ("correct_predictions", "Correct predictions", "{:.3f}"),
    ("idle_fraction", "Idle fraction", "{:.2f}"),
]


def _job(args):
    scenario, sched_name, seed = args
    t = time.time()
    world = build_world(scenario, seed=seed)
    m, _, _ = run_episode(world, make_scheduler(sched_name), seed=seed)
    return {"scenario": scenario, "scheduler": sched_name, "seed": seed, "wall_s": time.time() - t, **m}


def run_benchmark(scenarios, schedulers, seeds, workers: int = 8) -> pd.DataFrame:
    jobs = [(sc, sh, s) for sc in scenarios for sh in schedulers for s in seeds]
    # torch-backed schedulers are run in-process to avoid re-initialising CUDA per worker
    heavy = [j for j in jobs if j[1] == "d3qn"]
    light = [j for j in jobs if j[1] != "d3qn"]
    rows = [_job(j) for j in heavy]
    if workers > 1 and len(light) > 1:
        with ProcessPoolExecutor(workers) as ex:
            rows += list(ex.map(_job, light))
    else:
        rows += [_job(j) for j in light]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c, _, _ in HEADLINE if c in df.columns]
    return df.groupby(["scenario", "scheduler"])[cols].agg(["mean", "std"])


def to_markdown(df: pd.DataFrame, order: list[str]) -> str:
    lines = []
    mean = df.groupby(["scenario", "scheduler"]).mean(numeric_only=True)
    for sc in sorted(df["scenario"].unique()):
        lines.append(f"\n### {sc}\n")
        hdr = ["Scheduler"] + [label for c, label, _ in HEADLINE if c in mean.columns]
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "---|" * len(hdr))
        for sh in order:
            if (sc, sh) not in mean.index:
                continue
            r = mean.loc[(sc, sh)]
            cells = [sh] + [("–" if pd.isna(r[c]) else fmt.format(r[c])) for c, _, fmt in HEADLINE if c in mean.columns]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(df: pd.DataFrame, out: Path, order: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scen = sorted(df["scenario"].unique())
    for metric, label in [("pd", "Probability of detection (Pd)"), ("pd_weighted", "Threat-weighted Pd"),
                          ("ttfi_censored_mean", "Time to first intercept [s]")]:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        w = 0.8 / len(order)
        for i, sh in enumerate(order):
            g = df[df.scheduler == sh].groupby("scenario")[metric]
            mu = [g.mean().get(s, np.nan) for s in scen]
            sd = [g.std().get(s, 0) for s in scen]
            ax.bar(np.arange(len(scen)) + i * w - 0.4 + w / 2, mu, w, yerr=sd, label=sh, capsize=2)
        ax.set_xticks(np.arange(len(scen)), scen)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(ncols=len(order), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.15))
        fig.tight_layout()
        fig.savefig(out / f"{metric}.png", dpi=130)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--schedulers", default="sweep,random,round_robin,bandit,smart")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=100, help="evaluation seeds start here (training uses < 100)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmark"))
    a = ap.parse_args(argv)
    scenarios = list_scenarios() if a.scenarios == "all" else a.scenarios.split(",")
    schedulers = a.schedulers.split(",")
    seeds = list(range(a.seed_offset, a.seed_offset + a.seeds))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    df = run_benchmark(scenarios, schedulers, seeds, a.workers)
    df.to_csv(out / "results.csv", index=False)
    md = to_markdown(df, schedulers)
    (out / "summary.md").write_text(f"# Benchmark ({len(seeds)} seeds, eval seeds {seeds[0]}..{seeds[-1]})\n{md}\n")
    plot(df, out, schedulers)
    print(md)
    print(f"\n{len(df)} episodes in {time.time() - t:.0f}s → {out}")


if __name__ == "__main__":
    main()
