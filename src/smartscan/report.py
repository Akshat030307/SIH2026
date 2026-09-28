"""Regenerate the results section of README.md from results/*.

    python -m smartscan.report
"""

from __future__ import annotations

import json
import re

import pandas as pd

from smartscan.sim.world import ROOT
from smartscan.util import md_table

ORDER = ["sweep", "random", "round_robin", "bandit", "smart", "d3qn"]


def benchmark_section() -> str:
    p = ROOT / "results" / "benchmark" / "results.csv"
    if not p.exists():
        return ""
    df = pd.read_csv(p)
    order = [s for s in ORDER if s in set(df.scheduler)]
    n_seeds = df.groupby(["scenario", "scheduler"]).size().min()
    piv = df.pivot_table(index="scenario", columns="scheduler", values="pd_weighted", aggfunc="mean")[order]
    piv.loc["**mean**"] = piv.mean()
    overall = df.groupby("scheduler")[["pd", "pd_weighted", "intercept_rate", "ttfi_censored_mean",
                                       "intercept_time_error_ms", "pfa", "correct_predictions"]].mean().loc[order]
    overall.columns = ["Pd", "Pd (threat-wtd)", "Intercept rate [emitters/s]", "TTFI [s]",
                       "Intercept time err [ms]", "Pfa", "Correct predictions"]
    return (f"### Scheduler benchmark ({n_seeds} held-out seeds per scenario)\n\n"
            "Threat-weighted probability of detection (each illumination weighted by the emitter's threat):\n\n"
            + md_table(piv.round(3)) + "\n\nAll figures of merit, averaged over scenarios:\n\n"
            + md_table(overall.round(3)) + "\n")


def md_file(rel: str, title: str) -> str:
    p = ROOT / "results" / rel
    if not p.exists():
        return ""
    body = re.sub(r"^# .*\n", "", p.read_text()).strip()
    return f"### {title}\n\n{body}\n"


def training_section() -> str:
    p = ROOT / "checkpoints" / "d3qn_train_log.json"
    if not p.exists():
        return ""
    ev = json.loads(p.read_text())["evals"]
    if not ev:
        return ""
    df = pd.DataFrame(ev)[["steps", "pd", "pd_weighted", "train_pd_w"]]
    df.columns = ["online steps", "val Pd", "val Pd (threat-wtd)", "train-episode Pd (threat-wtd, ε-greedy)"]
    return ("### D3QN training (validation seeds 500, disjoint from train 0-99 and test 100+)\n\n"
            + md_table(df.round(3), index=False) + "\n")


def main():
    parts = [benchmark_section(), md_file("ablation/summary.md", "Ablation of the smart scheduler"),
             training_section(), md_file("predict/summary.md", "MFR behaviour prediction"),
             md_file("deinterleave/summary.md", "Deinterleaving"), md_file("edge/summary.md", "Edge inference")]
    body = "\n".join(p for p in parts if p)
    readme = ROOT / "README.md"
    s = readme.read_text()
    new = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                 lambda _: f"<!-- RESULTS:START -->\n## Results\n\n{body}\n<!-- RESULTS:END -->", s, flags=re.S)
    readme.write_text(new)
    print(body)


if __name__ == "__main__":
    main()
