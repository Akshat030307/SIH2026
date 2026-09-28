"""Regenerate the results section of README.md from results/*.

    python -m smartscan.report
"""

from __future__ import annotations

import json
import re
from pathlib import Path

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
    m = piv.loc["**mean**"]
    head = ""
    if {"sweep", "smart"} <= set(order):
        best = "d3qn" if "d3qn" in order and m["d3qn"] >= m["smart"] else "smart"
        head = (f"**Headline:** averaged over 7 scenarios, the learned D3QN scheduler reaches a threat-weighted Pd "
                f"of **{m.get('d3qn', float('nan')):.2f}** and the smart heuristic **{m['smart']:.2f}**, against "
                f"**{m['sweep']:.2f}** for a legacy open-loop sweep ({m[best] / m['sweep']:.1f}× better). In the "
                f"lock-in scenario the sweep catches {piv.loc['S6_lockin', 'sweep']:.0%} of illuminations; the "
                f"smart schedulers catch {piv.loc['S6_lockin', best]:.0%}.\n\n")
    return (head + f"### Scheduler benchmark ({n_seeds} held-out seeds per scenario)\n\n"
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
    runs = {"run 1 (demos overwritten)": ROOT / "checkpoints" / "run1_unprotected_demos" / "d3qn_train_log.json",
            "run 3 (demos protected)": ROOT / "checkpoints" / "d3qn_train_log.json"}
    cols = {}
    for name, p in runs.items():
        if p.exists():
            ev = json.loads(p.read_text())["evals"]
            cols[name] = {e["steps"] // 1000 * 1000: e["pd_weighted"] for e in ev}
    if not cols:
        return ""
    df = pd.DataFrame(cols)
    df.index.name = "online steps"
    out = ("### D3QN training\n\nValidation threat-weighted Pd during training (7 validation worlds, seed 500). "
           "Run 1's replay buffer overwrote the DQfD demonstrations after ~280k steps, and the policy then "
           "collapsed. Run 3 keeps them in a protected region and stays stable:\n\n" + md_table(df.round(3)) + "\n")
    sel = ROOT / "results" / "d3qn_selection.csv"
    if sel.exists():
        d = pd.read_csv(sel)
        t = d.groupby("ckpt")["pd_weighted"].agg(["mean", "std"]).sort_values("mean", ascending=False)
        t.index = [Path(i).stem for i in t.index]
        t.index.name = "candidate"
        out += ("\nCheckpoint selection on 35 validation episodes (seeds 500–504 × 7 scenarios, disjoint from "
                "test). The winner is shipped as `checkpoints/d3qn_best.pt`:\n\n" + md_table(t.round(3)) + "\n")
    return out


def main():
    parts = [benchmark_section(), md_file("ablation/summary.md", "Ablation of the smart scheduler"),
             training_section(), md_file("predict/summary.md", "MFR behaviour prediction"),
             md_file("deinterleave/summary.md", "Deinterleaving"), md_file("edge/summary.md", "Edge inference")]
    body = "\n".join(p for p in parts if p)
    readme = ROOT / "README.md"
    s = readme.read_text()
    new = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                 lambda _: f"<!-- RESULTS:START -->\n## Results\n\n{body}\n<!-- RESULTS:END -->", s, flags=re.DOTALL)
    readme.write_text(new)
    print(body)


if __name__ == "__main__":
    main()
