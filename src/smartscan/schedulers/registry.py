"""Name → scheduler factory, shared by the evaluator, the API and the dashboard."""

from __future__ import annotations

from smartscan.schedulers.bandit import BanditScheduler
from smartscan.schedulers.baselines import LinearSweep, RandomScan, RoundRobinPrior
from smartscan.schedulers.base import Scheduler
from smartscan.schedulers.heuristic import SmartHeuristic

DESCRIPTIONS = {
    "sweep": "Open-loop linear sweep, 10 ms dwell (legacy baseline)",
    "random": "Uniform random channel, 10 ms dwell",
    "round_robin": "Pre-planned round-robin favouring radar bands",
    "bandit": "Phase 1 only: Discounted-UCB channel selection",
    "smart": "Bandit + tracker + period lock + acquisition (heuristic)",
    "d3qn": "Dueling Double DQN policy over tracker/bandit features",
}


def make_scheduler(name: str, **kw) -> Scheduler:
    if name == "sweep":
        return LinearSweep(**kw)
    if name == "random":
        return RandomScan(**kw)
    if name == "round_robin":
        return RoundRobinPrior(**kw)
    if name == "bandit":
        return BanditScheduler("ducb", **kw)
    if name.startswith("bandit_"):
        return BanditScheduler(name.split("_", 1)[1], **kw)
    if name == "smart":
        return SmartHeuristic(**kw)
    if name.startswith("smart_"):  # smart_<bandit kind>
        return SmartHeuristic(bandit=name.split("_", 1)[1], **kw)
    if name == "d3qn":
        from smartscan.rl.policy import D3QNScheduler

        return D3QNScheduler(**kw)
    raise KeyError(f"unknown scheduler {name!r}; known: {sorted(DESCRIPTIONS)}")
