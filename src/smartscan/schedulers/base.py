"""Scheduler interface.

The runner loops: ``action = scheduler.act(t)`` → engine step →
``scheduler.observe(result, pdws_without_truth)``.

A scheduler only ever sees PDWs with ground truth removed. It may attach an
*intent* to an action, meaning "I expect emitter track X to illuminate at
time t". The evaluator uses intents for intercept-time-error, Pfa and
prediction-accuracy metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartscan.sim.receiver import ReceiverSpec


@dataclass
class Intent:
    track_id: int
    predicted_time: float  # predicted illumination center / first arrival
    predicted_mode: int | None = None


@dataclass
class Action:
    channel: int
    dwell_idx: int
    intent: Intent | None = None
    reason: str = ""  # free-text tag for the dashboard: explore / track / sweep ...


class Scheduler:
    name = "base"

    def reset(self, rx: ReceiverSpec, duration: float, seed: int = 0) -> None:
        self.rx = rx
        self.duration = duration
        self.rng = np.random.default_rng(seed)

    def act(self, t: float, channel: int) -> Action:
        raise NotImplementedError

    def observe(self, t_listen: float, t_end: float, channel: int, pdws: np.ndarray) -> np.ndarray | None:
        """Consume the (truth-stripped) PDWs from the last dwell.

        Returns an optional per-PDW track-id array (-1 = unassigned). The
        evaluator uses it to map tracks to ground-truth emitters.
        """
        return None

    def tracks_snapshot(self) -> list[dict]:
        """For the dashboard: current track table (may be empty)."""
        return []
