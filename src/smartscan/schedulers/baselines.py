"""Open-loop reference schedulers (the PDF's "traditional" scanning)."""

from __future__ import annotations

import numpy as np

from smartscan.schedulers.base import Action, Scheduler


def _dwell_index(rx, dwell_s: float) -> int:
    return int(np.argmin(np.abs(np.asarray(rx.dwells_s) - dwell_s)))


class LinearSweep(Scheduler):
    """Deterministic sweep: channel 0..K-1 in order, fixed dwell."""

    name = "sweep"

    def __init__(self, dwell_s: float = 0.01):
        self.dwell_s = dwell_s

    def reset(self, rx, duration, seed=0):
        super().reset(rx, duration, seed)
        self._k = 0
        self._d = _dwell_index(rx, self.dwell_s)

    def act(self, t, channel):
        ch = self._k % self.rx.n_channels
        self._k += 1
        return Action(ch, self._d, reason="sweep")


class RandomScan(Scheduler):
    """Uniformly random channel, fixed dwell."""

    name = "random"

    def __init__(self, dwell_s: float = 0.01):
        self.dwell_s = dwell_s

    def reset(self, rx, duration, seed=0):
        super().reset(rx, duration, seed)
        self._d = _dwell_index(rx, self.dwell_s)

    def act(self, t, channel):
        return Action(int(self.rng.integers(self.rx.n_channels)), self._d, reason="random")


class RoundRobinPrior(Scheduler):
    """Round-robin with a pre-mission revisit weighting per channel.

    Stands in for a classic "pre-programmed" search plan built from prior
    intelligence. Here the prior just favours the radar bands (S, C, X, Ku),
    because the problem states no reliable prior exists.
    """

    name = "round_robin"

    BANDS_GHZ = ((2.7, 3.5), (5.2, 5.9), (8.5, 10.5), (13.0, 17.5))

    def __init__(self, dwell_s: float = 0.01, boost: int = 2):
        self.dwell_s = dwell_s
        self.boost = boost

    def reset(self, rx, duration, seed=0):
        super().reset(rx, duration, seed)
        self._d = _dwell_index(rx, self.dwell_s)
        seq = []
        for ch, fc in enumerate(rx.centers_hz / 1e9):
            w = self.boost if any(lo <= fc <= hi for lo, hi in self.BANDS_GHZ) else 1
            seq.extend([ch] * w)
        # interleave repeats so boosted channels are spread over the cycle
        order = np.argsort(np.array([i / seq.count(c) + 1e-3 * c for c in set(seq)
                                     for i in range(seq.count(c))]), kind="stable")
        flat = [c for c in sorted(set(seq)) for _ in range(seq.count(c))]
        self._seq = [flat[i] for i in order]
        self._k = 0

    def act(self, t, channel):
        ch = self._seq[self._k % len(self._seq)]
        self._k += 1
        return Action(ch, self._d, reason="plan")
