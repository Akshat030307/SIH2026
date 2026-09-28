"""Phase 1: zero-knowledge spectrum exploration as a multi-armed bandit.

Each channel is an arm. The payoff of a visit is the (threat-weighted) amount
of *new* information it brought: a new emitter track, or a hit on a track
whose illumination times aren't predictable yet.

The PDF specifies UCB1:  A_t = argmax_a Q_t(a) + c·sqrt(ln t / N_t(a)).
UCB1 assumes stationary payoffs, but emitters switch on/off and lose value
once they are locked. So we also provide Discounted UCB, Sliding-Window UCB
and Thompson sampling. D-UCB is the default.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from smartscan.schedulers.base import Action, Scheduler
from smartscan.tracker.tracker import Tracker, infer_threat


class Bandit:
    def __init__(self, k: int, rng: np.random.Generator):
        self.k = k
        self.rng = rng

    def select(self) -> int: ...

    def update(self, arm: int, reward: float) -> None: ...

    def values(self) -> np.ndarray:
        """Per-arm value estimate (used as an RL feature)."""
        raise NotImplementedError


class UCB1(Bandit):
    def __init__(self, k, rng, c: float = 0.5):
        super().__init__(k, rng)
        self.c = c
        self.n = np.zeros(k)
        self.s = np.zeros(k)
        self.t = 0

    def select(self):
        self.t += 1
        if (self.n == 0).any():
            return int(self.rng.choice(np.flatnonzero(self.n == 0)))
        q = self.s / self.n
        return int(np.argmax(q + self.c * np.sqrt(np.log(self.t) / self.n)))

    def update(self, arm, reward):
        self.n[arm] += 1
        self.s[arm] += reward

    def values(self):
        return np.where(self.n > 0, self.s / np.maximum(self.n, 1e-9), 0.0)

    def counts(self):
        return self.n


class DiscountedUCB(UCB1):
    """D-UCB (Garivier & Moulines 2011): statistics decay by γ at every play."""

    def __init__(self, k, rng, c: float = 0.5, gamma: float = 0.995):
        super().__init__(k, rng, c)
        self.gamma = gamma

    def select(self):
        if (self.n < 1e-6).any():
            return int(self.rng.choice(np.flatnonzero(self.n < 1e-6)))
        tot = self.n.sum()
        q = self.s / self.n
        return int(np.argmax(q + self.c * np.sqrt(np.log(max(tot, 1.0 + 1e-9)) / self.n)))

    def update(self, arm, reward):
        self.n *= self.gamma
        self.s *= self.gamma
        self.n[arm] += 1
        self.s[arm] += reward


class SlidingWindowUCB(UCB1):
    def __init__(self, k, rng, c: float = 0.5, window: int = 400):
        super().__init__(k, rng, c)
        self.hist: deque = deque()
        self.window = window

    def update(self, arm, reward):
        self.hist.append((arm, reward))
        self.n[arm] += 1
        self.s[arm] += reward
        if len(self.hist) > self.window:
            a, r = self.hist.popleft()
            self.n[a] -= 1
            self.s[a] -= r

    def select(self):
        if (self.n == 0).any():
            return int(self.rng.choice(np.flatnonzero(self.n == 0)))
        q = self.s / self.n
        return int(np.argmax(q + self.c * np.sqrt(np.log(min(len(self.hist), self.window) + 1) / self.n)))


class Thompson(Bandit):
    """Beta-Bernoulli Thompson sampling with forgetting."""

    def __init__(self, k, rng, gamma: float = 0.995):
        super().__init__(k, rng)
        self.a = np.ones(k)
        self.b = np.ones(k)
        self.gamma = gamma

    def select(self):
        return int(np.argmax(self.rng.beta(self.a, self.b)))

    def update(self, arm, reward):
        self.a = 1 + (self.a - 1) * self.gamma
        self.b = 1 + (self.b - 1) * self.gamma
        self.a[arm] += reward
        self.b[arm] += 1 - reward

    def values(self):
        return self.a / (self.a + self.b)

    def counts(self):
        return self.a + self.b - 2


BANDITS = {"ucb1": UCB1, "ducb": DiscountedUCB, "swucb": SlidingWindowUCB, "thompson": Thompson}


def novelty_reward(touched, max_threat: float = 5.0, unlocked_weight: float = 0.0) -> float:
    """Exploration payoff of one dwell, in [0, 1].

    Discovering a new emitter is the main payoff. Re-seeing a known but
    unlocked emitter is worth ``unlocked_weight`` (0 in the smart scheduler,
    whose acquisition mode already covers those; > 0 for the stand-alone bandit).
    """
    r = 0.0
    for tr, is_new, n in touched:
        if n < 2:
            continue
        w = infer_threat(tr) / max_threat
        if is_new:
            r += 0.6 + 0.4 * w
        elif not tr.locked and unlocked_weight > 0 and getattr(tr, "_dirty", False):
            r += unlocked_weight * (0.5 + 0.5 * w)
    return float(min(r, 1.0))


class BanditScheduler(Scheduler):
    """Pure Phase-1 scheduler: bandit channel choice, fixed dwell, tracker for rewards."""

    def __init__(self, kind: str = "ducb", dwell_s: float = 0.01, **kw):
        self.kind, self.dwell_s, self.kw = kind, dwell_s, kw
        self.name = f"bandit_{kind}"

    def reset(self, rx, duration, seed=0):
        super().reset(rx, duration, seed)
        self.bandit = BANDITS[self.kind](rx.n_channels, self.rng, **self.kw)
        self.tracker = Tracker(rx)
        self._d = int(np.argmin(np.abs(np.asarray(rx.dwells_s) - self.dwell_s)))

    def act(self, t, channel):
        return Action(self.bandit.select(), self._d, reason="explore")

    def observe(self, t_listen, t_end, channel, pdws):
        assign, touched = self.tracker.update(t_listen, t_end, channel, pdws)
        for tr, _, _ in touched:
            self.tracker.refresh_estimate(tr)
        self.bandit.update(channel, novelty_reward(touched, unlocked_weight=0.5))
        return assign
