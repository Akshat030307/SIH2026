"""Gymnasium environment for training the D3QN scheduler.

Observation: flattened per-channel feature matrix plus global features (see perception.py).
Action:      Discrete(K·D), a = channel·D + dwell_idx.
Reward (PDF §Phase 3, R_t = α·H_t − β·M_t − γ·Δt_err − δ·C_tune):
  H_t   threat-weighted illumination events newly intercepted this step, plus
        a discovery bonus for an emitter's first intercept
  M_t   threat-weighted events of *already known* emitters that ended un-intercepted
  Δt_err|predicted-window center − first pulse|, for prediction-driven dwells
  C_tune 1 if the local oscillator was retuned
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from smartscan import pdw as P
from smartscan.rl.perception import F_CH, F_GLOBAL, Perception
from smartscan.sim.engine import RFEngine
from smartscan.sim.world import build_world

DEFAULT_SCENARIOS = ("S1_sparse", "S2_dense", "S3_mfr", "S4_agile_lpi", "S5_popup", "S6_lockin", "S7_colocated")


@dataclass
class RewardCfg:
    alpha: float = 1.0
    discovery: float = 1.0
    beta: float = 0.5
    gamma_err: float = 2.0  # per second of timing error
    delta_tune: float = 0.002
    scale: float = 10.0  # keeps Q-values on the scale of the DQfD margin (0.8)


class SmartScanEnv(gym.Env):
    metadata = {"render_modes": []}  # noqa: RUF012 (gymnasium convention)

    def __init__(self, scenarios=DEFAULT_SCENARIOS, seed: int = 0, reward: RewardCfg | None = None,
                 duration: float | None = None, behaviour=None):
        self.scenarios = list(scenarios)
        self.reward_cfg = reward or RewardCfg()
        self.duration_override = duration
        self.behaviour = behaviour
        self._rng = np.random.default_rng(seed)
        w = build_world(self.scenarios[0], seed=0)
        self.K, self.D = w.receiver.n_channels, w.receiver.n_dwells
        self.observation_space = gym.spaces.Box(-2.0, 2.0, (self.K * F_CH + F_GLOBAL,), np.float32)
        self.action_space = gym.spaces.Discrete(self.K * self.D)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        sc = (options or {}).get("scenario") or self.scenarios[int(self._rng.integers(len(self.scenarios)))]
        wseed = int(self._rng.integers(0, 100))  # training seeds 0..99; evaluation uses ≥ 100
        self.world = build_world(sc, seed=wseed)
        if self.duration_override:
            self.world.duration = self.duration_override
        self.eng = RFEngine(self.world).reset(seed=int(self._rng.integers(1 << 30)))
        self.per = Perception(behaviour=self.behaviour)
        self.per.reset(self.world.receiver, self.world.duration, seed=int(self._rng.integers(1 << 30)))
        self.known = np.zeros(len(self.world.emitters), dtype=bool)
        obs = self.per.features(0.0, -1)
        return obs, {"scenario": sc, "expert": self.per.expert_action(0.0, -1)}

    def step(self, a: int):
        ch, d = divmod(int(a), self.D)
        t, cur = self.eng.t, self.eng.channel
        act = self.per.commit_action(t, cur, ch, d)
        res = self.eng.step(ch, d)
        self.per.observe(res.t_listen, res.t_end, res.channel, P.strip_truth(res.pdws))
        r = self._reward(res, act)
        obs = self.per.features(self.eng.t, self.eng.channel)
        info = {"reason": act.reason}
        if not res.done:
            info["expert"] = self.per.expert_action(self.eng.t, self.eng.channel)
        else:
            gt = self.eng.truth
            info["ep_pd"] = float(gt.intercepted.mean()) if gt.n else float("nan")
            info["ep_pd_w"] = float((gt.priority * gt.intercepted).sum() / max(gt.priority.sum(), 1e-9))
        return obs, r, res.done, False, info

    def _reward(self, res, act) -> float:
        c, gt = self.reward_cfg, self.eng.truth
        r = 0.0
        for e in res.new_intercepts:
            eid = gt.emitter[e]
            r += c.alpha * gt.priority[e] / 5.0
            if not self.known[eid]:
                self.known[eid] = True
                r += c.discovery * gt.priority[e] / 5.0
        for e in res.missed:
            if self.known[gt.emitter[e]]:
                r -= c.beta * gt.priority[e] / 5.0
        if act.intent is not None:
            true = res.pdws[res.pdws["emitter"] >= 0]
            if len(true):
                r -= c.gamma_err * min(abs(float(true["toa"].min()) - act.intent.predicted_time), 0.05)
        if res.retuned:
            r -= c.delta_tune
        return float(c.scale * r)
