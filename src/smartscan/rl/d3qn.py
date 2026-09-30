"""Dueling Double Deep Q-Network scheduler (PDF §Phase 3).

Network: a shared per-channel encoder θ (weights shared across all K
channels, then a 1-D convolution across neighbouring channels) feeds
  * a value stream       V(s; θ, α)  from pooled channel context,
  * an advantage stream  A(s, a; θ, β), one head per channel with D dwell outputs,
  combined as  Q = V + (A − mean_a' A)  (Wang et al. 2016; the PDF's aggregation).
Because the encoder is shared, a lesson learned on one band carries over to all bands.

Learning: Double-DQN targets (online net selects, target net evaluates),
n-step returns, prioritized experience replay (proportional, sum-tree), and
DQfD-style demonstrations (Hester et al. 2018). The replay is seeded with
SmartHeuristic trajectories, and a large-margin loss keeps the agent close to
the expert until TD learning finds something better.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from smartscan.rl.perception import F_CH, F_GLOBAL


class DuelingQNet(nn.Module):
    def __init__(self, K: int, D: int, d: int = 128):
        super().__init__()
        self.K, self.D = K, D
        self.enc = nn.Sequential(nn.Linear(F_CH + F_GLOBAL, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.mix = nn.Conv1d(d, d, kernel_size=3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * d + F_GLOBAL, d), nn.ReLU())
        self.adv = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, D))
        self.val = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        f = obs[:, : self.K * F_CH].reshape(B, self.K, F_CH)
        g = obs[:, self.K * F_CH:]
        h = self.enc(torch.cat([f, g[:, None, :].expand(B, self.K, F_GLOBAL)], -1))  # (B, K, d)
        h = h + F.relu(self.mix(h.transpose(1, 2))).transpose(1, 2)
        ctx = self.glob(torch.cat([h.mean(1), h.amax(1), g], -1))  # (B, d)
        A = self.adv(torch.cat([h, ctx[:, None, :].expand_as(h)], -1)).reshape(B, self.K * self.D)
        V = self.val(ctx)
        return V + A - A.mean(1, keepdim=True)


class SumTree:
    """Array-backed sum tree for proportional prioritized replay (vectorized)."""

    def __init__(self, capacity: int):
        self.cap = 1
        while self.cap < capacity:
            self.cap *= 2
        self.tree = np.zeros(2 * self.cap)

    def update(self, idx: np.ndarray, p: np.ndarray):
        i = np.asarray(idx) + self.cap
        self.tree[i] = p
        # all leaves sit at the same depth, so walk up one level at a time until the root
        while True:
            i = np.unique(i // 2)
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            if i[0] == 1:
                break

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def sample(self, n: int, rng) -> np.ndarray:
        u = (np.arange(n) + rng.random(n)) * (self.total / n)
        i = np.ones(n, dtype=np.int64)
        while i[0] < self.cap:
            left = self.tree[2 * i]
            go_right = u > left
            u = np.where(go_right, u - left, u)
            i = 2 * i + go_right
        return i - self.cap


class Replay:
    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6, eps: float = 1e-3):
        self.cap, self.alpha, self.eps = capacity, alpha, eps
        self.s = np.zeros((capacity, obs_dim), np.float16)
        self.s2 = np.zeros((capacity, obs_dim), np.float16)
        self.a = np.zeros(capacity, np.int64)
        self.a_exp = np.full(capacity, -1, np.int64)  # expert action in s (-1 = unknown)
        self.r = np.zeros(capacity, np.float32)
        self.g = np.zeros(capacity, np.float32)  # γ^n, 0 if terminal
        self.demo = np.zeros(capacity, bool)
        self.tree = SumTree(capacity)
        self.n, self.ptr, self.max_p = 0, 0, 1.0
        self.floor = 0  # slots below this index are permanent (demonstrations)

    def protect(self) -> None:
        """Make everything stored so far permanent: later writes cycle over the remaining slots.

        DQfD keeps demonstrations for the whole run. Without this, the ring buffer
        overwrites them and the margin loss silently disappears.
        """
        self.floor = self.n
        self.ptr = self.n % self.cap

    def add(self, s, a, r, s2, g, a_exp=-1, demo=False):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.g[i] = s, a, r, s2, g
        self.a_exp[i], self.demo[i] = a_exp, demo
        self.tree.update(np.array([i]), np.array([self.max_p]))
        self.ptr = self.ptr + 1
        if self.ptr >= self.cap:
            self.ptr = self.floor
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs: int, rng, beta: float = 0.4):
        idx = self.tree.sample(bs, rng)
        idx = np.minimum(idx, self.n - 1)
        p = self.tree.tree[idx + self.tree.cap] / max(self.tree.total, 1e-12)
        w = (self.n * np.maximum(p, 1e-12)) ** (-beta)
        w = w / w.max()
        return idx, w.astype(np.float32)

    def update_priorities(self, idx, td):
        p = (np.abs(td) + self.eps) ** self.alpha
        p = np.where(self.demo[idx], p + 0.05, p)  # demos stay a bit more likely (DQfD ε_d)
        self.max_p = max(self.max_p, float(p.max()))
        self.tree.update(idx, p)


class NStep:
    """Per-environment n-step return accumulator."""

    def __init__(self, n: int, gamma: float):
        self.n, self.gamma = n, gamma
        self.buf: deque = deque()

    def push(self, s, a, r, s2, done, a_exp):
        self.buf.append((s, a, r, s2, done, a_exp))
        out = []
        if done:
            while self.buf:
                out.append(self._emit())
                self.buf.popleft()
        elif len(self.buf) >= self.n:
            out.append(self._emit())
            self.buf.popleft()
        return out

    def _emit(self):
        R, g = 0.0, 1.0
        s0, a0, _, _, _, ae0 = self.buf[0]
        for (_, _, r, s2, done, _) in self.buf:
            R += g * r
            g *= self.gamma
            last_s2, last_done = s2, done
            if done:
                break
        return s0, a0, R, last_s2, 0.0 if last_done else g, ae0


class D3QNAgent:
    def __init__(self, K: int, D: int, obs_dim: int, lr: float = 2.5e-4, gamma: float = 0.99, n_step: int = 5,
                 margin: float = 0.8, lambda_demo: float = 1.0, tau: float = 0.005, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.K, self.D = K, D
        self.q = DuelingQNet(K, D).to(self.device)
        self.qt = DuelingQNet(K, D).to(self.device)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.n_step, self.margin, self.lambda_demo, self.tau = gamma, n_step, margin, lambda_demo, tau

    @torch.inference_mode()
    def act(self, obs: np.ndarray, eps: float, rng) -> np.ndarray:
        q = self.q(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        a = q.argmax(1).cpu().numpy()
        rnd = rng.random(len(a)) < eps
        a[rnd] = rng.integers(0, self.K * self.D, rnd.sum())
        return a

    def update(self, rb: Replay, bs: int, rng, beta: float) -> dict:
        idx, w = rb.sample(bs, rng, beta)
        dev = self.device
        s = torch.as_tensor(rb.s[idx], dtype=torch.float32, device=dev)
        s2 = torch.as_tensor(rb.s2[idx], dtype=torch.float32, device=dev)
        a = torch.as_tensor(rb.a[idx], device=dev)
        r = torch.as_tensor(rb.r[idx], device=dev)
        g = torch.as_tensor(rb.g[idx], device=dev)
        ae = torch.as_tensor(rb.a_exp[idx], device=dev)
        wt = torch.as_tensor(w, device=dev)
        q = self.q(s)
        qa = q.gather(1, a[:, None])[:, 0]
        with torch.no_grad():
            a2 = self.q(s2).argmax(1, keepdim=True)  # double DQN: online selects ...
            y = r + g * self.qt(s2).gather(1, a2)[:, 0]  # ... target evaluates
        td = y - qa
        loss_td = (wt * F.smooth_l1_loss(qa, y, reduction="none")).mean()
        # large-margin imitation loss where an expert action is known (demonstrations)
        has = ae >= 0
        loss_m = torch.zeros((), device=dev)
        if has.any():
            qh = q[has]
            l = torch.full_like(qh, self.margin)
            l.scatter_(1, ae[has][:, None], 0.0)
            loss_m = ((qh + l).amax(1) - qh.gather(1, ae[has][:, None])[:, 0]).mean()
        loss = loss_td + self.lambda_demo * loss_m
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.qt.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
        rb.update_priorities(idx, td.detach().cpu().numpy())
        return {"loss_td": loss_td.item(), "loss_margin": loss_m.item(), "q_mean": qa.mean().item()}

    def save(self, path, meta: dict | None = None):
        torch.save({"q": self.q.state_dict(), "K": self.K, "D": self.D, "meta": meta or {}}, path)
