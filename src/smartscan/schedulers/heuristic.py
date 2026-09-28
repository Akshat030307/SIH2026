"""Smart heuristic scheduler: bandit exploration + period lock + priority arbitration.

This is the strong non-learned baseline the D3QN agent has to beat, and the
fallback demo. Each decision:

  1. For every locked track (scan period estimated), predict the next main-beam
     window [t_c − h, t_c + h] and its channel.
  2. If a window is due (it opens before we could finish an exploration dwell),
     tune there and dwell until it closes: a *prediction-driven* dwell. When
     several windows are due at once, the highest inferred threat wins.
  3. Otherwise, *acquire*: detected but not yet locked tracks have their
     channel revisited at roughly half the observed illumination width, so no
     illumination is skipped and a period lock forms within about 3 rotations.
     Acquisition may use at most ``acq_budget`` of the time, split by threat.
  4. Otherwise explore: the bandit picks the channel, and the dwell is the
     largest option that still ends before the next due window. Exploration
     dwell lengths are randomly jittered, so the revisit pattern can't
     phase-lock with a radar's rotation period.
  5. After a prediction-driven dwell, a hit resets the track's miss counter.
     Repeated misses drop the lock (mode change / wrong period multiple).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from smartscan.schedulers.bandit import BANDITS, novelty_reward
from smartscan.schedulers.base import Action, Intent, Scheduler
from smartscan.tracker.tracker import Track, Tracker, infer_threat


@dataclass
class Window:
    track: Track
    center: float
    open: float
    close: float
    channel: int
    priority: float


class SmartHeuristic(Scheduler):
    name = "smart_heuristic"

    def __init__(self, bandit: str = "ducb", explore_dwell_s: float = 0.01, n_need: int = 4,
                 guard_s: float = 0.0015, sigma_k: float = 2.5, max_window_s: float = 0.06,
                 acq_budget: float = 0.5, **bandit_kw):
        self.bandit_kind = bandit
        self.bandit_kw = bandit_kw
        self.explore_dwell_s = explore_dwell_s
        self.n_need = n_need
        self.guard_s = guard_s
        self.sigma_k = sigma_k
        self.max_window_s = max_window_s
        self.acq_budget = acq_budget

    def reset(self, rx, duration, seed=0):
        super().reset(rx, duration, seed)
        self.tracker = Tracker(rx)
        self.bandit = BANDITS[self.bandit_kind](rx.n_channels, self.rng, **self.bandit_kw)
        self.dwells = np.asarray(rx.dwells_s)
        self._commit: Window | None = None
        self._steps = 0
        self.last_visit = np.full(rx.n_channels, -np.inf)
        self._acq_log: deque = deque()  # (t_end, duration) of acquisition dwells
        self._acq_time = 0.0

    # --------------------------------------------------------------- windows
    def half_width(self, tr: Track, cycles: int) -> float:
        """Half-width of the dwell around a predicted beam center.

        Any overlap with the illumination is enough, so the center uncertainty
        is discounted by the observed illumination width.
        """
        sigma = tr.est.center_std(cycles)
        pri = tr.pri if not np.isnan(tr.pri) else 0.003
        h = max(self.sigma_k * sigma - 0.4 * tr.illum_width(), 0.5 * self.n_need * pri) + self.guard_s
        return float(min(h, self.max_window_s / 2))

    def windows(self, t: float) -> list[Window]:
        out = []
        for tr in self.tracker.tracks.values():
            if tr.est is None:
                continue
            h0 = self.half_width(tr, 1)
            tc, k = tr.est.next_center(t - h0)
            h = self.half_width(tr, k)
            out.append(Window(tr, tc, tc - h, tc + h, tr.best_channel, infer_threat(tr)))
        out.sort(key=lambda w: w.open)
        return out

    def _dwell_to_cover(self, need: float) -> int:
        i = np.searchsorted(self.dwells, need - 1e-9)
        return int(min(i, len(self.dwells) - 1))

    # ------------------------------------------------------------------- act
    def act(self, t, channel):
        a = self._act(t, channel)
        self._reason = a.reason
        return a

    def _act(self, t, channel):
        self._steps += 1
        tune = self.rx.tune_s
        wins = [w for w in self.windows(t) if w.close > t + tune]
        self._commit = None
        if wins:
            lead = self.dwells[0] + 2 * tune
            due = [w for w in wins if w.open <= t + lead]
            if due:
                w = max(due, key=lambda w: (w.priority, -w.close))
                t_listen = t + (tune if w.channel != channel else 0.0)
                d = self._dwell_to_cover(w.close - t_listen)
                self._commit = w
                return Action(w.channel, d, Intent(w.track.id, w.center), reason="track")
            slack = wins[0].open - t - 2 * tune
        else:
            slack = np.inf
        acq = self._acquisition_channel(t)
        if acq is not None:
            ch, ri = acq
            cap = min(self.explore_dwell_s, 0.5 * ri, slack)
            ok = np.flatnonzero(self.dwells <= cap + 1e-12)
            return Action(ch, int(ok[-1]) if len(ok) else 0, reason="acquire")
        cap = min(self.explore_dwell_s * self.rng.choice([0.5, 1.0, 1.0, 2.0]), slack)
        ok = np.flatnonzero(self.dwells <= cap + 1e-12)
        d = int(ok[-1]) if len(ok) else 0
        return Action(self.bandit.select(), d, reason="explore")

    def _acquisition_channel(self, t: float):
        """Most overdue channel holding an unlocked track, if within the time budget."""
        while self._acq_log and self._acq_log[0][0] < t - 2.0:
            self._acq_time -= self._acq_log.popleft()[1]
        if self._acq_time > self.acq_budget * 2.0:
            return None
        best, best_score = None, 1.0
        for tr in self.tracker.tracks.values():
            if tr.est is not None or not tr.hits:
                continue
            ch = tr.best_channel
            ri = float(np.clip(0.5 * tr.illum_width(), 0.015, 0.12))
            overdue = (t - self.last_visit[ch]) / ri
            score = overdue * (0.5 + infer_threat(tr) / 5.0)
            if overdue >= 1.0 and score > best_score:
                best, best_score = (ch, ri), score
        return best

    # --------------------------------------------------------------- observe
    def observe(self, t_listen, t_end, channel, pdws):
        assign, touched = self.tracker.update(t_listen, t_end, channel, pdws)
        for tr, _, _ in touched:
            self.tracker.refresh_estimate(tr)
        self.last_visit[channel] = t_end
        for tr, _, _ in touched:
            tr.threat_cache = infer_threat(tr)
        if self._reason == "acquire":
            self._acq_log.append((t_end, t_end - t_listen))
            self._acq_time += t_end - t_listen
        w = self._commit
        if w is not None:
            seen = any(tr.id == w.track.id for tr, _, n in touched)
            if seen:
                w.track.misses = 0
                w.track.locked_hits += 1
            elif w.track.id in self.tracker.tracks and t_end >= w.close - 1e-6:
                self.tracker.register_miss(w.track)
        elif self._reason == "explore":
            self.bandit.update(channel, novelty_reward(touched))
        if self._steps % 200 == 0:
            self.tracker.prune(t_end)
        return assign

    def tracks_snapshot(self):
        out = []
        for tr in self.tracker.tracks.values():
            out.append(dict(id=tr.id, aoa=round(tr.aoa, 1), rf_ghz=round(tr.rf / 1e9, 3),
                            pw_us=round(float(np.exp(tr.log_pw)) * 1e6, 2),
                            pri_us=None if np.isnan(tr.pri) else round(tr.pri * 1e6, 1),
                            agile=tr.agile, channel=tr.best_channel, threat=round(infer_threat(tr), 2),
                            period=None if tr.est is None else round(tr.est.period, 4),
                            hits=len(tr.hits), locked_hits=tr.locked_hits, last_seen=round(tr.last_seen, 3)))
        return out
