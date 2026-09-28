"""Shared perception for learned schedulers + the per-channel state encoding.

``Perception`` is the SmartHeuristic's machinery (tracker, bandit,
predicted windows, acquisition bookkeeping) with the decision rule taken out.
It turns the scheduler's belief into the D3QN state S_t (PDF §Phase 3):

per channel (K rows, F_CH columns):
   0  time since last visit (log-scaled)
   1  bandit discovery value Q(c)
   2  bandit exploration bonus (uncertainty)
   3  # tracks on the channel
   4  # period-locked tracks on the channel
   5  time until the next predicted window opens (clipped, in 50 ms units)
   6  that window is due now (would be missed by any dwell that doesn't go there)
   7  predicted window length
   8  inferred threat of that window's emitter
   9  max inferred threat of unlocked tracks on the channel
  10  acquisition overdue ratio of the most overdue unlocked track
  11  is the receiver currently tuned here (retune cost)
  12  recent pulse activity (log)
  13  behaviour model: P(next radar word is a track-mode word) for emitters here
global (G columns): mission progress, share of time in acquisition, # tracks, # locked,
                    budget headroom.
"""

from __future__ import annotations

import numpy as np

from smartscan.schedulers.base import Action, Intent
from smartscan.schedulers.heuristic import SmartHeuristic, Window

F_CH = 14
F_GLOBAL = 5


class Perception(SmartHeuristic):
    name = "perception"

    def __init__(self, behaviour=None, **kw):
        super().__init__(**kw)
        self.behaviour = behaviour  # optional next-word Predictor (smartscan.predict)

    def reset(self, rx, duration, seed=0):
        super().reset(rx, duration, seed)
        self.activity = np.zeros(rx.n_channels)
        self._wins: list[Window] = []

    # ------------------------------------------------------------ features
    def features(self, t: float, channel: int) -> np.ndarray:
        K = self.rx.n_channels
        f = np.zeros((K, F_CH), np.float32)
        f[:, 0] = np.log1p(np.minimum(t - self.last_visit, 60.0) / 0.05) / 7.0
        b = self.bandit
        n = np.maximum(b.counts(), 1e-6)
        f[:, 1] = b.values()
        f[:, 2] = np.minimum(np.sqrt(np.log(max(n.sum(), 2.0)) / n), 3.0) / 3.0
        tune = self.rx.tune_s
        lead = self.dwells[0] + 2 * tune
        self._wins = [w for w in self.windows(t) if w.close > t + tune]
        soon = {}
        for w in self._wins:
            if w.channel not in soon:
                soon[w.channel] = w
        for c, w in soon.items():
            f[c, 5] = np.clip((w.open - t) / 0.05, -1.0, 5.0) / 5.0
            f[c, 6] = float(w.open <= t + lead)
            f[c, 7] = min((w.close - w.open) / 0.05, 2.0) / 2.0
            f[c, 8] = w.priority / 5.0
        for tr in self.tracker.tracks.values():
            c = tr.best_channel
            f[c, 3] += 0.2
            if tr.est is not None:
                f[c, 4] += 0.2
            else:
                th = tr.threat_cache / 5.0
                f[c, 9] = max(f[c, 9], th)
                if tr.hits:
                    ri = float(np.clip(0.5 * tr.illum_width(), 0.015, 0.12))
                    f[c, 10] = max(f[c, 10], min((t - self.last_visit[c]) / ri, 5.0) / 5.0)
            words = getattr(tr, "words", None)
            if self.behaviour is not None and words is not None and len(words) >= 2:
                if getattr(tr, "_bp_n", -1) != len(words):  # cache: recompute only on a new word
                    p = self.behaviour.predict_proba(np.asarray(words[-12:]))
                    tr._bp, tr._bp_n = float(p[:3].sum()), len(words)  # PRI < 150 µs words = track mode
                f[c, 13] = max(f[c, 13], tr._bp)
        f[:, 3] = np.minimum(f[:, 3], 1.0)
        f[:, 4] = np.minimum(f[:, 4], 1.0)
        if 0 <= channel < K:
            f[channel, 11] = 1.0
        f[:, 12] = np.log1p(self.activity) / 5.0
        g = np.array([
            t / self.duration,
            self._acq_time / 2.0,
            min(len(self.tracker.tracks) / 50.0, 1.0),
            min(sum(tr.est is not None for tr in self.tracker.tracks.values()) / 50.0, 1.0),
            float(self._acq_time <= self.acq_budget * 2.0),
        ], np.float32)
        return np.concatenate([f.ravel(), g])

    # ------------------------------------------------ act on external choice
    def commit_action(self, t: float, channel: int, ch: int, d: int) -> Action:
        """Wrap an externally chosen (channel, dwell) as an Action with the right bookkeeping.

        If the chosen channel holds a predicted window that the dwell overlaps,
        the dwell counts as prediction-driven: it gets an Intent, and a miss
        will count against the track's lock.
        """
        tune = self.rx.tune_s if ch != channel else 0.0
        t0, t1 = t + tune, t + tune + self.dwells[d]
        self._commit = None
        for w in self._wins:
            if w.channel == ch and w.open < t1 and w.close > t0:
                self._commit = w
                self._reason = "track"
                return Action(ch, d, Intent(w.track.id, w.center), reason="track")
        acq = self._acquisition_channel(t)
        self._reason = "acquire" if acq is not None and acq[0] == ch else "explore"
        return Action(ch, d, reason=self._reason)

    def observe(self, t_listen, t_end, channel, pdws):
        self.activity *= 0.98
        self.activity[channel] += len(pdws)
        assign = super().observe(t_listen, t_end, channel, pdws)
        if self.behaviour is not None:
            _update_words(self.tracker, channel)
        return assign

    def expert_action(self, t: float, channel: int) -> int:
        """The heuristic's choice in this state (for demonstrations). Leaves bookkeeping untouched."""
        saved = (self._commit, self._reason if hasattr(self, "_reason") else None, self._steps)
        a = self._act(t, channel)
        self._commit, self._reason, self._steps = saved
        return self.rx.encode_action(a.channel, a.dwell_idx)


def _update_words(tracker, channel):
    """Append a radar word to a track each time it gains a new illumination hit."""
    from smartscan.predict.symbols import symbolize

    for tr in tracker.tracks.values():
        if tr.best_channel != channel or np.isnan(tr.pri):
            continue
        if not tr.hits:
            continue
        key = round(tr.hits[-1].t_first, 4)
        if getattr(tr, "_last_word_hit", None) != key:
            tr._last_word_hit = key
            if getattr(tr, "words", None) is None:
                tr.words = []
            tr.words.append(int(symbolize(tr.pri, float(np.exp(tr.log_pw)))))
