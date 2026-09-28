"""Discrete-event RF environment engine.

The receiver repeatedly picks (channel, dwell). The engine then:
  1. applies the local-oscillator retune dead time if the channel changed,
  2. generates every emitter pulse whose carrier lies in that channel during
     the listening window,
  3. runs each pulse through the one-way link budget and the detection
     threshold, adds measurement noise, and injects noise-triggered false PDWs,
  4. books detected pulses against ground-truth *illumination events* (periods
     when an emitter's main beam puts the receiver above threshold) for scoring.

An illumination event counts as intercepted once ``n_min`` of its pulses have
been detected. This is the unit behind Pd and intercept rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartscan import pdw as P
from smartscan.sim.propagation import snr_db
from smartscan.sim.world import World


@dataclass
class GroundTruth:
    """All illumination events of the episode, in one flat table."""

    emitter: np.ndarray  # (E,) emitter id
    start: np.ndarray
    end: np.ndarray
    mode: np.ndarray
    priority: np.ndarray
    peak_snr: np.ndarray
    count: np.ndarray  # detected pulses so far
    intercept_time: np.ndarray  # NaN until intercepted
    by_emitter: dict  # emitter id -> event indices sorted by start

    @property
    def n(self) -> int:
        return len(self.emitter)

    @property
    def intercepted(self) -> np.ndarray:
        return ~np.isnan(self.intercept_time)


@dataclass
class StepResult:
    t_start: float
    t_listen: float
    t_end: float
    channel: int
    dwell_idx: int
    retuned: bool
    pdws: np.ndarray  # includes ground-truth labels; strip before passing to schedulers
    new_intercepts: np.ndarray  # event indices intercepted during this step
    missed: np.ndarray  # event indices that ended during this step without being intercepted
    done: bool


class RFEngine:
    def __init__(self, world: World, n_min: int = 3):
        self.world = world
        self.rx = world.receiver
        self.n_min = n_min
        self.emitters = world.emitters
        rx = self.rx
        noise = rx.noise_dbw

        # per-emitter link budget constants and pulse-generation windows
        self._peak_snr = np.zeros(len(self.emitters))
        self._sidelobe_visible = np.zeros(len(self.emitters), dtype=bool)
        self._gen_windows: list[np.ndarray] = []
        self._chan_emitters: list[list[int]] = [[] for _ in range(rx.n_channels)]
        ev_rows = []
        for e in self.emitters:
            f_mid = float(np.mean(e.rf_all))
            peak = float(snr_db(e.pt_w, e.gain_db, rx.gr_db, f_mid, e.range_m, rx.loss_db, noise))
            self._peak_snr[e.id] = peak
            pad = 3.0 * rx.amp_noise_db
            self._sidelobe_visible[e.id] = peak + e.sidelobe_db > rx.threshold_db - pad
            margin = peak - rx.threshold_db
            if margin > 0:
                for s, en, m in e.illumination_windows(0.0, world.duration, margin_db=min(margin, 40.0)):
                    ev_rows.append((e.id, s, en, int(m), e.priority_at(int(m)), peak))
            gen_margin = min(margin + pad, 45.0)
            self._gen_windows.append(
                e.illumination_windows(0.0, world.duration, margin_db=gen_margin)[:, :2]
                if gen_margin > 0 else np.zeros((0, 2))
            )
            for ch in np.unique(rx.channel_of(e.rf_all)):
                if ch >= 0:
                    self._chan_emitters[ch].append(e.id)

        ev = np.array(ev_rows, dtype=float).reshape(-1, 6)
        order = np.lexsort((ev[:, 1], ev[:, 0]))
        ev = ev[order]
        emitter_ids = ev[:, 0].astype(int)
        by_emitter = {int(i): np.flatnonzero(emitter_ids == i) for i in np.unique(emitter_ids)}
        self.truth = GroundTruth(
            emitter=emitter_ids, start=ev[:, 1], end=ev[:, 2], mode=ev[:, 3].astype(int),
            priority=ev[:, 4], peak_snr=ev[:, 5], count=np.zeros(len(ev), dtype=int),
            intercept_time=np.full(len(ev), np.nan), by_emitter=by_emitter,
        )
        self._end_order = np.argsort(self.truth.end, kind="stable")
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None):
        self.rng = np.random.default_rng(self.world.seed * 1_000_003 + 17 if seed is None else seed)
        self.t = 0.0
        self.channel = -1
        self.truth.count[:] = 0
        self.truth.intercept_time[:] = np.nan
        self._end_ptr = 0
        return self

    @property
    def done(self) -> bool:
        return self.t >= self.world.duration

    def step(self, channel: int, dwell_idx: int) -> StepResult:
        rx = self.rx
        t0 = self.t
        retuned = channel != self.channel
        t_listen = t0 + (rx.tune_s if retuned else 0.0)
        t_end = min(t_listen + rx.dwells_s[dwell_idx], self.world.duration)
        pdws = self._observe(channel, t_listen, t_end) if t_end > t_listen else P.empty()
        new = self._book(pdws)
        missed = self._advance_missed(t_end)
        self.t, self.channel = t_end, channel
        return StepResult(t0, t_listen, t_end, channel, dwell_idx, retuned, pdws, new, missed, self.done)

    # ------------------------------------------------------------ internals
    def _observe(self, ch: int, a: float, b: float) -> np.ndarray:
        rx, rng = self.rx, self.rng
        lo, hi = rx.channel_bounds(ch)
        chunks = []
        for eid in self._chan_emitters[ch]:
            e = self.emitters[eid]
            if b <= e.t_on or a >= e.t_off:
                continue
            if self._sidelobe_visible[eid]:
                spans = [(a, b)]
            else:
                w = self._gen_windows[eid]
                if len(w) == 0:
                    continue
                i0 = np.searchsorted(w[:, 1], a, side="right")
                spans = []
                for s, en in w[i0:]:
                    if s >= b:
                        break
                    spans.append((max(s, a), min(en, b)))
            for sa, sb in spans:
                toa, rf, pw, mode, gain = e.pulses(sa, sb)
                if len(toa) == 0:
                    continue
                inb = (rf >= lo) & (rf < hi)
                if not inb.any():
                    continue
                toa, rf, pw, mode, gain = toa[inb], rf[inb], pw[inb], mode[inb], gain[inb]
                snr = snr_db(e.pt_w, gain, rx.gr_db, rf, e.range_m, rx.loss_db, rx.noise_dbw)
                amp = snr + rng.normal(0.0, rx.amp_noise_db, len(snr))
                det = amp >= rx.threshold_db
                n = int(det.sum())
                if n == 0:
                    continue
                out = P.empty(n)
                out["toa"] = toa[det] + rng.normal(0.0, rx.toa_noise_s, n)
                out["rf"] = rf[det] + rng.normal(0.0, rx.rf_noise_hz, n)
                out["pw"] = pw[det] * (1.0 + rng.normal(0.0, rx.pw_noise_frac, n))
                out["aoa"] = (e.bearing_deg + rng.normal(0.0, rx.aoa_noise_deg, n)) % 360.0
                out["amp"] = amp[det]
                out["emitter"] = eid
                out["mode"] = mode[det]
                chunks.append(out)
        # noise-triggered false pulses
        nf = rng.poisson(rx.false_pulse_rate * (b - a))
        if nf:
            fp = P.empty(nf)
            fp["toa"] = rng.uniform(a, b, nf)
            fp["rf"] = rng.uniform(lo, hi, nf)
            fp["pw"] = np.exp(rng.uniform(np.log(0.1e-6), np.log(50e-6), nf))
            fp["aoa"] = rng.uniform(0, 360, nf)
            fp["amp"] = rx.threshold_db + rng.exponential(1.5, nf)
            fp["emitter"] = P.NOISE_ID
            fp["mode"] = -1
            chunks.append(fp)
        return P.concat(chunks)

    def _book(self, pdws: np.ndarray) -> np.ndarray:
        """Assign detected pulses to ground-truth events; return newly intercepted event indices."""
        gt = self.truth
        new = []
        if len(pdws) == 0:
            return np.zeros(0, dtype=int)
        for eid in np.unique(pdws["emitter"]):
            if eid < 0 or eid not in gt.by_emitter:
                continue
            idx = gt.by_emitter[eid]
            toa = pdws["toa"][pdws["emitter"] == eid]
            j = np.searchsorted(gt.start[idx], toa, side="right") - 1
            ok = j >= 0
            j, toa = j[ok], toa[ok]
            ev = idx[j]
            inside = toa <= gt.end[ev]
            ev, toa = ev[inside], toa[inside]
            for e_idx in np.unique(ev):
                before = gt.count[e_idx]
                ts = toa[ev == e_idx]
                gt.count[e_idx] += len(ts)
                if before < self.n_min <= gt.count[e_idx]:
                    gt.intercept_time[e_idx] = ts[self.n_min - before - 1]
                    new.append(e_idx)
        return np.array(new, dtype=int)

    def _advance_missed(self, t_end: float) -> np.ndarray:
        gt, order = self.truth, self._end_order
        i = self._end_ptr
        out = []
        while i < len(order) and gt.end[order[i]] <= t_end:
            k = order[i]
            if np.isnan(gt.intercept_time[k]):
                out.append(k)
            i += 1
        self._end_ptr = i
        return np.array(out, dtype=int)
