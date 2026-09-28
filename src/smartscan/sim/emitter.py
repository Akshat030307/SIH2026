"""Emitter models.

An emitter has:
  * geometry: bearing and range from the receiver,
  * a transmitter: peak power, antenna gain, beamwidth, sidelobe level,
  * a scan type:
      - ``circular``: a mechanically rotating antenna. The main beam sweeps
        the receiver once per rotation period T_e.
      - ``electronic``: an AESA / MFR. The beam dwells on the receiver's
        direction for ``illum`` seconds every ``revisit`` seconds; both depend
        on the current mode.
  * one or more *segments*, each a time span in which the waveform and scan
    parameters are fixed. A non-MFR emitter has one segment
    [t_on, t_off]. An MFR has one segment per mode visit, drawn from a
    semi-Markov mode chain.

The pulse train is deterministic and generated lazily for any time window.
Per-pulse randomness (jitter, frequency hops) comes from a counter-based hash
of (seed, pulse index), so calling ``pulses(a, b)`` twice returns identical
pulses, and any window can be queried without simulating the pulses before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from smartscan.sim.propagation import beam_gain_db, mainlobe_halfwidth_deg

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def hash_uniform(seed: int, idx: np.ndarray) -> np.ndarray:
    """Counter-based U[0,1) via splitmix64. Vectorized, deterministic."""
    with np.errstate(over="ignore"):
        z = np.asarray(idx, dtype=np.int64).astype(np.uint64) + np.uint64(seed & 0xFFFFFFFFFFFF) * _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (2.0**-53)


@dataclass
class Waveform:
    intervals: np.ndarray  # PRI sequence [s], repeated cyclically
    pw: float  # pulse width [s]
    rf: np.ndarray  # carrier set [Hz]; len 1 → fixed frequency
    jitter: float = 0.0  # fractional PRI jitter (± jitter·PRI)
    hop: str = "fixed"  # fixed | pulse | burst
    burst_len: int = 16
    pri_type: str = "fixed"  # descriptive label: fixed|stagger|jitter|sliding|dwell_switch

    def __post_init__(self):
        self.intervals = np.atleast_1d(np.asarray(self.intervals, dtype=float))
        self.rf = np.atleast_1d(np.asarray(self.rf, dtype=float))
        self._offsets = np.concatenate([[0.0], np.cumsum(self.intervals)[:-1]])
        self._period = float(self.intervals.sum())

    @property
    def mean_pri(self) -> float:
        return self._period / len(self.intervals)


@dataclass
class Segment:
    t0: float
    t1: float
    mode: int
    wf: Waveform
    revisit: float = 0.0  # electronic scan only
    illum: float = 0.0  # electronic scan only
    phase: float = 0.0  # time of first illumination after t0 (electronic) / pulse-train phase
    seed: int = 0


@dataclass
class Emitter:
    id: int
    name: str
    cls: str
    priority: float  # threat weight (ground truth; the scheduler must estimate its own)
    bearing_deg: float
    range_m: float
    pt_w: float
    gain_db: float
    beamwidth_deg: float
    sidelobe_db: float
    scan: str  # circular | electronic
    segments: list[Segment]
    scan_period: float = 0.0  # circular only [s]
    scan_phase: float = 0.0  # circular only: time at which the beam first points at the receiver
    mode_names: list[str] = field(default_factory=lambda: ["default"])
    mode_priority: list[float] | None = None  # per-mode threat weight (MFR)

    @property
    def t_on(self) -> float:
        return self.segments[0].t0

    @property
    def t_off(self) -> float:
        return self.segments[-1].t1

    @property
    def rf_all(self) -> np.ndarray:
        return np.unique(np.concatenate([s.wf.rf for s in self.segments]))

    def priority_at(self, mode: int) -> float:
        if self.mode_priority is None:
            return self.priority
        return self.mode_priority[mode]

    # ------------------------------------------------------------------ gain
    def gain_at(self, t: np.ndarray, seg: Segment | None = None) -> np.ndarray:
        """Transmit antenna gain toward the receiver [dBi] at times ``t``."""
        t = np.asarray(t, dtype=float)
        if self.scan == "circular":
            offset = 360.0 * (t - self.scan_phase) / self.scan_period
            return beam_gain_db(offset, self.gain_db, self.beamwidth_deg, self.sidelobe_db)
        # electronic: flat main beam during illumination windows, sidelobe otherwise
        g = np.full(t.shape, self.gain_db + self.sidelobe_db)
        segs = [seg] if seg is not None else self.segments
        for s in segs:
            m = (t >= s.t0) & (t < s.t1)
            if not m.any():
                continue
            ph = (t[m] - s.t0 - s.phase) % s.revisit
            g[m] = np.where(ph < s.illum, self.gain_db, g[m])
        return g

    # ------------------------------------------------------ illumination
    def illumination_windows(self, t0: float, t1: float, margin_db: float = 3.0) -> np.ndarray:
        """Intervals in [t0, t1) where gain ≥ peak − ``margin_db``.

        Returns an (N, 3) array of (start, end, mode). With the default 3 dB
        margin these are the "illumination events" used for scoring.
        """
        out = []
        for s in self.segments:
            a, b = max(t0, s.t0), min(t1, s.t1)
            if a >= b:
                continue
            if self.scan == "circular":
                hw = mainlobe_halfwidth_deg(margin_db, self.beamwidth_deg) / 360.0 * self.scan_period
                k0 = int(np.floor((a - self.scan_phase - hw) / self.scan_period))
                k1 = int(np.ceil((b - self.scan_phase + hw) / self.scan_period))
                centers = self.scan_phase + np.arange(k0, k1 + 1) * self.scan_period
                starts, ends = centers - hw, centers + hw
            else:
                k0 = int(np.floor((a - s.t0 - s.phase) / s.revisit))
                k1 = int(np.ceil((b - s.t0 - s.phase) / s.revisit))
                starts = s.t0 + s.phase + np.arange(max(k0, 0), k1 + 1) * s.revisit
                ends = starts + s.illum
            starts, ends = np.maximum(starts, a), np.minimum(ends, b)
            keep = ends > starts
            for st, en in zip(starts[keep], ends[keep]):
                out.append((st, en, s.mode))
        return np.array(out, dtype=float).reshape(-1, 3)

    # ------------------------------------------------------------ pulses
    def pulses(self, t0: float, t1: float):
        """Emitted pulses with TOA in [t0, t1).

        Returns (toa, rf, pw, mode, gain_db) arrays. Gain is the transmit gain
        toward the receiver at each pulse's TOA.
        """
        toas, rfs, pws, modes, gains = [], [], [], [], []
        for si, s in enumerate(self.segments):
            a, b = max(t0, s.t0), min(t1, s.t1)
            if a >= b:
                continue
            wf = s.wf
            m = len(wf.intervals)
            S = wf._period
            jmax = wf.jitter * wf.intervals.max()
            base = s.t0 + (s.phase % S if self.scan == "circular" else 0.0)
            c_lo = int(np.floor((a - base - jmax) / S))
            c_hi = int(np.ceil((b - base + jmax) / S))
            cycles = np.arange(max(c_lo, 0), max(c_hi, 0) + 1)
            if len(cycles) == 0:
                continue
            k = (cycles[:, None] * m + np.arange(m)[None, :]).ravel()
            t = base + (k // m) * S + wf._offsets[k % m]
            if wf.jitter > 0:
                u = hash_uniform(s.seed * 7919 + 1, k)
                t = t + (2.0 * u - 1.0) * wf.jitter * wf.intervals[k % m]
            sel = (t >= a) & (t < b)
            t, k = t[sel], k[sel]
            if len(t) == 0:
                continue
            if wf.hop == "fixed" or len(wf.rf) == 1:
                rf = np.full(len(t), wf.rf[0])
            else:
                key = k if wf.hop == "pulse" else k // wf.burst_len
                rf = wf.rf[(hash_uniform(s.seed * 104729 + 3, key) * len(wf.rf)).astype(int)]
            toas.append(t)
            rfs.append(rf)
            pws.append(np.full(len(t), wf.pw))
            modes.append(np.full(len(t), s.mode, dtype=np.int16))
            gains.append(self.gain_at(t, s))
        if not toas:
            e = np.empty(0)
            return e, e, e, np.empty(0, dtype=np.int16), e
        return (
            np.concatenate(toas),
            np.concatenate(rfs),
            np.concatenate(pws),
            np.concatenate(modes),
            np.concatenate(gains),
        )

    def mode_at(self, t: float) -> int:
        for s in self.segments:
            if s.t0 <= t < s.t1:
                return s.mode
        return -1
