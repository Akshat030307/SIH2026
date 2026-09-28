"""Online deinterleaver and emitter track store.

Each dwell's truth-stripped PDWs are:
  1. clustered within the dwell by gap-splitting on AOA, then on log pulse width
     (fast, O(n log n), with no hyperparameters beyond the gates),
  2. matched to existing tracks by gated nearest neighbour on (AOA, PW, RF),
  3. turned into new tracks when an unmatched cluster has ≥ ``min_new`` pulses.
     Isolated noise PDWs rarely line up, so they are dropped.

A track keeps a *hit* list: one entry per illumination it was observed in,
holding a beam-center time estimate. Hits feed the period estimator, which
predicts the next illumination window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from smartscan.tracker.period import PeriodEstimate, estimate_period


def _circ_diff(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def _split_gaps(x: np.ndarray, gap: float, circular: bool = False) -> np.ndarray:
    """Label 1-D points into groups separated by gaps > ``gap``."""
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(x)
    xs = x[order]
    d = np.diff(xs)
    brk = d > gap
    labels_sorted = np.concatenate([[0], np.cumsum(brk)])
    if circular and n > 1 and (xs[0] + 360.0 - xs[-1]) <= gap and labels_sorted[-1] > 0:
        labels_sorted[labels_sorted == labels_sorted[-1]] = 0
    labels = np.empty(n, dtype=int)
    labels[order] = labels_sorted
    return labels


def cluster_dwell(pdws: np.ndarray, aoa_gap: float = 6.0, logpw_gap: float = 0.18) -> np.ndarray:
    """Cluster labels for one dwell's PDWs (AOA gaps, then log-PW gaps)."""
    if len(pdws) == 0:
        return np.zeros(0, dtype=int)
    la = _split_gaps(pdws["aoa"], aoa_gap, circular=True)
    lpw = np.log(np.maximum(pdws["pw"], 1e-9))
    out = np.full(len(pdws), -1)
    nxt = 0
    for g in np.unique(la):
        idx = np.flatnonzero(la == g)
        sub = _split_gaps(lpw[idx], logpw_gap)
        for s in np.unique(sub):
            out[idx[sub == s]] = nxt
            nxt += 1
    return out


@dataclass
class Hit:
    """One observed illumination (possibly stitched from several dwells)."""

    t_first: float
    t_last: float
    center: float
    amp: float
    sigma: float  # 1-σ uncertainty of ``center`` [s]
    n: int


@dataclass
class Track:
    id: int
    aoa: float
    log_pw: float
    rf_min: float
    rf_max: float
    amp_max: float
    created: float
    last_seen: float
    n_pulses: int = 0
    pri: float = np.nan
    channels: dict = field(default_factory=dict)
    est: PeriodEstimate | None = None
    misses: int = 0
    locked_hits: int = 0
    threat_cache: float = 1.0
    visits: int = 0  # dwells on the track's channel since creation
    seen: int = 0  # ... of which the track was detected
    hits: list = field(default_factory=list)  # derived from chunks by Tracker.refresh_estimate
    # per-visit records on this track's channel(s): (t_listen, t_end, amp_max, toa[], amp[], n)
    # amp_max = -inf and n = 0 for visits where the track was not detected
    chunks: list = field(default_factory=list)
    chunk_amps: list = field(default_factory=list)

    @property
    def agile(self) -> bool:
        return (self.rf_max - self.rf_min) > 30e6

    @property
    def rf(self) -> float:
        return 0.5 * (self.rf_min + self.rf_max)

    @property
    def best_channel(self) -> int:
        return max(self.channels.items(), key=lambda kv: kv[1])[0]

    @property
    def locked(self) -> bool:
        return self.est is not None

    @property
    def continuous(self) -> bool:
        """Seen on most visits → sidelobes are above threshold (strong / close emitter)."""
        return self.visits >= 5 and self.seen / self.visits > 0.6

    @property
    def amp_floor(self) -> float:
        if not self.chunk_amps:
            return -np.inf
        return float(np.percentile(self.chunk_amps[-60:], 30))

    def is_mainlobe(self, amp: float) -> bool:
        return (not self.continuous) or amp >= self.amp_floor + 10.0

    def illum_width(self) -> float:
        """Main-beam dwell on the receiver, from the widest stitched observation."""
        if not self.hits:
            return 0.05
        return float(np.clip(max(h.t_last - h.t_first for h in self.hits), 0.005, 0.3))

    def mainlobe_hits(self) -> list[Hit]:
        return self.hits


class Tracker:
    def __init__(
        self,
        rx,
        aoa_gate: float = 9.0,
        logpw_gate: float = 0.3,
        rf_gate: float = 40e6,
        min_new: int = 3,
        merge_gap: float = 0.25,
        max_chunks: int = 1000,
        max_hits: int = 12,
        drop_after: float = 20.0,
    ):
        self.rx = rx
        self.aoa_gate, self.logpw_gate, self.rf_gate = aoa_gate, logpw_gate, rf_gate
        self.min_new, self.merge_gap = min_new, merge_gap
        self.max_chunks, self.max_hits = max_chunks, max_hits
        self.drop_after = drop_after
        self.tracks: dict[int, Track] = {}
        self._next = 0

    # --------------------------------------------------------------- update
    def update(self, t_listen: float, t_end: float, channel: int, pdws: np.ndarray):
        """Returns (assignments per PDW, list of (track, is_new, n_pulses) touched this dwell).

        A touched track whose new detection could be a main-lobe hit gets
        ``_dirty = True``; call ``refresh_estimate`` on it.
        """
        assign = np.full(len(pdws), -1)
        touched: list[tuple[Track, bool, int]] = []
        watch = [tr for tr in self.tracks.values() if tr.channels and tr.best_channel == channel]
        for tr in watch:
            tr.visits += 1
        self._t_listen = t_listen
        if len(pdws) == 0:
            self._log_empty(watch, t_listen, t_end)
            return assign, touched
        labels = cluster_dwell(pdws)
        for lab in np.unique(labels):
            idx = np.flatnonzero(labels == lab)
            c = pdws[idx]
            aoa = float(np.rad2deg(np.angle(np.exp(1j * np.deg2rad(c["aoa"])).mean())) % 360.0)
            lpw = float(np.log(np.maximum(c["pw"], 1e-9)).mean())
            rf = float(np.median(c["rf"]))
            tr = self._associate(aoa, lpw, rf, len(idx))
            is_new = False
            if tr is None:
                if len(idx) < self.min_new:
                    continue
                tr = Track(self._next, aoa, lpw, rf, rf, float(c["amp"].max()), t_listen, t_end)
                self.tracks[tr.id] = tr
                self._next += 1
                is_new = True
                tr.visits = 1
            self._absorb(tr, c, aoa, lpw, channel, t_end)
            assign[idx] = tr.id
            touched.append((tr, is_new, len(idx)))
        seen_ids = {tr.id for tr, _, _ in touched}
        self._log_empty([tr for tr in watch if tr.id not in seen_ids], t_listen, t_end)
        return assign, touched

    def _log_empty(self, tracks, t_listen, t_end):
        for tr in tracks:
            tr.chunks.append((t_listen, t_end, -np.inf, None, None, 0))
            if len(tr.chunks) > self.max_chunks:
                del tr.chunks[: len(tr.chunks) - self.max_chunks]

    def _associate(self, aoa, lpw, rf, n) -> Track | None:
        best, best_cost = None, 1.0
        for tr in self.tracks.values():
            da = abs(_circ_diff(aoa, tr.aoa)) / self.aoa_gate
            if da >= 1.0:
                continue
            dp = abs(lpw - tr.log_pw) / self.logpw_gate
            if dp >= 1.0:
                continue
            if rf < tr.rf_min - self.rf_gate or rf > tr.rf_max + self.rf_gate:
                # a carrier jump with matching AOA/PW is most likely a frequency hop of the same emitter
                dr = 0.2 if tr.agile else 0.5
            else:
                dr = 0.0
            cost = max(da, dp) + dr
            if cost < best_cost:
                best, best_cost = tr, cost
        return best

    def _absorb(self, tr: Track, c: np.ndarray, aoa, lpw, channel, t_end):
        n = len(c)
        w = n / (tr.n_pulses + n)
        tr.aoa = float((tr.aoa + w * _circ_diff(aoa, tr.aoa)) % 360.0)
        tr.log_pw = float(tr.log_pw + w * (lpw - tr.log_pw))
        tr.rf_min = min(tr.rf_min, float(np.percentile(c["rf"], 5)))
        tr.rf_max = max(tr.rf_max, float(np.percentile(c["rf"], 95)))
        tr.n_pulses += n
        tr.seen += 1
        tr.channels[channel] = tr.channels.get(channel, 0) + n
        amax = float(c["amp"].max())
        tr.amp_max = max(tr.amp_max, amax)
        tr.chunk_amps.append(amax)
        if n >= 3:
            pri = float(np.median(np.diff(np.sort(c["toa"]))))
            tr.pri = pri if np.isnan(tr.pri) else min(tr.pri * 1.02, 0.7 * tr.pri + 0.3 * pri)
        toa, amp = c["toa"], c["amp"]
        if len(toa) > 64:  # keep the parabola fit cheap
            k = np.linspace(0, len(toa) - 1, 64).astype(int)
            toa, amp = toa[k], amp[k]
        if tr.chunks and tr.chunks[-1][0] == self._t_listen and tr.chunks[-1][5] > 0:
            # a second cluster of the same track in this dwell: merge
            p = tr.chunks[-1]
            tr.chunks[-1] = (p[0], p[1], max(p[2], amax), np.concatenate([p[3], toa]),
                             np.concatenate([p[4], amp]), p[5] + n)
        else:
            tr.chunks.append((self._t_listen, t_end, amax, toa, amp, n))
        if len(tr.chunks) > self.max_chunks:
            del tr.chunks[: len(tr.chunks) - self.max_chunks]
        if len(tr.chunk_amps) > self.max_chunks:
            del tr.chunk_amps[: len(tr.chunk_amps) - self.max_chunks]
        tr._dirty = n >= 2 and tr.is_mainlobe(amax)
        tr.last_seen = t_end

    # ------------------------------------------------------------ hits / lock
    def build_hits(self, tr: Track) -> list[Hit]:
        """Group main-lobe chunks into illuminations and estimate each beam center."""
        floor = tr.amp_floor
        cont = tr.continuous
        pri = 0.003 if np.isnan(tr.pri) else tr.pri
        groups: list[list] = []
        open_group = False
        for ch in tr.chunks:
            kept = ch[5] >= 2 and (not cont or ch[2] >= floor + 10.0)
            if kept:
                if open_group and ch[0] - groups[-1][-1][1] < self.merge_gap:
                    groups[-1].append(ch)
                else:
                    groups.append([ch])
                open_group = True
            elif ch[1] - ch[0] >= 2.5 * pri:
                # a long-enough look that saw no main-lobe pulses ends the current illumination
                open_group = False
        hits, fits = [], []
        for g in groups[-self.max_hits:]:
            toa = np.concatenate([x[3] for x in g])
            amp = np.concatenate([x[4] for x in g])
            center, fit_ok = beam_center(toa, amp)
            hits.append(Hit(float(toa.min()), float(toa.max()), center, float(amp.max()), 0.0, len(toa)))
            fits.append(fit_ok)
        width = float(np.clip(max((h.t_last - h.t_first for h in hits), default=0.05), 0.005, 0.3))
        for h, fit_ok in zip(hits, fits):
            # a clean parabola vertex or a near-peak sample pins the beam center; an edge-only view does not
            h.sigma = 0.01 if (fit_ok or h.amp >= tr.amp_max - 6.0) else max(0.01, 0.3 * width)
        tr.hits = hits
        return hits

    def refresh_estimate(self, tr: Track) -> None:
        if not getattr(tr, "_dirty", True):
            return
        tr._dirty = False
        hits = self.build_hits(tr)
        if len(hits) < 3:
            tr.est = None
            return
        tr.est = estimate_period(np.array([h.center for h in hits]),
                                 sigmas=np.array([h.sigma for h in hits]))

    def register_miss(self, tr: Track, max_misses: int = 2) -> None:
        tr.misses += 1
        if tr.misses >= max_misses:
            # probably a mode change or a wrong (multiple) period: restart from recent evidence
            tr.est = None
            keep = [c for c in tr.chunks if c[1] >= tr.last_seen - 1.0]
            tr.chunks = keep or tr.chunks[-1:]
            tr.misses = 0
            tr._dirty = True

    def prune(self, t: float) -> None:
        stale = [k for k, tr in self.tracks.items() if t - tr.last_seen > self.drop_after]
        for k in stale:
            del self.tracks[k]


def beam_center(toa: np.ndarray, amp: np.ndarray) -> tuple[float, bool]:
    """Estimate when the beam pointed at the receiver during one illumination.

    For a Gaussian main lobe, SNR in dB is quadratic in time, so fit a parabola
    and take its vertex (this works from a partial view of the beam). Fall back
    to the time of peak amplitude when the fit isn't concave or is unreliable.
    """
    if len(toa) >= 6 and toa.max() - toa.min() > 0:
        t0 = toa.mean()
        x = toa - t0
        try:
            a, b, _ = np.polyfit(x, amp, 2)
            if a < 0:
                v = -b / (2 * a)
                span = x.max() - x.min()
                if abs(v) < 0.75 * span + 0.003:
                    return float(t0 + v), True
        except (np.linalg.LinAlgError, ValueError):
            pass
    k = int(np.argmax(amp))
    return float(toa[k]), False


def infer_threat(tr: Track) -> float:
    """Threat weight from observables only (no pre-mission database).

    Higher carrier (X/Ku fire-control bands), high PRF, frequency agility and
    fast revisit all point to a lethal emitter.
    """
    s = 1.0
    f = tr.rf / 1e9
    s += 1.0 if f >= 8.0 else (0.5 if f >= 4.0 else 0.0)
    if not np.isnan(tr.pri):
        s += 1.5 if tr.pri < 200e-6 else (0.5 if tr.pri < 600e-6 else 0.0)
    if tr.agile:
        s += 1.0
    if tr.est is not None and tr.est.period < 1.0:
        s += 1.0
    return float(min(s, 5.0))
