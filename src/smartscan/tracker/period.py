"""Scan-period estimation from sparse, unevenly spaced illumination hits.

The PDF estimates the rotation period from the autocorrelation of one
continuous SNR series. A scanning receiver rarely has that: it only sees an
emitter on the occasions it happens to be tuned to the right channel while the
main beam passes. So the main estimator here works on the *hit times* t_i
(beam-center estimates):

  * every pair difference must be close to an integer multiple of the period
    P: d_ij ≈ n_ij·P;
  * each divisor P/k fits the hits equally well, so we take the **largest**
    P that fits (an approximate GCD), bounded above by the smallest spacing
    between hits (one hit per illumination);
  * then (φ, P) is refined by least squares on t_i = φ + n_i·P.

``autocorr_period`` implements the PDF's SNR-autocorrelation estimator for
the case where a continuous dwell is affordable (used by tests and demos).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PeriodEstimate:
    period: float
    phase: float  # time of a reference illumination center
    resid: float  # RMS fit residual [s]
    n_hits: int
    score: float
    period_std: float

    def next_center(self, t: float) -> tuple[float, int]:
        """First predicted center strictly after ``t``, and cycles ahead of the last hit."""
        k = np.floor((t - self.phase) / self.period) + 1
        return self.phase + k * self.period, int(k)

    def center_std(self, cycles_ahead: int) -> float:
        return float(np.hypot(self.resid, abs(cycles_ahead) * self.period_std))


def estimate_period(
    hits: np.ndarray,
    sigmas: np.ndarray | None = None,
    p_min: float = 0.05,
    p_max: float = 15.0,
    tol: float = 0.006,
    min_hits: int = 3,
    min_score: float = 0.85,
) -> PeriodEstimate | None:
    order = np.argsort(np.asarray(hits, dtype=float))
    t = np.asarray(hits, dtype=float)[order]
    sig = np.full(len(t), tol) if sigmas is None else np.maximum(np.asarray(sigmas, dtype=float)[order], 1e-4)
    if len(t) < min_hits:
        return None
    gaps = np.diff(t)
    if gaps.min() <= 0:
        return None
    hi = min(p_max, gaps.min() * 1.02)
    if hi <= p_min:
        return None
    i, j = np.triu_indices(len(t), 1)
    d = t[j] - t[i]
    tij = np.hypot(sig[i], sig[j])
    span = d.max()
    # relative grid resolution small enough that the largest difference stays within tolerance
    step = max(tij.min() / (4.0 * span), 1e-5)
    grid = np.exp(np.arange(np.log(hi), np.log(p_min), -step))  # descending
    # chunk to bound memory
    best = None
    for c0 in range(0, len(grid), 4096):
        g = grid[c0:c0 + 4096][:, None]
        r = d[None, :] - np.round(d[None, :] / g) * g
        score = np.exp(-0.5 * (r / tij[None, :]) ** 2).mean(axis=1)
        ok = np.flatnonzero(score >= min_score)
        if len(ok):
            best = float(grid[c0 + ok[0]])  # largest period meeting the score
            break
    if best is None:
        return None
    # least-squares refinement
    n = np.round((t - t[0]) / best)
    if len(np.unique(n)) < 2:
        return None
    A = np.stack([np.ones_like(n), n], axis=1)
    wts = 1.0 / sig
    coef, *_ = np.linalg.lstsq(A * wts[:, None], t * wts, rcond=None)
    phase, period = float(coef[0]), float(coef[1])
    res = t - A @ coef
    dof = max(len(t) - 2, 1)
    chi2 = float(((res * wts) ** 2).sum() / dof)
    # residual of the *fit* at a well-measured point, inflated if the hits disagree with the model
    resid = max(float(sig.min()) * np.sqrt(max(chi2, 1.0)), 1e-4)
    cov = max(chi2, 1.0) * np.linalg.pinv((A * wts[:, None]).T @ (A * wts[:, None]))
    rr = d - np.round(d / period) * period
    score = float(np.exp(-0.5 * (rr / tij) ** 2).mean())
    # anchor the phase at the most recent hit to keep extrapolation short
    phase = phase + n[-1] * period
    return PeriodEstimate(period, phase, resid, len(t), score, float(np.sqrt(max(cov[1, 1], 0.0))))


def autocorr_period(snr_series: np.ndarray, dt: float, p_min: float, p_max: float) -> float | None:
    """PDF §Interception Algorithms: ρ̂(k) = 1/N Σ (SNR_i − μ)(SNR_{i+k} − μ).

    Returns the lag of the strongest autocorrelation peak in [p_min, p_max].
    """
    x = np.asarray(snr_series, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 4:
        return None
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n] / n
    k0, k1 = int(np.ceil(p_min / dt)), min(int(p_max / dt), n - 1)
    if k1 <= k0:
        return None
    seg = ac[k0:k1 + 1]
    # local maxima only
    peaks = np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])) + 1
    if len(peaks) == 0:
        return None
    k = k0 + peaks[np.argmax(seg[peaks])]
    return float(k * dt)
