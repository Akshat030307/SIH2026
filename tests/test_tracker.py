import numpy as np
import pytest

from smartscan import pdw as P
from smartscan.tracker.period import autocorr_period, estimate_period
from smartscan.tracker.tracker import beam_center, cluster_dwell


def test_period_from_sparse_hits_with_gaps():
    rng = np.random.default_rng(0)
    T, phase = 3.7, 0.4
    k = np.array([0, 1, 3, 4, 7, 9])  # missed illuminations
    hits = phase + k * T + rng.normal(0, 0.004, len(k))
    est = estimate_period(hits, sigmas=np.full(len(k), 0.01))
    assert est is not None
    assert est.period == pytest.approx(T, rel=2e-3)
    nxt, _ = est.next_center(hits[-1] + 0.1)
    assert nxt == pytest.approx(phase + 10 * T, abs=0.03)


def test_period_rejects_random_hits():
    rng = np.random.default_rng(1)
    hits = np.sort(rng.uniform(0, 40, 5))
    est = estimate_period(hits, sigmas=np.full(5, 0.005))
    assert est is None or est.score < 0.95


def test_autocorrelation_period():
    dt, T = 0.01, 2.5
    t = np.arange(0, 20, dt)
    snr = 60 - 12 * ((((t - 0.3) % T) - T / 2) / 0.05) ** 2
    snr = np.maximum(snr, 15) + np.random.default_rng(0).normal(0, 1, len(t))
    assert autocorr_period(snr, dt, 0.5, 6.0) == pytest.approx(T, abs=2 * dt)


def test_beam_center_from_partial_view():
    tc = 1.000
    toa = np.linspace(0.985, 1.005, 30)  # sees the rising edge and a bit past the peak
    amp = 50 - 12 * ((toa - tc) / 0.02) ** 2
    c, ok = beam_center(toa, amp)
    assert ok and c == pytest.approx(tc, abs=1e-3)


def test_cluster_dwell_separates_emitters():
    p = P.empty(40)
    p["aoa"][:20] = 30 + np.random.default_rng(0).normal(0, 2, 20)
    p["aoa"][20:] = 200 + np.random.default_rng(1).normal(0, 2, 20)
    p["pw"] = 1e-6
    lab = cluster_dwell(p)
    assert len(set(lab[:20])) == 1 and len(set(lab[20:])) == 1 and lab[0] != lab[25]
