import numpy as np
import pytest
import torch

from smartscan import pdw as P
from smartscan.deinterleave.classical import dbscan_deinterleave, hdbscan_deinterleave
from smartscan.deinterleave.dataset import N_FREQ, WINDOW, make_windows, pulse_inputs
from smartscan.deinterleave.learned import PulseEncoder, embed, learned_deinterleave


def _synthetic_pdws(n: int = 500, n_emitters: int = 3, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pdws = P.empty(n)
    emitters = rng.integers(0, n_emitters, n)
    pdws["emitter"] = emitters

    base_rf = np.linspace(9e9, 11e9, n_emitters)
    base_pw = np.linspace(1e-6, 10e-6, n_emitters)
    base_aoa = np.linspace(30.0, 150.0, n_emitters)

    pdws["rf"] = base_rf[emitters] + rng.normal(0, 1e6, n)
    pdws["pw"] = base_pw[emitters] + rng.normal(0, 0.1e-6, n)
    pdws["aoa"] = (base_aoa[emitters] + rng.normal(0, 1.0, n)) % 360.0
    pdws["amp"] = rng.uniform(20.0, 45.0, n)
    pdws["toa"] = np.sort(rng.uniform(0.0, 0.5, n))
    pdws["mode"] = 0
    return pdws


def test_pulse_inputs():
    pdws = _synthetic_pdws(n=100)
    feat = pulse_inputs(pdws)
    assert feat.shape == (100, 5 + 2 * N_FREQ)
    assert np.isfinite(feat).all()
    assert np.all(feat[:, :5] > -20.0) and np.all(feat[:, :5] < 20.0)
    assert np.all(feat[:, 5:] >= -1.0 - 1e-5) and np.all(feat[:, 5:] <= 1.0 + 1e-5)


def test_make_windows():
    pdws = _synthetic_pdws(n=600)
    rng = np.random.default_rng(0)
    windows = make_windows(pdws, rng, n=5, window=WINDOW)
    assert len(windows) == 5
    for xs, ys in windows:
        assert xs.shape == (WINDOW, 5 + 2 * N_FREQ)
        assert ys.shape == (WINDOW,)
        assert np.isfinite(xs).all()


def test_pulse_encoder_model():
    model = PulseEncoder(d_in=5 + 2 * N_FREQ, d=64, layers=2, heads=2, d_out=16)
    x = torch.randn(4, 64, 5 + 2 * N_FREQ)
    out = model(x)
    assert out.shape == (4, 64, 16)
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_embed_short_and_long_sequences():
    model = PulseEncoder(d_in=5 + 2 * N_FREQ, d=64, layers=2, heads=2, d_out=16).eval()

    # Short sequence (less than WINDOW)
    short_pdws = _synthetic_pdws(n=50)
    z_short = embed(model, short_pdws, window=WINDOW)
    assert z_short.shape == (50, 16)
    assert np.isfinite(z_short).all()
    norms_short = np.linalg.norm(z_short, axis=-1)
    assert np.allclose(norms_short, 1.0, atol=1e-4)

    # Long sequence (multiple WINDOWs)
    long_pdws = _synthetic_pdws(n=600)
    z_long = embed(model, long_pdws, window=WINDOW, batch_size=16)
    assert z_long.shape == (600, 16)
    assert np.isfinite(z_long).all()
    norms_long = np.linalg.norm(z_long, axis=-1)
    assert np.allclose(norms_long, 1.0, atol=1e-4)


def test_classical_deinterleave():
    pdws = _synthetic_pdws(n=300, n_emitters=3)
    labels_db = dbscan_deinterleave(pdws)
    assert len(labels_db) == 300
    assert len(np.unique(labels_db)) >= 2

    labels_hdb = hdbscan_deinterleave(pdws, min_cluster_size=15)
    assert len(labels_hdb) == 300
    assert len(np.unique(labels_hdb)) >= 2


def test_learned_deinterleave():
    model = PulseEncoder(d_in=5 + 2 * N_FREQ, d=64, layers=2, heads=2, d_out=16).eval()
    pdws = _synthetic_pdws(n=400, n_emitters=3)
    labels = learned_deinterleave(model, pdws, min_cluster_size=20)
    assert len(labels) == 400
    assert labels.dtype == np.int64 or labels.dtype == np.int32
