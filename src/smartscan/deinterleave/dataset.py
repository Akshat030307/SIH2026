"""Training windows for the learned deinterleaver (torch-free, cached to disk)."""

from __future__ import annotations

import numpy as np

from smartscan.sim.world import ROOT

WINDOW = 256
N_FREQ = 16
CACHE = ROOT / "data" / "deinterleave_windows.npz"


PERIODS = np.geomspace(20e-6, 20e-3, N_FREQ).astype(np.float32)
TWO_PI_OVER_PERIODS = (2 * np.pi / PERIODS).astype(np.float32)


def pulse_inputs(pdws: np.ndarray) -> np.ndarray:
    """(N, 5 + 2·N_FREQ) float32 inputs. Time features are relative to the first pulse."""
    a = np.deg2rad(pdws["aoa"])
    toa = pdws["toa"] - pdws["toa"][0]
    ph = toa[:, None] * TWO_PI_OVER_PERIODS[None, :]
    static = np.stack([
        (pdws["rf"] - 10e9) / 5e9,
        np.log10(np.maximum(pdws["pw"], 1e-9) * 1e6),
        np.sin(a), np.cos(a),
        (pdws["amp"] - 30.0) / 15.0,
    ], axis=1)
    return np.concatenate([static, np.sin(ph), np.cos(ph)], axis=1).astype(np.float32)


def make_windows(pdws: np.ndarray, rng, n: int, window: int = WINDOW):
    x = pulse_inputs(pdws)
    y = pdws["emitter"].astype(np.int64)
    toa = pdws["toa"]
    L = len(x)
    out = []
    for _ in range(n):
        s = 0 if L <= window else int(rng.integers(0, L - window))
        xs = x[s : s + window].copy()
        ys = y[s : s + window]
        dt = (toa[s : s + window] - toa[s])[:, None]
        ph = dt * TWO_PI_OVER_PERIODS[None, :]
        xs[:, 5 : 5 + N_FREQ] = np.sin(ph)
        xs[:, 5 + N_FREQ :] = np.cos(ph)
        out.append((xs, ys))
    return out


def build_training_set(seeds, scenarios=("S7_colocated", "S2_dense"), duration=1.5, per_stream=40):
    from smartscan.deinterleave.data import busiest_channels, simulate_stream

    rng = np.random.default_rng(0)
    data = []
    for sc in scenarios:
        for s in seeds:
            ch = busiest_channels(sc, s, k=6)
            p = simulate_stream(sc, seed=s, duration=duration, channels=ch, t0=float(rng.uniform(0, 30)))
            if len(p) < 64:
                continue
            data += make_windows(p, rng, per_stream)
    return data



def cached_training_set(n_seeds: int = 120):
    """Build (or load) the training windows as (X: (N, WINDOW, F), Y: (N, WINDOW), M: (N, WINDOW))."""
    if CACHE.exists():
        z = np.load(CACHE)
        if int(z["n_seeds"]) == n_seeds:
            return z["X"], z["Y"], z["M"]
    data = build_training_set(range(n_seeds))
    F = data[0][0].shape[1]
    X = np.zeros((len(data), WINDOW, F), np.float32)
    Y = np.full((len(data), WINDOW), -1, np.int64)
    M = np.zeros((len(data), WINDOW), bool)
    for i, (x, y) in enumerate(data):
        X[i, :len(x)], Y[i, :len(y)], M[i, :len(x)] = x, y, True
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X, Y=Y, M=M, n_seeds=n_seeds)
    return X, Y, M
