"""Radar-word symbolization and MFR behaviour sequences.

The receiver never sees the emitter's mode. Each observed illumination is
reduced to a discrete *radar word* built only from measurable parameters:

    symbol = PRI band (4 levels) × pulse-width band (3 levels)   →  12 symbols

Sequences of these symbols (one per illumination) are what the predictors
model. Simulator ground-truth modes are used **only** for scoring.
"""

from __future__ import annotations

import numpy as np

from smartscan.sim.engine import RFEngine
from smartscan.sim.world import build_world

PRI_EDGES = np.array([150e-6, 450e-6, 1000e-6])
PW_EDGES = np.array([3e-6, 10e-6])
N_SYMBOLS = (len(PRI_EDGES) + 1) * (len(PW_EDGES) + 1)


def symbolize(pri: np.ndarray, pw: np.ndarray) -> np.ndarray:
    pi = np.searchsorted(PRI_EDGES, np.asarray(pri))
    wi = np.searchsorted(PW_EDGES, np.asarray(pw))
    return (pi * (len(PW_EDGES) + 1) + wi).astype(int)


def emitter_sequences(scenario: str = "S3_mfr", seeds=range(50), classes=("mfr",), drop: float = 0.0,
                      pri_noise: float = 0.05, pw_noise: float = 0.08, rng_seed: int = 0):
    """Per-emitter (symbols, modes, times) from ground-truth illumination events.

    ``drop`` removes each illumination at random with that probability,
    mimicking a receiver that misses some illuminations.
    """
    rng = np.random.default_rng(rng_seed)
    out = []
    for s in seeds:
        world = build_world(scenario, seed=s)
        eng = RFEngine(world)
        gt = eng.truth
        for e in world.emitters:
            if e.cls not in classes or e.id not in gt.by_emitter:
                continue
            idx = gt.by_emitter[e.id]
            t = gt.start[idx]
            modes = gt.mode[idx]
            pri, pw = np.empty(len(idx)), np.empty(len(idx))
            for k, tt in enumerate(t):
                seg = next(sg for sg in e.segments if sg.t0 <= tt < sg.t1)
                pri[k] = seg.wf.mean_pri
                pw[k] = seg.wf.pw
            pri *= np.exp(rng.normal(0, pri_noise, len(pri)))
            pw *= np.exp(rng.normal(0, pw_noise, len(pw)))
            sym = symbolize(pri, pw)
            keep = rng.random(len(sym)) >= drop
            if keep.sum() >= 8:
                out.append((sym[keep], modes[keep], t[keep]))
    return out
