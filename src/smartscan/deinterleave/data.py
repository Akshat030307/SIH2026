"""Labelled interleaved pulse streams for deinterleaving research.

``simulate_stream`` records every detectable pulse in a set of channels over
a time span, like the Turing dataset's "stare" receiver restricted to those
channels. The PDWs carry ground-truth emitter labels.
"""

from __future__ import annotations

import numpy as np

from smartscan import pdw as P
from smartscan.sim.engine import RFEngine
from smartscan.sim.world import build_world


def simulate_stream(scenario: str = "S2_dense", seed: int = 0, t0: float = 0.0, duration: float = 2.0,
                    channels: list[int] | None = None, include_noise: bool = False) -> np.ndarray:
    world = build_world(scenario, seed=seed)
    eng = RFEngine(world).reset(seed=seed + 12345)
    rx = world.receiver
    if channels is None:
        channels = list(range(rx.n_channels))
    chunks = [eng._observe(ch, t0, t0 + duration) for ch in channels]
    out = P.concat(chunks)
    if not include_noise:
        out = out[out["emitter"] >= 0]
    return out


def busiest_channels(scenario: str, seed: int, k: int = 4, duration: float = 1.0) -> list[int]:
    world = build_world(scenario, seed=seed)
    counts = np.zeros(world.receiver.n_channels)
    for e in world.emitters:
        for ch in np.unique(world.receiver.channel_of(e.rf_all)):
            counts[ch] += 1
    return [int(c) for c in np.argsort(-counts)[:k]]


def features(pdws: np.ndarray, rf_scale: float = 1e9) -> np.ndarray:
    """Per-pulse static features: rf [GHz], log10 pw [µs], sin/cos aoa, amp/10."""
    a = np.deg2rad(pdws["aoa"])
    return np.stack([
        pdws["rf"] / rf_scale,
        np.log10(np.maximum(pdws["pw"], 1e-9) * 1e6),
        np.sin(a),
        np.cos(a),
        pdws["amp"] / 10.0,
    ], axis=1).astype(np.float32)
