"""Fit / load the radar-word behaviour model used as an RL state feature."""

from __future__ import annotations

import pickle

from smartscan.predict.models import NGram
from smartscan.predict.symbols import emitter_sequences
from smartscan.sim.world import ROOT

PATH = ROOT / "checkpoints" / "behaviour_ngram3.pkl"


def load_behaviour(fit_if_missing: bool = True):
    if PATH.exists():
        with open(PATH, "rb") as f:
            return pickle.load(f)
    if not fit_if_missing:
        return None
    seqs = [s for s, _, _ in emitter_sequences("S3_mfr", seeds=range(0, 100), drop=0.3, rng_seed=5)]
    model = NGram(3).fit(seqs)
    PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PATH, "wb") as f:
        pickle.dump(model, f)
    return model
