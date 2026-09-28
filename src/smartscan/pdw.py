"""Pulse Descriptor Word (PDW) representation.

A PDW is one detected pulse. Units: seconds, Hz, degrees, dB.

``emitter`` and ``mode`` are simulator ground truth. They are kept in the same
record so the evaluator can score results, but schedulers only ever receive
``strip_truth(pdws)``, where both fields are set to -1.
"""

from __future__ import annotations

import numpy as np

PDW_DTYPE = np.dtype(
    [
        ("toa", "f8"),  # time of arrival [s]
        ("rf", "f8"),  # carrier frequency [Hz]
        ("pw", "f8"),  # pulse width [s]
        ("aoa", "f8"),  # angle of arrival [deg, 0..360)
        ("amp", "f8"),  # measured SNR [dB]
        ("emitter", "i4"),  # ground-truth emitter id (-1 = noise / hidden)
        ("mode", "i2"),  # ground-truth emitter mode (-1 = n/a / hidden)
    ]
)

NOISE_ID = -1


def empty(n: int = 0) -> np.ndarray:
    return np.zeros(n, dtype=PDW_DTYPE)


def strip_truth(pdws: np.ndarray) -> np.ndarray:
    """Copy of ``pdws`` with ground-truth labels removed."""
    out = pdws.copy()
    out["emitter"] = -1
    out["mode"] = -1
    return out


def concat(chunks: list[np.ndarray]) -> np.ndarray:
    chunks = [c for c in chunks if len(c)]
    if not chunks:
        return empty()
    out = np.concatenate(chunks)
    return out[np.argsort(out["toa"], kind="stable")]
