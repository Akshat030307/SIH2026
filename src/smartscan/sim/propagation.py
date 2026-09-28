"""Link budget: one-way radar equation and antenna beam patterns."""

from __future__ import annotations

import numpy as np

C = 299_792_458.0
K_BOLTZMANN = 1.380649e-23


def db(x):
    return 10.0 * np.log10(x)


def undb(x):
    return 10.0 ** (np.asarray(x) / 10.0)


def noise_power_dbw(t_eff_k: float, bandwidth_hz: float, noise_figure_db: float) -> float:
    """Receiver noise floor k·T_eff·B·F in dBW."""
    return float(db(K_BOLTZMANN * t_eff_k * bandwidth_hz) + noise_figure_db)


def snr_db(
    pt_w: float,
    gt_db,
    gr_db: float,
    freq_hz,
    range_m: float,
    loss_db: float,
    noise_dbw: float,
):
    """One-way radar equation (PDF §Physics of Interception).

    SNR = Pt·Gt·Gr·λ² / ((4πR)²·k·T_eff·B·F·L)
    """
    lam = C / np.asarray(freq_hz, dtype=float)
    pr_dbw = (
        db(pt_w)
        + np.asarray(gt_db)
        + gr_db
        + 20.0 * np.log10(lam)
        - 20.0 * np.log10(4.0 * np.pi * range_m)
        - loss_db
    )
    return pr_dbw - noise_dbw


def wrap_deg(x):
    """Wrap angle(s) to (-180, 180]."""
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def beam_gain_db(offset_deg, peak_gain_db: float, beamwidth_deg: float, sidelobe_db: float):
    """Gaussian main lobe with a flat sidelobe floor.

    ``offset_deg`` is the angle between antenna boresight and the receiver.
    ``sidelobe_db`` is relative to peak (negative, e.g. -40).
    Gaussian main lobe: G(δ) = Gmax - 12·(δ/θ3dB)² dB (≈ -3 dB at δ = θ/2).
    """
    d = wrap_deg(offset_deg)
    main = peak_gain_db - 12.0 * (d / beamwidth_deg) ** 2
    return np.maximum(main, peak_gain_db + sidelobe_db)


def mainlobe_halfwidth_deg(margin_db: float, beamwidth_deg: float) -> float:
    """Half-width of the angular region where the main lobe is within ``margin_db`` of peak."""
    if margin_db <= 0:
        return 0.0
    return float(beamwidth_deg * np.sqrt(margin_db / 12.0))
