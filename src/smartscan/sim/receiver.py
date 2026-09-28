"""Narrowband scanning superheterodyne ESM receiver model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from smartscan.sim.propagation import noise_power_dbw


@dataclass
class ReceiverSpec:
    f_min_hz: float = 2.0e9
    f_max_hz: float = 18.0e9
    ibw_hz: float = 500e6  # instantaneous bandwidth = one channel
    dwells_s: tuple[float, ...] = (0.002, 0.005, 0.01, 0.02, 0.05)
    tune_s: float = 250e-6  # local-oscillator retune dead time
    gr_db: float = 0.0  # receive antenna gain
    t_eff_k: float = 290.0
    detect_bw_hz: float = 20e6  # detection (channelizer sub-band) bandwidth
    noise_figure_db: float = 12.0
    loss_db: float = 6.0  # system + propagation losses
    threshold_db: float = 13.0  # detection threshold on measured SNR
    amp_noise_db: float = 1.5  # std of amplitude measurement noise
    rf_noise_hz: float = 1.0e6
    pw_noise_frac: float = 0.05
    aoa_noise_deg: float = 2.5
    toa_noise_s: float = 50e-9
    false_pulse_rate: float = 30.0  # noise-triggered PDWs per second per channel
    centers_hz: np.ndarray = field(init=False)

    def __post_init__(self):
        n = int(np.ceil((self.f_max_hz - self.f_min_hz) / self.ibw_hz))
        self.centers_hz = self.f_min_hz + self.ibw_hz * (np.arange(n) + 0.5)
        self.dwells_s = tuple(float(d) for d in self.dwells_s)

    @property
    def n_channels(self) -> int:
        return len(self.centers_hz)

    @property
    def n_dwells(self) -> int:
        return len(self.dwells_s)

    @property
    def n_actions(self) -> int:
        return self.n_channels * self.n_dwells

    @property
    def noise_dbw(self) -> float:
        return noise_power_dbw(self.t_eff_k, self.detect_bw_hz, self.noise_figure_db)

    def channel_of(self, rf_hz) -> np.ndarray:
        """Channel index for each frequency (-1 if outside the band)."""
        rf = np.asarray(rf_hz, dtype=float)
        ch = np.floor((rf - self.f_min_hz) / self.ibw_hz).astype(int)
        return np.where((rf >= self.f_min_hz) & (rf < self.f_max_hz), ch, -1)

    def channel_bounds(self, ch: int) -> tuple[float, float]:
        lo = self.f_min_hz + ch * self.ibw_hz
        return lo, lo + self.ibw_hz

    def encode_action(self, channel: int, dwell_idx: int) -> int:
        return channel * self.n_dwells + dwell_idx

    def decode_action(self, a: int) -> tuple[int, int]:
        return divmod(int(a), self.n_dwells)

    @classmethod
    def from_dict(cls, d: dict | None) -> "ReceiverSpec":
        d = dict(d or {})
        conv = {"f_min_ghz": ("f_min_hz", 1e9), "f_max_ghz": ("f_max_hz", 1e9), "ibw_mhz": ("ibw_hz", 1e6),
                "tune_us": ("tune_s", 1e-6), "detect_bw_mhz": ("detect_bw_hz", 1e6)}
        for k, (nk, s) in conv.items():
            if k in d:
                d[nk] = d.pop(k) * s
        if "dwells_ms" in d:
            d["dwells_s"] = tuple(x * 1e-3 for x in d.pop("dwells_ms"))
        return cls(**d)
