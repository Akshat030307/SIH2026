"""Vendor-agnostic scheduler interface for integrating with a real RF front-end.

    from smartscan.api import SmartScanScheduler
    sched = SmartScanScheduler({"f_min_ghz": 2, "f_max_ghz": 18, "ibw_mhz": 500}, policy="smart")
    cmd = sched.next_command(t_now)                           # → tune the LO to cmd.center_freq_hz
    ...                                                       #   and listen for cmd.dwell_s
    sched.ingest(pdw_array, t_listen, t_end)                  # PDWs the receiver produced

PDW input: a structured array with fields toa [s], rf [Hz], pw [s], aoa [deg],
amp [dB], or a plain (N, 5) float array in that column order. No
receiver-specific types cross this boundary, so the same scheduler serves an
airborne RWR or a ground SIGINT vehicle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from smartscan import pdw as P
from smartscan.schedulers.registry import make_scheduler
from smartscan.sim.receiver import ReceiverSpec


@dataclass
class Command:
    channel: int
    center_freq_hz: float
    dwell_s: float
    reason: str
    predicted_track: int | None = None
    predicted_time_s: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class SmartScanScheduler:
    def __init__(self, receiver: ReceiverSpec | dict | None = None, policy: str = "smart", mission_s: float = 3600.0,
                 seed: int = 0, **policy_kw):
        self.rx = receiver if isinstance(receiver, ReceiverSpec) else ReceiverSpec.from_dict(receiver)
        self.impl = make_scheduler(policy, **policy_kw)
        self.impl.reset(self.rx, mission_s, seed=seed)
        self._channel = -1

    def next_command(self, t_now: float) -> Command:
        a = self.impl.act(t_now, self._channel)
        self._channel = a.channel
        it = a.intent
        return Command(a.channel, float(self.rx.centers_hz[a.channel]), float(self.rx.dwells_s[a.dwell_idx]), a.reason,
                       None if it is None else it.track_id, None if it is None else it.predicted_time)

    def ingest(self, pdws, t_listen: float, t_end: float) -> np.ndarray:
        """Feed the PDWs of the dwell just completed. Returns per-pulse track ids (-1 = unassigned)."""
        arr = self._as_pdw(pdws)
        out = self.impl.observe(t_listen, t_end, self._channel, arr)
        return np.full(len(arr), -1) if out is None else out

    def tracks(self) -> list[dict]:
        return self.impl.tracks_snapshot()

    @staticmethod
    def _as_pdw(x) -> np.ndarray:
        if isinstance(x, np.ndarray) and x.dtype.names:
            out = P.empty(len(x))
            for k in ("toa", "rf", "pw", "aoa", "amp"):
                out[k] = x[k]
        else:
            a = np.asarray(x, dtype=float).reshape(-1, 5)
            out = P.empty(len(a))
            for i, k in enumerate(("toa", "rf", "pw", "aoa", "amp")):
                out[k] = a[:, i]
        out["emitter"], out["mode"] = -1, -1
        return out[np.argsort(out["toa"], kind="stable")]
