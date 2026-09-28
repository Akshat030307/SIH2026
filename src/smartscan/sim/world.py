"""Scenario loading and random emitter instantiation from the class library."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from smartscan.sim.emitter import Emitter, Segment, Waveform
from smartscan.sim.receiver import ReceiverSpec

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs"
LIBRARY_PATH = CONFIG_DIR / "emitters" / "library.yaml"
SCENARIO_DIR = CONFIG_DIR / "scenarios"


def load_library(path: Path = LIBRARY_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["classes"]


def list_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))


def load_scenario(name_or_path: str | Path) -> dict:
    p = Path(name_or_path)
    if not p.suffix:
        p = SCENARIO_DIR / f"{name_or_path}.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


@dataclass
class World:
    """One instantiated scenario: a receiver, emitters, and the mission duration."""

    name: str
    duration: float
    receiver: ReceiverSpec
    emitters: list[Emitter]
    seed: int
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- sampling
def _u(rng, spec, log: bool = False) -> float:
    if isinstance(spec, dict) and "choice" in spec:
        return float(rng.choice(spec["choice"]))
    if np.isscalar(spec):
        return float(spec)
    lo, hi = float(spec[0]), float(spec[1])
    if log and lo > 0:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


def _pri_pattern(rng, pri_type: str, pri: float) -> tuple[np.ndarray, float]:
    """Return (intervals, jitter_fraction) for a PRI modulation type."""
    if pri_type == "fixed":
        return np.array([pri]), 0.0
    if pri_type == "jitter":
        return np.array([pri]), float(rng.uniform(0.05, 0.15))
    if pri_type == "stagger":
        m = int(rng.integers(2, 6))
        return pri * (1.0 + rng.uniform(-0.3, 0.3, m)), 0.0
    if pri_type == "sliding":
        n = int(rng.integers(16, 33))
        return np.linspace(0.8 * pri, 1.2 * pri, n), 0.0
    if pri_type == "dwell_switch":
        m = int(rng.integers(2, 5))
        levels = pri * (1.0 + rng.uniform(-0.35, 0.35, m))
        reps = rng.integers(8, 33, m)
        return np.repeat(levels, reps), 0.0
    raise ValueError(f"unknown PRI type {pri_type}")


def _rf_set(rng, c: dict) -> tuple[np.ndarray, str, int]:
    rf0 = _u(rng, c["rf_ghz"]) * 1e9
    hop = c.get("hop", "fixed")
    burst = int(_u(rng, c.get("burst_len", 16)))
    if hop == "fixed":
        return np.array([rf0]), "fixed", burst
    span = _u(rng, c["agile_span_mhz"]) * 1e6
    n = int(_u(rng, c["agile_channels"]))
    lo, hi = c["rf_ghz"][0] * 1e9, c["rf_ghz"][1] * 1e9
    center = np.clip(rf0, lo + span / 2, hi - span / 2)
    return center + np.linspace(-span / 2, span / 2, n), hop, burst


def _waveform(rng, spec: dict, rf, hop, burst) -> Waveform:
    pri_type = str(rng.choice(spec["pri_types"]))
    pri = _u(rng, spec["pri_us"], log=True) * 1e-6
    intervals, jitter = _pri_pattern(rng, pri_type, pri)
    pw = _u(rng, spec["pw_us"], log=True) * 1e-6
    pw = min(pw, 0.3 * intervals.min())
    return Waveform(intervals=intervals, pw=pw, rf=rf, jitter=jitter, hop=hop, burst_len=burst, pri_type=pri_type)


def make_emitter(rng, eid: int, cls_name: str, c: dict, t_on: float, t_off: float) -> Emitter:
    rf, hop, burst = _rf_set(rng, c)
    common = dict(
        id=eid,
        name=f"{cls_name}-{eid:02d}",
        cls=cls_name,
        priority=float(c["priority"]),
        bearing_deg=float(rng.uniform(0, 360)),
        range_m=_u(rng, c["range_km"]) * 1e3,
        pt_w=_u(rng, c["pt_kw"], log=True) * 1e3,
        gain_db=_u(rng, c["gain_db"]),
        beamwidth_deg=_u(rng, c["beamwidth_deg"]),
        sidelobe_db=_u(rng, c["sidelobe_db"]),
        scan=c["scan"],
    )
    seed = int(rng.integers(1, 2**31))
    if c["scan"] == "circular":
        wf = _waveform(rng, c, rf, hop, burst)
        period = _u(rng, c["scan_period_s"])
        seg = Segment(t_on, t_off, 0, wf, phase=float(rng.uniform(0, wf._period)), seed=seed)
        return Emitter(**common, segments=[seg], scan_period=period,
                       scan_phase=float(rng.uniform(0, period)))

    # electronic scan / MFR: per-radar "radar words" fixed per mode, semi-Markov mode chain
    mode_names = list(c["modes"].keys())
    words = []
    for mname in mode_names:
        ms = c["modes"][mname]
        words.append(dict(wf=_waveform(rng, ms, rf, hop, burst),
                          revisit=_u(rng, ms["revisit_s"]),
                          illum=_u(rng, ms["illum_ms"]) * 1e-3,
                          dur=ms["duration_s"]))
    trans = np.asarray(c["transitions"], dtype=float)
    segs, t, mode, k = [], t_on, 0, 0
    while t < t_off:
        w = words[mode]
        dur = _u(rng, w["dur"])
        t1 = min(t + dur, t_off)
        segs.append(Segment(t, t1, mode, w["wf"], revisit=w["revisit"], illum=w["illum"],
                            phase=float(rng.uniform(0, w["revisit"])), seed=seed + 7 * k))
        t, k = t1, k + 1
        mode = int(rng.choice(len(mode_names), p=trans[mode]))
    return Emitter(**common, segments=segs, mode_names=mode_names,
                   mode_priority=[float(c["modes"][m]["priority"]) for m in mode_names])


def build_world(scenario: str | dict, seed: int = 0, library: dict | None = None) -> World:
    sc = load_scenario(scenario) if not isinstance(scenario, dict) else scenario
    lib = library or load_library()
    rng = np.random.default_rng(seed)
    duration = float(sc.get("duration_s", 40))
    receiver = ReceiverSpec.from_dict(sc.get("receiver"))
    overrides = sc.get("overrides", {})

    emitters: list[Emitter] = []

    def add(cls_name: str, n: int, popup: bool):
        c = {**lib[cls_name], **overrides.get(cls_name, {})}
        for _ in range(int(n)):
            t_on = 0.0
            if popup:
                lo, hi = sc.get("popup_window", [0.25, 0.7])
                t_on = float(rng.uniform(lo, hi)) * duration
            emitters.append(make_emitter(rng, len(emitters), cls_name, c, t_on, duration))

    for cls_name, n in (sc.get("emitters") or {}).items():
        add(cls_name, n, popup=False)
    for cls_name, n in (sc.get("popups") or {}).items():
        add(cls_name, n, popup=True)

    # co-located sites: several radars share (almost) the same bearing, so AOA can't separate them
    n_sites = int(sc.get("sites", 0))
    if n_sites > 0:
        site_brg = rng.uniform(0, 360, n_sites)
        site_rng = rng.uniform(40e3, 120e3, n_sites)
        for e in emitters:
            k = int(rng.integers(n_sites))
            e.bearing_deg = float((site_brg[k] + rng.normal(0, 0.3)) % 360)
            e.range_m = float(site_rng[k] * rng.uniform(0.97, 1.03))

    return World(name=sc.get("name", "custom"), duration=duration, receiver=receiver,
                 emitters=emitters, seed=seed, meta={"description": sc.get("description", "")})
