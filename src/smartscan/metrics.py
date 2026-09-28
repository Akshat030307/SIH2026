"""Figures of merit (PDF §Performance Evaluation), scored against simulator ground truth.

Definitions:
  Pd                  intercepted illumination events / all illumination events
                      (an event = one main-beam passage above the detection
                      threshold; intercepted = ≥ n_min of its pulses detected).
  Pd (weighted)       same, weighted by the emitter's (mode-dependent) threat priority.
  Intercept rate      unique emitters intercepted per second of mission time.
  TTFI                time from an emitter's first illumination to its first intercept.
  Intercept time err  for dwells the scheduler placed on a *prediction*:
                      |predicted arrival − first detected pulse of that emitter|.
  Pfa                 fraction of prediction-driven dwells where the predicted
                      emitter did not show up at all.
  Correct predictions fraction of prediction-driven dwells where the emitter
                      showed up (and, if a mode was predicted, in that mode).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from smartscan.sim.engine import GroundTruth


@dataclass
class EpisodeLog:
    t_start: list = field(default_factory=list)
    t_listen: list = field(default_factory=list)
    t_end: list = field(default_factory=list)
    channel: list = field(default_factory=list)
    dwell_idx: list = field(default_factory=list)
    retuned: list = field(default_factory=list)
    n_pdw: list = field(default_factory=list)
    n_true: list = field(default_factory=list)
    reason: list = field(default_factory=list)
    new_intercepts: list = field(default_factory=list)
    # prediction-driven dwells: (track_id, predicted_time, predicted_mode, {emitter: (first_toa, mode)})
    intents: list = field(default_factory=list)
    # track id -> emitter id -> pulse count (for mapping tracks to truth)
    confusion: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    rewards: list = field(default_factory=list)


def track_mapping(conf: dict) -> tuple[dict, float]:
    """Majority-vote track→emitter map, plus overall track purity."""
    mapping, hit, tot = {}, 0, 0
    for tid, row in conf.items():
        if tid < 0 or not row:
            continue
        eid, c = max(row.items(), key=lambda kv: kv[1])
        mapping[tid] = eid
        hit += c
        tot += sum(row.values())
    return mapping, (hit / tot if tot else float("nan"))


def compute_metrics(gt: GroundTruth, log: EpisodeLog, duration: float, tune_s: float,
                    emitter_classes: dict[int, str] | None = None) -> dict:
    m: dict = {}
    n = gt.n
    icpt = gt.intercepted
    m["events"] = n
    m["pd"] = float(icpt.mean()) if n else float("nan")
    m["pd_weighted"] = float((gt.priority * icpt).sum() / gt.priority.sum()) if n else float("nan")

    emitters = np.unique(gt.emitter)
    first_ev, first_ic = {}, {}
    for e in emitters:
        idx = gt.by_emitter[int(e)]
        first_ev[e] = gt.start[idx].min()
        t_ic = gt.intercept_time[idx]
        if (~np.isnan(t_ic)).any():
            first_ic[e] = np.nanmin(t_ic)
    m["emitters_visible"] = len(emitters)
    m["emitters_intercepted"] = len(first_ic)
    m["intercept_rate"] = len(first_ic) / duration  # unique emitters / s (PDF definition)
    m["event_intercept_rate"] = float(icpt.sum() / duration)
    ttfi = [first_ic[e] - first_ev[e] for e in first_ic]
    censored = ttfi + [duration - first_ev[e] for e in emitters if e not in first_ic]
    m["ttfi_mean"] = float(np.mean(ttfi)) if ttfi else float("nan")
    m["ttfi_censored_mean"] = float(np.mean(censored)) if censored else float("nan")

    # prediction-driven dwells
    mapping, purity = track_mapping(log.confusion)
    m["track_purity"] = purity
    m["n_tracks"] = len(mapping)
    errs, hits, mode_ok, n_int, n_mode = [], 0, 0, 0, 0
    for tid, t_pred, p_mode, seen in log.intents:
        n_int += 1
        eid = mapping.get(tid)
        if eid is None or eid not in seen:
            continue
        hits += 1
        first_toa, mode = seen[eid]
        errs.append(abs(t_pred - first_toa))
        if p_mode is not None:
            n_mode += 1
            mode_ok += int(p_mode == mode)
    m["predicted_dwells"] = n_int
    m["pfa"] = 1.0 - hits / n_int if n_int else float("nan")
    m["correct_predictions"] = hits / n_int if n_int else float("nan")
    m["mode_prediction_acc"] = mode_ok / n_mode if n_mode else float("nan")
    m["intercept_time_error_ms"] = 1e3 * float(np.mean(errs)) if errs else float("nan")

    listen = np.asarray(log.t_end) - np.asarray(log.t_listen)
    m["tune_overhead"] = float(np.sum(log.retuned) * tune_s / duration)
    m["idle_fraction"] = float(listen[np.asarray(log.n_true) == 0].sum() / max(listen.sum(), 1e-12))
    m["steps"] = len(log.t_end)
    if log.rewards:
        m["return"] = float(np.sum(log.rewards))

    if emitter_classes:
        cls_of_ev = np.array([emitter_classes[int(e)] for e in gt.emitter])
        for c in sorted(set(emitter_classes.values())):
            sel = cls_of_ev == c
            if sel.any():
                m[f"pd[{c}]"] = float(icpt[sel].mean())
    return m
