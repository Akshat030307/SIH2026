"""Episode runner: couples a scheduler to the RF engine and records the log."""

from __future__ import annotations

from typing import Callable

import numpy as np

from smartscan import pdw as P
from smartscan.metrics import EpisodeLog, compute_metrics
from smartscan.schedulers.base import Scheduler
from smartscan.sim.engine import RFEngine, StepResult
from smartscan.sim.world import World, build_world


def run_episode(world: World, scheduler: Scheduler, seed: int = 0,
                on_step: Callable[[StepResult, object, np.ndarray | None], None] | None = None):
    eng = RFEngine(world).reset()
    scheduler.reset(world.receiver, world.duration, seed=seed)
    log = EpisodeLog()
    while not eng.done:
        act = scheduler.act(eng.t, eng.channel)
        res = eng.step(act.channel, act.dwell_idx)
        seen = P.strip_truth(res.pdws)
        assign = scheduler.observe(res.t_listen, res.t_end, res.channel, seen)
        record(log, res, act, assign)
        if on_step is not None:
            on_step(res, act, assign)
    classes = {e.id: e.cls for e in world.emitters}
    metrics = compute_metrics(eng.truth, log, world.duration, world.receiver.tune_s, classes)
    return metrics, log, eng


def record(log: EpisodeLog, res: StepResult, act, assign) -> None:
    pd = res.pdws
    true = pd["emitter"] >= 0
    log.t_start.append(res.t_start)
    log.t_listen.append(res.t_listen)
    log.t_end.append(res.t_end)
    log.channel.append(res.channel)
    log.dwell_idx.append(res.dwell_idx)
    log.retuned.append(res.retuned)
    log.n_pdw.append(len(pd))
    log.n_true.append(int(true.sum()))
    log.reason.append(act.reason)
    log.new_intercepts.append(len(res.new_intercepts))
    if assign is not None and len(pd):
        for tid, eid in zip(assign.tolist(), pd["emitter"].tolist()):
            if tid >= 0:
                log.confusion[tid][eid] += 1
    if act.intent is not None:
        seen = {}
        for eid in np.unique(pd["emitter"][true]):
            sel = pd["emitter"] == eid
            modes = pd["mode"][sel]
            seen[int(eid)] = (float(pd["toa"][sel].min()), int(np.bincount(modes[modes >= 0]).argmax())
                              if (modes >= 0).any() else -1)
        it = act.intent
        log.intents.append((it.track_id, it.predicted_time, it.predicted_mode, seen))


def evaluate(scenario: str, scheduler_factory: Callable[[], Scheduler], seeds=(0, 1, 2)) -> list[dict]:
    rows = []
    for s in seeds:
        world = build_world(scenario, seed=s)
        sched = scheduler_factory()
        m, _, _ = run_episode(world, sched, seed=s)
        rows.append({"scenario": scenario, "scheduler": sched.name, "seed": s, **m})
    return rows
