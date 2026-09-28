"""Demo backend: FastAPI + WebSocket streaming of side-by-side scheduler runs.

    uv run uvicorn smartscan.server:app --port 8000
    (in production mode the built React app in dashboard/web/dist is served at /)

WebSocket /api/run  (first client message = JSON config)
  {"scenario": "S6_lockin", "seed": 100, "schedulers": ["sweep", "smart"], "speed": 4.0, "frame_s": 0.1}
Server → client messages:
  {"type": "init", "emitters": [...], "channels": [...], "duration": 40, "schedulers": [...]}
  {"type": "frame", "t": 1.3, "runs": {name: {"dwells": [...], "metrics": {...}, "tracks": [...]}},
   "events": [...ground-truth illuminations that ended in this frame...]}
  {"type": "done", "final": {name: metrics}}
Client → server during a run: {"cmd": "speed", "value": 8} | {"cmd": "pause"} | {"cmd": "resume"} | {"cmd": "stop"}
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from smartscan import pdw as P
from smartscan.metrics import EpisodeLog, compute_metrics
from smartscan.runner import record
from smartscan.schedulers.registry import DESCRIPTIONS, make_scheduler
from smartscan.sim.engine import RFEngine
from smartscan.sim.world import ROOT, build_world, list_scenarios, load_scenario

app = FastAPI(title="Smart Scan EW")


def _warm_up():
    """Import torch and load the D3QN stack once, so the first mission doesn't stall on a cold import."""
    if (ROOT / "checkpoints" / "d3qn_best.pt").exists():
        make_scheduler("d3qn")


threading.Thread(target=_warm_up, daemon=True).start()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
DIST = ROOT / "dashboard" / "web" / "dist"


def _clean(x):
    """JSON-safe floats (NaN → None)."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else round(float(x), 5)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def available_schedulers() -> list[dict]:
    out = []
    for k, v in DESCRIPTIONS.items():
        if k == "d3qn" and not (ROOT / "checkpoints" / "d3qn_best.pt").exists():
            continue
        out.append({"name": k, "description": v})
    return out


@app.get("/api/scenarios")
def scenarios():
    out = []
    for s in list_scenarios():
        sc = load_scenario(s)
        out.append({"name": s, "description": sc.get("description", ""), "duration": sc.get("duration_s", 40),
                    "emitters": {**(sc.get("emitters") or {}), **{f"{k} (pop-up)": v for k, v in
                                                                 (sc.get("popups") or {}).items()}}})
    return out


@app.get("/api/schedulers")
def schedulers():
    return available_schedulers()


def _read_md(p: Path):
    return p.read_text() if p.exists() else None


@app.get("/api/results")
def results():
    """Offline benchmark tables (produced by smartscan.eval and friends)."""
    res = {}
    csv = ROOT / "results" / "benchmark" / "results.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        cols = ["pd", "pd_weighted", "intercept_rate", "ttfi_censored_mean", "intercept_time_error_ms", "pfa",
                "correct_predictions"]
        g = df.groupby(["scenario", "scheduler"])[cols]
        mean, std = g.mean(), g.std()
        res["benchmark"] = [
            {"scenario": sc, "scheduler": sh, **{c: mean.loc[(sc, sh), c] for c in cols},
             **{f"{c}_std": std.loc[(sc, sh), c] for c in cols}, "n": int(g.size().loc[(sc, sh)])}
            for sc, sh in mean.index
        ]
    for key, p in [("deinterleave", "deinterleave/summary.md"), ("predict", "predict/summary.md"),
                   ("edge", "edge/summary.md"), ("ablation", "ablation/summary.md")]:
        res[key] = _read_md(ROOT / "results" / p)
    log = ROOT / "checkpoints" / "d3qn_train_log.json"
    if log.exists():
        res["training"] = json.loads(log.read_text())
    return JSONResponse(_clean(res))


class Stream:
    """One scheduler running against its own copy of the same world."""

    def __init__(self, name: str, scenario: str, seed: int):
        self.name = name
        self.world = build_world(scenario, seed=seed)
        self.eng = RFEngine(self.world).reset()
        self.sched = make_scheduler(name)
        self.sched.reset(self.world.receiver, self.world.duration, seed=seed)
        self.log = EpisodeLog()
        self.classes = {e.id: e.cls for e in self.world.emitters}

    def advance(self, t_until: float) -> list:
        dwells = []
        eng = self.eng
        while not eng.done and eng.t < t_until:
            act = self.sched.act(eng.t, eng.channel)
            res = eng.step(act.channel, act.dwell_idx)
            assign = self.sched.observe(res.t_listen, res.t_end, res.channel, P.strip_truth(res.pdws))
            record(self.log, res, act, assign)
            dwells.append([round(res.t_listen, 5), round(res.t_end, 5), res.channel, act.reason,
                           int((res.pdws["emitter"] >= 0).sum()), len(res.new_intercepts)])
        return dwells

    def metrics(self) -> dict:
        gt = self.eng.truth
        t = self.eng.t
        ended = gt.end <= t
        m = compute_metrics(gt, self.log, max(t, 1e-3), self.world.receiver.tune_s)
        # "so far" versions: only events that have finished
        m["pd"] = float(gt.intercepted[ended].mean()) if ended.any() else None
        m["pd_weighted"] = (float((gt.priority * gt.intercepted)[ended].sum() / gt.priority[ended].sum())
                            if ended.any() else None)
        m["events_done"] = int(ended.sum())
        m["events_intercepted"] = int(gt.intercepted[ended].sum())
        m["emitters_intercepted"] = len(np.unique(gt.emitter[gt.intercepted]))
        return m


def _init_payload(streams: list[Stream]) -> dict:
    w = streams[0].world
    rx = w.receiver
    emitters = []
    for e in w.emitters:
        chans = sorted({int(c) for c in rx.channel_of(e.rf_all) if c >= 0})
        emitters.append({"id": e.id, "name": e.name, "cls": e.cls, "priority": e.priority,
                         "bearing": round(e.bearing_deg, 1), "range_km": round(e.range_m / 1e3, 1),
                         "rf_ghz": round(float(np.mean(e.rf_all)) / 1e9, 3), "channels": chans,
                         "t_on": round(e.t_on, 2), "scan": e.scan,
                         "period": round(e.scan_period, 3) if e.scan == "circular" else None})
    return {"type": "init", "scenario": w.name, "description": w.meta.get("description", ""),
            "duration": w.duration, "channels": [round(c / 1e9, 3) for c in rx.centers_hz],
            "emitters": emitters, "schedulers": [s.name for s in streams],
            "events_total": int(streams[0].eng.truth.n)}


@app.websocket("/api/run")
async def run(ws: WebSocket):
    await ws.accept()
    try:
        cfg = json.loads(await ws.receive_text())
        scenario = cfg.get("scenario", "S6_lockin")
        seed = int(cfg.get("seed", 100))
        names = cfg.get("schedulers") or ["sweep", "smart"]
        speed = float(cfg.get("speed", 4.0))
        frame_s = float(cfg.get("frame_s", 0.1))
        streams = await asyncio.to_thread(lambda: [Stream(n, scenario, seed) for n in names])
        await ws.send_text(json.dumps(_clean(_init_payload(streams))))
        gt = streams[0].eng.truth
        ev_order = np.argsort(gt.end)
        ev_ptr = 0
        t, frame = 0.0, 0
        ctl = {"speed": speed, "paused": False, "stop": False}

        async def reader():
            try:
                while True:
                    msg = json.loads(await ws.receive_text())
                    if msg.get("cmd") == "speed":
                        ctl["speed"] = max(0.1, float(msg["value"]))
                    elif msg.get("cmd") in ("pause", "resume"):
                        ctl["paused"] = msg["cmd"] == "pause"
                    elif msg.get("cmd") == "stop":
                        ctl["stop"] = True
            except (WebSocketDisconnect, RuntimeError):
                ctl["stop"] = True

        reader_task = asyncio.create_task(reader())
        loop = asyncio.get_running_loop()
        while t < streams[0].world.duration:
            if ctl["stop"]:
                reader_task.cancel()
                return
            if ctl["paused"]:
                await asyncio.sleep(0.05)
                continue
            speed = ctl["speed"]
            wall0 = loop.time()
            t = min(t + frame_s, streams[0].world.duration)
            runs = {}
            for s in streams:
                dw = await asyncio.to_thread(s.advance, t)
                runs[s.name] = {"dwells": dw, "metrics": s.metrics()}
                if frame % 5 == 0:
                    runs[s.name]["tracks"] = s.sched.tracks_snapshot()
            events = []
            while ev_ptr < len(ev_order) and gt.end[ev_order[ev_ptr]] <= t:
                k = ev_order[ev_ptr]
                caught = {s.name: bool(s.eng.truth.intercepted[k]) for s in streams}
                events.append([int(gt.emitter[k]), round(float(gt.start[k]), 4), round(float(gt.end[k]), 4),
                               int(gt.mode[k]), caught])
                ev_ptr += 1
            await ws.send_text(json.dumps(_clean({"type": "frame", "t": t, "runs": runs, "events": events})))
            frame += 1
            await asyncio.sleep(max(0.0, frame_s / speed - (loop.time() - wall0)))
        final = {s.name: s.metrics() for s in streams}
        await ws.send_text(json.dumps(_clean({"type": "done", "final": final})))
        reader_task.cancel()
    except WebSocketDisconnect:
        return


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(DIST / "index.html")
