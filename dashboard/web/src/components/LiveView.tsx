import { useEffect, useRef, useState } from "react";
import type { DoneMsg, Dwell, FrameMsg, GtEvent, InitMsg, Metrics, ScenarioInfo, SchedulerInfo, TrackRow } from "../types";
import { CLASS_COLORS, REASON_COLORS, REASON_LABELS } from "../theme";
import { Waterfall } from "./Waterfall";
import { Legend, LineChart } from "./Charts";

const WINDOW_S = 6;

interface RunState {
  init: InitMsg;
  t: number;
  dwells: Record<string, Dwell[]>;
  events: GtEvent[];
  metrics: Record<string, Metrics>;
  history: Record<string, [number, number][]>;
  tracks: Record<string, TrackRow[]>;
  done: boolean;
}

const fmt = (v: number | null | undefined, d = 3) => (v == null || Number.isNaN(v) ? "–" : v.toFixed(d));

export function LiveView({ scenarios, schedulers }: { scenarios: ScenarioInfo[]; schedulers: SchedulerInfo[] }) {
  const [scenario, setScenario] = useState("S6_lockin");
  const [seed, setSeed] = useState(100);
  const [chosen, setChosen] = useState<string[]>(["sweep", "smart"]);
  const [speed, setSpeed] = useState(4);
  const [paused, setPaused] = useState(false);
  const [run, setRun] = useState<RunState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (schedulers.some((s) => s.name === "d3qn")) setChosen(["sweep", "smart", "d3qn"]);
  }, [schedulers]);

  useEffect(() => () => ws.current?.close(), []);

  // demo presets: ?autorun=S6_lockin&schedulers=sweep,smart&speed=4&seed=100
  const autorun = useRef(new URLSearchParams(location.search));
  useEffect(() => {
    const q = autorun.current;
    const sc = q.get("autorun");
    if (!sc || scenarios.length === 0 || schedulers.length === 0) return;
    q.delete("autorun");
    const cfg = {
      scenario: sc,
      seed: Number(q.get("seed") ?? 100),
      schedulers: (q.get("schedulers") ?? "sweep,smart").split(","),
      speed: Number(q.get("speed") ?? 4),
    };
    setScenario(cfg.scenario);
    setSeed(cfg.seed);
    setChosen(cfg.schedulers);
    setSpeed(cfg.speed);
    start(cfg);
  }, [scenarios, schedulers]);

  const start = (cfg = { scenario, seed, schedulers: chosen, speed }) => {
    ws.current?.close();
    setError(null);
    setPaused(false);
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const sock = new WebSocket(`${proto}://${location.host}/api/run`);
    ws.current = sock;
    sock.onopen = () => sock.send(JSON.stringify({ ...cfg, frame_s: 0.1 }));
    sock.onerror = () => setError("Connection to the simulation server failed. Is `uvicorn smartscan.server:app` running?");
    sock.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as InitMsg | FrameMsg | DoneMsg;
      if (msg.type === "init") {
        const empty = Object.fromEntries(msg.schedulers.map((s) => [s, []]));
        setRun({ init: msg, t: 0, dwells: { ...empty }, events: [], metrics: {}, history: { ...empty }, tracks: {}, done: false });
      } else if (msg.type === "frame") {
        setRun((r) => {
          if (!r) return r;
          const cut = msg.t - WINDOW_S - 0.5;
          const dwells: Record<string, Dwell[]> = {};
          const history = { ...r.history };
          const metrics = { ...r.metrics };
          const tracks = { ...r.tracks };
          for (const [name, data] of Object.entries(msg.runs)) {
            dwells[name] = r.dwells[name].filter((d) => d[1] >= cut).concat(data.dwells);
            metrics[name] = data.metrics;
            const pw = data.metrics.pd_weighted;
            if (pw != null) history[name] = [...history[name], [msg.t, pw]];
            if (data.tracks) tracks[name] = data.tracks;
          }
          const events = r.events.filter((e) => e[2] >= cut).concat(msg.events);
          return { ...r, t: msg.t, dwells, events, metrics, history, tracks };
        });
      } else if (msg.type === "done") {
        setRun((r) => (r ? { ...r, done: true, metrics: { ...r.metrics, ...msg.final } } : r));
      }
    };
  };

  const control = (cmd: object) => ws.current?.readyState === WebSocket.OPEN && ws.current.send(JSON.stringify(cmd));

  const sc = scenarios.find((s) => s.name === scenario);
  const trackOwner = run?.init.schedulers.find((s) => (run.tracks[s] ?? []).length > 0);

  return (
    <div className="live">
      <section className="controls panel">
        <label>
          Scenario
          <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
            {scenarios.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Seed
          <input type="number" value={seed} min={100} onChange={(e) => setSeed(Number(e.target.value))} />
        </label>
        <fieldset>
          <legend>Schedulers (same world, side by side)</legend>
          {schedulers.map((s) => (
            <label key={s.name} className="check" title={s.description}>
              <input
                type="checkbox"
                checked={chosen.includes(s.name)}
                onChange={(e) =>
                  setChosen((c) => (e.target.checked ? [...c, s.name] : c.filter((x) => x !== s.name)).slice(0, 4))
                }
              />
              {s.name}
            </label>
          ))}
        </fieldset>
        <label>
          Speed ×{speed}
          <input
            type="range"
            min={0.5}
            max={20}
            step={0.5}
            value={speed}
            onChange={(e) => {
              setSpeed(Number(e.target.value));
              control({ cmd: "speed", value: Number(e.target.value) });
            }}
          />
        </label>
        <div className="buttons">
          <button className="primary" onClick={() => start()} disabled={chosen.length === 0}>
            {run && !run.done ? "Restart" : "Run mission"}
          </button>
          {run && !run.done && (
            <button
              onClick={() => {
                control({ cmd: paused ? "resume" : "pause" });
                setPaused(!paused);
              }}
            >
              {paused ? "Resume" : "Pause"}
            </button>
          )}
        </div>
        {sc && <p className="desc">{sc.description}</p>}
        {error && <p className="error">{error}</p>}
      </section>

      {!run && (
        <section className="panel empty">
          <h2>Ready</h2>
          <p>
            Pick a scenario and two or more schedulers, then press <b>Run mission</b>. Each scheduler drives its own
            receiver through an identical radar environment. The waterfalls show where each receiver listened
            (filled blocks) against every main-beam illumination that reached it (outlines: green = intercepted, red =
            missed).
          </p>
          <p className="hint">Try S6_lockin: an open-loop sweep phase-locks with the radars and sees almost nothing.</p>
        </section>
      )}

      {run && (
        <>
          <section className="panel emitters">
            <div className="row-head">
              <h2>
                {run.init.scenario} · t = {run.t.toFixed(1)} / {run.init.duration}s {run.done && <span className="tag">complete</span>}
              </h2>
              <div className="progress">
                <div style={{ width: `${(100 * run.t) / run.init.duration}%` }} />
              </div>
            </div>
            <div className="chips">
              {run.init.emitters.map((e) => (
                <span
                  key={e.id}
                  className="chip"
                  style={{ borderColor: CLASS_COLORS[e.cls], opacity: run.t >= e.t_on ? 1 : 0.35 }}
                  title={`${e.name}: ${e.rf_ghz} GHz, bearing ${e.bearing}°, ${e.range_km} km${e.period ? `, T=${e.period}s` : ""}${e.t_on > 0 ? `, on at ${e.t_on}s` : ""}`}
                >
                  <i style={{ background: CLASS_COLORS[e.cls] }} />
                  {e.cls}
                </span>
              ))}
            </div>
          </section>

          <section className="runs" style={{ gridTemplateColumns: `repeat(${Math.min(run.init.schedulers.length, 3)}, minmax(0, 1fr))` }}>
            {run.init.schedulers.map((name) => {
              const m = run.metrics[name];
              const desc = schedulers.find((s) => s.name === name)?.description;
              return (
                <article key={name} className="panel run">
                  <header>
                    <h3>{name}</h3>
                    <span className="muted">{desc}</span>
                  </header>
                  <div className="kpis">
                    <Kpi label="Pd" value={fmt(m?.pd)} />
                    <Kpi label="Threat-wtd Pd" value={fmt(m?.pd_weighted)} strong />
                    <Kpi label="Events caught" value={m ? `${m.events_intercepted}/${m.events_done}` : "–"} />
                    <Kpi label="Emitters" value={m ? `${m.emitters_intercepted}/${run.init.emitters.length}` : "–"} />
                    <Kpi label="Predicted dwells" value={m ? String(m.predicted_dwells) : "–"} />
                    <Kpi label="Pfa" value={fmt(m?.pfa, 2)} />
                  </div>
                  <Waterfall
                    name={name}
                    t={run.t}
                    window={WINDOW_S}
                    channels={run.init.channels}
                    dwells={run.dwells[name]}
                    events={run.events}
                    emitters={run.init.emitters}
                  />
                </article>
              );
            })}
          </section>
          <div className="legend reasons">
            {Object.entries(REASON_LABELS).map(([k, v]) => (
              <span key={k}>
                <i style={{ background: REASON_COLORS[k] }} />
                {v}
              </span>
            ))}
            <span>
              <i className="outline good" />
              illumination intercepted
            </span>
            <span>
              <i className="outline bad" />
              illumination missed
            </span>
            <span className="muted">y-axis: channel centre [GHz]</span>
          </div>

          <section className="bottom">
            <article className="panel">
              <h3>Threat-weighted Pd over the mission</h3>
              <LineChart
                series={run.init.schedulers.map((s) => ({ name: s, points: run.history[s] }))}
                xMax={run.init.duration}
                yLabel="Threat-weighted Pd"
              />
              <Legend names={run.init.schedulers} />
            </article>
            <article className="panel">
              <h3>Emitter tracks {trackOwner ? `(${trackOwner})` : ""}</h3>
              {trackOwner ? (
                <TrackTable rows={run.tracks[trackOwner]} />
              ) : (
                <p className="muted">Open-loop schedulers keep no tracks.</p>
              )}
            </article>
          </section>
        </>
      )}
    </div>
  );
}

function Kpi({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className={`kpi ${strong ? "strong" : ""}`}>
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

function TrackTable({ rows }: { rows: TrackRow[] }) {
  const sorted = [...rows].sort((a, b) => b.threat - a.threat || b.hits - a.hits).slice(0, 14);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>RF GHz</th>
            <th>AOA°</th>
            <th>PRI µs</th>
            <th>PW µs</th>
            <th>Scan period</th>
            <th>Threat</th>
            <th>Hits</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.id}>
              <td>{r.id}</td>
              <td>
                {r.rf_ghz.toFixed(2)}
                {r.agile ? " ⇄" : ""}
              </td>
              <td>{r.aoa.toFixed(0)}</td>
              <td>{r.pri_us ?? "–"}</td>
              <td>{r.pw_us}</td>
              <td>{r.period ? <span className="locked">{r.period.toFixed(3)}s 🔒</span> : <span className="muted">acquiring</span>}</td>
              <td>{r.threat.toFixed(1)}</td>
              <td>
                {r.hits}
                {r.locked_hits ? ` (+${r.locked_hits})` : ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
