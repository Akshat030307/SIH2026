import { useEffect, useRef, useState } from "react";
import type { DoneMsg, Dwell, FrameMsg, GtEvent, InitMsg, Metrics, ScenarioInfo, SchedulerInfo, TrackRow } from "../types";
import { CLASS_COLORS, REASON_COLORS, REASON_LABELS, SCHEDULER_TAGS } from "../theme";
import { Waterfall } from "./Waterfall";
import { Legend, LineChart } from "./Charts";
import { LockIcon, PauseIcon, PlayIcon, RestartIcon, SparklesIcon } from "./Icons";
import { createClientSimulation } from "../clientSim";

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
const fmtClock = (sec: number) => {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m.toString().padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}s`;
};

const PRESET_SPEEDS = [1, 2, 4, 8, 16];

export function LiveView({ scenarios, schedulers }: { scenarios: ScenarioInfo[]; schedulers: SchedulerInfo[] }) {
  const [scenario, setScenario] = useState("S6_lockin");
  const [seed, setSeed] = useState(100);
  const [chosen, setChosen] = useState<string[]>(["sweep", "smart"]);
  const [speed, setSpeed] = useState(4);
  const [paused, setPaused] = useState(false);
  const [run, setRun] = useState<RunState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDemo, setIsDemo] = useState(false);
  const ws = useRef<WebSocket | null>(null);
  const simDriver = useRef<{ setSpeed: (s: number) => void; pause: () => void; resume: () => void; stop: () => void } | null>(null);

  useEffect(() => {
    if (schedulers.some((s) => s.name === "d3qn") && !chosen.includes("d3qn")) {
      setChosen(["sweep", "smart", "d3qn"]);
    }
  }, [schedulers]);

  useEffect(() => () => ws.current?.close(), []);

  // Demo URL presets: ?autorun=S6_lockin&schedulers=sweep,smart&speed=4&seed=100
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

  const handleFrame = (msg: FrameMsg) => {
    setRun((r) => {
      if (!r) return r;
      const cut = msg.t - WINDOW_S - 0.5;
      const dwells: Record<string, Dwell[]> = {};
      const history = { ...r.history };
      const metrics = { ...r.metrics };
      const tracks = { ...r.tracks };
      for (const [name, data] of Object.entries(msg.runs)) {
        dwells[name] = (r.dwells[name] ?? []).filter((d) => d[1] >= cut).concat(data.dwells);
        metrics[name] = data.metrics;
        const pw = data.metrics.pd_weighted;
        if (pw != null) history[name] = [...(history[name] ?? []), [msg.t, pw]];
        if (data.tracks) tracks[name] = data.tracks;
      }
      const events = r.events.filter((e) => e[2] >= cut).concat(msg.events);
      return { ...r, t: msg.t, dwells, events, metrics, history, tracks };
    });
  };

  const start = (cfg = { scenario, seed, schedulers: chosen, speed }, forceDemo = false) => {
    ws.current?.close();
    simDriver.current?.stop();
    setError(null);
    setPaused(false);

    if (forceDemo) {
      setIsDemo(true);
      const empty = Object.fromEntries(cfg.schedulers.map((s) => [s, []]));
      simDriver.current = createClientSimulation(
        { scenario: cfg.scenario, seed: cfg.seed, schedulers: cfg.schedulers, speed: cfg.speed },
        (initMsg) => {
          setRun({
            init: initMsg,
            t: 0,
            dwells: { ...empty },
            events: [],
            metrics: {},
            history: { ...empty },
            tracks: {},
            done: false,
          });
        },
        (frameMsg) => handleFrame(frameMsg),
        (doneMsg) => {
          setRun((r) => (r ? { ...r, done: true, metrics: { ...r.metrics, ...doneMsg.final } } : r));
        }
      );
      return;
    }

    setIsDemo(false);
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const sock = new WebSocket(`${proto}://${location.host}/api/run`);
    ws.current = sock;
    sock.onopen = () => sock.send(JSON.stringify({ ...cfg, frame_s: 0.1 }));
    sock.onerror = () => {
      // Gracefully switch to interactive client-side simulation when backend is unreachable
      setError("Backend server offline. Automatically switched to interactive client simulation engine.");
      start(cfg, true);
    };
    sock.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as InitMsg | FrameMsg | DoneMsg;
      if (msg.type === "init") {
        const empty = Object.fromEntries(msg.schedulers.map((s) => [s, []]));
        setRun({
          init: msg,
          t: 0,
          dwells: { ...empty },
          events: [],
          metrics: {},
          history: { ...empty },
          tracks: {},
          done: false,
        });
      } else if (msg.type === "frame") {
        handleFrame(msg);
      } else if (msg.type === "done") {
        setRun((r) => (r ? { ...r, done: true, metrics: { ...r.metrics, ...msg.final } } : r));
      }
    };
  };

  const control = (cmd: { cmd: string; value?: number }) => {
    if (isDemo && simDriver.current) {
      if (cmd.cmd === "speed" && cmd.value != null) simDriver.current.setSpeed(cmd.value);
      else if (cmd.cmd === "pause") simDriver.current.pause();
      else if (cmd.cmd === "resume") simDriver.current.resume();
      else if (cmd.cmd === "stop") simDriver.current.stop();
      return;
    }
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(cmd));
    }
  };

  const sc = scenarios.find((s) => s.name === scenario);
  const trackOwner = run?.init.schedulers.find((s) => (run.tracks[s] ?? []).length > 0);

  // Find leader (highest Threat-weighted Pd)
  let leaderName: string | null = null;
  let leaderScore = -1;
  if (run && run.init.schedulers.length > 1) {
    for (const name of run.init.schedulers) {
      const score = run.metrics[name]?.pd_weighted ?? -1;
      if (score > leaderScore && score > 0) {
        leaderScore = score;
        leaderName = name;
      }
    }
  }

  const isRunning = !!(run && !run.done);

  // Keyboard shortcut listener (Space = pause/resume)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (["INPUT", "SELECT", "TEXTAREA"].includes((e.target as HTMLElement)?.tagName)) return;
      if (e.code === "Space" && run && !run.done) {
        e.preventDefault();
        const next = !paused;
        setPaused(next);
        control({ cmd: next ? "pause" : "resume" });
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [run, paused]);

  return (
    <div className="live">
      {/* Sleek Command Deck */}
      <section className="command-deck panel">
        <div className="command-row">
          <div className="command-group">
            <span className="field-label">Scenario</span>
            <div className="select-wrap">
              <select value={scenario} onChange={(e) => setScenario(e.target.value)} disabled={isRunning}>
                {scenarios.map((s) => (
                  <option key={s.name} value={s.name}>
                    {s.name} ({s.duration}s)
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="command-group">
            <span className="field-label">Schedulers ({chosen.length})</span>
            <div className="scheduler-pills">
              {schedulers.map((s) => {
                const active = chosen.includes(s.name);
                const tag = SCHEDULER_TAGS[s.name];
                return (
                  <button
                    key={s.name}
                    type="button"
                    className={`pill-btn ${active ? "active" : ""}`}
                    disabled={isRunning}
                    onClick={() => {
                      setChosen((c) =>
                        active ? (c.length > 1 ? c.filter((x) => x !== s.name) : c) : [...c, s.name]
                      );
                    }}
                    title={s.description}
                  >
                    <span className="pill-dot" style={{ background: active ? tag?.color ?? "var(--accent)" : "transparent" }} />
                    <span className="pill-name">{s.name}</span>
                    {tag && <span className="pill-badge">{tag.label}</span>}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="command-group compact">
            <span className="field-label">Seed</span>
            <input
              type="number"
              className="number-input"
              value={seed}
              min={1}
              disabled={isRunning}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </div>

          <div className="command-group speed-group">
            <span className="field-label">Speed</span>
            <div className="speed-segmented">
              {PRESET_SPEEDS.map((sp) => (
                <button
                  key={sp}
                  type="button"
                  className={`speed-pill ${speed === sp ? "active" : ""}`}
                  onClick={() => {
                    setSpeed(sp);
                    control({ cmd: "speed", value: sp });
                  }}
                >
                  {sp}×
                </button>
              ))}
            </div>
          </div>

          <div className="command-actions">
            {isDemo && <span className="engine-badge demo" title="Running in high-fidelity client simulation mode">⚡ Demo Sim</span>}
            <button
              className="btn btn-primary"
              onClick={() => start()}
              disabled={chosen.length === 0}
            >
              {isRunning ? <RestartIcon className="btn-icon" /> : <PlayIcon className="btn-icon" />}
              <span>{run && !run.done ? "Restart" : "Run Mission"}</span>
            </button>
            {isRunning && (
              <button
                className="btn btn-secondary"
                onClick={() => {
                  control({ cmd: paused ? "resume" : "pause" });
                  setPaused(!paused);
                }}
                title="Press Space to toggle"
              >
                {paused ? <PlayIcon className="btn-icon" /> : <PauseIcon className="btn-icon" />}
                <span>{paused ? "Resume" : "Pause"}</span>
              </button>
            )}
          </div>
        </div>

        {sc && (
          <div className="scenario-brief">
            <span className="brief-tag">Mission Brief:</span>
            <span className="brief-desc">{sc.description}</span>
          </div>
        )}

        {error && <div className="error-alert">{error}</div>}
      </section>

      {/* Ready / Empty Mission State */}
      {!run && (
        <section className="panel empty-hero">
          <div className="hero-content">
            <div className="hero-icon-wrap">
              <SparklesIcon className="hero-icon" />
            </div>
            <h2>Autonomous ESM Receiver Simulation</h2>
            <p>
              Compare agile machine learning scheduling strategies against legacy sweeps in real-time.
              Both schedulers face identical RF environments, ground-truth pulse trains, and main-beam illuminations.
            </p>
            <div className="hero-presets">
              <span className="presets-title">Quick Scenarios (Click to Launch):</span>
              <div className="preset-cards">
                <div
                  className={`preset-card ${scenario === "S6_lockin" ? "selected" : ""}`}
                  onClick={() => {
                    setScenario("S6_lockin");
                    start({ scenario: "S6_lockin", seed, schedulers: chosen, speed });
                  }}
                >
                  <div className="preset-name">S6 Phase-Locking Trap</div>
                  <div className="preset-hint">Legacy sweeps lock out; cognitive tracker achieves &gt;90% Pd.</div>
                </div>
                <div
                  className={`preset-card ${scenario === "S3_mfr" ? "selected" : ""}`}
                  onClick={() => {
                    setScenario("S3_mfr");
                    start({ scenario: "S3_mfr", seed, schedulers: chosen, speed });
                  }}
                >
                  <div className="preset-name">S3 Multi-Function Radar</div>
                  <div className="preset-hint">Search, acquisition &amp; track mode switches tested.</div>
                </div>
                <div
                  className={`preset-card ${scenario === "S5_popup" ? "selected" : ""}`}
                  onClick={() => {
                    setScenario("S5_popup");
                    start({ scenario: "S5_popup", seed, schedulers: chosen, speed });
                  }}
                >
                  <div className="preset-name">S5 Pop-up Threat Radars</div>
                  <div className="preset-hint">Sudden emergence of high-priority fire control emitters.</div>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Active Run Telemetry & Schedulers */}
      {run && (
        <>
          {/* Mission Progress & Emitters Ribbon */}
          <section className="panel telemetry-ribbon">
            <div className="telemetry-bar">
              <div className="telemetry-info">
                <span className="scenario-badge">{run.init.scenario}</span>
                <span className="clock-digits">
                  {fmtClock(run.t)} <span className="clock-total">/ {fmtClock(run.init.duration)}</span>
                </span>
                <span className="progress-percent">
                  {((100 * run.t) / run.init.duration).toFixed(1)}%
                </span>
                <span className={`status-pill ${run.done ? "done" : paused ? "paused" : "live"}`}>
                  <span className="status-dot" />
                  {run.done ? "COMPLETE" : paused ? "PAUSED" : "TRANSMITTING"}
                </span>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${(100 * run.t) / run.init.duration}%` }} />
              </div>
            </div>

            <div className="emitter-roster">
              <div className="roster-label">Target Emitters ({run.init.emitters.length})</div>
              <div className="roster-chips">
                {run.init.emitters.map((e) => {
                  const isActive = run.t >= e.t_on;
                  return (
                    <div
                      key={e.id}
                      className={`emitter-chip ${isActive ? "active" : "standby"}`}
                      style={{ borderColor: isActive ? CLASS_COLORS[e.cls] : "var(--border)" }}
                      title={`${e.name}: ${e.rf_ghz} GHz, Azimuth ${e.bearing}°, ${e.range_km} km${e.period ? `, Period ${e.period}s` : ""}${e.t_on > 0 ? `, Active at ${e.t_on}s` : ""}`}
                    >
                      <span className="chip-indicator" style={{ background: CLASS_COLORS[e.cls] }} />
                      <span className="chip-class">{e.cls}</span>
                      <span className="chip-rf">{e.rf_ghz}G</span>
                      <span className="chip-bearing">∠{e.bearing.toFixed(0)}°</span>
                      {e.period && <span className="chip-period">T={e.period.toFixed(2)}s</span>}
                    </div>
                  );
                })}
              </div>
            </div>
          </section>

          {/* Mission Debrief Banner when complete */}
          {run.done && (
            <section className="panel mission-debrief">
              <div className="debrief-left">
                <div className="debrief-badge">
                  <SparklesIcon className="debrief-icon" /> Mission Complete
                </div>
                <div className="debrief-summary">
                  {leaderName ? (
                    <span>
                      <b>{leaderName.toUpperCase()}</b> achieved top performance with{" "}
                      <b className="debrief-metric">{((run.metrics[leaderName]?.pd_weighted ?? 0) * 100).toFixed(1)}%</b> Threat-Weighted Pd
                      {run.metrics.sweep?.pd_weighted != null && leaderName !== "sweep" && (
                        <span className="debrief-gain">
                          {" "}(+{(((run.metrics[leaderName]?.pd_weighted ?? 0) - (run.metrics.sweep?.pd_weighted ?? 0)) * 100).toFixed(1)}% gain vs baseline sweep)
                        </span>
                      )}.
                    </span>
                  ) : (
                    <span>All schedulers reached mission duration.</span>
                  )}
                </div>
              </div>
              <button type="button" className="btn btn-secondary btn-sm" onClick={() => start()}>
                <RestartIcon className="btn-icon" /> Replay Mission
              </button>
            </section>
          )}

          {/* Schedulers Side-by-Side View */}
          <section
            className="runs-grid"
            style={{
              gridTemplateColumns: `repeat(${
                run.init.schedulers.length === 1
                  ? 1
                  : run.init.schedulers.length === 2 || run.init.schedulers.length === 4
                  ? 2
                  : 3
              }, minmax(0, 1fr))`,
            }}
          >
            {run.init.schedulers.map((name) => {
              const m = run.metrics[name];
              const desc = schedulers.find((s) => s.name === name)?.description;
              const tag = SCHEDULER_TAGS[name];
              const isLeader = leaderName === name;

              return (
                <article key={name} className={`panel run-card ${isLeader ? "leader-card" : ""}`}>
                  <header className="run-card-header">
                    <div className="run-title-row">
                      <div className="run-title-group">
                        <h3 className="run-name">{name}</h3>
                        {tag && <span className="run-tag">{tag.label}</span>}
                      </div>
                      {isLeader && (
                        <span className="leader-badge">
                          <SparklesIcon className="leader-icon" /> Leader
                        </span>
                      )}
                    </div>
                    <span className="run-desc">{desc}</span>
                  </header>

                  <div className="kpi-grid">
                    <KpiCard label="Threat-wtd Pd" value={fmt(m?.pd_weighted)} highlight />
                    <KpiCard label="Overall Pd" value={fmt(m?.pd)} />
                    <KpiCard
                      label="Caught"
                      value={m ? `${m.events_intercepted}/${m.events_done}` : "–"}
                      sub={m && m.events_done > 0 ? `${((100 * m.events_intercepted) / m.events_done).toFixed(0)}%` : undefined}
                    />
                    <KpiCard label="Emitters" value={m ? `${m.emitters_intercepted}/${run.init.emitters.length}` : "–"} />
                    <KpiCard label="Pred Dwells" value={m ? String(m.predicted_dwells) : "–"} />
                    <KpiCard label="Pfa" value={fmt(m?.pfa, 2)} />
                  </div>

                  <div className="waterfall-wrap">
                    <Waterfall
                      name={name}
                      t={run.t}
                      window={WINDOW_S}
                      channels={run.init.channels}
                      dwells={run.dwells[name]}
                      events={run.events}
                      emitters={run.init.emitters}
                    />
                  </div>
                </article>
              );
            })}
          </section>

          {/* Minimalist Legend Strip */}
          <div className="legend-strip panel">
            <span className="legend-title">Waterfall Legend:</span>
            <div className="legend-items">
              {Object.entries(REASON_LABELS).map(([k, v]) => (
                <span key={k} className="legend-tag">
                  <i style={{ background: REASON_COLORS[k] }} />
                  <span>{v}</span>
                </span>
              ))}
              <span className="legend-tag">
                <i className="outline good" />
                <span>Beam Intercepted</span>
              </span>
              <span className="legend-tag">
                <i className="outline bad" />
                <span>Beam Missed</span>
              </span>
            </div>
            <span className="legend-axis-note">Vertical axis: Frequency channel center (GHz)</span>
          </div>

          {/* Lower Analytical Deck */}
          <section className="analytical-deck">
            <article className="panel analytical-card">
              <div className="card-header">
                <h3>Threat-Weighted Pd Trajectory</h3>
                <span className="card-subtitle">Cumulative performance across elapsed mission duration</span>
              </div>
              <LineChart
                series={run.init.schedulers.map((s) => ({ name: s, points: run.history[s] }))}
                xMax={run.init.duration}
                yLabel="Threat-weighted Pd"
              />
              <Legend names={run.init.schedulers} />
            </article>

            <article className="panel analytical-card">
              <div className="card-header">
                <h3>Cognitive Emitter Tracks {trackOwner ? `(${trackOwner})` : ""}</h3>
                <span className="card-subtitle">Deinterleaved pulse clusters &amp; beam period estimators</span>
              </div>
              {trackOwner ? (
                <TrackTable rows={run.tracks[trackOwner]} />
              ) : (
                <div className="no-tracks">
                  <p>Open-loop legacy sweeps do not maintain tracks or estimate radar beam rotation periods.</p>
                </div>
              )}
            </article>
          </section>
        </>
      )}
    </div>
  );
}

function KpiCard({ label, value, sub, highlight }: { label: string; value: string; sub?: string; highlight?: boolean }) {
  return (
    <div className={`kpi-card ${highlight ? "highlight" : ""}`}>
      <span className="kpi-label">{label}</span>
      <div className="kpi-val-row">
        <b className="kpi-value">{value}</b>
        {sub && <span className="kpi-sub">{sub}</span>}
      </div>
    </div>
  );
}

function TrackTable({ rows }: { rows: TrackRow[] }) {
  const sorted = [...rows].sort((a, b) => b.threat - a.threat || b.hits - a.hits).slice(0, 14);
  return (
    <div className="table-wrap">
      <table className="clean-table">
        <thead>
          <tr>
            <th>#</th>
            <th>RF (GHz)</th>
            <th>Bearing (AOA)</th>
            <th>PRI (µs)</th>
            <th>PW (µs)</th>
            <th>Period &amp; Lock</th>
            <th>Threat</th>
            <th>Hits</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.id}>
              <td className="mono muted-text">{r.id}</td>
              <td className="mono">
                {r.rf_ghz.toFixed(2)}
                {r.agile && <span className="agile-tag" title="Frequency Agile"> ⇄</span>}
              </td>
              <td className="mono">∠{r.aoa.toFixed(0)}°</td>
              <td className="mono">{r.pri_us ? r.pri_us.toFixed(1) : "–"}</td>
              <td className="mono">{r.pw_us.toFixed(1)}</td>
              <td>
                {r.period ? (
                  <span className="lock-pill locked">
                    <LockIcon className="inline-icon" /> {r.period.toFixed(3)}s
                  </span>
                ) : (
                  <span className="lock-pill acquiring">Acquiring…</span>
                )}
              </td>
              <td>
                <span className={`threat-badge threat-${Math.min(Math.floor(r.threat), 3)}`}>
                  {r.threat.toFixed(1)}
                </span>
              </td>
              <td className="mono">
                {r.hits}
                {r.locked_hits > 0 && <span className="locked-hits"> (+{r.locked_hits})</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
