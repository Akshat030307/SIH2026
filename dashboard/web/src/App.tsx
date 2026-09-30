import { useEffect, useState } from "react";
import type { ScenarioInfo, SchedulerInfo } from "./types";
import { LiveView } from "./components/LiveView";
import { ResultsView } from "./components/ResultsView";
import { ActivityIcon, ChartBarIcon, CopyIcon, RadarIcon } from "./components/Icons";

const DEFAULT_SCENARIOS: ScenarioInfo[] = [
  { name: "S6_lockin", description: "Phase-locking trap: 6 circular radars with periods matching harmonic sweep rates.", duration: 40, emitters: { "Surveillance": 4, "Coastal Defense": 2 } },
  { name: "S1_sparse", description: "Low-density baseline environment with widely spaced surveillance radars.", duration: 40, emitters: { "Surveillance A": 1, "Surveillance B": 2 } },
  { name: "S2_dense", description: "High-density spectrum with overlapping multi-band pulse trains.", duration: 40, emitters: { "Dense Radars": 12 } },
  { name: "S3_mfr", description: "Multi-Function Radar executing search, acquisition, and track mode switches.", duration: 40, emitters: { "MFR Threat": 1 } },
  { name: "S4_agile_lpi", description: "Agile frequency-hopping and Low Probability of Intercept emitters.", duration: 40, emitters: { "LPI Radar": 2, "Agile Radar": 2 } },
  { name: "S5_popup", description: "Pop-up high-threat missile fire control radars appearing mid-mission.", duration: 40, emitters: { "Surveillance": 3, "Fire Control (Pop-up)": 2 } },
  { name: "S7_colocated", description: "Co-located emitters with identical bearing and overlapping frequencies.", duration: 40, emitters: { "Colocated Emitters": 4 } },
];

const DEFAULT_SCHEDULERS: SchedulerInfo[] = [
  { name: "sweep", description: "Open-loop linear sweep, 10 ms dwell (legacy baseline)" },
  { name: "smart", description: "Bandit + tracker + period lock + acquisition (heuristic)" },
  { name: "d3qn", description: "Dueling Double DQN policy over tracker/bandit features" },
  { name: "bandit", description: "Phase 1 only: Discounted-UCB channel selection" },
  { name: "round_robin", description: "Pre-planned round-robin favouring radar bands" },
  { name: "random", description: "Uniform random channel, 10 ms dwell" },
];

export default function App() {
  const [tab, setTab] = useState<"live" | "results">("live");
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>(DEFAULT_SCENARIOS);
  const [schedulers, setSchedulers] = useState<SchedulerInfo[]>(DEFAULT_SCHEDULERS);
  const [connected, setConnected] = useState<boolean | null>(null);
  const [bannerDismissed, setBannerDismissed] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch("/api/scenarios").then((r) => r.json()),
      fetch("/api/schedulers").then((r) => r.json()),
    ])
      .then(([sc, sh]) => {
        if (Array.isArray(sc) && sc.length > 0) setScenarios(sc);
        if (Array.isArray(sh) && sh.length > 0) setSchedulers(sh);
        setConnected(true);
      })
      .catch(() => {
        setConnected(false);
      });
  }, []);

  const handleCopyCmd = () => {
    navigator.clipboard.writeText("uv run uvicorn smartscan.server:app --port 8000");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="app">
      {/* Top Header Bar */}
      <header className="top-header">
        <div className="brand-group">
          <div className="logo-box" aria-hidden>
            <RadarIcon className="brand-radar" />
          </div>
          <div className="brand-text">
            <div className="brand-title-row">
              <span className="brand-name">JONATHAN</span>
              <span className="brand-tag">ESM // SIH26055</span>
            </div>
            <p className="brand-sub">Autonomous ESM Receiver Scheduling System</p>
          </div>
        </div>

        <div className="header-right">
          {/* Server Connection Status */}
          <div className={`status-badge ${connected ? "online" : "offline"}`}>
            <span className="status-indicator-dot" />
            <span className="status-text">{connected ? "Live Backend" : "Demo Engine Ready"}</span>
          </div>

          {/* Segmented Tab Navigation */}
          <nav className="tab-nav">
            <button
              type="button"
              className={`nav-tab ${tab === "live" ? "active" : ""}`}
              onClick={() => setTab("live")}
            >
              <ActivityIcon className="nav-icon" />
              <span>Live Mission</span>
            </button>
            <button
              type="button"
              className={`nav-tab ${tab === "results" ? "active" : ""}`}
              onClick={() => setTab("results")}
            >
              <ChartBarIcon className="nav-icon" />
              <span>Benchmark Results</span>
            </button>
          </nav>
        </div>
      </header>

      {/* Offline Alert Banner */}
      {connected === false && !bannerDismissed && (
        <aside className="offline-banner">
          <div className="banner-msg">
            <span className="banner-dot" />
            <span>Interactive demo simulation active. To connect live PyTorch/D3QN backend:</span>
            <code className="banner-code">uv run uvicorn smartscan.server:app --port 8000</code>
            <button type="button" className="copy-btn" onClick={handleCopyCmd} title="Copy command">
              <CopyIcon className="copy-icon" />
              <span>{copied ? "Copied!" : "Copy"}</span>
            </button>
          </div>
          <button type="button" className="banner-close" onClick={() => setBannerDismissed(true)}>
            ✕
          </button>
        </aside>
      )}

      {/* Main Content Area */}
      <main className="main-content">
        {tab === "live" ? (
          <LiveView scenarios={scenarios} schedulers={schedulers} />
        ) : (
          <ResultsView />
        )}
      </main>
    </div>
  );
}
