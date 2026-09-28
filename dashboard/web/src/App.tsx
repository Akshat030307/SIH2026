import { useEffect, useState } from "react";
import type { ScenarioInfo, SchedulerInfo } from "./types";
import { LiveView } from "./components/LiveView";
import { ResultsView } from "./components/ResultsView";

export default function App() {
  const [tab, setTab] = useState<"live" | "results">("live");
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [schedulers, setSchedulers] = useState<SchedulerInfo[]>([]);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    Promise.all([fetch("/api/scenarios").then((r) => r.json()), fetch("/api/schedulers").then((r) => r.json())])
      .then(([sc, sh]) => {
        setScenarios(sc);
        setSchedulers(sh);
      })
      .catch(() => setOffline(true));
  }, []);

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <span className="logo" aria-hidden>
            ◎
          </span>
          <div>
            <h1>Smart Scan</h1>
            <p>ML-driven ESM receiver scheduling · SIH26055</p>
          </div>
        </div>
        <nav>
          <button className={tab === "live" ? "active" : ""} onClick={() => setTab("live")}>
            Live mission
          </button>
          <button className={tab === "results" ? "active" : ""} onClick={() => setTab("results")}>
            Results
          </button>
        </nav>
      </header>
      {offline && (
        <p className="panel error">
          Can't reach the API. Start it with <code>uv run uvicorn smartscan.server:app --port 8000</code>.
        </p>
      )}
      <main>{tab === "live" ? <LiveView scenarios={scenarios} schedulers={schedulers} /> : <ResultsView />}</main>
    </div>
  );
}
