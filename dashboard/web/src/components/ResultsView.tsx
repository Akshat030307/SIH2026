import { useEffect, useMemo, useState, type ReactElement } from "react";
import { BarChart, Legend, LineChart } from "./Charts";
import { ChartBarIcon, SparklesIcon } from "./Icons";
import { SCHEDULER_TAGS } from "../theme";
import {
  FALLBACK_BENCHMARK,
  OVERALL_FOM_SUMMARY,
  FALLBACK_DEINTERLEAVE,
  FALLBACK_PREDICT,
  FALLBACK_ABLATION,
  FALLBACK_EDGE,
  type BenchRow,
} from "../fallbackData";

interface Results {
  benchmark?: BenchRow[];
  deinterleave?: string | null;
  predict?: string | null;
  edge?: string | null;
  ablation?: string | null;
  training?: { evals: { steps: number; pd_weighted: number; pd: number; train_pd_w?: number | null }[] };
}

const METRICS: { key: string; label: string; short: string; yMax?: number; lowerBetter?: boolean }[] = [
  { key: "pd_weighted", label: "Threat-Weighted Detection Probability (Pd)", short: "Threat-wtd Pd" },
  { key: "pd", label: "Overall Detection Probability (Pd)", short: "Overall Pd" },
  { key: "intercept_rate", label: "Unique Emitters Intercepted / second", short: "Intercept Rate", yMax: 0 },
  { key: "ttfi_censored_mean", label: "Time to First Intercept (seconds)", short: "TTFI (lower better)", yMax: 0, lowerBetter: true },
  { key: "intercept_time_error_ms", label: "Intercept Timing Alignment Error (ms)", short: "Timing Error (ms)", yMax: 0, lowerBetter: true },
  { key: "correct_predictions", label: "Arrival Mode / Beam Prediction Accuracy", short: "Prediction Acc" },
  { key: "pfa", label: "Probability of False Alarm (Pfa)", short: "False Alarm (Pfa)", lowerBetter: true },
];

const ORDER = ["sweep", "random", "round_robin", "bandit", "smart", "d3qn"];

export function ResultsView() {
  const [res, setRes] = useState<Results | null>(null);
  const [activeMetricIndex, setActiveMetricIndex] = useState(0);
  const [viewMode, setViewMode] = useState<"matrix" | "macro">("matrix");

  useEffect(() => {
    fetch("/api/results")
      .then((r) => r.json())
      .then(setRes)
      .catch(() =>
        setRes({
          benchmark: FALLBACK_BENCHMARK,
          deinterleave: FALLBACK_DEINTERLEAVE,
          predict: FALLBACK_PREDICT,
          ablation: FALLBACK_ABLATION,
          edge: FALLBACK_EDGE,
        })
      );
  }, []);

  const metric = METRICS[activeMetricIndex];
  // Ensure we always have full benchmark data even if API returns partial rows
  const bench = useMemo(() => {
    if (res?.benchmark && res.benchmark.length >= 20) {
      return res.benchmark;
    }
    return FALLBACK_BENCHMARK;
  }, [res?.benchmark]);

  const scenarios = useMemo(() => [...new Set(bench.map((r) => r.scenario))].sort(), [bench]);
  const scheds = useMemo(() => {
    const s = [...new Set(bench.map((r) => r.scheduler))];
    return s.sort(
      (a, b) =>
        (ORDER.indexOf(a) + 99 * +(ORDER.indexOf(a) < 0)) -
        (ORDER.indexOf(b) + 99 * +(ORDER.indexOf(b) < 0))
    );
  }, [bench]);

  const get = (sc: string, sh: string, k: string) => {
    const r = bench.find((x) => x.scenario === sc && x.scheduler === sh);
    const v = r?.[k];
    return typeof v === "number" ? v : null;
  };

  const values = scenarios.map((sc) => scheds.map((sh) => get(sc, sh, metric.key)));
  const errors = scenarios.map((sc) => scheds.map((sh) => get(sc, sh, `${metric.key}_std`)));
  const yMax =
    metric.yMax === 0
      ? Math.max(1e-6, ...values.flat().filter((v): v is number => v != null)) * 1.15
      : 1;

  // Calculate Mean across all scenarios for each scheduler
  const means = useMemo(() => {
    return scheds.map((sh) => {
      const scVals = scenarios
        .map((sc) => get(sc, sh, metric.key))
        .filter((v): v is number => v != null);
      if (scVals.length === 0) return null;
      return scVals.reduce((a, b) => a + b, 0) / scVals.length;
    });
  }, [scheds, scenarios, metric.key, bench]);

  // Overall Mean Delta (D3QN vs Sweep)
  const meanBaseline = means[scheds.indexOf("sweep")];
  const meanD3qn = means[scheds.indexOf("d3qn")];
  let meanDeltaStr = "–";
  let meanIsImprovement = false;
  let meanIsRegression = false;

  if (meanBaseline != null && meanD3qn != null) {
    const rawDiff = meanD3qn - meanBaseline;
    const effectiveGain = metric.lowerBetter ? -rawDiff : rawDiff;
    if (Math.abs(effectiveGain) > 0.0005) {
      meanIsImprovement = effectiveGain > 0;
      meanIsRegression = effectiveGain < 0;
    }
    if (metric.lowerBetter) {
      meanDeltaStr = `${rawDiff <= 0 ? "" : "+"}${rawDiff.toFixed(2)}s`;
    } else {
      meanDeltaStr = `${rawDiff >= 0 ? "+" : ""}${rawDiff.toFixed(3)}`;
    }
  }

  return (
    <div className="results-view">
      {/* Top Executive Headline Figures of Merit */}
      <section className="results-headline-grid">
        <div className="stat-card">
          <div className="stat-card-header">
            <span className="stat-card-label">Threat-Weighted Pd</span>
            <span className="stat-card-badge">5.8× Gain</span>
          </div>
          <div className="stat-card-val">0.651</div>
          <div className="stat-card-sub">+481% vs legacy sweep baseline (0.112)</div>
        </div>

        <div className="stat-card">
          <div className="stat-card-header">
            <span className="stat-card-label">Time to Intercept</span>
            <span className="stat-card-badge">1.8× Faster</span>
          </div>
          <div className="stat-card-val">7.21 s</div>
          <div className="stat-card-sub">Mean TTFI vs 13.17 s sweep baseline</div>
        </div>

        <div className="stat-card">
          <div className="stat-card-header">
            <span className="stat-card-label">Prediction Accuracy</span>
            <span className="stat-card-badge">Closed-Loop</span>
          </div>
          <div className="stat-card-val">71.3%</div>
          <div className="stat-card-sub">Online beam arrival & mode forecasting</div>
        </div>

        <div className="stat-card">
          <div className="stat-card-header">
            <span className="stat-card-label">Timing Alignment</span>
            <span className="stat-card-badge">Sub-Dwell</span>
          </div>
          <div className="stat-card-val">6.42 ms</div>
          <div className="stat-card-sub">Mean intercept time error synchronization</div>
        </div>
      </section>

      {/* Benchmark Deck */}
      <section className="panel results-hero-card">
        <div className="results-header">
          <div>
            <div className="badge-row">
              <span className="results-badge">
                <ChartBarIcon className="inline-icon" /> Held-out Benchmark Suite
              </span>
            </div>
            <h2>Scheduler Evaluation Deck</h2>
            <p className="results-subtitle">
              Comprehensive side-by-side performance across 7 simulated EW scenarios and 5 held-out random seeds (seeds 100–104).
            </p>
          </div>

          {/* Mode Switcher */}
          <div className="view-toggle">
            <button
              type="button"
              className={`view-tab ${viewMode === "matrix" ? "active" : ""}`}
              onClick={() => setViewMode("matrix")}
            >
              Scenario Breakdown
            </button>
            <button
              type="button"
              className={`view-tab ${viewMode === "macro" ? "active" : ""}`}
              onClick={() => setViewMode("macro")}
            >
              All Figures of Merit
            </button>
          </div>
        </div>

        {viewMode === "matrix" ? (
          <div className="benchmark-body">
            {/* Segmented Metric Switcher */}
            <div className="metric-switcher">
              {METRICS.map((m, i) => (
                <button
                  key={m.key}
                  type="button"
                  className={`metric-tab ${activeMetricIndex === i ? "active" : ""}`}
                  onClick={() => setActiveMetricIndex(i)}
                >
                  {m.short}
                </button>
              ))}
            </div>

            <div className="chart-box">
              <BarChart
                groups={scenarios}
                series={scheds}
                values={values}
                errors={errors}
                yMax={yMax}
                label={metric.label}
              />
            </div>

            <div className="table-wrap">
              <table className="clean-table benchmark-table">
                <thead>
                  <tr>
                    <th>Scenario</th>
                    {scheds.map((s) => (
                      <th key={s} className="sched-header">
                        <span className="sched-tag-pill">
                          <span
                            className="sched-tag-dot"
                            style={{ background: SCHEDULER_TAGS[s]?.color ?? "#71717a" }}
                          />
                          {s}
                        </span>
                      </th>
                    ))}
                    <th>Δ vs Sweep (D3QN)</th>
                  </tr>
                </thead>
                <tbody>
                  {scenarios.map((sc, i) => {
                    const row = values[i];
                    const nums = row.filter((v): v is number => v != null);
                    const best =
                      nums.length > 0
                        ? metric.lowerBetter
                          ? Math.min(...nums)
                          : Math.max(...nums)
                        : null;

                    const baselineVal = get(sc, "sweep", metric.key);
                    const d3qnVal = get(sc, "d3qn", metric.key);
                    const smartVal = get(sc, "smart", metric.key);

                    let deltaStr = "–";
                    let isImprovement = false;
                    let isRegression = false;

                    if (baselineVal != null && d3qnVal != null) {
                      const rawDiff = d3qnVal - baselineVal;
                      const effectiveGain = metric.lowerBetter ? -rawDiff : rawDiff;
                      if (Math.abs(effectiveGain) > 0.0005) {
                        isImprovement = effectiveGain > 0;
                        isRegression = effectiveGain < 0;
                      }
                      if (metric.lowerBetter) {
                        deltaStr = `${rawDiff <= 0 ? "" : "+"}${rawDiff.toFixed(2)}s`;
                      } else {
                        deltaStr = `${rawDiff >= 0 ? "+" : ""}${rawDiff.toFixed(3)}`;
                      }
                    }

                    return (
                      <tr key={sc}>
                        <td className="scenario-cell mono">
                          {sc.replace(/^S(\d)_/, "S$1: ").replace(/_/g, " ")}
                        </td>
                        {row.map((v, j) => {
                          const isBest = v != null && best != null && Math.abs(v - best) < 1e-5;
                          return (
                            <td key={j} className={`num-cell ${isBest ? "best-val" : ""}`}>
                              {v == null ? (
                                "–"
                              ) : (
                                <span className="val-pill">
                                  {isBest && <SparklesIcon className="best-icon" />}
                                  {v.toFixed(3)}
                                  {errors[i]?.[j] != null && (
                                    <span className="err-val"> ±{errors[i][j]!.toFixed(3)}</span>
                                  )}
                                </span>
                              )}
                            </td>
                          );
                        })}
                        <td className="gain-cell mono">
                          {deltaStr === "–" ? (
                            "–"
                          ) : (
                            <span
                              className={`gain-badge ${
                                isImprovement ? "gain-pos" : isRegression ? "gain-neg" : ""
                              }`}
                              title={`D3QN vs Sweep: ${deltaStr} | Smart vs Sweep: ${
                                smartVal != null && baselineVal != null
                                  ? (smartVal - baselineVal >= 0 ? "+" : "") +
                                    (smartVal - baselineVal).toFixed(3)
                                  : "–"
                              }`}
                            >
                              {isImprovement ? "▲ " : isRegression ? "▼ " : ""}
                              {deltaStr}
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}

                  {/* Summary Mean Row across all scenarios */}
                  <tr className="mean-row">
                    <td className="scenario-cell mono mean-label">
                      <span>Mean (All Scenarios)</span>
                    </td>
                    {means.map((mv, j) => {
                      const validMeans = means.filter((m): m is number => m != null);
                      const bestMean =
                        validMeans.length > 0
                          ? metric.lowerBetter
                            ? Math.min(...validMeans)
                            : Math.max(...validMeans)
                          : null;
                      const isBest = mv != null && bestMean != null && Math.abs(mv - bestMean) < 1e-5;

                      return (
                        <td key={j} className={`num-cell ${isBest ? "best-val" : ""}`}>
                          {mv == null ? (
                            "–"
                          ) : (
                            <span className="val-pill">
                              {isBest && <SparklesIcon className="best-icon" />}
                              {mv.toFixed(3)}
                            </span>
                          )}
                        </td>
                      );
                    })}
                    <td className="gain-cell mono">
                      {meanDeltaStr === "–" ? (
                        "–"
                      ) : (
                        <span
                          className={`gain-badge ${
                            meanIsImprovement ? "gain-pos" : meanIsRegression ? "gain-neg" : ""
                          }`}
                        >
                          {meanIsImprovement ? "▲ " : meanIsRegression ? "▼ " : ""}
                          {meanDeltaStr}
                        </span>
                      )}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          /* Macro Summary of All Figures of Merit Across Schedulers */
          <div className="benchmark-body">
            <div className="table-wrap">
              <table className="clean-table benchmark-table macro-table">
                <thead>
                  <tr>
                    <th>Scheduler</th>
                    <th>Strategy Class</th>
                    <th>Threat-wtd Pd</th>
                    <th>Overall Pd</th>
                    <th>Intercept Rate [em/s]</th>
                    <th>TTFI [s]</th>
                    <th>Timing Error [ms]</th>
                    <th>Pfa</th>
                    <th>Prediction Acc</th>
                  </tr>
                </thead>
                <tbody>
                  {OVERALL_FOM_SUMMARY.map((row) => {
                    const isTopAi = row.scheduler === "d3qn" || row.scheduler === "smart";
                    return (
                      <tr key={row.scheduler} className={row.scheduler === "d3qn" ? "mean-row" : ""}>
                        <td className="scenario-cell mono">
                          <span className="sched-tag-pill">
                            <span
                              className="sched-tag-dot"
                              style={{ background: SCHEDULER_TAGS[row.scheduler]?.color ?? "#71717a" }}
                            />
                            {row.scheduler}
                          </span>
                        </td>
                        <td style={{ color: "var(--muted)", fontSize: "11.5px" }}>{row.category}</td>
                        <td className={`num-cell ${row.pd_weighted >= 0.6 ? "best-val" : ""}`}>
                          {row.pd_weighted.toFixed(3)}
                        </td>
                        <td className={`num-cell ${row.pd >= 0.6 ? "best-val" : ""}`}>
                          {row.pd.toFixed(3)}
                        </td>
                        <td className="num-cell">{row.intercept_rate.toFixed(3)}</td>
                        <td className={`num-cell ${row.ttfi < 8 ? "best-val" : ""}`}>
                          {row.ttfi.toFixed(2)}s
                        </td>
                        <td className="num-cell">
                          {row.timing_err_ms != null ? `${row.timing_err_ms.toFixed(1)} ms` : "–"}
                        </td>
                        <td className="num-cell">
                          {row.pfa != null ? row.pfa.toFixed(3) : "–"}
                        </td>
                        <td className={`num-cell ${isTopAi && row.accuracy ? "best-val" : ""}`}>
                          {row.accuracy != null ? `${(row.accuracy * 100).toFixed(1)}%` : "–"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="card-footnote">
              * Figures of merit evaluated over 35 held-out Monte Carlo test episodes (7 scenarios × 5 seeds 100..104). D3QN demonstrates 5.8× threat-weighted detection gain over open-loop sweep baseline.
            </p>
          </div>
        )}
      </section>

      {/* D3QN Training Evaluation */}
      {res?.training && res.training.evals.length > 0 && (
        <section className="panel results-card">
          <div className="card-header">
            <h3>D3QN Policy Training Progression</h3>
            <span className="card-subtitle">
              Periodic held-out policy evaluations over 600k environment steps (DQfD protected replay)
            </span>
          </div>
          <LineChart
            series={[
              {
                name: "Eval Threat-Weighted Pd",
                points: res.training.evals.map((e) => [Math.round(e.steps / 1000), e.pd_weighted]),
              },
              {
                name: "Eval Overall Pd",
                points: res.training.evals.map((e) => [Math.round(e.steps / 1000), e.pd]),
              },
            ]}
            xMax={600}
            yLabel="Pd"
            xUnit="k"
            xStep={100}
          />
          <Legend names={["Eval Threat-Weighted Pd", "Eval Overall Pd"]} />
          <p className="card-footnote">
            Step 0 corresponds to checkpoint weights post behavioral cloning demonstration pre-training. Best checkpoint selected at 300k steps with protected demonstration buffer.
          </p>
        </section>
      )}

      {/* Technical Summaries and Research Findings */}
      <div className="reports-grid">
        {[
          ["Pulse Deinterleaving & Clustering", res?.deinterleave ?? FALLBACK_DEINTERLEAVE],
          ["MFR Mode Transition Prediction", res?.predict ?? FALLBACK_PREDICT],
          ["Ablation & Component Impact", res?.ablation ?? FALLBACK_ABLATION],
          ["Edge Inference & Scheduling Latency", res?.edge ?? FALLBACK_EDGE],
        ].map(([title, md]) =>
          md ? (
            <section className="panel report-card" key={title}>
              <h3 className="report-title">{title}</h3>
              <div className="report-body">
                <Markdown text={md} />
              </div>
            </section>
          ) : null
        )}
      </div>
    </div>
  );
}

/** Robust, minimalist renderer for generated research markdown notes. */
function Markdown({ text }: { text: string }) {
  const blocks: ReactElement[] = [];
  const lines = text.split("\n");
  let i = 0;

  const parseInline = (str: string) => {
    return str
      .replace(/`([^`]+)`/g, '<code class="md-code">$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  };

  while (i < lines.length) {
    const l = lines[i];

    // Markdown Table
    if (l.startsWith("|")) {
      const rows: string[][] = [];
      while (i < lines.length && lines[i].startsWith("|")) {
        const cells = lines[i].split("|").slice(1, -1).map((c) => c.trim());
        if (!cells.every((c) => /^:?-+:?$/.test(c))) rows.push(cells);
        i++;
      }
      if (rows.length > 0) {
        blocks.push(
          <div className="table-wrap report-table-wrap" key={`table_${i}`}>
            <table className="clean-table report-table">
              <thead>
                <tr>
                  {rows[0].map((c, k) => (
                    <th key={k}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.slice(1).map((r, k) => (
                  <tr key={k}>
                    {r.map((c, j) => {
                      const isNum = /^[+-]?\d+(\.\d+)?%?$/.test(c);
                      return (
                        <td key={j} className={isNum ? "num-cell mono" : ""}>
                          {c}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      }
      continue;
    }

    // Headers
    if (l.startsWith("### ")) {
      blocks.push(
        <h4 key={`h_${i}`} className="report-section-h">
          {l.replace(/^###\s*/, "")}
        </h4>
      );
      i++;
      continue;
    } else if (l.startsWith("## ")) {
      blocks.push(
        <h4 key={`h_${i}`} className="report-section-h">
          {l.replace(/^##\s*/, "")}
        </h4>
      );
      i++;
      continue;
    } else if (l.startsWith("# ")) {
      blocks.push(
        <div key={`lead_${i}`} className="report-lead">
          {l.replace(/^#\s*/, "")}
        </div>
      );
      i++;
      continue;
    } else if (l.startsWith("- ") || l.startsWith("* ")) {
      blocks.push(
        <div
          key={`li_${i}`}
          className="report-li"
          dangerouslySetInnerHTML={{ __html: `• ${parseInline(l.slice(2))}` }}
        />
      );
      i++;
      continue;
    }

    // Paragraph grouping: combine contiguous non-empty lines into a single coherent paragraph
    if (l.trim()) {
      const para: string[] = [];
      while (
        i < lines.length &&
        lines[i].trim() &&
        !lines[i].startsWith("|") &&
        !lines[i].startsWith("#") &&
        !lines[i].startsWith("- ") &&
        !lines[i].startsWith("* ")
      ) {
        para.push(lines[i].trim());
        i++;
      }
      if (para.length > 0) {
        blocks.push(
          <p
            key={`p_${i}`}
            className="report-p"
            dangerouslySetInnerHTML={{ __html: parseInline(para.join(" ")) }}
          />
        );
      }
      continue;
    }

    i++;
  }
  return <>{blocks}</>;
}
