import { useEffect, useMemo, useState, type ReactElement } from "react";
import { BarChart, Legend, LineChart } from "./Charts";

interface BenchRow {
  scenario: string;
  scheduler: string;
  n: number;
  [k: string]: string | number | null;
}

interface Results {
  benchmark?: BenchRow[];
  deinterleave?: string | null;
  predict?: string | null;
  edge?: string | null;
  ablation?: string | null;
  training?: { evals: { steps: number; pd_weighted: number; pd: number; train_pd_w?: number | null }[] };
}

const METRICS: { key: string; label: string; yMax?: number; lowerBetter?: boolean }[] = [
  { key: "pd_weighted", label: "Threat-weighted Pd" },
  { key: "pd", label: "Pd (all illuminations)" },
  { key: "intercept_rate", label: "Unique emitters intercepted / s", yMax: 0 },
  { key: "ttfi_censored_mean", label: "Time to first intercept [s] (lower is better)", yMax: 0, lowerBetter: true },
];

const ORDER = ["sweep", "random", "round_robin", "bandit", "smart", "d3qn"];

export function ResultsView() {
  const [res, setRes] = useState<Results | null>(null);
  const [metric, setMetric] = useState(METRICS[0]);
  useEffect(() => {
    fetch("/api/results")
      .then((r) => r.json())
      .then(setRes)
      .catch(() => setRes({}));
  }, []);

  const bench = res?.benchmark ?? [];
  const scenarios = useMemo(() => [...new Set(bench.map((r) => r.scenario))].sort(), [bench]);
  const scheds = useMemo(() => {
    const s = [...new Set(bench.map((r) => r.scheduler))];
    return s.sort((a, b) => (ORDER.indexOf(a) + 99 * +(ORDER.indexOf(a) < 0)) - (ORDER.indexOf(b) + 99 * +(ORDER.indexOf(b) < 0)));
  }, [bench]);
  const get = (sc: string, sh: string, k: string) => {
    const r = bench.find((x) => x.scenario === sc && x.scheduler === sh);
    const v = r?.[k];
    return typeof v === "number" ? v : null;
  };
  const values = scenarios.map((sc) => scheds.map((sh) => get(sc, sh, metric.key)));
  const errors = scenarios.map((sc) => scheds.map((sh) => get(sc, sh, `${metric.key}_std`)));
  const yMax = metric.yMax === 0 ? Math.max(1e-6, ...values.flat().filter((v): v is number => v != null)) * 1.15 : 1;

  if (!res) return <p className="panel">Loading…</p>;
  return (
    <div className="results">
      <section className="panel">
        <div className="row-head">
          <h2>Scheduler benchmark (held-out seeds)</h2>
          <select value={metric.key} onChange={(e) => setMetric(METRICS.find((m) => m.key === e.target.value)!)}>
            {METRICS.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
        {bench.length === 0 ? (
          <p className="muted">
            No benchmark yet. Run <code>uv run python -m smartscan.eval</code>.
          </p>
        ) : (
          <>
            <BarChart groups={scenarios} series={scheds} values={values} errors={errors} yMax={yMax} label={metric.label} />
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Scenario</th>
                    {scheds.map((s) => (
                      <th key={s}>{s}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {scenarios.map((sc, i) => {
                    const row = values[i];
                    const nums = row.filter((v): v is number => v != null);
                    const best = metric.lowerBetter ? Math.min(...nums) : Math.max(...nums);
                    return (
                      <tr key={sc}>
                        <td>{sc}</td>
                        {row.map((v, j) => (
                          <td key={j} className={v === best ? "best" : ""}>
                            {v == null ? "–" : v.toFixed(3)}
                            {errors[i][j] != null && <span className="muted"> ±{errors[i][j]!.toFixed(3)}</span>}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>

      {res.training && res.training.evals.length > 0 && (
        <section className="panel">
          <h2>D3QN training: held-out evaluation during training</h2>
          <LineChart
            series={[
              { name: "eval threat-weighted Pd", points: res.training.evals.map((e) => [e.steps / 1e5, e.pd_weighted]) },
              { name: "eval Pd", points: res.training.evals.map((e) => [e.steps / 1e5, e.pd]) },
            ]}
            xMax={Math.max(...res.training.evals.map((e) => e.steps / 1e5))}
            yLabel="Pd"
            xUnit="00k"
            xStep={2}
          />
          <Legend names={["eval threat-weighted Pd", "eval Pd"]} />
          <p className="muted">x-axis: online environment steps. Step 0 = after demonstration pre-training.</p>
        </section>
      )}

      {[
        ["Deinterleaving", res.deinterleave],
        ["MFR behaviour prediction", res.predict],
        ["Ablations", res.ablation],
        ["Edge inference", res.edge],
      ].map(([title, md]) =>
        md ? (
          <section className="panel" key={title}>
            <h2>{title}</h2>
            <Markdown text={md} />
          </section>
        ) : null,
      )}
    </div>
  );
}

/** Tiny renderer for the generated summaries: headings, paragraphs and pipe tables. */
function Markdown({ text }: { text: string }) {
  const blocks: ReactElement[] = [];
  const lines = text.split("\n");
  let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (l.startsWith("|")) {
      const rows: string[][] = [];
      while (i < lines.length && lines[i].startsWith("|")) {
        const cells = lines[i].split("|").slice(1, -1).map((c) => c.trim());
        if (!cells.every((c) => /^:?-+:?$/.test(c))) rows.push(cells);
        i++;
      }
      blocks.push(
        <div className="table-wrap" key={i}>
          <table>
            <thead>
              <tr>{rows[0].map((c, k) => <th key={k}>{c}</th>)}</tr>
            </thead>
            <tbody>
              {rows.slice(1).map((r, k) => (
                <tr key={k}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }
    if (l.startsWith("#")) {
      if (!l.startsWith("# ")) blocks.push(<h4 key={i}>{l.replace(/^#+\s*/, "")}</h4>);
    } else if (l.trim()) {
      blocks.push(<p key={i} dangerouslySetInnerHTML={{ __html: l.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>") }} />);
    }
    i++;
  }
  return <>{blocks}</>;
}
