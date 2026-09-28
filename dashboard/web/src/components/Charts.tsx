import { SERIES_COLORS } from "../theme";

export interface Series {
  name: string;
  points: [number, number][];
}

/** Minimal SVG line chart with a shared x axis. */
export function LineChart({ series, xMax, yMax = 1, height = 180, yLabel, xUnit = "s", xStep = 5 }: {
  series: Series[];
  xMax: number;
  yMax?: number;
  height?: number;
  yLabel?: string;
  xUnit?: string;
  xStep?: number;
}) {
  const W = 600;
  const H = height;
  const L = 36;
  const B = 20;
  const x = (v: number) => L + (v / Math.max(xMax, 1e-9)) * (W - L - 8);
  const y = (v: number) => 6 + (1 - v / yMax) * (H - B - 6);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={yLabel}>
      {ticks.map((v) => (
        <g key={v}>
          <line x1={L} x2={W - 8} y1={y(v)} y2={y(v)} className="gridline" />
          <text x={L - 5} y={y(v) + 3} className="axis" textAnchor="end">
            {yMax === 1 ? v.toFixed(2) : v.toFixed(0)}
          </text>
        </g>
      ))}
      {Array.from({ length: Math.floor(xMax / xStep) + 1 }, (_, i) => i * xStep).map((v) => (
        <text key={v} x={x(v)} y={H - 5} className="axis" textAnchor="middle">
          {v}
          {xUnit}
        </text>
      ))}
      {series.map((s, i) =>
        s.points.length > 1 ? (
          <polyline
            key={s.name}
            fill="none"
            stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
            strokeWidth={2}
            points={s.points.map(([a, b]) => `${x(a)},${y(b)}`).join(" ")}
          />
        ) : null,
      )}
    </svg>
  );
}

/** Grouped bar chart: groups × series, optional error bars. */
export function BarChart({ groups, series, values, errors, yMax = 1, label }: {
  groups: string[];
  series: string[];
  values: (number | null)[][];
  errors?: (number | null)[][];
  yMax?: number;
  label: string;
}) {
  const W = 900;
  const H = 260;
  const L = 40;
  const B = 34;
  const gw = (W - L - 10) / groups.length;
  const bw = (gw * 0.8) / series.length;
  const y = (v: number) => 8 + (1 - v / yMax) * (H - B - 8);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={label}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line x1={L} x2={W - 10} y1={y(f * yMax)} y2={y(f * yMax)} className="gridline" />
            <text x={L - 5} y={y(f * yMax) + 3} className="axis" textAnchor="end">
              {(f * yMax).toFixed(2)}
            </text>
          </g>
        ))}
        {groups.map((g, gi) => (
          <g key={g}>
            {series.map((s, si) => {
              const v = values[gi][si];
              if (v == null) return null;
              const x0 = L + gi * gw + gw * 0.1 + si * bw;
              const e = errors?.[gi][si] ?? 0;
              return (
                <g key={s}>
                  <rect x={x0} y={y(v)} width={bw - 2} height={y(0) - y(v)} fill={SERIES_COLORS[si % SERIES_COLORS.length]}>
                    <title>{`${g} · ${s}: ${v.toFixed(3)}`}</title>
                  </rect>
                  {e > 0 && (
                    <line x1={x0 + bw / 2 - 1} x2={x0 + bw / 2 - 1} y1={y(Math.min(v + e, yMax))} y2={y(Math.max(v - e, 0))} className="errbar" />
                  )}
                </g>
              );
            })}
            <text x={L + gi * gw + gw / 2} y={H - 12} className="axis" textAnchor="middle">
              {g.replace(/^S\d_/, "")}
            </text>
          </g>
        ))}
      </svg>
      <Legend names={series} />
    </div>
  );
}

export function Legend({ names }: { names: string[] }) {
  return (
    <div className="legend">
      {names.map((n, i) => (
        <span key={n}>
          <i style={{ background: SERIES_COLORS[i % SERIES_COLORS.length] }} />
          {n}
        </span>
      ))}
    </div>
  );
}
