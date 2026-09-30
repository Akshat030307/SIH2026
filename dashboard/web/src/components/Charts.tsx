import { useState, useRef, useId } from "react";
import { SERIES_COLORS, SCHEDULER_TAGS } from "../theme";

export interface Series {
  name: string;
  points: [number, number][];
}

interface HoverState {
  xVal: number;
  svgX: number;
  items: { name: string; val: number; color: string }[];
}

/** Minimalist SVG line chart with subtle area gradients, interactive crosshair and floating inspector. */
export function LineChart({
  series,
  xMax,
  yMax = 1,
  height = 180,
  yLabel,
  xUnit = "s",
  xStep = 5,
}: {
  series: Series[];
  xMax: number;
  yMax?: number;
  height?: number;
  yLabel?: string;
  xUnit?: string;
  xStep?: number;
}) {
  const chartId = useId().replace(/:/g, "_");
  const [hover, setHover] = useState<HoverState | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const W = 620;
  const H = height;
  const L = 38;
  const B = 22;
  const plotW = W - L - 10;
  const plotH = H - B - 8;

  const x = (v: number) => L + (v / Math.max(xMax, 1e-9)) * plotW;
  const y = (v: number) => 8 + (1 - Math.min(Math.max(v, 0), yMax) / yMax) * plotH;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax);

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg || xMax <= 0) return;
    const rect = svg.getBoundingClientRect();
    const mouseX = Math.max(L, Math.min(W - 10, ((e.clientX - rect.left) / rect.width) * W));
    const rawX = ((mouseX - L) / plotW) * xMax;

    const items: { name: string; val: number; color: string }[] = [];
    series.forEach((s, idx) => {
      if (s.points.length === 0) return;
      // find closest point to rawX
      let closest = s.points[0];
      let minDiff = Math.abs(closest[0] - rawX);
      for (let i = 1; i < s.points.length; i++) {
        const diff = Math.abs(s.points[i][0] - rawX);
        if (diff < minDiff) {
          minDiff = diff;
          closest = s.points[i];
        }
      }
      items.push({
        name: s.name,
        val: closest[1],
        color: SERIES_COLORS[idx % SERIES_COLORS.length],
      });
    });

    setHover({ xVal: rawX, svgX: mouseX, items });
  };

  const handleMouseLeave = () => setHover(null);

  return (
    <div style={{ position: "relative", width: "100%" }}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="chart"
        role="img"
        aria-label={yLabel}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <defs>
          {series.map((s, i) => {
            const col = SERIES_COLORS[i % SERIES_COLORS.length];
            return (
              <linearGradient key={s.name} id={`grad_${chartId}_${i}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={col} stopOpacity="0.18" />
                <stop offset="100%" stopColor={col} stopOpacity="0.0" />
              </linearGradient>
            );
          })}
        </defs>

        {/* Horizontal gridlines */}
        {ticks.map((v) => (
          <g key={v}>
            <line x1={L} x2={W - 10} y1={y(v)} y2={y(v)} className="gridline" />
            <text x={L - 6} y={y(v) + 3} className="axis" textAnchor="end">
              {yMax === 1 ? v.toFixed(2) : v.toFixed(0)}
            </text>
          </g>
        ))}

        {/* X axis labels */}
        {Array.from({ length: Math.floor(xMax / xStep) + 1 }, (_, i) => i * xStep).map((v) => (
          <text key={v} x={x(v)} y={H - 5} className="axis" textAnchor="middle">
            {v === 0 ? "0" : `${v}${xUnit}`}
          </text>
        ))}

        {/* Series area and line */}
        {series.map((s, i) => {
          if (s.points.length === 0) return null;
          const col = SERIES_COLORS[i % SERIES_COLORS.length];
          if (s.points.length === 1) {
            return (
              <circle
                key={s.name}
                cx={x(s.points[0][0])}
                cy={y(s.points[0][1])}
                r={3}
                fill={col}
              />
            );
          }
          const linePoints = s.points.map(([a, b]) => `${x(a)},${y(b)}`).join(" ");
          const firstX = x(s.points[0][0]);
          const lastX = x(s.points[s.points.length - 1][0]);
          const areaPoints = `${linePoints} ${lastX},${y(0)} ${firstX},${y(0)}`;

          return (
            <g key={s.name}>
              <polygon points={areaPoints} fill={`url(#grad_${chartId}_${i})`} />
              <polyline
                fill="none"
                stroke={col}
                strokeWidth={1.8}
                strokeLinecap="round"
                strokeLinejoin="round"
                points={linePoints}
              />
            </g>
          );
        })}

        {/* Interactive crosshair */}
        {hover && hover.items.length > 0 && (
          <line
            x1={hover.svgX}
            x2={hover.svgX}
            y1={8}
            y2={H - B}
            stroke="var(--accent-2)"
            strokeWidth="1"
            strokeDasharray="3 3"
            opacity="0.8"
          />
        )}
      </svg>

      {/* Floating tooltip */}
      {hover && hover.items.length > 0 && (
        <div
          className="chart-tooltip"
          style={{
            left: `${Math.min(Math.max((hover.svgX / W) * 100, 10), 90)}%`,
            top: "8px",
            transform: hover.svgX > W * 0.65 ? "translateX(-105%)" : "translateX(8px)",
          }}
        >
          <div className="tooltip-header">t = {hover.xVal.toFixed(1)}s</div>
          {hover.items.map((it) => (
            <div key={it.name} className="tooltip-row">
              <span className="tooltip-dot" style={{ background: it.color }} />
              <span className="tooltip-label">{it.name}:</span>
              <span className="tooltip-val">{it.val.toFixed(3)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** Grouped bar chart: groups × series, optional error bars, modern rounded bars, and interactive hover. */
export function BarChart({
  groups,
  series,
  values,
  errors,
  yMax = 1,
  label,
  colorMap,
}: {
  groups: string[];
  series: string[];
  values: (number | null)[][];
  errors?: (number | null)[][];
  yMax?: number;
  label: string;
  colorMap?: Record<string, string>;
}) {
  const [activeCell, setActiveCell] = useState<{ group: string; sched: string; val: number; err?: number; x: number; y: number } | null>(null);

  const W = 900;
  const H = 250;
  const L = 46;
  const B = 32;
  const gw = (W - L - 16) / Math.max(groups.length, 1);
  const bw = (gw * 0.82) / Math.max(series.length, 1);
  const y = (v: number) => 8 + (1 - Math.min(Math.max(v, 0), yMax) / yMax) * (H - B - 8);

  const getColor = (s: string, idx: number) =>
    colorMap?.[s] ?? SCHEDULER_TAGS[s]?.color ?? SERIES_COLORS[idx % SERIES_COLORS.length];

  return (
    <div style={{ position: "relative", width: "100%" }}>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={label} onMouseLeave={() => setActiveCell(null)}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line x1={L} x2={W - 12} y1={y(f * yMax)} y2={y(f * yMax)} className="gridline" />
            <text x={L - 6} y={y(f * yMax) + 3} className="axis" textAnchor="end">
              {(f * yMax).toFixed(2)}
            </text>
          </g>
        ))}

        {groups.map((g, gi) => (
          <g key={g}>
            {series.map((s, si) => {
              const v = values[gi]?.[si];
              if (v == null) return null;
              const x0 = L + gi * gw + gw * 0.09 + si * bw;
              const e = errors?.[gi]?.[si] ?? 0;
              const barH = Math.max(y(0) - y(v), 2);
              const isHovered = activeCell?.group === g && activeCell?.sched === s;
              const color = getColor(s, si);

              return (
                <g
                  key={s}
                  onMouseEnter={() => setActiveCell({ group: g, sched: s, val: v, err: e > 0 ? e : undefined, x: x0 + bw / 2, y: y(v) })}
                  style={{ cursor: "pointer" }}
                >
                  <rect
                    x={x0}
                    y={y(v)}
                    width={Math.max(bw - 2, 2)}
                    height={barH}
                    rx={2.5}
                    ry={2.5}
                    fill={color}
                    opacity={isHovered ? 1 : 0.88}
                    filter={isHovered ? "brightness(1.2)" : undefined}
                    style={{ transition: "opacity 0.15s, filter 0.15s" }}
                  />
                  {e > 0 && (
                    <line
                      x1={x0 + bw / 2 - 1}
                      x2={x0 + bw / 2 - 1}
                      y1={y(Math.min(v + e, yMax))}
                      y2={y(Math.max(v - e, 0))}
                      className="errbar"
                    />
                  )}
                </g>
              );
            })}
            <text x={L + gi * gw + gw / 2} y={H - 10} className="axis" textAnchor="middle" style={{ fontWeight: 500 }}>
              {g.replace(/^S(\d)_/, "S$1: ").replace(/_/g, " ")}
            </text>
          </g>
        ))}
      </svg>

      {/* Floating tooltip for BarChart with smart flip to prevent clipping */}
      {activeCell && (
        <div
          className="chart-tooltip"
          style={{
            left: `${Math.min(Math.max((activeCell.x / W) * 100, 12), 88)}%`,
            top: activeCell.y < 70 ? `${(activeCell.y / H) * 100 + 12}%` : `${(activeCell.y / H) * 100 - 8}%`,
            transform: activeCell.y < 70 ? "translate(-50%, 0)" : "translate(-50%, -100%)",
          }}
        >
          <div className="tooltip-header">{activeCell.group.replace(/^S(\d)_/, "S$1: ").replace(/_/g, " ")}</div>
          <div className="tooltip-row">
            <span
              className="tooltip-dot"
              style={{ background: getColor(activeCell.sched, series.indexOf(activeCell.sched)) }}
            />
            <span className="tooltip-label">{activeCell.sched}:</span>
            <span className="tooltip-val">
              {activeCell.val.toFixed(3)}
              {activeCell.err != null ? ` ±${activeCell.err.toFixed(3)}` : ""}
            </span>
          </div>
        </div>
      )}

      <Legend names={series} colorMap={colorMap} />
    </div>
  );
}

export function Legend({ names, colorMap }: { names: string[]; colorMap?: Record<string, string> }) {
  return (
    <div className="legend">
      {names.map((n, i) => (
        <span key={n} className="legend-item">
          <i style={{ background: colorMap?.[n] ?? SCHEDULER_TAGS[n]?.color ?? SERIES_COLORS[i % SERIES_COLORS.length] }} />
          <span>{SCHEDULER_TAGS[n]?.label ? `${n} (${SCHEDULER_TAGS[n].label})` : n}</span>
        </span>
      ))}
    </div>
  );
}
