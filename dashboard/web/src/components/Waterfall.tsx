import { useEffect, useRef, useState } from "react";
import type { Dwell, EmitterInfo, GtEvent } from "../types";
import { CLASS_COLORS, REASON_COLORS } from "../theme";

interface Props {
  name: string;
  t: number;
  window: number;
  channels: number[];
  dwells: Dwell[];
  events: GtEvent[];
  emitters: EmitterInfo[];
}

interface HoverInfo {
  x: number;
  y: number;
  timeSec: number;
  channelIdx: number;
  channelGhz: number;
  activeDwell?: {
    reason: string;
    tListen: number;
    tEnd: number;
    pulses: number;
  };
  activeEvent?: {
    emitterName: string;
    cls: string;
    start: number;
    end: number;
    caught: boolean;
  };
}

/**
 * High-precision Time × Frequency Waterfall Display
 *  - Dwell blocks: where receiver listened (colored by reason)
 *  - Outlines: ground-truth radar beam illuminations (green = intercepted, red = missed)
 *  - Live cursor & interactive hover inspector
 */
export function Waterfall({ name, t, window, channels, dwells, events, emitters }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<HoverInfo | null>(null);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const dpr = globalThis.devicePixelRatio || 1;
    const W = cv.clientWidth;
    const H = cv.clientHeight;
    if (cv.width !== W * dpr || cv.height !== H * dpr) {
      cv.width = W * dpr;
      cv.height = H * dpr;
    }
    const ctx = cv.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Pure black radar screen background
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, W, H);

    const left = 44;
    const bottom = 20;
    const plotW = W - left - 6;
    const plotH = H - bottom - 4;
    if (plotW <= 10 || plotH <= 10) return;

    const K = channels.length;
    const rowH = plotH / Math.max(K, 1);
    const t0 = Math.max(0, t - window);
    const x = (tt: number) => left + ((tt - t0) / window) * plotW;
    const y = (ch: number) => 4 + (K - 1 - ch) * rowH;

    // Horizontal channel gridlines & labels
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    ctx.font = "9.5px IBM Plex Mono, ui-monospace, monospace";
    ctx.fillStyle = "rgba(148, 163, 184, 0.65)";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    for (let ch = 0; ch < K; ch += 4) {
      const yy = y(ch) + rowH / 2;
      ctx.beginPath();
      ctx.moveTo(left, yy);
      ctx.lineTo(W - 6, yy);
      ctx.stroke();
      ctx.fillText(`${channels[ch].toFixed(1)}G`, left - 6, yy);
    }

    // Vertical time gridlines & timestamps
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let s = Math.ceil(t0); s <= t; s += 1) {
      const xx = x(s);
      ctx.beginPath();
      ctx.moveTo(xx, 4);
      ctx.lineTo(xx, 4 + plotH);
      ctx.stroke();
      ctx.fillText(`${s}s`, xx, H - bottom + 4);
    }

    // Clip plotting area to prevent bleed outside the oscilloscope frame
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, 4, plotW, plotH);
    ctx.clip();

    // Receiver dwells (where antenna tuned and listened)
    for (const d of dwells) {
      if (d[1] < t0 || d[0] > t) continue;
      if (d[2] < 0 || d[2] >= K) continue;
      const x0 = Math.max(x(d[0]), left);
      const w = Math.max(x(d[1]) - x0, 1.4);
      const isHit = d[4] > 0;
      ctx.fillStyle = REASON_COLORS[d[3]] ?? REASON_COLORS.explore;
      ctx.globalAlpha = isHit ? 0.95 : 0.45;
      ctx.fillRect(x0, y(d[2]) + 1, w, Math.max(rowH - 2, 1));

      if (isHit && w > 4) {
        // Subtle indicator for pulse detection inside dwell
        ctx.fillStyle = "#ffffff";
        ctx.globalAlpha = 0.85;
        ctx.fillRect(x0 + w / 2 - 1, y(d[2]) + rowH / 2 - 1, 2, 2);
      }
    }
    ctx.globalAlpha = 1;

    // Ground-truth illuminations (transmitters illuminating receiver)
    const byId = new Map(emitters.map((e) => [e.id, e]));
    for (const ev of events) {
      if (ev[2] < t0 || ev[1] > t) continue;
      const em = byId.get(ev[0]);
      if (!em || !em.channels || em.channels.length === 0) continue;
      const caught = !!ev[4]?.[name];
      const x0 = Math.max(x(ev[1]), left);
      const w = Math.max(x(ev[2]) - x0, 2);

      ctx.strokeStyle = caught ? "#10b981" : "#f43f5e";
      ctx.lineWidth = caught ? 1.6 : 1.2;

      for (const ch of em.channels) {
        if (ch >= 0 && ch < K) {
          ctx.strokeRect(x0, y(ch) + 0.5, w, Math.max(rowH - 1, 1));
        }
      }

      // Small radar emitter marker on channel center
      ctx.fillStyle = CLASS_COLORS[em.cls] ?? "#94a3b8";
      ctx.fillRect(x0, y(em.channels[0]) + rowH / 2 - 1, 2, 2);
    }

    // Scanning time cursor line on current t
    const curX = x(t);
    if (curX >= left && curX <= W - 6) {
      ctx.strokeStyle = "rgba(56, 189, 248, 0.45)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      ctx.moveTo(curX, 4);
      ctx.lineTo(curX, 4 + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.restore();
  }, [name, t, window, channels, dwells, events, emitters]);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const cv = canvasRef.current;
    if (!cv || channels.length === 0) return;
    const rect = cv.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    const left = 44;
    const bottom = 20;
    const plotW = W - left - 6;
    const plotH = H - bottom - 4;

    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (mouseX < left || mouseX > W - 6 || mouseY < 4 || mouseY > H - bottom) {
      setHover(null);
      return;
    }

    const t0 = Math.max(0, t - window);
    const timeSec = t0 + ((mouseX - left) / plotW) * window;
    const K = channels.length;
    const rowH = plotH / K;
    const chIdx = Math.max(0, Math.min(K - 1, K - 1 - Math.floor((mouseY - 4) / rowH)));
    const chGhz = channels[chIdx];

    // Find any overlapping dwell
    const activeDwell = dwells.find(
      (d) => d[2] === chIdx && timeSec >= d[0] && timeSec <= d[1]
    );

    // Find any overlapping event
    const activeEvent = events.find((ev) => {
      const em = emitters.find((e) => e.id === ev[0]);
      if (!em || !em.channels.includes(chIdx)) return false;
      return timeSec >= ev[1] && timeSec <= ev[2];
    });

    const em = activeEvent ? emitters.find((e) => e.id === activeEvent[0]) : undefined;

    setHover({
      x: mouseX,
      y: mouseY,
      timeSec,
      channelIdx: chIdx,
      channelGhz: chGhz,
      activeDwell: activeDwell
        ? {
            reason: activeDwell[3],
            tListen: activeDwell[0],
            tEnd: activeDwell[1],
            pulses: activeDwell[4],
          }
        : undefined,
      activeEvent:
        activeEvent && em
          ? {
              emitterName: em.name,
              cls: em.cls,
              start: activeEvent[1],
              end: activeEvent[2],
              caught: !!activeEvent[4][name],
            }
          : undefined,
    });
  };

  return (
    <div
      ref={containerRef}
      className="waterfall-container"
      onMouseMove={handleMouseMove}
      onMouseLeave={() => setHover(null)}
      style={{ position: "relative" }}
    >
      <canvas ref={canvasRef} className="waterfall" />
      {hover && (
        <div
          className="waterfall-tooltip"
          style={{
            left: `${canvasRef.current?.clientWidth ? (hover.x > canvasRef.current.clientWidth - 180 ? hover.x - 175 : hover.x + 12) : hover.x + 12}px`,
            top: `${Math.max(8, Math.min(hover.y - 45, (canvasRef.current?.clientHeight ?? 330) - 80))}px`,
          }}
        >
          <div className="wf-ch">
            Ch {hover.channelIdx} · <b>{hover.channelGhz.toFixed(2)} GHz</b>
          </div>
          <div className="wf-time">t = {hover.timeSec.toFixed(2)}s</div>
          {hover.activeDwell && (
            <div className="wf-detail dwell">
              Dwell: <span>{hover.activeDwell.reason}</span>
              {hover.activeDwell.pulses > 0 && <span className="pulses"> ({hover.activeDwell.pulses} pulses)</span>}
            </div>
          )}
          {hover.activeEvent && (
            <div className={`wf-detail event ${hover.activeEvent.caught ? "good" : "miss"}`}>
              {hover.activeEvent.caught ? "✓ Intercepted" : "✗ Missed"} · {hover.activeEvent.emitterName}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
