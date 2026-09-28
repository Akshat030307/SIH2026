import { useEffect, useRef } from "react";
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

/**
 * Time × frequency picture of one scheduler.
 *  - filled blocks: where the receiver listened (colour = why)
 *  - outlined bars: ground-truth main-beam illuminations, green if this scheduler caught it, red if missed
 */
export function Waterfall({ name, t, window, channels, dwells, events, emitters }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
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
    const css = getComputedStyle(document.documentElement);
    const bg = css.getPropertyValue("--panel-2").trim();
    const grid = css.getPropertyValue("--grid").trim();
    const text = css.getPropertyValue("--muted").trim();
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    const left = 46;
    const bottom = 18;
    const plotW = W - left - 6;
    const plotH = H - bottom - 4;
    const K = channels.length;
    const rowH = plotH / K;
    const t0 = Math.max(0, t - window);
    const x = (tt: number) => left + ((tt - t0) / window) * plotW;
    const y = (ch: number) => 4 + (K - 1 - ch) * rowH;

    // grid + frequency labels
    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.font = "10px IBM Plex Mono, monospace";
    ctx.fillStyle = text;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let ch = 0; ch < K; ch += 4) {
      const yy = y(ch) + rowH / 2;
      ctx.beginPath();
      ctx.moveTo(left, yy);
      ctx.lineTo(W - 6, yy);
      ctx.stroke();
      ctx.fillText(`${channels[ch].toFixed(1)}`, left - 5, yy);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let s = Math.ceil(t0); s <= t; s += 1) {
      const xx = x(s);
      ctx.beginPath();
      ctx.moveTo(xx, 4);
      ctx.lineTo(xx, 4 + plotH);
      ctx.stroke();
      ctx.fillText(`${s}s`, xx, H - bottom + 3);
    }

    // dwells
    for (const d of dwells) {
      if (d[1] < t0) continue;
      const x0 = Math.max(x(d[0]), left);
      const w = Math.max(x(d[1]) - x0, 1.2);
      ctx.fillStyle = REASON_COLORS[d[3]] ?? REASON_COLORS.explore;
      ctx.globalAlpha = d[4] > 0 ? 0.95 : 0.5;
      ctx.fillRect(x0, y(d[2]) + 1, w, rowH - 2);
    }
    ctx.globalAlpha = 1;

    // ground-truth illuminations
    const byId = new Map(emitters.map((e) => [e.id, e]));
    for (const ev of events) {
      if (ev[2] < t0) continue;
      const em = byId.get(ev[0]);
      if (!em) continue;
      const caught = ev[4][name];
      const x0 = Math.max(x(ev[1]), left);
      const w = Math.max(x(ev[2]) - x0, 2);
      ctx.strokeStyle = caught ? "#3ddc97" : "#ff5d73";
      ctx.lineWidth = caught ? 1.6 : 1.1;
      for (const ch of em.channels) {
        ctx.strokeRect(x0, y(ch) + 0.5, w, rowH - 1);
      }
      ctx.fillStyle = CLASS_COLORS[em.cls] ?? "#999";
      ctx.fillRect(x0, y(em.channels[0]) + rowH / 2 - 1, 2, 2);
    }
  }, [name, t, window, channels, dwells, events, emitters]);

  return <canvas ref={ref} className="waterfall" />;
}
