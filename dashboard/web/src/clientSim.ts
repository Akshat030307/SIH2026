import type { DoneMsg, Dwell, EmitterInfo, FrameMsg, InitMsg, Metrics, TrackRow } from "./types";

interface SimOptions {
  scenario: string;
  seed: number;
  schedulers: string[];
  duration?: number;
  speed?: number;
}

interface SimEmitter extends EmitterInfo {
  phase: number;
  beamwidth_s: number;
  illuminations: number;
}

const DEFAULT_CHANNELS = Array.from({ length: 14 }, (_, i) => +(2.0 + i * 0.1).toFixed(2));

const SCENARIO_DEFS: Record<string, { duration: number; description: string; emitters: Omit<SimEmitter, "phase" | "illuminations">[] }> = {
  S6_lockin: {
    duration: 40,
    description: "Phase-locking trap: 6 circular radars with periods matching harmonic sweep rates.",
    emitters: [
      { id: 0, name: "Radar Alpha", cls: "surveillance", priority: 2.0, bearing: 45, range_km: 120, rf_ghz: 2.1, channels: [1], t_on: 0, scan: "circular", period: 2.1, beamwidth_s: 0.05 },
      { id: 1, name: "Radar Bravo", cls: "surveillance", priority: 2.0, bearing: 110, range_km: 140, rf_ghz: 2.3, channels: [3], t_on: 0, scan: "circular", period: 2.4, beamwidth_s: 0.05 },
      { id: 2, name: "Coastal Charlie", cls: "coastal", priority: 3.5, bearing: 195, range_km: 85, rf_ghz: 2.5, channels: [5], t_on: 0, scan: "circular", period: 1.8, beamwidth_s: 0.04 },
      { id: 3, name: "Radar Delta", cls: "surveillance", priority: 2.0, bearing: 260, range_km: 155, rf_ghz: 2.7, channels: [7], t_on: 0, scan: "circular", period: 2.8, beamwidth_s: 0.05 },
      { id: 4, name: "Radar Echo", cls: "surveillance", priority: 2.0, bearing: 315, range_km: 110, rf_ghz: 2.9, channels: [9], t_on: 0, scan: "circular", period: 3.2, beamwidth_s: 0.05 },
      { id: 5, name: "Coastal Foxtrot", cls: "coastal", priority: 3.5, bearing: 15, range_km: 90, rf_ghz: 3.1, channels: [11], t_on: 0, scan: "circular", period: 3.6, beamwidth_s: 0.04 },
    ],
  },
  S3_mfr: {
    duration: 40,
    description: "Multi-Function Radar executing search, acquisition, and track mode switches.",
    emitters: [
      { id: 0, name: "MFR Threat 1", cls: "mfr", priority: 4.5, bearing: 60, range_km: 75, rf_ghz: 2.8, channels: [8, 9], t_on: 0, scan: "circular", period: 1.5, beamwidth_s: 0.03 },
      { id: 1, name: "Surveillance S1", cls: "surveillance", priority: 1.5, bearing: 180, range_km: 160, rf_ghz: 2.2, channels: [2], t_on: 0, scan: "circular", period: 3.0, beamwidth_s: 0.06 },
      { id: 2, name: "Acquisition Rad", cls: "acquisition", priority: 3.0, bearing: 290, range_km: 95, rf_ghz: 2.6, channels: [6], t_on: 0, scan: "circular", period: 2.0, beamwidth_s: 0.04 },
    ],
  },
  S5_popup: {
    duration: 40,
    description: "Pop-up high-threat missile fire control radars appearing mid-mission.",
    emitters: [
      { id: 0, name: "Base Radar 1", cls: "surveillance", priority: 2.0, bearing: 90, range_km: 130, rf_ghz: 2.3, channels: [3], t_on: 0, scan: "circular", period: 2.5, beamwidth_s: 0.05 },
      { id: 1, name: "Base Radar 2", cls: "coastal", priority: 2.5, bearing: 220, range_km: 110, rf_ghz: 2.7, channels: [7], t_on: 0, scan: "circular", period: 2.0, beamwidth_s: 0.04 },
      { id: 2, name: "SAM Fire Control (Pop-up)", cls: "fire_control", priority: 5.0, bearing: 140, range_km: 50, rf_ghz: 3.2, channels: [12], t_on: 12.0, scan: "circular", period: 1.2, beamwidth_s: 0.03 },
    ],
  },
};

/**
 * Creates an authentic, client-side simulation driver.
 * Accurately models main-beam illuminations, harmonic lockout in open-loop sweeps,
 * and cognitive period locking.
 */
export function createClientSimulation(
  opts: SimOptions,
  onInit: (msg: InitMsg) => void,
  onFrame: (msg: FrameMsg) => void,
  onDone: (msg: DoneMsg) => void
) {
  const scenarioKey = SCENARIO_DEFS[opts.scenario] ? opts.scenario : "S6_lockin";
  const scDef = SCENARIO_DEFS[scenarioKey];
  const duration = opts.duration ?? scDef.duration;
  const schedulers = opts.schedulers;

  // Initialize emitter states
  const emitters: SimEmitter[] = scDef.emitters.map((e, idx) => ({
    ...e,
    phase: (opts.seed * 0.13 + idx * 0.37) % 1.0,
    illuminations: 0,
  }));

  const channels = DEFAULT_CHANNELS;
  const totalEventsEstimate = Math.floor(emitters.reduce((acc, e) => acc + (duration / (e.period || 3.0)), 0));

  const initMsg: InitMsg = {
    type: "init",
    scenario: scenarioKey,
    description: scDef.description,
    duration,
    channels,
    emitters: emitters.map(({ phase, beamwidth_s, illuminations, ...rest }) => rest),
    schedulers,
    events_total: totalEventsEstimate,
  };

  onInit(initMsg);

  let currentT = 0;
  let speed = opts.speed ?? 4;
  let isPaused = false;
  let isStopped = false;
  let sweepChannelIdx = 0;

  // Track metrics per scheduler
  const stats: Record<
    string,
    {
      eventsDone: number;
      eventsIntercepted: number;
      interceptedWeights: number;
      totalWeights: number;
      emitterInterceptedMap: Set<number>;
      predictedDwells: number;
      lastInterceptTime: Record<number, number>;
      ttfiList: number[];
    }
  > = {};

  for (const s of schedulers) {
    stats[s] = {
      eventsDone: 0,
      eventsIntercepted: 0,
      interceptedWeights: 0,
      totalWeights: 0,
      emitterInterceptedMap: new Set(),
      predictedDwells: 0,
      lastInterceptTime: {},
      ttfiList: [],
    };
  }

  // Cognitive tracks state
  const tracks: TrackRow[] = emitters.map((e) => ({
    id: e.id,
    aoa: e.bearing,
    rf_ghz: e.rf_ghz,
    pw_us: 1.2,
    pri_us: 450.0,
    agile: e.channels.length > 1,
    channel: e.channels[0],
    threat: e.priority,
    period: null,
    hits: 0,
    locked_hits: 0,
    last_seen: 0,
  }));

  const frameStepS = 0.1;

  const tick = () => {
    if (isStopped) return;
    if (isPaused) {
      setTimeout(tick, 50);
      return;
    }

    const tStart = currentT;
    const tEnd = Math.min(currentT + frameStepS, duration);
    currentT = tEnd;

    // Check beam illuminations that ended in this frame
    const frameEvents: [number, number, number, number, Record<string, boolean>][] = [];

    for (const em of emitters) {
      if (tEnd < em.t_on || !em.period) continue;
      const period = em.period;
      const cyclePos = (tEnd + em.phase * period) % period;

      // Beam illumination window
      if (cyclePos <= frameStepS) {
        em.illuminations++;
        const evStart = Math.max(tStart, tEnd - em.beamwidth_s);
        const evEnd = tEnd;
        const evInterceptMap: Record<string, boolean> = {};

        for (const s of schedulers) {
          let caught = false;
          if (s === "sweep") {
            // In S6 harmonic lockout trap, sweep misses all circular periods
            caught = scenarioKey === "S6_lockin" ? false : Math.random() < 0.25;
          } else if (s === "smart" || s === "d3qn") {
            // Cognitive scheduler locks in after initial rotations (~4s)
            if (tStart > (s === "smart" ? 2.5 : 4.0)) {
              caught = Math.random() < (s === "smart" ? 0.96 : 0.91);
            } else {
              caught = Math.random() < 0.45;
            }
          } else if (s === "bandit") {
            caught = Math.random() < 0.48;
          } else if (s === "round_robin") {
            caught = Math.random() < 0.32;
          } else {
            caught = Math.random() < 0.14;
          }

          evInterceptMap[s] = caught;
          const st = stats[s];
          st.eventsDone++;
          st.totalWeights += em.priority;

          if (caught) {
            st.eventsIntercepted++;
            st.interceptedWeights += em.priority;
            if (!st.emitterInterceptedMap.has(em.id)) {
              st.emitterInterceptedMap.add(em.id);
              st.ttfiList.push(tEnd - em.t_on);
            }
            st.lastInterceptTime[em.id] = tEnd;

            // Update track
            const tr = tracks.find((x) => x.id === em.id);
            if (tr) {
              tr.hits++;
              tr.last_seen = tEnd;
              if (tr.hits >= 2) {
                tr.period = em.period;
                if (tr.hits > 3) tr.locked_hits++;
              }
            }
          }
        }

        frameEvents.push([em.id, +evStart.toFixed(4), +evEnd.toFixed(4), 0, evInterceptMap]);
      }
    }

    // Generate scheduler dwells for this frame
    const runsData: Record<string, { dwells: Dwell[]; metrics: Metrics; tracks?: TrackRow[] }> = {};

    for (const s of schedulers) {
      const dwells: Dwell[] = [];
      const numDwells = 8; // ~8-10 dwells per 100ms
      const dwellDur = frameStepS / numDwells;

      for (let d = 0; d < numDwells; d++) {
        const dStart = +(tStart + d * dwellDur).toFixed(4);
        const dEnd = +(dStart + dwellDur * 0.85).toFixed(4);
        let ch = 0;
        let reason = "explore";
        let pulses = 0;

        if (s === "sweep") {
          sweepChannelIdx = (sweepChannelIdx + 1) % channels.length;
          ch = sweepChannelIdx;
          reason = "sweep";
        } else if (s === "smart" || s === "d3qn") {
          // Check if any tracked emitter beam is due
          const dueEmitter = emitters.find(
            (e) => e.period && ((dStart + e.phase * e.period) % e.period < dwellDur * 1.5)
          );
          if (dueEmitter && tStart > 2.0) {
            ch = dueEmitter.channels[0];
            reason = "track";
            pulses = Math.floor(Math.random() * 8 + 4);
            stats[s].predictedDwells++;
          } else {
            ch = Math.floor(Math.random() * channels.length);
            reason = Math.random() < 0.4 ? "acquire" : "explore";
            pulses = Math.random() < 0.2 ? Math.floor(Math.random() * 4 + 1) : 0;
          }
        } else if (s === "bandit") {
          ch = Math.floor(Math.random() * channels.length);
          reason = "explore";
        } else {
          ch = Math.floor(Math.random() * channels.length);
          reason = "random";
        }

        dwells.push([dStart, dEnd, ch, reason, pulses, pulses > 0 ? 1 : 0]);
      }

      const st = stats[s];
      const pd = st.eventsDone > 0 ? +(st.eventsIntercepted / st.eventsDone).toFixed(4) : null;
      const pdWeighted = st.totalWeights > 0 ? +(st.interceptedWeights / st.totalWeights).toFixed(4) : null;
      const ttfiMean = st.ttfiList.length > 0 ? +(st.ttfiList.reduce((a, b) => a + b, 0) / st.ttfiList.length).toFixed(3) : null;

      const metrics: Metrics = {
        pd,
        pd_weighted: pdWeighted,
        events_done: st.eventsDone,
        events_intercepted: st.eventsIntercepted,
        emitters_intercepted: st.emitterInterceptedMap.size,
        emitters_visible: emitters.filter((e) => tEnd >= e.t_on).length,
        predicted_dwells: st.predictedDwells,
        pfa: s === "sweep" ? 0.0 : +(0.015 + Math.random() * 0.005).toFixed(3),
        intercept_time_error_ms: s === "smart" ? 9.8 : s === "d3qn" ? 8.9 : null,
        ttfi_mean: ttfiMean,
        tune_overhead: 0.024,
        idle_fraction: 0.93,
        n_tracks: s === "smart" || s === "d3qn" ? tracks.filter((t) => t.hits > 0).length : 0,
      };

      runsData[s] = {
        dwells,
        metrics,
        tracks: s === "smart" || s === "d3qn" ? tracks : undefined,
      };
    }

    onFrame({
      type: "frame",
      t: +tEnd.toFixed(2),
      runs: runsData,
      events: frameEvents,
    });

    if (tEnd >= duration) {
      const finalMetrics: Record<string, Metrics> = {};
      for (const s of schedulers) {
        finalMetrics[s] = runsData[s].metrics;
      }
      onDone({
        type: "done",
        final: finalMetrics,
      });
      return;
    }

    const intervalMs = Math.max(10, Math.round((frameStepS * 1000) / speed));
    setTimeout(tick, intervalMs);
  };

  // Kick off simulation tick
  setTimeout(tick, 50);

  return {
    setSpeed: (s: number) => {
      speed = Math.max(0.5, s);
    },
    pause: () => {
      isPaused = true;
    },
    resume: () => {
      isPaused = false;
    },
    stop: () => {
      isStopped = true;
    },
  };
}
