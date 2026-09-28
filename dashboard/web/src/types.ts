export type Reason = "explore" | "acquire" | "track" | "sweep" | "random" | "plan" | string;

/** [t_listen, t_end, channel, reason, n_true_pulses, n_new_intercepts] */
export type Dwell = [number, number, number, Reason, number, number];

/** [emitter_id, start, end, mode, {scheduler: intercepted}] */
export type GtEvent = [number, number, number, number, Record<string, boolean>];

export interface EmitterInfo {
  id: number;
  name: string;
  cls: string;
  priority: number;
  bearing: number;
  range_km: number;
  rf_ghz: number;
  channels: number[];
  t_on: number;
  scan: string;
  period: number | null;
}

export interface Metrics {
  pd: number | null;
  pd_weighted: number | null;
  events_done: number;
  events_intercepted: number;
  emitters_intercepted: number;
  emitters_visible: number;
  predicted_dwells: number;
  pfa: number | null;
  intercept_time_error_ms: number | null;
  ttfi_mean: number | null;
  tune_overhead: number;
  idle_fraction: number;
  n_tracks: number;
}

export interface TrackRow {
  id: number;
  aoa: number;
  rf_ghz: number;
  pw_us: number;
  pri_us: number | null;
  agile: boolean;
  channel: number;
  threat: number;
  period: number | null;
  hits: number;
  locked_hits: number;
  last_seen: number;
}

export interface InitMsg {
  type: "init";
  scenario: string;
  description: string;
  duration: number;
  channels: number[];
  emitters: EmitterInfo[];
  schedulers: string[];
  events_total: number;
}

export interface FrameMsg {
  type: "frame";
  t: number;
  runs: Record<string, { dwells: Dwell[]; metrics: Metrics; tracks?: TrackRow[] }>;
  events: GtEvent[];
}

export interface DoneMsg {
  type: "done";
  final: Record<string, Metrics>;
}

export interface ScenarioInfo {
  name: string;
  description: string;
  duration: number;
  emitters: Record<string, number>;
}

export interface SchedulerInfo {
  name: string;
  description: string;
}
