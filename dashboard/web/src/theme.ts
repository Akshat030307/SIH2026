export const REASON_COLORS: Record<string, string> = {
  explore: "#38bdf8", // Sky blue for exploration
  acquire: "#fbbf24", // Amber for acquisition
  track: "#34d399",   // Crisp emerald for locked predicted beam
  sweep: "#64748b",   // Slate for open-loop baseline
  random: "#94a3b8",
  plan: "#a78bfa",
};

export const REASON_LABELS: Record<string, string> = {
  explore: "Explore (Bandit / Prior)",
  acquire: "Acquire (Tracking Radar)",
  track: "Predicted Beam (Locked 🔒)",
  sweep: "Open-loop Sweep / Plan",
};

export const CLASS_COLORS: Record<string, string> = {
  surveillance: "#38bdf8",
  coastal: "#34d399",
  acquisition: "#fbbf24",
  agile: "#f87171",
  lpi: "#a78bfa",
  mfr: "#f472b6",
  fire_control: "#fb923c",
};

export const CLASS_LABELS: Record<string, string> = {
  surveillance: "Surveillance",
  coastal: "Coastal Defense",
  acquisition: "Target Acquisition",
  agile: "Frequency Agile",
  lpi: "Low Probability of Intercept",
  mfr: "Multi-Function Radar",
  fire_control: "Fire Control",
};

export const SERIES_COLORS = [
  "#94a3b8", // sweep (slate)
  "#10b981", // smart (emerald)
  "#6366f1", // d3qn (indigo)
  "#f59e0b", // bandit (amber)
  "#ec4899", // round_robin (pink)
  "#06b6d4", // random (cyan)
  "#8b5cf6", // other
];

export const SCHEDULER_TAGS: Record<string, { label: string; color: string }> = {
  sweep: { label: "Baseline", color: "#64748b" },
  random: { label: "Stochastic", color: "#94a3b8" },
  round_robin: { label: "Prioritized", color: "#ec4899" },
  bandit: { label: "Contextual Bandit", color: "#f59e0b" },
  smart: { label: "Cognitive Heuristic", color: "#10b981" },
  d3qn: { label: "Deep RL (D3QN)", color: "#6366f1" },
};
