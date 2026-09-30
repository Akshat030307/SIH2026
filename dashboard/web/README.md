# SmartScan ESM Dashboard (`dashboard/web`)

Modern, high-performance web dashboard for real-time visualization and held-out benchmarking of machine learning-driven ESM (Electronic Support Measures) receiver scheduling.

Built with **React 19**, **TypeScript**, **Vite**, and clean **CSS custom properties** with a tactical dark theme.

---

## Features

- **Side-by-Side Schedulers**: Run and visualize multiple schedulers (`sweep`, `smart`, `d3qn`, `bandit`, `round_robin`, `random`) simultaneously against the exact same RF environment and ground-truth pulse trains.
- **Real-Time Waterfall Oscilloscope**: Canvas-rendered frequency channel dwells, target emitter pulse hits, and main-beam illuminations with precise clipping and hover inspection.
- **Adaptive Balanced Layout**: Dynamically adjusts between 1, 2, or 3 columns, and automatically organizes 4 schedulers into a balanced 2×2 grid with spacious 3×2 KPI metric cards.
- **Offline Client Simulation Engine (`clientSim.ts`)**: Built-in client-side simulation driver reproducing circular scan geometries, open-loop harmonic sweep lockouts, and cognitive period locks at 10 Hz. Allows full interactive mission demos even when the PyTorch backend is offline.
- **Mission Debrief**: Automatic post-mission telemetry summary highlighting the leading policy and threat-weighted performance margin.
- **Benchmark Evaluation Browser**: View held-out test seed results, compare metrics across scenarios, inspect gain relative to baseline sweep (`Δ vs Baseline`), and read technical research notes with formatted Markdown tables and code pills.
- **Keyboard Shortcuts**: Press `Spacebar` anytime to pause or resume an active mission.

---

## Directory Structure

```
dashboard/web/
├── src/
│   ├── components/
│   │   ├── LiveView.tsx        # Main live mission execution view & scheduler cards
│   │   ├── ResultsView.tsx     # Held-out benchmark metrics browser & markdown reader
│   │   ├── Waterfall.tsx       # HTML5 Canvas RF waterfall & dwell oscilloscope
│   │   ├── Charts.tsx          # Bar and line chart components
│   │   └── Icons.tsx           # Lightweight custom SVG icon set
│   ├── App.tsx                 # Top-level shell, header, connection badge & routing
│   ├── clientSim.ts            # Client-side EW simulation fallback engine
│   ├── theme.ts                # Color palettes, scheduler tags, emitter class hues
│   ├── types.ts                # TypeScript interfaces for WebSocket protocol & telemetry
│   ├── styles.css              # Dark theme styling, grid layouts, and responsive breakpoints
│   └── main.tsx                # React entry point
├── dist/                       # Production assets served directly by FastAPI backend
├── index.html                  # HTML entry point
├── package.json                # Scripts & dependencies
├── tsconfig.json               # TypeScript compiler configuration
└── vite.config.ts              # Vite bundler & API proxy configuration
```

---

## Getting Started

### 1. Install Dependencies
```bash
npm install
```

### 2. Development Mode (with Hot-Module Reloading)
```bash
npm run dev
```
Starts Vite at **`http://localhost:5173`**. API and WebSocket calls (`/api/...`) are automatically proxied to the Python backend on `http://localhost:8000`.

### 3. Production Build
```bash
npm run build
```
Compiles TypeScript (`tsc -b`) and generates production bundles in `dist/`. The FastAPI server (`smartscan.server:app`) automatically serves these assets at `/` and `/assets`.

---

## WebSocket Protocol (`/api/run`)

1. **Client Init**: On connect, the frontend sends a JSON configuration:
   ```json
   {
     "scenario": "S6_lockin",
     "seed": 100,
     "schedulers": ["sweep", "smart", "d3qn"],
     "speed": 4.0,
     "frame_s": 0.1
   }
   ```
2. **Server Init**: Backend sends scene parameters, channels, and emitter roster:
   ```json
   {
     "type": "init",
     "scenario": "S6_lockin",
     "duration": 40.0,
     "channels": [2.0, 2.1],
     "emitters": [...]
   }
   ```
3. **Telemetry Streaming**: Backend sends 10 Hz frame updates containing dwell windows, ground-truth events, and real-time metrics:
   ```json
   {
     "type": "frame",
     "t": 1.4,
     "runs": {
       "smart": { "dwells": [...], "metrics": { "pd_weighted": 0.94 } }
     },
     "events": [...]
   }
   ```
4. **Live Controls**: While running, the client sends control messages:
   - `{"cmd": "speed", "value": 8}`
   - `{"cmd": "pause"}` / `{"cmd": "resume"}`
   - `{"cmd": "stop"}`
