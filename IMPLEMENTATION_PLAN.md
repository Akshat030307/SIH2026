# Implementation Plan — SIH26055: Smart Scan Strategy for Electronic Warfare

Source: `SIH26055 Electronic Warfare Smart Scan.pdf` (analysis & framework document).
Goal: a **closed-loop, ML-driven receiver scheduler** that decides *which frequency band to tune to next* and *how long to dwell*, with **no prior intelligence** about emitters, and that measurably beats open-loop (sequential) scanning on the SIH figures of merit.

> **Implementation status:** M0–M9 are completely implemented and deeply verified. See `README.md` for benchmark results across all 7 scenarios and how to reproduce them.
> Key accomplishments & architectural updates:
> - Python package structure in `src/smartscan/`.
> - Full simulation physics with scenarios S1–S7 (including S7 co-located air-defence emitters).
> - Deinterleaving fine-tuned and benchmarked on the real Alan Turing Institute Synthetic Radar Dataset (`checkpoints/deinterleaver_turing.pt`, `results/deinterleave/turing_summary.md`).
> - Production dashboard built with FastAPI backend streaming + modern React 19 / TypeScript / Vite frontend, including authentic client-side fallback simulation (`clientSim.ts`).

---

## 1. What we are actually building

| # | Component | Purpose | PDF section |
|---|-----------|---------|-------------|
| 1 | **RF environment simulator** (Gymnasium env) | Ground truth of every emitter in every band/time slot; produces the PDWs a narrowband scanning receiver would see | Simulation Environments |
| 2 | **Baseline schedulers** | Linear sweep, random, round-robin: the "open-loop" reference we must beat | Intro / Performance |
| 3 | **Phase 1: Bandit explorer** | UCB over frequency sub-bands for zero-knowledge spectrum discovery | Phase 1 |
| 4 | **Deinterleaver** | Split the interleaved PDW stream into per-emitter tracks | Deinterleaving |
| 5 | **Periodic-scan interceptor** | Estimate antenna rotation period → schedule dwells exactly at predicted main-beam arrival | Periodic Scan Receivers |
| 6 | **Phase 2: Behaviour predictor (PSR)** | Predict next mode / next emission of multi-function radars from past PDWs | Phase 2 |
| 7 | **Phase 3: D3QN scheduler** | Dueling Double DQN picking (channel, dwell) from a state that fuses 3–6 | Phase 3 |
| 8 | **Metrics & evaluation harness** | Pd, intercept rate, intercept time error, Pfa, % correct predictions, reward convergence | Performance Evaluation |
| 9 | **Edge inference package** | ONNX export + INT8 quantization + latency benchmark; vendor-agnostic PDW API | Deployment |
| 10 | **Demo dashboard** | Live waterfall + scheduler decisions + metric comparison vs baselines | (demo for judges) |

### Key design decisions (and where we deviate from the PDF)

1. **The simulator is the product's foundation, not the Turing dataset.** The Turing Synthetic Radar Dataset is a stream of PDWs recorded under a *fixed* receiver schedule (scan/stare). An RL scheduler has to control the receiver, so it needs an interactive environment. We use the Turing data **only to train and evaluate the deinterleaver**, and to sanity-check the simulator's pulse statistics.
2. **Use non-stationary UCB.** The PDF gives plain UCB1, which assumes stationary payoffs. Emitters switch on and off, so we implement **Sliding-Window UCB / Discounted UCB** (and Thompson sampling as an alternative). The PDF's "quantum-inspired" MAB is plain UCB1 in its maths, so we skip it unless judges ask.
3. **Estimate periods from sparse detections, not continuous SNR.** The PDF's autocorrelation needs one continuous SNR series over several rotations (2–12 s of dwelling). A scanning receiver rarely has that. Our main estimator works from the timestamps of beam-illumination detections, which arrive unevenly (period search by epoch folding, or a Lomb–Scargle periodogram). Continuous-dwell autocorrelation is a fallback for high-priority emitters.
4. **Build a strong heuristic scheduler before the RL agent.** UCB exploration, period lock, and priority-based conflict resolution together already make a smart scheduler. It is the fallback demo, and the bar D3QN must clear. If D3QN can't beat it, we show the heuristic and present RL as an ablation.
5. **Anti-lock-in dithering.** When the receiver's revisit interval shares a common factor with a radar's rotation period, the two can stay out of step forever. Every revisit schedule adds small random jitter to prevent this.
6. **Build PSR incrementally.** Start with an n-gram / variable-order Markov predictor, then a GRU next-event model, then spectral (Hankel/SVD) PSR learning as the "proper" PSR. All three share one `Predictor` interface, so the scheduler doesn't care which one runs.

---

## 2. Tech stack

| Area | Choice |
|------|--------|
| Language | Python 3.12 |
| Simulation | NumPy (vectorized per-dwell pulse generation), Gymnasium API |
| ML | PyTorch 2.x (D3QN, GRU predictor, deinterleaving Transformer encoder) |
| Clustering | scikit-learn DBSCAN / HDBSCAN |
| Signal proc. | SciPy (Lomb–Scargle, autocorrelation, period estimation) |
| Config | YAML + pydantic (scenario & emitter library) |
| Experiment tracking | TensorBoard + CSV evaluation logging |
| Edge | ONNX + ONNX Runtime (FP32 & INT8 quantization); latency benchmark |
| Dashboard | FastAPI WebSocket streaming backend + React 19 / TypeScript / Vite UI |
| Testing | pytest (28 tests covering env, sim, tracker, schedulers, deinterleaving, prediction) |

---

## 3. Repository layout

```
SIH2026/
├── configs/
│   ├── emitters/            # emitter class library (YAML): periodic, MFR, agile, LPI
│   ├── scenarios/           # S1..S7 scenario definitions
│   └── train/               # D3QN / predictor hyperparameters
├── src/smartscan/
│   ├── sim/                 # RF propagation, emitters, receiver, world timeline, Gymnasium env
│   ├── pdw.py               # PDW dataclass / structured numpy dtype (TOA, RF, PW, AOA, AMP)
│   ├── deinterleave/        # classical (DBSCAN/HDBSCAN), learned transformer, Turing fine-tuning
│   ├── tracker/             # per-emitter track store, period estimation, next-beam prediction
│   ├── predict/             # ngram.py, gru.py, spectral_psr.py (radar-word symbol prediction)
│   ├── schedulers/          # sweep, random, round-robin, bandit, heuristic, d3qn
│   ├── rl/                  # D3QN network, agent, prioritized replay, DQfD pre-training
│   ├── metrics.py           # all SIH figures of merit
│   ├── eval.py              # scenario × scheduler × seed benchmark harness
│   ├── export/              # ONNX FP32/INT8 export & latency benchmarking
│   └── server.py            # FastAPI + WebSocket streaming backend
├── dashboard/web/           # Modern React 19 + TypeScript + Vite frontend
│   ├── src/                 # LiveView, ResultsView, Waterfall, Charts, clientSim, Icons
│   └── dist/                # Pre-built assets served directly by FastAPI server
├── checkpoints/             # Trained models: d3qn_best.pt, deinterleaver_turing.pt
├── results/                 # Evaluation summaries, benchmark CSVs, plots
├── data/turing/             # Turing synthetic radar dataset recordings
└── tests/                   # 28 comprehensive automated unit and integration tests
```

---

## 4. Detailed work breakdown

### M0: Setup & domain grounding
- Get the **official SIH26055 problem statement** (the PDF cites `SIH_2026_Problem_Statements.json` but doesn't include it). Confirm the required deliverables and metrics word for word.
- Repo scaffolding, CI (pytest + ruff), shared `PDW` dtype, config loading.
- Check the **Turing Synthetic Radar Dataset** on Hugging Face (`alan-turing-institute/turing-synthetic-radar-dataset`): format, size, license. Stream a **subset** only; the full set is about 4 B pulses.
- Check access to the **"JC Wise Radar Emitter Database"**. We couldn't confirm it is freely available. Fallback: build our own emitter parameter tables from the open sources cited in the PDF (Air Power Australia pages, radar handbooks), stored in `configs/emitters/`.

**Exit:** repo builds, tests run, dataset subset loaded in a notebook, emitter-parameter source decided.

### M1: RF environment simulator (critical path)

**Receiver model**
- Total band configurable (default 0.5–18 GHz), split into `K` channels of instantaneous bandwidth (e.g. 500 MHz → ~35 channels).
- Discrete dwell set `D` (e.g. {1, 2, 5, 10, 20} ms); tuning latency `C_tune` (e.g. 50–500 µs, dead time).
- Pulse detected if `SNR + noise > threshold`. Also inject false PDWs by Poisson noise at a set per-channel rate.

**Emitter models** (each yields pulses lazily over `[t, t+dwell]`, vectorized):
| Type | Key parameters |
|------|---------------|
| Periodic mechanical scan | rotation period `T_e` (2–12 s), beamwidth → illumination `τ_e`, sidelobe level, fixed/stagger/jitter PRI |
| MFR (AESA) | semi-Markov mode chain {search, acquisition, TWS, STT}; each mode has its own PRI/PW/RF/beam-revisit params ("radar words") |
| Frequency agile | pulse-to-pulse / burst-to-burst hopping across a sub-band |
| LPI | low peak power (SNR near the noise floor), long PW / CW-like |
| Pop-up | emitter switches on mid-episode (tests re-exploration) |

**Physics:** one-way SNR from the PDF equation
`SNR = P_t G_t G_r λ² / ((4πR)² k T_eff B F L)`, with `G_t` = beam-pattern gain (Gaussian/sinc² main lobe + sidelobe floor) at the current antenna angle relative to the receiver bearing.

**Ground truth / info dict:** every emitter's illumination windows, true mode, true pulses (needed for Pd, time error, and prediction accuracy).

**Gymnasium API:** `action = (channel, dwell_idx)` flattened to `K·|D|` discrete actions. `obs` = PDWs detected in that dwell (the env stays unaware of the scheduler's internal state; feature building happens in the agent).

**Scenario suite** (`configs/scenarios/`), fixed seeds:
- S1 sparse: 3–5 periodic scanners
- S2 dense: 30–100 emitters, parameter overlap (Turing-like density)
- S3 MFR-heavy: modes switching
- S4 frequency-agile + LPI
- S5 pop-up threats mid-mission
- S6 lock-in trap: sweep period deliberately commensurate with rotation periods

**Tests:** hand-computable cases (a single emitter with known period gives known illumination times); SNR matches the formula; determinism under a seed.
**Performance target:** simulate ≥ 1000 dwells/s on a laptop for RL training.

### M2: Metrics, baselines, evaluation harness
Define metrics precisely (in `metrics.py`), measured against simulator ground truth:

| Metric | Definition we implement |
|--------|------------------------|
| **Pd** | intercepted illumination events / total illumination events that reached the receiver above sensitivity. "Intercepted" = ≥ `N_min` (e.g. 3) pulses of that emitter detected within the event. Report per emitter class and priority-weighted |
| **Average intercept rate** | unique emitters intercepted per second. Also **time-to-first-intercept** (TTFI) per emitter, a standard ESM metric worth showing |
| **Average intercept time error** | for dwells triggered by a prediction: mean abs(predicted arrival − actual first pulse arrival) |
| **Pfa (scheduler-level)** | fraction of prediction-triggered dwells where the predicted emitter produced no pulse |
| **% correct predictions** | accuracy of next-mode prediction (MFR) + hit rate of predicted illumination windows |
| **Reward convergence** | episodic return curve (mean ± std over seeds), plateau detection |

- Baselines: linear sweep, random, round-robin.
- `eval.py`: runs scenario × scheduler × N seeds, writes a CSV, and produces a comparison table + plots (plots feed the presentation).

**Exit:** a baseline results table exists for S1–S6. Every later phase has to improve on it.

### M3: Phase 1, bandit exploration
- Arms = channels. Reward = new-emitter discovery + priority-weighted intercepts (not raw pulse count, or a busy channel dominates).
- Implement UCB1 (the PDF's formula), **SW-UCB**, **D-UCB**, Thompson (Beta-Bernoulli on "channel active").
- Tune `c` and window/discount per scenario; the S5 pop-up scenario is the key test for non-stationarity.

**Exit:** the bandit beats sweep on intercept rate and TTFI in S1/S2/S5.

### M4: Deinterleaving
- **Classical baseline:** DBSCAN/HDBSCAN on normalized (RF, PW, AOA), then CDIF/SDIF PRI histogram to split same-parameter emitters.
- **Learned:** an encoder (MLP/Transformer over PDW windows) trained with a contrastive / metric-learning loss on Turing dataset labels, then HDBSCAN in the latent space (the PDF's approach).
- Evaluate with Adjusted Rand Index / V-measure on the Turing held-out split and on simulator output.
- Output: `Track` objects (emitter id, PDW history, parameter estimates) kept up to date by `tracker/`.

**Exit:** ARI ≥ target on the Turing subset. Tracks feed M5/M6 in real time. If learned lags classical, the classical version ships.

### M5: Periodic-scan interception (period lock)
- Per track: collect the timestamps of beam-illumination detections (amplitude peaks).
- Estimate `T_e` by **epoch folding / Lomb–Scargle** on uneven samples. For high-priority tracks, optionally spend one continuous dwell and use the PDF's **SNR autocorrelation** `ρ̂(k)` to refine.
- Predict the next illumination window `[t̂, t̂+τ̂_e]` with uncertainty. Schedule a short dwell centered on it, then refine the estimate after each hit (a Kalman-style update of phase and period).
- **Heuristic scheduler** (`heuristic.py`): imminent predicted windows get priority → conflicts resolved by threat priority and uncertainty → leftover time goes to the bandit explorer, with anti-lock-in jitter.

**Exit:** in S1/S6, period estimate error < 1% after a few rotations. Heuristic Pd ≫ sweep, and intercept time error drops sharply.

### M6: Phase 2, behaviour prediction (PSR track)
- Symbolize each MFR track into "radar words" (PRI bucket, PW bucket, RF-agility flag), using sim ground truth modes as labels for evaluation.
- Predictors behind one interface `predict(history) → (next-mode dist, next-emission time dist)`:
  1. n-gram / variable-order Markov (quick baseline)
  2. GRU/Transformer next-event model
  3. **Spectral PSR** (TPSR): Hankel matrix of histories × tests → SVD → linear predictive state update. This is the PDF's ⟨𝒪, ℋ, ℰ, Q⟩ formulation.
- Output the predictive-state vector as input features for D3QN.

**Exit:** % correct mode predictions reported for all three. The best one is wired into the tracker.

### M7: Phase 3, D3QN scheduler
**State `S_t`:** a per-channel feature matrix `K × F`, plus global features.
- Per channel: time since last visit, bandit mean & count, # tracked emitters, min predicted time-to-next-illumination, predicted window width, max threat priority, recent intercept count, PSR state summary.
- Global: current channel (for tuning cost), mission time.

**Action `A_t`:** `(channel, dwell)` → `K·|D|` discrete outputs, with **action masking** for invalid/hard-constrained actions. The period lock from M5 can inject hard constraints, as the PDF describes.

**Network:** a shared per-channel encoder θ (1D conv / shared MLP across channels) feeds into:
- value stream `V(s; θ, α)`
- advantage stream `A(s, a; θ, β)`
- combined as `Q = V + (A − mean_a A)`, exactly as in the PDF.

**Training:** Double DQN target, prioritized experience replay, n-step returns, target-network sync, ε-greedy decay. Curriculum S1 → S3 → S2 → S4/S5 with domain randomization over the emitter library.

**Reward:** `R_t = α·H_t − β·M_t − γ·|Δt_err| − δ·C_tune`.
- `H_t` = priority-weighted *new illumination events intercepted* (not raw pulses, which would let the agent "park" on a high-PRF emitter).
- `M_t` = predicted-and-missed illuminations.
- Tune α–δ; log the individual reward terms separately to spot reward hacking.

**Ablations:** −bandit features, −PSR features, −period lock, −dueling, −double.

**Exit:** D3QN ≥ heuristic on priority-weighted Pd in ≥ 4/6 scenarios; return curve converges across ≥ 3 seeds.

### M8: Edge deployment & API
- `SmartScanScheduler` API: `ingest(pdws: np.ndarray) → next_command(channel_freq_hz, dwell_s)`. Vendor-agnostic, takes standard PDW arrays.
- Export policy + predictor to ONNX, INT8 post-training quantization, and check the accuracy drop is < 1–2% on the metrics.
- Latency benchmark (CPU; Jetson/Raspberry Pi if available). Target: **sub-millisecond per decision** for the policy network. FPGA deployment is out of scope; describe it as the path forward.

### M9: Demo dashboard & presentation
- Streamlit app: scenario picker → live **spectrum waterfall** (ground truth vs. what the receiver saw), scheduler timeline, tracks with predicted windows, live metric counters.
- Side-by-side race: **sweep vs. heuristic vs. D3QN** on the same seed.
- Static results: metric table across S1–S6, reward convergence plot, ablation bars, latency numbers.
- Deliverables: PPT, 2–3 min demo video, README with reproduction commands.

---

## 5. Milestone order & dependencies

```
M0 ─► M1 (simulator) ─► M2 (metrics+baselines) ─┬─► M3 (bandit) ──┐
                                                 ├─► M5 (period lock) ─► heuristic scheduler ─┐
        M0 ─► M4 (deinterleaver, Turing data) ───┘                 │                          ├─► M7 (D3QN) ─► M8 (edge) ─► M9 (demo)
                                                  M6 (PSR) ─────────┴──────────────────────────┘
```

Suggested timeline for a 6-person team, ~8 weeks (compress for a hackathon sprint by skipping the ⭐ items):

| Week | Focus |
|------|-------|
| 1 | M0, M1 skeleton (periodic emitters + receiver + env), PDW dtype |
| 2 | M1 complete (MFR, agile, LPI, scenarios), M2 metrics + baselines, M4 classical deinterleaver |
| 3 | M3 bandits, M5 period estimation, heuristic scheduler, first comparison table |
| 4 | M6 n-gram + GRU predictors, M4 learned deinterleaver ⭐, D3QN implementation |
| 5 | D3QN training + curriculum + reward tuning, spectral PSR ⭐ |
| 6 | Ablations, M8 ONNX/quantization/latency, dashboard v1 |
| 7 | Hardening, full evaluation sweep, dashboard polish |
| 8 | Presentation, video, buffer |

**Minimum viable demo (if time collapses):** M1 + M2 + M3 + M5 → heuristic scheduler beating sweep on the dashboard. This already covers the core of the problem statement: closed-loop, no prior intelligence, periodic-scan interception, figures of merit.

### Suggested team split (6)
1. **Simulator lead:** M1, scenario suite
2. **Metrics/eval + dashboard:** M2, M9
3. **Bandits + period lock + heuristic:** M3, M5
4. **Deinterleaving:** M4 (Turing data)
5. **Behaviour prediction:** M6
6. **RL + edge:** M7, M8

---

## 6. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Simulator too slow for RL | Vectorize per-dwell pulse generation; Numba; parallel envs (Gymnasium vector env); coarser time resolution during training |
| D3QN doesn't beat the heuristic | Heuristic is the shipping fallback; present RL with ablations honestly; add heuristic features to the state and warm-start via imitation of the heuristic |
| Reward hacking (parking on busy channels) | Reward counts illumination *events* and new discoveries, not pulses; log reward terms separately |
| Turing dataset too large / format issues | Stream a subset; the simulator makes its own labelled PDWs as backup |
| JC Wise DB unavailable | Hand-curated YAML emitter library from open sources |
| Sim-to-real gap (judges' question) | Domain randomization; parameters from real system specs; vendor-agnostic PDW API |
| Lock-in with periodic emitters | Revisit jitter; S6 scenario as a regression test |

---

## 7. Definition of done
- [ ] `python -m smartscan.eval --scenarios all --schedulers all --seeds 5` produces the full metric table reproducibly
- [ ] Smart scheduler beats linear sweep on Pd, intercept rate, TTFI, and intercept time error in every scenario
- [ ] Periodic-scan interception shown working, including the lock-in scenario
- [ ] MFR mode prediction accuracy reported
- [ ] D3QN reward convergence plot + ablations
- [ ] ONNX model with measured per-decision latency
- [ ] Dashboard demo + presentation + README
