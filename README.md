# Smart Scan: ML-driven ESM receiver scheduling (SIH26055)

A closed-loop scheduler for a narrowband scanning ESM/RWR receiver. It decides
**which frequency band to tune to next and how long to listen**, with no
prior intelligence about the emitters. It learns the RF environment online,
predicts when scanning and multi-function radars will illuminate the receiver,
and places its dwells there.

Everything runs against a physics-based RF simulator whose ground truth
scores the SIH figures of merit: probability of detection, intercept rate,
intercept time error, Pfa, correct predictions, and reward convergence.

<!-- RESULTS:START -->
<!-- RESULTS:END -->

## Quick start

```bash
uv sync                                    # Python 3.12 env (CUDA PyTorch from the cu128 index)
uv run pytest                              # unit tests

# benchmark all schedulers on all scenarios (held-out seeds 100..104)
uv run python -m smartscan.eval --seeds 5

# dashboard: API + React UI
uv run uvicorn smartscan.server:app --port 8000
cd dashboard/web && npm install && npm run dev   # → http://localhost:5173
#   or: npm run build, then open http://localhost:8000 (the API serves the built app)
#   demo link: http://localhost:5173/?autorun=S6_lockin&schedulers=sweep,smart,d3qn
```

## What's inside

| PDF component | Module | Notes |
|---|---|---|
| RF environment / ground truth | `smartscan/sim/` | One-way radar equation, Gaussian beams with sidelobe floor, circular scanners, MFR semi-Markov modes, PRI stagger/jitter/sliding/dwell-switch, pulse/burst frequency agility, LPI, pop-ups, co-located sites; superhet receiver with retune dead time, detection threshold, PDW noise, false pulses. Pulse trains are generated lazily and deterministically (35k–98k dwells/s). |
| Emitter database | `configs/emitters/library.yaml` | Class-level parameter ranges from open sources (the "JC Wise" DB wasn't available). |
| Scenarios | `configs/scenarios/S1..S7` | sparse, dense (~40 emitters), MFR, agile+LPI, pop-up threats, lock-in trap, co-located air-defence sites |
| Figures of merit | `smartscan/metrics.py` | Pd, threat-weighted Pd, intercept rate, TTFI, intercept time error, Pfa, correct predictions |
| Open-loop baselines | `schedulers/baselines.py` | linear sweep, random, round-robin |
| Phase 1: bandit exploration | `schedulers/bandit.py` | UCB1 (PDF), Discounted UCB (default), SW-UCB, Thompson |
| Deinterleaving (online) | `tracker/tracker.py` | gap-split clustering + gated association; frequency-hop aware |
| Deinterleaving (offline) | `deinterleave/` | DBSCAN/HDBSCAN, SDIF, learned Transformer embeddings + HDBSCAN; Turing dataset loader |
| Periodic-scan interception | `tracker/period.py` | scan period from sparse illumination hits (approx. GCD + weighted LS); the PDF's SNR autocorrelation |
| Smart heuristic scheduler | `schedulers/heuristic.py` | predicted-beam dwells + acquisition + D-UCB exploration + anti-lock-in jitter |
| Phase 2: behaviour prediction | `predict/` | radar-word symbols; n-gram, spectral PSR (Hankel/SVD), GRU |
| Phase 3: D3QN | `rl/` | dueling double DQN, channel-shared encoder, n-step, prioritized replay, DQfD demonstrations |
| Edge deployment | `export/edge.py`, `api.py` | ONNX + INT8, latency benchmark, vendor-agnostic `SmartScanScheduler` API |
| Demo | `server.py`, `dashboard/web` | live side-by-side missions, results browser |

## Reproducing the results

```bash
uv run python -m smartscan.eval --seeds 5                        # results/benchmark/
uv run python -m smartscan.ablation --seeds 5                    # results/ablation/
uv run python -m smartscan.predict.benchmark                     # results/predict/
uv run python -m smartscan.deinterleave.learned train && \
uv run python -m smartscan.deinterleave.learned eval             # results/deinterleave/
uv run python -m smartscan.rl.train --steps 1000000 --envs 12    # checkpoints/d3qn_best.pt, runs/ (TensorBoard)
uv run python -m smartscan.eval --seeds 5 --schedulers sweep,random,round_robin,bandit,smart,d3qn
uv run python -m smartscan.export.edge                           # results/edge/
```

Training uses world seeds 0–99; every reported number uses held-out seeds 100–104; D3QN checkpoint selection uses validation seeds 500–501.

### Turing Synthetic Radar Dataset

The dataset is gated (auto-approved) on Hugging Face. Accept the terms at
<https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset>,
then:

```bash
export HF_TOKEN=hf_...
uv run python -m smartscan.deinterleave.turing download --split train_scan --n 30
uv run python -m smartscan.deinterleave.turing download --split test_scan --n 10
uv run python -m smartscan.deinterleave.turing inspect --split train_scan
```

## Integrating with a receiver

```python
from smartscan.api import SmartScanScheduler

sched = SmartScanScheduler({"f_min_ghz": 2, "f_max_ghz": 18, "ibw_mhz": 500}, policy="smart")  # or "d3qn"
while True:
    cmd = sched.next_command(t_now)              # cmd.center_freq_hz, cmd.dwell_s, cmd.reason
    pdws = receiver.dwell(cmd.center_freq_hz, cmd.dwell_s)   # (N, 5): toa[s], rf[Hz], pw[s], aoa[deg], amp[dB]
    sched.ingest(pdws, t_listen, t_end)
```

## Design notes and deviations from the analysis PDF

* **The simulator, not the Turing dataset, is the RL environment.** The Turing
  data was recorded under a fixed receiver schedule, so it can't respond to
  scheduling decisions. It's used for deinterleaving only.
* **Non-stationary bandits.** Plain UCB1 assumes stationary payoffs.
  Discounted UCB is the default, and exploration pays out only for newly
  discovered emitters.
* **Period estimation from sparse hits.** A scanning receiver rarely has the
  continuous SNR series the PDF's autocorrelation needs. The period comes from
  the timing of the illuminations actually observed; the autocorrelation
  estimator is still implemented.
* **The heuristic comes first.** The heuristic scheduler is the bar D3QN has
  to clear. The agent sees the same perception state and is seeded with the
  heuristic's demonstrations (DQfD).
* **Radar words, not raw PDWs, for behaviour prediction.** Modes are hidden.
  Predictors model sequences of PRI×PW words, and ground-truth modes are used
  only for scoring.
