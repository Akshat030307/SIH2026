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
## Results

**Headline:** averaged over 7 scenarios, the learned D3QN scheduler reaches a threat-weighted Pd of **0.65** and the smart heuristic **0.62**, against **0.11** for a legacy open-loop sweep (5.8× better). In the lock-in scenario the sweep catches 1% of illuminations; the smart schedulers catch 92%.

### Scheduler benchmark (5 held-out seeds per scenario)

Threat-weighted probability of detection (each illumination weighted by the emitter's threat):

| scenario | sweep | random | round_robin | bandit | smart | d3qn |
|---|---|---|---|---|---|---|
| S1_sparse | 0.208 | 0.2 | 0.213 | 0.204 | 0.678 | 0.782 |
| S2_dense | 0.108 | 0.116 | 0.142 | 0.121 | 0.477 | 0.504 |
| S3_mfr | 0.074 | 0.089 | 0.097 | 0.091 | 0.606 | 0.649 |
| S4_agile_lpi | 0.227 | 0.243 | 0.267 | 0.256 | 0.826 | 0.825 |
| S5_popup | 0.079 | 0.081 | 0.101 | 0.086 | 0.55 | 0.524 |
| S6_lockin | 0.011 | 0.136 | 0.186 | 0.052 | 0.878 | 0.925 |
| S7_colocated | 0.08 | 0.087 | 0.101 | 0.089 | 0.349 | 0.35 |
| **mean** | 0.112 | 0.136 | 0.158 | 0.128 | 0.623 | 0.651 |

All figures of merit, averaged over scenarios:

| scheduler | Pd | Pd (threat-wtd) | Intercept rate [emitters/s] | TTFI [s] | Intercept time err [ms] | Pfa | Correct predictions |
|---|---|---|---|---|---|---|---|
| sweep | 0.125 | 0.112 | 0.319 | 13.17 | – | – | – |
| random | 0.138 | 0.136 | 0.328 | 11.954 | – | – | – |
| round_robin | 0.168 | 0.158 | 0.329 | 9.488 | – | – | – |
| bandit | 0.139 | 0.128 | 0.32 | 12.682 | – | – | – |
| smart | 0.625 | 0.623 | 0.332 | 6.47 | 6.659 | 0.265 | 0.735 |
| d3qn | 0.643 | 0.651 | 0.309 | 7.214 | 6.417 | 0.287 | 0.713 |

### Ablation of the smart scheduler

| scheduler | variant | S1_sparse | S2_dense | S3_mfr | S4_agile_lpi | S5_popup | S6_lockin | S7_colocated | mean | TTFI [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| smart | full scheduler | 0.678 | 0.477 | 0.606 | 0.826 | 0.55 | 0.878 | 0.349 | 0.623 | 6.47 |
| smart-no_lock | no period lock (no predicted-beam dwells) | 0.359 | 0.333 | 0.406 | 0.69 | 0.16 | 0.362 | 0.156 | 0.352 | 5.645 |
| smart-no_acquire | no acquisition revisits (locks form only by chance) | 0.153 | 0.258 | 0.318 | 0.333 | 0.379 | 0.176 | 0.294 | 0.273 | 12.663 |
| smart-no_lock_no_acquire | tracker only: bandit exploration | 0.133 | 0.117 | 0.095 | 0.251 | 0.084 | 0.123 | 0.083 | 0.127 | 12.854 |
| smart-sweep_explore | exploration by linear sweep instead of D-UCB (no jitter) | 0.671 | 0.529 | 0.625 | 0.752 | 0.515 | 0.338 | 0.36 | 0.541 | 10.297 |
| smart-random_explore | exploration uniformly at random instead of D-UCB | 0.654 | 0.478 | 0.652 | 0.831 | 0.542 | 0.838 | 0.318 | 0.616 | 6.937 |
| smart-no_jitter | D-UCB exploration without dwell jitter | 0.692 | 0.485 | 0.615 | 0.796 | 0.494 | 0.641 | 0.331 | 0.579 | 8.395 |

### D3QN training

Validation threat-weighted Pd during training (7 validation worlds, seed 500). Run 1's replay buffer overwrote the DQfD demonstrations after ~280k steps, and the policy then collapsed. Run 3 keeps them in a protected region and stays stable:

| online steps | run 1 (demos overwritten) | run 3 (demos protected) |
|---|---|---|
| 100000 | 0.578 | 0.603 |
| 200000 | 0.594 | 0.609 |
| 300000 | 0.62 | 0.533 |
| 400000 | 0.586 | 0.595 |
| 500000 | 0.509 | 0.613 |
| 600000 | 0.349 | 0.588 |

Checkpoint selection on 35 validation episodes (seeds 500–504 × 7 scenarios, disjoint from test). The winner is shipped as `checkpoints/d3qn_best.pt`:

| candidate | mean | std |
|---|---|---|
| run1_300k | 0.654 | 0.23 |
| run3_600k | 0.641 | 0.21 |
| run3_500k | 0.625 | 0.183 |
| run2_pretrained | 0.591 | 0.201 |
| run3_pretrained | 0.54 | 0.234 |

### MFR behaviour prediction

| observed | model | next_word_acc | next_mode_acc | transition_acc | change_auc | where_next_acc | nll | steps | transitions |
|---|---|---|---|---|---|---|---|---|---|
| all | unigram | 0.707 | 0.707 | 0.283 | 0.744 | 0.489 | 1.135 | 26116 | 2311 |
| all | persistence | 0.892 | 0.912 | 0 | 0.237 | 0.675 | 0.757 | 26116 | 2311 |
| all | ngram1 | 0.892 | 0.911 | 0 | 0.754 | 0.785 | 0.451 | 26116 | 2311 |
| all | ngram3 | 0.894 | 0.912 | 0.003 | 0.803 | 0.777 | 0.418 | 26116 | 2311 |
| all | ngram6 | 0.89 | 0.91 | 0.019 | 0.819 | 0.757 | 0.429 | 26116 | 2311 |
| all | psr8 | 0.884 | 0.902 | 0.002 | 0.765 | 0.782 | 0.467 | 26116 | 2311 |
| all | psr16 | 0.881 | 0.899 | 0.02 | 0.765 | 0.782 | 0.529 | 26116 | 2311 |
| all | psr24 | 0.874 | 0.89 | 0.013 | 0.762 | 0.775 | 0.542 | 26116 | 2311 |
| all | gru | 0.894 | 0.91 | 0.044 | 0.865 | 0.756 | 0.346 | 26116 | 2311 |
| 70% | unigram | 0.707 | 0.707 | 0.297 | 0.75 | 0.507 | 1.135 | 18257 | 2195 |
| 70% | persistence | 0.862 | 0.88 | 0 | 0.237 | 0.657 | 0.967 | 18257 | 2195 |
| 70% | ngram1 | 0.862 | 0.88 | 0 | 0.754 | 0.775 | 0.541 | 18257 | 2195 |
| 70% | ngram3 | 0.861 | 0.879 | 0.012 | 0.797 | 0.757 | 0.518 | 18257 | 2195 |
| 70% | ngram6 | 0.859 | 0.876 | 0.034 | 0.81 | 0.736 | 0.534 | 18257 | 2195 |
| 70% | psr8 | 0.851 | 0.867 | 0.002 | 0.766 | 0.76 | 0.57 | 18257 | 2195 |
| 70% | psr16 | 0.847 | 0.864 | 0.014 | 0.765 | 0.762 | 0.586 | 18257 | 2195 |
| 70% | psr24 | 0.837 | 0.852 | 0.01 | 0.761 | 0.745 | 0.634 | 18257 | 2195 |
| 70% | gru | 0.862 | 0.878 | 0.044 | 0.84 | 0.737 | 0.435 | 18257 | 2195 |

### Deinterleaving

| scenario | method | ari | ami | v_measure | homogeneity | completeness | n_true | n_pred | sec |
|---|---|---|---|---|---|---|---|---|---|
| S2_dense | dbscan | 0.989 | 0.969 | 0.971 | 0.991 | 0.952 | 27.2 | 47.2 | 0.23 |
| S2_dense | hdbscan | 0.995 | 0.981 | 0.982 | 0.99 | 0.974 | 27.2 | 30.2 | 0.146 |
| S2_dense | learned | 0.978 | 0.966 | 0.967 | 0.964 | 0.971 | 27.2 | 20.6 | 0.908 |
| S7_colocated | dbscan | 0.8 | 0.843 | 0.844 | 0.979 | 0.759 | 15.1 | 42 | 0.691 |
| S7_colocated | hdbscan | 0.808 | 0.852 | 0.853 | 0.993 | 0.763 | 15.1 | 29.4 | 0.368 |
| S7_colocated | learned | 0.885 | 0.916 | 0.916 | 0.91 | 0.935 | 15.1 | 16.4 | 3.015 |

### Edge inference

Model sizes: d3qn_fp32.onnx 601 KiB, d3qn_int8.onnx 171 KiB

FP32 vs INT8 greedy-action agreement on 12935 recorded states: **0.764**

## Latency per decision (µs)

| path | p50_us | p99_us |
|---|---|---|
| PyTorch CPU | 155.4 | 1169.6 |
| ONNX Runtime FP32 (1 thread) | 93.6 | 143.6 |
| ONNX Runtime INT8 (1 thread) | 197.2 | 270.2 |
| Full decision (features + INT8), S2_dense | 755.7 | 1680.2 |

## Scheduling metrics, FP32 vs INT8 (seed 150)

| scenario | model | pd | pd_weighted |
|---|---|---|---|
| S1_sparse | fp32 | 0.643 | 0.705 |
| S1_sparse | int8 | 0.929 | 0.919 |
| S3_mfr | fp32 | 0.729 | 0.762 |
| S3_mfr | int8 | 0.704 | 0.736 |
| S6_lockin | fp32 | 0.883 | 0.867 |
| S6_lockin | int8 | 0.745 | 0.793 |
**Recommendation:** deploy the FP32 ONNX policy. On this ~150k-parameter network, dynamic INT8
is slower on CPU (quantize/dequantize overhead dominates) and changes a noticeable share of greedy
actions. The decision loop is dominated by Python feature extraction, which is the part to port
to C++/FPGA for microsecond-level budgets.

<!-- RESULTS:END -->

## Quick start

```bash
uv sync                                    # Python 3.12 env; pulls CUDA PyTorch (cu128, ~3 GB)
#   CPU-only alternative (all models here are small enough; this repo's results were trained on CPU):
#   uv pip install "torch==2.11.0+cpu" --index-url https://download.pytorch.org/whl/cpu
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

## Limitations

* **Simulated data only.** All results come from our simulator, whose emitter parameters are
  representative, not taken from any specific real system. The Turing loader is in place but
  was not run (it needs a Hugging Face token).
* **D3QN vs heuristic.** D3QN beats the hand-built heuristic on average (0.65 vs 0.62 threat-weighted
  Pd) but not everywhere. It is slightly worse on pop-up threats (S5) and on time to first intercept.
  Most of the gain over open-loop scanning comes from the tracker, period lock and acquisition, which
  both schedulers share (see the ablation).
* **Training variance.** The DQfD pre-training outcome varies between runs (0.54–0.63 validation).
  The shipped checkpoint was chosen on validation seeds only.
* **Co-located emitters stay hard.** S7 threat-weighted Pd is ~0.35. The online tracker associates
  pulses by AOA/PW/RF gating. The learned deinterleaver separates co-located radars better offline
  (ARI 0.885 vs 0.808), but it is not wired into the online tracker, which runs per dwell.
* **INT8 quantization** doesn't help this small policy network; deploy FP32 ONNX.

