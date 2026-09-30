# Benchmark (1 seeds, eval seeds 100..100)

### S6_lockin

| Scheduler | Pd | Pd (threat-wtd) | Intercept rate [emitters/s] | TTFI [s] | Intercept time err [ms] | Pfa | Correct predictions | Idle fraction |
|---|---|---|---|---|---|---|---|---|
| sweep | 0.000 | 0.000 | 0.000 | 38.59 | – | – | – | 1.00 |
| smart | 0.962 | 0.962 | 0.150 | 0.51 | 10.5 | 0.015 | 0.985 | 0.93 |
| d3qn | 0.911 | 0.906 | 0.150 | 1.27 | 8.9 | 0.016 | 0.984 | 0.93 |
