# Benchmark (5 seeds, eval seeds 100..104)

### S1_sparse

| Scheduler | Pd | Pd (threat-wtd) | Intercept rate [emitters/s] | TTFI [s] | Intercept time err [ms] | Pfa | Correct predictions | Idle fraction |
|---|---|---|---|---|---|---|---|---|
| sweep | 0.239 | 0.208 | 0.115 | 7.92 | – | – | – | 0.95 |
| random | 0.197 | 0.200 | 0.100 | 15.02 | – | – | – | 0.95 |
| round_robin | 0.237 | 0.213 | 0.105 | 12.60 | – | – | – | 0.94 |
| bandit | 0.223 | 0.204 | 0.110 | 10.72 | – | – | – | 0.95 |
| smart | 0.673 | 0.678 | 0.125 | 9.37 | 7.9 | 0.087 | 0.913 | 0.82 |
| d3qn | 0.757 | 0.782 | 0.115 | 8.73 | 8.1 | 0.100 | 0.900 | 0.81 |

### S2_dense

| Scheduler | Pd | Pd (threat-wtd) | Intercept rate [emitters/s] | TTFI [s] | Intercept time err [ms] | Pfa | Correct predictions | Idle fraction |
|---|---|---|---|---|---|---|---|---|
| sweep | 0.131 | 0.108 | 0.850 | 8.24 | – | – | – | 0.86 |
| random | 0.129 | 0.116 | 0.805 | 11.79 | – | – | – | 0.86 |
| round_robin | 0.164 | 0.142 | 0.820 | 9.19 | – | – | – | 0.83 |
| bandit | 0.141 | 0.121 | 0.825 | 10.88 | – | – | – | 0.85 |
| smart | 0.489 | 0.477 | 0.800 | 7.31 | 6.1 | 0.373 | 0.627 | 0.36 |
| d3qn | 0.505 | 0.504 | 0.720 | 9.66 | 6.5 | 0.406 | 0.594 | 0.34 |

### S3_mfr

| Scheduler | Pd | Pd (threat-wtd) | Intercept rate [emitters/s] | TTFI [s] | Intercept time err [ms] | Pfa | Correct predictions | Idle fraction |
|---|---|---|---|---|---|---|---|---|
| sweep | 0.079 | 0.074 | 0.240 | 9.01 | – | – | – | 0.96 |
| random | 0.092 | 0.089 | 0.230 | 9.97 | – | – | – | 0.96 |
| round_robin | 0.101 | 0.097 | 0.230 | 10.11 | – | – | – | 0.94 |
| bandit | 0.096 | 0.091 | 0.250 | 8.83 | – | – | – | 0.95 |
| smart | 0.586 | 0.606 | 0.250 | 4.97 | 4.6 | 0.528 | 0.472 | 0.56 |
| d3qn | 0.623 | 0.649 | 0.235 | 5.85 | 4.7 | 0.601 | 0.399 | 0.55 |

### S4_agile_lpi

| Scheduler | Pd | Pd (threat-wtd) | Intercept rate [emitters/s] | TTFI [s] | Intercept time err [ms] | Pfa | Correct predictions | Idle fraction |
|---|---|---|---|---|---|---|---|---|
| sweep | 0.224 | 0.227 | 0.225 | 11.52 | – | – | – | 0.99 |
| random | 0.231 | 0.243 | 0.220 | 12.37 | – | – | – | 0.99 |
| round_robin | 0.258 | 0.267 | 0.200 | 11.36 | – | – | – | 0.99 |
| bandit | 0.255 | 0.256 | 0.215 | 10.02 | – | – | – | 0.99 |
| smart | 0.837 | 0.826 | 0.215 | 5.45 | 7.8 | 0.086 | 0.914 | 0.93 |
| d3qn | 0.814 | 0.825 | 0.225 | 6.52 | 7.2 | 0.084 | 0.916 | 0.93 |

### S5_popup

| Scheduler | Pd | Pd (threat-wtd) | Intercept rate [emitters/s] | TTFI [s] | Intercept time err [ms] | Pfa | Correct predictions | Idle fraction |
|---|---|---|---|---|---|---|---|---|
| sweep | 0.099 | 0.079 | 0.310 | 8.59 | – | – | – | 0.93 |
| random | 0.091 | 0.081 | 0.305 | 9.83 | – | – | – | 0.93 |
| round_robin | 0.119 | 0.101 | 0.305 | 5.90 | – | – | – | 0.91 |
| bandit | 0.100 | 0.086 | 0.300 | 8.91 | – | – | – | 0.93 |
| smart | 0.552 | 0.550 | 0.310 | 5.14 | 5.6 | 0.388 | 0.612 | 0.59 |
| d3qn | 0.528 | 0.524 | 0.305 | 5.78 | 4.8 | 0.447 | 0.553 | 0.65 |

### S6_lockin

| Scheduler | Pd | Pd (threat-wtd) | Intercept rate [emitters/s] | TTFI [s] | Intercept time err [ms] | Pfa | Correct predictions | Idle fraction |
|---|---|---|---|---|---|---|---|---|
| sweep | 0.015 | 0.011 | 0.005 | 37.12 | – | – | – | 0.99 |
| random | 0.132 | 0.136 | 0.130 | 15.56 | – | – | – | 0.98 |
| round_robin | 0.183 | 0.186 | 0.150 | 8.99 | – | – | – | 0.98 |
| bandit | 0.062 | 0.052 | 0.035 | 29.52 | – | – | – | 0.98 |
| smart | 0.881 | 0.878 | 0.150 | 4.40 | 10.3 | 0.039 | 0.961 | 0.89 |
| d3qn | 0.923 | 0.925 | 0.150 | 1.80 | 8.9 | 0.031 | 0.969 | 0.84 |

### S7_colocated

| Scheduler | Pd | Pd (threat-wtd) | Intercept rate [emitters/s] | TTFI [s] | Intercept time err [ms] | Pfa | Correct predictions | Idle fraction |
|---|---|---|---|---|---|---|---|---|
| sweep | 0.090 | 0.080 | 0.485 | 9.78 | – | – | – | 0.87 |
| random | 0.096 | 0.087 | 0.505 | 9.13 | – | – | – | 0.86 |
| round_robin | 0.113 | 0.101 | 0.490 | 8.26 | – | – | – | 0.83 |
| bandit | 0.098 | 0.089 | 0.505 | 9.89 | – | – | – | 0.86 |
| smart | 0.359 | 0.349 | 0.475 | 8.65 | 4.4 | 0.357 | 0.643 | 0.40 |
| d3qn | 0.353 | 0.350 | 0.415 | 12.15 | 4.8 | 0.340 | 0.660 | 0.38 |
