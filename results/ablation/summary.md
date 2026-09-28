# Smart-scheduler ablation (threat-weighted Pd, held-out seeds)

| scheduler | variant | S1_sparse | S2_dense | S3_mfr | S4_agile_lpi | S5_popup | S6_lockin | S7_colocated | mean | TTFI [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| smart | full scheduler | 0.678 | 0.476 | 0.572 | 0.819 | 0.529 | 0.878 | 0.317 | 0.61 | 6.549 |
| smart-no_lock | no period lock (no predicted-beam dwells) | 0.359 | 0.333 | 0.406 | 0.69 | 0.16 | 0.362 | 0.156 | 0.352 | 5.645 |
| smart-no_acquire | no acquisition revisits (locks form only by chance) | 0.153 | 0.171 | 0.184 | 0.341 | 0.283 | 0.176 | 0.192 | 0.214 | 12.921 |
| smart-no_lock_no_acquire | tracker only: bandit exploration | 0.133 | 0.117 | 0.095 | 0.251 | 0.084 | 0.123 | 0.083 | 0.127 | 12.854 |
| smart-sweep_explore | exploration by linear sweep instead of D-UCB (no jitter) | 0.671 | 0.472 | 0.604 | 0.758 | 0.459 | 0.338 | 0.315 | 0.517 | 10.403 |
| smart-random_explore | exploration uniformly at random instead of D-UCB | 0.654 | 0.466 | 0.612 | 0.839 | 0.462 | 0.828 | 0.287 | 0.593 | 7.059 |
| smart-no_jitter | D-UCB exploration without dwell jitter | 0.692 | 0.463 | 0.639 | 0.804 | 0.461 | 0.641 | 0.289 | 0.57 | 8.691 |
