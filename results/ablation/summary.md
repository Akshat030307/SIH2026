# Smart-scheduler ablation (threat-weighted Pd, held-out seeds)

| scheduler | variant | S1_sparse | S2_dense | S3_mfr | S4_agile_lpi | S5_popup | S6_lockin | S7_colocated | mean | TTFI [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| smart | full scheduler | 0.678 | 0.477 | 0.606 | 0.826 | 0.55 | 0.878 | 0.349 | 0.623 | 6.47 |
| smart-no_lock | no period lock (no predicted-beam dwells) | 0.359 | 0.333 | 0.406 | 0.69 | 0.16 | 0.362 | 0.156 | 0.352 | 5.645 |
| smart-no_acquire | no acquisition revisits (locks form only by chance) | 0.153 | 0.258 | 0.318 | 0.333 | 0.379 | 0.176 | 0.294 | 0.273 | 12.663 |
| smart-no_lock_no_acquire | tracker only: bandit exploration | 0.133 | 0.117 | 0.095 | 0.251 | 0.084 | 0.123 | 0.083 | 0.127 | 12.854 |
| smart-sweep_explore | exploration by linear sweep instead of D-UCB (no jitter) | 0.671 | 0.529 | 0.625 | 0.752 | 0.515 | 0.338 | 0.36 | 0.541 | 10.297 |
| smart-random_explore | exploration uniformly at random instead of D-UCB | 0.654 | 0.478 | 0.652 | 0.831 | 0.542 | 0.838 | 0.318 | 0.616 | 6.937 |
| smart-no_jitter | D-UCB exploration without dwell jitter | 0.692 | 0.485 | 0.615 | 0.796 | 0.494 | 0.641 | 0.331 | 0.579 | 8.395 |
