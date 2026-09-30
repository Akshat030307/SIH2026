# Turing Synthetic Radar Dataset Benchmark (test_scan)

Evaluated on 10 pulse train recordings (subset of 5000 pulses each).

| method | ari | ami | v_measure | homogeneity | completeness | n_true | n_pred | noise_frac | sec |
|---|---|---|---|---|---|---|---|---|---|
| dbscan | 0.428 | 0.586 | 0.592 | 0.983 | 0.464 | 12.5 | 41.8 | 0.011 | 0.087 |
| hdbscan | 0.428 | 0.59 | 0.595 | 0.984 | 0.468 | 12.5 | 33.8 | 0.007 | 0.071 |
| learned_sim | 0.44 | 0.618 | 0.623 | 0.929 | 0.536 | 12.5 | 20.6 | 0.014 | 0.445 |
| learned_turing | 0.461 | 0.591 | 0.595 | 0.955 | 0.48 | 12.5 | 25 | 0.02 | 0.404 |
