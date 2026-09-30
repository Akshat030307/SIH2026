# Deinterleaving (held-out seeds)

| scenario | method | ari | ami | v_measure | homogeneity | completeness | n_true | n_pred | sec |
|---|---|---|---|---|---|---|---|---|---|
| S2_dense | dbscan | 0.989 | 0.969 | 0.971 | 0.991 | 0.952 | 27.2 | 47.2 | 0.23 |
| S2_dense | hdbscan | 0.995 | 0.981 | 0.982 | 0.99 | 0.974 | 27.2 | 30.2 | 0.146 |
| S2_dense | learned | 0.978 | 0.966 | 0.967 | 0.964 | 0.971 | 27.2 | 20.6 | 0.908 |
| S7_colocated | dbscan | 0.8 | 0.843 | 0.844 | 0.979 | 0.759 | 15.1 | 42 | 0.691 |
| S7_colocated | hdbscan | 0.808 | 0.852 | 0.853 | 0.993 | 0.763 | 15.1 | 29.4 | 0.368 |
| S7_colocated | learned | 0.885 | 0.916 | 0.916 | 0.91 | 0.935 | 15.1 | 16.4 | 3.015 |

## Turing Synthetic Radar Dataset Benchmark (test_scan)

Evaluated on 10 pulse train recordings (subset of 5000 pulses each).

| method | ari | ami | v_measure | homogeneity | completeness | n_true | n_pred | noise_frac | sec |
|---|---|---|---|---|---|---|---|---|---|
| dbscan | 0.428 | 0.586 | 0.592 | 0.983 | 0.464 | 12.5 | 41.8 | 0.011 | 0.087 |
| hdbscan | 0.428 | 0.59 | 0.595 | 0.984 | 0.468 | 12.5 | 33.8 | 0.007 | 0.071 |
| learned_sim | 0.44 | 0.618 | 0.623 | 0.929 | 0.536 | 12.5 | 20.6 | 0.014 | 0.445 |
| learned_turing | 0.461 | 0.591 | 0.595 | 0.955 | 0.48 | 12.5 | 25 | 0.02 | 0.404 |

