"""Classical deinterleavers.

* ``dbscan_deinterleave``: density clustering on scaled (RF, log PW, AOA),
  the classical baseline.
* ``hdbscan_deinterleave``: the same with HDBSCAN (no global ε, copes with
  mixed densities).
* ``sdif_split``: sequential difference histogram (SDIF). Splits one cluster
  into several emitters that share RF/PW/AOA but differ in PRI.
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm
from sklearn.cluster import DBSCAN


def _scaled(pdws: np.ndarray, rf_mhz: float = 8.0, pw_log: float = 0.08, aoa_deg: float = 4.0) -> np.ndarray:
    """Scale features so that one unit ≈ typical within-emitter spread."""
    a = np.deg2rad(pdws["aoa"])
    r = aoa_deg / 57.2958
    return np.stack([
        pdws["rf"] / (rf_mhz * 1e6),
        np.log(np.maximum(pdws["pw"], 1e-9)) / pw_log,
        np.sin(a) / r,
        np.cos(a) / r,
    ], axis=1)


def dbscan_deinterleave(pdws: np.ndarray, eps: float = 1.5, min_samples: int = 5, **scale) -> np.ndarray:
    if len(pdws) == 0:
        return np.zeros(0, dtype=int)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(_scaled(pdws, **scale))


def hdbscan_deinterleave(pdws: np.ndarray, min_cluster_size: int = 15, min_samples: int = 5,
                         core_dist_n_jobs: int = -1, **scale) -> np.ndarray:
    import hdbscan

    if len(pdws) < min_cluster_size:
        return np.zeros(len(pdws), dtype=int)
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                           core_dist_n_jobs=core_dist_n_jobs).fit_predict(_scaled(pdws, **scale))


def sdif_split(toa: np.ndarray, max_emitters: int = 4, bins: int = 400, thresh: float = 0.35) -> np.ndarray:
    """Split a pulse set by PRI with a sequential-difference histogram.

    For difference level c = 1..4, histogram TOA_{i+c} − TOA_i. A PRI shows up
    as a sharp peak. For each detected PRI, extract the pulse chain that
    follows it, remove it, and repeat.
    """
    n = len(toa)
    labels = np.full(n, -1)
    order = np.argsort(toa)
    t = toa[order]
    remaining = np.ones(n, dtype=bool)
    lab = 0
    for _ in range(max_emitters):
        idx = np.flatnonzero(remaining)
        if len(idx) < 6:
            break
        ts = t[idx]
        found = None
        for c in range(1, 5):
            d = ts[c:] - ts[:-c]
            if len(d) < 5:
                break
            hist, edges = np.histogram(d, bins=bins, range=(0, np.percentile(d, 95) * 1.2 + 1e-12))
            k = int(np.argmax(hist))
            if hist[k] > thresh * (len(ts) - c):
                found = 0.5 * (edges[k] + edges[k + 1])
                tol = (edges[1] - edges[0]) * 1.5
                break
        if found is None:
            break
        # chain extraction: greedily follow pulses separated by ≈ found
        chain = np.zeros(len(ts), dtype=bool)
        for start in range(len(ts)):
            if chain[start]:
                continue
            j, cur = start, [start]
            while True:
                nxt = np.searchsorted(ts, ts[j] + found - tol)
                if nxt >= len(ts) or ts[nxt] > ts[j] + found + tol:
                    break
                cur.append(nxt)
                j = nxt
            if len(cur) >= 4:
                chain[cur] = True
        if chain.sum() < 4:
            break
        labels[order[idx[chain]]] = lab
        remaining[idx[chain]] = False
        lab += 1
    if remaining.any():
        labels[order[remaining]] = lab
    return labels


def clustering_scores(true: np.ndarray, pred: np.ndarray) -> dict:
    """Turing Deinterleaving Challenge metrics (clustering subset)."""
    h, c, v = skm.homogeneity_completeness_v_measure(true, pred)
    return dict(
        ari=float(skm.adjusted_rand_score(true, pred)),
        ami=float(skm.adjusted_mutual_info_score(true, pred)),
        v_measure=float(v), homogeneity=float(h), completeness=float(c),
        n_true=len(np.unique(true)), n_pred=len(np.unique(pred[pred >= 0])),
        noise_frac=float((pred < 0).mean()),
    )
