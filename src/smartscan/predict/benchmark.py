"""Behaviour-prediction benchmark (the PDF's "percentage of correct predictions").

    python -m smartscan.predict.benchmark

For every MFR in held-out scenario seeds, each predictor forecasts the next
radar word from the words seen so far. Reported:
  next_word_acc   top-1 accuracy of the next symbol
  next_mode_acc   accuracy of the implied next *mode* (symbol → mode map fitted on training data)
  transition_acc  top-1 next-mode accuracy only at steps where the true mode changes
  change_auc      ROC-AUC of P(mode changes at the next word) against the true changes: does
                  the model know *when* a switch is coming (dwell-time awareness)?
  where_next_acc  at true changes, is the most likely *other* mode the right one? (does the
                  model know the mode grammar, e.g. search → acquisition → track?)
  nll             mean negative log-likelihood of the next symbol [nats]
Two observation regimes: every illumination seen, and 30 % of them missed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartscan.predict.models import NGram, Persistence, SpectralPSR, Unigram
from smartscan.predict.symbols import N_SYMBOLS, emitter_sequences
from smartscan.sim.world import ROOT


def symbol_to_mode(train) -> np.ndarray:
    c = np.zeros((N_SYMBOLS, 8))
    for sym, modes, _ in train:
        np.add.at(c, (sym, modes), 1)
    return c.argmax(1)


def score(pred, seqs, s2m) -> dict:
    from sklearn.metrics import roc_auc_score

    n_modes = int(s2m.max()) + 1
    M = np.zeros((N_SYMBOLS, n_modes))
    M[np.arange(N_SYMBOLS), s2m] = 1.0
    acc, macc, tacc, wacc, tn, nll, n = 0, 0, 0, 0, 0, 0.0, 0
    pchg, ychg = [], []
    for sym, modes, _ in seqs:
        P = pred.predict_all(sym)
        y, ym = sym[1:], modes[1:]
        cur_obs = s2m[sym[:-1]]  # mode implied by the last observed word (no labels used)
        top = P.argmax(1)
        acc += (top == y).sum()
        mhat = s2m[top]
        macc += (mhat == ym).sum()
        PM = P @ M  # (T, modes)
        p_stay = PM[np.arange(len(y)), cur_obs]
        pchg.append(1.0 - p_stay)
        ychg.append(ym != cur_obs)
        tr = ym != cur_obs
        tacc += (mhat[tr] == ym[tr]).sum()
        other = PM[tr].copy()
        other[np.arange(tr.sum()), cur_obs[tr]] = -1
        wacc += (other.argmax(1) == ym[tr]).sum()
        tn += tr.sum()
        nll += -np.log(np.maximum(P[np.arange(len(y)), y], 1e-9)).sum()
        n += len(y)
    pchg, ychg = np.concatenate(pchg), np.concatenate(ychg)
    auc = roc_auc_score(ychg, pchg) if 0 < ychg.sum() < len(ychg) else float("nan")
    return dict(next_word_acc=acc / n, next_mode_acc=macc / n, transition_acc=tacc / max(tn, 1),
                change_auc=auc, where_next_acc=wacc / max(tn, 1), nll=nll / n, steps=n, transitions=tn)


def main():
    rows = []
    for drop in (0.0, 0.3):
        train = emitter_sequences("S3_mfr", seeds=range(0, 150), drop=drop, rng_seed=1)
        test = emitter_sequences("S3_mfr", seeds=range(1000, 1040), drop=drop, rng_seed=2)
        seqs = [s for s, _, _ in train]
        s2m = symbol_to_mode(train)
        preds = [Unigram(), Persistence(), NGram(1), NGram(3), NGram(6), SpectralPSR(8, 2), SpectralPSR(16, 2),
                 SpectralPSR(24, 3)]
        try:
            from smartscan.predict.gru import GRUPredictor

            preds.append(GRUPredictor())
        except ImportError:
            pass
        for p in preds:
            p.fit(seqs)
            rows.append(dict(observed="all" if drop == 0 else f"{int(100 * (1 - drop))}%", model=p.name,
                             **score(p, test, s2m)))
            print(rows[-1])
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "predict"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    (out / "summary.md").write_text("# MFR behaviour prediction (held-out seeds)\n\n"
                                    + df.round(3).to_markdown(index=False) + "\n")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
