"""Next-radar-word predictors behind one interface.

    fit(sequences)            learn from unlabelled symbol sequences
    predict_proba(history)    distribution over the next symbol, shape (N_SYMBOLS,)

Implementations:
  * ``NGram``: variable-order Markov model with interpolated back-off
    (quick baseline; its memory is limited to the order).
  * ``SpectralPSR``: transformed predictive state representation learned by
    SVD of empirical Hankel matrices (Boots, Siddiqi & Gordon 2010). The state
    is a k-dimensional linear projection of the predictions of future tests,
    i.e. the PDF's ⟨𝒪, ℋ, ℰ, Q⟩ with core tests found automatically by the
    SVD. It needs no prior knowledge of the number or meaning of modes.
  * ``GRUPredictor``: neural sequence model, in ``gru.py``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from smartscan.predict.symbols import N_SYMBOLS


class Predictor:
    name = "base"

    def fit(self, seqs: list[np.ndarray]) -> "Predictor":
        return self

    def predict_proba(self, history: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_all(self, seq: np.ndarray) -> np.ndarray:
        """(len(seq)-1, N) predictions of seq[t+1] from seq[:t+1]."""
        return np.stack([self.predict_proba(seq[: t + 1]) for t in range(len(seq) - 1)])


class Unigram(Predictor):
    name = "unigram"

    def fit(self, seqs):
        c = np.bincount(np.concatenate(seqs), minlength=N_SYMBOLS) + 1.0
        self.p = c / c.sum()
        return self

    def predict_proba(self, history):
        return self.p


class Persistence(Predictor):
    """'Next word = last word': the natural baseline for slowly switching modes."""

    name = "persistence"

    def predict_proba(self, history):
        p = np.full(N_SYMBOLS, 1e-3)
        p[history[-1]] = 1.0
        return p / p.sum()


class NGram(Predictor):
    def __init__(self, order: int = 3, alpha: float = 0.1):
        self.order, self.alpha = order, alpha
        self.name = f"ngram{order}"

    def fit(self, seqs):
        self.counts = [defaultdict(lambda: np.zeros(N_SYMBOLS)) for _ in range(self.order + 1)]
        for s in seqs:
            for t in range(len(s)):
                for k in range(self.order + 1):
                    if t - k < 0:
                        break
                    ctx = tuple(s[t - k:t])
                    self.counts[k][ctx][s[t]] += 1
        return self

    def predict_proba(self, history):
        p = np.full(N_SYMBOLS, 1.0 / N_SYMBOLS)
        for k in range(self.order + 1):  # interpolate from short to long contexts
            if k > len(history):
                break
            ctx = tuple(history[len(history) - k:]) if k else ()
            c = self.counts[k].get(ctx)
            if c is None or c.sum() == 0:
                continue
            n = c.sum()
            lam = n / (n + 5.0)
            p = lam * (c + self.alpha * p) / (n + self.alpha) + (1 - lam) * p
        return p / p.sum()


class SpectralPSR(Predictor):
    """Transformed PSR learned with the spectral algorithm.

    Histories are indicative events of length ≤ ``h_len`` (the last symbols).
    Tests are the next symbol, and the next-plus-one symbol for the observable
    operators. With
        P_H[h]          = Pr[history h]
        P_TH[τ, h]      = Pr[next = τ, history h]
        P_TxH[x][τ, h]  = Pr[next = x, next+1 = τ, history h]
    and U the top-k left singular vectors of P_TH:
        B_x   = Uᵀ P_TxH[x] (Uᵀ P_TH)⁺
        b_∞ᵀ  = P_Hᵀ (Uᵀ P_TH)⁺
        b_h   = Uᵀ P_TH[:, h] / P_H[h]        (state straight from a history)
    Filtering: b ← B_x b / (b_∞ᵀ B_x b), and Pr[next = x | b] = b_∞ᵀ B_x b.
    """

    def __init__(self, rank: int = 10, h_len: int = 2):
        self.rank, self.h_len = rank, h_len
        self.name = f"psr{rank}"

    def _hidx(self, ctx) -> int:
        # index of an indicative event: all contexts of length exactly h_len (base-N number)
        v = 0
        for x in ctx:
            v = v * N_SYMBOLS + int(x)
        return v

    def fit(self, seqs):
        N, L = N_SYMBOLS, self.h_len
        nh = N**L
        PH = np.zeros(nh)
        PTH = np.zeros((N, nh))
        PTxH = np.zeros((N, N, nh))
        tot = 0
        for s in seqs:
            for t in range(L - 1, len(s) - 2):
                h = self._hidx(s[t - L + 1:t + 1])
                PH[h] += 1
                PTH[s[t + 1], h] += 1
                PTxH[s[t + 1], s[t + 2], h] += 1
                tot += 1
        keep = PH > 0
        self._keep = np.flatnonzero(keep)
        self._col = {h: i for i, h in enumerate(self._keep)}
        PH, PTH, PTxH = PH[keep] / tot, PTH[:, keep] / tot, PTxH[:, :, keep] / tot
        U, S, _ = np.linalg.svd(PTH, full_matrices=False)
        k = min(self.rank, int((S > 1e-8).sum()))
        self.U = U[:, :k]
        UP = self.U.T @ PTH
        pinv = np.linalg.pinv(UP, rcond=1e-6)
        self.B = np.stack([self.U.T @ PTxH[x] @ pinv for x in range(N)])
        self.binf = PH @ pinv
        self.PTH, self.PH = PTH, PH
        self.bstar = UP.sum(axis=1) / PH.sum()
        return self

    def state_from_history(self, ctx) -> np.ndarray:
        h = self._hidx(ctx)
        i = self._col.get(h)
        if i is None:
            return self.bstar.copy()
        return self.U.T @ self.PTH[:, i] / self.PH[i]

    def _probs(self, b):
        p = np.einsum("k,xkj,j->x", self.binf, self.B, b)
        p = np.maximum(p, 1e-6)
        return p / p.sum()

    def filter(self, seq):
        """Yield the predictive state after each prefix (reset from a short history, then filter)."""
        L = self.h_len
        b = None
        for t in range(len(seq)):
            if t == L - 1:
                b = self.state_from_history(seq[t - L + 1:t + 1])
            elif b is not None:
                nb = self.B[seq[t]] @ b
                z = float(self.binf @ nb)
                b = nb / z if abs(z) > 1e-9 else self.state_from_history(seq[max(0, t - L + 1):t + 1])
            yield b

    def predict_proba(self, history):
        if len(history) < self.h_len:
            return self._probs(self.bstar)
        b = None
        for b in self.filter(history):
            pass
        return self._probs(b)

    def predict_all(self, seq):
        out = []
        for t, b in enumerate(self.filter(seq[:-1])):
            out.append(self._probs(b if b is not None else self.bstar))
        return np.stack(out)
