import numpy as np

from smartscan.predict.models import NGram, SpectralPSR
from smartscan.predict.symbols import N_SYMBOLS, symbolize


def _cyclic_sequences(n=40, length=60, seed=0):
    """A deterministic grammar 0→1→2→0 with random dwell lengths: learnable transitions."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s, cur = [], int(rng.integers(3))
        while len(s) < length:
            s += [cur] * int(rng.integers(2, 6))
            cur = (cur + 1) % 3
        out.append(np.array(s[:length]))
    return out


def test_symbolize_ranges():
    assert symbolize(100e-6, 1e-6) == 0
    assert symbolize(2000e-6, 20e-6) == N_SYMBOLS - 1


def test_ngram_and_psr_learn_the_grammar():
    train, test = _cyclic_sequences(seed=0), _cyclic_sequences(n=5, seed=1)
    for model in (NGram(3), SpectralPSR(rank=4, h_len=2)):
        model.fit(train)
        for seq in test:
            P = model.predict_all(seq)
            assert P.shape == (len(seq) - 1, N_SYMBOLS)
            assert np.allclose(P.sum(1), 1.0)
            # probability mass on the grammatical successor or staying put
            t = np.arange(len(seq) - 1)
            ok = P[t, seq[:-1]] + P[t, (seq[:-1] + 1) % 3]
            assert ok.mean() > 0.9, model.name
