"""GRU next-radar-word predictor."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from smartscan.predict.models import Predictor
from smartscan.predict.symbols import N_SYMBOLS


class _Net(nn.Module):
    def __init__(self, d: int = 64, layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(N_SYMBOLS, d)
        self.gru = nn.GRU(d, d, layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(d, N_SYMBOLS)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.head(h)


class GRUPredictor(Predictor):
    name = "gru"

    def __init__(self, epochs: int = 30, chunk: int = 64, lr: float = 3e-3, device: str | None = None):
        self.epochs, self.chunk, self.lr = epochs, chunk, lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, seqs):
        torch.manual_seed(0)
        self.net = _Net().to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # chop into fixed-length chunks for batching
        chunks = []
        for s in seqs:
            for i in range(0, max(len(s) - 1, 1), self.chunk):
                c = s[i:i + self.chunk + 1]
                if len(c) >= 4:
                    chunks.append(c)
        rng = np.random.default_rng(0)
        for _ in range(self.epochs):
            for b in np.array_split(rng.permutation(len(chunks)), max(len(chunks) // 64, 1)):
                L = max(len(chunks[i]) for i in b)
                X = np.zeros((len(b), L - 1), np.int64)
                Y = np.full((len(b), L - 1), -100, np.int64)
                for r, i in enumerate(b):
                    c = chunks[i]
                    X[r, :len(c) - 1], Y[r, :len(c) - 1] = c[:-1], c[1:]
                X, Y = torch.from_numpy(X).to(self.device), torch.from_numpy(Y).to(self.device)
                loss = nn.functional.cross_entropy(self.net(X).reshape(-1, N_SYMBOLS), Y.reshape(-1),
                                                   ignore_index=-100)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def predict_all(self, seq):
        x = torch.as_tensor(np.asarray(seq[:-1]), dtype=torch.long, device=self.device)[None]
        return torch.softmax(self.net(x)[0], -1).cpu().numpy()

    def predict_proba(self, history):
        return self.predict_all(np.append(history, 0))[-1]
