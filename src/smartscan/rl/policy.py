"""Trained D3QN policy wrapped as a Scheduler (for the evaluator, API and dashboard)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from smartscan.rl.d3qn import DuelingQNet
from smartscan.rl.perception import Perception
from smartscan.sim.world import ROOT

DEFAULT_CKPT = ROOT / "checkpoints" / "d3qn_best.pt"


class D3QNScheduler(Perception):
    name = "d3qn"

    def __init__(self, ckpt: str | Path | None = None, device: str | None = None, onnx: str | Path | None = None,
                 net: DuelingQNet | None = None, behaviour="default", **kw):
        if behaviour == "default":
            from smartscan.rl.behaviour import load_behaviour

            behaviour = load_behaviour()
        super().__init__(behaviour=behaviour, **kw)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._ort = None
        if net is not None:
            self.net = net
            self.device = next(net.parameters()).device
        elif onnx is not None:
            import onnxruntime as ort

            self._ort = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            self._in = self._ort.get_inputs()[0].name
        else:
            ck = torch.load(ckpt or DEFAULT_CKPT, map_location=self.device)
            self.net = DuelingQNet(ck["K"], ck["D"]).to(self.device).eval()
            self.net.load_state_dict(ck["q"])

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        if self._ort is not None:
            return self._ort.run(None, {self._in: obs[None].astype(np.float32)})[0][0]
        x = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)
        return self.net(x)[0].cpu().numpy()

    def act(self, t, channel):
        self._steps += 1
        obs = self.features(t, channel)
        a = int(np.argmax(self.q_values(obs)))
        ch, d = self.rx.decode_action(a)
        return self.commit_action(t, channel, ch, d)
