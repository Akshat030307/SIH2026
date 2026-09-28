"""Edge packaging: ONNX export, INT8 quantization, latency and fidelity benchmark (PDF §Deployment).

    python -m smartscan.export.edge

Writes checkpoints/d3qn_fp32.onnx and d3qn_int8.onnx and results/edge/summary.md:
  * per-decision latency (p50 / p99) for PyTorch CPU/GPU and ONNX Runtime FP32/INT8,
    measured on the policy network alone and on the full decision loop
    (feature extraction + network),
  * FP32 vs INT8 action agreement on states recorded from real episodes,
  * scheduling metrics of the INT8 policy vs FP32 on held-out scenarios.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from smartscan.rl.d3qn import DuelingQNet
from smartscan.rl.perception import F_CH, F_GLOBAL
from smartscan.rl.policy import DEFAULT_CKPT, D3QNScheduler
from smartscan.sim.world import ROOT, build_world
from smartscan.util import md_table

OUT = ROOT / "checkpoints"


def export_onnx(ckpt=DEFAULT_CKPT):
    ck = torch.load(ckpt, map_location="cpu")
    net = DuelingQNet(ck["K"], ck["D"]).eval()
    net.load_state_dict(ck["q"])
    obs_dim = net.K * F_CH + F_GLOBAL
    fp32 = OUT / "d3qn_fp32.onnx"
    torch.onnx.export(net, torch.zeros(1, obs_dim), str(fp32), input_names=["obs"], output_names=["q"],
                      dynamic_axes={"obs": {0: "batch"}, "q": {0: "batch"}}, opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = OUT / "d3qn_int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    return net, fp32, int8


def _lat(fn, x, n=2000):
    for _ in range(50):
        fn(x)
    ts = np.empty(n)
    for i in range(n):
        t = time.perf_counter()
        fn(x)
        ts[i] = time.perf_counter() - t
    return float(np.percentile(ts, 50) * 1e6), float(np.percentile(ts, 99) * 1e6)


def record_states(n_eps=3):
    from smartscan.runner import run_episode

    states = []
    sched = D3QNScheduler()
    orig = sched.features

    def spy(t, ch):
        o = orig(t, ch)
        states.append(o)
        return o

    sched.features = spy
    for s, sc in enumerate(["S2_dense", "S3_mfr", "S5_popup"][:n_eps]):
        run_episode(build_world(sc, seed=200 + s), sched, seed=200 + s)
    return np.stack(states).astype(np.float32)


def main():
    import onnxruntime as ort
    import pandas as pd

    from smartscan.runner import run_episode

    net, fp32, int8 = export_onnx()
    states = record_states()
    x1 = states[:1]
    rows = []
    with torch.no_grad():
        tx = torch.from_numpy(x1)
        rows.append(("PyTorch CPU", *_lat(lambda x: net(x), tx)))
        if torch.cuda.is_available():
            gnet = net.cuda()
            gx = tx.cuda()

            def gpu(x):
                gnet(x)
                torch.cuda.synchronize()

            rows.append(("PyTorch GPU", *_lat(gpu, gx)))
            net = net.cpu()
    sess = {}
    for name, path in [("ONNX Runtime FP32", fp32), ("ONNX Runtime INT8", int8)]:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        sess[name] = s
        rows.append((name + " (1 thread)", *_lat(lambda x, s=s: s.run(None, {"obs": x}), x1)))
    # fidelity
    q32 = sess["ONNX Runtime FP32"].run(None, {"obs": states})[0]
    q8 = sess["ONNX Runtime INT8"].run(None, {"obs": states})[0]
    agree = float((q32.argmax(1) == q8.argmax(1)).mean())
    # full decision loop latency (features + INT8 network)
    sched = D3QNScheduler(onnx=int8)
    world = build_world("S2_dense", seed=300)
    loop = []
    orig_act = sched.act

    def timed(t, ch):
        t0 = time.perf_counter()
        a = orig_act(t, ch)
        loop.append(time.perf_counter() - t0)
        return a

    sched.act = timed
    run_episode(world, sched, seed=300)
    loop = np.array(loop) * 1e6
    rows.append(("Full decision (features + INT8), S2_dense", float(np.percentile(loop, 50)),
                 float(np.percentile(loop, 99))))
    lat = pd.DataFrame(rows, columns=["path", "p50_us", "p99_us"]).round(1)
    # metrics FP32 vs INT8
    mrows = []
    for sc in ["S1_sparse", "S3_mfr", "S6_lockin"]:
        for name, kw in [("fp32", {"onnx": fp32}), ("int8", {"onnx": int8})]:
            m, _, _ = run_episode(build_world(sc, seed=150), D3QNScheduler(**kw), seed=150)
            mrows.append(dict(scenario=sc, model=name, pd=m["pd"], pd_weighted=m["pd_weighted"]))
    met = pd.DataFrame(mrows).round(3)
    sizes = {p.name: p.stat().st_size / 1024 for p in [fp32, int8]}
    out = ROOT / "results" / "edge"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Edge inference benchmark", "", "Model sizes: " + ", ".join(f"{k} {v:.0f} KiB" for k, v in sizes.items()),
          "", f"FP32 vs INT8 greedy-action agreement on {len(states)} recorded states: **{agree:.3f}**", "",
          "## Latency per decision (µs)", "", md_table(lat, index=False), "",
          "## Scheduling metrics, FP32 vs INT8 (seed 150)", "", md_table(met, index=False), ""]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
