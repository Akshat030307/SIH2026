"""Turing Synthetic Radar Dataset (TSRD): subset download + loader.

The dataset is gated (auto-approved) on Hugging Face:
  1. accept the terms at https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset
  2. export HF_TOKEN=hf_...   (a read token from https://huggingface.co/settings/tokens)
  3. python -m smartscan.deinterleave.turing download --split train_scan --n 30
     python -m smartscan.deinterleave.turing download --split test_scan --n 10
     python -m smartscan.deinterleave.turing inspect

Only *scan-mode* files are fetched by default (~94k pulses per train, a few MB
each). Stare mode averages 1.3 M pulses per file.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from smartscan import pdw as P
from smartscan.sim.world import ROOT

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(ROOT / ".env")
    load_dotenv(Path.home() / ".env")
except ImportError:
    pass

REPO = "alan-turing-institute/turing-synthetic-radar-dataset"
DATA_DIR = ROOT / "data" / "turing"


def _req(url: str):
    headers = {"User-Agent": "smartscan"}
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def list_files(split: str, mode: str = "scan") -> list[dict]:
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{mode}/{split}"
    return [x for x in json.load(_req(url)) if x.get("type") == "file"]


def download(split: str = "train_scan", n: int = 30, mode: str = "scan", seed: int = 0, max_workers: int = 8) -> list[Path]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    files = list_files(split, mode)
    if n >= len(files):
        pick = list(range(len(files)))
    else:
        rng = np.random.default_rng(seed)
        pick = sorted(rng.choice(len(files), size=n, replace=False).tolist())

    out_dir = DATA_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [files[i] for i in pick]
    print(f"Downloading {len(targets)} files for {split} ({mode}) using {max_workers} threads...")

    def _fetch(f):
        dst = out_dir / Path(f["path"]).name
        if not dst.exists() or dst.stat().st_size == 0:
            url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{f['path']}"
            for attempt in range(4):
                try:
                    with _req(url) as r:
                        dst.write_bytes(r.read())
                    print(f"  + {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
                    break
                except Exception as e:
                    if attempt == 3:
                        print(f"  x Failed {dst.name}: {e}")
                        raise
                    time.sleep(1.0 * (attempt + 1))
        else:
            print(f"  = {dst.name} (cached)")
        return dst

    got = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch, f) for f in targets]
        for fut in as_completed(futures):
            try:
                got.append(fut.result())
            except Exception:
                pass
    return got


def inspect(path: Path) -> None:
    import h5py

    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: {obj.shape} {obj.dtype} attrs={dict(obj.attrs)}")
        else:
            print(f"  {name}/ attrs={dict(obj.attrs)}")

    with h5py.File(path, "r") as f:
        print(path.name, dict(f.attrs))
        f.visititems(show)


# Column order documented in the dataset card: ToA [µs], centre frequency [MHz],
# pulse width [µs], AoA [deg], amplitude [dB].
_FEATURE_KEYS = ("data", "pdws", "x", "features")
_LABEL_KEYS = ("labels", "y", "emitter", "emitter_id")


def load(path: Path) -> np.ndarray:
    """Load one TSRD file into our PDW structured array (units converted to s / Hz)."""
    import h5py

    with h5py.File(path, "r") as f:
        keys = {k.lower(): k for k in f}
        xk = next((keys[k] for k in _FEATURE_KEYS if k in keys), None)
        yk = next((keys[k] for k in _LABEL_KEYS if k in keys), None)
        if xk is None or yk is None:
            raise KeyError(f"unrecognised layout in {path.name}: {list(f.keys())} (run `inspect`)")
        x = np.asarray(f[xk])
        y = np.asarray(f[yk]).reshape(-1)
    out = P.empty(len(x))
    out["toa"] = x[:, 0] * 1e-6
    out["rf"] = x[:, 1] * 1e6
    out["pw"] = x[:, 2] * 1e-6
    out["aoa"] = x[:, 3] % 360.0
    out["amp"] = x[:, 4]
    out["emitter"] = y.astype(np.int32)
    out["mode"] = -1
    return out[np.argsort(out["toa"], kind="stable")]


def local_files(split: str) -> list[Path]:
    return sorted((DATA_DIR / split).glob("*.h5"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Turing Synthetic Radar Dataset tooling")
    ap.add_argument("cmd", choices=["download", "inspect", "eval", "train"])
    ap.add_argument("--split", default="test_scan")
    ap.add_argument("--mode", default="scan", choices=["scan", "stare", "archive"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-pulses", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args(argv)
    if a.cmd == "download":
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            raise SystemExit("HF_TOKEN is not set; see the module docstring")
        download(split=a.split, n=a.n, mode=a.mode, max_workers=a.workers)
    elif a.cmd == "inspect":
        files = local_files(a.split)
        if not files:
            raise SystemExit(f"no files in {DATA_DIR / a.split}")
        inspect(files[0])
    elif a.cmd == "eval":
        from smartscan.deinterleave.benchmark_turing import evaluate_tsrd
        evaluate_tsrd(split=a.split, max_pulses=a.max_pulses)
    elif a.cmd == "train":
        from smartscan.deinterleave.train_turing import train_turing
        train_turing(epochs=a.epochs)


if __name__ == "__main__":
    main()

