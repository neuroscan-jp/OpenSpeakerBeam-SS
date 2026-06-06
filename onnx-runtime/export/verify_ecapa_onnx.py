"""Verify ECAPA embedding ONNX vs PyTorch encode_batch."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import load_ecapa_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        type=Path,
        default=ROOT / "onnx-runtime/models/ecapa_embedding.onnx",
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    device = torch.device("cpu")
    clf = load_ecapa_model(device)
    wav = torch.randn(1, int(16_000 * args.seconds))
    with torch.no_grad():
        ref = clf.encode_batch(wav).squeeze().numpy()
        feats = clf.mods["compute_features"](wav).numpy()

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"features": feats})[0].squeeze()
    diff = np.abs(ref - out)
    print(f"max_abs_diff={diff.max():.6e}, mean={diff.mean():.6e}")
    if diff.max() > 1e-3:
        raise SystemExit("ECAPA ONNX parity check failed")


if __name__ == "__main__":
    main()
