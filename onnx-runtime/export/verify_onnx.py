"""
Verify ONNX export against PyTorch (ep110).

Usage:
  python verify_onnx.py --onnx onnx-runtime/models/speakerbeam_ep110.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

EXPORT_DIR = Path(__file__).resolve().parent
ROOT = EXPORT_DIR.parents[1]
sys.path.insert(0, str(EXPORT_DIR))
sys.path.insert(0, str(ROOT))

from model import SpeakerBeamSS  # noqa: E402
from onnx_utils import patch_s4d_for_onnx, restore_s4d  # noqa: E402


class Wrapper(nn.Module):
    def __init__(self, model: SpeakerBeamSS):
        super().__init__()
        self.model = model

    def forward(self, mixture: torch.Tensor, spk_embedding: torch.Tensor) -> torch.Tensor:
        return self.model(mixture, spk_embedding)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/scratch_v2_lowsir/best_model.pth")
    parser.add_argument("--onnx", type=Path, default=ROOT / "onnx-runtime/models/speakerbeam_ep110.onnx")
    parser.add_argument("--t_seconds", type=float, default=10.0)
    args = parser.parse_args()

    T = int(16000 * args.t_seconds)
    mixture = torch.randn(1, 1, T)
    emb = torch.randn(1, 192)
    emb = torch.nn.functional.normalize(emb, dim=-1)

    model = SpeakerBeamSS()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    backups = patch_s4d_for_onnx(model)
    try:
        with torch.no_grad():
            pt_out = Wrapper(model)(mixture, emb).numpy()
    finally:
        restore_s4d(backups)

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    ort_out = sess.run(
        None,
        {
            "mixture": mixture.numpy().astype(np.float32),
            "spk_embedding": emb.numpy().astype(np.float32),
        },
    )[0]

    diff = np.abs(pt_out - ort_out)
    print(f"shape pt={pt_out.shape} ort={ort_out.shape}")
    print(f"max_abs_diff={diff.max():.6e}")
    print(f"mean_abs_diff={diff.mean():.6e}")
    if diff.max() > 1e-3:
        raise SystemExit("FAIL: max diff > 1e-3")
    print("OK: ONNX matches PyTorch within tolerance")


if __name__ == "__main__":
    main()
