"""
End-to-end parity: PyTorch inference.py path vs ONNX on real sample audio.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPORT_DIR))
sys.path.insert(0, str(ROOT))

from model import SpeakerBeamSS  # noqa: E402
from onnx_utils import patch_s4d_for_onnx, restore_s4d  # noqa: E402
from extract_embedding import extract_enrollment_embedding, load_mono_16k  # noqa: E402
from tools import load_ecapa_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", type=Path, default=ROOT / "data/sample/mixture_000001.wav")
    parser.add_argument("--enrollment", type=Path, default=ROOT / "data/sample/enrollment_000001.wav")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/scratch_v2_lowsir/best_model.pth")
    parser.add_argument("--onnx", type=Path, default=ROOT / "onnx-runtime/models/speakerbeam_ep110.onnx")
    args = parser.parse_args()

    mixture = load_mono_16k(args.mixture)
    T = mixture.shape[-1]
    device = torch.device("cpu")
    encoder = load_ecapa_model(device)
    enrollment = load_mono_16k(args.enrollment).to(device)
    emb = extract_enrollment_embedding(enrollment, encoder)
    emb_t = torch.from_numpy(emb).unsqueeze(0)

    model = SpeakerBeamSS()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    backups = patch_s4d_for_onnx(model)
    try:
        with torch.no_grad():
            pt_out = model(mixture, emb_t).numpy()
    finally:
        restore_s4d(backups)

    mix_np = mixture.numpy().astype(np.float32)

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    ort_out = sess.run(
        None,
        {"mixture": mix_np, "spk_embedding": emb.astype(np.float32).reshape(1, -1)},
    )[0]

    n = min(pt_out.shape[-1], ort_out.shape[-1])
    diff = np.abs(pt_out[..., :n] - ort_out[..., :n])
    print(f"max_abs_diff={diff.max():.6e}")
    print(f"pt rms={np.sqrt((pt_out**2).mean()):.4f} ort rms={np.sqrt((ort_out**2).mean()):.4f}")


if __name__ == "__main__":
    main()
