"""
Offline batch ONNX inference — matches PyTorch SpeakerBeamSS batch quality (GLN path).

Streaming ONNX (encoder/decoder/separator_cgln) uses causal cgLN and is lower quality.
Use this script or speakerbeam-cli without --stream for batch parity.

Example:
  python run_offline_onnx_infer.py \\
    --onnx ../models/jtube10k_ep5/speakerbeam_batch.onnx \\
    --mixture path/to/mixture.wav \\
    --embedding-npy path/to/enrollment.npy \\
    --output separated.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

EXPORT_DIR = Path(__file__).resolve().parent
ROOT = EXPORT_DIR.parents[1]
sys.path.insert(0, str(EXPORT_DIR))
sys.path.insert(0, str(ROOT))

from extract_embedding import extract_enrollment_embedding, load_mono_16k  # noqa: E402
from tools import load_ecapa_model  # noqa: E402


def onnx_fixed_samples(session: ort.InferenceSession) -> int:
    shape = session.get_inputs()[0].shape
    if len(shape) != 3 or shape[2] in (None, "time", "T"):
        raise ValueError(f"ONNX mixture input must be fixed [1,1,T], got {shape}")
    return int(shape[2])


def pad_mixture(mixture: np.ndarray, fixed_samples: int) -> np.ndarray:
    if mixture.shape[-1] > fixed_samples:
        raise ValueError(
            f"mixture length {mixture.shape[-1]} exceeds ONNX trace length {fixed_samples}; "
            "re-export with a larger --t_seconds"
        )
    out = np.zeros(fixed_samples, dtype=np.float32)
    out[: mixture.shape[-1]] = mixture
    return out.reshape(1, 1, -1)


def load_embedding(args: argparse.Namespace) -> np.ndarray:
    if args.embedding_npy is not None:
        emb = np.load(args.embedding_npy).astype(np.float32)
    else:
        if args.enrollment is None:
            raise ValueError("pass --embedding-npy or --enrollment")
        device = __import__("torch").device("cpu")
        encoder = load_ecapa_model(device)
        enrollment = load_mono_16k(args.enrollment)
        emb = extract_enrollment_embedding(enrollment, encoder)
    emb = emb.reshape(-1)
    if emb.shape[0] != 192:
        raise ValueError(f"expected 192-d embedding, got {emb.shape}")
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline batch SpeakerBeam ONNX inference")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument("--embedding-npy", type=Path, default=None)
    parser.add_argument("--enrollment", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mixture = load_mono_16k(args.mixture).numpy().astype(np.float32)
    orig_len = mixture.shape[0]
    emb = load_embedding(args)

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    fixed = onnx_fixed_samples(sess)
    mix_in = pad_mixture(mixture, fixed)
    out = sess.run(
        None,
        {"mixture": mix_in, "spk_embedding": emb.reshape(1, -1)},
    )[0].squeeze()
    out = out[:orig_len]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, out, 16000)
    rms = float(np.sqrt(np.mean(out**2)))
    print(f"saved: {args.output}")
    print(f"orig_len={orig_len} ({orig_len / 16000:.2f}s) onnx_fixed={fixed}")
    print(f"out_rms={rms:.4f}")


if __name__ == "__main__":
    main()
