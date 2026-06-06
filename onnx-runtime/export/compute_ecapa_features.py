"""
Compute SpeechBrain-compatible log-mel features for ECAPA embedding ONNX.

Used by Rust when --embedding-backend onnx (features in Python, embedding in ONNX).

Usage:
  python compute_ecapa_features.py --wav enrollment.wav --output features.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import load_ecapa_model  # noqa: E402


def load_mono_16k(path: Path, sample_rate: int = 16_000) -> torch.Tensor:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != sample_rate:
        raise RuntimeError(f"Expected {sample_rate} Hz, got {sr}: {path}")
    mono = wav.mean(axis=1)
    return torch.from_numpy(mono).float().unsqueeze(0)


def compute_features(waveform: torch.Tensor, classifier) -> np.ndarray:
    """Return float32 array shape (T, 80)."""
    with torch.no_grad():
        feats = classifier.mods["compute_features"](waveform)
    return feats.squeeze(0).cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample_rate", type=int, default=16_000)
    args = parser.parse_args()

    device = torch.device("cpu")
    waveform = load_mono_16k(args.wav, args.sample_rate)
    classifier = load_ecapa_model(device)
    features = compute_features(waveform, classifier)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, features)
    print(f"Saved features shape={features.shape} -> {args.output}")


if __name__ == "__main__":
    main()
