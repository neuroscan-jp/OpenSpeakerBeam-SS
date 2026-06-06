"""
Extract ECAPA speaker embedding from enrollment audio (matches inference.py).

Usage:
  python extract_embedding.py --enrollment data/sample/enrollment_000001.wav \\
      --output embedding_000001.npy
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

from tools import get_speaker_embeddings_batch, load_ecapa_model  # noqa: E402


def load_mono_16k(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != sample_rate:
        raise RuntimeError(f"Expected {sample_rate} Hz, got {sr} Hz: {path}")
    mono = wav.mean(axis=1)
    return torch.from_numpy(mono).float().unsqueeze(0).unsqueeze(0)


def extract_enrollment_embedding(
    enrollment: torch.Tensor,
    speaker_encoder,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Multi-segment mean embedding (same logic as inference.py)."""
    segment_samples = 5 * sample_rate
    T = enrollment.shape[-1]
    embeddings_list = []
    if T <= segment_samples:
        emb = get_speaker_embeddings_batch(speaker_encoder, enrollment)
        embeddings_list.append(emb)
    else:
        step = max(1, (T - segment_samples) // 3)
        starts = list(range(0, T - segment_samples + 1, step))[:4]
        for s in starts:
            seg = enrollment[..., s : s + segment_samples]
            emb = get_speaker_embeddings_batch(speaker_encoder, seg)
            embeddings_list.append(emb)
    stacked = torch.stack(embeddings_list, dim=0).mean(dim=0)
    emb = stacked.squeeze(0).cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(emb) + 1e-8
    return (emb / norm).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract ECAPA embedding to .npy")
    parser.add_argument("--enrollment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enrollment = load_mono_16k(args.enrollment, args.sample_rate).to(device)
    encoder = load_ecapa_model(device)
    embedding = extract_enrollment_embedding(enrollment, encoder, args.sample_rate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embedding)
    print(f"Saved embedding shape={embedding.shape} -> {args.output}")


if __name__ == "__main__":
    main()
