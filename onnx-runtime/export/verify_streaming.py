"""Verify chunked streaming inference vs full forward pass."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPORT.parent))

from model import SpeakerBeamSS  # noqa: E402
from model.streaming import SpeakerBeamSSStream  # noqa: E402
from export.extract_embedding import extract_enrollment_embedding, load_mono_16k  # noqa: E402
from tools import load_ecapa_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", type=Path, default=ROOT / "data/sample/mixture_000001.wav")
    parser.add_argument("--enrollment", type=Path, default=ROOT / "data/sample/enrollment_000001.wav")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/scratch_v2_lowsir/best_model.pth")
    parser.add_argument("--hop_ms", type=float, default=100.0)
    parser.add_argument(
        "--compare",
        choices=("cgln", "gln"),
        default="cgln",
        help="cgln: stream vs cgLN full (should match). gln: stream cgLN vs gLN batch.",
    )
    args = parser.parse_args()

    hop = int(16000 * args.hop_ms / 1000.0)
    mixture = load_mono_16k(args.mixture)
    enrollment = load_mono_16k(args.enrollment).to(mixture.device)

    encoder = load_ecapa_model(torch.device("cpu"))
    emb = extract_enrollment_embedding(enrollment, encoder)
    emb_t = torch.from_numpy(emb).unsqueeze(0)

    model = SpeakerBeamSS()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    if args.compare == "gln":
        with torch.no_grad():
            reference = model(mixture, emb_t).numpy().squeeze()
    else:
        from model.streaming import swap_gln_to_cgln  # noqa: E402

        swap_gln_to_cgln(model)
        with torch.no_grad():
            reference = model(mixture, emb_t).numpy().squeeze()

    stream = SpeakerBeamSSStream(
        model, hop_samples=hop, use_cgln=(args.compare == "gln")
    )
    stream.set_embedding(emb_t)
    stream.reset(batch_size=1)
    parts = []
    T = mixture.shape[-1]
    for start in range(0, T, hop):
        chunk = mixture[..., start : start + hop]
        out = stream.push(chunk)
        parts.append(out.numpy().squeeze())
    tail = stream.flush().numpy().squeeze()
    if tail.size:
        parts.append(tail)

    chunked = np.concatenate([p for p in parts if p.size]) if parts else np.array([])
    n = min(len(reference), len(chunked))
    diff = np.abs(reference[:n] - chunked[:n])
    print(f"hop={hop} samples, compare={args.compare}")
    print(f"lengths ref={len(reference)} stream={len(chunked)}")
    print(f"max_abs_diff={diff.max():.6e}, mean={diff.mean():.6e}")
    threshold = 1e-3 if args.compare == "cgln" else 0.35
    if diff.max() > threshold:
        raise SystemExit("streaming diverged from reference")


if __name__ == "__main__":
    main()
