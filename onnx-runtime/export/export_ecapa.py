"""
Export ECAPA-TDNN embedding head to ONNX + SpeechBrain FBank weights for Rust.

Full waveform → ONNX is blocked by STFT/complex ops. Phase 2 exports:
  - ecapa_embedding.onnx  : log-mel features (B, T, 80) → embedding (B, 192)
  - ecapa_fbank.npz       : STFT window + fbank_matrix for Rust feature extraction

Usage:
  python onnx-runtime/export/export_ecapa.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

EXPORT_DIR = Path(__file__).resolve().parent
ROOT = EXPORT_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from tools import load_ecapa_model  # noqa: E402


class EcapaEmbeddingOnnx(nn.Module):
    """mean_var_norm + embedding_model (matches encode_batch without L2 post-norm)."""

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        wav_lens = torch.ones(features.shape[0], device=features.device)
        feats = self.classifier.mods["mean_var_norm"](features, wav_lens)
        emb = self.classifier.mods["embedding_model"](feats, wav_lens)
        return emb.squeeze(1)


def export_fbank_weights(classifier, output: Path) -> None:
    fb = classifier.mods["compute_features"]
    fbank = fb.compute_fbanks
    # Trigger lazy filter construction.
    _ = fb(torch.randn(1, 16_000))

    f_central_mat = fbank.f_central.repeat(
        fbank.all_freqs_mat.shape[1], 1
    ).transpose(0, 1)
    band_mat = fbank.band.repeat(fbank.all_freqs_mat.shape[1], 1).transpose(0, 1)
    fbank_matrix = fbank._create_fbank_matrix(f_central_mat, band_mat)

    np.savez(
        output,
        fbank_matrix=fbank_matrix.cpu().numpy().astype(np.float32),
        window=fb.compute_STFT.window.cpu().numpy().astype(np.float32),
        n_fft=np.int32(fb.compute_STFT.n_fft),
        hop_length=np.int32(fb.compute_STFT.hop_length),
        win_length=np.int32(fb.compute_STFT.win_length),
        sample_rate=np.int32(fb.compute_STFT.sample_rate),
        amin=np.float32(fbank.amin),
        log_multiplier=np.float32(fbank.multiplier),
    )


def main():
    parser = argparse.ArgumentParser(description="Export ECAPA embedding ONNX + fbank weights")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "onnx-runtime/models/ecapa_embedding.onnx",
    )
    parser.add_argument(
        "--fbank-npz",
        type=Path,
        default=ROOT / "onnx-runtime/models/ecapa_fbank.npz",
    )
    parser.add_argument("--trace-seconds", type=float, default=5.0)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    device = torch.device("cpu")
    classifier = load_ecapa_model(device)
    classifier.eval()

    t_samples = int(16_000 * args.trace_seconds)
    dummy_wav = torch.randn(1, t_samples, device=device)
    with torch.no_grad():
        dummy_features = classifier.mods["compute_features"](dummy_wav)

    wrapper = EcapaEmbeddingOnnx(classifier).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_features,
            str(args.output),
            input_names=["features"],
            output_names=["embedding"],
            dynamic_axes={"features": {1: "time"}, "embedding": {0: "batch"}},
            opset_version=args.opset,
            dynamo=False,
            do_constant_folding=True,
        )

    export_fbank_weights(classifier, args.fbank_npz)
    print(f"Exported embedding ONNX -> {args.output}")
    print(f"Exported fbank weights -> {args.fbank_npz}")
    print(f"  trace features shape: {tuple(dummy_features.shape)}")


if __name__ == "__main__":
    main()
