"""
Export split ONNX models for Rust streaming inference (cgLN separator).

Outputs:
  encoder_frame.onnx     : (B, 1, 320) → (B, 4096, 1)  [dynamic B]
  decoder.onnx           : (1, 4096, L) → (1, 1, T_out)  [dynamic L]
  separator_cgln.onnx    : (1, 4096, Lpad) + (1, 192) → (1, 4096, Lpad)  [fixed Lpad]
  separator_chunk.onnx   : (1, 4096, Lchunk) + (1, 192) → (1, 4096, Lchunk)  [fixed Lchunk]

`separator_chunk` is for short-window benchmarks / future native-S4D hybrid paths.

Usage:
  python onnx-runtime/export/export_streaming_models.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

EXPORT_DIR = Path(__file__).resolve().parent
ROOT = EXPORT_DIR.parents[1]
sys.path.insert(0, str(EXPORT_DIR))
sys.path.insert(0, str(ROOT))

from model import SpeakerBeamSS  # noqa: E402
from model.streaming import swap_gln_to_cgln  # noqa: E402
from onnx_utils import patch_s4d_for_onnx, restore_s4d  # noqa: E402


class EncoderFrameOnnx(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.encoder(waveform)


class DecoderOnnx(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


class SeparatorCglnOnnx(nn.Module):
    def __init__(self, separator):
        super().__init__()
        self.separator = separator

    def forward(self, latent: torch.Tensor, spk_embedding: torch.Tensor) -> torch.Tensor:
        return self.separator(latent, spk_embedding)


def main():
    parser = argparse.ArgumentParser(description="Export streaming split ONNX models")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/scratch_v2_lowsir/best_model.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "onnx-runtime/models",
    )
    parser.add_argument(
        "--separator-latent",
        type=int,
        default=2048,
        help="Fixed latent frames for separator_cgln.onnx (zero-pad shorter)",
    )
    parser.add_argument(
        "--separator-chunk-latent",
        type=int,
        default=16,
        help="Fixed latent frames for separator_chunk.onnx",
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    device = torch.device("cpu")
    model = SpeakerBeamSS().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    swap_gln_to_cgln(model)
    backups = patch_s4d_for_onnx(model)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    enc_path = args.output_dir / "encoder_frame.onnx"
    dec_path = args.output_dir / "decoder.onnx"
    sep_path = args.output_dir / "separator_cgln.onnx"
    sep_chunk_path = args.output_dir / "separator_chunk.onnx"

    try:
        with torch.no_grad():
            torch.onnx.export(
                EncoderFrameOnnx(model.encoder).eval(),
                torch.randn(2, 1, 320, device=device),
                str(enc_path),
                input_names=["waveform"],
                output_names=["latent"],
                dynamic_axes={
                    "waveform": {0: "batch"},
                    "latent": {0: "batch"},
                },
                opset_version=args.opset,
                dynamo=False,
            )
            l_trace = 64
            torch.onnx.export(
                DecoderOnnx(model.decoder).eval(),
                torch.randn(1, 4096, l_trace, device=device),
                str(dec_path),
                input_names=["latent"],
                output_names=["waveform"],
                dynamic_axes={"latent": {2: "time"}, "waveform": {2: "time_out"}},
                opset_version=args.opset,
                dynamo=False,
            )
            l_sep = args.separator_latent
            torch.onnx.export(
                SeparatorCglnOnnx(model.separator).eval(),
                (
                    torch.randn(1, 4096, l_sep, device=device),
                    torch.randn(1, 192, device=device),
                ),
                str(sep_path),
                input_names=["latent", "spk_embedding"],
                output_names=["separated"],
                opset_version=args.opset,
                dynamo=False,
            )
            l_chunk = args.separator_chunk_latent
            torch.onnx.export(
                SeparatorCglnOnnx(model.separator).eval(),
                (
                    torch.randn(1, 4096, l_chunk, device=device),
                    torch.randn(1, 192, device=device),
                ),
                str(sep_chunk_path),
                input_names=["latent", "spk_embedding"],
                output_names=["separated"],
                opset_version=args.opset,
                dynamo=False,
            )
    finally:
        restore_s4d(backups)

    print(f"Exported encoder_frame -> {enc_path}")
    print(f"Exported decoder      -> {dec_path}")
    print(f"Exported separator    -> {sep_path} (fixed L={args.separator_latent})")
    print(f"Exported sep chunk    -> {sep_chunk_path} (fixed L={args.separator_chunk_latent})")


if __name__ == "__main__":
    main()
