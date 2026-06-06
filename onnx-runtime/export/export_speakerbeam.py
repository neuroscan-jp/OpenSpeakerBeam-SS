"""
Export SpeakerBeamSS (ep110) to ONNX — **offline batch inference only**.

Streaming uses model/streaming.py (cgLN + state); do not rely on this single graph.

S4D FFT is replaced with conv1d during export (onnx_utils.patch_s4d_for_onnx).
Dynamic-length export fails on S4D conv; use --no_dynamic_time (default trace 10 s).

Usage:
  python onnx-runtime/export/export_speakerbeam.py --no_dynamic_time
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
from onnx_utils import patch_s4d_for_onnx, restore_s4d  # noqa: E402


class SpeakerBeamOnnxWrapper(nn.Module):
    def __init__(self, model: SpeakerBeamSS):
        super().__init__()
        self.model = model

    def forward(self, mixture: torch.Tensor, spk_embedding: torch.Tensor) -> torch.Tensor:
        return self.model(mixture, spk_embedding)


def main():
    parser = argparse.ArgumentParser(description="Export SpeakerBeamSS to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/scratch_v2_lowsir/best_model.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "onnx-runtime/models/speakerbeam_ep110.onnx",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--t_seconds", type=float, default=10.0,
                        help="Dummy trace length for export verification")
    parser.add_argument("--no_dynamic_time", action="store_true", default=True,
                        help="Fixed trace length (required; dynamic S4D export unsupported)")
    parser.add_argument("--dynamic_time", action="store_true",
                        help="Attempt dynamic time axis (likely fails on S4D conv)")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = SpeakerBeamSS().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    t_samples = int(16000 * args.t_seconds)
    backups = patch_s4d_for_onnx(model)
    wrapper = SpeakerBeamOnnxWrapper(model).eval()
    dummy_mix = torch.randn(1, 1, t_samples, device=device)
    dummy_emb = torch.randn(1, 192, device=device)
    dummy_emb = torch.nn.functional.normalize(dummy_emb, dim=-1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.dynamic_time:
        args.no_dynamic_time = False

    dynamic_axes = None
    if not args.no_dynamic_time:
        dynamic_axes = {
            "mixture": {2: "time"},
            "enhanced": {2: "time_out"},
        }

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_mix, dummy_emb),
                str(args.output),
                input_names=["mixture", "spk_embedding"],
                output_names=["enhanced"],
                dynamic_axes=dynamic_axes,
                opset_version=args.opset,
                dynamo=False,
                do_constant_folding=True,
            )
    finally:
        restore_s4d(backups)

    print(f"Exported ONNX -> {args.output}")
    print(f"  trace length: {t_samples} samples ({args.t_seconds}s @ 16kHz)")
    print(f"  dynamic time axis: {not args.no_dynamic_time}")


if __name__ == "__main__":
    main()
