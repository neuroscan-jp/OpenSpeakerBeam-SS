"""Verify streaming_separator.npz weights vs PyTorch SeparatorStream."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import SpeakerBeamSS  # noqa: E402
from model.incremental_streaming import SeparatorStream  # noqa: E402
from model.streaming import swap_gln_to_cgln  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "onnx-runtime/models/streaming_separator.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/scratch_v2_lowsir/best_model.pth",
    )
    args = parser.parse_args()

    _ = np.load(args.weights)  # export sanity: file readable
    model = SpeakerBeamSS()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    swap_gln_to_cgln(model)
    model.eval()

    lat = torch.randn(1, 4096, 23)
    emb = torch.randn(1, 192)
    sep_stream = SeparatorStream(model.separator)
    states = sep_stream.initial_states(1, "cpu")
    caches = [None] * sep_stream.num_blocks

    with torch.no_grad():
        out1, states, caches = sep_stream(lat[..., :11], emb, states, caches, n_prev=0)
        out2, states, caches = sep_stream(lat, emb, states, caches, n_prev=11)
        ref = model.separator(lat, emb)

    print("chunk2 vs batch maxdiff", (out2 - ref).abs().max().item())
    print("weights file", args.weights, "ok")


if __name__ == "__main__":
    main()
