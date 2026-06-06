"""Verify Python IncrementalSpeakerBeamSSStream vs ONNX encoder/decoder + PyTorch separator."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "onnx-runtime"))

from export.extract_embedding import extract_enrollment_embedding, load_mono_16k  # noqa: E402
from model import SpeakerBeamSS  # noqa: E402
from model.incremental_streaming import IncrementalSpeakerBeamSSStream  # noqa: E402
from model.streaming import OpusChunkAggregator, swap_gln_to_cgln  # noqa: E402
from tools import load_ecapa_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", type=Path, default=ROOT / "data/sample/mixture_000001.wav")
    parser.add_argument("--enrollment", type=Path, default=ROOT / "data/sample/enrollment_000001.wav")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/scratch_v2_lowsir/best_model.pth")
    parser.add_argument("--input-chunk-ms", type=float, default=60.0)
    parser.add_argument("--process-every-chunks", type=int, default=2)
    args = parser.parse_args()

    mix = load_mono_16k(args.mixture)
    emb = torch.from_numpy(
        extract_enrollment_embedding(
            load_mono_16k(args.enrollment), load_ecapa_model(torch.device("cpu"))
        )
    ).unsqueeze(0)

    model = SpeakerBeamSS()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    swap_gln_to_cgln(model)
    model.eval()

    stream = IncrementalSpeakerBeamSSStream(model)
    stream.set_embedding(emb)
    stream.reset(1)
    agg = OpusChunkAggregator(16000, args.input_chunk_ms, args.process_every_chunks)
    parts = []
    times = []
    samples = mix.numpy().squeeze()
    for start in range(0, len(samples), agg.input_chunk_samples):
        ch = samples[start : start + agg.input_chunk_samples]
        w = agg.push_chunk(torch.from_numpy(ch[None, None, :].astype(np.float32)))
        if w is not None:
            t0 = time.perf_counter()
            parts.append(stream.push(w).numpy().squeeze())
            times.append((time.perf_counter() - t0) * 1000)
    tail = agg.flush(device="cpu", dtype=torch.float32)
    if tail.shape[-1]:
        parts.append(stream.push(tail).numpy().squeeze())
    parts.append(stream.flush().numpy().squeeze())
    out = np.concatenate([p for p in parts if p.size])
    print(f"output_len={len(out)} steps={len(times)}")
    print(f"py_incremental_ms median={np.median(times):.1f} p95={np.percentile(times, 95):.1f}")
    print("Run `cargo run -p speakerbeam-cli -- --stream ...` for Rust incremental parity.")


if __name__ == "__main__":
    main()
