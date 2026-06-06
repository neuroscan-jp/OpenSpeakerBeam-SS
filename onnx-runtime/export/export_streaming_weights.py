"""
Export SeparatorStream weights for native Rust incremental inference.

Outputs:
  streaming_separator.npz — conv / cgLN / S4D-stream / FiLM weights + metadata

Usage:
  python onnx-runtime/export/export_streaming_weights.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

EXPORT_DIR = Path(__file__).resolve().parent
ROOT = EXPORT_DIR.parents[1]
sys.path.insert(0, str(EXPORT_DIR))
sys.path.insert(0, str(ROOT))

from model import SpeakerBeamSS, S4DBlockStream  # noqa: E402
from model.s4d import S4DStream  # noqa: E402
from model.streaming import swap_gln_to_cgln  # noqa: E402


def _tensor_dict(module: torch.nn.Module, prefix: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, param in module.named_parameters():
        out[f"{prefix}.{name}"] = param.detach().cpu().numpy()
    for name, buf in module.named_buffers():
        out[f"{prefix}.{name}"] = buf.detach().cpu().numpy()
    return out


def _export_s4d_stream(s4d: S4D, prefix: str) -> dict[str, np.ndarray]:
    stream = S4DStream(s4d)
    out = _tensor_dict(stream, prefix)
    out[f"{prefix}.h"] = np.array([stream.h], dtype=np.int64)
    out[f"{prefix}.n"] = np.array([stream.n], dtype=np.int64)
    return out


def _export_s4d_block_stream(block, prefix: str) -> dict[str, np.ndarray]:
    stream = S4DBlockStream(block)
    out: dict[str, np.ndarray] = {}
    out.update(_tensor_dict(stream.ln_s4d, f"{prefix}.ln_s4d"))
    out.update(_export_s4d_stream(block.s4d, f"{prefix}.s4d"))
    out.update(_tensor_dict(stream.linear1, f"{prefix}.linear1"))
    out.update(_tensor_dict(stream.glu_conv, f"{prefix}.glu_conv"))
    out.update(_tensor_dict(stream.ln_ff2, f"{prefix}.ln_ff2"))
    out.update(_tensor_dict(stream.ff2_linear1, f"{prefix}.ff2_linear1"))
    out.update(_tensor_dict(stream.ff2_linear2, f"{prefix}.ff2_linear2"))
    return out


def _export_conv_block(conv_block, prefix: str) -> dict[str, np.ndarray]:
    shared = conv_block.shared_block
    in_conv = shared[0]
    prelu1 = shared[1]
    cgln1 = shared[2]
    depth = shared[3][0] if isinstance(shared[3], torch.nn.Sequential) else shared[3]
    chop = shared[3][1].chop_size if isinstance(shared[3], torch.nn.Sequential) else 0
    prelu2 = shared[4]
    cgln2 = shared[5]
    out: dict[str, np.ndarray] = {}
    out.update(_tensor_dict(in_conv, f"{prefix}.in_conv"))
    out.update(_tensor_dict(depth, f"{prefix}.depth_conv"))
    out.update(_tensor_dict(conv_block.res_conv, f"{prefix}.res_conv"))
    out[f"{prefix}.prelu1.weight"] = prelu1.weight.detach().cpu().numpy()
    out[f"{prefix}.prelu2.weight"] = prelu2.weight.detach().cpu().numpy()
    out[f"{prefix}.cgln1.gamma"] = cgln1.gamma.detach().cpu().numpy()
    out[f"{prefix}.cgln1.beta"] = cgln1.beta.detach().cpu().numpy()
    out[f"{prefix}.cgln2.gamma"] = cgln2.gamma.detach().cpu().numpy()
    out[f"{prefix}.cgln2.beta"] = cgln2.beta.detach().cpu().numpy()
    out[f"{prefix}.chop_size"] = np.array([chop], dtype=np.int64)
    out[f"{prefix}.depth_kernel"] = np.array([depth.kernel_size[0]], dtype=np.int64)
    out[f"{prefix}.depth_dilation"] = np.array([depth.dilation[0]], dtype=np.int64)
    out[f"{prefix}.depth_padding"] = np.array([depth.padding[0]], dtype=np.int64)
    return out


def main():
    parser = argparse.ArgumentParser(description="Export SeparatorStream weights for Rust")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/scratch_v2_lowsir/best_model.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "onnx-runtime/models/streaming_separator.npz",
    )
    args = parser.parse_args()

    model = SpeakerBeamSS()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    swap_gln_to_cgln(model)
    sep = model.separator

    arrays: dict[str, np.ndarray] = {}
    arrays.update(_tensor_dict(sep.layer_norm_in, "layer_norm_in"))
    arrays.update(_tensor_dict(sep.in_conv1x1, "in_conv1x1"))
    arrays.update(_tensor_dict(sep.out_conv1x1, "out_conv1x1"))
    arrays.update(_tensor_dict(sep.layer_norm_out, "layer_norm_out"))
    arrays.update(_tensor_dict(sep.spk_proj[0], "spk_proj.0"))
    arrays.update(_tensor_dict(sep.spk_proj[3], "spk_proj.3"))

    idx = 0
    for conv_block, s4d_block in sep.blocks1:
        arrays.update(_export_conv_block(conv_block, f"blocks.{idx}.conv"))
        arrays.update(_export_s4d_block_stream(s4d_block, f"blocks.{idx}.s4d"))
        idx += 1
    for conv_block, s4d_block in sep.blocks2:
        arrays.update(_export_conv_block(conv_block, f"blocks.{idx}.conv"))
        arrays.update(_export_s4d_block_stream(s4d_block, f"blocks.{idx}.s4d"))
        idx += 1

    arrays.update(_tensor_dict(model.decoder.deconv, "decoder.deconv"))

    arrays["num_blocks"] = np.array([idx], dtype=np.int64)
    arrays["latent_channels"] = np.array([4096], dtype=np.int64)
    arrays["dec_kernel"] = np.array([model.decoder.deconv.kernel_size[0]], dtype=np.int64)
    arrays["dec_stride"] = np.array([model.decoder.deconv.stride[0]], dtype=np.int64)
    arrays["sep_channels"] = np.array([256], dtype=np.int64)
    arrays["hidden_channels"] = np.array([512], dtype=np.int64)
    arrays["embed_dim"] = np.array([192], dtype=np.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Exported streaming separator weights -> {args.output}")
    print(f"  num_blocks={idx}, arrays={len(arrays)}")


if __name__ == "__main__":
    main()
