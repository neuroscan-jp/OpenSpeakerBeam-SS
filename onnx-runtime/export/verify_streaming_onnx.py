"""Verify Rust-oriented split ONNX streaming vs PyTorch SpeakerBeamSSStream."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
EXPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPORT.parent))

from model import SpeakerBeamSS  # noqa: E402
from model.streaming import SpeakerBeamSSStream, swap_gln_to_cgln  # noqa: E402
from export.extract_embedding import extract_enrollment_embedding, load_mono_16k  # noqa: E402
from tools import load_ecapa_model  # noqa: E402

ENC_KERNEL = 320
ENC_STRIDE = 160
LATENT_C = 4096
LOOKAHEAD = 1


class OnnxStream:
    """Python reimplementation of Rust StreamingSession for parity check."""

    def __init__(self, enc_sess, dec_sess, sep_sess, sep_cap: int, emb: np.ndarray):
        self.enc = enc_sess
        self.dec = dec_sess
        self.sep = sep_sess
        self.sep_cap = sep_cap
        self.emb = emb.astype(np.float32)
        self.audio: list[float] = []
        self.latent: list[np.ndarray] = []
        self.n_latent = 0
        self.wav_emitted = 0

    def reset(self):
        self.audio.clear()
        self.latent.clear()
        self.n_latent = 0
        self.wav_emitted = 0

    def _append_latent(self):
        t = len(self.audio)
        if t < ENC_KERNEL:
            return
        n_frames = (t - ENC_KERNEL) // ENC_STRIDE + 1
        while self.n_latent < n_frames:
            i = self.n_latent
            win = np.array(self.audio[i * ENC_STRIDE : i * ENC_STRIDE + ENC_KERNEL], dtype=np.float32)
            lat = self.enc.run(None, {"waveform": win[None, None, :]})[0]
            self.latent.append(lat)
            self.n_latent += 1

    def _emit(self) -> np.ndarray:
        if self.n_latent == 0:
            return np.array([], dtype=np.float32)
        n = self.n_latent
        latent = np.concatenate(self.latent, axis=-1)
        pad = np.zeros((1, LATENT_C, self.sep_cap), dtype=np.float32)
        pad[..., :n] = latent
        sep = self.sep.run(
            None,
            {"latent": pad, "spk_embedding": self.emb[None, :]},
        )[0][..., :n]
        wav = self.dec.run(None, {"latent": sep})[0].squeeze()
        safe = wav.shape[-1] - LOOKAHEAD * ENC_STRIDE
        if safe <= self.wav_emitted:
            return np.array([], dtype=np.float32)
        out = wav[self.wav_emitted : safe]
        self.wav_emitted = safe
        return out

    def push(self, chunk: np.ndarray) -> np.ndarray:
        self.audio.extend(chunk.tolist())
        self._append_latent()
        return self._emit()

    def flush(self) -> np.ndarray:
        if self.n_latent == 0:
            return np.array([], dtype=np.float32)
        n = self.n_latent
        latent = np.concatenate(self.latent, axis=-1)
        pad = np.zeros((1, LATENT_C, self.sep_cap), dtype=np.float32)
        pad[..., :n] = latent
        sep = self.sep.run(
            None,
            {"latent": pad, "spk_embedding": self.emb[None, :]},
        )[0][..., :n]
        wav = self.dec.run(None, {"latent": sep})[0].squeeze()
        if wav.shape[-1] <= self.wav_emitted:
            return np.array([], dtype=np.float32)
        out = wav[self.wav_emitted :]
        self.wav_emitted = wav.shape[-1]
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=ROOT / "onnx-runtime/models")
    parser.add_argument("--mixture", type=Path, default=ROOT / "data/sample/mixture_000001.wav")
    parser.add_argument("--enrollment", type=Path, default=ROOT / "data/sample/enrollment_000001.wav")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/scratch_v2_lowsir/best_model.pth")
    parser.add_argument("--hop-ms", type=float, default=100.0)
    args = parser.parse_args()

    hop = int(16000 * args.hop_ms / 1000.0)
    mix = load_mono_16k(args.mixture)
    enc_ecapa = load_ecapa_model(torch.device("cpu"))
    emb = extract_enrollment_embedding(
        load_mono_16k(args.enrollment).to(mix.device), enc_ecapa
    )

    model = SpeakerBeamSS()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    swap_gln_to_cgln(model)
    model.eval()

    with torch.no_grad():
        ref_parts = []
        stream = SpeakerBeamSSStream(model, hop_samples=hop, use_cgln=False)
        stream.set_embedding(torch.from_numpy(emb).unsqueeze(0))
        stream.reset(1)
        T = mix.shape[-1]
        for start in range(0, T, hop):
            ref_parts.append(stream.push(mix[..., start : start + hop]).numpy().squeeze())
        tail = stream.flush().numpy().squeeze()
        if tail.size:
            ref_parts.append(tail)
        reference = np.concatenate([p for p in ref_parts if p.size])

    md = args.models_dir
    enc_sess = ort.InferenceSession(str(md / "encoder_frame.onnx"), providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(md / "decoder.onnx"), providers=["CPUExecutionProvider"])
    sep_sess = ort.InferenceSession(str(md / "separator_cgln.onnx"), providers=["CPUExecutionProvider"])
    sep_cap = sep_sess.get_inputs()[0].shape[2]

    onnx_stream = OnnxStream(enc_sess, dec_sess, sep_sess, int(sep_cap), emb)
    onnx_parts = []
    samples = mix.numpy().squeeze()
    for start in range(0, len(samples), hop):
        onnx_parts.append(onnx_stream.push(samples[start : start + hop]))
    tail = onnx_stream.flush()
    if tail.size:
        onnx_parts.append(tail)
    candidate = np.concatenate([p for p in onnx_parts if p.size])

    n = min(len(reference), len(candidate))
    diff = np.abs(reference[:n] - candidate[:n])
    print(f"lengths ref={len(reference)} onnx={len(candidate)}")
    print(f"max_abs_diff={diff.max():.6e}, mean={diff.mean():.6e}")
    if diff.max() > 1e-2:
        raise SystemExit("split ONNX streaming diverged from PyTorch cgLN stream")


if __name__ == "__main__":
    main()
