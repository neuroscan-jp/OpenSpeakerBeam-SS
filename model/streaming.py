"""
Streaming inference for SpeakerBeamSS.

Processes arbitrary-length audio in chunks. Uses cumulative layer norm (cgLN)
instead of global layer norm (gLN) in Conv1D blocks so short chunks normalize
the same way as a growing buffer. Offline batch inference still uses gLN.

Encoder: one latent frame per 320-sample window, stride 160.
Decoder: emits audio after ``lookahead_frames`` latent frames (default 1 ≈ 10 ms).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from asteroid.masknn.norms import CumLN, GlobLN
from model import SpeakerBeamSS

ENC_KERNEL = 320
ENC_STRIDE = 160
DEFAULT_LOOKAHEAD_FRAMES = 1


class OpusChunkAggregator:
    """
    Buffer Opus-sized chunks (e.g. 60 ms) and emit a process window every N chunks.

    60 ms @ 16 kHz = 960 samples; N=2 → 120 ms (1920 samples), N=3 → 180 ms (2880).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        input_chunk_ms: float = 60.0,
        process_every_n_chunks: int = 2,
    ):
        self.input_chunk_samples = int(round(sample_rate * input_chunk_ms / 1000.0))
        self.process_every_n_chunks = max(1, process_every_n_chunks)
        self._chunks = 0
        self._buf: list[float] = []

    @property
    def process_window_samples(self) -> int:
        return self.input_chunk_samples * self.process_every_n_chunks

    def push_chunk(self, chunk: torch.Tensor) -> Optional[torch.Tensor]:
        """chunk: (B, 1, T) with T <= input_chunk_samples."""
        self._buf.extend(chunk.reshape(-1).tolist())
        if chunk.shape[-1] == self.input_chunk_samples:
            self._chunks += 1
        if self._chunks >= self.process_every_n_chunks:
            n = min(self.process_window_samples, len(self._buf))
            window = torch.tensor(self._buf[:n], dtype=chunk.dtype, device=chunk.device)
            self._buf = self._buf[n:]
            self._chunks = 0
            return window.view(chunk.shape[0], 1, -1)
        return None

    def flush(self, device=None, dtype=torch.float32) -> torch.Tensor:
        if not self._buf:
            return torch.zeros(1, 1, 0, device=device, dtype=dtype)
        out = torch.tensor(self._buf, dtype=dtype, device=device).view(1, 1, -1)
        self._buf.clear()
        self._chunks = 0
        return out


def swap_gln_to_cgln(module: nn.Module) -> None:
    """Replace GlobLN with CumLN (gamma/beta copied) for streamable normalization."""
    for name, child in list(module.named_children()):
        if isinstance(child, GlobLN):
            dev = child.gamma.device
            cg = CumLN(child.channel_size).to(dev)
            cg.gamma.data.copy_(child.gamma.data)
            cg.beta.data.copy_(child.beta.data)
            setattr(module, name, cg)
        else:
            swap_gln_to_cgln(child)


@dataclass
class StreamState:
    audio_buf: torch.Tensor  # (B, 1, T_total)
    latent_buf: torch.Tensor  # (B, 4096, F)
    wav_emitted: int = 0
    total_in: int = 0


class SpeakerBeamSSStream:
    """
    Chunk-wise streaming wrapper around a trained :class:`SpeakerBeamSS`.

    Parameters
    ----------
    hop_samples : int
        Expected chunk size per :meth:`push` (e.g. 1600 = 100 ms @ 16 kHz).
    use_cgln : bool
        If True (default), swap gLN → cgLN for streamable normalization.
    lookahead_frames : int
        Latent frames to buffer before emitting decoder output (default 1).
    """

    def __init__(
        self,
        model: SpeakerBeamSS,
        hop_samples: int = 1600,
        use_cgln: bool = True,
        lookahead_frames: int = DEFAULT_LOOKAHEAD_FRAMES,
    ):
        self.model = model.eval()
        if use_cgln:
            swap_gln_to_cgln(self.model)
        self.hop_samples = hop_samples
        self.lookahead_frames = lookahead_frames
        self._spk_emb: Optional[torch.Tensor] = None
        self._state: Optional[StreamState] = None

    def set_embedding(self, spk_embedding: torch.Tensor) -> None:
        """Set target speaker embedding (B, 192), computed once offline."""
        self._spk_emb = spk_embedding

    def reset(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        dev = device or next(self.model.parameters()).device
        self._state = StreamState(
            audio_buf=torch.zeros(batch_size, 1, 0, device=dev),
            latent_buf=torch.zeros(batch_size, 4096, 0, device=dev),
        )

    def _append_latent_frames(self, state: StreamState) -> None:
        """Encode new latent frames from audio received so far."""
        t_total = state.audio_buf.shape[-1]
        n_frames = (t_total - ENC_KERNEL) // ENC_STRIDE + 1
        if n_frames <= 0:
            return
        cur = state.latent_buf.shape[-1]
        if n_frames <= cur:
            return
        new_frames = []
        audio = state.audio_buf
        for i in range(cur, n_frames):
            win = audio[..., i * ENC_STRIDE : i * ENC_STRIDE + ENC_KERNEL]
            new_frames.append(self.model.encoder(win))
        state.latent_buf = torch.cat([state.latent_buf] + new_frames, dim=-1)

    def _emit_ready(self, state: StreamState) -> torch.Tensor:
        """Run separator/decoder on current latent buffer; return newly ready audio."""
        if state.latent_buf.shape[-1] == 0:
            return torch.zeros(
                state.audio_buf.shape[0], 1, 0, device=state.audio_buf.device
            )

        with torch.no_grad():
            sep = self.model.separator(state.latent_buf, self._spk_emb)
            wav = self.model.decoder(sep)

        safe_end = wav.shape[-1] - self.lookahead_frames * ENC_STRIDE
        if safe_end <= state.wav_emitted:
            return torch.zeros(wav.shape[0], 1, 0, device=wav.device)

        out = wav[..., state.wav_emitted : safe_end]
        state.wav_emitted = safe_end
        return out

    def push(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Process one audio chunk.

        Parameters
        ----------
        chunk : Tensor (B, 1, T)

        Returns
        -------
        Tensor (B, 1, T_out) — separated audio ready after decoder lookahead.
        """
        if self._spk_emb is None:
            raise RuntimeError("call set_embedding() before push()")
        if self._state is None:
            self.reset(batch_size=chunk.shape[0], device=chunk.device)

        state = self._state
        state.audio_buf = torch.cat([state.audio_buf, chunk], dim=-1)
        state.total_in += chunk.shape[-1]

        self._append_latent_frames(state)
        return self._emit_ready(state)

    def flush(self) -> torch.Tensor:
        """Emit remaining audio (call once at end of stream)."""
        if self._state is None or self._spk_emb is None:
            return torch.zeros(1, 1, 0)

        state = self._state
        if state.latent_buf.shape[-1] == 0:
            return torch.zeros(state.audio_buf.shape[0], 1, 0, device=state.audio_buf.device)

        with torch.no_grad():
            sep = self.model.separator(state.latent_buf, self._spk_emb)
            wav = self.model.decoder(sep)

        if wav.shape[-1] <= state.wav_emitted:
            return torch.zeros(wav.shape[0], 1, 0, device=wav.device)

        out = wav[..., state.wav_emitted :]
        state.wav_emitted = wav.shape[-1]
        return out
