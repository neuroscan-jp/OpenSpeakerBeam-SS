"""
Incremental streaming separator: full conv path + S4D state on new frames only.

Conv/cgLN run on the full latent buffer each chunk (matches batch cgLN stats).
S4D blocks keep state and only process newly added latent frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from model import S4DBlockStream, Separator, SpeakerBeamSS
from model.streaming import swap_gln_to_cgln

ENC_KERNEL = 320
ENC_STRIDE = 160
DEFAULT_LOOKAHEAD_FRAMES = 1


class SeparatorStream(nn.Module):
    """Separator with S4DBlockStream; conv runs on full latent, S4D on new frames only."""

    def __init__(self, separator: Separator):
        super().__init__()
        self.sep = separator
        self.s4d_streams1 = nn.ModuleList(
            S4DBlockStream(pair[1]) for pair in separator.blocks1
        )
        self.s4d_streams2 = nn.ModuleList(
            S4DBlockStream(pair[1]) for pair in separator.blocks2
        )

    def initial_states(self, batch_size: int, device) -> List[torch.Tensor]:
        states = []
        for m in list(self.s4d_streams1) + list(self.s4d_streams2):
            states.append(m.initial_state(batch_size, device))
        return states

    @property
    def num_blocks(self) -> int:
        return len(self.sep.blocks1) + len(self.sep.blocks2)

    def forward(
        self,
        x: torch.Tensor,
        spk: torch.Tensor,
        s4d_states: List[torch.Tensor],
        block_caches: List[Optional[torch.Tensor]],
        n_prev: int = 0,
    ) -> tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        input_orig = x
        x = x.transpose(1, 2)
        x = self.sep.layer_norm_in(x)
        x = x.transpose(1, 2)
        x = self.sep.in_conv1x1(x)
        spk = self.sep.spk_proj(spk)

        new_caches: List[torch.Tensor] = []
        idx = 0
        for (conv_block, _), adapt, drop, s4d_stream in zip(
            self.sep.blocks1, self.sep.adapt1, self.sep.drop1, self.s4d_streams1
        ):
            x = conv_block(x)
            x_new, s4d_states[idx] = s4d_stream(x[..., n_prev:], s4d_states[idx])
            if n_prev > 0 and block_caches[idx] is not None:
                x_new = adapt(x_new, spk)
                x_new = drop(x_new)
                x = torch.cat([block_caches[idx][..., :n_prev], x_new], dim=-1)
            else:
                x = adapt(x_new, spk)
                x = drop(x)
            idx += 1
            new_caches.append(x)

        for (conv_block, _), adapt, drop, s4d_stream in zip(
            self.sep.blocks2, self.sep.adapt2, self.sep.drop2, self.s4d_streams2
        ):
            x = conv_block(x)
            x_new, s4d_states[idx] = s4d_stream(x[..., n_prev:], s4d_states[idx])
            if n_prev > 0 and block_caches[idx] is not None:
                x_new = adapt(x_new, spk)
                x_new = drop(x_new)
                x = torch.cat([block_caches[idx][..., :n_prev], x_new], dim=-1)
            else:
                x = adapt(x_new, spk)
                x = drop(x)
            idx += 1
            new_caches.append(x)

        x = self.sep.out_conv1x1(x)
        x = x.transpose(1, 2)
        x = self.sep.layer_norm_out(x)
        x = x.transpose(1, 2)
        x = torch.relu(x)
        x = x * input_orig
        return x, s4d_states, new_caches


@dataclass
class IncrementalStreamState:
    audio_buf: torch.Tensor
    latent_buf: torch.Tensor
    sep_buf: torch.Tensor
    s4d_states: List[torch.Tensor]
    block_caches: List[Optional[torch.Tensor]]
    processed_latent: int = 0
    wav_emitted: int = 0
    total_in: int = 0


class IncrementalSpeakerBeamSSStream:
    """
    Incremental streaming: encoder frame-wise + separator with incremental S4D.
    """

    def __init__(
        self,
        model: SpeakerBeamSS,
        lookahead_frames: int = DEFAULT_LOOKAHEAD_FRAMES,
        use_cgln: bool = True,
    ):
        self.model = model.eval()
        if use_cgln:
            swap_gln_to_cgln(self.model)
        self.lookahead_frames = lookahead_frames
        self.separator_stream = SeparatorStream(self.model.separator)
        self._spk_emb: Optional[torch.Tensor] = None
        self._state: Optional[IncrementalStreamState] = None

    def set_embedding(self, spk_embedding: torch.Tensor) -> None:
        self._spk_emb = spk_embedding

    def reset(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        dev = device or next(self.model.parameters()).device
        n_blocks = self.separator_stream.num_blocks
        self._state = IncrementalStreamState(
            audio_buf=torch.zeros(batch_size, 1, 0, device=dev),
            latent_buf=torch.zeros(batch_size, 4096, 0, device=dev),
            sep_buf=torch.zeros(batch_size, 4096, 0, device=dev),
            s4d_states=self.separator_stream.initial_states(batch_size, dev),
            block_caches=[None] * n_blocks,
        )

    def _append_latent_frames(self, state: IncrementalStreamState) -> int:
        t_total = state.audio_buf.shape[-1]
        n_frames = (t_total - ENC_KERNEL) // ENC_STRIDE + 1
        if n_frames <= state.latent_buf.shape[-1]:
            return 0
        cur = state.latent_buf.shape[-1]
        new_frames = []
        audio = state.audio_buf
        for i in range(cur, n_frames):
            win = audio[..., i * ENC_STRIDE : i * ENC_STRIDE + ENC_KERNEL]
            new_frames.append(self.model.encoder(win))
        state.latent_buf = torch.cat([state.latent_buf] + new_frames, dim=-1)
        return n_frames - cur

    def _run_separator(self, state: IncrementalStreamState) -> None:
        n_prev = state.processed_latent
        n_total = state.latent_buf.shape[-1]
        sep_chunk, state.s4d_states, state.block_caches = self.separator_stream(
            state.latent_buf,
            self._spk_emb,
            state.s4d_states,
            state.block_caches,
            n_prev=n_prev,
        )
        sep_new = sep_chunk[..., n_prev:n_total]
        if sep_new.shape[-1] > 0:
            state.sep_buf = torch.cat([state.sep_buf, sep_new], dim=-1)
        state.processed_latent = n_total

    def _emit_ready(self, state: IncrementalStreamState, n_new: int) -> torch.Tensor:
        if n_new <= 0 or self._spk_emb is None:
            return torch.zeros(state.audio_buf.shape[0], 1, 0, device=state.audio_buf.device)

        with torch.no_grad():
            self._run_separator(state)
            wav = self.model.decoder(state.sep_buf)

        safe_end = wav.shape[-1] - self.lookahead_frames * ENC_STRIDE
        if safe_end <= state.wav_emitted:
            return torch.zeros(wav.shape[0], 1, 0, device=wav.device)

        out = wav[..., state.wav_emitted : safe_end]
        state.wav_emitted = safe_end
        return out

    def push(self, chunk: torch.Tensor) -> torch.Tensor:
        if self._spk_emb is None:
            raise RuntimeError("call set_embedding() before push()")
        if self._state is None:
            self.reset(batch_size=chunk.shape[0], device=chunk.device)

        state = self._state
        state.audio_buf = torch.cat([state.audio_buf, chunk], dim=-1)
        state.total_in += chunk.shape[-1]
        n_new = self._append_latent_frames(state)
        return self._emit_ready(state, n_new)

    def flush(self) -> torch.Tensor:
        if self._state is None or self._spk_emb is None:
            return torch.zeros(1, 1, 0)

        state = self._state
        n_new = state.latent_buf.shape[-1] - state.processed_latent

        with torch.no_grad():
            if n_new > 0:
                self._run_separator(state)
            if state.sep_buf.shape[-1] == 0:
                return torch.zeros(
                    state.audio_buf.shape[0], 1, 0, device=state.audio_buf.device
                )
            wav = self.model.decoder(state.sep_buf)

        if wav.shape[-1] <= state.wav_emitted:
            return torch.zeros(wav.shape[0], 1, 0, device=wav.device)
        out = wav[..., state.wav_emitted :]
        state.wav_emitted = wav.shape[-1]
        return out
