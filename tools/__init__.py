import torch
import numpy as np
from functools import lru_cache
from silero_vad import load_silero_vad, get_speech_timestamps
# SpeechBrain は load_ecapa_model 内で遅延ロードする（torch._dynamo との衝突回避）


MAX_EMBEDDING_AUDIO_SAMPLES = 5 * 16000

# ECAPA-TDNN の出力次元
ECAPA_EMBED_DIM = 192
_ECAPA_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
_ECAPA_SAVEDIR = "pretrained_models/spkrec-ecapa-voxceleb"


@lru_cache(maxsize=1)
def _get_silero_vad_model():
    return load_silero_vad()


def _extract_speech_only(waveform_np: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    vad_model = _get_silero_vad_model()
    waveform_tensor = torch.from_numpy(waveform_np).float().cpu()
    speech_timestamps = get_speech_timestamps(
        waveform_tensor,
        vad_model,
        sampling_rate=sample_rate,
    )

    if not speech_timestamps:
        return waveform_np

    speech_segments = []
    for timestamp in speech_timestamps:
        segment = waveform_tensor[timestamp["start"]:timestamp["end"]]
        if segment.numel() > 0:
            speech_segments.append(segment)

    if not speech_segments:
        return waveform_np

    speech_waveform = torch.cat(speech_segments)
    if speech_waveform.numel() < sample_rate // 2:
        return waveform_np

    return speech_waveform.numpy()


def load_ecapa_model(device):
    """ECAPA-TDNN (SpeechBrain) をロードして返す。"""
    import sys

    # ── Python 3.13 + torch 互換パッチ ──────────────────────────────────
    # SpeechBrain の LazyModule (k2_fsa 等) は未ロード時に __file__ などを
    # 読もうとすると ImportError を投げる。
    # torch.library._register_fake → inspect.getmodule → hasattr(mod, '__file__')
    # の経路でこのクラッシュが発生するため、ロード前にパッチを当てる。
    from speechbrain.utils.importutils import LazyModule
    if not getattr(LazyModule, "_file_access_patched", False):
        _orig_getattr = LazyModule.__getattr__
        _safe_attrs = frozenset(("__file__", "__spec__", "__loader__", "__path__",
                                  "__package__", "__cached__"))
        def _safe_getattr(self, attr):
            if attr in _safe_attrs and self.lazy_module is None:
                return None  # 未ロードモジュールの属性は None で返す
            return _orig_getattr(self, attr)
        LazyModule.__getattr__ = _safe_getattr
        LazyModule._file_access_patched = True
    # ────────────────────────────────────────────────────────────────────

    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    # SpeechBrain は "cuda:0" 形式を要求するため正規化する
    device_str = str(device)
    if device_str == "cuda":
        device_str = "cuda:0"
    model = EncoderClassifier.from_hparams(
        source=_ECAPA_MODEL_SOURCE,
        savedir=_ECAPA_SAVEDIR,
        run_opts={"device": device_str},
        local_strategy=LocalStrategy.COPY,
    )
    return model


def get_speaker_embedding(speaker_encoder, waveform: torch.Tensor) -> np.ndarray:
    """
    ECAPA-TDNN で単一 enrollment 音声からスピーカーエンベディングを抽出する。

    Args:
        speaker_encoder: SpeechBrain EncoderClassifier (ECAPA-TDNN)
        waveform: shape (1, T) の tensor (16kHz, モノラル)

    Returns:
        embedding: shape (ECAPA_EMBED_DIM,) の numpy array (float32)
    """
    waveform_np = waveform.squeeze(0).cpu().numpy()  # (T,)
    if waveform_np.shape[0] > MAX_EMBEDDING_AUDIO_SAMPLES:
        waveform_np = waveform_np[:MAX_EMBEDDING_AUDIO_SAMPLES]
    waveform_np = _extract_speech_only(waveform_np)
    if waveform_np.shape[0] > MAX_EMBEDDING_AUDIO_SAMPLES:
        waveform_np = waveform_np[:MAX_EMBEDDING_AUDIO_SAMPLES]

    wav_tensor = torch.from_numpy(waveform_np).float().unsqueeze(0)  # (1, T)
    with torch.no_grad():
        embeddings = speaker_encoder.encode_batch(wav_tensor)  # (1, 1, 192)
    return embeddings.squeeze().cpu().numpy().astype(np.float32)  # (192,)


def get_speaker_embeddings_batch(speaker_encoder, enrollment_batch: torch.Tensor) -> torch.Tensor:
    """
    enrollment 音声のバッチから、各サンプルの ECAPA-TDNN エンベディングを生成する関数。

    Args:
        speaker_encoder: SpeechBrain EncoderClassifier (ECAPA-TDNN)
        enrollment_batch: 入力 enrollment 音声テンソル、形状は (B, 1, T)

    Returns:
        torch.Tensor: 各 enrollment に対応するエンベディングテンソル、形状は (B, ECAPA_EMBED_DIM)
    """
    embeddings = []
    B = enrollment_batch.size(0)
    with torch.no_grad():
        for i in range(B):
            embeddings.append(get_speaker_embedding(speaker_encoder, enrollment_batch[i]))
    embeddings_np = np.stack(embeddings, axis=0)  # (B, ECAPA_EMBED_DIM)
    return torch.from_numpy(embeddings_np).to(enrollment_batch.device)


if __name__ == "__main__":
    # enrollment 音声テンソル (バッチサイズ2, チャンネル1, サンプル数)
    waveform1 = torch.randn(1, 203776)
    waveform2 = torch.randn(1, 203776)
    batch_waveform = torch.stack([waveform1, waveform2], dim=0)
    print("batch_waveform shape:", batch_waveform.shape)  # (2, 1, 203776)

    speaker_encoder = VoiceEncoder(device="cpu")
    speaker_embeddings = get_speaker_embeddings_batch(speaker_encoder, batch_waveform)
    print("speaker_embeddings shape:", speaker_embeddings.shape)
    # (2, 256)