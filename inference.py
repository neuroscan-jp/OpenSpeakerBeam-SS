import os
import torch
import torchaudio
import argparse
import soundfile as sf
from model import SpeakerBeamSS
from tools import get_speaker_embeddings_batch, load_ecapa_model


def speaker_verification_filter(
    enhanced: torch.Tensor,
    enrollment_embedding: torch.Tensor,
    speaker_encoder,
    sample_rate: int = 16000,
    device: torch.device = torch.device("cpu"),
    win_sec: float = 1.0,
    hop_sec: float = 0.25,
    threshold: float = 0.25,
    smooth_frames: int = 5,
) -> torch.Tensor:
    """
    スライディングウィンドウ話者照合フィルタ。

    分離出力の各フレームをenrollment埋め込みとcosine類似度で照合し、
    閾値以下のフレームをゼロ抑制する（フェードで平滑化）。

    - 001冒頭問題: ターゲット話者が無音の最初の数秒を抑制
    - 002末尾問題: ターゲット話者が終了した後のサブ話者漏れを抑制
    - 主の声には影響しない（類似度が高いフレームはそのまま）

    Args:
        enhanced: (B, 1, T) 分離出力
        enrollment_embedding: (B, emb_dim) enrollment埋め込み（正規化済み）
        speaker_encoder: ECAPA-TDNNモデル
        sample_rate: サンプリングレート
        win_sec: 照合ウィンドウ長（秒）
        hop_sec: ウィンドウシフト（秒）
        threshold: cosine類似度の閾値（これ以下でゼロ抑制）
        smooth_frames: ゲートを平滑化するフレーム数
    """
    B, C, T = enhanced.shape
    win_len = int(win_sec * sample_rate)
    hop_len = int(hop_sec * sample_rate)

    # enrollment埋め込みを正規化（cosine類似度用）
    enroll_norm = torch.nn.functional.normalize(enrollment_embedding, dim=-1)  # (B, D)

    result = enhanced.clone()

    for b in range(B):
        signal = enhanced[b, 0]  # (T,)
        n_frames = max(1, (T - win_len) // hop_len + 1)
        scores = []

        for i in range(n_frames):
            start = i * hop_len
            end = min(start + win_len, T)
            frame = signal[start:end]

            # エネルギーが極めて小さいフレームは照合スキップ（スコア=0）
            if frame.abs().max() < 1e-5:
                scores.append(0.0)
                continue

            # ECAPA-TDNNで埋め込みを計算
            frame_input = frame.unsqueeze(0).unsqueeze(0)  # (1, 1, t)
            with torch.no_grad():
                frame_emb = get_speaker_embeddings_batch(speaker_encoder, frame_input.to(device))
            frame_norm = torch.nn.functional.normalize(frame_emb, dim=-1)  # (1, D)

            sim = (frame_norm * enroll_norm[b:b+1]).sum(dim=-1).item()
            scores.append(sim)

        # 適応的閾値：1D k-means (k=2) でスコアを2クラスタに分割
        # onset/absent と present を自動分離する
        if len(scores) >= 4:
            import numpy as np
            scores_arr = np.array(scores, dtype=np.float32)
            # k-means初期化: min と max
            c1 = float(scores_arr.min())
            c2 = float(scores_arr.max())
            for _ in range(50):  # 収束まで反復
                mid = (c1 + c2) / 2.0
                cl1 = scores_arr[scores_arr < mid]
                cl2 = scores_arr[scores_arr >= mid]
                new_c1 = float(cl1.mean()) if len(cl1) > 0 else c1
                new_c2 = float(cl2.mean()) if len(cl2) > 0 else c2
                if abs(new_c1 - c1) < 1e-6 and abs(new_c2 - c2) < 1e-6:
                    break
                c1, c2 = new_c1, new_c2
            adaptive_threshold = (c1 + c2) / 2.0
            print(f"[Filter] b={b}: n_frames={len(scores)}, kmeans c1={c1:.4f} c2={c2:.4f}, adaptive_threshold={adaptive_threshold:.4f}")
        else:
            adaptive_threshold = threshold

        # フレームスコアをサンプルレベルのゲートに変換（zeros→高類似フレームのみ1.0にセット）
        gate_samples = torch.zeros(T, device=enhanced.device)
        for i, score in enumerate(scores):
            start = i * hop_len
            end = min(start + win_len, T)
            if score >= adaptive_threshold:
                gate_samples[start:end] = 1.0

        # ゲートを平滑化（急激なミュートを避ける）
        smooth_len = smooth_frames * hop_len
        kernel = torch.ones(smooth_len, device=enhanced.device) / smooth_len
        gate_samples_2d = gate_samples.unsqueeze(0).unsqueeze(0)
        kernel_3d = kernel.view(1, 1, -1)
        gate_smooth = torch.nn.functional.conv1d(
            gate_samples_2d,
            kernel_3d,
            padding=smooth_len // 2,
        ).squeeze()[:T]
        gate_smooth = gate_smooth.clamp(0.0, 1.0)

        result[b, 0] = signal * gate_smooth

    return result


def load_audio(path):
    waveform_np, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform_np.T.copy())
    return waveform, sample_rate


def save_audio(path, waveform, sample_rate):
    waveform_np = waveform.detach().cpu().transpose(0, 1).numpy()
    sf.write(path, waveform_np, sample_rate)

def main():
    parser = argparse.ArgumentParser(description="Inference for SpeakerBeam-SS")
    parser.add_argument("--mixture", type=str, required=True,
                        help="Path to the input mixture audio file")
    parser.add_argument("--enrollment", type=str, required=True,
                        help="Path to the enrollment audio file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the output (enhanced) wav file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing the best_model.pth checkpoint")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Target sample rate (default: 16000)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- 1. 混合音声の読み込みと前処理 -----
    mixture_waveform, sr_m = load_audio(args.mixture)
    if sr_m != args.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr_m, new_freq=args.sample_rate)
        mixture_waveform = resampler(mixture_waveform)
    # モノラル化（必要なら平均化）
    if mixture_waveform.shape[0] > 1:
        mixture_waveform = torch.mean(mixture_waveform, dim=0, keepdim=True)
    # バッチ次元追加（形状: (B, 1, T) ）
    if mixture_waveform.dim() == 2:
        mixture_waveform = mixture_waveform.unsqueeze(0)
    mixture_waveform = mixture_waveform.to(device)

    # ----- 2. enrollment 音声の読み込みと前処理 -----
    enrollment_waveform, sr_e = load_audio(args.enrollment)
    if sr_e != args.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr_e, new_freq=args.sample_rate)
        enrollment_waveform = resampler(enrollment_waveform)
    if enrollment_waveform.shape[0] > 1:
        enrollment_waveform = torch.mean(enrollment_waveform, dim=0, keepdim=True)
    if enrollment_waveform.dim() == 2:
        enrollment_waveform = enrollment_waveform.unsqueeze(0)
    enrollment_waveform = enrollment_waveform.to(device)

    # ----- 3. 学習済みモデルと Speaker Encoder のロード -----
    model = SpeakerBeamSS().to(device)
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Checkpoint not found at {best_model_path}. Train the model first.")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # SpeakerEncoder を初期化 (ECAPA-TDNN)
    speaker_encoder = load_ecapa_model(device)

    # ----- 4. enrollment音声からスピーカー埋め込みを取得（マルチセグメント平均） -----
    # enrollment を複数区間に分割して各埋め込みを平均 → より安定した話者表現
    T = enrollment_waveform.shape[-1]
    segment_samples = 5 * args.sample_rate  # 5秒ずつ
    embeddings_list = []
    if T <= segment_samples:
        # 音声が短い場合はそのまま1回
        emb = get_speaker_embeddings_batch(speaker_encoder, enrollment_waveform)
        embeddings_list.append(emb)
    else:
        # 最大4区間（オーバーラップあり）で埋め込みを計算して平均
        step = max(1, (T - segment_samples) // 3)
        starts = list(range(0, T - segment_samples + 1, step))[:4]
        for s in starts:
            seg = enrollment_waveform[..., s: s + segment_samples]
            emb = get_speaker_embeddings_batch(speaker_encoder, seg)
            embeddings_list.append(emb)

    speaker_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)  # (B, emb_dim)

    # ----- 5. 推論実行 -----
    with torch.no_grad():
        enhanced = model(mixture_waveform, speaker_embeddings)
    # enhanced の形状: (B, 1, T)

    # ----- 5.5. 話者照合フィルタ（スライディングウィンドウ）-----
    # 各フレームをenrollment話者埋め込みと照合し、類似度が低いフレームを抑制する。
    # 001の冒頭・002の末尾のような「ターゲット話者が無音のフレームにサブ話者が漏れる」
    # 問題を直接修正する。
    enhanced = speaker_verification_filter(
        enhanced, speaker_embeddings, speaker_encoder, args.sample_rate, device
    )

    # ----- 6. 推論結果を wav として保存 -----
    enhanced = enhanced.cpu()
    enhanced = enhanced.squeeze(0)
    save_audio(args.output, enhanced, args.sample_rate)
    print(f"Inference complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()
