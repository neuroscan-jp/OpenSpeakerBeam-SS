import os
import torch
import torchaudio
import argparse
import soundfile as sf
from model import SpeakerBeamSS
from tools import get_speaker_embeddings_batch, load_ecapa_model


def _frame_enrollment_scores(
    signal: torch.Tensor,
    enroll_norm: torch.Tensor,
    speaker_encoder,
    device: torch.device,
    win_len: int,
    hop_len: int,
    n_frames: int,
):
    """フレームごとの ECAPA×enrollment 類似度と RMS を返す。"""
    scores, rms_list = [], []
    for i in range(n_frames):
        start = i * hop_len
        end = min(start + win_len, signal.shape[-1])
        frame = signal[start:end]
        rms_list.append(float(frame.pow(2).mean().sqrt().item()))
        if frame.abs().max() < 1e-5:
            scores.append(0.0)
            continue
        frame_input = frame.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            frame_emb = get_speaker_embeddings_batch(speaker_encoder, frame_input.to(device))
        frame_norm = torch.nn.functional.normalize(frame_emb, dim=-1)
        scores.append(float((frame_norm * enroll_norm).sum(dim=-1).item()))
    return scores, rms_list


def _adaptive_threshold(scores_arr, rms_arr, threshold: float, rms_threshold: float = 0.03):
    """有声フレーム 3/2 クラスタ k-means で適応閾値を決める。"""
    import numpy as np

    if len(scores_arr) < 4:
        return threshold

    voiced_mask = rms_arr >= rms_threshold
    voiced_scores = scores_arr[voiced_mask]

    if len(voiced_scores) >= 3:
        c_low = float(voiced_scores.min())
        c_high = float(voiced_scores.max())
        c_mid = float((c_low + c_high) / 2.0)
        for _ in range(100):
            dists = np.stack([
                np.abs(voiced_scores - c_low),
                np.abs(voiced_scores - c_mid),
                np.abs(voiced_scores - c_high),
            ], axis=1)
            labels = dists.argmin(axis=1)
            new_c_low = float(voiced_scores[labels == 0].mean()) if (labels == 0).any() else c_low
            new_c_mid = float(voiced_scores[labels == 1].mean()) if (labels == 1).any() else c_mid
            new_c_high = float(voiced_scores[labels == 2].mean()) if (labels == 2).any() else c_high
            if (abs(new_c_low - c_low) < 1e-6 and
                abs(new_c_mid - c_mid) < 1e-6 and
                abs(new_c_high - c_high) < 1e-6):
                break
            c_low, c_mid, c_high = new_c_low, new_c_mid, new_c_high
        return (c_mid + c_high) / 2.0, "3kmeans", c_low, c_mid, c_high, int(voiced_mask.sum())

    c1 = float(scores_arr.min())
    c2 = float(scores_arr.max())
    for _ in range(50):
        mid = (c1 + c2) / 2.0
        cl1 = scores_arr[scores_arr < mid]
        cl2 = scores_arr[scores_arr >= mid]
        new_c1 = float(cl1.mean()) if len(cl1) > 0 else c1
        new_c2 = float(cl2.mean()) if len(cl2) > 0 else c2
        if abs(new_c1 - c1) < 1e-6 and abs(new_c2 - c2) < 1e-6:
            break
        c1, c2 = new_c1, new_c2
    return (c1 + c2) / 2.0, "2kmeans", c1, c2, None, int(voiced_mask.sum())


def speaker_verification_filter(
    enhanced: torch.Tensor,
    enrollment_embedding: torch.Tensor,
    speaker_encoder,
    sample_rate: int = 16000,
    device: torch.device = torch.device("cpu"),
    mixture: torch.Tensor = None,
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
        mixture: (B, 1, T) 混合音声。指定時は mixture 側の話者照合も併用し、
                 混合音にターゲットがいない区間の誤抽出を抑制する。
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

    import numpy as np

    for b in range(B):
        signal = enhanced[b, 0]  # (T,)
        n_frames = max(1, (T - win_len) // hop_len + 1)

        scores, rms_list = _frame_enrollment_scores(
            signal, enroll_norm[b:b + 1], speaker_encoder, device, win_len, hop_len, n_frames,
        )
        scores_arr = np.array(scores, dtype=np.float32)
        rms_arr = np.array(rms_list, dtype=np.float32)

        out_thr, mode, *clusters, voiced_n = _adaptive_threshold(scores_arr, rms_arr, threshold)
        mix_thr = None
        mix_scores_arr = None
        if mixture is not None:
            mix_signal = mixture[b, 0]
            mix_scores, mix_rms = _frame_enrollment_scores(
                mix_signal, enroll_norm[b:b + 1], speaker_encoder, device, win_len, hop_len, n_frames,
            )
            mix_scores_arr = np.array(mix_scores, dtype=np.float32)
            mix_rms_arr = np.array(mix_rms, dtype=np.float32)
            mix_thr, mix_mode, *_, _ = _adaptive_threshold(mix_scores_arr, mix_rms_arr, threshold)

        if mode == "3kmeans":
            c_low, c_mid, c_high = clusters
            print(f"[Filter] b={b}: n_frames={len(scores)}, voiced={voiced_n}, "
                  f"3kmeans c_low={c_low:.4f} c_mid={c_mid:.4f} c_high={c_high:.4f}, "
                  f"out_threshold={out_thr:.4f}"
                  + (f", mix_threshold={mix_thr:.4f}" if mix_thr is not None else ""))
        else:
            c1, c2 = clusters
            print(f"[Filter] b={b}: n_frames={len(scores)}, 2kmeans fallback "
                  f"c1={c1:.4f} c2={c2:.4f}, out_threshold={out_thr:.4f}"
                  + (f", mix_threshold={mix_thr:.4f}" if mix_thr is not None else ""))

        # 出力がターゲットらしく、かつ mixture にもターゲットがいるフレームのみ通す
        gate_samples = torch.zeros(T, device=enhanced.device)
        for i, score in enumerate(scores):
            start = i * hop_len
            end = min(start + win_len, T)
            keep = score >= out_thr
            if mix_scores_arr is not None:
                keep = keep and mix_scores_arr[i] >= mix_thr
            if keep:
                gate_samples[start:end] = 1.0

        # 因果的ゲート平滑化（アタック即時・リリースのみ）
        # 対称畳み込みだと後続フレームの ON が過去（冒頭無音区間）へ漏れるため、
        # 過去→未来の一方向のみリリースをかける。
        if smooth_frames > 0:
            release_step = 1.0 / max(smooth_frames * hop_len, 1)
            gate_smooth = torch.zeros(T, device=enhanced.device)
            level = 0.0
            for t in range(T):
                target = gate_samples[t].item()
                if target > level:
                    level = target
                else:
                    level = max(target, level - release_step)
                gate_smooth[t] = level
        else:
            gate_smooth = gate_samples

        result[b, 0] = signal * gate_smooth

    return result


def _suppress_transients(
    signal: torch.Tensor,
    sample_rate: int,
    win_ms: float = 8.0,
    hop_ms: float = 4.0,
    crest_factor: float = 6.0,
    rms_floor: float = 0.012,
    attenuation: float = 0.0,
) -> torch.Tensor:
    """短いポツン音（低エネルギー背景上の鋭いピーク）を抑制する。"""
    x = signal.clone()
    win = max(1, int(sample_rate * win_ms / 1000))
    hop = max(1, int(sample_rate * hop_ms / 1000))
    T = x.shape[-1]
    for start in range(0, T - win + 1, hop):
        seg = x[start:start + win]
        rms = float(seg.pow(2).mean().sqrt().item())
        peak = float(seg.abs().max().item())
        if rms < rms_floor and peak / (rms + 1e-8) >= crest_factor:
            x[start:start + win] *= attenuation
    return x


def _suppress_mad_spikes(
    signal: torch.Tensor,
    sample_rate: int,
    end_sec: float = 4.0,
    win_ms: float = 20.0,
    hop_ms: float = 5.0,
    k: float = 4.0,
    max_rms: float = 0.045,
    attenuation: float = 0.05,
) -> torch.Tensor:
    """冒頭区間の局所MADベースでポツン・クリック状スパイクを抑制。"""
    x = signal.clone()
    end = min(int(end_sec * sample_rate), x.shape[-1])
    if end <= 0:
        return x
    win = max(1, int(sample_rate * win_ms / 1000))
    hop = max(1, int(sample_rate * hop_ms / 1000))
    absx = x[:end].abs()
    for start in range(0, end - win + 1, hop):
        seg = absx[start:start + win]
        rms = float((seg.pow(2).mean().sqrt().item()))
        if rms > max_rms:
            continue
        med = float(seg.median().item())
        mad = float((seg - med).abs().median().item())
        thresh = med + k * max(mad, 0.0015)
        mask = seg > thresh
        if mask.any():
            x[start:start + win][mask] *= attenuation
    return x


def refine_output_filter(
    enhanced: torch.Tensor,
    enrollment_embedding: torch.Tensor,
    speaker_encoder,
    sample_rate: int = 16000,
    device: torch.device = torch.device("cpu"),
    mixture: torch.Tensor = None,
    reject_enrollment: torch.Tensor = None,
    win_sec: float = 0.75,
    hop_sec: float = 0.125,
    margin_threshold: float = 0.08,
    focus_sec: float = 4.0,
    smooth_frames: int = 1,
) -> torch.Tensor:
    """
    分離済み音声のリファイン（nofilt 出力向け）。

    0〜focus_sec（混在焦点区間）ではソフト減衰を使わず、
    mixture・出力の両方で「男性優勢」と判定されたフレームだけをミュートする。
    女性が混在する区間は nofilt を極力維持する。
    """
    import numpy as np

    B, _, T = enhanced.shape
    win_len = int(win_sec * sample_rate)
    hop_len = int(hop_sec * sample_rate)
    enroll_norm = torch.nn.functional.normalize(enrollment_embedding, dim=-1)
    reject_norm = None
    if reject_enrollment is not None:
        reject_norm = torch.nn.functional.normalize(reject_enrollment, dim=-1)

    result = enhanced.clone()
    for b in range(B):
        signal = enhanced[b, 0]
        n_frames = max(1, (T - win_len) // hop_len + 1)

        out_s1, out_rms = _frame_enrollment_scores(
            signal, enroll_norm[b:b + 1], speaker_encoder, device, win_len, hop_len, n_frames,
        )
        out_s2 = mix_s1 = mix_s2 = None
        if reject_norm is not None:
            out_s2, _ = _frame_enrollment_scores(
                signal, reject_norm[b:b + 1], speaker_encoder, device, win_len, hop_len, n_frames,
            )
        if mixture is not None:
            mix_s1, _ = _frame_enrollment_scores(
                mixture[b, 0], enroll_norm[b:b + 1], speaker_encoder, device, win_len, hop_len, n_frames,
            )
            if reject_norm is not None:
                mix_s2, _ = _frame_enrollment_scores(
                    mixture[b, 0], reject_norm[b:b + 1], speaker_encoder, device, win_len, hop_len, n_frames,
                )

        gate_samples = torch.ones(T, device=enhanced.device)
        for i in range(n_frames):
            start = i * hop_len
            end = min(start + win_len, T)
            t_mid = (start + end) / (2 * sample_rate)
            gate = 1.0

            if out_s2 is not None:
                om = out_s1[i] - out_s2[i]
                mm = mix_s1[i] - mix_s2[i] if (mix_s1 is not None and mix_s2 is not None) else 0.0

                if t_mid < focus_sec:
                    # 混在焦点区間: mixture で男性優勢と判定されたフレームのみミュート
                    # （出力スコア単独では判定しない — 女性が薄い区間を誤って消すため）
                    male_in_mix = mm < 0.0
                    if male_in_mix and om < 0.05:
                        gate = 0.0
                    elif male_in_mix and mm < -0.08 and om < 0.10:
                        gate = 0.0
                else:
                    if om < margin_threshold:
                        gate = 0.0
                    elif om < margin_threshold + 0.05:
                        gate = 0.4
                    elif mm < 0.0 and om < 0.18:
                        gate = min(gate, 0.15)

            gate_samples[start:end] = torch.minimum(
                gate_samples[start:end],
                torch.tensor(gate, device=enhanced.device),
            )

        if smooth_frames > 0:
            release_step = 1.0 / max(smooth_frames * hop_len, 1)
            gate_smooth = torch.zeros(T, device=enhanced.device)
            level = 1.0
            for t in range(T):
                target = gate_samples[t].item()
                if target > level:
                    level = target
                else:
                    level = max(target, level - release_step)
                gate_smooth[t] = level
        else:
            gate_smooth = gate_samples

        result[b, 0] = signal * gate_smooth

        muted = int((gate_smooth < 0.5).sum().item())
        print(f"[Refine] b={b}: focus<{focus_sec:.0f}s selective-mute, "
              f"hop={hop_sec:.2f}s, muted_samples~={muted}")

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
    parser.add_argument("--no_filter", action="store_true",
                        help="Skip speaker verification post-filter")
    parser.add_argument("--refine_filter", action="store_true",
                        help="分離後リファイン（干渉話者・ポツン音抑制。--no_filter と併用推奨）")
    parser.add_argument("--reject_enrollment", type=str, default=None,
                        help="拒否話者 enrollment（干渉話者照合用、任意）")
    parser.add_argument("--filter_smooth_frames", type=int, default=2,
                        help="Release smoothing length in hop frames (0=hard gate)")
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
    if not args.no_filter:
        enhanced = speaker_verification_filter(
            enhanced, speaker_embeddings, speaker_encoder, args.sample_rate, device,
            mixture=mixture_waveform,
            smooth_frames=args.filter_smooth_frames,
        )

    if args.refine_filter:
        reject_embeddings = None
        if args.reject_enrollment:
            reject_waveform, sr_r = load_audio(args.reject_enrollment)
            if sr_r != args.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr_r, new_freq=args.sample_rate)
                reject_waveform = resampler(reject_waveform)
            if reject_waveform.shape[0] > 1:
                reject_waveform = torch.mean(reject_waveform, dim=0, keepdim=True)
            if reject_waveform.dim() == 2:
                reject_waveform = reject_waveform.unsqueeze(0)
            reject_waveform = reject_waveform.to(device)
            reject_embeddings = get_speaker_embeddings_batch(speaker_encoder, reject_waveform)

        enhanced = refine_output_filter(
            enhanced, speaker_embeddings, speaker_encoder, args.sample_rate, device,
            mixture=mixture_waveform,
            reject_enrollment=reject_embeddings,
        )

    # ----- 6. 推論結果を wav として保存 -----
    enhanced = enhanced.cpu()
    enhanced = enhanced.squeeze(0)
    save_audio(args.output, enhanced, args.sample_rate)
    print(f"Inference complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()
