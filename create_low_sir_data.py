"""
create_low_sir_data.py

既存の (mixture, enrollment, target) トリプルから、
干渉話者成分を再スケールして低 SIR（強い同時発話）の mixture を追加生成する。

  residual = mixture_orig - target_orig   # 干渉 + ノイズ
  mixture_new = target + scale(residual)  # 指定 SIR [dB] に合わせてスケール

均一 SIR に加え、区間可変 SIR（例: 0〜2秒は -12〜-8 dB、2〜10秒は -5〜+2 dB）も生成できる。
sample 001 のような「冒頭だけ極端に低 SIR」パターンを本流学習データに投入する用途。

ターゲット・ enrollment は既存ファイルをそのまま流用する。
"""
import os
import random
import argparse
import pandas as pd
import numpy as np
import soundfile as sf


SEGMENT_LENGTH = 16000 * 10  # 10秒固定


def load_wav(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 16000:
        raise RuntimeError(f"Expected 16kHz, got {sr}Hz: {path}")
    return wav.mean(axis=1)


def save_wav(path: str, wav: np.ndarray, sr: int = 16000):
    sf.write(path, wav, sr, subtype="PCM_16")


def measure_sir_db(target: np.ndarray, residual: np.ndarray, eps: float = 1e-8) -> float:
    pt = np.mean(target ** 2)
    pr = np.mean(residual ** 2)
    return 10.0 * np.log10(pt / (pr + eps) + eps)


def rescale_residual_to_sir(target: np.ndarray, residual: np.ndarray, sir_db: float) -> np.ndarray:
    """target と residual から、指定 SIR になる mixture を返す。"""
    pt = np.mean(target ** 2)
    pr = np.mean(residual ** 2)
    if pr < 1e-12:
        return target.copy()
    target_pr = pt / (10 ** (sir_db / 10))
    scale = np.sqrt(target_pr / (pr + 1e-8))
    return target + residual * scale


def build_zone_mixture(
    target: np.ndarray,
    residual: np.ndarray,
    sir_head: float,
    sir_tail: float,
    zone_split: int,
) -> np.ndarray:
    """先頭区間と残りで別 SIR の mixture を構築する。"""
    out = target.copy()
    head_t, head_r = target[:zone_split], residual[:zone_split]
    tail_t, tail_r = target[zone_split:], residual[zone_split:]
    out[:zone_split] = rescale_residual_to_sir(head_t, head_r, sir_head)
    out[zone_split:] = rescale_residual_to_sir(tail_t, tail_r, sir_tail)
    peak = np.abs(out).max()
    if peak > 1.0:
        out /= peak
    return out


def sample_sir_db(
    sir_min: float,
    sir_max: float,
    focus_min: float,
    focus_max: float,
    focus_prob: float,
) -> float:
    """
    SIR [dB] をサンプリングする。
    focus_prob の確率で [focus_min, focus_max]、残りはその区間外を一様サンプリング。
    """
    if focus_prob <= 0.0 or focus_min >= focus_max:
        return random.uniform(sir_min, sir_max)

    if random.random() < focus_prob:
        return random.uniform(focus_min, focus_max)

    low_w = max(0.0, focus_min - sir_min)
    high_w = max(0.0, sir_max - focus_max)
    if low_w + high_w <= 1e-8:
        return random.uniform(focus_min, focus_max)
    if random.random() < low_w / (low_w + high_w):
        return random.uniform(sir_min, focus_min)
    return random.uniform(focus_max, sir_max)


def next_file_index(df: pd.DataFrame) -> int:
    indices = set()
    for p in df["mixture_path"]:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            indices.add(int(stem.split("_")[-1]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate low-SIR mixture augmentations from existing training triples",
    )
    parser.add_argument("--train_csv", default="data_csv/train/metadata.csv")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--out_mixture_dir", default="data/mixtures")
    parser.add_argument("--out_csv", default="data_csv/train/metadata_low_sir.csv")
    parser.add_argument("--sir_min", type=float, default=-15.0,
                        help="Minimum target SIR [dB]")
    parser.add_argument("--sir_max", type=float, default=5.0,
                        help="Maximum target SIR [dB]")
    parser.add_argument("--sir_focus_min", type=float, default=-12.0,
                        help="Lower bound of emphasized SIR band [dB]")
    parser.add_argument("--sir_focus_max", type=float, default=-8.0,
                        help="Upper bound of emphasized SIR band [dB]")
    parser.add_argument("--sir_focus_prob", type=float, default=0.6,
                        help="Probability of sampling from the focus band")
    parser.add_argument("--max_orig_sir", type=float, default=None,
                        help="Only use source triples whose original SIR <= this value [dB]")
    parser.add_argument("--zone_num_samples", type=int, default=0,
                        help="Number of zone-varying SIR augmentations (0=disabled)")
    parser.add_argument("--zone_split_sec", type=float, default=2.0,
                        help="Head zone length [s] for zone-varying SIR")
    parser.add_argument("--zone_head_sir_min", type=float, default=-13.0)
    parser.add_argument("--zone_head_sir_max", type=float, default=-8.0)
    parser.add_argument("--zone_head_focus_min", type=float, default=-12.0,
                        help="Emphasized head SIR band lower bound [dB]")
    parser.add_argument("--zone_head_focus_max", type=float, default=-9.0,
                        help="Emphasized head SIR band upper bound [dB]")
    parser.add_argument("--zone_head_focus_prob", type=float, default=0.7,
                        help="Probability of sampling head SIR from focus band")
    parser.add_argument("--zone_tail_sir_min", type=float, default=-5.0)
    parser.add_argument("--zone_tail_sir_max", type=float, default=2.0)
    parser.add_argument("--zone_tail_focus_min", type=float, default=-3.0)
    parser.add_argument("--zone_tail_focus_max", type=float, default=1.0)
    parser.add_argument("--zone_tail_focus_prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    df_orig = pd.read_csv(args.train_csv)
    print(f"既存 train CSV: {len(df_orig)} 件")
    print(f"SIR 範囲: [{args.sir_min}, {args.sir_max}] dB")
    print(f"重点帯: [{args.sir_focus_min}, {args.sir_focus_max}] dB "
          f"(prob={args.sir_focus_prob:.0%})")
    if args.zone_num_samples > 0:
        print(f"区間可変 SIR: {args.zone_num_samples} 件")
        print(f"  head [0, {args.zone_split_sec}s]: "
              f"[{args.zone_head_sir_min}, {args.zone_head_sir_max}] dB "
              f"(focus [{args.zone_head_focus_min}, {args.zone_head_focus_max}] "
              f"prob={args.zone_head_focus_prob:.0%})")
        print(f"  tail [{args.zone_split_sec}, 10s]: "
              f"[{args.zone_tail_sir_min}, {args.zone_tail_sir_max}] dB")

    if args.num_samples <= 0 and args.zone_num_samples <= 0:
        raise ValueError("num_samples か zone_num_samples のいずれかを 1 以上にしてください。")

    os.makedirs(args.out_mixture_dir, exist_ok=True)
    next_idx = next_file_index(df_orig)
    zone_split = int(args.zone_split_sec * 16000)
    if zone_split <= 0 or zone_split >= SEGMENT_LENGTH:
        raise ValueError(f"zone_split_sec must be in (0, 10), got {args.zone_split_sec}")

    new_rows = []
    sampled_sirs = []
    zone_head_sirs = []
    zone_tail_sirs = []
    generated = 0
    attempts = 0
    max_attempts = max(args.num_samples, 1) * 10

    while generated < args.num_samples and attempts < max_attempts:
        attempts += 1
        row = df_orig.sample(1).iloc[0]
        mix_path = row["mixture_path"]
        enr_path = row["enrollment_path"]
        tgt_path = row["target_path"]

        if not all(os.path.exists(p) for p in (mix_path, tgt_path, enr_path)):
            continue

        try:
            mix_wav = load_wav(mix_path)
            tgt_wav = load_wav(tgt_path)
        except Exception as e:
            print(f"  skip: {e}")
            continue

        if len(mix_wav) < SEGMENT_LENGTH or len(tgt_wav) < SEGMENT_LENGTH:
            continue

        mix_wav = mix_wav[:SEGMENT_LENGTH]
        tgt_wav = tgt_wav[:SEGMENT_LENGTH]
        residual = mix_wav - tgt_wav

        if np.mean(residual ** 2) < 1e-10:
            continue

        orig_sir = measure_sir_db(tgt_wav, residual)
        if args.max_orig_sir is not None and orig_sir > args.max_orig_sir:
            continue

        sir_db = sample_sir_db(
            args.sir_min, args.sir_max,
            args.sir_focus_min, args.sir_focus_max,
            args.sir_focus_prob,
        )
        new_mix = rescale_residual_to_sir(tgt_wav, residual, sir_db)

        peak = np.abs(new_mix).max()
        if peak > 1.0:
            new_mix = new_mix / peak

        idx_str = f"{next_idx:06d}"
        new_mix_fname = f"low_sir_{idx_str}.wav"
        new_mix_path = os.path.join(args.out_mixture_dir, new_mix_fname)

        try:
            save_wav(new_mix_path, new_mix)
        except Exception as e:
            print(f"  save error: {e}")
            continue

        new_rows.append({
            "mixture_path": new_mix_path,
            "enrollment_path": enr_path,
            "target_path": tgt_path,
        })
        sampled_sirs.append(sir_db)
        next_idx += 1
        generated += 1

        if generated % 500 == 0 and args.num_samples > 0:
            print(f"  uniform: {generated}/{args.num_samples} generated...")

    zone_generated = 0
    zone_attempts = 0
    max_zone_attempts = max(args.zone_num_samples, 1) * 10

    while zone_generated < args.zone_num_samples and zone_attempts < max_zone_attempts:
        zone_attempts += 1
        row = df_orig.sample(1).iloc[0]
        mix_path = row["mixture_path"]
        enr_path = row["enrollment_path"]
        tgt_path = row["target_path"]

        if not all(os.path.exists(p) for p in (mix_path, tgt_path, enr_path)):
            continue

        try:
            mix_wav = load_wav(mix_path)
            tgt_wav = load_wav(tgt_path)
        except Exception as e:
            print(f"  zone skip: {e}")
            continue

        if len(mix_wav) < SEGMENT_LENGTH or len(tgt_wav) < SEGMENT_LENGTH:
            continue

        mix_wav = mix_wav[:SEGMENT_LENGTH]
        tgt_wav = tgt_wav[:SEGMENT_LENGTH]
        residual = mix_wav - tgt_wav

        if np.mean(residual ** 2) < 1e-10:
            continue

        head_r = residual[:zone_split]
        tail_r = residual[zone_split:]
        if np.mean(head_r ** 2) < 1e-10 or np.mean(tail_r ** 2) < 1e-10:
            continue

        sir_head = sample_sir_db(
            args.zone_head_sir_min, args.zone_head_sir_max,
            args.zone_head_focus_min, args.zone_head_focus_max,
            args.zone_head_focus_prob,
        )
        sir_tail = sample_sir_db(
            args.zone_tail_sir_min, args.zone_tail_sir_max,
            args.zone_tail_focus_min, args.zone_tail_focus_max,
            args.zone_tail_focus_prob,
        )
        new_mix = build_zone_mixture(tgt_wav, residual, sir_head, sir_tail, zone_split)

        idx_str = f"{next_idx:06d}"
        new_mix_fname = f"low_sir_zone_{idx_str}.wav"
        new_mix_path = os.path.join(args.out_mixture_dir, new_mix_fname)

        try:
            save_wav(new_mix_path, new_mix)
        except Exception as e:
            print(f"  zone save error: {e}")
            continue

        new_rows.append({
            "mixture_path": new_mix_path,
            "enrollment_path": enr_path,
            "target_path": tgt_path,
        })
        zone_head_sirs.append(sir_head)
        zone_tail_sirs.append(sir_tail)
        next_idx += 1
        zone_generated += 1

        if zone_generated % 500 == 0:
            print(f"  zone: {zone_generated}/{args.zone_num_samples} generated...")

    total_new = generated + zone_generated
    if total_new == 0:
        raise RuntimeError("No samples generated. Check paths and filters.")

    if generated > 0:
        sir_arr = np.array(sampled_sirs)
        in_focus = ((sir_arr >= args.sir_focus_min) & (sir_arr <= args.sir_focus_max)).mean()
        print(f"\n均一 SIR 生成: {generated} 件")
        print(f"  SIR mean={sir_arr.mean():.2f} dB, median={np.median(sir_arr):.2f} dB, "
              f"min={sir_arr.min():.2f}, max={sir_arr.max():.2f}")
        print(f"  重点帯 [{args.sir_focus_min}, {args.sir_focus_max}] の割合: {in_focus:.1%}")

    if zone_generated > 0:
        head_arr = np.array(zone_head_sirs)
        tail_arr = np.array(zone_tail_sirs)
        head_focus = ((head_arr >= args.zone_head_focus_min) &
                      (head_arr <= args.zone_head_focus_max)).mean()
        print(f"\n区間可変 SIR 生成: {zone_generated} 件")
        print(f"  head SIR mean={head_arr.mean():.2f} dB, median={np.median(head_arr):.2f} dB")
        print(f"  tail SIR mean={tail_arr.mean():.2f} dB, median={np.median(tail_arr):.2f} dB")
        print(f"  head 重点帯 [{args.zone_head_focus_min}, {args.zone_head_focus_max}] "
              f"の割合: {head_focus:.1%}")

    print(f"\n合計新規: {total_new} 件（均一 {generated} + zone {zone_generated}）")

    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_orig, df_new], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df_combined.to_csv(args.out_csv, index=False)
    print(f"CSV保存: {args.out_csv} → {len(df_combined)} 件（既存{len(df_orig)}+新規{len(df_new)}）")


if __name__ == "__main__":
    main()
