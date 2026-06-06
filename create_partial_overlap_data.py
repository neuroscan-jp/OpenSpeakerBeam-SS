"""
create_partial_overlap_data.py

既存の data/mixtures, data/enrollment, data/target を再利用して、
「ターゲット話者が途中で沈黙するパターン」の部分重複ミックスを生成し、
data_csv/train/metadata.csv に追記する。

生成ルール:
  - 既存の (mixture, enrollment, target) トリプルをランダム選択
  - target を [0, T/2] のどこかで無音にカット（前半のみ or 後半のみ）
  - mixture = カットされた target + interference（residual = mixture_orig - target_orig で近似）
  - 新 mixture = partial_target + (mixture_orig - target_orig) [= 干渉話者分]
  - 保存先: data/mixtures/partial_XXXXXX.wav 等
"""
import os
import random
import argparse
import pandas as pd
import numpy as np
import soundfile as sf


SEGMENT_LENGTH = 16000 * 10  # 10秒固定


def load_wav(path: str) -> np.ndarray:
    """モノラル float32 (T,) で読み込む"""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 16000:
        raise RuntimeError(f"Expected 16kHz, got {sr}Hz: {path}")
    # モノラル化
    return wav.mean(axis=1)


def save_wav(path: str, wav: np.ndarray, sr: int = 16000):
    sf.write(path, wav, sr, subtype="PCM_16")


def apply_partial_silence(wav: np.ndarray, mode: str, silence_ratio: float) -> np.ndarray:
    """
    wav (T,) のうち一部を無音にして返す。
    mode: 'tail' → 後半を無音に（ターゲットが途中で終わる）
          'head' → 前半を無音に（ターゲットが遅れて入る）
    silence_ratio: 0.3〜0.7（無音にする割合）
    """
    T = len(wav)
    silence_len = int(T * silence_ratio)
    out = wav.copy()
    if mode == "tail":
        out[T - silence_len:] = 0.0
    else:  # head
        out[:silence_len] = 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data_csv/train/metadata.csv")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="生成する部分重複サンプル数")
    parser.add_argument("--out_mixture_dir", default="data/mixtures")
    parser.add_argument("--out_target_dir", default="data/target")
    parser.add_argument("--out_csv", default="data_csv/train/metadata_v4.csv",
                        help="出力先CSV（既存CSVとは別に保存。後でマージして使う）")
    parser.add_argument("--silence_ratio_min", type=float, default=0.3)
    parser.add_argument("--silence_ratio_max", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 既存CSV読み込み
    df_orig = pd.read_csv(args.train_csv)
    print(f"既存 train CSV: {len(df_orig)} 件")

    os.makedirs(args.out_mixture_dir, exist_ok=True)
    os.makedirs(args.out_target_dir, exist_ok=True)

    # 既存インデックスの最大値を取得して連番を決める
    existing_indices = set()
    for p in df_orig["mixture_path"]:
        fname = os.path.basename(p)
        stem = os.path.splitext(fname)[0]  # e.g. "mixture_009926" or "partial_000001"
        try:
            idx = int(stem.split("_")[-1])
            existing_indices.add(idx)
        except ValueError:
            pass
    next_idx = max(existing_indices) + 1 if existing_indices else 0

    new_rows = []
    generated = 0
    attempts = 0
    max_attempts = args.num_samples * 5

    while generated < args.num_samples and attempts < max_attempts:
        attempts += 1

        # ランダムに既存トリプルを選択
        row = df_orig.sample(1).iloc[0]
        mix_path = row["mixture_path"]
        enr_path = row["enrollment_path"]
        tgt_path = row["target_path"]

        if not (os.path.exists(mix_path) and os.path.exists(tgt_path) and os.path.exists(enr_path)):
            continue

        try:
            mix_wav = load_wav(mix_path)
            tgt_wav = load_wav(tgt_path)
        except Exception as e:
            print(f"  skip: {e}")
            continue

        T = SEGMENT_LENGTH
        if len(mix_wav) < T or len(tgt_wav) < T:
            continue

        mix_wav = mix_wav[:T]
        tgt_wav = tgt_wav[:T]

        # 干渉話者成分 = mixture - target（近似）
        interference_wav = mix_wav - tgt_wav

        # ターゲットに部分無音を適用
        mode = random.choice(["tail", "head"])
        ratio = random.uniform(args.silence_ratio_min, args.silence_ratio_max)
        partial_tgt = apply_partial_silence(tgt_wav, mode, ratio)

        # 新しいミックス = 部分ターゲット + 干渉話者成分
        new_mix = partial_tgt + interference_wav

        # クリッピング防止
        peak = np.abs(new_mix).max()
        if peak > 1.0:
            new_mix = new_mix / peak
            partial_tgt = partial_tgt / peak

        # ファイル保存
        idx_str = f"{next_idx:06d}"
        new_mix_fname = f"partial_{idx_str}.wav"
        new_tgt_fname = f"partial_target_{idx_str}.wav"

        new_mix_path = os.path.join(args.out_mixture_dir, new_mix_fname)
        new_tgt_path = os.path.join(args.out_target_dir, new_tgt_fname)

        try:
            save_wav(new_mix_path, new_mix)
            save_wav(new_tgt_path, partial_tgt)
        except Exception as e:
            print(f"  save error: {e}")
            continue

        new_rows.append({
            "mixture_path": new_mix_path,
            "enrollment_path": enr_path,  # enrollmentは既存のものを流用
            "target_path": new_tgt_path,
        })

        next_idx += 1
        generated += 1

        if generated % 500 == 0:
            print(f"  {generated}/{args.num_samples} generated...")

    print(f"\n生成完了: {generated} 件")

    # 既存CSVと新規データを結合して別ファイルに保存（既存CSVは変更しない）
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_orig, df_new], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)  # シャッフル
    df_combined.to_csv(args.out_csv, index=False)
    print(f"CSV保存: {args.out_csv} → {len(df_combined)} 件（既存{len(df_orig)}+新規{len(df_new)}）")


if __name__ == "__main__":
    main()
