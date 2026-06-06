"""
既存の mixture/target ペアから「ターゲット話者不在区間」を持つサンプルを生成する。

問題の根本原因:
  現在の学習データは mixture 10 秒全区間にターゲット話者が存在する前提で生成されている。
  実際の音声では冒頭・末尾にターゲット不在区間があり、その区間でサブ話者を出力する誤動作が起きる。

修正方針:
  - 既存の data/target/*.wav の前半/後半/全体をゼロにする
  - data/mixtures/*.wav はそのまま使う（干渉話者の音声は全区間存在するまま）
  - これで「ターゲット不在フレームの正解 = ゼロ」を学習させる

生成パターン（各1/3ずつ）:
  (a) onset:  冒頭 1〜5 秒がゼロ → 001 の問題に対応
  (b) offset: 末尾 1〜5 秒がゼロ → 002 の問題に対応
  (c) full:   全区間ゼロ（完全不在）→ 汎化用

使い方:
  python create_absent_target_data.py \
    --source_csv tmp/train_fast5k.csv \
    --output_dir data/absent_target \
    --num_samples 10000 \
    --output_csv tmp/absent_target_10k.csv
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import soundfile as sf

SEGMENT_LENGTH = 16000 * 10  # 10秒


def create_absent_samples(source_csv: str, output_dir: str, num_samples: int,
                          output_csv: str, seed: int = 42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(source_csv)
    total = len(df)
    if total == 0:
        raise ValueError(f"CSVが空です: {source_csv}")

    rows = []
    generated = 0
    attempts = 0
    max_attempts = num_samples * 3

    print(f"ソースサンプル数: {total}")
    print(f"生成目標: {num_samples} サンプル")
    print(f"出力ディレクトリ: {output_dir}")

    while generated < num_samples and attempts < max_attempts:
        attempts += 1
        row = df.iloc[random.randint(0, total - 1)]
        target_path = row["target_path"]
        mix_path = row["mixture_path"]
        enroll_path = row["enrollment_path"]

        if not os.path.exists(target_path) or not os.path.exists(mix_path):
            continue

        try:
            target_np, sr = sf.read(target_path, dtype='float32')
        except Exception as e:
            print(f"  読み込み失敗: {target_path}: {e}")
            continue

        # ステレオの場合はモノラルに変換
        if target_np.ndim == 2:
            target_np = target_np.mean(axis=1)
        target_np = target_np.copy()
        T = len(target_np)

        # パターンをランダムに選択（fullは除外: 全区間ゼロにすると「常に無音」を過学習するため）
        pattern = random.choice(["onset", "offset"])

        if pattern == "onset":
            # 冒頭 1〜5 秒をゼロ
            absent_sec = random.uniform(1.0, 5.0)
            absent_samples = min(int(absent_sec * 16000), T - 1600)
            target_np[:absent_samples] = 0.0
            label = f"onset{absent_sec:.1f}s"

        elif pattern == "offset":
            # 末尾 1〜5 秒をゼロ
            absent_sec = random.uniform(1.0, 5.0)
            absent_samples = min(int(absent_sec * 16000), T - 1600)
            target_np[-absent_samples:] = 0.0
            label = f"offset{absent_sec:.1f}s"

        else:  # full
            # 全区間ゼロ（ターゲット完全不在）
            target_np[:] = 0.0
            label = "full"

        # 変形後の target を保存
        out_fname = f"absent_{generated:06d}_{label}.wav"
        out_path = os.path.join(output_dir, out_fname)
        sf.write(out_path, target_np, 16000)

        rows.append({
            "mixture_path": mix_path,
            "enrollment_path": enroll_path,
            "target_path": out_path,
        })

        generated += 1
        if generated % 500 == 0:
            print(f"  {generated}/{num_samples} 生成完了")

    if generated < num_samples:
        print(f"警告: {generated} サンプルのみ生成（目標: {num_samples}）")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f"\n完了: {generated} サンプル生成")
    print(f"CSV: {output_csv}")

    # パターン別の統計
    patterns = {"onset": 0, "offset": 0}
    for r in rows:
        fname = os.path.basename(r["target_path"])
        for p in patterns:
            if p in fname:
                patterns[p] += 1
                break
    print(f"パターン内訳: {patterns}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", type=str, default="tmp/train_fast5k.csv")
    parser.add_argument("--output_dir", type=str, default="data/absent_target")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--output_csv", type=str, default="tmp/absent_target_10k.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_absent_samples(
        source_csv=args.source_csv,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        output_csv=args.output_csv,
        seed=args.seed,
    )
