"""
sample mixture_000001 の干渉成分から、001 専用 fine-tune 用データを生成する。

sample mixture は学習用 mixture_000001 と別物で、0〜2秒 SIR ≈ -10 dB が特徴。
区間別 SIR（0〜2秒 / 2〜10秒）で residual を再スケールし、バリエーションを作る。
"""
import os
import random
import argparse
import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
SEG_LEN = SR * 10
ZONE_SPLIT = SR * 2  # 0〜2秒


def load_mono(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        raise RuntimeError(f"Expected 16kHz: {path}")
    w = wav.mean(axis=1)
    if len(w) < SEG_LEN:
        raise RuntimeError(f"Too short ({len(w)} samples): {path}")
    return w[:SEG_LEN]


def measure_sir_db(target: np.ndarray, residual: np.ndarray, eps: float = 1e-8) -> float:
    pt = np.mean(target ** 2)
    pr = np.mean(residual ** 2)
    return 10.0 * np.log10(pt / (pr + eps) + eps)


def rescale_zone(target: np.ndarray, residual: np.ndarray, sir_db: float) -> np.ndarray:
    pt = np.mean(target ** 2)
    pr = np.mean(residual ** 2)
    if pr < 1e-12:
        return target.copy()
    target_pr = pt / (10 ** (sir_db / 10))
    scale = np.sqrt(target_pr / (pr + 1e-8))
    return target + residual * scale


def build_zone_mixture(target: np.ndarray, residual: np.ndarray, sir_head: float, sir_tail: float) -> np.ndarray:
    out = target.copy()
    head_t, head_r = target[:ZONE_SPLIT], residual[:ZONE_SPLIT]
    tail_t, tail_r = target[ZONE_SPLIT:], residual[ZONE_SPLIT:]
    out[:ZONE_SPLIT] = rescale_zone(head_t, head_r, sir_head)
    out[ZONE_SPLIT:] = rescale_zone(tail_t, tail_r, sir_tail)
    peak = np.abs(out).max()
    if peak > 1.0:
        out /= peak
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate sample-001 dedicated fine-tune data")
    parser.add_argument("--mixture", default="data/sample/mixture_000001.wav")
    parser.add_argument("--target", default="data/target/target_000001.wav")
    parser.add_argument("--enrollment", default="data/sample/enrollment_000001.wav")
    parser.add_argument("--out_dir", default="data/sample_finetune")
    parser.add_argument("--out_csv", default="data_csv/train/metadata_sample_001.csv")
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--head_sir_min", type=float, default=-13.0)
    parser.add_argument("--head_sir_max", type=float, default=-8.0)
    parser.add_argument("--tail_sir_min", type=float, default=-5.0)
    parser.add_argument("--tail_sir_max", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    target = load_mono(args.target)
    mix_sample = load_mono(args.mixture)
    residual = mix_sample - target

    rows = []
    head_sirs, tail_sirs = [], []

    # 実サンプルそのもの（SIR 基準）
    ref_path = os.path.join(args.out_dir, "mixture_000000.wav")
    sf.write(ref_path, mix_sample, SR, subtype="PCM_16")
    rows.append({
        "mixture_path": ref_path,
        "enrollment_path": args.enrollment,
        "target_path": args.target,
    })
    head_sirs.append(measure_sir_db(target[:ZONE_SPLIT], residual[:ZONE_SPLIT]))
    tail_sirs.append(measure_sir_db(target[ZONE_SPLIT:], residual[ZONE_SPLIT:]))

    for i in range(1, args.num_samples):
        sir_head = random.uniform(args.head_sir_min, args.head_sir_max)
        sir_tail = random.uniform(args.tail_sir_min, args.tail_sir_max)
        new_mix = build_zone_mixture(target, residual, sir_head, sir_tail)
        path = os.path.join(args.out_dir, f"mixture_{i:06d}.wav")
        sf.write(path, new_mix, SR, subtype="PCM_16")
        rows.append({
            "mixture_path": path,
            "enrollment_path": args.enrollment,
            "target_path": args.target,
        })
        head_sirs.append(sir_head)
        tail_sirs.append(sir_tail)

    df = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)

    print(f"Generated {len(rows)} samples -> {args.out_csv}")
    print(f"  ref 0-2s SIR: {head_sirs[0]:.2f} dB, 2-10s SIR: {tail_sirs[0]:.2f} dB")
    print(f"  aug head SIR mean: {np.mean(head_sirs[1:]):.2f} dB")
    print(f"  aug tail SIR mean: {np.mean(tail_sirs[1:]):.2f} dB")


if __name__ == "__main__":
    main()
