import os
import sys
import time
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from tqdm import tqdm
from model import SpeakerBeamSS
from tools import get_speaker_embedding, load_ecapa_model

_log_file = None

def log(msg):
    """標準出力とログファイルの両方に出力する。"""
    print(msg, flush=True)
    if _log_file is not None:
        _log_file.write(msg + "\n")
        _log_file.flush()


def load_audio(path, target_sample_rate=16000):
    waveform_np, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform_np.T.copy())
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
        sample_rate = target_sample_rate
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def collate_speech_batch(batch):
    mixtures, enrollment_paths, targets = zip(*batch)

    def _pad_waveforms(waveforms):
        padded = pad_sequence([waveform.squeeze(0) for waveform in waveforms], batch_first=True)
        return padded.unsqueeze(1)

    return (
        _pad_waveforms(mixtures),
        _pad_waveforms(targets),
        list(enrollment_paths),
    )


def get_cached_speaker_embeddings(enrollment_paths, cache_dir, device, expected_dim=192):
    """キャッシュ済み埋め込みをロードして返す（事前計算済み前提）。"""
    embeddings = []
    for enrollment_path in enrollment_paths:
        cache_key = hashlib.sha1(os.path.normpath(enrollment_path).encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{cache_key}.npy")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Embedding cache not found: {cache_path}. Run precompute first.")
        embedding = torch.from_numpy(np.load(cache_path)).float()
        embeddings.append(embedding)
    return torch.stack(embeddings, dim=0).to(device)


def precompute_all_embeddings(csv_files, cache_dir, speaker_encoder, device, expected_dim=192):
    """全CSVのenrollment音声の埋め込みを事前計算してキャッシュに保存する。
    これにより学習ループ中にSpeechBrainを呼び出す必要がなくなる。"""
    os.makedirs(cache_dir, exist_ok=True)

    all_enrollment_paths = set()
    for csv_file in csv_files:
        if csv_file and os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            all_enrollment_paths.update(df["enrollment_path"].tolist())

    total = len(all_enrollment_paths)
    log(f"Pre-computing speaker embeddings for {total} unique enrollment files...")

    with tqdm(sorted(all_enrollment_paths), total=total, desc="Embeddings", unit="file", dynamic_ncols=True) as pbar:
        for enrollment_path in pbar:
            cache_key = hashlib.sha1(os.path.normpath(enrollment_path).encode("utf-8")).hexdigest()
            cache_path = os.path.join(cache_dir, f"{cache_key}.npy")

            # 既存キャッシュの次元チェック（不一致なら再計算して上書き）
            if os.path.exists(cache_path):
                cached = np.load(cache_path)
                if cached.shape[-1] == expected_dim:
                    continue

            try:
                waveform, _ = load_audio(enrollment_path)
                embedding_np = get_speaker_embedding(speaker_encoder, waveform)
                np.save(cache_path, embedding_np)
            except Exception as e:
                pbar.write(f"  Warning: Failed for {enrollment_path}: {e}")

    log("Pre-computation complete.\n")


# ========================================
# 1. SI-SNR loss 関数
# ========================================
def si_snr_loss(s, s_hat, eps=1e-8):
    """
    SI-SNR loss を計算する関数。

    Args:
        s (Tensor): 正解音声 (B, T)
        s_hat (Tensor): 推定音声 (B, T)
        eps (float): 数値安定性のための微小値
    Returns:
        loss (Tensor): 平均の負の SI-SNR (dB) 値
    """
    # 各サンプルごとに平均を引いてゼロ平均化
    s = s - torch.mean(s, dim=1, keepdim=True)
    s_hat = s_hat - torch.mean(s_hat, dim=1, keepdim=True)

    # 正解信号への射影（スケール不変）
    s_target = torch.sum(s_hat * s, dim=1, keepdim=True) / (torch.sum(s * s, dim=1, keepdim=True) + eps) * s
    e_noise = s_hat - s_target
    ratio = torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps)
    # SI-SNR [dB]
    si_snr = 10 * torch.log10(ratio + eps)

    # 損失は負の SI-SNR の平均（最大化が目的なので最小化問題に変換）
    loss = -torch.mean(si_snr)
    return loss


def spectral_loss(s, s_hat, n_fft=512, hop_length=128, eps=1e-8):
    """
    STFT対数振幅L1損失。

    女性音声の高周波数域（2kHz以上）の再現精度を高めるための補助損失。
    SI-SNRが捉えにくい倍音構造・声質をスペクトル領域で監視する。
    """
    window = torch.hann_window(n_fft, device=s.device)
    S = torch.stft(s, n_fft=n_fft, hop_length=hop_length,
                   window=window, return_complex=True)
    S_hat = torch.stft(s_hat, n_fft=n_fft, hop_length=hop_length,
                       window=window, return_complex=True)
    loss = torch.mean(torch.abs(torch.log(S.abs() + eps) - torch.log(S_hat.abs() + eps)))
    return loss


def sdr_loss(s, s_hat, eps=1e-8):
    """
    SDR (Signal-to-Distortion Ratio) 損失。

    SI-SNR と異なりスケール不変でないため、出力レベルがターゲットと
    ずれる場合もペナルティを与える。残留干渉を直接最小化する効果。

    Args:
        s (Tensor): 正解音声 (B, T)
        s_hat (Tensor): 推定音声 (B, T)
    Returns:
        loss (Tensor): 負の SDR の平均 [dB]
    """
    s = s - s.mean(dim=-1, keepdim=True)
    s_hat = s_hat - s_hat.mean(dim=-1, keepdim=True)
    target_energy = (s ** 2).sum(dim=-1)
    distortion = s_hat - s
    distortion_energy = (distortion ** 2).sum(dim=-1)
    sdr = 10 * torch.log10(target_energy / (distortion_energy + eps) + eps)
    return -sdr.mean()


def _voiced_mask(s, frame_len=1600, eps=1e-8):
    """ターゲット有声区間のサンプルマスク (B, T)。"""
    B, T = s.shape
    if T < frame_len:
        return torch.ones(B, T, device=s.device)
    hop = frame_len // 2
    s_frames = s.unfold(-1, frame_len, hop)
    frame_energy = s_frames.pow(2).mean(-1)
    peak_energy = frame_energy.amax(dim=-1, keepdim=True).clamp(min=eps)
    voiced_frame_mask = (frame_energy >= peak_energy * 0.01).float()
    voiced_mask = torch.zeros(B, T, device=s.device)
    for i in range(voiced_frame_mask.shape[1]):
        start = i * hop
        end = min(start + frame_len, T)
        voiced_mask[:, start:end] = torch.max(
            voiced_mask[:, start:end],
            voiced_frame_mask[:, i:i + 1].expand(-1, end - start),
        )
    return voiced_mask


def interference_suppression_loss_per_sample(s, s_hat, eps=1e-8, frame_len=1600):
    """
    残留干渉直接ペナルティ（有声マスク付き）をサンプルごとに返す。

    Returns:
        Tensor: (B,) 各サンプルの正規化残留干渉エネルギー
    """
    residual = s_hat - s
    target_peak = s.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    residual_norm = residual / target_peak
    voiced_mask = _voiced_mask(s, frame_len=frame_len, eps=eps)
    residual_norm = residual_norm * voiced_mask
    n_voiced = voiced_mask.sum(dim=-1).clamp(min=1.0)
    return residual_norm.pow(2).sum(dim=-1) / n_voiced


def interference_suppression_loss(s, s_hat, eps=1e-8, frame_len=1600):
    """
    残留干渉直接ペナルティ損失（有声マスク付き）。

    出力とターゲットの差（= 残留干渉）をターゲットのピークで正規化して最小化。
    SI-SNR ・ SDR が対数スケールの頭打ちで感度が下がる高 SNR 域でも
    線形スケールで干渉を押し込み続ける。

    ターゲットが無音の区間（partial overlap等）ではresidualをマスクし、
    有声区間のみで損失を計算する。無音区間の干渉抑制は energy_consistency_loss に任せる。
    """
    return interference_suppression_loss_per_sample(s, s_hat, eps=eps, frame_len=frame_len).mean()


def energy_consistency_loss(target, output, frame_len=1600, hop_len=800, threshold_db=-20, eps=1e-8):
    """
    フレームレベルのエネルギー一致損失。

    ターゲット話者が無音のフレームでは出力も無音になるよう誘導する。
    (002の最後5秒問題や001の冒頭漏れに対処)

    threshold_db=-20 (v5の-40より緩め) → より多くのフレームに適用される。
    """
    B, T = target.shape
    if T < frame_len:
        return torch.tensor(0.0, device=target.device)

    target_frames = target.unfold(-1, frame_len, hop_len)   # [B, nf, frame_len]
    output_frames = output.unfold(-1, frame_len, hop_len)

    target_energy = target_frames.pow(2).mean(-1)           # [B, nf]
    output_energy = output_frames.pow(2).mean(-1)

    peak_energy = target_energy.amax(dim=-1, keepdim=True).clamp(min=eps)  # [B, 1]
    thresh = peak_energy * (10 ** (threshold_db / 10))      # -40dB閘値

    silence_mask = (target_energy < thresh).float()         # 1=無音フレーム
    # output_energy / peak_energy の爆発を防ぐため clamp を追加
    ratio = (output_energy / peak_energy).clamp(max=10.0)
    loss = (silence_mask * ratio).mean()
    return loss

# ========================================
# 2. Dataset の定義
# ========================================
class SpeechDataset(Dataset):
    """
    CSVファイルに記載された音声パスから、mixture, enrollment, target のペアを返す Dataset
    CSV ファイルは、少なくとも以下のカラムを含むものとする:
        - mixture_path
        - enrollment_path
        - target_path
    """

    def __init__(self, csv_file, transform=None, enroll_crop_sec=5, sample_rate=16000,
                 random_enroll_crop=False, preload_to_ram=False):
        self.metadata = pd.read_csv(csv_file)
        self.transform = transform
        self.enroll_crop_samples = enroll_crop_sec * sample_rate
        self.random_enroll_crop = random_enroll_crop
        self.preload_to_ram = preload_to_ram
        self.cache = {}
        if preload_to_ram:
            print(f"RAM preloading {len(self.metadata)} samples...", flush=True)
            for i, row in tqdm(self.metadata.iterrows(), total=len(self.metadata),
                               desc="RAM load", dynamic_ncols=True):
                m, _ = load_audio(row["mixture_path"])
                t, _ = load_audio(row["target_path"])
                self.cache[i] = (m, t)
            print("RAM preload complete.", flush=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        if self.preload_to_ram:
            mixture, target = self.cache[idx]
        else:
            mixture, _ = load_audio(row["mixture_path"])
            target, _ = load_audio(row["target_path"])
        return mixture, row["enrollment_path"], target


# ========================================
# 3. 検証 / テスト時用の評価関数
# ========================================
@torch.no_grad()
def evaluate(model, dataloader, cache_dir, device):
    """DevやTestでSI-SNRを計算する共通関数"""
    model.eval()
    total_loss = 0.0
    for mixture, target, enrollment_paths in tqdm(dataloader, desc="Dev", leave=False, dynamic_ncols=True):
        mixture = mixture.to(device)
        target = target.to(device)

        speaker_embeddings = get_cached_speaker_embeddings(enrollment_paths, cache_dir, device)
        output = model(mixture, speaker_embeddings)

        output = output.squeeze(1)
        target = target.squeeze(1)

        loss = si_snr_loss(target, output)
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    model.train()  # ここで学習モードに戻す
    return avg_loss


# ========================================
# 4. メイン学習関数
# ========================================
def train_and_validate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # (A) DataLoader の準備
    # ---------------------------
    # Trainデータ（エンロールメントをランダムクロップして汎化性向上）
    train_dataset = SpeechDataset(csv_file=args.train_csv, random_enroll_crop=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_speech_batch,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Devデータ（ハイパーパラメータ調整・性能検証用）
    dev_dataset = SpeechDataset(csv_file=args.dev_csv)
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_speech_batch,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # ---------------------------
    # (B) モデルやオプティマイザの定義
    # ---------------------------
    model = SpeakerBeamSS().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ReduceLROnPlateau の設定
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.reduce_patience
    )

    # 事前学習済みモデルからfine-tune（--pretrained_model指定時）
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        pretrained_sd = torch.load(args.pretrained_model, map_location=device)
        current_sd = model.state_dict()
        loaded, skipped = [], []
        for k, v in pretrained_sd.items():
            if k in current_sd and current_sd[k].shape == v.shape:
                current_sd[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(current_sd)
        log(f"Loaded pretrained weights from {args.pretrained_model}")
        log(f"  Loaded: {len(loaded)} keys")
        if skipped:
            log(f"  Skipped (shape mismatch / new): {skipped}")

    # 音声埋め込みエンコーダー (ECAPA-TDNN)
    # torch._dynamo 初期化後にロードすることで speechbrain との衝突を回避
    speaker_encoder = load_ecapa_model(device)

    # 学習ループ開始前に全埋め込みを事前計算（学習中にSpeechBrainを呼び出さないため）
    precompute_all_embeddings(
        [args.train_csv, args.dev_csv],
        args.embedding_cache_dir,
        speaker_encoder,
        device,
    )

    # ---------------------------
    # Resume 処理
    # ---------------------------
    start_epoch = 0
    best_dev_loss = float("inf")
    patience_count = 0

    resume_ckpt = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
    if args.resume and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch    = ckpt["epoch"]          # 次に実行するエポック番号 (0-indexed)
        best_dev_loss  = ckpt["best_dev_loss"]
        patience_count = ckpt["patience_count"]
        log(f"Resumed from {resume_ckpt} (epoch {start_epoch}, best_dev_loss={best_dev_loss:.4f})")
    elif args.resume:
        log(f"WARNING: --resume specified but {resume_ckpt} not found. Starting fresh.")

    # 学習開始
    model.train()
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.0

        # ---------------------------
        # (C) Trainエポック
        # ---------------------------
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f"Epoch {epoch+1}/{args.num_epochs}", dynamic_ncols=True)
        _step_t0 = time.perf_counter()   # ステップ速度計測の起点
        for batch_idx, (mixture, target, enrollment_paths) in train_pbar:
            mixture = mixture.to(device)
            target = target.to(device)

            speaker_embeddings = get_cached_speaker_embeddings(
                enrollment_paths,
                args.embedding_cache_dir,
                device,
            )

            optimizer.zero_grad()
            # モデルの順伝播: mixture と speaker_embeddings を入力し、推定音声を出力
            output = model(mixture, speaker_embeddings)
            # 出力・target の形状: (B, 1, T) → SI-SNRは (B, T) で計算するため squeeze する
            output = output.squeeze(1)
            target = target.squeeze(1)

            loss_sisnr = si_snr_loss(target, output)
            loss_sdr = sdr_loss(target, output)
            loss_spec = spectral_loss(target, output)
            loss_energy = energy_consistency_loss(target, output)
            loss_interference = 5.0 * interference_suppression_loss(target, output)
            # SI-SNR: 基本的な分離品質を安定気く学習
            # SDR:    スケール非不変、絶対的な残留干渉を直接ペナルティ
            # Spec:   高周波音質補完
            # Energy: 無音フレームでの干渉漏れ抱制
            # Interference: 線形スケールで干渉残留を押し込む
            loss = loss_sisnr + loss_sdr + 0.1 * loss_spec + 0.1 * loss_energy + loss_interference
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # ステップ速度を計算して tqdm に表示
            _elapsed = time.perf_counter() - _step_t0
            _steps_done = batch_idx + 1  # 0-indexed なので +1
            _sps = _steps_done / max(_elapsed, 1e-9)   # steps per second
            _sec_per_step = 1.0 / _sps

            train_pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "SI-SNR": f"{loss_sisnr.item():.3f}",
                "Interf": f"{loss_interference.item():.4f}",
                "sps": f"{_sps:.2f}",
            })
            if batch_idx % args.log_interval == 0:
                log(f"[Train] Epoch {epoch+1}/{args.num_epochs}, "
                    f"Step {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f} "
                    f"(SI-SNR={loss_sisnr.item():.4f}, SDR={loss_sdr.item():.4f}, "
                    f"Spec={loss_spec.item():.4f}, Energy={loss_energy.item():.4f}, Interf={loss_interference.item():.4f}) "
                    f"[{_sps:.2f} steps/s, {_sec_per_step:.2f} s/step]")

        avg_train_loss = epoch_loss / len(train_loader)
        _epoch_elapsed = time.perf_counter() - _step_t0
        log(f"[Train] Epoch {epoch+1} done: avg_loss={avg_train_loss:.4f}, "
            f"elapsed={_epoch_elapsed:.1f}s, "
            f"{len(train_loader)/_epoch_elapsed:.2f} steps/s")

        # ---------------------------
        # (D) Devエポック (検証)
        # ---------------------------
        dev_loss = evaluate(model, dev_loader, args.embedding_cache_dir, device)
        log(f"[Dev]   Epoch {epoch+1}/{args.num_epochs}, Dev Loss: {dev_loss:.4f}")

        # スケジューラにDev損失を渡して学習率を調整（ReduceLROnPlateauなど）
        scheduler.step(dev_loss)

        # ---------------------------
        # (E) ベストモデルの更新 & 早期停止判定
        # ---------------------------
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        # モデルのみのエポック別チェックポイント（推論用）
        epoch_ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1:03d}.pth")
        torch.save(model.state_dict(), epoch_ckpt_path)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_count = 0

            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            log(f"=> Best model updated! Dev Loss = {dev_loss:.4f}")
        else:
            patience_count += 1
            if patience_count >= args.early_stop_patience:
                log("Early stopping triggered.")
                # フル checkpoint を保存してから終了
                torch.save({
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_dev_loss": best_dev_loss,
                    "patience_count": patience_count,
                }, os.path.join(args.checkpoint_dir, "checkpoint_latest.pt"))
                break

        # フル checkpoint（再開用）を毎エポック上書き保存
        torch.save({
            "epoch": epoch + 1,          # 次回はこのエポックから開始
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dev_loss": best_dev_loss,
            "patience_count": patience_count,
        }, os.path.join(args.checkpoint_dir, "checkpoint_latest.pt"))

        log(f"[Train] Epoch {epoch+1} finished! Average Train Loss: {avg_train_loss:.4f}\n")

    # ---------------------------
    # 学習終了後、最終的にベストモデルをロードしておく
    # ---------------------------
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        log(f"Loaded best model from {best_model_path}")
    else:
        log("No best model found (no improvement on Dev set).")

    return model


# ========================================
# 5. テスト時の評価関数
# ========================================
def test_model(args):
    """
    学習済み(ベスト)モデルを使ってテストデータで評価する関数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) テストデータローダーの用意
    test_dataset = SpeechDataset(csv_file=args.test_csv)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_speech_batch,
        pin_memory=False,
    )

    # 2) モデルをロードし、ECAPA-TDNN は後から初期化
    model = SpeakerBeamSS().to(device)

    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}. Train the model first.")

    # 3) ベストモデルを読み込み
    model.load_state_dict(torch.load(best_model_path))
    log(f"Loaded best model from {best_model_path} for testing.")

    # 4) ECAPA-TDNN をモデルロード後に初期化（torch._dynamo との衝突回避）
    speaker_encoder = load_ecapa_model(device)

    # テストデータの埋め込みを事前計算
    precompute_all_embeddings(
        [args.test_csv],
        args.embedding_cache_dir,
        speaker_encoder,
        device,
    )

    # 5) テストデータ上で評価 (SI-SNR)
    test_loss = evaluate(model, test_loader, speaker_encoder, device, args.embedding_cache_dir)
    log(f"[Test] Test Loss (SI-SNR): {test_loss:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train & Validate SpeakerBeam-SS with SI-SNR loss")
    parser.add_argument("--train_csv", type=str, default="data_csv/train/metadata.csv")
    parser.add_argument("--dev_csv", type=str, default="data_csv/dev/metadata.csv")
    parser.add_argument("--test_csv", type=str, default="data_csv/test/metadata.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--embedding_cache_dir", type=str, default=os.path.join("cache", "speaker_embeddings"))

    # Early Stopping用パラメータ
    parser.add_argument("--early_stop_patience", type=int, default=120,
                        help="Number of epochs to wait for dev_loss improvement before early stopping.")
    # ReduceLROnPlateau用パラメータ
    parser.add_argument("--reduce_patience", type=int, default=20,
                        help="Number of epochs with no improvement after which learning rate will be reduced.")

    parser.add_argument("--mode", type=str, default="train",
                        help="Specify 'train' or 'test'. If 'test', evaluate on test data.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file. If specified, output is also written to this file.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Path to pretrained model .pth to fine-tune from (e.g. v4's best_model.pth). "
                             "New layers (temporal_gate) are randomly initialized.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint_latest.pt in --checkpoint_dir.")

    args = parser.parse_args()

    _log_file = None
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) if os.path.dirname(args.log_file) else ".", exist_ok=True)
        # resume 時は追記モードでログを開く
        log_open_mode = "a" if args.resume else "w"
        _log_file = open(args.log_file, log_open_mode, encoding="utf-8", buffering=1)
        if args.resume:
            _log_file.write("\n" + "="*60 + "\n")
            _log_file.write("[RESUMED]\n")
            _log_file.write("="*60 + "\n")

    try:
        if args.mode == "train":
            trained_model = train_and_validate(args)
        elif args.mode == "test":
            test_model(args)
        else:
            raise ValueError("--mode should be 'train' or 'test'.")
    finally:
        if _log_file is not None:
            _log_file.close()
