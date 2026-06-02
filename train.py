import os
import sys
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
    mixtures, enrollments, targets, enrollment_paths = zip(*batch)

    def _pad_waveforms(waveforms):
        padded = pad_sequence([waveform.squeeze(0) for waveform in waveforms], batch_first=True)
        return padded.unsqueeze(1)

    enrollment_lengths = torch.tensor([waveform.shape[-1] for waveform in enrollments], dtype=torch.int64)

    return (
        _pad_waveforms(mixtures),
        _pad_waveforms(enrollments),
        _pad_waveforms(targets),
        enrollment_lengths,
        list(enrollment_paths),
    )


def get_cached_speaker_embeddings(
    speaker_encoder,
    enrollment_batch,
    enrollment_lengths,
    enrollment_paths,
    cache_dir,
    device,
):
    os.makedirs(cache_dir, exist_ok=True)
    embeddings = []

    for index, enrollment_path in enumerate(enrollment_paths):
        cache_key = hashlib.sha1(os.path.normpath(enrollment_path).encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{cache_key}.npy")

        if os.path.exists(cache_path):
            embedding = torch.from_numpy(np.load(cache_path))
        else:
            enrollment_length = int(enrollment_lengths[index].item())
            waveform = enrollment_batch[index, :, :enrollment_length]
            embedding_np = get_speaker_embedding(speaker_encoder, waveform)
            np.save(cache_path, embedding_np)
            embedding = torch.from_numpy(embedding_np)

        embeddings.append(embedding.float())

    return torch.stack(embeddings, dim=0).to(device)


def precompute_all_embeddings(csv_files, cache_dir, speaker_encoder, device):
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

    done = 0
    for enrollment_path in sorted(all_enrollment_paths):
        cache_key = hashlib.sha1(os.path.normpath(enrollment_path).encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{cache_key}.npy")

        if os.path.exists(cache_path):
            done += 1
            continue

        try:
            waveform, _ = load_audio(enrollment_path)
            embedding_np = get_speaker_embedding(speaker_encoder, waveform)
            np.save(cache_path, embedding_np)
        except Exception as e:
            log(f"  Warning: Failed for {enrollment_path}: {e}")

        done += 1
        if done % 200 == 0 or done == total:
            log(f"  [{done}/{total}] embeddings cached")

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

    def __init__(self, csv_file, transform=None, enroll_crop_sec=5, sample_rate=16000, random_enroll_crop=False):
        self.metadata = pd.read_csv(csv_file)
        self.transform = transform
        self.enroll_crop_samples = enroll_crop_sec * sample_rate
        self.random_enroll_crop = random_enroll_crop

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        mixture, sr1 = load_audio(row["mixture_path"])
        enrollment, sr2 = load_audio(row["enrollment_path"])
        target, sr3 = load_audio(row["target_path"])

        # 学習時はenrollmentのランダムな区間をクロップ→モデルをより汎用的に
        if self.random_enroll_crop:
            T = enrollment.shape[-1]
            crop_len = min(self.enroll_crop_samples, T)
            if T > crop_len:
                start = torch.randint(0, T - crop_len + 1, (1,)).item()
                enrollment = enrollment[..., start: start + crop_len]

        if self.transform:
            mixture = self.transform(mixture)
            enrollment = self.transform(enrollment)
            target = self.transform(target)
        return mixture, enrollment, target, row["enrollment_path"]


# ========================================
# 3. 検証 / テスト時用の評価関数
# ========================================
@torch.no_grad()
def evaluate(model, dataloader, speaker_encoder, device, cache_dir):
    """DevやTestでSI-SNRを計算する共通関数"""
    model.eval()
    total_loss = 0.0
    for mixture, enrollment, target, enrollment_lengths, enrollment_paths in dataloader:
        mixture = mixture.to(device)
        target = target.to(device)

        speaker_embeddings = get_cached_speaker_embeddings(
            speaker_encoder,
            enrollment,
            enrollment_lengths,
            enrollment_paths,
            cache_dir,
            device,
        )
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
        num_workers=0,
        collate_fn=collate_speech_batch,
        pin_memory=False,
    )

    # Devデータ（ハイパーパラメータ調整・性能検証用）
    dev_dataset = SpeechDataset(csv_file=args.dev_csv)
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_speech_batch,
        pin_memory=False,
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
        state_dict = torch.load(args.pretrained_model, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log(f"Loaded pretrained weights from {args.pretrained_model}")
        if missing:
            log(f"  New (randomly init) keys: {missing}")
        if unexpected:
            log(f"  Ignored keys: {unexpected}")

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

    # 学習開始
    model.train()
    global_step = 0

    best_dev_loss = float("inf")  # Devの最小損失を追跡
    patience_count = 0            # Early Stopping用カウンタ

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0

        # ---------------------------
        # (C) Trainエポック
        # ---------------------------
        for batch_idx, (mixture, enrollment, target, enrollment_lengths, enrollment_paths) in enumerate(train_loader):
            # mixture, enrollment, target は形状が (B, 1, T)
            mixture = mixture.to(device)
            target = target.to(device)

            # enrollment 音声からスピーカーエンベディングを取得
            speaker_embeddings = get_cached_speaker_embeddings(
                speaker_encoder,
                enrollment,
                enrollment_lengths,
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
            loss_spec = spectral_loss(target, output)
            loss_energy = energy_consistency_loss(target, output)
            loss = loss_sisnr + 0.1 * loss_spec + 0.3 * loss_energy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if batch_idx % args.log_interval == 0:
                log(f"[Train] Epoch {epoch+1}/{args.num_epochs}, "
                    f"Step {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f} "
                    f"(SI-SNR={loss_sisnr.item():.4f}, Spec={loss_spec.item():.4f}, Energy={loss_energy.item():.4f})")

        avg_train_loss = epoch_loss / len(train_loader)

        # ---------------------------
        # (D) Devエポック (検証)
        # ---------------------------
        dev_loss = evaluate(model, dev_loader, speaker_encoder, device, args.embedding_cache_dir)
        log(f"[Dev]   Epoch {epoch+1}/{args.num_epochs}, Dev Loss: {dev_loss:.4f}")

        # スケジューラにDev損失を渡して学習率を調整（ReduceLROnPlateauなど）
        scheduler.step(dev_loss)

        # ---------------------------
        # (E) ベストモデルの更新 & 早期停止判定
        # ---------------------------
        # 毎エポックのチェックポイントを保存
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        epoch_ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1:03d}.pth")
        torch.save(model.state_dict(), epoch_ckpt_path)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_count = 0

            # ベストモデルを保存（Dev損失が改善したとき）
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            log(f"=> Best model updated! Dev Loss = {dev_loss:.4f}")
        else:
            # 改善しなかった場合
            patience_count += 1
            if patience_count >= args.early_stop_patience:
                log("Early stopping triggered.")
                break

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

    args = parser.parse_args()

    _log_file = None
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) if os.path.dirname(args.log_file) else ".", exist_ok=True)
        _log_file = open(args.log_file, "w", encoding="utf-8", buffering=1)

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
