# OpenSpeakerBeam-SS ONNX Runtime (Rust)

Python 学習パイプライン（リポジトリルート）とは **別ツリー** の、推論専用 Rust 実装です。

| 領域 | パス | 役割 |
|---|---|---|
| 学習・評価（Python） | `../`（リポジトリルート） | `train.py`, `inference.py`, `model/` |
| **推論ランタイム（Rust）** | **`onnx-runtime/`** | ONNX Runtime による CPU 推論 |

## ドキュメント

- **[DESIGN.md](./DESIGN.md)** — 詳細設計（アーキテクチャ、I/O 契約、クレート構成、段階的実装計画）

## ディレクトリ概要

```
onnx-runtime/
├── DESIGN.md              # 詳細設計書
├── Cargo.toml             # Rust ワークスペース
├── crates/
│   ├── speakerbeam-core/  # 型・音声前処理・埋め込み集約
│   ├── speakerbeam-onnx/  # ONNX Session ラッパー
│   └── speakerbeam-cli/   # CLI バイナリ
├── export/                # PyTorch → ONNX 変換（Python）
├── models/                # 配置する .onnx 重み（git 管理外推奨）
└── tests/                 # Python との数値一致テスト
```

## 現行ベースモデル

- チェックポイント: `../checkpoints/scratch_v2_lowsir/best_model.pth`（**ep110**）
- サンプルレート: 16 kHz mono
- 話者埋め込み: ECAPA-TDNN 192 次元（Phase 1 は Python 前処理、Phase 2 で ONNX 化）

## Phase 2 クイックスタート

### 1. モデルエクスポート

```powershell
cd d:\OpenSpeakerBeam-SS
.\.venv\Scripts\python.exe onnx-runtime/export/export_ecapa.py
.\.venv\Scripts\python.exe onnx-runtime/export/export_streaming_models.py
.\.venv\Scripts\python.exe onnx-runtime/export/verify_ecapa_onnx.py
.\.venv\Scripts\python.exe onnx-runtime/export/verify_streaming_onnx.py
```

### 2. Rust CLI（ストリーミング + ECAPA ONNX）

```powershell
cd onnx-runtime
cargo build --release -p speakerbeam-cli
.\target\release\speakerbeam-cli.exe `
  --mixture ..\data\sample\mixture_000001.wav `
  --enrollment ..\data\sample\enrollment_000001.wav `
  --output ..\data\sample\result_rust_stream.wav `
  --embedding-backend onnx `
  --stream
```

Python でも任意長ストリーミング可能:

```powershell
.\.venv\Scripts\python.exe inference.py `
  --mixture data/sample/mixture_000001.wav `
  --enrollment data/sample/enrollment_000001.wav `
  --output data/sample/result_stream.wav `
  --stream --no_filter
```

## Phase 1 クイックスタート（オフライン ONNX 10s 固定）

### 1. ONNX エクスポート（Python）

```powershell
cd d:\OpenSpeakerBeam-SS
.\.venv\Scripts\python.exe -m pip install onnx onnxruntime onnxscript
.\.venv\Scripts\python.exe onnx-runtime/export/export_speakerbeam.py
.\.venv\Scripts\python.exe onnx-runtime/export/verify_onnx.py
.\.venv\Scripts\python.exe onnx-runtime/export/verify_real_audio.py
```

### 2. 埋め込み事前計算（高速パス）

```powershell
.\.venv\Scripts\python.exe onnx-runtime/export/extract_embedding.py `
  --enrollment data/sample/enrollment_000001.wav `
  --output onnx-runtime/models/embedding_000001.npy
```

### 3. Rust ビルド & 推論

[Rust toolchain](https://rustup.rs/) インストール後:

```powershell
cd onnx-runtime
cargo build --release

cargo run --release -p speakerbeam-cli -- `
  --mixture ../data/sample/mixture_000001.wav `
  --embedding-npy models/embedding_000001.npy `
  --output ../data/sample/result_000001_onnx_rust.wav `
  --model models/speakerbeam_ep110.onnx
```

`--enrollment` を指定すると Python で ECAPA 抽出してから推論（初回のみ遅い）。

**注意:** 現 ONNX は **固定 10 秒（160000 samples）** 向けエクスポート。それ以外の長さはパディング/トリムされます。
