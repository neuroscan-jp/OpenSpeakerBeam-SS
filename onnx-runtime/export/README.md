# ONNX Export (Python)

PyTorch `SpeakerBeamSS`（ep110）を ONNX に変換するスクリプト群。

| ファイル | 役割 |
|----------|------|
| `export_speakerbeam.py` | `best_model.pth` → `../models/speakerbeam_ep110.onnx`（オフライン10s） |
| `export_ecapa.py` | ECAPA → `ecapa_embedding.onnx` + `ecapa_fbank.npz`（Phase 2） |
| `export_streaming_models.py` | 分割 ONNX: encoder_frame / decoder / separator_cgln / separator_chunk |
| `export_streaming_weights.py` | 増分 Rust 用 `streaming_separator.npz`（S4D 状態 + conv/cgLN 重み） |
| `verify_incremental_streaming.py` | Python 増分ストリーム性能・Rust 連携手順 |
| `verify_native_separator.py` | SeparatorStream 重み export スモークテスト |
| `compute_ecapa_features.py` | enrollment wav → FBank `.npy`（Rust ONNX 連携） |
| `verify_onnx.py` | PyTorch vs ONNX 数値一致（Phase 1） |
| `verify_ecapa_onnx.py` | ECAPA embedding ONNX 検証（Phase 2） |
| `verify_streaming_onnx.py` | 分割 ONNX ストリーミング検証（Phase 2） |
| `verify_streaming.py` | PyTorch ストリーミング検証 |
| `extract_embedding.py` | enrollment → `.npy`（Phase 1 Rust 連携用） |

実行時はリポジトリルートの `.venv` を使用:

```bash
cd onnx-runtime/export
../../.venv/Scripts/python.exe export_speakerbeam.py \
  --checkpoint ../../checkpoints/scratch_v2_lowsir/best_model.pth \
  --output ../models/speakerbeam_ep110.onnx
```
