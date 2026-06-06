# Parity Tests

Python `inference.py`（ep110, `--no_filter`）と Rust ONNX 出力の一致を確認する。

## 手順（Phase 1）

1. `export/export_speakerbeam.py` で ONNX 生成
2. `export/verify_onnx.py` で PyTorch vs ONNX（同一 embedding）
3. `cargo test -p speakerbeam-onnx` で Rust vs ONNX

許容: max abs diff < 1e-4, SI-SNR 差 < 0.1 dB
