//! Shared types and audio preprocessing for the ONNX runtime.
//!
//! See `onnx-runtime/DESIGN.md` for architecture.

pub mod audio;
pub mod chunk_buffer;
pub mod config;
pub mod embedding;

pub use chunk_buffer::{ms_to_samples, ChunkAggregator};
pub use config::RuntimeConfig;
