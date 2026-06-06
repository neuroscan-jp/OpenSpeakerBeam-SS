//! ONNX Runtime inference for SpeakerBeamSS (ep110).
//!
//! Model I/O contract — see `onnx-runtime/DESIGN.md` §4.

pub mod ecapa_session;
pub mod incremental_streaming_session;
pub mod native;
pub mod onnx_util;
pub mod session;
pub mod streaming_session;

pub use ecapa_session::EcapaSession;
pub use incremental_streaming_session::IncrementalStreamingSession;
pub use session::{SpeakerBeamSession, EMBED_DIM, FIXED_SAMPLES};
pub use streaming_session::{StreamingSession, ENC_KERNEL, ENC_STRIDE, DEFAULT_LOOKAHEAD_FRAMES};
