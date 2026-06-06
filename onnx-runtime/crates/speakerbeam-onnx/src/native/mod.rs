//! Native SeparatorStream (incremental S4D + conv/cgLN) for real-time CPU inference.

mod decoder;
mod ops;
mod separator;
pub mod weights;

pub use decoder::NativeDecoder;
pub use separator::NativeSeparatorStream;
pub use weights::{DecoderWeights, SeparatorWeights, WeightsError};
