//! Native SeparatorStream (incremental S4D + conv/cgLN) for real-time CPU inference.

mod ops;
mod separator;
pub mod weights;

pub use separator::NativeSeparatorStream;
pub use weights::{SeparatorWeights, WeightsError};
