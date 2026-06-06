use std::path::Path;

use ndarray::{Array2, Array3};
use ort::session::Session;
use thiserror::Error;

use crate::onnx_util::{open_session, tensor_from_array2, tensor_from_array3, tensor_to_vec};

pub const EMBED_DIM: usize = 192;
pub const FIXED_SAMPLES: usize = 160_000;

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("ONNX runtime error: {0}")]
    Ort(#[from] ort::Error),
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("invalid mixture length {0}, expected {1} for this ONNX export")]
    InvalidLength(usize, usize),
    #[error("invalid embedding length {0}, expected {1}")]
    InvalidEmbedding(usize, usize),
    #[error("unexpected output shape: {0:?}")]
    OutputShape(Vec<usize>),
}

/// Wrapper around ONNX Runtime `Session` for SpeakerBeamSS (ep110).
pub struct SpeakerBeamSession {
    session: Session,
}

impl SpeakerBeamSession {
    pub fn from_file(path: &Path, threads: usize) -> Result<Self, SessionError> {
        if !path.exists() {
            return Err(SessionError::ModelNotFound(path.display().to_string()));
        }
        let session = open_session(path, threads)?;
        Ok(Self { session })
    }

    pub fn run(&mut self, mixture: &[f32], embedding: &[f32]) -> Result<Vec<f32>, SessionError> {
        if mixture.len() != FIXED_SAMPLES {
            return Err(SessionError::InvalidLength(mixture.len(), FIXED_SAMPLES));
        }
        if embedding.len() != EMBED_DIM {
            return Err(SessionError::InvalidEmbedding(embedding.len(), EMBED_DIM));
        }

        let mix = Array3::from_shape_vec((1, 1, mixture.len()), mixture.to_vec())
            .map_err(|_| SessionError::InvalidLength(mixture.len(), FIXED_SAMPLES))?;
        let emb = Array2::from_shape_vec((1, EMBED_DIM), embedding.to_vec())
            .map_err(|_| SessionError::InvalidEmbedding(embedding.len(), EMBED_DIM))?;

        let outputs = self.session.run(ort::inputs![
            "mixture" => tensor_from_array3(mix)?,
            "spk_embedding" => tensor_from_array2(emb)?,
        ])?;

        let flat = tensor_to_vec(&outputs["enhanced"])?;
        match flat.len() {
            FIXED_SAMPLES => Ok(flat),
            _ => Err(SessionError::OutputShape(vec![flat.len()])),
        }
    }
}
