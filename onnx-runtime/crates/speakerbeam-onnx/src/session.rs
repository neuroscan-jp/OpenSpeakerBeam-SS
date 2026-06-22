use std::path::Path;

use ndarray::{Array2, Array3};
use ort::session::Session;
use thiserror::Error;

use crate::onnx_util::{open_session, tensor_from_array2, tensor_from_array3, tensor_to_vec};

pub const EMBED_DIM: usize = 192;
/// Default trace length for legacy 10 s exports (`speakerbeam_ep110.onnx`).
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

/// Wrapper around ONNX Runtime `Session` for SpeakerBeamSS (offline batch / GLN).
pub struct SpeakerBeamSession {
    session: Session,
    fixed_samples: usize,
}

fn fixed_samples_from_session(session: &Session) -> Result<usize, SessionError> {
    let input = session
        .inputs()
        .iter()
        .find(|i| i.name() == "mixture")
        .ok_or_else(|| SessionError::ModelNotFound("mixture input not found".into()))?;
    let shape = input
        .dtype()
        .tensor_shape()
        .ok_or_else(|| SessionError::OutputShape(vec![]))?;
    if shape.len() != 3 {
        return Err(SessionError::OutputShape(
            shape.iter().map(|&d| d.max(0) as usize).collect(),
        ));
    }
    let t = shape[2];
    if t <= 0 {
        return Err(SessionError::OutputShape(vec![t.max(0) as usize]));
    }
    Ok(t as usize)
}

impl SpeakerBeamSession {
    pub fn from_file(path: &Path, threads: usize) -> Result<Self, SessionError> {
        if !path.exists() {
            return Err(SessionError::ModelNotFound(path.display().to_string()));
        }
        let session = open_session(path, threads)?;
        let fixed_samples = fixed_samples_from_session(&session)?;
        Ok(Self {
            session,
            fixed_samples,
        })
    }

    pub fn fixed_samples(&self) -> usize {
        self.fixed_samples
    }

    pub fn run(&mut self, mixture: &[f32], embedding: &[f32]) -> Result<Vec<f32>, SessionError> {
        if mixture.len() != self.fixed_samples {
            return Err(SessionError::InvalidLength(
                mixture.len(),
                self.fixed_samples,
            ));
        }
        if embedding.len() != EMBED_DIM {
            return Err(SessionError::InvalidEmbedding(embedding.len(), EMBED_DIM));
        }

        let mix = Array3::from_shape_vec((1, 1, mixture.len()), mixture.to_vec()).map_err(|_| {
            SessionError::InvalidLength(mixture.len(), self.fixed_samples)
        })?;
        let emb = Array2::from_shape_vec((1, EMBED_DIM), embedding.to_vec())
            .map_err(|_| SessionError::InvalidEmbedding(embedding.len(), EMBED_DIM))?;

        let outputs = self.session.run(ort::inputs![
            "mixture" => tensor_from_array3(mix)?,
            "spk_embedding" => tensor_from_array2(emb)?,
        ])?;

        let flat = tensor_to_vec(&outputs["enhanced"])?;
        if flat.len() < self.fixed_samples {
            return Err(SessionError::OutputShape(vec![flat.len()]));
        }
        Ok(flat)
    }
}
