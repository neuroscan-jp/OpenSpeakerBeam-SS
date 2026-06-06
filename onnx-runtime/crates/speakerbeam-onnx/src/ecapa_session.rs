use std::path::Path;

use ndarray::Array3;
use ort::session::Session;
use thiserror::Error;

use crate::onnx_util::{open_session, tensor_from_array3, tensor_to_vec};
use speakerbeam_core::config::RuntimeConfig;

#[derive(Debug, Error)]
pub enum EcapaError {
    #[error("ONNX runtime error: {0}")]
    Ort(#[from] ort::Error),
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("invalid features shape: {0:?}")]
    FeatureShape(Vec<usize>),
    #[error("unexpected embedding shape: {0:?}")]
    OutputShape(Vec<usize>),
}

pub struct EcapaSession {
    session: Session,
    embed_dim: usize,
}

impl EcapaSession {
    pub fn from_file(path: &Path, threads: usize, cfg: &RuntimeConfig) -> Result<Self, EcapaError> {
        if !path.exists() {
            return Err(EcapaError::ModelNotFound(path.display().to_string()));
        }
        let session = open_session(path, threads)?;
        Ok(Self {
            session,
            embed_dim: cfg.embed_dim,
        })
    }

    pub fn embed_features(
        &mut self,
        features: &[f32],
        n_frames: usize,
        n_mels: usize,
    ) -> Result<Vec<f32>, EcapaError> {
        if features.len() != n_frames * n_mels {
            return Err(EcapaError::FeatureShape(vec![n_frames, n_mels]));
        }
        let feats = Array3::from_shape_vec((1, n_frames, n_mels), features.to_vec())
            .map_err(|_| EcapaError::FeatureShape(vec![n_frames, n_mels]))?;

        let outputs = self
            .session
            .run(ort::inputs!["features" => tensor_from_array3(feats)?])?;
        let flat = tensor_to_vec(&outputs["embedding"])?;
        if flat.len() == self.embed_dim {
            Ok(flat)
        } else {
            Err(EcapaError::OutputShape(vec![flat.len()]))
        }
    }

    pub fn normalize_embedding(embedding: &mut [f32]) {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in embedding.iter_mut() {
                *x /= norm;
            }
        }
    }
}
