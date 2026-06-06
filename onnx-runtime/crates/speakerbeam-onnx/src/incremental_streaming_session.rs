use std::path::Path;

use ndarray::Array3;
use ort::session::Session;
use thiserror::Error;

use crate::native::{NativeSeparatorStream, SeparatorWeights, WeightsError};
use crate::onnx_util::{open_session, tensor_from_array3, tensor_to_vec};
use crate::session::EMBED_DIM;
use crate::streaming_session::{ENC_KERNEL, ENC_STRIDE, LATENT_CHANNELS};

#[derive(Debug, Error)]
pub enum IncrementalStreamingError {
    #[error("ONNX runtime error: {0}")]
    Ort(#[from] ort::Error),
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("weights error: {0}")]
    Weights(#[from] WeightsError),
    #[error("invalid embedding length {0}, expected {1}")]
    InvalidEmbedding(usize, usize),
    #[error("unexpected tensor shape: {0:?}")]
    TensorShape(Vec<usize>),
}

/// Incremental streaming: encoder/decoder ONNX + native SeparatorStream (S4D state).
pub struct IncrementalStreamingSession {
    encoder: Session,
    decoder: Session,
    separator: NativeSeparatorStream,
    lookahead_frames: usize,
    audio_buf: Vec<f32>,
    latent_buf: Vec<f32>,
    sep_buf: Vec<f32>,
    n_latent_frames: usize,
    processed_latent: usize,
    wav_emitted: usize,
    embedding: Vec<f32>,
}

impl IncrementalStreamingSession {
    pub fn from_files(
        encoder_path: &Path,
        decoder_path: &Path,
        weights_path: &Path,
        threads: usize,
        lookahead_frames: usize,
    ) -> Result<Self, IncrementalStreamingError> {
        for (label, path) in [("encoder", encoder_path), ("decoder", decoder_path)] {
            if !path.exists() {
                return Err(IncrementalStreamingError::ModelNotFound(format!(
                    "{label}: {}",
                    path.display()
                )));
            }
        }
        if !weights_path.exists() {
            return Err(IncrementalStreamingError::ModelNotFound(format!(
                "separator weights: {}",
                weights_path.display()
            )));
        }

        let weights = SeparatorWeights::from_npz(weights_path)?;
        let separator = NativeSeparatorStream::from_weights(weights);

        Ok(Self {
            encoder: open_session(encoder_path, threads)?,
            decoder: open_session(decoder_path, threads)?,
            separator,
            lookahead_frames,
            audio_buf: Vec::new(),
            latent_buf: Vec::new(),
            sep_buf: Vec::new(),
            n_latent_frames: 0,
            processed_latent: 0,
            wav_emitted: 0,
            embedding: Vec::new(),
        })
    }

    pub fn set_embedding(&mut self, embedding: &[f32]) -> Result<(), IncrementalStreamingError> {
        if embedding.len() != EMBED_DIM {
            return Err(IncrementalStreamingError::InvalidEmbedding(
                embedding.len(),
                EMBED_DIM,
            ));
        }
        self.embedding = embedding.to_vec();
        self.separator.set_embedding(embedding)?;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.audio_buf.clear();
        self.latent_buf.clear();
        self.sep_buf.clear();
        self.n_latent_frames = 0;
        self.processed_latent = 0;
        self.wav_emitted = 0;
        self.separator.reset();
        if !self.embedding.is_empty() {
            let _ = self.separator.set_embedding(&self.embedding);
        }
    }

    fn append_latent_frames(&mut self) -> Result<usize, IncrementalStreamingError> {
        let t_total = self.audio_buf.len();
        if t_total < ENC_KERNEL {
            return Ok(0);
        }
        let n_frames = (t_total - ENC_KERNEL) / ENC_STRIDE + 1;
        let mut added = 0usize;
        while self.n_latent_frames < n_frames {
            let i = self.n_latent_frames;
            let start = i * ENC_STRIDE;
            let window = self.audio_buf[start..start + ENC_KERNEL].to_vec();
            let frame = self.run_encoder_frame(&window)?;
            self.latent_buf.extend_from_slice(&frame);
            self.n_latent_frames += 1;
            added += 1;
        }
        Ok(added)
    }

    fn run_encoder_frame(&mut self, window: &[f32]) -> Result<Vec<f32>, IncrementalStreamingError> {
        let wav = Array3::from_shape_vec((1, 1, ENC_KERNEL), window.to_vec())
            .map_err(|_| IncrementalStreamingError::TensorShape(vec![1, 1, ENC_KERNEL]))?;
        let outputs = self
            .encoder
            .run(ort::inputs!["waveform" => tensor_from_array3(wav)?])?;
        Ok(tensor_to_vec(&outputs["latent"])?)
    }

    fn run_separator_incremental(&mut self) -> Result<(), IncrementalStreamingError> {
        let n_total = self.n_latent_frames;
        let n_prev = self.processed_latent;
        if n_total <= n_prev {
            return Ok(());
        }
        let sep_chunk = self.separator.forward(
            &self.latent_buf[..n_total * LATENT_CHANNELS],
            n_total,
            n_prev,
            &self.embedding,
        )?;
        let n_new = n_total - n_prev;
        let start = self.sep_buf.len();
        self.sep_buf.resize(start + n_new * LATENT_CHANNELS, 0.0);
        for f in 0..n_new {
            let src = (n_prev + f) * LATENT_CHANNELS;
            let dst = start + f * LATENT_CHANNELS;
            self.sep_buf[dst..dst + LATENT_CHANNELS]
                .copy_from_slice(&sep_chunk[src..src + LATENT_CHANNELS]);
        }
        self.processed_latent = n_total;
        Ok(())
    }

    fn run_decoder(&mut self, n_frames: usize) -> Result<Vec<f32>, IncrementalStreamingError> {
        let mut sep_cf = vec![0.0f32; LATENT_CHANNELS * n_frames];
        for f in 0..n_frames {
            for c in 0..LATENT_CHANNELS {
                sep_cf[c * n_frames + f] = self.sep_buf[f * LATENT_CHANNELS + c];
            }
        }
        let latent = Array3::from_shape_vec((1, LATENT_CHANNELS, n_frames), sep_cf)
            .map_err(|_| IncrementalStreamingError::TensorShape(vec![1, LATENT_CHANNELS, n_frames]))?;
        let outputs = self
            .decoder
            .run(ort::inputs!["latent" => tensor_from_array3(latent)?])?;
        Ok(tensor_to_vec(&outputs["waveform"])?)
    }

    fn emit_ready(&mut self, n_new: usize) -> Result<Vec<f32>, IncrementalStreamingError> {
        if n_new == 0 || self.embedding.is_empty() {
            return Ok(Vec::new());
        }
        self.run_separator_incremental()?;
        if self.sep_buf.is_empty() {
            return Ok(Vec::new());
        }
        let n_sep = self.sep_buf.len() / LATENT_CHANNELS;
        let wav = self.run_decoder(n_sep)?;
        let safe_end = wav.len().saturating_sub(self.lookahead_frames * ENC_STRIDE);
        if safe_end <= self.wav_emitted {
            return Ok(Vec::new());
        }
        let out = wav[self.wav_emitted..safe_end].to_vec();
        self.wav_emitted = safe_end;
        Ok(out)
    }

    pub fn push(&mut self, chunk: &[f32]) -> Result<Vec<f32>, IncrementalStreamingError> {
        self.audio_buf.extend_from_slice(chunk);
        let n_new = self.append_latent_frames()?;
        self.emit_ready(n_new)
    }

    pub fn flush(&mut self) -> Result<Vec<f32>, IncrementalStreamingError> {
        if self.embedding.is_empty() {
            return Ok(Vec::new());
        }
        self.run_separator_incremental()?;
        if self.sep_buf.is_empty() {
            return Ok(Vec::new());
        }
        let n_sep = self.sep_buf.len() / LATENT_CHANNELS;
        let wav = self.run_decoder(n_sep)?;
        if wav.len() <= self.wav_emitted {
            return Ok(Vec::new());
        }
        let out = wav[self.wav_emitted..].to_vec();
        self.wav_emitted = wav.len();
        Ok(out)
    }
}
