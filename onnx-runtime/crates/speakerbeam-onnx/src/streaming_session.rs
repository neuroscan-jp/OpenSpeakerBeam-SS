use std::path::Path;

use ndarray::{Array2, Array3};
use ort::session::Session;
use thiserror::Error;

use crate::onnx_util::{open_session, tensor_from_array2, tensor_from_array3, tensor_to_vec};
use crate::session::EMBED_DIM;

pub const ENC_KERNEL: usize = 320;
pub const ENC_STRIDE: usize = 160;
pub const LATENT_CHANNELS: usize = 4096;
pub const DEFAULT_LOOKAHEAD_FRAMES: usize = 1;

#[derive(Debug, Error)]
pub enum StreamingError {
    #[error("ONNX runtime error: {0}")]
    Ort(#[from] ort::Error),
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("separator latent length {0} exceeds fixed capacity {1}")]
    SeparatorCapacity(usize, usize),
    #[error("invalid embedding length {0}, expected {1}")]
    InvalidEmbedding(usize, usize),
    #[error("unexpected tensor shape: {0:?}")]
    TensorShape(Vec<usize>),
}

/// Legacy streaming: full separator ONNX (L=2048 pad) each chunk.
pub struct StreamingSession {
    encoder: Session,
    decoder: Session,
    separator: Session,
    separator_latent_cap: usize,
    lookahead_frames: usize,
    audio_buf: Vec<f32>,
    latent_buf: Vec<f32>,
    n_latent_frames: usize,
    wav_emitted: usize,
    embedding: Vec<f32>,
}

impl StreamingSession {
    pub fn from_files(
        encoder_path: &Path,
        decoder_path: &Path,
        separator_path: &Path,
        threads: usize,
        lookahead_frames: usize,
    ) -> Result<Self, StreamingError> {
        for (label, path) in [
            ("encoder", encoder_path),
            ("decoder", decoder_path),
            ("separator", separator_path),
        ] {
            if !path.exists() {
                return Err(StreamingError::ModelNotFound(format!(
                    "{label}: {}",
                    path.display()
                )));
            }
        }

        Ok(Self {
            encoder: open_session(encoder_path, threads)?,
            decoder: open_session(decoder_path, threads)?,
            separator: open_session(separator_path, threads)?,
            separator_latent_cap: 2048,
            lookahead_frames,
            audio_buf: Vec::new(),
            latent_buf: Vec::new(),
            n_latent_frames: 0,
            wav_emitted: 0,
            embedding: Vec::new(),
        })
    }

    pub fn set_embedding(&mut self, embedding: &[f32]) -> Result<(), StreamingError> {
        if embedding.len() != EMBED_DIM {
            return Err(StreamingError::InvalidEmbedding(embedding.len(), EMBED_DIM));
        }
        self.embedding = embedding.to_vec();
        Ok(())
    }

    pub fn reset(&mut self) {
        self.audio_buf.clear();
        self.latent_buf.clear();
        self.n_latent_frames = 0;
        self.wav_emitted = 0;
    }

    fn append_latent_frames(&mut self) -> Result<(), StreamingError> {
        let t_total = self.audio_buf.len();
        if t_total < ENC_KERNEL {
            return Ok(());
        }
        let n_frames = (t_total - ENC_KERNEL) / ENC_STRIDE + 1;
        while self.n_latent_frames < n_frames {
            let i = self.n_latent_frames;
            let start = i * ENC_STRIDE;
            let window = self.audio_buf[start..start + ENC_KERNEL].to_vec();
            let frame = self.run_encoder_frame(&window)?;
            self.latent_buf.extend_from_slice(&frame);
            self.n_latent_frames += 1;
        }
        Ok(())
    }

    fn run_encoder_frame(&mut self, window: &[f32]) -> Result<Vec<f32>, StreamingError> {
        let wav = Array3::from_shape_vec((1, 1, ENC_KERNEL), window.to_vec())
            .map_err(|_| StreamingError::TensorShape(vec![1, 1, ENC_KERNEL]))?;
        let outputs = self
            .encoder
            .run(ort::inputs!["waveform" => tensor_from_array3(wav)?])?;
        Ok(tensor_to_vec(&outputs["latent"])?)
    }

    fn run_separator(&mut self, n_frames: usize) -> Result<Vec<f32>, StreamingError> {
        if n_frames > self.separator_latent_cap {
            return Err(StreamingError::SeparatorCapacity(
                n_frames,
                self.separator_latent_cap,
            ));
        }
        let mut padded = vec![0.0f32; self.separator_latent_cap * LATENT_CHANNELS];
        for f in 0..n_frames {
            for c in 0..LATENT_CHANNELS {
                padded[c * self.separator_latent_cap + f] =
                    self.latent_buf[f * LATENT_CHANNELS + c];
            }
        }
        let latent = Array3::from_shape_vec(
            (1, LATENT_CHANNELS, self.separator_latent_cap),
            padded,
        )
        .map_err(|_| {
            StreamingError::TensorShape(vec![1, LATENT_CHANNELS, self.separator_latent_cap])
        })?;
        let emb = Array2::from_shape_vec((1, EMBED_DIM), self.embedding.clone())
            .map_err(|_| StreamingError::InvalidEmbedding(self.embedding.len(), EMBED_DIM))?;

        let outputs = self.separator.run(ort::inputs![
            "latent" => tensor_from_array3(latent)?,
            "spk_embedding" => tensor_from_array2(emb)?,
        ])?;
        let sep = tensor_to_vec(&outputs["separated"])?;
        let mut out = vec![0.0f32; n_frames * LATENT_CHANNELS];
        for f in 0..n_frames {
            for c in 0..LATENT_CHANNELS {
                out[f * LATENT_CHANNELS + c] = sep[c * self.separator_latent_cap + f];
            }
        }
        Ok(out)
    }

    fn run_decoder(&mut self, sep_latent: &[f32], n_frames: usize) -> Result<Vec<f32>, StreamingError> {
        let mut sep_cf = vec![0.0f32; LATENT_CHANNELS * n_frames];
        for f in 0..n_frames {
            for c in 0..LATENT_CHANNELS {
                sep_cf[c * n_frames + f] = sep_latent[f * LATENT_CHANNELS + c];
            }
        }
        let latent = Array3::from_shape_vec((1, LATENT_CHANNELS, n_frames), sep_cf)
            .map_err(|_| StreamingError::TensorShape(vec![1, LATENT_CHANNELS, n_frames]))?;
        let outputs = self
            .decoder
            .run(ort::inputs!["latent" => tensor_from_array3(latent)?])?;
        Ok(tensor_to_vec(&outputs["waveform"])?)
    }

    fn emit_ready(&mut self) -> Result<Vec<f32>, StreamingError> {
        if self.n_latent_frames == 0 || self.embedding.is_empty() {
            return Ok(Vec::new());
        }
        let sep = self.run_separator(self.n_latent_frames)?;
        let wav = self.run_decoder(&sep, self.n_latent_frames)?;
        let safe_end = wav.len().saturating_sub(self.lookahead_frames * ENC_STRIDE);
        if safe_end <= self.wav_emitted {
            return Ok(Vec::new());
        }
        let out = wav[self.wav_emitted..safe_end].to_vec();
        self.wav_emitted = safe_end;
        Ok(out)
    }

    pub fn push(&mut self, chunk: &[f32]) -> Result<Vec<f32>, StreamingError> {
        self.audio_buf.extend_from_slice(chunk);
        self.append_latent_frames()?;
        self.emit_ready()
    }

    pub fn flush(&mut self) -> Result<Vec<f32>, StreamingError> {
        if self.n_latent_frames == 0 || self.embedding.is_empty() {
            return Ok(Vec::new());
        }
        let sep = self.run_separator(self.n_latent_frames)?;
        let wav = self.run_decoder(&sep, self.n_latent_frames)?;
        if wav.len() <= self.wav_emitted {
            return Ok(Vec::new());
        }
        let out = wav[self.wav_emitted..].to_vec();
        self.wav_emitted = wav.len();
        Ok(out)
    }

    pub fn process_all(&mut self, mixture: &[f32]) -> Result<Vec<f32>, StreamingError> {
        self.reset();
        let mut out = self.push(mixture)?;
        out.extend(self.flush()?);
        Ok(out)
    }
}
