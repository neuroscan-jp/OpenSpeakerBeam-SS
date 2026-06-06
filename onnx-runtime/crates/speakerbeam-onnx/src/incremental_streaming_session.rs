use std::path::Path;

use ndarray::Array3;
use ort::session::Session;
use thiserror::Error;

use crate::native::{NativeDecoder, NativeSeparatorStream, SeparatorWeights, WeightsError};
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

/// Incremental streaming: encoder ONNX + native separator/decoder (S4D state).
pub struct IncrementalStreamingSession {
    encoder: Session,
    decoder: NativeDecoder,
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
        if !encoder_path.exists() {
            return Err(IncrementalStreamingError::ModelNotFound(format!(
                "encoder: {}",
                encoder_path.display()
            )));
        }
        if !decoder_path.exists() {
            eprintln!(
                "note: decoder ONNX not used; native decoder loaded from {}",
                weights_path.display()
            );
        }
        if !weights_path.exists() {
            return Err(IncrementalStreamingError::ModelNotFound(format!(
                "separator weights: {}",
                weights_path.display()
            )));
        }

        let weights = SeparatorWeights::from_npz(weights_path)?;
        let decoder = NativeDecoder::from_weights(weights.decoder_weights()?);
        let separator = NativeSeparatorStream::from_weights(weights);

        Ok(Self {
            encoder: open_session(encoder_path, threads)?,
            decoder,
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
        self.decoder.reset();
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
        let n_pending = n_frames.saturating_sub(self.n_latent_frames);
        if n_pending == 0 {
            return Ok(0);
        }
        self.run_encoder_batch(self.n_latent_frames, n_pending)?;
        self.n_latent_frames += n_pending;
        Ok(n_pending)
    }

    fn run_encoder_batch(
        &mut self,
        start_frame: usize,
        batch: usize,
    ) -> Result<(), IncrementalStreamingError> {
        let mut wav_batch = Vec::with_capacity(batch * ENC_KERNEL);
        for i in 0..batch {
            let frame_idx = start_frame + i;
            let start = frame_idx * ENC_STRIDE;
            wav_batch.extend_from_slice(&self.audio_buf[start..start + ENC_KERNEL]);
        }
        let wav = Array3::from_shape_vec((batch, 1, ENC_KERNEL), wav_batch)
            .map_err(|_| IncrementalStreamingError::TensorShape(vec![batch, 1, ENC_KERNEL]))?;
        let outputs = self
            .encoder
            .run(ort::inputs!["waveform" => tensor_from_array3(wav)?])?;
        let flat = tensor_to_vec(&outputs["latent"])?;
        let frame_len = LATENT_CHANNELS;
        if flat.len() != batch * frame_len {
            return Err(IncrementalStreamingError::TensorShape(vec![
                batch,
                LATENT_CHANNELS,
                1,
            ]));
        }
        let prev = self.latent_buf.len();
        self.latent_buf.resize(prev + flat.len(), 0.0);
        self.latent_buf[prev..].copy_from_slice(&flat);
        Ok(())
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
        self.sep_buf.resize(n_total * LATENT_CHANNELS, 0.0);
        let dst = n_prev * LATENT_CHANNELS;
        let nbytes = n_new * LATENT_CHANNELS;
        self.sep_buf[dst..dst + nbytes].copy_from_slice(&sep_chunk[..nbytes]);
        self.processed_latent = n_total;
        Ok(())
    }

    fn run_decoder(&mut self, n_frames: usize) {
        self.decoder
            .decode_fm(&self.sep_buf[..n_frames * LATENT_CHANNELS], n_frames);
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
        self.run_decoder(n_sep);
        let wav = self.decoder.wav();
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
        self.run_decoder(n_sep);
        let wav = self.decoder.wav();
        if wav.len() <= self.wav_emitted {
            return Ok(Vec::new());
        }
        let out = wav[self.wav_emitted..].to_vec();
        self.wav_emitted = wav.len();
        Ok(out)
    }
}
