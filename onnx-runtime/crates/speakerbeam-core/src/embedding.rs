use std::path::Path;
use std::process::Command;

use ndarray::{Array1, ArrayD};
use ndarray_npy::ReadNpyExt;
use thiserror::Error;

use crate::config::RuntimeConfig;

pub const N_MELS: usize = 80;

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("failed to read embedding file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to read wav: {0}")]
    Wav(#[from] hound::Error),
    #[error("failed to read npy: {0}")]
    Npy(String),
    #[error("invalid embedding length {0}, expected {1}")]
    InvalidLength(usize, usize),
    #[error("embedding must be 1-D, got rank {0}")]
    InvalidRank(usize),
    #[error("feature extraction failed: {0}")]
    FeatureExtract(String),
    #[error("invalid features shape {0:?}")]
    FeatureShape(Vec<usize>),
}

/// Load a precomputed speaker embedding from `.npy` (float32, shape `[192]` or `[1, 192]`).
pub fn load_embedding_npy(path: &Path, expected_dim: usize) -> Result<Vec<f32>, EmbeddingError> {
    let mut file = std::fs::File::open(path)?;
    let arr: ArrayD<f32> = ArrayD::read_npy(&mut file).map_err(|e| EmbeddingError::Npy(e.to_string()))?;
    let rank = arr.ndim();
    let flat: Array1<f32> = match rank {
        1 => arr
            .into_dimensionality()
            .map_err(|_| EmbeddingError::InvalidRank(rank))?,
        2 if arr.shape()[0] == 1 => Array1::from_iter(arr.iter().copied()),
        n => return Err(EmbeddingError::InvalidRank(n)),
    };
    if flat.len() != expected_dim {
        return Err(EmbeddingError::InvalidLength(flat.len(), expected_dim));
    }
    Ok(flat.to_vec())
}

/// Segment start indices for multi-segment enrollment (matches inference.py).
pub fn enrollment_segment_starts(total_samples: usize, cfg: &RuntimeConfig) -> Vec<usize> {
    let seg = cfg.enroll_segment_samples;
    if total_samples <= seg {
        return vec![0];
    }
    let step = ((total_samples - seg) / 3).max(1);
    let mut starts = Vec::new();
    let mut s = 0usize;
    while s + seg <= total_samples && starts.len() < cfg.enroll_max_segments {
        starts.push(s);
        s += step;
    }
    if starts.is_empty() {
        starts.push(0);
    }
    starts
}

/// Call Python `compute_ecapa_features.py` for one waveform segment.
pub fn extract_features_python(
    python: &Path,
    wav_path: &Path,
    output_npy: &Path,
    script_dir: &Path,
) -> Result<(), EmbeddingError> {
    let script = script_dir.join("compute_ecapa_features.py");
    let status = Command::new(python)
        .arg(&script)
        .arg("--wav")
        .arg(wav_path)
        .arg("--output")
        .arg(output_npy)
        .status()
        .map_err(EmbeddingError::Io)?;
    if !status.success() {
        return Err(EmbeddingError::FeatureExtract(format!(
            "compute_ecapa_features.py exited with {status}"
        )));
    }
    Ok(())
}

/// Load feature matrix `.npy` shaped `[T, 80]`.
pub fn load_features_npy(path: &Path) -> Result<(Vec<f32>, usize), EmbeddingError> {
    let mut file = std::fs::File::open(path)?;
    let arr: ArrayD<f32> = ArrayD::read_npy(&mut file).map_err(|e| EmbeddingError::Npy(e.to_string()))?;
    match arr.ndim() {
        2 => {
            let n_frames = arr.shape()[0];
            let n_mels = arr.shape()[1];
            if n_mels != N_MELS {
                return Err(EmbeddingError::FeatureShape(vec![n_frames, n_mels]));
            }
            Ok((arr.iter().copied().collect(), n_frames))
        }
        n => Err(EmbeddingError::FeatureShape(vec![n])),
    }
}

/// Write a mono WAV snippet for segment feature extraction.
pub fn write_wav_segment(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
) -> Result<(), EmbeddingError> {
    use hound::{SampleFormat, WavSpec, WavWriter};
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Extract enrollment embedding via Python features + caller-supplied ONNX embed fn.
pub fn extract_enrollment_via_python_features<E>(
    enrollment_samples: &[f32],
    cfg: &RuntimeConfig,
    python: &Path,
    export_dir: &Path,
    mut embed_fn: E,
) -> Result<Vec<f32>, EmbeddingError>
where
    E: FnMut(&[f32], usize) -> Result<Vec<f32>, EmbeddingError>,
{
    let tmp = std::env::temp_dir().join("speakerbeam_ecapa");
    std::fs::create_dir_all(&tmp)?;

    let starts = enrollment_segment_starts(enrollment_samples.len(), cfg);
    let mut segment_embeddings = Vec::new();

    for (i, &start) in starts.iter().enumerate() {
        let end = (start + cfg.enroll_segment_samples).min(enrollment_samples.len());
        let seg_wav = &enrollment_samples[start..end];
        let wav_path = tmp.join(format!("enroll_seg_{i}.wav"));
        let feat_path = tmp.join(format!("enroll_seg_{i}_feat.npy"));
        write_wav_segment(&wav_path, seg_wav, cfg.sample_rate)?;
        extract_features_python(python, &wav_path, &feat_path, export_dir)?;
        let (features, n_frames) = load_features_npy(&feat_path)?;
        let emb = embed_fn(&features, n_frames)?;
        if emb.len() != cfg.embed_dim {
            return Err(EmbeddingError::InvalidLength(emb.len(), cfg.embed_dim));
        }
        segment_embeddings.push(emb);
    }

    aggregate_embeddings(segment_embeddings, cfg.embed_dim)
}

/// Mean aggregate + L2 normalize.
pub fn aggregate_embeddings(segments: Vec<Vec<f32>>, embed_dim: usize) -> Result<Vec<f32>, EmbeddingError> {
    if segments.is_empty() {
        return Err(EmbeddingError::InvalidLength(0, embed_dim));
    }
    let mut mean = vec![0.0f32; embed_dim];
    for seg in &segments {
        if seg.len() != embed_dim {
            return Err(EmbeddingError::InvalidLength(seg.len(), embed_dim));
        }
        for (m, v) in mean.iter_mut().zip(seg.iter()) {
            *m += v;
        }
    }
    let n = segments.len() as f32;
    for m in &mut mean {
        *m /= n;
    }
    normalize_l2(&mut mean);
    Ok(mean)
}

pub fn normalize_l2(embedding: &mut [f32]) {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in embedding.iter_mut() {
            *x /= norm;
        }
    }
}
