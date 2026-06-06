use std::path::Path;

use hound::{SampleFormat, WavSpec, WavWriter};
use thiserror::Error;

use crate::config::RuntimeConfig;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("failed to read wav: {0}")]
    Read(#[from] hound::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported sample rate {0}, expected {1}")]
    SampleRate(u32, u32),
    #[error("empty audio")]
    Empty,
}

/// Load mono f32 samples at 16 kHz.
pub fn load_wav_mono_16k(path: &Path, cfg: &RuntimeConfig) -> Result<Vec<f32>, AudioError> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != cfg.sample_rate {
        return Err(AudioError::SampleRate(spec.sample_rate, cfg.sample_rate));
    }
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()?,
        SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
    };
    if samples.is_empty() {
        return Err(AudioError::Empty);
    }
    Ok(samples)
}

/// Save mono f32 PCM16 WAV.
pub fn save_wav_mono_16k(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), AudioError> {
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
        let clamped = s.clamp(-1.0, 1.0);
        let v = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Pad or trim mixture to fixed ONNX trace length (10 s @ 16 kHz).
pub fn fit_fixed_length(samples: &[f32], target_len: usize) -> Vec<f32> {
    if samples.len() == target_len {
        return samples.to_vec();
    }
    if samples.len() > target_len {
        return samples[..target_len].to_vec();
    }
    let mut out = samples.to_vec();
    out.resize(target_len, 0.0);
    out
}
