use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use ndarray::{IxDyn, OwnedRepr};
use ndarray_npy::NpzReader;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WeightsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("NPZ error: {0}")]
    Npz(String),
    #[error("missing weight array: {0}")]
    Missing(String),
}

#[derive(Clone)]
pub struct SeparatorWeights {
    pub arrays: HashMap<String, Vec<f32>>,
    pub num_blocks: usize,
    pub latent_channels: usize,
    pub sep_channels: usize,
    pub hidden_channels: usize,
    pub embed_dim: usize,
}

impl SeparatorWeights {
    pub fn from_npz(path: &Path) -> Result<Self, WeightsError> {
        let file = File::open(path)?;
        let mut reader = NpzReader::new(file).map_err(|e| WeightsError::Npz(e.to_string()))?;
        let mut arrays = HashMap::new();
        for name in reader
            .names()
            .map_err(|e| WeightsError::Npz(e.to_string()))?
        {
            let values = if let Ok(arr) =
                reader.by_name::<OwnedRepr<f32>, IxDyn>(&name)
            {
                let (vec, _) = arr.into_raw_vec_and_offset();
                vec
            } else if let Ok(arr) = reader.by_name::<OwnedRepr<i64>, IxDyn>(&name) {
                arr.iter().map(|&v| v as f32).collect()
            } else {
                return Err(WeightsError::Npz(format!(
                    "unsupported dtype for array '{name}'"
                )));
            };
            arrays.insert(name, values);
        }
        let num_blocks = arrays
            .get("num_blocks")
            .and_then(|v| v.first().copied())
            .unwrap_or(6.0) as usize;
        let latent_channels = arrays
            .get("latent_channels")
            .and_then(|v| v.first().copied())
            .unwrap_or(4096.0) as usize;
        let sep_channels = arrays
            .get("sep_channels")
            .and_then(|v| v.first().copied())
            .unwrap_or(256.0) as usize;
        let embed_dim = arrays
            .get("embed_dim")
            .and_then(|v| v.first().copied())
            .unwrap_or(192.0) as usize;
        let hidden_channels = arrays
            .get("hidden_channels")
            .and_then(|v| v.first().copied())
            .unwrap_or(512.0) as usize;
        Ok(Self {
            arrays,
            num_blocks,
            latent_channels,
            sep_channels,
            hidden_channels,
            embed_dim,
        })
    }

    pub fn get(&self, key: &str) -> Result<&[f32], WeightsError> {
        self.arrays
            .get(key)
            .map(|v| v.as_slice())
            .ok_or_else(|| WeightsError::Missing(key.to_string()))
    }

    pub fn get_usize(&self, key: &str) -> Result<usize, WeightsError> {
        Ok(self.get(key)?.first().copied().unwrap_or(0.0) as usize)
    }

    pub fn decoder_weights(&self) -> Result<DecoderWeights, WeightsError> {
        let raw = self.get("decoder.deconv.weight")?;
        let in_ch = self.latent_channels;
        let kernel = self.get_usize("dec_kernel").unwrap_or(320);
        let stride = self.get_usize("dec_stride").unwrap_or(160);
        if raw.len() != in_ch * kernel {
            return Err(WeightsError::Npz(format!(
                "decoder.deconv.weight len {} != in_ch*kernel {}",
                raw.len(),
                in_ch * kernel
            )));
        }
        let bias = self.get("decoder.deconv.bias")?.first().copied().unwrap_or(0.0);
        Ok(DecoderWeights {
            weight: raw.to_vec(),
            bias,
            in_ch,
            kernel,
            stride,
        })
    }
}

#[derive(Clone)]
pub struct DecoderWeights {
    pub weight: Vec<f32>,
    pub bias: f32,
    pub in_ch: usize,
    pub kernel: usize,
    pub stride: usize,
}
