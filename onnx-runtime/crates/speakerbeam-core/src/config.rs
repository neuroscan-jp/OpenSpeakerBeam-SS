/// Runtime constants aligned with Python `inference.py` / ep110.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub sample_rate: u32,
    pub embed_dim: usize,
    pub enroll_segment_samples: usize,
    pub enroll_max_segments: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            embed_dim: 192,
            enroll_segment_samples: 5 * 16_000,
            enroll_max_segments: 4,
        }
    }
}
