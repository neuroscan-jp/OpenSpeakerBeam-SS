//! Aggregate small input chunks (e.g. Opus 60 ms frames) before SpeakerBeam inference.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ChunkBufferError {
    #[error("chunk length {0} exceeds input_chunk_samples {1}")]
    ChunkTooLong(usize, usize),
}

/// Buffers `input_chunk_ms` samples per call; emits a window every `process_every_n_chunks`.
///
/// Example: Opus 60 ms @ 16 kHz → 960 samples/chunk; `process_every_n_chunks=2`
/// yields 120 ms (1920 samples) per inference step.
#[derive(Debug)]
pub struct ChunkAggregator {
    input_chunk_samples: usize,
    process_every_n_chunks: usize,
    chunks_received: usize,
    buffer: Vec<f32>,
}

impl ChunkAggregator {
    pub fn from_ms(sample_rate: u32, input_chunk_ms: f32, process_every_n_chunks: usize) -> Self {
        let input_chunk_samples = ms_to_samples(sample_rate, input_chunk_ms);
        Self {
            input_chunk_samples,
            process_every_n_chunks: process_every_n_chunks.max(1),
            chunks_received: 0,
            buffer: Vec::new(),
        }
    }

    pub fn input_chunk_samples(&self) -> usize {
        self.input_chunk_samples
    }

    pub fn process_window_samples(&self) -> usize {
        self.input_chunk_samples * self.process_every_n_chunks
    }

    pub fn process_window_ms(&self, sample_rate: u32) -> f32 {
        self.process_window_samples() as f32 * 1000.0 / sample_rate as f32
    }

    /// Push one input chunk (typically fixed 60 ms). Returns a full process window when ready.
    pub fn push_chunk(&mut self, chunk: &[f32]) -> Result<Option<Vec<f32>>, ChunkBufferError> {
        if chunk.len() > self.input_chunk_samples {
            return Err(ChunkBufferError::ChunkTooLong(
                chunk.len(),
                self.input_chunk_samples,
            ));
        }
        self.buffer.extend_from_slice(chunk);
        if chunk.len() == self.input_chunk_samples {
            self.chunks_received += 1;
        }
        if self.chunks_received >= self.process_every_n_chunks {
            let n = self.process_window_samples().min(self.buffer.len());
            if n == 0 {
                return Ok(None);
            }
            let out = self.buffer.drain(..n).collect();
            self.chunks_received = 0;
            Ok(Some(out))
        } else {
            Ok(None)
        }
    }

    /// Drain any remaining samples (end of stream).
    pub fn flush(&mut self) -> Vec<f32> {
        self.chunks_received = 0;
        self.buffer.drain(..).collect()
    }
}

pub fn ms_to_samples(sample_rate: u32, ms: f32) -> usize {
    ((sample_rate as f32) * ms / 1000.0).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opus_60ms_two_chunks() {
        let mut agg = ChunkAggregator::from_ms(16_000, 60.0, 2);
        assert_eq!(agg.input_chunk_samples(), 960);
        assert_eq!(agg.process_window_samples(), 1920);
        assert!(agg.push_chunk(&vec![0.0; 960]).unwrap().is_none());
        let win = agg.push_chunk(&vec![0.0; 960]).unwrap();
        assert_eq!(win.as_ref().map(|w| w.len()), Some(1920));
    }

    #[test]
    fn opus_60ms_three_chunks() {
        let mut agg = ChunkAggregator::from_ms(16_000, 60.0, 3);
        assert_eq!(agg.process_window_samples(), 2880);
        agg.push_chunk(&vec![0.0; 960]).unwrap();
        agg.push_chunk(&vec![0.0; 960]).unwrap();
        let win = agg.push_chunk(&vec![0.0; 960]).unwrap();
        assert_eq!(win.unwrap().len(), 2880);
    }
}
