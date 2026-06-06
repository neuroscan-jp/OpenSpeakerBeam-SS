use crate::native::ops::{
    conv_transpose1d_decode, conv_transpose1d_decode_range,
    conv_transpose1d_decode_range_fm,
};
use crate::native::weights::DecoderWeights;

/// Native decoder with incremental tail recompute (parity with full PyTorch decode).
pub struct NativeDecoder {
    weights: DecoderWeights,
    wav_cache: Vec<f32>,
    decoded_sep_frames: usize,
}

impl NativeDecoder {
    pub fn from_weights(weights: DecoderWeights) -> Self {
        Self {
            weights,
            wav_cache: Vec::new(),
            decoded_sep_frames: 0,
        }
    }

    pub fn reset(&mut self) {
        self.wav_cache.clear();
        self.decoded_sep_frames = 0;
    }

    pub fn wav(&self) -> &[f32] {
        &self.wav_cache
    }

    /// Frame-major latent: `sep_fm[frame * in_ch + ch]` (matches separator output).
    pub fn decode_fm(&mut self, sep_fm: &[f32], n_frames: usize) {
        let w = &self.weights;
        if n_frames == 0 {
            self.wav_cache.clear();
            self.decoded_sep_frames = 0;
            return;
        }
        if n_frames == self.decoded_sep_frames && !self.wav_cache.is_empty() {
            return;
        }

        let out_len = (n_frames - 1) * w.stride + w.kernel;
        let n_prev = self.decoded_sep_frames;
        if n_prev == 0 || n_prev >= n_frames {
            self.wav_cache.resize(out_len, 0.0);
            conv_transpose1d_decode_range_fm(
                sep_fm,
                w.in_ch,
                n_frames,
                &w.weight,
                w.bias,
                w.kernel,
                w.stride,
                0,
                &mut self.wav_cache,
            );
            self.decoded_sep_frames = n_frames;
            return;
        }

        let out_start = (n_prev - 1) * w.stride;
        if self.wav_cache.len() < out_start {
            self.wav_cache.resize(out_start, 0.0);
        }
        self.wav_cache.resize(out_len, 0.0);
        conv_transpose1d_decode_range_fm(
            sep_fm,
            w.in_ch,
            n_frames,
            &w.weight,
            w.bias,
            w.kernel,
            w.stride,
            out_start,
            &mut self.wav_cache,
        );
        self.decoded_sep_frames = n_frames;
    }

    /// Channel-major latent: `sep_cf[c * n_frames + f]`.
    pub fn decode(&mut self, sep_cf: &[f32], n_frames: usize) -> Vec<f32> {
        let w = &self.weights;
        if n_frames == 0 {
            return Vec::new();
        }
        let n_prev = self.decoded_sep_frames;
        if n_prev == 0 || n_prev >= n_frames {
            let out = conv_transpose1d_decode(
                sep_cf,
                w.in_ch,
                n_frames,
                &w.weight,
                w.bias,
                w.kernel,
                w.stride,
            );
            self.wav_cache = out.clone();
            self.decoded_sep_frames = n_frames;
            return out;
        }

        let out_start = (n_prev - 1) * w.stride;
        let out_len = (n_frames - 1) * w.stride + w.kernel;
        let mut out = vec![0.0f32; out_len];
        if out_start > 0 && self.wav_cache.len() >= out_start {
            out[..out_start].copy_from_slice(&self.wav_cache[..out_start]);
        }
        conv_transpose1d_decode_range(
            sep_cf,
            w.in_ch,
            n_frames,
            &w.weight,
            w.bias,
            w.kernel,
            w.stride,
            out_start,
            &mut out,
        );
        self.wav_cache = out.clone();
        self.decoded_sep_frames = n_frames;
        out
    }
}
