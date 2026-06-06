use super::ops::{
    add_residual, conv1d, cum_ln, gelu, glu, layer_norm_per_frame, prelu_channel, cmul,
};
use super::weights::{SeparatorWeights, WeightsError};

pub struct NativeSeparatorStream {
    weights: SeparatorWeights,
    s4d_states: Vec<Vec<f32>>,
    block_caches: Vec<Option<Vec<f32>>>,
    spk_mul: Vec<f32>,
    spk_add: Vec<f32>,
}

impl NativeSeparatorStream {
    pub fn from_weights(weights: SeparatorWeights) -> Self {
        let n = weights.num_blocks;
        Self {
            weights,
            s4d_states: (0..n).map(|_| Vec::new()).collect(),
            block_caches: (0..n).map(|_| None).collect(),
            spk_mul: Vec::new(),
            spk_add: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        for s in &mut self.s4d_states {
            s.clear();
        }
        for c in &mut self.block_caches {
            *c = None;
        }
        self.spk_mul.clear();
        self.spk_add.clear();
    }

    pub fn set_embedding(&mut self, emb: &[f32]) -> Result<(), WeightsError> {
        let (mul, add) = self.spk_film(emb)?;
        self.spk_mul = mul;
        self.spk_add = add;
        Ok(())
    }

    fn init_s4d_state(&self, block: usize) -> Vec<f32> {
        let h = self.weights.sep_channels;
        let n = self.weights.get_usize(&format!("blocks.{block}.s4d.s4d.n")).unwrap_or(32);
        vec![0.0; h * (n / 2) * 2]
    }

    /// Frame-major latent `(n_frames * latent_channels)` → frame-major separated latent.
    pub fn forward(
        &mut self,
        latent_fm: &[f32],
        n_frames: usize,
        n_prev: usize,
        embedding: &[f32],
    ) -> Result<Vec<f32>, WeightsError> {
        let lc = self.weights.latent_channels;
        let sc = self.weights.sep_channels;

        if self.spk_mul.is_empty() {
            self.set_embedding(embedding)?;
        }

        let ln_in_w = self.weights.get("layer_norm_in.weight")?;
        let ln_in_b = self.weights.get("layer_norm_in.bias")?;
        let mut x = vec![0.0f32; lc * n_frames];
        for f in 0..n_frames {
            let frame = &latent_fm[f * lc..(f + 1) * lc];
            let normed = layer_norm_per_frame(frame, ln_in_w, ln_in_b, lc);
            for c in 0..lc {
                x[c * n_frames + f] = normed[c];
            }
        }

        x = conv1d(
            &x,
            self.weights.get("in_conv1x1.weight")?,
            Some(self.weights.get("in_conv1x1.bias")?),
            lc,
            sc,
            n_frames,
            1,
            1,
            0,
            1,
            1,
        );

        for block in 0..self.weights.num_blocks {
            if self.s4d_states[block].is_empty() {
                self.s4d_states[block] = self.init_s4d_state(block);
            }
            x = self.forward_conv_block(block, &x, n_frames)?;
            let (x_new, state) = self.forward_s4d_block_stream(block, &x, n_prev, n_frames)?;
            self.s4d_states[block] = state;

            let x_out = if n_prev > 0 {
                let cache = self.block_caches[block]
                    .as_ref()
                    .ok_or_else(|| WeightsError::Missing(format!("block_cache.{block}")))?;
                let mut joined = vec![0.0; sc * n_frames];
                for c in 0..sc {
                    for f in 0..n_prev {
                        joined[c * n_frames + f] = cache[c * n_prev + f];
                    }
                }
                let adapted_new =
                    self.film_drop_new(&x_new, n_prev, n_frames)?;
                for c in 0..sc {
                    for f in 0..(n_frames - n_prev) {
                        joined[c * n_frames + (n_prev + f)] =
                            adapted_new[c * (n_frames - n_prev) + f];
                    }
                }
                joined
            } else {
                self.film_drop_new(&x_new, 0, n_frames)?
            };
            self.block_caches[block] = Some(x_out.clone());
            x = x_out;
        }

        x = conv1d(
            &x,
            self.weights.get("out_conv1x1.weight")?,
            Some(self.weights.get("out_conv1x1.bias")?),
            sc,
            lc,
            n_frames,
            1,
            1,
            0,
            1,
            1,
        );

        let ln_out_w = self.weights.get("layer_norm_out.weight")?;
        let ln_out_b = self.weights.get("layer_norm_out.bias")?;
        let mut sep_fm = vec![0.0f32; lc * n_frames];
        let mut frame = vec![0.0f32; lc];
        for f in 0..n_frames {
            for c in 0..lc {
                frame[c] = x[c * n_frames + f];
            }
            let normed = layer_norm_per_frame(&frame, ln_out_w, ln_out_b, lc);
            for c in 0..lc {
                sep_fm[f * lc + c] = normed[c].max(0.0) * latent_fm[f * lc + c];
            }
        }
        Ok(sep_fm)
    }

    fn spk_film(&self, emb: &[f32]) -> Result<(Vec<f32>, Vec<f32>), WeightsError> {
        let h0 = linear(
            emb,
            self.weights.get("spk_proj.0.weight")?,
            self.weights.get("spk_proj.0.bias")?,
            self.weights.embed_dim,
            256,
        );
        let h0: Vec<f32> = h0.into_iter().map(|v| v.max(0.0)).collect();
        let h1 = linear(
            &h0,
            self.weights.get("spk_proj.3.weight")?,
            self.weights.get("spk_proj.3.bias")?,
            256,
            512,
        );
        let mut mul = vec![0.0; 256];
        let mut add = vec![0.0; 256];
        mul.copy_from_slice(&h1[..256]);
        add.copy_from_slice(&h1[256..]);
        Ok((mul, add))
    }

    fn film_drop_new(
        &self,
        x_new: &[f32],
        n_prev: usize,
        n_frames: usize,
    ) -> Result<Vec<f32>, WeightsError> {
        let sc = self.weights.sep_channels;
        let n_new = n_frames - n_prev;
        let mut out = vec![0.0; sc * n_new];
        for c in 0..sc {
            let mul = self.spk_mul[c];
            let add = self.spk_add[c];
            for f in 0..n_new {
                let v = x_new[c * n_new + f];
                out[c * n_new + f] = mul * v + add;
            }
        }
        Ok(out)
    }

    fn forward_conv_block(
        &self,
        block: usize,
        x: &[f32],
        n_frames: usize,
    ) -> Result<Vec<f32>, WeightsError> {
        let p = format!("blocks.{block}.conv");
        let sc = self.weights.sep_channels;
        let hc = self.weights.hidden_channels;
        let k = self.weights.get_usize(&format!("{p}.depth_kernel"))?;
        let dilation = self.weights.get_usize(&format!("{p}.depth_dilation"))?;
        let padding = self.weights.get_usize(&format!("{p}.depth_padding"))?;
        let chop = self.weights.get_usize(&format!("{p}.chop_size"))?;

        let mut y = conv1d(
            x,
            self.weights.get(&format!("{p}.in_conv.weight"))?,
            Some(self.weights.get(&format!("{p}.in_conv.bias"))?),
            sc,
            hc,
            n_frames,
            1,
            1,
            0,
            1,
            1,
        );
        let prelu1 = self.weights.get(&format!("{p}.prelu1.weight"))?;
        for (i, v) in y.iter_mut().enumerate() {
            let c = (i / n_frames) % hc;
            *v = prelu_channel(*v, prelu1, c);
        }
        let g1 = self.weights.get(&format!("{p}.cgln1.gamma"))?;
        let b1 = self.weights.get(&format!("{p}.cgln1.beta"))?;
        y = cum_ln(&y, g1, b1, hc, n_frames);

        let mut depth = conv1d(
            &y,
            self.weights.get(&format!("{p}.depth_conv.weight"))?,
            Some(self.weights.get(&format!("{p}.depth_conv.bias"))?),
            hc,
            hc,
            n_frames,
            k,
            1,
            padding,
            dilation,
            hc,
        );
        let mut depth_len = depth.len() / hc;
        if chop > 0 && depth_len > chop {
            depth_len -= chop;
            let mut trimmed = vec![0.0; hc * depth_len];
            for c in 0..hc {
                for t in 0..depth_len {
                    trimmed[c * depth_len + t] = depth[c * (depth_len + chop) + t];
                }
            }
            depth = trimmed;
        }
        let prelu2 = self.weights.get(&format!("{p}.prelu2.weight"))?;
        for (i, v) in depth.iter_mut().enumerate() {
            let c = (i / depth_len) % hc;
            *v = prelu_channel(*v, prelu2, c);
        }
        let g2 = self.weights.get(&format!("{p}.cgln2.gamma"))?;
        let b2 = self.weights.get(&format!("{p}.cgln2.beta"))?;
        depth = cum_ln(&depth, g2, b2, hc, depth_len);

        Ok(conv1d(
            &depth,
            self.weights.get(&format!("{p}.res_conv.weight"))?,
            Some(self.weights.get(&format!("{p}.res_conv.bias"))?),
            hc,
            sc,
            depth_len,
            1,
            1,
            0,
            1,
            1,
        ))
    }

    fn forward_s4d_block_stream(
        &self,
        block: usize,
        x: &[f32],
        n_prev: usize,
        n_frames: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), WeightsError> {
        let sc = self.weights.sep_channels;
        let n_new = n_frames - n_prev;
        let mut outs = vec![0.0; sc * n_new];
        let mut h = self.s4d_states[block].clone();
        if h.is_empty() {
            h = self.init_s4d_state(block);
        }
        let mut frame = vec![0.0f32; sc];
        for t in 0..n_new {
            for c in 0..sc {
                frame[c] = x[c * n_frames + (n_prev + t)];
            }
            let (y, h_new) = self.s4d_block_step(block, &frame, &h)?;
            h = h_new;
            for c in 0..sc {
                outs[c * n_new + t] = y[c];
            }
        }
        Ok((outs, h))
    }

    fn s4d_block_step(
        &self,
        block: usize,
        x_t: &[f32],
        h: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), WeightsError> {
        let p = format!("blocks.{block}.s4d");
        let sc = self.weights.sep_channels;
        let n = self.weights.get_usize(&format!("{p}.s4d.n"))?;
        let half = n / 2;

        let ln = layer_norm_per_frame(
            x_t,
            self.weights.get(&format!("{p}.ln_s4d.weight"))?,
            self.weights.get(&format!("{p}.ln_s4d.bias"))?,
            sc,
        );
        let (mut y, h_new) = self.s4d_step(&format!("{p}.s4d"), &ln, h, sc, half)?;
        for v in &mut y {
            *v = gelu(*v);
        }
        y = conv1d(
            &y,
            self.weights.get(&format!("{p}.linear1.weight"))?,
            Some(self.weights.get(&format!("{p}.linear1.bias"))?),
            sc,
            sc,
            1,
            1,
            1,
            0,
            1,
            1,
        );

        let mut cat = vec![0.0; 2 * sc];
        for c in 0..sc {
            cat[c] = x_t[c];
            cat[c + sc] = y[c];
        }
        let z = conv1d(
            &cat,
            self.weights.get(&format!("{p}.glu_conv.weight"))?,
            Some(self.weights.get(&format!("{p}.glu_conv.bias"))?),
            2 * sc,
            2 * sc,
            1,
            1,
            1,
            0,
            1,
            1,
        );
        let z = glu(&z, 2 * sc, 1);
        let a = add_residual(x_t, &z);

        let a_ln = layer_norm_per_frame(
            &a,
            self.weights.get(&format!("{p}.ln_ff2.weight"))?,
            self.weights.get(&format!("{p}.ln_ff2.bias"))?,
            sc,
        );
        let mut ff = conv1d(
            &a_ln,
            self.weights.get(&format!("{p}.ff2_linear1.weight"))?,
            Some(self.weights.get(&format!("{p}.ff2_linear1.bias"))?),
            sc,
            sc,
            1,
            1,
            1,
            0,
            1,
            1,
        );
        for v in &mut ff {
            *v = gelu(*v);
        }
        ff = conv1d(
            &ff,
            self.weights.get(&format!("{p}.ff2_linear2.weight"))?,
            Some(self.weights.get(&format!("{p}.ff2_linear2.bias"))?),
            sc,
            sc,
            1,
            1,
            1,
            0,
            1,
            1,
        );
        Ok((add_residual(&a, &ff), h_new))
    }

    fn s4d_step(
        &self,
        prefix: &str,
        u: &[f32],
        h: &[f32],
        h_dim: usize,
        half: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), WeightsError> {
        let a_bar = self.weights.get(&format!("{prefix}._A_bar"))?;
        let b_bar = self.weights.get(&format!("{prefix}._B_bar"))?;
        let c = self.weights.get(&format!("{prefix}._C"))?;
        let d = self.weights.get(&format!("{prefix}._D"))?;

        let mut h_new = vec![0.0; h.len()];
        let mut y = vec![0.0; h_dim];
        for hi in 0..h_dim {
            let mut yr = 0.0f32;
            for ni in 0..half {
                let idx = (hi * half + ni) * 2;
                let ar = a_bar[idx];
                let ai = a_bar[idx + 1];
                let br = b_bar[idx];
                let bi = b_bar[idx + 1];
                let cr = c[idx];
                let ci = c[idx + 1];
                let hr = h[idx];
                let hi_c = h[idx + 1];
                let (hnr, hni) = cmul(ar, ai, hr, hi_c);
                let (bnr, bni) = cmul(br, bi, u[hi], 0.0);
                h_new[idx] = hnr + bnr;
                h_new[idx + 1] = hni + bni;
                let (pr, _pi) = cmul(cr, ci, h_new[idx], h_new[idx + 1]);
                yr += 2.0 * pr;
            }
            y[hi] = yr + d[hi] * u[hi];
        }
        for v in &mut y {
            *v = gelu(*v);
        }
        y = conv1d(
            &y,
            self.weights.get(&format!("{prefix}.output_linear.0.weight"))?,
            Some(self.weights.get(&format!("{prefix}.output_linear.0.bias"))?),
            h_dim,
            2 * h_dim,
            1,
            1,
            1,
            0,
            1,
            1,
        );
        y = glu(&y, 2 * h_dim, 1);
        Ok((y, h_new))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
    }

    fn prep_after_in_conv(weights: &SeparatorWeights, latent_fm: &[f32], n_frames: usize) -> Vec<f32> {
        let lc = weights.latent_channels;
        let sc = weights.sep_channels;
        let ln_in_w = weights.get("layer_norm_in.weight").unwrap();
        let ln_in_b = weights.get("layer_norm_in.bias").unwrap();
        let mut x = vec![0.0f32; lc * n_frames];
        for f in 0..n_frames {
            let frame = &latent_fm[f * lc..(f + 1) * lc];
            let normed = layer_norm_per_frame(frame, ln_in_w, ln_in_b, lc);
            for c in 0..lc {
                x[c * n_frames + f] = normed[c];
            }
        }
        conv1d(
            &x,
            weights.get("in_conv1x1.weight").unwrap(),
            Some(weights.get("in_conv1x1.bias").unwrap()),
            lc,
            sc,
            n_frames,
            1,
            1,
            0,
            1,
            1,
        )
    }

    #[test]
    fn block0_conv_matches_python() {
        let ref_path = repo_root().join("onnx-runtime/models/sep_ref/block0_conv.npz");
        if !ref_path.exists() {
            return;
        }
        let weights =
            SeparatorWeights::from_npz(&repo_root().join("onnx-runtime/models/streaming_separator.npz"))
                .unwrap();
        let sep = NativeSeparatorStream::from_weights(weights.clone());
        let file = std::fs::File::open(ref_path).unwrap();
        let mut reader = ndarray_npy::NpzReader::new(file).unwrap();
        let x_in: ndarray::Array3<f32> = reader.by_name("x_in").unwrap();
        let x_ref: ndarray::Array3<f32> = reader.by_name("x_conv").unwrap();
        let n_frames = x_in.shape()[2];
        let sc = weights.sep_channels;
        let x_prep = prep_after_in_conv(&weights, &{
            let mut step1 = ndarray_npy::NpzReader::new(std::fs::File::open(
                repo_root().join("onnx-runtime/models/sep_ref/step1.npz"),
            ).unwrap()).unwrap();
            let lat: ndarray::Array2<f32> = step1.by_name("latent_fm").unwrap();
            lat.iter().copied().collect::<Vec<_>>()
        }, n_frames);
        let y = sep.forward_conv_block(0, &x_prep, n_frames).unwrap();
        let mut maxdiff = 0.0f32;
        for c in 0..sc {
            for t in 0..n_frames {
                let rust = y[c * n_frames + t];
                let py = x_ref[[0, c, t]];
                maxdiff = maxdiff.max((rust - py).abs());
            }
        }
        eprintln!("block0 conv maxdiff={maxdiff:.6e}");
        assert!(maxdiff < 1e-2, "block0 conv maxdiff={maxdiff:.6e}");
    }
}

fn linear(x: &[f32], w: &[f32], b: &[f32], in_d: usize, out_d: usize) -> Vec<f32> {
    (0..out_d)
        .map(|o| {
            let mut s = b[o];
            for i in 0..in_d {
                s += x[i] * w[o * in_d + i];
            }
            s
        })
        .collect()
}
