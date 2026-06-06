use super::ops::{
    add_residual_into, conv1d, conv1d_into, conv1d_into_from, cum_ln, cum_ln_into_from,
    cum_ln_state_at, gelu, glu_into, layer_norm_per_frame_into,
    pointwise_matvec_into, prelu_channel, cmul,
};
use super::weights::{SeparatorWeights, WeightsError};

fn resize_buf(buf: &mut Vec<f32>, len: usize) {
    if buf.len() != len {
        buf.resize(len, 0.0);
    }
}

fn copy_channel_major_prefix(
    dst: &mut [f32],
    src: &[f32],
    channels: usize,
    dst_frames: usize,
    n_copy: usize,
    src_frames: usize,
) {
    for c in 0..channels {
        let db = c * dst_frames;
        let sb = c * src_frames;
        dst[db..db + n_copy].copy_from_slice(&src[sb..sb + n_copy]);
    }
}

fn extend_cm_cache(buf: &mut Vec<f32>, channels: usize, n_prev: usize, n_frames: usize) {
    let old = std::mem::take(buf);
    resize_buf(buf, channels * n_frames);
    if n_prev > 0 && old.len() >= channels * n_prev {
        copy_channel_major_prefix(buf, &old, channels, n_frames, n_prev, n_prev);
    }
}

fn apply_film_tail(
    sc: usize,
    n_prev: usize,
    n_frames: usize,
    x_new: &[f32],
    mul: &[f32],
    add: &[f32],
    out: &mut [f32],
) {
    let n_new = n_frames - n_prev;
    for c in 0..sc {
        let m = mul[c];
        let a = add[c];
        for f in 0..n_new {
            out[c * n_frames + (n_prev + f)] = m * x_new[c * n_new + f] + a;
        }
    }
}

struct SeparatorScratch {
    x: Vec<f32>,
    x_ln: Vec<f32>,
    depth_raw: Vec<f32>,
    frame: Vec<f32>,
    sep_tail: Vec<f32>,
    s4d_tail: Vec<f32>,
    s4d_ln: Vec<f32>,
    s4d_core: Vec<f32>,
    s4d_wide: Vec<f32>,
    s4d_linear1: Vec<f32>,
    s4d_cat: Vec<f32>,
    s4d_glu_z: Vec<f32>,
    s4d_glu: Vec<f32>,
    s4d_a: Vec<f32>,
    s4d_ln_ff: Vec<f32>,
    s4d_ff1: Vec<f32>,
    s4d_ff2: Vec<f32>,
    s4d_out: Vec<f32>,
}

struct InConvCache {
    n_frames: usize,
    x: Vec<f32>,
}

struct ConvBlockCache {
    n_frames: usize,
    after_in_conv: Vec<f32>,
    after_cumln1: Vec<f32>,
    after_depth: Vec<f32>,
    after_cumln2: Vec<f32>,
    output: Vec<f32>,
}

struct ConvBlockIntermediates {
    after_in_conv: Vec<f32>,
    after_cumln1: Vec<f32>,
    after_depth: Vec<f32>,
    after_cumln2: Vec<f32>,
}

pub struct NativeSeparatorStream {
    weights: SeparatorWeights,
    s4d_states: Vec<Vec<f32>>,
    block_caches: Vec<Vec<f32>>,
    conv_block_caches: Vec<ConvBlockCache>,
    in_conv_cache: Option<InConvCache>,
    spk_mul: Vec<f32>,
    spk_add: Vec<f32>,
    scratch: SeparatorScratch,
}

impl NativeSeparatorStream {
    pub fn from_weights(weights: SeparatorWeights) -> Self {
        let n = weights.num_blocks;
        Self {
            weights,
            s4d_states: (0..n).map(|_| Vec::new()).collect(),
            block_caches: (0..n).map(|_| Vec::new()).collect(),
            in_conv_cache: None,
            conv_block_caches: (0..n)
                .map(|_| ConvBlockCache {
                    n_frames: 0,
                    after_in_conv: Vec::new(),
                    after_cumln1: Vec::new(),
                    after_depth: Vec::new(),
                    after_cumln2: Vec::new(),
                    output: Vec::new(),
                })
                .collect(),
            spk_mul: Vec::new(),
            spk_add: Vec::new(),
            scratch: SeparatorScratch {
                x: Vec::new(),
                x_ln: Vec::new(),
                depth_raw: Vec::new(),
                frame: Vec::new(),
                sep_tail: Vec::new(),
                s4d_tail: Vec::new(),
                s4d_ln: Vec::new(),
                s4d_core: Vec::new(),
                s4d_wide: Vec::new(),
                s4d_linear1: Vec::new(),
                s4d_cat: Vec::new(),
                s4d_glu_z: Vec::new(),
                s4d_glu: Vec::new(),
                s4d_a: Vec::new(),
                s4d_ln_ff: Vec::new(),
                s4d_ff1: Vec::new(),
                s4d_ff2: Vec::new(),
                s4d_out: Vec::new(),
            },
        }
    }

    pub fn reset(&mut self) {
        for s in &mut self.s4d_states {
            s.clear();
        }
        for c in &mut self.block_caches {
            c.clear();
        }
        self.in_conv_cache = None;
        for c in &mut self.conv_block_caches {
            c.n_frames = 0;
            c.after_in_conv.clear();
            c.after_cumln1.clear();
            c.after_depth.clear();
            c.after_cumln2.clear();
            c.output.clear();
        }
        self.spk_mul.clear();
        self.spk_add.clear();
        self.scratch.x.clear();
        self.scratch.x_ln.clear();
        self.scratch.depth_raw.clear();
        self.scratch.frame.clear();
        self.scratch.sep_tail.clear();
        self.scratch.s4d_tail.clear();
        self.scratch.s4d_ln.clear();
        self.scratch.s4d_core.clear();
        self.scratch.s4d_wide.clear();
        self.scratch.s4d_linear1.clear();
        self.scratch.s4d_cat.clear();
        self.scratch.s4d_glu_z.clear();
        self.scratch.s4d_glu.clear();
        self.scratch.s4d_a.clear();
        self.scratch.s4d_ln_ff.clear();
        self.scratch.s4d_ff1.clear();
        self.scratch.s4d_ff2.clear();
        self.scratch.s4d_out.clear();
    }

    fn merge_film_block(
        &mut self,
        block: usize,
        sc: usize,
        n_prev: usize,
        n_frames: usize,
    ) {
        if n_prev > 0 {
            extend_cm_cache(&mut self.block_caches[block], sc, n_prev, n_frames);
            apply_film_tail(
                sc,
                n_prev,
                n_frames,
                &self.scratch.s4d_tail,
                &self.spk_mul,
                &self.spk_add,
                &mut self.block_caches[block],
            );
        } else {
            resize_buf(&mut self.block_caches[block], sc * n_frames);
            apply_film_tail(
                sc,
                0,
                n_frames,
                &self.scratch.s4d_tail,
                &self.spk_mul,
                &self.spk_add,
                &mut self.block_caches[block],
            );
        }
        resize_buf(&mut self.scratch.x, sc * n_frames);
        self.scratch
            .x
            .copy_from_slice(&self.block_caches[block]);
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

        self.prepare_in_conv(latent_fm, n_frames, n_prev)?;

        for block in 0..self.weights.num_blocks {
            if self.s4d_states[block].is_empty() {
                self.s4d_states[block] = self.init_s4d_state(block);
            }
            self.forward_conv_block_cached(block, n_frames, n_prev)?;
            self.forward_s4d_block_stream(block, n_prev, n_frames)?;
            self.merge_film_block(block, sc, n_prev, n_frames);
        }

        let out_start = if n_prev > 0 { n_prev } else { 0 };
        let n_out = n_frames - out_start;
        resize_buf(&mut self.scratch.sep_tail, lc * n_out);
        let x = self.scratch.x.as_slice();
        let sep_tail = &mut self.scratch.sep_tail;
        Self::finish_out_conv(
            &self.weights,
            x,
            latent_fm,
            n_frames,
            out_start,
            sep_tail,
        )?;
        Ok(std::mem::take(&mut self.scratch.sep_tail))
    }

    fn prepare_in_conv(
        &mut self,
        latent_fm: &[f32],
        n_frames: usize,
        n_prev: usize,
    ) -> Result<(), WeightsError> {
        let lc = self.weights.latent_channels;
        let sc = self.weights.sep_channels;
        let ln_in_w = self.weights.get("layer_norm_in.weight")?;
        let ln_in_b = self.weights.get("layer_norm_in.bias")?;
        let w = self.weights.get("in_conv1x1.weight")?;
        let b = self.weights.get("in_conv1x1.bias")?;

        let start = if n_prev > 0
            && self
                .in_conv_cache
                .as_ref()
                .map(|c| c.n_frames == n_prev)
                .unwrap_or(false)
        {
            n_prev
        } else {
            0
        };

        resize_buf(&mut self.scratch.x_ln, lc * n_frames);
        resize_buf(&mut self.scratch.frame, lc);
        for f in start..n_frames {
            let frame = &latent_fm[f * lc..(f + 1) * lc];
            layer_norm_per_frame_into(&mut self.scratch.frame, frame, ln_in_w, ln_in_b, lc);
            for c in 0..lc {
                self.scratch.x_ln[c * n_frames + f] = self.scratch.frame[c];
            }
        }

        resize_buf(&mut self.scratch.x, sc * n_frames);
        if start > 0 {
            let cache = self.in_conv_cache.as_ref().unwrap();
            copy_channel_major_prefix(
                &mut self.scratch.x,
                &cache.x,
                sc,
                n_frames,
                start,
                n_prev,
            );
            for f in start..n_frames {
                for oc in 0..sc {
                    let mut acc = b[oc];
                    let w_row = oc * lc;
                    for ic in 0..lc {
                        acc += w[w_row + ic] * self.scratch.x_ln[ic * n_frames + f];
                    }
                    self.scratch.x[oc * n_frames + f] = acc;
                }
            }
        } else {
            conv1d_into(
                &mut self.scratch.x,
                &self.scratch.x_ln,
                w,
                Some(b),
                lc,
                sc,
                n_frames,
                n_frames,
                1,
                1,
                0,
                1,
                1,
            );
        }

        if let Some(cache) = &mut self.in_conv_cache {
            resize_buf(&mut cache.x, sc * n_frames);
            cache.x.copy_from_slice(&self.scratch.x);
            cache.n_frames = n_frames;
        } else {
            self.in_conv_cache = Some(InConvCache {
                n_frames,
                x: self.scratch.x.clone(),
            });
        }
        Ok(())
    }

    fn finish_out_conv(
        weights: &SeparatorWeights,
        x: &[f32],
        latent_fm: &[f32],
        n_frames: usize,
        start_f: usize,
        sep_fm: &mut [f32],
    ) -> Result<(), WeightsError> {
        let lc = weights.latent_channels;
        let sc = weights.sep_channels;
        let ln_out_w = weights.get("layer_norm_out.weight")?;
        let ln_out_b = weights.get("layer_norm_out.bias")?;

        if start_f == 0 {
            let mut out = vec![0.0f32; lc * n_frames];
            conv1d_into(
                &mut out,
                x,
                weights.get("out_conv1x1.weight")?,
                Some(weights.get("out_conv1x1.bias")?),
                sc,
                lc,
                n_frames,
                n_frames,
                1,
                1,
                0,
                1,
                1,
            );
            let mut frame = vec![0.0f32; lc];
            let mut normed = vec![0.0f32; lc];
            for f in 0..n_frames {
                for c in 0..lc {
                    frame[c] = out[c * n_frames + f];
                }
                layer_norm_per_frame_into(&mut normed, &frame, ln_out_w, ln_out_b, lc);
                for c in 0..lc {
                    sep_fm[f * lc + c] = normed[c].max(0.0) * latent_fm[f * lc + c];
                }
            }
            return Ok(());
        }

        let w = weights.get("out_conv1x1.weight")?;
        let b = weights.get("out_conv1x1.bias")?;
        let mut frame = vec![0.0f32; lc];
        let mut normed = vec![0.0f32; lc];
        for f in start_f..n_frames {
            for oc in 0..lc {
                let mut acc = b[oc];
                let w_row = oc * sc;
                for ic in 0..sc {
                    acc += w[w_row + ic] * x[ic * n_frames + f];
                }
                frame[oc] = acc;
            }
            layer_norm_per_frame_into(&mut normed, &frame, ln_out_w, ln_out_b, lc);
            let out_f = f - start_f;
            for c in 0..lc {
                sep_fm[out_f * lc + c] = normed[c].max(0.0) * latent_fm[f * lc + c];
            }
        }
        Ok(())
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

    fn forward_conv_block_cached(
        &mut self,
        block: usize,
        n_frames: usize,
        n_prev: usize,
    ) -> Result<(), WeightsError> {
        let cache = &self.conv_block_caches[block];
        if n_prev > 0 && cache.n_frames == n_prev {
            return self.forward_conv_block_incr(block, n_frames, n_prev);
        }
        let x = std::mem::take(&mut self.scratch.x);
        let (out, inter) = self.forward_conv_block_staged(block, &x, n_frames)?;
        let cache = &mut self.conv_block_caches[block];
        cache.n_frames = n_frames;
        cache.after_in_conv = inter.after_in_conv;
        cache.after_cumln1 = inter.after_cumln1;
        cache.after_depth = inter.after_depth;
        cache.after_cumln2 = inter.after_cumln2;
        resize_buf(&mut cache.output, out.len());
        cache.output.copy_from_slice(&out);
        self.scratch.x = out;
        Ok(())
    }

    fn forward_conv_block_incr(
        &mut self,
        block: usize,
        n_frames: usize,
        n_prev: usize,
    ) -> Result<(), WeightsError> {
        let x = &self.scratch.x;
        let p = format!("blocks.{block}.conv");
        let sc = self.weights.sep_channels;
        let hc = self.weights.hidden_channels;
        let k = self.weights.get_usize(&format!("{p}.depth_kernel"))?;
        let dilation = self.weights.get_usize(&format!("{p}.depth_dilation"))?;
        let padding = self.weights.get_usize(&format!("{p}.depth_padding"))?;
        let chop = self.weights.get_usize(&format!("{p}.chop_size"))?;
        let in_w = self.weights.get(&format!("{p}.in_conv.weight"))?;
        let in_b = self.weights.get(&format!("{p}.in_conv.bias"))?;
        let prelu1 = self.weights.get(&format!("{p}.prelu1.weight"))?;
        let g1 = self.weights.get(&format!("{p}.cgln1.gamma"))?;
        let b1 = self.weights.get(&format!("{p}.cgln1.beta"))?;
        let prelu2 = self.weights.get(&format!("{p}.prelu2.weight"))?;
        let g2 = self.weights.get(&format!("{p}.cgln2.gamma"))?;
        let b2 = self.weights.get(&format!("{p}.cgln2.beta"))?;
        let res_w = self.weights.get(&format!("{p}.res_conv.weight"))?;
        let res_b = self.weights.get(&format!("{p}.res_conv.bias"))?;

        let cache = &mut self.conv_block_caches[block];
        extend_cm_cache(&mut cache.after_in_conv, hc, n_prev, n_frames);
        for t in n_prev..n_frames {
            for oc in 0..hc {
                let mut acc = in_b[oc];
                let w_row = oc * sc;
                for ic in 0..sc {
                    acc += in_w[w_row + ic] * x[ic * n_frames + t];
                }
                cache.after_in_conv[oc * n_frames + t] = prelu_channel(acc, prelu1, oc);
            }
        }

        extend_cm_cache(&mut cache.after_cumln1, hc, n_prev, n_frames);
        let mut c1 = cum_ln_state_at(&cache.after_in_conv, hc, n_frames, n_prev);
        cum_ln_into_from(
            &mut cache.after_cumln1,
            &cache.after_in_conv,
            g1,
            b1,
            hc,
            n_frames,
            n_prev,
            &mut c1,
        );

        let eff_k = (k - 1) * dilation + 1;
        let depth_raw_len = (n_frames + 2 * padding).saturating_sub(eff_k) + 1;
        let depth_len = if chop > 0 && depth_raw_len > chop {
            depth_raw_len - chop
        } else {
            depth_raw_len
        };
        resize_buf(&mut self.scratch.depth_raw, hc * depth_raw_len);
        conv1d_into_from(
            &mut self.scratch.depth_raw,
            &cache.after_cumln1,
            self.weights.get(&format!("{p}.depth_conv.weight"))?,
            Some(self.weights.get(&format!("{p}.depth_conv.bias"))?),
            hc,
            hc,
            n_frames,
            depth_raw_len,
            k,
            1,
            padding,
            dilation,
            hc,
            n_prev,
        );

        let prev_depth_raw_len =
            (n_prev + 2 * padding).saturating_sub(eff_k) + 1;
        let prev_depth_len = if chop > 0 && prev_depth_raw_len > chop {
            prev_depth_raw_len - chop
        } else {
            prev_depth_raw_len
        };
        let old_depth = std::mem::take(&mut cache.after_depth);
        resize_buf(&mut cache.after_depth, hc * depth_len);
        let depth_copy = n_prev.min(depth_len).min(prev_depth_len);
        if depth_copy > 0 && old_depth.len() >= hc * prev_depth_len {
            copy_channel_major_prefix(
                &mut cache.after_depth,
                &old_depth,
                hc,
                depth_len,
                depth_copy,
                prev_depth_len,
            );
        }
        for t in n_prev..depth_len {
            for c in 0..hc {
                let v = self.scratch.depth_raw[c * depth_raw_len + t];
                cache.after_depth[c * depth_len + t] = prelu_channel(v, prelu2, c);
            }
        }

        let old_depth_cum = std::mem::take(&mut cache.after_cumln2);
        resize_buf(&mut cache.after_cumln2, hc * depth_len);
        if depth_copy > 0 && old_depth_cum.len() >= hc * prev_depth_len {
            copy_channel_major_prefix(
                &mut cache.after_cumln2,
                &old_depth_cum,
                hc,
                depth_len,
                depth_copy,
                prev_depth_len,
            );
        }
        let mut c2 = cum_ln_state_at(&cache.after_depth, hc, depth_len, n_prev);
        cum_ln_into_from(
            &mut cache.after_cumln2,
            &cache.after_depth,
            g2,
            b2,
            hc,
            depth_len,
            n_prev,
            &mut c2,
        );

        extend_cm_cache(&mut cache.output, sc, n_prev, n_frames);
        for t in n_prev..n_frames.min(depth_len) {
            for oc in 0..sc {
                let mut acc = res_b[oc];
                let w_row = oc * hc;
                for ic in 0..hc {
                    acc += res_w[w_row + ic] * cache.after_cumln2[ic * depth_len + t];
                }
                cache.output[oc * n_frames + t] = acc;
            }
        }

        cache.n_frames = n_frames;
        resize_buf(&mut self.scratch.x, sc * n_frames);
        self.scratch.x.copy_from_slice(&cache.output);
        Ok(())
    }

    fn forward_conv_block_staged(
        &self,
        block: usize,
        x: &[f32],
        n_frames: usize,
    ) -> Result<(Vec<f32>, ConvBlockIntermediates), WeightsError> {
        let inter = self.forward_conv_block_intermediates(block, x, n_frames)?;
        let depth_len = inter.after_depth.len() / self.weights.hidden_channels;
        let out = self.forward_conv_block_res(&inter.after_cumln2, block, depth_len)?;
        Ok((out, inter))
    }

    fn forward_conv_block_intermediates(
        &self,
        block: usize,
        x: &[f32],
        n_frames: usize,
    ) -> Result<ConvBlockIntermediates, WeightsError> {
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
        let after_in_conv = y.clone();
        let g1 = self.weights.get(&format!("{p}.cgln1.gamma"))?;
        let b1 = self.weights.get(&format!("{p}.cgln1.beta"))?;
        y = cum_ln(&y, g1, b1, hc, n_frames);
        let after_cumln1 = y.clone();

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
        let after_depth = depth.clone();
        let g2 = self.weights.get(&format!("{p}.cgln2.gamma"))?;
        let b2 = self.weights.get(&format!("{p}.cgln2.beta"))?;
        let after_cumln2 = cum_ln(&depth, g2, b2, hc, depth_len);

        Ok(ConvBlockIntermediates {
            after_in_conv,
            after_cumln1,
            after_depth,
            after_cumln2,
        })
    }

    fn forward_conv_block_res(
        &self,
        depth_cum: &[f32],
        block: usize,
        depth_len: usize,
    ) -> Result<Vec<f32>, WeightsError> {
        let p = format!("blocks.{block}.conv");
        let sc = self.weights.sep_channels;
        let hc = self.weights.hidden_channels;
        Ok(conv1d(
            depth_cum,
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

    fn forward_conv_block(
        &self,
        block: usize,
        x: &[f32],
        n_frames: usize,
    ) -> Result<Vec<f32>, WeightsError> {
        let inter = self.forward_conv_block_intermediates(block, x, n_frames)?;
        let depth_len = inter.after_depth.len() / self.weights.hidden_channels;
        self.forward_conv_block_res(&inter.after_cumln2, block, depth_len)
    }

    fn forward_s4d_block_stream(
        &mut self,
        block: usize,
        n_prev: usize,
        n_frames: usize,
    ) -> Result<(), WeightsError> {
        let sc = self.weights.sep_channels;
        let n_new = n_frames - n_prev;
        resize_buf(&mut self.scratch.s4d_tail, sc * n_new);
        if self.s4d_states[block].is_empty() {
            self.s4d_states[block] = self.init_s4d_state(block);
        }
        resize_buf(&mut self.scratch.frame, sc);
        resize_buf(&mut self.scratch.s4d_ln, sc);
        resize_buf(&mut self.scratch.s4d_core, sc);
        resize_buf(&mut self.scratch.s4d_wide, 2 * sc);
        resize_buf(&mut self.scratch.s4d_linear1, sc);
        resize_buf(&mut self.scratch.s4d_cat, 2 * sc);
        resize_buf(&mut self.scratch.s4d_glu_z, 2 * sc);
        resize_buf(&mut self.scratch.s4d_glu, sc);
        resize_buf(&mut self.scratch.s4d_a, sc);
        resize_buf(&mut self.scratch.s4d_ln_ff, sc);
        resize_buf(&mut self.scratch.s4d_ff1, sc);
        resize_buf(&mut self.scratch.s4d_ff2, sc);
        resize_buf(&mut self.scratch.s4d_out, sc);

        let p = format!("blocks.{block}.s4d");
        let n = self.weights.get_usize(&format!("{p}.s4d.n"))?;
        let half = n / 2;
        for t in 0..n_new {
            let ft = n_prev + t;
            for c in 0..sc {
                self.scratch.frame[c] = self.scratch.x[c * n_frames + ft];
            }
            let x_t = self.scratch.frame.clone();
            let mut h = std::mem::take(&mut self.s4d_states[block]);
            self.s4d_block_step_into(&p, half, &x_t, &mut h)?;
            self.s4d_states[block] = h;
            for c in 0..sc {
                self.scratch.s4d_tail[c * n_new + t] = self.scratch.s4d_out[c];
            }
        }
        Ok(())
    }

    fn s4d_block_step_into(
        &mut self,
        p: &str,
        half: usize,
        x_t: &[f32],
        h: &mut [f32],
    ) -> Result<(), WeightsError> {
        let sc = self.weights.sep_channels;
        let s4d_prefix = format!("{p}.s4d");

        layer_norm_per_frame_into(
            &mut self.scratch.s4d_ln,
            x_t,
            self.weights.get(&format!("{p}.ln_s4d.weight"))?,
            self.weights.get(&format!("{p}.ln_s4d.bias"))?,
            sc,
        );
        Self::s4d_step_into(
            &self.weights,
            &s4d_prefix,
            &self.scratch.s4d_ln,
            h,
            &mut self.scratch.s4d_core,
            &mut self.scratch.s4d_wide,
            sc,
            half,
        )?;
        for v in &mut self.scratch.s4d_core {
            *v = gelu(*v);
        }
        pointwise_matvec_into(
            &mut self.scratch.s4d_linear1,
            &self.scratch.s4d_core,
            self.weights.get(&format!("{p}.linear1.weight"))?,
            Some(self.weights.get(&format!("{p}.linear1.bias"))?),
            sc,
            sc,
        );

        for c in 0..sc {
            self.scratch.s4d_cat[c] = x_t[c];
            self.scratch.s4d_cat[c + sc] = self.scratch.s4d_linear1[c];
        }
        pointwise_matvec_into(
            &mut self.scratch.s4d_glu_z,
            &self.scratch.s4d_cat,
            self.weights.get(&format!("{p}.glu_conv.weight"))?,
            Some(self.weights.get(&format!("{p}.glu_conv.bias"))?),
            2 * sc,
            2 * sc,
        );
        glu_into(&mut self.scratch.s4d_glu, &self.scratch.s4d_glu_z, 2 * sc, 1);
        add_residual_into(&mut self.scratch.s4d_a, x_t, &self.scratch.s4d_glu);

        layer_norm_per_frame_into(
            &mut self.scratch.s4d_ln_ff,
            &self.scratch.s4d_a,
            self.weights.get(&format!("{p}.ln_ff2.weight"))?,
            self.weights.get(&format!("{p}.ln_ff2.bias"))?,
            sc,
        );
        pointwise_matvec_into(
            &mut self.scratch.s4d_ff1,
            &self.scratch.s4d_ln_ff,
            self.weights.get(&format!("{p}.ff2_linear1.weight"))?,
            Some(self.weights.get(&format!("{p}.ff2_linear1.bias"))?),
            sc,
            sc,
        );
        for v in &mut self.scratch.s4d_ff1 {
            *v = gelu(*v);
        }
        pointwise_matvec_into(
            &mut self.scratch.s4d_ff2,
            &self.scratch.s4d_ff1,
            self.weights.get(&format!("{p}.ff2_linear2.weight"))?,
            Some(self.weights.get(&format!("{p}.ff2_linear2.bias"))?),
            sc,
            sc,
        );
        add_residual_into(&mut self.scratch.s4d_out, &self.scratch.s4d_a, &self.scratch.s4d_ff2);
        Ok(())
    }

    fn s4d_step_into(
        weights: &SeparatorWeights,
        prefix: &str,
        u: &[f32],
        h: &mut [f32],
        core: &mut [f32],
        wide: &mut [f32],
        h_dim: usize,
        half: usize,
    ) -> Result<(), WeightsError> {
        let a_bar = weights.get(&format!("{prefix}._A_bar"))?;
        let b_bar = weights.get(&format!("{prefix}._B_bar"))?;
        let c = weights.get(&format!("{prefix}._C"))?;
        let d = weights.get(&format!("{prefix}._D"))?;

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
                h[idx] = hnr + bnr;
                h[idx + 1] = hni + bni;
                let (pr, _pi) = cmul(cr, ci, h[idx], h[idx + 1]);
                yr += 2.0 * pr;
            }
            core[hi] = yr + d[hi] * u[hi];
        }
        for v in core.iter_mut() {
            *v = gelu(*v);
        }
        pointwise_matvec_into(
            wide,
            core,
            weights.get(&format!("{prefix}.output_linear.0.weight"))?,
            Some(weights.get(&format!("{prefix}.output_linear.0.bias"))?),
            h_dim,
            2 * h_dim,
        );
        glu_into(core, wide, 2 * h_dim, 1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ops::layer_norm_per_frame;
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
