use matrixmultiply::sgemm;
use rayon::prelude::*;

const EPS: f32 = 1e-8;
const SGEMM_MIN_OPS: usize = 64 * 64;

pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((0.7978845608 * (x + 0.044715 * x * x * x)).tanh()))
}

pub fn prelu_channel(x: f32, alpha: &[f32], channel: usize) -> f32 {
    let slope = if alpha.len() == 1 {
        alpha[0]
    } else {
        alpha[channel]
    };
    if x >= 0.0 {
        x
    } else {
        slope * x
    }
}

pub fn prelu_channels_inplace(x: &mut [f32], alpha: &[f32], channels: usize, length: usize) {
    for t in 0..length {
        for c in 0..channels {
            let idx = c * length + t;
            let v = x[idx];
            if v < 0.0 {
                x[idx] = prelu_channel(v, alpha, c);
            }
        }
    }
}

pub fn layer_norm_per_frame(x: &[f32], gamma: &[f32], beta: &[f32], c: usize) -> Vec<f32> {
    let mean = x.iter().sum::<f32>() / c as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / c as f32;
    let inv = 1.0 / (var + EPS).sqrt();
    (0..c)
        .map(|i| gamma[i] * (x[i] - mean) * inv + beta[i])
        .collect()
}

pub fn layer_norm_per_frame_into(
    out: &mut [f32],
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    c: usize,
) {
    let mean = x.iter().sum::<f32>() / c as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / c as f32;
    let inv = 1.0 / (var + EPS).sqrt();
    for i in 0..c {
        out[i] = gamma[i] * (x[i] - mean) * inv + beta[i];
    }
}

pub fn cum_ln(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    length: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; channels * length];
    cum_ln_into(&mut out, x, gamma, beta, channels, length);
    out
}

#[derive(Clone, Copy, Default, Debug)]
pub struct CumLnState {
    pub cum_sum: f32,
    pub cum_pow: f32,
}

pub fn cum_ln_state_at(
    x: &[f32],
    channels: usize,
    length: usize,
    end_t: usize,
) -> CumLnState {
    let mut state = CumLnState::default();
    if end_t == 0 {
        return state;
    }
    for t in 0..end_t {
        let mut frame_sum = 0.0f32;
        let mut frame_pow = 0.0f32;
        for c in 0..channels {
            let v = x[c * length + t];
            frame_sum += v;
            frame_pow += v * v;
        }
        state.cum_sum += frame_sum;
        state.cum_pow += frame_pow;
    }
    state
}

pub fn cum_ln_into(
    out: &mut [f32],
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    length: usize,
) {
    cum_ln_into_from(out, x, gamma, beta, channels, length, 0, &mut CumLnState::default());
}

pub fn cum_ln_into_from(
    out: &mut [f32],
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    length: usize,
    start_t: usize,
    state: &mut CumLnState,
) {
    let mut cum_sum = state.cum_sum;
    let mut cum_pow = state.cum_pow;
    for t in start_t..length {
        let mut frame_sum = 0.0f32;
        let mut frame_pow = 0.0f32;
        for c in 0..channels {
            let v = x[c * length + t];
            frame_sum += v;
            frame_pow += v * v;
        }
        cum_sum += frame_sum;
        cum_pow += frame_pow;
        let cnt = channels as f32 * (t + 1) as f32;
        let mean = cum_sum / cnt;
        let var = cum_pow / cnt - mean * mean;
        let inv = 1.0 / (var + EPS).sqrt();
        for c in 0..channels {
            let idx = c * length + t;
            out[idx] = gamma[c] * ((x[idx] - mean) * inv) + beta[c];
        }
    }
    state.cum_sum = cum_sum;
    state.cum_pow = cum_pow;
}

/// Conv1d: input (in_ch, in_len), weight (out_ch, in_ch/groups, k), bias optional.
pub fn conv1d(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
    k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Vec<f32> {
    let eff_k = (k - 1) * dilation + 1;
    let out_len = (in_len + 2 * padding).saturating_sub(eff_k) / stride + 1;
    let mut y = vec![0.0f32; out_ch * out_len];
    conv1d_into(
        &mut y,
        x,
        weight,
        bias,
        in_ch,
        out_ch,
        in_len,
        out_len,
        k,
        stride,
        padding,
        dilation,
        groups,
    );
    y
}

pub fn conv1d_into(
    y: &mut [f32],
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
    out_len: usize,
    k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) {
    conv1d_into_from(
        y, x, weight, bias, in_ch, out_ch, in_len, out_len, k, stride, padding, dilation, groups, 0,
    );
}

pub fn conv1d_into_from(
    y: &mut [f32],
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
    out_len: usize,
    k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    out_start: usize,
) {
    debug_assert_eq!(y.len(), out_ch * out_len);
    if k == 1 && stride == 1 && padding == 0 && dilation == 1 && groups == 1 {
        if out_start == 0 {
            conv1d_pointwise_into(y, x, weight, bias, in_ch, out_ch, in_len);
        } else {
            y.par_chunks_mut(in_len)
                .enumerate()
                .for_each(|(oc, row)| {
                    let w_row = oc * in_ch;
                    let b = bias.map(|bias| bias[oc]).unwrap_or(0.0);
                    for t in out_start..in_len {
                        let mut acc = b;
                        for ic in 0..in_ch {
                            acc += weight[w_row + ic] * x[ic * in_len + t];
                        }
                        row[t] = acc;
                    }
                });
        }
        return;
    }
    if groups == in_ch && groups == out_ch && stride == 1 {
        conv1d_depthwise_into_from(
            y, x, weight, bias, in_ch, in_len, out_len, k, padding, dilation, out_start,
        );
        return;
    }
    conv1d_naive_into(
        y,
        x,
        weight,
        bias,
        in_ch,
        out_ch,
        in_len,
        out_len,
        k,
        stride,
        padding,
        dilation,
        groups,
    );
}

fn conv1d_pointwise_into(
    y: &mut [f32],
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
) {
    debug_assert_eq!(x.len(), in_ch * in_len);
    debug_assert_eq!(weight.len(), out_ch * in_ch);
    debug_assert_eq!(y.len(), out_ch * in_len);

    if in_ch * out_ch * in_len >= SGEMM_MIN_OPS
        && conv1d_pointwise_sgemm(y, x, weight, in_ch, out_ch, in_len)
    {
        if let Some(bias) = bias {
            y.par_chunks_mut(in_len)
                .zip(bias.par_iter())
                .for_each(|(row, &b)| {
                    for v in row.iter_mut() {
                        *v += b;
                    }
                });
        }
        return;
    }

    y.par_chunks_mut(in_len)
        .enumerate()
        .for_each(|(oc, row)| {
            let w_row = oc * in_ch;
            let b = bias.map(|bias| bias[oc]).unwrap_or(0.0);
            for t in 0..in_len {
                let mut acc = b;
                let x_col = t;
                for ic in 0..in_ch {
                    acc += weight[w_row + ic] * x[ic * in_len + x_col];
                }
                row[t] = acc;
            }
        });
}

/// Y = W @ X with row-major layouts: W(out_ch×in_ch), X(in_ch×in_len), Y(out_ch×in_len).
fn conv1d_pointwise_sgemm(
    y: &mut [f32],
    x: &[f32],
    weight: &[f32],
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
) -> bool {
    if weight.len() != out_ch * in_ch || x.len() != in_ch * in_len || y.len() != out_ch * in_len {
        return false;
    }
    y.fill(0.0);
    unsafe {
        sgemm(
            out_ch,
            in_ch,
            in_len,
            1.0,
            weight.as_ptr(),
            in_ch as isize,
            1,
            x.as_ptr(),
            in_len as isize,
            1,
            0.0,
            y.as_mut_ptr(),
            in_len as isize,
            1,
        );
    }
    true
}

#[inline]
fn conv_transpose_sample_cf(
    x: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    bias: f32,
    kernel: usize,
    stride: usize,
    ot: usize,
) -> f32 {
    let it_lo = ot.saturating_sub(kernel - 1).div_ceil(stride);
    let it_hi = (ot / stride).min(in_len.saturating_sub(1));
    let mut acc = bias;
    for it in it_lo..=it_hi {
        let ki = ot - it * stride;
        if ki >= kernel {
            continue;
        }
        for ic in 0..in_ch {
            acc += weight[ic * kernel + ki] * x[ic * in_len + it];
        }
    }
    acc
}

#[inline]
fn conv_transpose_sample_fm(
    x: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    bias: f32,
    kernel: usize,
    stride: usize,
    ot: usize,
) -> f32 {
    let it_lo = ot.saturating_sub(kernel - 1).div_ceil(stride);
    let it_hi = (ot / stride).min(in_len.saturating_sub(1));
    let mut acc = bias;
    for it in it_lo..=it_hi {
        let ki = ot - it * stride;
        if ki >= kernel {
            continue;
        }
        let x_base = it * in_ch;
        for ic in 0..in_ch {
            acc += weight[ic * kernel + ki] * x[x_base + ic];
        }
    }
    acc
}

/// ConvTranspose1d decode: channel-major `x[in_ch * in_len]`, weight `[in_ch * kernel]`.
pub fn conv_transpose1d_decode(
    x: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    bias: f32,
    kernel: usize,
    stride: usize,
) -> Vec<f32> {
    let out_len = (in_len - 1) * stride + kernel;
    let mut out = vec![0.0f32; out_len];
    conv_transpose1d_decode_range(x, in_ch, in_len, weight, bias, kernel, stride, 0, &mut out);
    out
}

pub fn conv_transpose1d_decode_range(
    x: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    bias: f32,
    kernel: usize,
    stride: usize,
    out_start: usize,
    out: &mut [f32],
) {
    let out_len = out.len();
    if out_start >= out_len {
        return;
    }
    out[out_start..out_len].par_iter_mut().enumerate().for_each(|(i, slot)| {
        *slot = conv_transpose_sample_cf(
            x, in_ch, in_len, weight, bias, kernel, stride, out_start + i,
        );
    });
}

/// Frame-major latent: `x[in_len * in_ch]` with `x[frame * in_ch + ch]`.
pub fn conv_transpose1d_decode_fm(
    x: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    bias: f32,
    kernel: usize,
    stride: usize,
) -> Vec<f32> {
    let out_len = (in_len - 1) * stride + kernel;
    let mut out = vec![0.0f32; out_len];
    conv_transpose1d_decode_range_fm(x, in_ch, in_len, weight, bias, kernel, stride, 0, &mut out);
    out
}

pub fn conv_transpose1d_decode_range_fm(
    x: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    bias: f32,
    kernel: usize,
    stride: usize,
    out_start: usize,
    out: &mut [f32],
) {
    let out_len = out.len();
    if out_start >= out_len {
        return;
    }
    out[out_start..out_len].par_iter_mut().enumerate().for_each(|(i, slot)| {
        *slot = conv_transpose_sample_fm(
            x, in_ch, in_len, weight, bias, kernel, stride, out_start + i,
        );
    });
}

fn conv1d_depthwise_into(
    y: &mut [f32],
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    channels: usize,
    in_len: usize,
    out_len: usize,
    k: usize,
    padding: usize,
    dilation: usize,
) {
    conv1d_depthwise_into_from(
        y, x, weight, bias, channels, in_len, out_len, k, padding, dilation, 0,
    );
}

fn conv1d_depthwise_into_from(
    y: &mut [f32],
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    _channels: usize,
    in_len: usize,
    out_len: usize,
    k: usize,
    padding: usize,
    dilation: usize,
    out_start: usize,
) {
    y.par_chunks_mut(out_len)
        .enumerate()
        .for_each(|(c, row)| {
            let b = bias.map(|bias| bias[c]).unwrap_or(0.0);
            let w_base = c * k;
            let x_row = c * in_len;
            for ot in out_start..out_len {
                let mut acc = b;
                for ki in 0..k {
                    let t = ot + ki * dilation;
                    let x_t = t as i32 - padding as i32;
                    if x_t >= 0 && (x_t as usize) < in_len {
                        acc += x[x_row + x_t as usize] * weight[w_base + ki];
                    }
                }
                row[ot] = acc;
            }
        });
}

fn conv1d_naive_into(
    y: &mut [f32],
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
    out_len: usize,
    k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) {
    let in_plen = in_len + 2 * padding;
    let in_per_group = in_ch / groups;
    let out_per_group = out_ch / groups;
    let w_stride = in_per_group * k;

    let in_padded = if padding > 0 {
        let mut p = vec![0.0f32; in_ch * in_plen];
        for t in 0..in_len {
            for c in 0..in_ch {
                p[c * in_plen + (t + padding)] = x[c * in_len + t];
            }
        }
        p
    } else {
        x.to_vec()
    };

    y.fill(0.0);
    for g in 0..groups {
        for oc in 0..out_per_group {
            let out_c = g * out_per_group + oc;
            let w_row = out_c * w_stride;
            for ot in 0..out_len {
                let it = ot * stride;
                let mut acc = bias.map(|b| b[out_c]).unwrap_or(0.0);
                for ic in 0..in_per_group {
                    let in_c = g * in_per_group + ic;
                    let x_row = in_c * in_plen;
                    for ki in 0..k {
                        let t = it + ki * dilation;
                        if t < in_plen {
                            acc += in_padded[x_row + t] * weight[w_row + ic * k + ki];
                        }
                    }
                }
                y[out_c * out_len + ot] = acc;
            }
        }
    }
}

pub fn glu(x: &[f32], channels: usize, length: usize) -> Vec<f32> {
    let half = channels / 2;
    let mut out = vec![0.0; half * length];
    glu_into(&mut out, x, channels, length);
    out
}

pub fn glu_into(out: &mut [f32], x: &[f32], channels: usize, length: usize) {
    let half = channels / 2;
    for t in 0..length {
        for c in 0..half {
            let a = x[c * length + t];
            let b = x[(c + half) * length + t];
            out[c * length + t] = a * sigmoid(b);
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn add_residual(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub fn add_residual_into(out: &mut [f32], a: &[f32], b: &[f32]) {
    for (o, (&x, &y)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
        *o = x + y;
    }
}

/// 1x1 conv with a single time step: `out[oc] = bias[oc] + sum_ic w[oc,ic]*x[ic]`.
#[inline]
pub fn pointwise_matvec_into(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
) {
    debug_assert_eq!(x.len(), in_ch);
    debug_assert_eq!(out.len(), out_ch);
    debug_assert_eq!(weight.len(), out_ch * in_ch);
    for oc in 0..out_ch {
        let mut acc = bias.map(|b| b[oc]).unwrap_or(0.0);
        let w_row = oc * in_ch;
        for ic in 0..in_ch {
            acc += weight[w_row + ic] * x[ic];
        }
        out[oc] = acc;
    }
}

pub fn gelu_vec_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = gelu(*v);
    }
}

pub fn cmul(ar: f32, ai: f32, br: f32, bi: f32) -> (f32, f32) {
    (ar * br - ai * bi, ar * bi + ai * br)
}
