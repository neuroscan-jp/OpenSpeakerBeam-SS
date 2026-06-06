use std::path::PathBuf;
use std::time::Instant;

use ndarray::Array2;
use ndarray_npy::NpzReader;
use speakerbeam_onnx::native::{NativeDecoder, SeparatorWeights};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

fn pack_sep_fm(sep_cf_arr: &Array2<f32>, n_frames: usize) -> Vec<f32> {
    let lc = sep_cf_arr.nrows();
    let mut sep_fm = vec![0.0f32; lc * n_frames];
    for f in 0..n_frames {
        for c in 0..lc {
            sep_fm[f * lc + c] = sep_cf_arr[[c, f]];
        }
    }
    sep_fm
}

fn main() {
    let ref_path = repo_root().join("onnx-runtime/models/sep_ref/decoder_ref.npz");
    let weights_path = repo_root().join("onnx-runtime/models/streaming_separator.npz");
    if !ref_path.exists() || !weights_path.exists() {
        eprintln!("skip bench: missing weights or reference");
        return;
    }

    let weights = SeparatorWeights::from_npz(&weights_path).expect("weights");
    let dec_w = weights.decoder_weights().expect("decoder weights");
    let file = std::fs::File::open(&ref_path).expect("ref");
    let mut reader = NpzReader::new(file).expect("npz");
    let sep_cf_arr: Array2<f32> = reader.by_name("sep_cf").expect("sep_cf");
    let n_frames = sep_cf_arr.ncols();
    let lc = sep_cf_arr.nrows();

    let mut long_sep = Vec::new();
    let repeat = (500usize + n_frames - 1) / n_frames;
    for _ in 0..repeat {
        long_sep.extend(pack_sep_fm(&sep_cf_arr, n_frames));
    }
    let n_total = 500usize;
    let n_prev = n_total - 12;

    let mut dec = NativeDecoder::from_weights(dec_w);
    let sep_full = &long_sep[..n_total * lc];
    let t0 = Instant::now();
    for _ in 0..10 {
        dec.reset();
        dec.decode_fm(sep_full, n_total);
    }
    let full_ms = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!("decoder_fm full n_frames={n_total}: {full_ms:.2} ms");

    let sep_prev = &long_sep[..n_prev * lc];
    let tail_iters = 200;
    let mut tail_elapsed = 0.0;
    for _ in 0..tail_iters {
        dec.reset();
        dec.decode_fm(sep_prev, n_prev);
        let t0 = Instant::now();
        dec.decode_fm(sep_full, n_total);
        tail_elapsed += t0.elapsed().as_secs_f64() * 1000.0;
    }
    println!(
        "decoder_fm tail-only n_prev={n_prev} -> {n_total}: {:.2} ms",
        tail_elapsed / tail_iters as f64
    );

    dec.reset();
    let t0 = Instant::now();
    let stream_iters = 12;
    for step in 1..=stream_iters {
        let n = n_prev + step;
        let sep = &long_sep[..n * lc];
        dec.decode_fm(sep, n);
    }
    let stream_ms = t0.elapsed().as_secs_f64() * 1000.0 / stream_iters as f64;
    println!("decoder_fm streaming +1 frame x{stream_iters} (from {n_prev}): {stream_ms:.2} ms/step");
}
