use std::path::PathBuf;

use ndarray::Array2;
use ndarray_npy::NpzReader;
use speakerbeam_onnx::native::{NativeDecoder, SeparatorWeights};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

fn pack_sep_cf(sep_cf_arr: &Array2<f32>, n_frames: usize) -> Vec<f32> {
    let lc = sep_cf_arr.nrows();
    let mut sep_cf = vec![0.0f32; lc * n_frames];
    for f in 0..n_frames {
        for c in 0..lc {
            sep_cf[c * n_frames + f] = sep_cf_arr[[c, f]];
        }
    }
    sep_cf
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

#[test]
fn native_decoder_matches_python_reference() {
    let ref_path = repo_root().join("onnx-runtime/models/sep_ref/decoder_ref.npz");
    if !ref_path.exists() {
        eprintln!("skip: missing {}", ref_path.display());
        return;
    }

    let weights_path = repo_root().join("onnx-runtime/models/streaming_separator.npz");
    let weights = SeparatorWeights::from_npz(&weights_path).expect("load weights");
    let dec_w = weights.decoder_weights().expect("decoder weights");
    let mut dec = NativeDecoder::from_weights(dec_w);

    let file = std::fs::File::open(&ref_path).expect("open decoder ref");
    let mut reader = NpzReader::new(file).expect("npz");
    let waveform: ndarray::Array1<f32> = reader.by_name("waveform").expect("waveform");
    let sep_cf_arr: Array2<f32> = reader.by_name("sep_cf").expect("sep_cf");
    let n_frames = sep_cf_arr.ncols();
    let lc = sep_cf_arr.nrows();
    let sep_cf = pack_sep_cf(&sep_cf_arr, n_frames);
    let full = dec.decode(&sep_cf, n_frames);
    assert_eq!(full.len(), waveform.len());
    let mut maxdiff = 0.0f32;
    for (a, b) in full.iter().zip(waveform.iter()) {
        maxdiff = maxdiff.max((a - b).abs());
    }
    eprintln!("native decoder vs python maxdiff={maxdiff:.6e}");
    assert!(maxdiff < 1e-3, "decoder python parity failed: {maxdiff:.6e}");

    dec.reset();
    let n_prev = n_frames / 2;
    let sep_prev = pack_sep_cf(&sep_cf_arr, n_prev);
    let sep_full = pack_sep_cf(&sep_cf_arr, n_frames);
    let _ = dec.decode(&sep_prev, n_prev);
    let incr = dec.decode(&sep_full, n_frames);
    let mut max_incr = 0.0f32;
    for (a, b) in full.iter().zip(incr.iter()) {
        max_incr = max_incr.max((a - b).abs());
    }
    eprintln!("native decoder incremental maxdiff={max_incr:.6e}");
    assert!(max_incr < 1e-3, "decoder incremental parity failed: {max_incr:.6e}");

    let mut dec_fm = NativeDecoder::from_weights(weights.decoder_weights().expect("decoder weights"));
    let sep_fm = pack_sep_fm(&sep_cf_arr, n_frames);
    dec_fm.decode_fm(&sep_fm, n_frames);
    let full_fm = dec_fm.wav().to_vec();
    let mut fm_diff = 0.0f32;
    for (a, b) in full.iter().zip(full_fm.iter()) {
        fm_diff = fm_diff.max((a - b).abs());
    }
    eprintln!("native decoder frame-major maxdiff={fm_diff:.6e}");
    assert!(fm_diff < 1e-3, "decoder frame-major parity failed: {fm_diff:.6e}");

    dec_fm.reset();
    let sep_fm_prev = pack_sep_fm(&sep_cf_arr, n_prev);
    let sep_fm_full = pack_sep_fm(&sep_cf_arr, n_frames);
    dec_fm.decode_fm(&sep_fm_prev, n_prev);
    dec_fm.decode_fm(&sep_fm_full, n_frames);
    let mut fm_incr = 0.0f32;
    for (a, b) in full_fm.iter().zip(dec_fm.wav().iter()) {
        fm_incr = fm_incr.max((a - b).abs());
    }
    eprintln!("native decoder frame-major incremental maxdiff={fm_incr:.6e}");
    assert!(fm_incr < 1e-3, "decoder frame-major incremental failed: {fm_incr:.6e}");
}
