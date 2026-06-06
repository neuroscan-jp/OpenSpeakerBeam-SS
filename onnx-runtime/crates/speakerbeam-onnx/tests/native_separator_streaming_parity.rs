use std::path::PathBuf;

use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use speakerbeam_onnx::native::{NativeSeparatorStream, SeparatorWeights};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

#[test]
fn streaming_steps_match_full_forward() {
    let ref_path = repo_root().join("onnx-runtime/models/sep_ref/step1.npz");
    if !ref_path.exists() {
        return;
    }
    let weights_path = repo_root().join("onnx-runtime/models/streaming_separator.npz");
    let weights = SeparatorWeights::from_npz(&weights_path).expect("weights");
    let file = std::fs::File::open(&ref_path).expect("ref");
    let mut reader = NpzReader::new(file).expect("npz");
    let latent_fm: Array2<f32> = reader.by_name("latent_fm").expect("latent_fm");
    let emb: Array1<f32> = reader.by_name("emb").expect("emb");
    let lc = latent_fm.ncols();
    let frame = latent_fm.row(0).to_owned();

    let n_total = 500usize;
    let mut latent = Vec::with_capacity(n_total * lc);
    for _ in 0..n_total {
        latent.extend_from_slice(frame.as_slice().unwrap());
    }

    let mut full_sep = NativeSeparatorStream::from_weights(weights.clone());
    full_sep.set_embedding(emb.as_slice().unwrap()).unwrap();
    let full_out = full_sep
        .forward(&latent, n_total, 0, emb.as_slice().unwrap())
        .expect("full");

    let mut incr_sep = NativeSeparatorStream::from_weights(weights);
    incr_sep.set_embedding(emb.as_slice().unwrap()).unwrap();
    let chunk = 12usize;
    let mut incr_out = vec![0.0f32; n_total * lc];
    let mut n_prev = 0usize;
    while n_prev < n_total {
        let n_frames = (n_prev + chunk).min(n_total);
        let out = incr_sep
            .forward(&latent[..n_frames * lc], n_frames, n_prev, emb.as_slice().unwrap())
            .expect("incr");
        for f in n_prev..n_frames {
            incr_out[f * lc..(f + 1) * lc]
                .copy_from_slice(&out[(f - n_prev) * lc..(f - n_prev + 1) * lc]);
        }
        n_prev = n_frames;
    }

    let mut maxdiff = 0.0f32;
    for i in 0..full_out.len() {
        maxdiff = maxdiff.max((full_out[i] - incr_out[i]).abs());
    }
    eprintln!("streaming {n_total} frames maxdiff={maxdiff:.6e}");
    assert!(maxdiff < 1e-3, "streaming parity failed: {maxdiff:.6e}");
}
