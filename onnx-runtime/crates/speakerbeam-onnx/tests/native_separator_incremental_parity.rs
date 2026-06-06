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
fn incremental_separator_matches_full_forward() {
    let ref_path = repo_root().join("onnx-runtime/models/sep_ref/step1.npz");
    if !ref_path.exists() {
        eprintln!("skip: missing {}", ref_path.display());
        return;
    }

    let weights_path = repo_root().join("onnx-runtime/models/streaming_separator.npz");
    let weights = SeparatorWeights::from_npz(&weights_path).expect("load weights");
    let file = std::fs::File::open(&ref_path).expect("open ref");
    let mut reader = NpzReader::new(file).expect("npz");
    let latent_fm: Array2<f32> = reader.by_name("latent_fm").expect("latent_fm");
    let emb: Array1<f32> = reader.by_name("emb").expect("emb");
    let n_total = latent_fm.nrows();
    let lc = latent_fm.ncols();
    let latent: Vec<f32> = latent_fm.iter().copied().collect();

    let mut full_sep = NativeSeparatorStream::from_weights(weights.clone());
    full_sep.set_embedding(emb.as_slice().unwrap()).unwrap();
    let full_out = full_sep
        .forward(&latent, n_total, 0, emb.as_slice().unwrap())
        .expect("full");

    let mut incr_sep = NativeSeparatorStream::from_weights(weights);
    incr_sep.set_embedding(emb.as_slice().unwrap()).unwrap();
    let mut n_prev = 0usize;
    let mut incr_out = vec![0.0f32; n_total * lc];
    let steps = [3usize, 4, 4];
    for &n_new in &steps {
        let n_frames = n_prev + n_new;
        let out = incr_sep
            .forward(&latent[..n_frames * lc], n_frames, n_prev, emb.as_slice().unwrap())
            .expect("incr");
        for f in 0..n_new {
            let dst = (n_prev + f) * lc;
            incr_out[dst..dst + lc].copy_from_slice(&out[f * lc..(f + 1) * lc]);
        }
        n_prev = n_frames;
    }

    let mut maxdiff = 0.0f32;
    for f in 0..n_total {
        for c in 0..lc {
            let a = full_out[f * lc + c];
            let b = incr_out[f * lc + c];
            maxdiff = maxdiff.max((a - b).abs());
        }
    }
    eprintln!("incremental vs full separator maxdiff={maxdiff:.6e}");
    assert!(
        maxdiff < 1e-2,
        "incremental separator parity failed: maxdiff={maxdiff:.6e}"
    );
}
