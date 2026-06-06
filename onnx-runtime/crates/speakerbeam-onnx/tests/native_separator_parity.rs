use std::path::PathBuf;

use ndarray::{Array1, Array2, IxDyn, OwnedRepr};
use ndarray_npy::NpzReader;
use speakerbeam_onnx::native::{NativeSeparatorStream, SeparatorWeights};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

#[test]
fn native_separator_matches_python_reference() {
    let ref_path = repo_root().join("onnx-runtime/models/sep_ref/step1.npz");
    if !ref_path.exists() {
        eprintln!("skip: missing {}", ref_path.display());
        return;
    }

    let weights_path = repo_root().join("onnx-runtime/models/streaming_separator.npz");
    let weights = SeparatorWeights::from_npz(&weights_path).expect("load weights");
    let mut sep = NativeSeparatorStream::from_weights(weights);

    let file = std::fs::File::open(&ref_path).expect("open ref");
    let mut reader = NpzReader::new(file).expect("npz");
    let latent_fm: Array2<f32> = reader.by_name("latent_fm").expect("latent_fm");
    let out_ref: Array2<f32> = reader.by_name("out_fm").expect("out_fm");
    let emb: Array1<f32> = reader.by_name("emb").expect("emb");

    let n_frames = latent_fm.nrows();
    let lc = latent_fm.ncols();
    let latent_flat: Vec<f32> = latent_fm.iter().copied().collect();

    let out = sep
        .forward(&latent_flat, n_frames, 0, emb.as_slice().unwrap())
        .expect("forward");

    assert_eq!(out.len(), n_frames * lc);
    let mut maxdiff = 0.0f32;
    for f in 0..n_frames {
        for c in 0..lc {
            let rust = out[f * lc + c];
            let py = out_ref[[f, c]];
            maxdiff = maxdiff.max((rust - py).abs());
        }
    }
    eprintln!("native vs python separator maxdiff={maxdiff:.6e}");
    assert!(
        maxdiff < 1e-3,
        "separator parity failed: maxdiff={maxdiff:.6e}"
    );
}
