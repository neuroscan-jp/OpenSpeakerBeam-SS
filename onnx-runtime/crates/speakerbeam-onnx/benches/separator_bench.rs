use std::path::PathBuf;
use std::time::Instant;

use speakerbeam_onnx::native::{NativeSeparatorStream, SeparatorWeights};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

fn main() {
    let weights_path = repo_root().join("onnx-runtime/models/streaming_separator.npz");
    let ref_path = repo_root().join("onnx-runtime/models/sep_ref/step1.npz");
    if !weights_path.exists() || !ref_path.exists() {
        eprintln!("skip bench: missing weights or reference");
        return;
    }

    let weights = SeparatorWeights::from_npz(&weights_path).expect("weights");
    let mut sep = NativeSeparatorStream::from_weights(weights);
    let file = std::fs::File::open(&ref_path).expect("ref");
    let mut reader = ndarray_npy::NpzReader::new(file).expect("npz");
    let latent_fm: ndarray::Array2<f32> = reader.by_name("latent_fm").expect("latent");
    let emb: ndarray::Array1<f32> = reader.by_name("emb").expect("emb");
    let n_frames = latent_fm.nrows();
    let latent: Vec<f32> = latent_fm.iter().copied().collect();

    sep.set_embedding(emb.as_slice().unwrap()).expect("embed");

    let warmup = sep
        .forward(&latent, n_frames, 0, emb.as_slice().unwrap())
        .expect("forward");
    assert!(!warmup.is_empty());

    let frames = [11usize, 50, 200, 500];
    for &n in &frames {
        if n > n_frames {
            continue;
        }
        sep.reset();
        sep.set_embedding(emb.as_slice().unwrap()).unwrap();
        let t0 = Instant::now();
        let iters = if n > 100 { 5 } else { 20 };
        for _ in 0..iters {
            let _ = sep
                .forward(&latent[..n * 4096], n, 0, emb.as_slice().unwrap())
                .unwrap();
            sep.reset();
            sep.set_embedding(emb.as_slice().unwrap()).unwrap();
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("separator forward n_frames={n}: {ms:.2} ms");
    }
}
