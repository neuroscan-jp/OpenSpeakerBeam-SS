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

    let n_total = 500usize;
    let n_prev = n_total - 12;
    let mut long_latent = Vec::with_capacity(n_total * 4096);
    for f in 0..n_total {
        let src = (f % n_frames) * 4096;
        long_latent.extend_from_slice(&latent[src..src + 4096]);
    }

    sep.reset();
    sep.set_embedding(emb.as_slice().unwrap()).unwrap();
    let t0 = Instant::now();
    for _ in 0..5 {
        sep.reset();
        sep.set_embedding(emb.as_slice().unwrap()).unwrap();
        let _ = sep
            .forward(&long_latent[..n_total * 4096], n_total, 0, emb.as_slice().unwrap())
            .unwrap();
    }
    let full_ms = t0.elapsed().as_secs_f64() * 1000.0 / 5.0;
    println!("separator full n_frames={n_total}: {full_ms:.2} ms");

    sep.reset();
    sep.set_embedding(emb.as_slice().unwrap()).unwrap();
    let _ = sep
        .forward(&long_latent[..n_prev * 4096], n_prev, 0, emb.as_slice().unwrap())
        .unwrap();
    let t0 = Instant::now();
    let iters = 30;
    for _ in 0..iters {
        let _ = sep
            .forward(
                &long_latent[..n_total * 4096],
                n_total,
                n_prev,
                emb.as_slice().unwrap(),
            )
            .unwrap();
    }
    let incr_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("separator incremental n_prev={n_prev} -> {n_total}: {incr_ms:.2} ms");
}
