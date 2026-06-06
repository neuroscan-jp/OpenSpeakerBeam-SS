//! OpenSpeakerBeam-SS ONNX inference CLI (Phase 1 offline + Phase 2 streaming/ECAPA).

use std::path::PathBuf;
use std::process::Command;

use clap::{Parser, ValueEnum};
use speakerbeam_core::audio::{fit_fixed_length, load_wav_mono_16k, save_wav_mono_16k};
use speakerbeam_core::chunk_buffer::ChunkAggregator;
use speakerbeam_core::embedding::{
    extract_enrollment_via_python_features, load_embedding_npy,
};
use speakerbeam_core::RuntimeConfig;
use speakerbeam_onnx::{
    EcapaSession, IncrementalStreamingSession, SpeakerBeamSession, StreamingSession, EMBED_DIM,
    FIXED_SAMPLES,
};
use tracing::info;

#[derive(Debug, Clone, ValueEnum)]
enum EmbeddingBackend {
    /// Precomputed `.npy` or Python `extract_embedding.py` (Phase 1).
    Python,
    /// Python FBank features + ECAPA embedding ONNX (Phase 2).
    Onnx,
}

#[derive(Debug, Parser)]
#[command(name = "speakerbeam-cli", about = "OpenSpeakerBeam-SS ONNX Runtime inference")]
struct Args {
    #[arg(long)]
    mixture: PathBuf,
    /// Precomputed `.npy` embedding (192-d). Skips ECAPA at runtime.
    #[arg(long)]
    embedding_npy: Option<PathBuf>,
    /// Enrollment wav (required unless `--embedding-npy` is set).
    #[arg(long)]
    enrollment: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    /// Offline batch SpeakerBeam ONNX (fixed 10 s).
    #[arg(long, default_value = "models/speakerbeam_ep110.onnx")]
    model: PathBuf,
    /// ECAPA embedding ONNX (Phase 2).
    #[arg(long, default_value = "models/ecapa_embedding.onnx")]
    ecapa_model: PathBuf,
    /// Streaming: encoder frame ONNX.
    #[arg(long, default_value = "models/encoder_frame.onnx")]
    encoder_model: PathBuf,
    /// Streaming: decoder ONNX.
    #[arg(long, default_value = "models/decoder.onnx")]
    decoder_model: PathBuf,
    /// Streaming: cgLN separator ONNX (fixed latent pad, legacy slow path).
    #[arg(long, default_value = "models/separator_cgln.onnx")]
    separator_model: PathBuf,
    /// Incremental native separator weights (.npz).
    #[arg(long, default_value = "models/streaming_separator.npz")]
    separator_weights: PathBuf,
    /// Legacy full-history separator ONNX per chunk (slow).
    #[arg(long)]
    stream_full_separator: bool,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long, value_enum, default_value_t = EmbeddingBackend::Python)]
    embedding_backend: EmbeddingBackend,
    /// Streaming inference (任意長). Uses split ONNX + cgLN separator.
    #[arg(long)]
    stream: bool,
    /// Direct push window when `--input-chunk-ms` is unset (default 100 ms).
    #[arg(long, default_value_t = 100.0)]
    stream_hop_ms: f32,
    /// Input chunk duration, e.g. Opus frame period (60 ms → 960 samples @ 16 kHz).
    #[arg(long)]
    input_chunk_ms: Option<f32>,
    /// Process every N input chunks (e.g. 2–3 for Opus 60 ms → 120–180 ms window).
    #[arg(long, default_value_t = 2)]
    process_every_chunks: usize,
    #[arg(long, default_value = "../../.venv/Scripts/python.exe")]
    python: PathBuf,
    #[arg(long, default_value = "export")]
    export_dir: PathBuf,
}

fn extract_embedding_python(
    python: &PathBuf,
    enrollment: &PathBuf,
    out_npy: &PathBuf,
    export_dir: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let script = export_dir.join("extract_embedding.py");
    let status = Command::new(python)
        .arg(&script)
        .arg("--enrollment")
        .arg(enrollment)
        .arg("--output")
        .arg(out_npy)
        .status()?;
    if !status.success() {
        return Err(format!("extract_embedding.py failed: {status}").into());
    }
    Ok(())
}

fn resolve_path(base: &PathBuf, p: &PathBuf) -> PathBuf {
    if p.is_absolute() {
        p.clone()
    } else {
        base.join(p)
    }
}

trait StreamInfer {
    fn push(&mut self, chunk: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>>;
    fn flush(&mut self) -> Result<Vec<f32>, Box<dyn std::error::Error>>;
}

impl StreamInfer for StreamingSession {
    fn push(&mut self, chunk: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(StreamingSession::push(self, chunk)?)
    }

    fn flush(&mut self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(StreamingSession::flush(self)?)
    }
}

impl StreamInfer for IncrementalStreamingSession {
    fn push(&mut self, chunk: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(IncrementalStreamingSession::push(self, chunk)?)
    }

    fn flush(&mut self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(IncrementalStreamingSession::flush(self)?)
    }
}

fn push_stream_chunks(
    enhanced: &mut Vec<f32>,
    mixture: &[f32],
    cfg: &RuntimeConfig,
    args: &Args,
    session: &mut impl StreamInfer,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(input_ms) = args.input_chunk_ms {
        let mut agg = ChunkAggregator::from_ms(cfg.sample_rate, input_ms, args.process_every_chunks);
        info!(
            "Opus-style input: {} ms/chunk, process every {} chunks (~{:.0} ms)",
            input_ms,
            args.process_every_chunks,
            agg.process_window_ms(cfg.sample_rate)
        );
        let in_samples = agg.input_chunk_samples();
        for chunk in mixture.chunks(in_samples) {
            if let Some(window) = agg.push_chunk(chunk).map_err(|e| -> Box<dyn std::error::Error> { e.into() })? {
                enhanced.extend(session.push(&window)?);
            }
        }
        let tail = agg.flush();
        if !tail.is_empty() {
            enhanced.extend(session.push(&tail)?);
        }
    } else {
        let hop = (cfg.sample_rate as f32 * args.stream_hop_ms / 1000.0) as usize;
        for chunk in mixture.chunks(hop) {
            enhanced.extend(session.push(chunk)?);
        }
    }
    enhanced.extend(session.flush()?);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let cfg = RuntimeConfig::default();
    let cwd = std::env::current_dir()?;

    info!("loading mixture: {}", args.mixture.display());
    let mixture = load_wav_mono_16k(&args.mixture, &cfg)?;

    let embedding = if let Some(ref npy_path) = args.embedding_npy {
        info!("loading embedding: {}", npy_path.display());
        load_embedding_npy(npy_path, cfg.embed_dim)?
    } else if let Some(ref enrollment) = args.enrollment {
        match args.embedding_backend {
            EmbeddingBackend::Python => {
                let tmp = std::env::temp_dir().join("speakerbeam_embedding.npy");
                info!("extracting embedding via Python: {}", enrollment.display());
                let export_dir = resolve_path(&cwd, &args.export_dir);
                extract_embedding_python(&args.python, enrollment, &tmp, &export_dir)?;
                load_embedding_npy(&tmp, cfg.embed_dim)?
            }
            EmbeddingBackend::Onnx => {
                info!(
                    "extracting embedding via ECAPA ONNX (features: Python, embed: ONNX): {}",
                    enrollment.display()
                );
                let enrollment_samples = load_wav_mono_16k(enrollment, &cfg)?;
                let export_dir = resolve_path(&cwd, &args.export_dir);
                let ecapa_path = resolve_path(&cwd, &args.ecapa_model);
                let mut ecapa = EcapaSession::from_file(&ecapa_path, args.threads, &cfg)?;
                extract_enrollment_via_python_features(
                    &enrollment_samples,
                    &cfg,
                    &args.python,
                    &export_dir,
                    |features, n_frames| {
                        ecapa
                            .embed_features(features, n_frames, speakerbeam_core::embedding::N_MELS)
                            .map_err(|e| speakerbeam_core::embedding::EmbeddingError::FeatureExtract(e.to_string()))
                    },
                )?
            }
        }
    } else {
        return Err("either --embedding-npy or --enrollment is required".into());
    };

    if embedding.len() != EMBED_DIM {
        return Err(format!("embedding dim {} != {EMBED_DIM}", embedding.len()).into());
    }

    if args.stream {
        let enc_path = resolve_path(&cwd, &args.encoder_model);
        let dec_path = resolve_path(&cwd, &args.decoder_model);
        let mut enhanced = Vec::new();

        if args.stream_full_separator {
            let sep_path = resolve_path(&cwd, &args.separator_model);
            info!("streaming inference (legacy cgLN separator ONNX, L=2048)");
            let mut session = StreamingSession::from_files(
                &enc_path,
                &dec_path,
                &sep_path,
                args.threads,
                speakerbeam_onnx::DEFAULT_LOOKAHEAD_FRAMES,
            )?;
            session.set_embedding(&embedding)?;
            push_stream_chunks(&mut enhanced, &mixture, &cfg, &args, &mut session)?;
        } else {
            let weights_path = resolve_path(&cwd, &args.separator_weights);
            info!("incremental streaming (native S4D separator + encoder/decoder ONNX)");
            let mut session = IncrementalStreamingSession::from_files(
                &enc_path,
                &dec_path,
                &weights_path,
                args.threads,
                speakerbeam_onnx::DEFAULT_LOOKAHEAD_FRAMES,
            )?;
            session.set_embedding(&embedding)?;
            push_stream_chunks(&mut enhanced, &mixture, &cfg, &args, &mut session)?;
        }
        save_wav_mono_16k(&args.output, &enhanced, cfg.sample_rate)?;
    } else {
        let model_path = resolve_path(&cwd, &args.model);
        info!("loading ONNX model: {}", model_path.display());
        let mut session = SpeakerBeamSession::from_file(&model_path, args.threads)?;
        let mixture_fixed = fit_fixed_length(&mixture, FIXED_SAMPLES);
        info!("running offline inference (fixed {} samples)", FIXED_SAMPLES);
        let enhanced = session.run(&mixture_fixed, &embedding)?;
        save_wav_mono_16k(&args.output, &enhanced, cfg.sample_rate)?;
    }

    info!("saved: {}", args.output.display());
    Ok(())
}
