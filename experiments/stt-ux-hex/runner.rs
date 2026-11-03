//! Saved WAV accuracy harness. No network clients, microphone, or timing output.
use std::{error::Error, fs::File, io::{self, BufRead, Write}};
use serde_json::{json, Value};
use transcribe_cpp::{Backend, Model, ModelOptions, RunOptions, SessionOptions, StreamOptions, TimestampKind};

mod transcription {
    pub mod pause_segmenter;
}
mod hex_pause;
use hex_pause::{PauseCap, StreamFeed};

// These are source constants, frozen before recognition, not experiment knobs.
const MINIMUM_SAMPLES: usize = 24_000;
const TRAILING_SAMPLES: usize = 6_400;
const FEED_SAMPLES: usize = 2_560;

#[derive(Default)]
pub struct Partial { stable: String, tentative: String }

fn wav(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 16_000
        || spec.bits_per_sample != 32 || spec.sample_format != hound::SampleFormat::Float {
        return Err("expected canonical mono Float32 16 kHz WAV".into());
    }
    let samples = reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?;
    if samples.iter().any(|x| !x.is_finite()) { return Err("nonfinite WAV sample".into()); }
    Ok(samples)
}

fn main() -> Result<(), Box<dyn Error>> {
    transcribe_cpp::disable_logging();
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 { return Err("usage: runner MODEL MANIFEST_JSONL stream|offline".into()); }
    let mode = args[3].as_str();
    if !matches!(mode, "stream" | "offline") { return Err("invalid mode".into()); }
    let model = Model::load_with(&args[1], &ModelOptions { backend: Backend::Metal, ..Default::default() })?;
    let device = model.device()?;
    if device.kind != "metal" || model.arch() != "parakeet" || model.variant() != "unified-en-0.6b" {
        return Err("runtime identity mismatch".into());
    }
    let caps = model.capabilities()?;
    let timestamps = if mode == "stream" { TimestampKind::Segment } else {
        match caps.max_timestamp_kind {
            TimestampKind::Word | TimestampKind::Token => TimestampKind::Word,
            x => x,
        }
    };
    let options = RunOptions { language: Some("en".into()), timestamps, ..Default::default() };
    let session_options = SessionOptions { n_threads: 1, ..Default::default() };
    // No capability dependent fallback or selection by recognition score.
    if mode == "stream" && !caps.supports_streaming { return Err("stream unsupported".into()); }
    let stdout = io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "{}", json!({"kind":"identity", "backend":device.kind, "device":device.name,
        "arch":model.arch(), "variant":model.variant(), "mode":mode,
        "supportsStreaming":caps.supports_streaming, "maxAudioMs":caps.max_audio_ms,
        "maxTimestampKind":format!("{:?}",caps.max_timestamp_kind)}))?;
    out.flush()?;
    for line in io::BufReader::new(File::open(&args[2])?).lines() {
        let row: Value = serde_json::from_str(&line?)?;
        let samples = wav(row["audioPath"].as_str().ok_or("missing audioPath")?)?;
        let n = samples.len();
        let mut session = model.session_with(&session_options)?;
        let record = if mode == "offline" {
            let mut prepared = samples;
            prepared.resize(n.max(MINIMUM_SAMPLES) + TRAILING_SAMPLES, 0.0);
            let result = match session.run(&prepared, &options) {
                Err(transcribe_cpp::Error::OutputRepetition { partial: Some(partial), .. }) => *partial,
                other => other?,
            };
            json!({"takeId":row["takeId"],"mode":mode,"inputSamples":n,
                "preparedSamples":prepared.len(),"rawRecognizer":result.raw_text,
                "recognizerText":result.text,"hexFinal":result.text,
                "postProcessed":null,"pauseDroppedSamples":0})
        } else {
            let mut stream = session.stream(&options, &StreamOptions::default())?;
            let mut cap = PauseCap::new(&mut stream, TRAILING_SAMPLES);
            for chunk in samples.chunks(FEED_SAMPLES) { cap.feed(chunk)?; }
            let hex_final = cap.finalize()?;
            let dropped = cap.dropped;
            drop(cap);
            let snapshot = stream.snapshot();
            json!({"takeId":row["takeId"],"mode":mode,"inputSamples":n,
                "rawRecognizer":snapshot.raw_text,"recognizerText":stream.text().full,
                "hexFinal":hex_final,"postProcessed":null,"pauseDroppedSamples":dropped})
        };
        writeln!(out, "{record}")?;
        out.flush()?;
        // Hex recreates its session after every run/stream.
    }
    Ok(())
}
