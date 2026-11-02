//! Persistent Hex Parakeet Unified worker for the browser voice host. Loads
//! the model once on Metal, then transcribes each take through Hex's own
//! stream pipeline: 2,560 sample feeds through PauseCap, then 6,400 trailing
//! zeros. Ported from experiments/stt-ux-hex/runner.rs (stt-ux-research).
//!
//! stdin: per take, a u32 LE sample count then that many f32 LE samples at
//! 16 kHz mono; a zero count or end of input exits. stdout: one JSON line at
//! ready, then one JSON line per take. No network client, no files written.
use std::{error::Error, io::{self, Read, Write}, time::Instant};
use serde_json::json;
use transcribe_cpp::{Backend, Model, ModelOptions, RunOptions, SessionOptions, StreamOptions, TimestampKind};

mod transcription {
    pub mod pause_segmenter;
}
mod hex_pause;
use hex_pause::{PauseCap, StreamFeed};

const TRAILING_SAMPLES: usize = 6_400;
const FEED_SAMPLES: usize = 2_560;
const MAX_SAMPLES: usize = 1_920_000;

#[derive(Default)]
pub struct Partial { stable: String, tentative: String }

fn read_take(input: &mut impl Read) -> Result<Option<Vec<f32>>, Box<dyn Error>> {
    let mut count = [0u8; 4];
    if let Err(error) = input.read_exact(&mut count) {
        return if error.kind() == io::ErrorKind::UnexpectedEof { Ok(None) } else { Err(error.into()) };
    }
    let count = u32::from_le_bytes(count) as usize;
    if count == 0 { return Ok(None); }
    if count > MAX_SAMPLES { return Err("take exceeds the sample limit".into()); }
    let mut bytes = vec![0u8; count * 4];
    input.read_exact(&mut bytes)?;
    let samples: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    if samples.iter().any(|x| !x.is_finite()) { return Err("nonfinite sample".into()); }
    Ok(Some(samples))
}

fn main() -> Result<(), Box<dyn Error>> {
    transcribe_cpp::disable_logging();
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 { return Err("usage: hex-unified-worker MODEL".into()); }
    let started = Instant::now();
    let model = Model::load_with(&args[1], &ModelOptions { backend: Backend::Metal, ..Default::default() })?;
    let device = model.device()?;
    if device.kind != "metal" || model.arch() != "parakeet" || model.variant() != "unified-en-0.6b" {
        return Err("runtime identity mismatch".into());
    }
    let caps = model.capabilities()?;
    if !caps.supports_streaming { return Err("stream unsupported".into()); }
    let options = RunOptions { language: Some("en".into()), timestamps: TimestampKind::Segment, ..Default::default() };
    let session_options = SessionOptions { n_threads: 1, ..Default::default() };
    let stdout = io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "{}", json!({"kind":"ready", "backend":device.kind, "device":device.name, "arch":model.arch(),
        "variant":model.variant(), "supportsStreaming":caps.supports_streaming,
        "loadMs":started.elapsed().as_secs_f64() * 1000.0}))?;
    out.flush()?;
    let mut input = io::stdin().lock();
    while let Some(samples) = read_take(&mut input)? {
        let started = Instant::now();
        // Hex recreates its session for every dictation.
        let mut session = model.session_with(&session_options)?;
        let mut stream = session.stream(&options, &StreamOptions::default())?;
        let mut cap = PauseCap::new(&mut stream, TRAILING_SAMPLES);
        let outcome = (|| -> Result<String, String> {
            for chunk in samples.chunks(FEED_SAMPLES) { cap.feed(chunk)?; }
            cap.finalize()
        })();
        let dropped = cap.dropped;
        drop(cap);
        let record = match outcome {
            Ok(text) => json!({"kind":"result", "text":text, "rawText":stream.snapshot().raw_text,
                "processingMs":started.elapsed().as_secs_f64() * 1000.0, "pauseDroppedSamples":dropped}),
            Err(error) => json!({"kind":"error", "error":error}),
        };
        writeln!(out, "{record}")?;
        out.flush()?;
    }
    Ok(())
}
