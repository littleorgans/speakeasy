//! Persistent Hex Parakeet Unified worker for the browser voice host. Loads
//! the model once on Metal, then transcribes each take through Hex's own
//! stream pipeline: 2,560 sample feeds through PauseCap, then 6,400 trailing
//! zeros. Ported from experiments/stt-ux-hex/runner.rs (stt-ux-research).
//! Live words while the owner speaks come from the same pipeline: each feed
//! reports the stream's committed and tentative text, as
//! experiments/stt-ux-timing/runner.rs captures it after every feed.
//!
//! stdin: per request, one op byte, a u32 LE sample count, then that many f32
//! LE samples at 16 kHz mono. `T` transcribes a whole take; `S` starts a live
//! stream and `F` feeds it, each answered with the stream's partial text. `S`
//! carries the stream's chunk and lookahead, two i32 LE ms, before its samples:
//! a pair from transcribe.cpp's Unified menu. Any other request ends the live
//! stream unfinalized. End of input exits. `B` is a `T` take with Cohere's names
//! bias, one f32 LE, before its samples.
//! stdout: one JSON line at ready, then one JSON line per request. No network
//! client, no files written.
//!
//! With `--whole=vocabulary` or `--whole=cohere` after the model, each `T`
//! take runs whole instead (whole.rs) and live streams are refused.
use std::{error::Error, io::{self, Read, Write}, time::Instant};
use serde_json::json;
use transcribe_cpp::{Backend, Model, ModelOptions, ParakeetBufferedStreamOptions, RunOptions, SessionOptions,
    StreamExtension, StreamOptions, TimestampKind};

mod transcription {
    pub mod pause_segmenter;
}
mod hex_pause;
use hex_pause::{PauseCap, StreamFeed};
mod whole;
use whole::Whole;

const TRAILING_SAMPLES: usize = 6_400;
const FEED_SAMPLES: usize = 2_560;
const MAX_SAMPLES: usize = 1_920_000;

#[derive(Default)]
pub struct Partial { stable: String, tentative: String }

/// A live stream's (chunk, right) in ms; Hex's own stream keeps the model default.
type Lookahead = (i32, i32);
enum Request { Take { samples: Vec<f32>, names_bias: f32 }, Feed { start: Option<Lookahead>, samples: Vec<f32> } }

fn read_request(input: &mut impl Read) -> Result<Option<Request>, Box<dyn Error>> {
    let mut head = [0u8; 5];
    if let Err(error) = input.read_exact(&mut head) {
        return if error.kind() == io::ErrorKind::UnexpectedEof { Ok(None) } else { Err(error.into()) };
    }
    let count = u32::from_le_bytes([head[1], head[2], head[3], head[4]]) as usize;
    if count > MAX_SAMPLES { return Err("request exceeds the sample limit".into()); }
    let mut lookahead = [0u8; 8];
    if head[0] == b'S' { input.read_exact(&mut lookahead)?; }
    let lookahead = (i32::from_le_bytes([lookahead[0], lookahead[1], lookahead[2], lookahead[3]]),
        i32::from_le_bytes([lookahead[4], lookahead[5], lookahead[6], lookahead[7]]));
    let mut bias = [0u8; 4];
    if head[0] == b'B' { input.read_exact(&mut bias)?; }
    let names_bias = f32::from_le_bytes(bias);
    if !names_bias.is_finite() { return Err("nonfinite names bias".into()); }
    let mut bytes = vec![0u8; count * 4];
    input.read_exact(&mut bytes)?;
    let samples: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    if samples.iter().any(|x| !x.is_finite()) { return Err("nonfinite sample".into()); }
    match head[0] {
        b'T' | b'B' if count > 0 => Ok(Some(Request::Take { samples, names_bias })),
        b'S' => Ok(Some(Request::Feed { start: Some(lookahead), samples })),
        b'F' => Ok(Some(Request::Feed { start: None, samples })),
        _ => Err("unknown request".into()),
    }
}

fn emit(out: &mut impl Write, record: serde_json::Value) -> Result<(), Box<dyn Error>> {
    writeln!(out, "{record}")?;
    out.flush()?;
    Ok(())
}

/// One live stream, fed in Hex's own 2,560 sample chunks; a remainder waits
/// for the next feed. Returns the request that ended it.
fn live(model: &Model, options: &RunOptions, session_options: &SessionOptions, (chunk_ms, right_ms): Lookahead,
    mut samples: Vec<f32>, input: &mut impl Read, out: &mut impl Write) -> Result<Option<Request>, Box<dyn Error>> {
    let mut session = model.session_with(session_options)?;
    let buffered = ParakeetBufferedStreamOptions { left_ms: None, chunk_ms: Some(chunk_ms), right_ms: Some(right_ms) };
    let stream_options = StreamOptions { family: Some(StreamExtension::ParakeetBuffered(buffered)), ..Default::default() };
    let mut stream = session.stream(options, &stream_options)?;
    let mut cap = PauseCap::new(&mut stream, TRAILING_SAMPLES);
    let mut pending: Vec<f32> = Vec::new();
    loop {
        let started = Instant::now();
        pending.append(&mut samples);
        let whole = pending.len() / FEED_SAMPLES * FEED_SAMPLES;
        let fed = pending[..whole].chunks(FEED_SAMPLES).try_for_each(|chunk| cap.feed(chunk));
        pending.drain(..whole);
        match fed {
            Ok(()) => {
                let Partial { stable, tentative } = cap.partial();
                emit(out, json!({"kind":"partial", "stable":stable, "tentative":tentative,
                    "processingMs":started.elapsed().as_secs_f64() * 1000.0}))?;
            }
            Err(error) => emit(out, json!({"kind":"error", "error":error}))?,
        }
        match read_request(input)? {
            Some(Request::Feed { start: None, samples: next }) => samples = next,
            other => return Ok(other),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    transcribe_cpp::disable_logging();
    let args: Vec<String> = std::env::args().collect();
    let whole = match args.len() {
        2 => None,
        3 => Some(Whole::parse(&args[2]).ok_or("unknown mode")?),
        _ => return Err("usage: hex-unified-worker MODEL [--whole=vocabulary|--whole=cohere]".into()),
    };
    let started = Instant::now();
    let model = Model::load_with(&args[1], &ModelOptions { backend: Backend::Metal, ..Default::default() })?;
    let device = model.device()?;
    let (arch, variant) = whole.map_or(("parakeet", "unified-en-0.6b"), Whole::identity);
    if device.kind != "metal" || model.arch() != arch || model.variant() != variant {
        return Err("runtime identity mismatch".into());
    }
    let caps = model.capabilities()?;
    if whole.is_none() && !caps.supports_streaming { return Err("stream unsupported".into()); }
    let limit = match whole { Some(mode) => mode.limit(&model)?, None => 0 };
    let options = RunOptions { language: Some("en".into()), timestamps: TimestampKind::Segment, ..Default::default() };
    let session_options = SessionOptions { n_threads: 1, ..Default::default() };
    let stdout = io::stdout();
    let mut out = stdout.lock();
    emit(&mut out, json!({"kind":"ready", "backend":device.kind, "device":device.name, "arch":model.arch(),
        "variant":model.variant(), "supportsStreaming":caps.supports_streaming,
        "loadMs":started.elapsed().as_secs_f64() * 1000.0}))?;
    let mut input = io::stdin().lock();
    if let Some(mode) = whole {
        while let Some(next) = read_request(&mut input)? {
            let Request::Take { samples, names_bias } = next else {
                return Err("whole take mode has no live stream".into());
            };
            let started = Instant::now();
            let mut reply = match mode.run(&model, limit, &samples, names_bias) {
                Ok(result) => result,
                Err(error) => { emit(&mut out, json!({"kind":"error", "error":error}))?; continue; }
            };
            reply["kind"] = json!("result");
            reply["processingMs"] = json!(started.elapsed().as_secs_f64() * 1000.0);
            emit(&mut out, reply)?;
        }
        return Ok(());
    }
    let mut request = read_request(&mut input)?;
    while let Some(next) = request {
        request = match next {
            // A feed with no stream open starts one at Hex's own lookahead.
            Request::Feed { start, samples } => live(&model, &options, &session_options, start.unwrap_or((1040, 1040)),
                samples, &mut input, &mut out)?,
            Request::Take { names_bias, .. } if names_bias != 0.0 => {
                emit(&mut out, json!({"kind":"error", "error":"names bias is Cohere only"}))?;
                read_request(&mut input)?
            }
            Request::Take { samples, .. } => {
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
                emit(&mut out, match outcome {
                    Ok(text) => json!({"kind":"result", "text":text, "rawText":stream.snapshot().raw_text,
                        "processingMs":started.elapsed().as_secs_f64() * 1000.0, "pauseDroppedSamples":dropped}),
                    Err(error) => json!({"kind":"error", "error":error}),
                })?;
                read_request(&mut input)?
            }
        };
    }
    Ok(())
}
