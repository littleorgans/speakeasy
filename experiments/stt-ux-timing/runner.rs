//! Paced engine replay. Independent producer preserves the original input clock.
use std::{cell::Cell, error::Error, fs::File, io::{self, BufRead, Write},
    rc::Rc, sync::mpsc, thread, time::{Duration, Instant}};
use serde_json::{json, Value};
use transcribe_cpp::{Backend, Model, ModelOptions, ParakeetBufferedStreamOptions,
    RunOptions, SessionOptions, Stream, StreamExtension, StreamOptions, StreamText,
    StreamUpdate, TimestampKind};

mod transcription { pub mod pause_segmenter; }
mod hex_pause;
use hex_pause::{PauseCap, StreamFeed};

const FEED: usize = 2560;
const TRAILING: usize = 6400;
#[derive(Default)]
pub struct Partial { stable: String, tentative: String }

fn ms(origin: Instant) -> f64 { origin.elapsed().as_secs_f64() * 1000.0 }
fn at(samples: usize) -> Duration { Duration::from_secs_f64(samples as f64 / 16000.0) }

fn wav(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 16000 || spec.bits_per_sample != 32
        || spec.sample_format != hound::SampleFormat::Float {
        return Err("expected canonical Float32 mono 16 kHz".into());
    }
    let samples = reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?;
    if samples.iter().any(|x| !x.is_finite()) { return Err("nonfinite PCM".into()); }
    Ok(samples)
}

fn options(condition: &str) -> Result<StreamOptions, Box<dyn Error>> {
    match condition {
        "default" => Ok(StreamOptions::default()),
        "low160" => Ok(StreamOptions { family: Some(StreamExtension::ParakeetBuffered(
            ParakeetBufferedStreamOptions { chunk_ms: Some(160), right_ms: Some(160),
                ..Default::default() })), ..Default::default() }),
        _ => Err("invalid condition".into()),
    }
}

struct Observer<'a> {
    stream: Stream<'a>, origin: Instant, input: Rc<Cell<usize>>,
    phase: Rc<Cell<&'static str>>, snapshots: Vec<Value>, previous: StreamText,
    model_samples: usize,
}

impl Observer<'_> {
    fn capture(&mut self, update: StreamUpdate, call_ms: f64, force: bool) {
        let wall_ms = ms(self.origin);
        let text = self.stream.text();
        if force || update.result_changed || text != self.previous {
            self.snapshots.push(json!({"phase":self.phase.get(), "wallMs":wall_ms,
                "inputSamples":self.input.get(), "modelSamples":self.model_samples,
                "full":text.full, "committed":text.committed, "tentative":text.tentative,
                "revision":update.revision, "resultChanged":update.result_changed,
                "isFinal":update.is_final, "inputReceivedMs":update.input_received_ms,
                "audioCommittedMs":update.audio_committed_ms, "bufferedMs":update.buffered_ms,
                "committedChanged":update.committed_changed,
                "tentativeChanged":update.tentative_changed, "callMs":call_ms}));
        }
        self.previous = text;
    }
}

impl StreamFeed for Observer<'_> {
    fn feed(&mut self, samples: &[f32]) -> Result<(), String> {
        if self.phase.get() == "end_flush" {
            self.capture(StreamUpdate { revision:self.stream.revision(),
                ..Default::default() }, 0.0, true);
            self.phase.set("suffix_feed");
        }
        let start = Instant::now();
        self.model_samples += samples.len();
        let result = self.stream.feed(samples).map_err(|e| e.to_string());
        if let Ok(update) = result.as_ref() { self.capture(*update, ms(start), false); }
        result.map(drop)
    }
    fn finalize(&mut self) -> Result<String, String> {
        self.phase.set("pre_finalize");
        self.capture(StreamUpdate { revision:self.stream.revision(),
            ..Default::default() }, 0.0, true);
        self.phase.set("finalize");
        let start = Instant::now();
        let update = self.stream.finalize().map_err(|e| e.to_string())?;
        self.capture(update, ms(start), true);
        Ok(self.stream.text().full.trim().to_string())
    }
    fn partial(&self) -> Partial {
        let text = self.stream.text();
        Partial { stable: text.committed.trim_start().to_string(), tentative: text.tentative }
    }
}

fn energy_onset(samples: &[f32]) -> Option<f64> {
    samples.chunks(480).position(|block| {
        let mean = block.iter().map(|&x| x as f64).sum::<f64>() / block.len() as f64;
        let power = block.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>()
            / block.len() as f64;
        power > 1e-5 // -50 dBFS; proxy only, no VAD or acoustic word annotation.
    }).map(|i| i as f64 * 30.0)
}

fn emit(row: &Value) -> Result<(), Box<dyn Error>> {
    let mut out = io::stdout().lock();
    writeln!(out, "{row}")?;
    out.flush()?;
    Ok(())
}

fn replay(model: &Model, row: &Value, run: &RunOptions, opts: &StreamOptions)
    -> Result<(), Box<dyn Error>> {
    let samples = wav(row["audioPath"].as_str().ok_or("missing audioPath")?)?;
    let n = samples.len();
    emit(&json!({"kind":"sample_start", "takeId":row["takeId"]}))?;
    let setup = Instant::now();
    let mut session = model.session_with(&SessionOptions { n_threads:1, ..Default::default() })?;
    let stream = session.stream(run, opts)?;
    let setup_ms = ms(setup);
    let origin = Instant::now();
    let input = Rc::new(Cell::new(0));
    let phase = Rc::new(Cell::new("initial"));
    let mut obs = Observer { stream, origin, input:input.clone(), phase:phase.clone(),
        snapshots:Vec::new(), previous:StreamText::default(), model_samples:0 };
    obs.capture(StreamUpdate::default(), 0.0, true);
    let mut feeds = Vec::new();
    let mut final_result: Result<String, String> = Err("no final result".into());
    let mut dropped = 0;
    let mut flush_start = 0.0;
    let mut final_wall = 0.0;
    thread::scope(|scope| {
        let (tx, rx) = mpsc::channel();
        let producer_samples = &samples;
        scope.spawn(move || {
            for (i, chunk) in producer_samples.chunks(FEED).enumerate() {
                let end = ((i+1)*FEED).min(n);
                let deadline = origin + at(end);
                if let Some(wait) = deadline.checked_duration_since(Instant::now()) {
                    thread::sleep(wait);
                }
                if tx.send((end, ms(origin), chunk)).is_err() { break; }
            }
        });
        let mut cap = PauseCap::new(&mut obs, TRAILING);
        phase.set("feed");
        let mut failed = false;
        for (end, release_ms, chunk) in rx {
            let started = ms(origin);
            input.set(end);
            let result = cap.feed(chunk);
            let ended = ms(origin);
            feeds.push(json!([end, release_ms, started, ended]));
            if let Err(e) = result { final_result = Err(e); failed = true; break; }
        }
        phase.set("end_flush");
        flush_start = ms(origin);
        if !failed { final_result = cap.finalize(); }
        final_wall = ms(origin);
        dropped = cap.dropped;
    });
    let raw = obs.stream.snapshot().raw_text;
    let text = obs.stream.text();
    let last_release = feeds.last().map(|f| f[1].as_f64().unwrap()).unwrap_or(0.0);
    let error = final_result.as_ref().err().cloned();
    let hex_final = final_result.ok();
    emit(&json!({"kind":"sample", "takeId":row["takeId"], "inputSamples":n,
        "sessionSetupMs":setup_ms, "scheduledEndMs":n as f64 / 16.0,
        "actualEndReleaseMs":last_release, "flushStartMs":flush_start,
        "finalWallMs":final_wall, "modelSamples":obs.model_samples,
        "pauseDroppedSamples":dropped, "rawRecognizer":raw,
        "recognizerText":text.full, "hexFinal":hex_final, "error":error,
        "energyOnsetProxyMs":energy_onset(&samples), "postProcessed":null,
        "snapshots":obs.snapshots, "feeds":feeds}))?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    transcribe_cpp::disable_logging();
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 { return Err("usage: runner MODEL MANIFEST default|low160|probe".into()); }
    emit(&json!({"kind":"load_start"}))?;
    let load = Instant::now();
    let model = Model::load_with(&args[1], &ModelOptions { backend:Backend::Metal, ..Default::default() })?;
    let load_ms = ms(load);
    let device = model.device()?;
    let caps = model.capabilities()?;
    if device.kind != "metal" || model.arch() != "parakeet"
        || model.variant() != "unified-en-0.6b" || !caps.supports_streaming {
        return Err("runtime identity mismatch".into());
    }
    let run = RunOptions { language:Some("en".into()), timestamps:TimestampKind::Segment,
        ..Default::default() };
    emit(&json!({"kind":"identity", "loadMs":load_ms, "backend":device.kind,
        "device":device.name, "arch":model.arch(), "variant":model.variant(),
        "maxAudioMs":caps.max_audio_ms, "maxTimestampKind":format!("{:?}",caps.max_timestamp_kind),
        "supportsStreaming":caps.supports_streaming}))?;
    if args[3] == "probe" {
        for condition in ["default", "low160"] {
            let mut session = model.session_with(&SessionOptions { n_threads:1, ..Default::default() })?;
            let stream = session.stream(&run, &options(condition)?)?;
            let text = stream.text();
            emit(&json!({"kind":"geometry_probe", "condition":condition,
                "accepted":true, "initialFull":text.full, "initialCommitted":text.committed,
                "initialTentative":text.tentative}))?;
        }
        return Ok(());
    }
    let opts = options(&args[3])?;
    emit(&json!({"kind":"warmup_start"}))?;
    let warm = Instant::now();
    let mut session = model.session_with(&SessionOptions { n_threads:1, ..Default::default() })?;
    let mut stream = session.stream(&run, &opts)?;
    let mut warm_texts = Vec::new();
    let mut warm_samples = 0;
    for chunk in vec![0.0;160000+TRAILING].chunks(FEED) {
        let update = stream.feed(chunk)?;
        warm_samples += chunk.len();
        if update.result_changed {
            warm_texts.push(json!({"modelSamples":warm_samples,"full":stream.text().full}));
        }
    }
    stream.finalize()?;
    emit(&json!({"kind":"warmup", "wallMs":ms(warm), "modelSamples":warm_samples,
        "texts":warm_texts, "final":stream.text().full}))?;
    drop(stream);
    drop(session);
    for line in io::BufReader::new(File::open(&args[2])?).lines() {
        replay(&model, &serde_json::from_str::<Value>(&line?)?, &run, &opts)?;
    }
    Ok(())
}
