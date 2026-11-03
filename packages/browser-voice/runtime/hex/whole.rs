//! Whole take modes of the transcribe.cpp worker, ported from the stt-ux
//! research gguf-controls runner (control.rs `Condition::options`, runner.rs
//! `whole_take`): one `session.run` per window, no stream and no pause cap.
//! `vocabulary` is Hex Unified with the research word list at boost 2.5;
//! `cohere` is Cohere Transcribe in windows of at most 35 s.
use serde_json::{json, Value};
use transcribe_cpp::{Error, ExtSlot, Model, ParakeetVocabularyOptions, RunExtension, RunOptions, SessionOptions,
    TimestampKind};

const MINIMUM_SAMPLES: usize = 24_000;
const TRAILING_SAMPLES: usize = 6_400;
const COHERE_WINDOW: usize = 560_000;
const NAMES: [&str; 6] = ["Claude", "GPT", "Grok", "Codex", "TM", "whoami"];

#[derive(Clone, Copy, PartialEq)]
pub enum Whole { Vocabulary, Cohere }

impl Whole {
    pub fn parse(flag: &str) -> Option<Self> {
        match flag { "--whole=vocabulary" => Some(Self::Vocabulary), "--whole=cohere" => Some(Self::Cohere), _ => None }
    }

    /// The model this mode requires: architecture, then variant.
    pub fn identity(self) -> (&'static str, &'static str) {
        match self { Self::Vocabulary => ("parakeet", "unified-en-0.6b"), Self::Cohere => ("cohere_asr", "cohere-transcribe-03-2026") }
    }

    /// The largest window one run takes, after checking the model accepts this mode.
    pub fn limit(self, model: &Model) -> Result<usize, String> {
        if self == Self::Vocabulary {
            let kind = transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_PARAKEET_VOCABULARY;
            return if model.accepts_ext(ExtSlot::Run, kind) { Ok(usize::MAX) } else { Err("vocabulary unsupported".into()) };
        }
        let max_audio_ms = model.capabilities().map_err(|error| error.to_string())?.max_audio_ms;
        let limit = if max_audio_ms > 0 { COHERE_WINDOW.min(max_audio_ms as usize * 16) } else { COHERE_WINDOW };
        if limit < MINIMUM_SAMPLES { Err("runtime capacity below minimum padding".into()) } else { Ok(limit) }
    }

    fn options(self) -> RunOptions {
        let timestamps = if self == Self::Cohere { TimestampKind::None } else { TimestampKind::Word };
        let mut options = RunOptions { language: Some("en".into()), timestamps, ..Default::default() };
        if self == Self::Vocabulary {
            options.vocabulary = NAMES.iter().map(|name| name.to_string()).collect();
            options.family = Some(RunExtension::ParakeetVocabulary(ParakeetVocabularyOptions {
                boost: Some(2.5), sounds_like: Vec::new(),
            }));
        }
        options
    }

    /// Each window padded to the minimum, plus Parakeet's trailing zeros; window texts
    /// joined. A repetition stop keeps its partial text, as Hex does.
    pub fn run(self, model: &Model, limit: usize, samples: &[f32]) -> Result<Value, String> {
        let options = self.options();
        let mut texts = Vec::new();
        let mut raw = Vec::new();
        let mut recovered = false;
        for (start, end, prepared_len) in windows(samples.len(), limit) {
            let mut prepared = samples[start..end].to_vec();
            prepared.resize(prepared_len + if self == Self::Cohere { 0 } else { TRAILING_SAMPLES }, 0.0);
            let result = model.session_with(&SessionOptions { n_threads: 1, ..Default::default() })
                .and_then(|mut session| session.run(&prepared, &options));
            let transcript = match result {
                Ok(transcript) => transcript,
                Err(Error::OutputRepetition { partial: Some(partial), .. }) => { recovered = true; *partial }
                Err(error) => return Err(error.to_string()),
            };
            raw.push(transcript.raw_text);
            texts.push(transcript.text.trim().to_string());
        }
        let text = texts.into_iter().filter(|text| !text.is_empty()).collect::<Vec<_>>().join(" ");
        Ok(json!({"text":text, "rawText":raw.concat(), "windows":raw.len(), "repetitionRecovered":recovered}))
    }
}

/// (start, end, padded length) per window, as the research runner splits a take.
fn windows(n: usize, limit: usize) -> Vec<(usize, usize, usize)> {
    (0..n.max(MINIMUM_SAMPLES)).step_by(limit).map(|start| {
        let end = (start + limit).min(n.max(MINIMUM_SAMPLES));
        (start, end.min(n), (end - start).max(MINIMUM_SAMPLES))
    }).collect()
}
