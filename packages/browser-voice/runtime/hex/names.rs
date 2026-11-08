//! The one names list a whole take mode applies, from `--names-json` at
//! startup: validated once, then for Cohere encoded once with the loaded
//! GGUF's own SentencePiece tokenizer (cohere-sequence-bias.patch).
use serde_json::Value;
use sha2::{Digest, Sha256};
use transcribe_cpp::Model;

const MAX_NAMES: usize = 64;
const MAX_NAME_BYTES: usize = 128;

pub struct Names {
    pub list: Vec<String>,
    /// sha256 hex of the canonical JSON array of `list`.
    pub digest: String,
    /// Cohere's piece ids, once `encode_cohere` has run.
    pub sequences: Vec<Vec<i32>>,
}

impl Names {
    /// A JSON array of strings: each trimmed, blanks dropped, later duplicates
    /// dropped. Absent (`None`) is the empty list.
    pub fn parse(json: Option<&str>) -> Result<Self, String> {
        let items = match json.map(serde_json::from_str::<Value>) {
            None => Vec::new(),
            Some(Ok(Value::Array(items))) => items,
            Some(_) => return Err("--names-json must be a JSON array of strings".into()),
        };
        let mut list: Vec<String> = Vec::new();
        for item in items {
            let Value::String(name) = item else { return Err("--names-json must be a JSON array of strings".into()) };
            let name = name.trim();
            if name.contains('\0') { return Err(format!("name {name:?} contains NUL")); }
            if name.len() > MAX_NAME_BYTES { return Err(format!("name {name:?} exceeds {MAX_NAME_BYTES} UTF8 bytes")); }
            if !name.is_empty() && !list.iter().any(|kept| kept == name) { list.push(name.to_string()); }
        }
        if list.len() > MAX_NAMES { return Err(format!("{} names exceed the limit of {MAX_NAMES}", list.len())); }
        let canonical = serde_json::to_string(&list).map_err(|error| error.to_string())?;
        let digest = Sha256::digest(canonical.as_bytes()).iter().map(|byte| format!("{byte:02x}")).collect();
        Ok(Self { list, digest, sequences: Vec::new() })
    }

    /// Every name bare, then every name after a space: the two forms Cohere
    /// writes a name in, as the loaded GGUF's piece ids.
    pub fn encode_cohere(&mut self, model: &Model) -> Result<(), String> {
        let bare = self.list.iter().map(String::clone);
        self.sequences = bare.chain(self.list.iter().map(|name| format!(" {name}"))).map(|form| {
            model.tokenize(&form).ok().filter(|ids| !ids.is_empty())
                .ok_or_else(|| format!("name {:?} does not encode with the model tokenizer", form.trim_start()))
        }).collect::<Result<_, _>>()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Names;
    use transcribe_cpp::{Backend, Model, ModelOptions};

    #[test]
    fn validation_trims_drops_blanks_and_keeps_first_duplicate() {
        let names = Names::parse(Some(r#"[" Grok ", "", "  ", "TM", "Grok", "GPT-SOL"]"#)).unwrap();
        assert_eq!(names.list, ["Grok", "TM", "GPT-SOL"]);
        assert_eq!(names.digest, Names::parse(Some(r#"["Grok","TM","GPT-SOL"]"#)).unwrap().digest);
        assert_eq!(Names::parse(None).unwrap().list.len(), 0);
        // sha256 of `[]`
        assert_eq!(Names::parse(None).unwrap().digest, "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945");
    }

    #[test]
    fn digest_is_sha256_of_the_compact_array_hosts_check() {
        // sha256 of `["Grok","Claude","Codex","GPT","GPT-SOL","TM","Hibiki","Qwen"]`, as JSON.stringify writes it.
        let eight = Names::parse(Some(EIGHT)).unwrap();
        assert_eq!(eight.digest, "59d6626926c652e50c7e90fea5f95afdea75f6c27e11aa07c2996b89d51fc068");
    }

    #[test]
    fn validation_refuses_invalid_lists() {
        let long = format!("[\"{}\"]", "a".repeat(129));
        let many = serde_json::to_string(&(0..65).map(|i| i.to_string()).collect::<Vec<_>>()).unwrap();
        for bad in ["", "{}", "\"Grok\"", "[1]", "[\"Grok\", null]", "[\"a\\u0000b\"]", &long, &many] {
            assert!(Names::parse(Some(bad)).is_err(), "{bad}");
        }
        assert!(Names::parse(Some(&format!("[\"{}\"]", "a".repeat(128)))).is_ok());
    }

    /// The one names list, spaced as a host might send it.
    const EIGHT: &str = r#"[ "Grok", "Claude", "Codex", "GPT", "GPT-SOL", "TM", "Hibiki", "Qwen" ]"#;
    /// The six names compiled into whole.rs before the names list, bare then spaced.
    const SIX: &str = r#"["Grok","Claude","Codex","GPT","TM","Hibiki"]"#;
    const SIX_SEQUENCES: [&[i32]; 12] = [&[13842, 564, 13788], &[13814, 750, 1974], &[13814, 2295, 13809],
        &[13842, 13815, 13806], &[13806, 13816], &[13837, 838, 1720, 13770], &[710, 564, 13788], &[627, 750, 1974],
        &[627, 2295, 13809], &[710, 13815, 13806], &[598, 13816], &[694, 838, 1720, 13770]];

    #[test]
    #[ignore = "needs SPEAKEASY_COHERE_GGUF, the pinned Cohere GGUF"]
    fn cohere_names_encode_with_the_gguf_tokenizer() {
        let path = std::env::var("SPEAKEASY_COHERE_GGUF").expect("SPEAKEASY_COHERE_GGUF");
        let model = Model::load_with(&path, &ModelOptions { backend: Backend::Metal, ..Default::default() }).unwrap();
        let encode = |json: &str| {
            let mut names = Names::parse(Some(json)).unwrap();
            names.encode_cohere(&model).map(|()| names.sequences)
        };
        assert_eq!(encode(SIX).unwrap(), SIX_SEQUENCES.map(<[i32]>::to_vec));
        // GPT-SOL and Qwen as map/cohere-scripts/reproduce_names.py's independent encoder gives them.
        let eight = encode(EIGHT).unwrap();
        assert_eq!([&eight[4], &eight[7], &eight[12], &eight[15]], [&vec![13842, 13815, 13806, 13830, 13801, 13834, 13836],
            &vec![13949, 13782, 514], &vec![710, 13815, 13806, 13830, 13801, 13834, 13836], &vec![1205, 13782, 514]]);
        assert!(encode(r#"["Grok\uE000"]"#).is_err());
    }
}
