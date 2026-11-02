# Candidate: Pocket TTS 18M (October 2026)

Kyutai Pocket TTS, the English 18.2M parameter checkpoint (15.6M flow LM plus 2.6M Mimi decoder), from the no voice cloning snapshot, on PyTorch CPU. Engine `pocket-18m-bill` (voice bill_boerst, precomputed state). `render.py` follows [CONTRACT.md](../../CONTRACT.md): it defines the engine and hands the CLI and loops to `fasttts.run_engine`.

## Pins

| What | Pin |
| --- | --- |
| Weights | `kyutai/pocket-tts-without-voice-cloning@a7056f505a4a8ca72d8b5fb1440071d6f1a35739`, `languages/english_18m_2026-10/`: `model.safetensors` (111,728,528 bytes, decoder included), `tokenizer.model` (59,339 bytes; byte identical at the config's tokenizer pin `08ffc58775e62aad9409511b9591c9af626270f3`), `embeddings/bill_boerst.safetensors` (4,360,600 bytes); sha256 checked by `render.py` before load |
| Runtime | `pocket-tts` 4.0.1 from `kyutai-labs/pocket-tts@91a9e96433bfe23375eea719b03f0081466f8e20`, `torch==2.15.0` (resolved), Python 3.12 |
| Config | `english_18m_2026-10.local.yaml`: upstream `pocket_tts/config/english_18m_2026-10.yaml` at that commit with local weights and tokenizer paths. Upstream tries the gated cloning weights first and falls back to the no cloning snapshot; the local config selects the snapshot directly, so the run is deterministic and needs no token. |

## Licences

- Code: pocket-tts MIT.
- Weights: CC BY 4.0, attribution required (model card; the card also lists prohibited uses, including voice impersonation without consent).
- Voice: bill_boerst from Kyutai's CC0 Voice Zero collection (per HOME/shortlist.md item 2). Other bundled voices come from Expresso, EARS and VCTK sources with their own terms; this adapter selects only bill_boerst and does no cloning.

## Setup

From the speak-easy repo root.

```bash
MODELS=~/Dev/LLM/DEV/helioy/speak-easy/models/tts
L=languages/english_18m_2026-10
$TMPDIR/venv-hf/bin/hf download kyutai/pocket-tts-without-voice-cloning $L/model.safetensors $L/tokenizer.model \
  $L/embeddings/bill_boerst.safetensors --revision a7056f505a4a8ca72d8b5fb1440071d6f1a35739 \
  --local-dir $MODELS/pocket-tts-english-18m-2026-10-a7056f50
uv venv -p 3.12 $TMPDIR/venv-pocket
uv pip install --python $TMPDIR/venv-pocket/bin/python \
  'git+https://github.com/kyutai-labs/pocket-tts.git@91a9e96433bfe23375eea719b03f0081466f8e20'
```

(`$TMPDIR/venv-hf` as in the kokoro-mlx README.) Source read at a shallow pinned clone, `~/Dev/research-clones/pocket-tts-91a9e964`.

## Run

```bash
PY=$TMPDIR/venv-pocket/bin/python
$PY -I experiments/fast-tts/candidates/pocket-18m/render.py --root $MODELS --engine pocket-18m-bill \
  --temperature warm --repeats 1 --json $TMPDIR/pocket-18m-bill--warm--stream.json --clips $ASSETS/clips
```

Flags match `render-baselines.ts`. `run-timed.sh` runs `pocket-18m-bill` by default, with `POCKET_PY` (default `$TMPDIR/venv-pocket/bin/python`).

## Stream policy

- Native incremental audio: the whole text goes to one `generate_audio_stream` call. Upstream splits it at sentences into chunks of at most 50 tokens, each generated afresh from the voice state. Latent frames (12.5 Hz, 80 ms, 1920 samples) are generated on one thread and decoded by Mimi on another; each yield is one decode of every frame queued at that moment, so the first chunk is one 80 ms frame and later chunks grow when the decoder lags.
- No host segmentation and no silence trim: chunks are the model's output as yielded.
- Voice state: loaded once from the bill_boerst safetensors at startup (inside `loadMs`); every call deep copies it (`copy_state=True`, upstream default).
- Threads: `pocket_tts` sets `torch.set_num_threads(1)` at import; kept (recorded in `engine.threads`). An untimed check at 4 threads showed no gain.
- Cancellation checkpoint: upstream's `stop` event, checked before every latent frame and every text chunk; frames already queued still decode. These runs never set it.
- Load (`process.loadMs`): torch and pocket_tts import, `load_model` on the local config, voice state; split in `loadDetail`.
- Full mode: `generate_audio`, which collects the same stream; one chunk at the end.
