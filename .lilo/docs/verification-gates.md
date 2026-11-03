# Verification gates
<!-- lilo-page kind=gates schema=1 -->
<!-- lilo-covers package.json pnpm-workspace.yaml packages/*/package.json packages/*/tsconfig.json packages/browser-voice/runtime/test_*.py packages/browser-voice/runtime/cleanup.py -->
<!-- lilo-verified digest=cfd125d41bb91d8a commit=e4aa4d930c1f date=2026-11-03 -->

Up: [speak-easy system map](index.md)

Run every gate from the repo root after `pnpm install`. Node must satisfy the `engines` floor in `code:package.json`; tests run TypeScript directly through `node --test`, so typecheck is a separate gate.

## Gates

| Gate | Command | Executed |
| --- | --- | --- |
| Scoped | `pnpm --filter <package> test` | exit 0 2026-10-23, as `pnpm --filter @speakeasy/llm test` |
| Structural | `pnpm typecheck` | exit 0 2026-10-23 |
| Full | `pnpm typecheck && pnpm test` | exit 0 2026-10-23 |
| CI parity | none | n/a |

## Conditional

- Scoped takes a workspace package name: `@speakeasy/speech-io`, `@speakeasy/llm`, `@speakeasy/convo-engine` or `@speakeasy/browser-voice`.
- Live provider smoke tests run only with an opt in flag and a key, so the suite never spends quota by accident: `SPEAKEASY_LIVE_CEREBRAS=1` with `CEREBRAS_API_KEY` (`code:packages/llm/src/cerebras.test.ts`), `SPEAKEASY_LIVE_MERCURY=1` with `INCEPTIONLABS_API_KEY` (`code:packages/llm/src/mercury.test.ts`), `SPEAKEASY_LIVE_CARTESIA=1` with `CARTESIA_API_KEY` (`code:packages/speech-io/src/tts/cartesia.test.ts`).
- Live onnxruntime model tests run only with `SPEAKEASY_LIVE_ORT=1` (macOS, downloads into `models/`), no key: `code:packages/speech-io/src/engines/silence-gate-live.test.ts` and the `ort-*-live.test.ts` files.
- Browser voice worker backends under `packages/browser-voice/runtime/` have standard library tests outside the Node gates: `python3 -m unittest discover -s packages/browser-voice/runtime`. They need Python 3.13.2; the cleanup tests fail on any other version (`code:packages/browser-voice/runtime/cleanup.py#runtime_versions`). With `clang++` installed, `test_cohere_bias.py` also compiles and checks the names bias selector (`code:packages/browser-voice/runtime/test_cohere_bias.py`).
- Benchmarks and demos (`pnpm bench`, `pnpm tts`, `pnpm cascade-sweep`) need downloaded models under `models/` or provider keys and are no gate.

## Hooks

none

## CI

none: the repo has no CI workflows.

## Expected failures

none: without the live flags the live smoke tests report as skipped.
