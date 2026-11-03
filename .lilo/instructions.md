# Repo instructions: speak-easy
Gates, from the repo root after `pnpm install`:
- Scoped: `pnpm --filter <package> test`
- Structural: `pnpm typecheck`
- Full: `pnpm typecheck && pnpm test`
- CI parity: none (no CI workflows, no hooks)

Hazards and conventions:
- Run commands from the repo root: model, results and hotword paths resolve from `process.cwd()`.
- Never pass `onProgress` to sherpa TTS `generateAsync`; it aborts the process. Stream via the sentence pipeline.
- Sherpa TTS stays on the `cpu` provider.
- Conversation states change only through `LEGAL_TRANSITIONS` in convo-engine `state.ts`.
- API keys come only from `process.env` or the gitignored `.env`; live tests need `SPEAKEASY_LIVE_<PROVIDER>=1`, plus the key where the provider has one.
- Size convention, unverified: files under 700 lines, functions under 150 lines.

Docs: `.lilo/docs/`, start at `index.md`.
- `index.md`: purpose, packages, vocabulary, rules, sources.
- `verification-gates.md`: gate commands, live test flags, Python worker tests, hooks and CI.
- `operations/road-test-server.md`: start, check and stop the road test server on 4870.
<!-- lilo-onboarding status=complete-quick date=2026-10-23 schema=1 -->
<!-- lilo-digest sha256:f87f0d4da21f1625 .lilo/docs/verification-gates.md -->
<!-- lilo-digest sha256:a18e1112c6a84481 .lilo/docs/index.md -->
