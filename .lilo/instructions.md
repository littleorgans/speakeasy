# Repo instructions: speak-easy
Gates, from the repo root after `pnpm install`:
- Scoped: `pnpm --filter <package> test`
- Structural: `pnpm typecheck`
- Full: `pnpm typecheck && pnpm test`
- CI parity: none (no CI workflows, no hooks)

Hazards and conventions:
- Run commands from the repo root: model, results and hotword paths resolve from `process.cwd()`.
- Never pass `onProgress` to sherpa TTS `generateAsync`; it aborts the process. Stream via the sentence pipeline.
- TTS stays on the `cpu` provider.
- Conversation states change only through `LEGAL_TRANSITIONS` in convo-engine `state.ts`.
- API keys come only from `process.env` or the gitignored `.env`; live tests need `SPEAKEASY_LIVE_<PROVIDER>=1` plus the key.
- Size convention, unverified: files under 700 lines, functions under 150 lines.

Docs: `.lilo/docs/`, start at `index.md`.
- `index.md`: purpose, packages, vocabulary, rules, sources.
- `verification-gates.md`: gate commands, live test flags, hooks and CI.
<!-- lilo-onboarding status=complete-quick date=2026-10-23 schema=1 -->
<!-- lilo-digest sha256:8a81b8edf64b029f .lilo/docs/verification-gates.md -->
<!-- lilo-digest sha256:891f6354b995c92f .lilo/docs/index.md -->
