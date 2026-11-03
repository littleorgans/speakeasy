# Road test server
<!-- lilo-page kind=operation schema=1 -->
<!-- lilo-covers package.json packages/browser-voice/src/server.ts packages/browser-voice/src/server-options.ts packages/browser-voice/src/host.ts -->
<!-- lilo-verified digest=4e57a13c4ca7fb15 commit=e4aa4d930c1f date=2026-11-03 -->

Up: [speak-easy system map](../index.md)

`pnpm road-test` serves the road test page on `http://127.0.0.1:4870/`, bound to loopback only (`code:package.json`, `code:packages/browser-voice/src/host.ts`). It serves the checkout it runs in, so start it from that checkout's root.

## Runtime file

Worker voices and speech in models read `models/road-test-runtime.json`; `--setup-runtime <file>` or `SPEAKEASY_SETUP_RUNTIME` names another (`code:packages/browser-voice/src/server-options.ts#DEFAULT_SETUP_RUNTIME`). Without the file the server still starts and those models show "needs setup"; `packages/browser-voice/road-test-runtime.example.json` shows its shape.

## Start

Start pnpm in a new session so the server outlives the launching shell and its process group. A `nohup pnpm road-test &` from an agent's shell can die with that shell. From the repo root, with `<log>` an absolute log path:

```sh
python3 -I -c 'import subprocess, sys; log = open(sys.argv[1], "wb"); p = subprocess.Popen(["pnpm", "road-test"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True); print(p.pid)' <log>
```

It prints the pnpm pid; the listener is a child `node` process.

## Check

`GET /` and `GET /speech-output/models` answer 200. The log's first line names the URL; a missing runtime file is logged after it.

## Stop

Find the listener with `lsof -nP -t -iTCP:4870 -sTCP:LISTEN` and send it `SIGINT`; the server closes its host on `SIGINT` or `SIGTERM`, and pnpm exits with it (`code:packages/browser-voice/src/server.ts`). Run the same `lsof` again to confirm the port is free.
