#!/usr/bin/env bash
# Compiles the web workers into web/, next to the page, so that both
# `flutter run -d chrome` and `flutter build web` serve them:
# - fit_host: runs the fit jobs (drivers, pool, WebGPU) off the page's thread;
# - fit_worker: the pool's workers, which fit_host starts (nested workers).
# Each as .wasm (+ .mjs) for the Wasm page and .js for the JS one. Without
# them the playground still fits: without fit_host from the page's thread,
# without fit_worker on the main thread.
set -euo pipefail
cd "$(dirname "$0")/.."
for w in fit_worker fit_host; do
  dart compile wasm -O4 --no-source-maps worker/$w.dart -o web/$w.wasm
  dart compile js -O2 --no-source-maps worker/$w.dart -o web/$w.js
  rm -f web/$w.js.deps web/$w.js.map web/$w.wasm.map
done
ls -l web/fit_worker.* web/fit_host.*
