#!/usr/bin/env bash
# Compiles the fit worker into web/, next to the page, so that both
# `flutter run -d chrome` and `flutter build web` serve it. Without it the
# playground still fits, on the main thread.
set -euo pipefail
cd "$(dirname "$0")/.."
dart compile wasm -O4 --no-source-maps worker/fit_worker.dart \
  -o web/fit_worker.wasm
dart compile js -O2 --no-source-maps worker/fit_worker.dart \
  -o web/fit_worker.js
rm -f web/fit_worker.js.deps web/fit_worker.js.map web/fit_worker.wasm.map
ls -l web/fit_worker.*
