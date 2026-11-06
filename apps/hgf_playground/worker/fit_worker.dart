// The Web Worker that the playground's fits run on (see
// `spawnPlaygroundWorkers`). Built into web/ by tool/build_worker.sh, as
// fit_worker.wasm (+ .mjs) for the Wasm page and fit_worker.js for the JS
// fallback.
import 'package:hgf_fit/fit_worker.dart';

void main() => runFitWorker();
