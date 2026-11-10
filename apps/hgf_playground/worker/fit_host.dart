// The Web Worker the playground's fit jobs run in on the web (see
// `spawnFitHost`): drivers, pool (nested fit_worker workers) and WebGPU off
// the page's thread. Built into web/ by tool/build_worker.sh, as
// fit_host.wasm (+ .mjs) for the Wasm page and fit_host.js for the JS one.
import 'package:hgf_playground/src/model/fit_host_worker.dart';

void main() => runFitHostWorker();
