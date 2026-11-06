// The Web Worker used by workers_web_test.dart. Compiled by
// `melos run test:web-workers` (or by hand):
//   dart compile js -O2 test/web/fit_worker.dart -o test/web/fit_worker.js
//   dart compile wasm -O2 test/web/fit_worker.dart -o test/web/fit_worker.wasm
import 'package:hgf_fit/fit_worker.dart';

import 'test_codec.dart';

void main() => runFitWorker(codec: testCodec);
