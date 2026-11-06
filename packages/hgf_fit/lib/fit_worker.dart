/// The entry point of a Web Worker for `FitWorkers` on the web.
///
/// Web Workers can't run the page's own compiled code, so the workers are a
/// small program of their own:
///
/// ```dart
/// // web/fit_worker.dart
/// import 'package:hgf_fit/fit_worker.dart';
///
/// void main() => runFitWorker();
/// ```
///
/// Compile it next to the page, with either compiler:
///
/// ```sh
/// dart compile js -O2 web/fit_worker.dart -o web/fit_worker.js
/// dart compile wasm -O2 web/fit_worker.dart -o web/fit_worker.wasm
/// ```
///
/// and point the pool at it:
///
/// ```dart
/// final workers = await FitWorkers.spawn(
///   script: FitWorkerScript.wasm('fit_worker.wasm'),
///   // or FitWorkerScript.js('fit_worker.js')
/// );
/// ```
///
/// Models reach the workers as `FitCodec` JSON. Custom priors, transforms
/// and response models need decoders in the codec passed to
/// [runFitWorker], custom couplings and response functions their names
/// (and the same response functions in the page's `FitWorkerScript.codec`).
///
/// Results are bit-identical to the single-threaded functions when the
/// worker is compiled like the page (both Wasm or both JS); across
/// compilers, `exp` and `log` may differ in the last bit.
library;

export 'src/codec.dart' show FitCodec, JsonDecoder;
export 'src/fit_worker_stub.dart'
    if (dart.library.js_interop) 'src/fit_worker_web.dart';
