import 'codec.dart';

/// Runs this program as a `FitWorkers` Web Worker. Only for programs
/// compiled for the web; on the VM, `FitWorkers` uses isolates.
void runFitWorker({FitCodec codec = const FitCodec()}) =>
    throw UnsupportedError(
      'runFitWorker is the entry point of a Web Worker compiled with '
      '`dart compile js` or `dart compile wasm`',
    );
