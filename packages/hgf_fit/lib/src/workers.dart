// `dart.library.isolate` is also defined on dart2wasm, where isolates throw
// at run time; `dart.library.io` is only defined where they work.
export 'workers_stub.dart' if (dart.library.io) 'workers_io.dart';
