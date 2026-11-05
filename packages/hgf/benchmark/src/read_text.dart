/// Platform-neutral text-file read and directory listing for the benchmarks
/// and `tool/parity_check.dart`: `dart:io` on the VM,
/// Node's `fs` on JS and Wasm.
library;

export 'inputs/read_stub.dart'
    if (dart.library.io) 'inputs/read_io.dart'
    if (dart.library.js_interop) 'inputs/read_web.dart';
