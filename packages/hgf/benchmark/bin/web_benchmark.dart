/// hgf.dart benchmark for JS and Wasm, run under Node (see
/// tool/bench_compare.dart and tool/web/).
///
/// Reads `benchmark/data/inputs.json` (relative to the working directory)
/// through Node's `fs`, and prints the results as one JSON line prefixed
/// with `BENCH_JSON:`.
library;

import 'dart:convert';
import 'dart:js_interop';

import '../src/scenarios.dart';

@JS('process.getBuiltinModule')
external _Fs _getBuiltinModule(String name);

extension type _Fs(JSObject _) implements JSObject {
  external String readFileSync(String path, String encoding);
}

void main() {
  const runtime = bool.fromEnvironment('dart.tool.dart2wasm')
      ? 'dart_wasm'
      : 'dart_js';
  final json = _getBuiltinModule(
    'fs',
  ).readFileSync('benchmark/data/inputs.json', 'utf8');
  final inputs = jsonDecode(json) as Map<String, Object?>;
  final results = <BenchResult>[
    ...seriesScenarios(inputs, runtime: runtime, repeat: 7),
    ...batchScenario(inputs, runtime: runtime, repeat: 3),
  ];
  print('BENCH_JSON:${jsonEncode(results)}');
}
