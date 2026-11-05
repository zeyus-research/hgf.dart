/// The parity suite as a standalone app, for builds `dart test` can't make:
/// `dart test -c dart2wasm` compiles without `-O4`, which release builds use.
/// From `packages/hgf`:
///
///     dart compile wasm -O4 tool/parity_check.dart -o /tmp/pc.wasm
///     node ../../tool/web/run_wasm.mjs /tmp/pc.wasm
///
/// Each fixture is also run as a `BatchNetwork`, which must match `Network`
/// bit for bit (`test/src/batch_check.dart`).
///
/// Also runs on the VM (`dart run tool/parity_check.dart`) and as dart2js
/// under Node. Prints one line per fixture and throws if any mismatch, so
/// the process exits non-zero.
library;

import 'dart:convert';

import '../benchmark/src/read_text.dart';
import '../test/src/batch_check.dart';
import '../test/src/parity_check.dart';

void main() {
  const dir = 'test/fixtures';
  final names = listDir(dir).where((n) => n.endsWith('.json')).toList()..sort();
  if (names.isEmpty) throw StateError('No fixtures in $dir');
  final failures = <String>[];
  for (final name in names) {
    final fixture = jsonDecode(readText('$dir/$name')) as Map<String, Object?>;
    final f = [...fixtureMismatches(fixture), ...batchMismatches(fixture)];
    print('${f.isEmpty ? 'ok  ' : 'FAIL'} ${fixture['name']}');
    failures.addAll(f);
  }
  if (failures.isNotEmpty) {
    failures.take(20).forEach(print);
    throw StateError('${failures.length} mismatches');
  }
  print('All ${names.length} fixtures match their reference.');
}
