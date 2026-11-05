/// BatchNetwork cost per element-trial by update mode and block size, against
/// the scalar Network looping over the same parameter sets (diagnostic).
///
/// Runs unchanged on every runtime, from `packages/hgf`:
///
///     dart compile exe benchmark/bin/batch_blocks.dart -o /tmp/bb && /tmp/bb
///     dart compile wasm -O4 benchmark/bin/batch_blocks.dart -o /tmp/bb.wasm
///     node ../../tool/web/run_wasm.mjs /tmp/bb.wasm
///
/// The workload is the benchmark's batch: the 2-level spec, its first 1000
/// trials, 1000 tonic volatilities of node 1 (also with the 3-level spec at
/// ω2 = ω3 = −6, where every mode stays finite; see profile_modes.dart).
library;

import 'dart:convert';
import 'dart:typed_data';

import '../src/read_text.dart';
import '../src/scenarios.dart';

void main(List<String> args) {
  final inputs =
      jsonDecode(readText('benchmark/data/inputs.json'))
          as Map<String, Object?>;
  final specs = inputs['specs'] as Map<String, Object?>;
  final data = Float64List.sublistView(
    floats(inputs['continuous']),
    0,
    batchTrials,
  );
  final omegas = floats(inputs['omegas']);
  final work = data.length * omegas.length;
  String ns(double ms) => (ms * 1e6 / work).toStringAsFixed(0).padLeft(5);

  Map<String, Object?> spec(String name, String mode) {
    final s = jsonDecode(jsonEncode(specs[name])) as Map<String, Object?>;
    (s['network'] as Map<String, Object?>)['volatility_updates'] = mode;
    if (name == 'continuous_3level') {
      for (final i in [2, 3]) {
        final node = (s['nodes'] as List<Object?>)[i] as Map<String, Object?>;
        (node['params'] as Map<String, Object?>)['tonic_volatility'] = -6.0;
      }
    }
    return s;
  }

  const blocks = [8, 16, 32, 64, 128, 256, 1000];
  print(
    'ns per element-trial; columns: Network loop, then BatchNetwork by '
    'block size ${blocks.join(', ')}',
  );
  for (final name in ['continuous_2level', 'continuous_3level']) {
    for (final mode in ['standard', 'eHGF', 'unbounded']) {
      final s = spec(name, mode);
      final ref = batchRun(s, data, omegas);
      var finite = 0;
      for (final x in ref) {
        if (x.isFinite) finite++;
      }
      final row = StringBuffer(
        '${name.padRight(18)} ${mode.padRight(9)} '
        '${ns(medianMs(() => batchRun(s, data, omegas), repeat: 5, warmup: 1))} |',
      );
      for (final b in blocks) {
        final ms = medianMs(
          () => batchNetworkRun(s, data, omegas, blockSize: b),
          repeat: 5,
          warmup: 1,
        );
        row.write(' ${ns(ms)}');
      }
      if (finite < ref.length) {
        row.write('  (${ref.length - finite} of ${ref.length} non-finite)');
      }
      print(row);
    }
  }
}
