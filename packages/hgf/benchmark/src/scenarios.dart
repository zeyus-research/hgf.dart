/// Benchmark scenarios shared by every runtime (VM JIT, AOT, JS, Wasm).
///
/// Mirrors `tool/parity/bench_python.py`: same inputs and network specs
/// (`benchmark/data/inputs.json`), median of `repeat` warm runs.
library;

import 'dart:typed_data';

import 'package:hgf/hgf.dart';

typedef BenchResult = Map<String, Object>;

BenchResult result(String name, double ms, {String? extra}) => {
  'name': name,
  'unit': 'ms',
  'value': ms,
  'extra': ?extra,
};

/// Median wall time per call of [fn] over [repeat] samples, after [warmup]
/// untimed calls. Short calls are batched so each sample spans at least
/// ~20 ms (the dart2js Stopwatch only has millisecond resolution under Node).
double medianMs(void Function() fn, {required int repeat, int warmup = 3}) {
  final sw = Stopwatch()..start();
  for (var i = 0; i < warmup; i++) {
    fn();
  }
  sw.stop();
  final perCall = warmup == 0 ? 1.0 : sw.elapsedMicroseconds / 1000.0 / warmup;
  final inner = perCall >= 20
      ? 1
      : (20 / (perCall <= 0 ? 0.01 : perCall)).ceil();
  final times = <double>[];
  for (var i = 0; i < repeat; i++) {
    sw
      ..reset()
      ..start();
    for (var k = 0; k < inner; k++) {
      fn();
    }
    sw.stop();
    times.add(sw.elapsedMicroseconds / 1000.0 / inner);
  }
  times.sort();
  return times[times.length ~/ 2];
}

Float64List floats(Object? v) => Float64List.fromList([
  for (final x in v as List<Object?>) (x as num).toDouble(),
]);

Network buildSpec(Map<String, Object?> inputs, String name) =>
    NetworkSpec.fromJson(
      (inputs['specs'] as Map<String, Object?>)[name] as Map<String, Object?>,
    ).build()..build();

/// Single-series scenarios: with trajectories (as used) and without.
List<BenchResult> seriesScenarios(
  Map<String, Object?> inputs, {
  required String runtime,
  required int repeat,
}) {
  final out = <BenchResult>[];
  final cont = floats(inputs['continuous']);
  final binary = floats(inputs['binary']);
  final usd = floats(inputs['usdchf']);
  for (final (name, spec, data) in [
    ('continuous_3level_usdchf_614', 'continuous_3level', usd),
    ('continuous_3level_10k', 'continuous_3level', cont),
    ('binary_3level_10k', 'binary_3level', binary),
  ]) {
    final net = buildSpec(inputs, spec);
    out.add(
      result(
        '$name/$runtime',
        medianMs(() => net.inputData(data), repeat: repeat),
      ),
    );
    out.add(
      result(
        '$name/${runtime}_no_traj',
        medianMs(() => net.inputData(data, record: false), repeat: repeat),
      ),
    );
    // Sanity: the run must stay finite, or the timing is meaningless.
    final last = net.get(1, NodeField.mean);
    if (!last.isFinite) throw StateError('$name diverged (mu1 = $last)');
  }
  return out;
}

/// Fitting-shaped batch: one forward pass per tonic volatility of node 1,
/// final state only. Returns the final mean of nodes 1 and 2 per run.
Float64List batchRun(
  Map<String, Object?> spec,
  Float64List data,
  Float64List omegas,
) {
  final net = NetworkSpec.fromJson(spec).build()..build();
  final out = Float64List(omegas.length * 2);
  for (var k = 0; k < omegas.length; k++) {
    net.setInitial(1, NodeField.tonicVolatility, omegas[k]);
    net.inputData(data, record: false);
    out[2 * k] = net.get(1, NodeField.mean);
    out[2 * k + 1] = net.get(2, NodeField.mean);
  }
  return out;
}

/// [batchRun] on a [BatchNetwork]: every ω in one lock-step pass.
Float64List batchNetworkRun(
  Map<String, Object?> spec,
  Float64List data,
  Float64List omegas, {
  int blockSize = defaultBlockSize,
}) {
  final batch = BatchNetwork(
    NetworkSpec.fromJson(spec).build(),
    omegas.length,
    blockSize: blockSize,
  )..setInitial(1, NodeField.tonicVolatility, omegas);
  batch.inputData(data);
  final out = Float64List(omegas.length * 2);
  for (var k = 0; k < omegas.length; k++) {
    out[2 * k] = batch.get(k, 1, NodeField.mean);
    out[2 * k + 1] = batch.get(k, 2, NodeField.mean);
  }
  return out;
}

/// [BatchNetwork]'s default block size.
const int defaultBlockSize = 256;

const int batchTrials = 1000;

List<BenchResult> batchScenario(
  Map<String, Object?> inputs, {
  required String runtime,
  required int repeat,
}) {
  final spec =
      (inputs['specs'] as Map<String, Object?>)['continuous_2level']
          as Map<String, Object?>;
  final data = Float64List.sublistView(
    floats(inputs['continuous']),
    0,
    batchTrials,
  );
  final omegas = floats(inputs['omegas']);
  // The two engines must agree (bit for bit), or the timing compares
  // different work.
  final loop = batchRun(spec, data, omegas);
  final lockStep = batchNetworkRun(spec, data, omegas);
  for (var k = 0; k < loop.length; k++) {
    if (loop[k] != lockStep[k] && !(loop[k].isNaN && lockStep[k].isNaN)) {
      throw StateError('BatchNetwork differs from Network at $k');
    }
  }
  return [
    result(
      'batch_1000x1000_continuous_2level/${runtime}_loop',
      medianMs(() => batchRun(spec, data, omegas), repeat: repeat, warmup: 1),
    ),
    result(
      'batch_1000x1000_continuous_2level/${runtime}_batch',
      medianMs(
        () => batchNetworkRun(spec, data, omegas),
        repeat: repeat,
        warmup: 1,
      ),
    ),
  ];
}
