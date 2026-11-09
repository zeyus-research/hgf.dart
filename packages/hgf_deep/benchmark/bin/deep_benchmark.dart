/// Times DeepNetwork's fit (sequential), batchUpdate and predict, and writes
/// each problem to JSON for `tool/parity/bench_deep.py` (pyhgf JAX, rshgf).
///
///     dart compile exe benchmark/bin/deep_benchmark.dart -o .dart_tool/deep_bench
///     .dart_tool/deep_bench --out .dart_tool/deep_problems.json
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf_deep/hgf_deep.dart';

final class Problem {
  Problem(this.name, this.sizes, this.samples, this.batch);
  final String name;

  /// Bottom (output) first.
  final List<int> sizes;
  final int samples;
  final int batch;
}

final problems = [
  Problem('small', [2, 8, 8, 4], 2000, 256),
  Problem('medium', [10, 64, 64, 16], 2000, 256),
  Problem('wide', [16, 256, 256, 32], 500, 256),
];

Matrix randn(int rows, int cols, math.Random rng) {
  final m = Matrix(rows, cols);
  for (var k = 0; k < m.data.length; k += 2) {
    final u1 = 1 - rng.nextDouble(), u2 = rng.nextDouble();
    final r = math.sqrt(-2 * math.log(u1));
    m.data[k] = r * math.cos(2 * math.pi * u2);
    if (k + 1 < m.data.length) m.data[k + 1] = r * math.sin(2 * math.pi * u2);
  }
  return m;
}

DeepNetwork build(Problem p, List<Matrix> weights) {
  final net = DeepNetwork(coupling: CouplingFn.tanh);
  for (final s in p.sizes) {
    net.addLayer(s);
  }
  net.setWeights(weights);
  return net;
}

double bestMs(void Function() run, {int repeat = 5}) {
  run(); // warm-up
  var best = double.infinity;
  for (var r = 0; r < repeat; r++) {
    final sw = Stopwatch()..start();
    run();
    best = math.min(best, sw.elapsedMicroseconds / 1000);
  }
  return best;
}

void main(List<String> args) {
  final outIndex = args.indexOf('--out');
  final out = <Map<String, Object?>>[];
  for (final p in problems) {
    final rng = math.Random(42);
    final weights = [
      for (var l = 1; l < p.sizes.length; l++)
        () {
          final w = randn(p.sizes[l - 1], p.sizes[l] + 1, rng);
          final scale = 1 / math.sqrt(p.sizes[l]);
          for (var k = 0; k < w.data.length; k++) {
            w.data[k] *= scale;
          }
          return w;
        }(),
    ];
    final x = randn(p.samples, p.sizes.last, rng);
    final y = randn(p.samples, p.sizes.first, rng);
    final xb = Matrix(
      p.batch,
      x.cols,
      Float64List.sublistView(x.data, 0, p.batch * x.cols),
    );
    final yb = Matrix(
      p.batch,
      y.cols,
      Float64List.sublistView(y.data, 0, p.batch * y.cols),
    );

    final fitMs = bestMs(
      () => build(p, weights).fit(x, y, optimiser: Adam(1e-3)),
      repeat: 3,
    );
    final net = build(p, weights);
    final batchMs = bestMs(
      () => net.batchUpdate(xb, yb, optimiser: Adam(1e-3)),
    );
    final predictMs = bestMs(() => net.predict(x));
    stdout.writeln(
      '${p.name.padRight(7)} ${p.sizes}  fit ${p.samples}: '
      '${fitMs.toStringAsFixed(1)} ms (${(fitMs * 1000 / p.samples).toStringAsFixed(1)} µs/sample)   '
      'batchUpdate ${p.batch}: ${batchMs.toStringAsFixed(2)} ms   '
      'predict ${p.samples}: ${predictMs.toStringAsFixed(2)} ms',
    );
    out.add({
      'name': p.name,
      'sizes': p.sizes,
      'batch': p.batch,
      'weights': [for (final w in weights) w.toRows()],
      'x': x.toRows(),
      'y': y.toRows(),
      'dart_ms': {'fit': fitMs, 'batch_update': batchMs, 'predict': predictMs},
    });
  }
  if (outIndex >= 0) {
    File(args[outIndex + 1]).writeAsStringSync(jsonEncode(out));
  }
}
