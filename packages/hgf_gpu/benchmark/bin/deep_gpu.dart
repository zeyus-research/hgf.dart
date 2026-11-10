/// hgf_deep's `DeepNetwork` on the GPU (`GpuDeepNetwork`, wgpu-native)
/// against the f64 CPU engine on one core, on the problems of hgf_deep's
/// `deep_benchmark.dart` (same weights and data). From `packages/hgf_gpu`:
///
///     dart compile exe benchmark/bin/deep_gpu.dart -o .dart_tool/deep_gpu
///     .dart_tool/deep_gpu [--batches 256,4096] [--no-cpu]
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf_deep/hgf_deep.dart';
import 'package:hgf_gpu/hgf_gpu.dart';
import 'package:hgf_gpu/native.dart';

final class Problem {
  Problem(this.name, this.sizes, this.samples);
  final String name;
  final List<int> sizes;
  final int samples;
}

final problems = [
  Problem('small', [2, 8, 8, 4], 2000),
  Problem('medium', [10, 64, 64, 16], 2000),
  Problem('wide', [16, 256, 256, 32], 500),
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

Matrix rows(Matrix m, int start, int n) => Matrix(
  n,
  m.cols,
  Float64List.sublistView(m.data, start * m.cols, (start + n) * m.cols),
);

/// The rows of [m] repeated to [n] rows.
Matrix tiled(Matrix m, int n) {
  final out = Matrix(n, m.cols);
  for (var r = 0; r < n; r++) {
    out.data.setRange(
      r * m.cols,
      (r + 1) * m.cols,
      m.data,
      (r % m.rows) * m.cols,
    );
  }
  return out;
}

Future<double> bestMs(Future<void> Function() run, {int repeat = 20}) async {
  // Warm-up: compilation, and the GPU out of its idle clocks.
  for (var i = 0; i < 3; i++) {
    await run();
  }
  var best = double.infinity;
  for (var r = 0; r < repeat; r++) {
    final sw = Stopwatch()..start();
    await run();
    best = math.min(best, sw.elapsedMicroseconds / 1000);
  }
  return best;
}

String ms(double v) => v.toStringAsFixed(v < 10 ? 2 : 1);

Future<void> main(List<String> args) async {
  final backend = WgpuBackend.create();
  if (backend == null) {
    stderr.writeln('No wgpu adapter');
    exitCode = 1;
    return;
  }
  final bi = args.indexOf('--batches');
  final batches = bi >= 0
      ? [for (final s in args[bi + 1].split(',')) int.parse(s)]
      : [256, 4096];
  final cpuToo = !args.contains('--no-cpu');
  print(backend.description);
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
    final x0 = randn(p.samples, p.sizes.last, rng);
    final y0 = randn(p.samples, p.sizes.first, rng);
    for (final b in batches) {
      // An epoch of 8 minibatches, and predict on the same rows.
      final n = 8 * b;
      final x = tiled(x0, n), y = tiled(y0, n);
      final xb = rows(x, 0, b), yb = rows(y, 0, b);
      final cpu = build(p, weights);
      final gpu = await GpuDeepNetwork.create(backend, cpu, capacity: b);
      final adam = Adam(1e-3);
      final step = await bestMs(() => gpu.batchUpdate(xb, yb, optimiser: adam));
      final epoch = await bestMs(() async {
        await gpu.trainEpoch(x, y, batchSize: b, optimiser: adam);
        await gpu.idle();
      });
      final predict = await bestMs(() => gpu.predict(x));
      gpu.dispose();
      var line =
          '${p.name.padRight(6)} ${p.sizes} B=$b  GPU: batchUpdate '
          '${ms(step)} ms (with readback), epoch ${ms(epoch / 8)} ms/step, '
          'predict $n: ${ms(predict)} ms';
      if (cpuToo) {
        final cpuStep = await bestMs(
          () async => cpu.batchUpdate(xb, yb, optimiser: adam),
          repeat: 3,
        );
        final cpuPredict = await bestMs(() async => cpu.predict(x), repeat: 3);
        line +=
            '   CPU f64: batchUpdate ${ms(cpuStep)} ms, predict '
            '${ms(cpuPredict)} ms   (${(cpuStep / (epoch / 8)).toStringAsFixed(1)}× / '
            '${(cpuPredict / predict).toStringAsFixed(1)}×)';
      }
      print(line);
    }
  }
}
