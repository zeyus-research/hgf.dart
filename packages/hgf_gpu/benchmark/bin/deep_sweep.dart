/// hgf_deep across network widths: `DeepNetwork` (f64, one core) and
/// `GpuDeepNetwork` (f32, wgpu-native) on tanh networks [4, H, H, 16]
/// (output first), with Adam. Writes each problem (weights, data) and the
/// Dart timings for `tool/parity/bench_deep.py --sweep` (pyhgf JAX on CPU
/// and GPU, rshgf). From `packages/hgf_gpu`:
///
///     dart compile exe benchmark/bin/deep_sweep.dart -o .dart_tool/deep_sweep
///     .dart_tool/deep_sweep --out ../../.dart_tool/deep_sweep.json
///
/// Timings are the best of several runs: `fit` (sequential, 500 samples),
/// `batch_update` (256 samples, with the GPU's readback of the errors) and
/// `predict` (2000 samples).
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:hgf_deep/hgf_deep.dart';
import 'package:hgf_gpu/hgf_gpu.dart';
import 'package:hgf_gpu/native.dart';

const fitSamples = 500, batch = 256, predictSamples = 2000;

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

Matrix head(Matrix m, int n) =>
    Matrix(n, m.cols, m.data.sublist(0, n * m.cols));

Future<double> bestMs(Future<void> Function() run, {int repeat = 5}) async {
  for (var i = 0; i < 2; i++) {
    await run(); // compile, and the GPU out of its idle clocks
  }
  var best = double.infinity;
  for (var r = 0; r < repeat; r++) {
    final sw = Stopwatch()..start();
    await run();
    best = math.min(best, sw.elapsedMicroseconds / 1000);
  }
  return best;
}

Future<void> main(List<String> args) async {
  var widths = [8, 32, 64, 128, 256, 512];
  String? out;
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--widths':
        widths = [for (final s in args[++i].split(',')) int.parse(s)];
      case '--out':
        out = args[++i];
    }
  }
  final backend = WgpuBackend.create();
  if (backend == null) stderr.writeln('No wgpu adapter: CPU only');
  final problems = <Map<String, Object?>>[];
  for (final h in widths) {
    final sizes = [4, h, h, 16];
    final rng = math.Random(42);
    final weights = [
      for (var l = 1; l < sizes.length; l++)
        () {
          final w = randn(sizes[l - 1], sizes[l] + 1, rng);
          final scale = 1 / math.sqrt(sizes[l]);
          for (var k = 0; k < w.data.length; k++) {
            w.data[k] *= scale;
          }
          return w;
        }(),
    ];
    DeepNetwork build() {
      final net = DeepNetwork(coupling: CouplingFn.tanh);
      for (final s in sizes) {
        net.addLayer(s);
      }
      return net..setWeights(weights);
    }

    final x = randn(predictSamples, sizes.last, rng);
    final y = randn(predictSamples, sizes.first, rng);
    final xf = head(x, fitSamples), yf = head(y, fitSamples);
    final xb = head(x, batch), yb = head(y, batch);

    final cpu = build();
    final ms = <String, Object?>{
      'cpu_fit': await bestMs(
        () async => build().fit(xf, yf, optimiser: Adam(1e-3)),
        repeat: 3,
      ),
      'cpu_batch_update': await bestMs(
        () async => cpu.batchUpdate(xb, yb, optimiser: Adam(1e-3)),
      ),
      'cpu_predict': await bestMs(() async => cpu.predict(x)),
    };
    if (backend != null) {
      final gpu = await GpuDeepNetwork.create(
        backend,
        build(),
        capacity: predictSamples,
      );
      final adam = Adam(1e-3);
      ms['gpu_batch_update'] = await bestMs(
        () => gpu.batchUpdate(xb, yb, optimiser: adam),
        repeat: 20,
      );
      ms['gpu_predict'] = await bestMs(() => gpu.predict(x), repeat: 20);
      gpu.dispose();
    }
    print(
      'H=$h ${[for (final e in ms.entries) '${e.key} ${(e.value! as double).toStringAsFixed(2)}'].join('  ')} ms',
    );
    problems.add({
      'name': 'h$h',
      'width': h,
      'sizes': sizes,
      'batch': batch,
      'fit_samples': fitSamples,
      'weights': [for (final w in weights) w.toRows()],
      'x': x.toRows(),
      'y': y.toRows(),
      'dart_ms': ms,
      if (backend != null)
        'device': backend.description.split(' / ').elementAtOrNull(1),
    });
  }
  if (out != null) File(out).writeAsStringSync(jsonEncode(problems));
}
