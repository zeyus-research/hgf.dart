/// `GpuDeepEnsemble` (hgf_deep's sequential `fit` for many networks at
/// once, wgpu-native) against `DeepNetwork.fit` (f64) on one CPU core. From
/// `packages/hgf_gpu`:
///
///     dart compile exe benchmark/bin/deep_ensemble.dart -o .dart_tool/deep_ensemble
///     .dart_tool/deep_ensemble [--members 1,64,1024] [--rows 500] \
///       [--wg 32,64,128,256] [--problems small,medium,wide] [--state]
///
/// `--state` times each case with the working state in storage, in
/// workgroup memory (`ws`), and with the parameters there too (`ws+wp`),
/// where they fit; by default the ensemble picks.
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:hgf_deep/hgf_deep.dart';
import 'package:hgf_gpu/hgf_gpu.dart';
import 'package:hgf_gpu/native.dart';

final problems = {
  'small': [2, 8, 8, 4],
  'medium': [10, 64, 64, 16],
  'wide': [16, 256, 256, 32],
};

Matrix randn(int rows, int cols, math.Random rng, [double scale = 1]) {
  final m = Matrix(rows, cols);
  for (var k = 0; k < m.data.length; k++) {
    final u1 = 1 - rng.nextDouble(), u2 = rng.nextDouble();
    m.data[k] =
        scale * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2);
  }
  return m;
}

DeepNetwork network(List<int> sizes, int seed) {
  final net = DeepNetwork(coupling: CouplingFn.tanh);
  for (final s in sizes) {
    net.addLayer(s);
  }
  return net..weightInitialisation(const XavierInit(), seed: seed);
}

List<int> intList(List<String> args, String flag, List<int> fallback) {
  final i = args.indexOf(flag);
  return i < 0
      ? fallback
      : [for (final s in args[i + 1].split(',')) int.parse(s)];
}

Future<void> main(List<String> args) async {
  final backend = WgpuBackend.create();
  if (backend == null) {
    stderr.writeln('wgpu-native not available');
    exit(1);
  }
  print(backend.description);
  final rows = intList(args, '--rows', [500]).first;
  final memberCounts = intList(args, '--members', [1, 16, 256, 1024, 4096]);
  final workgroups = intList(args, '--wg', []);
  // (state, params) in workgroup memory; null: the ensemble's choice.
  final stateModes = args.contains('--state')
      ? [(false, false), (true, false), (true, true)]
      : [(null, null)];
  final rng = math.Random(1);
  final only = args.contains('--problems')
      ? args[args.indexOf('--problems') + 1].split(',')
      : problems.keys.toList();
  for (final MapEntry(key: name, value: sizes) in problems.entries) {
    if (!only.contains(name)) continue;
    final x = randn(rows, sizes.last, rng, 0.5);
    final y = randn(rows, sizes.first, rng, 0.5);
    // CPU: one network, f64, one core.
    var cpuMs = double.infinity;
    for (var r = 0; r < 3; r++) {
      final net = network(sizes, 0);
      final sw = Stopwatch()..start();
      net.fit(x, y, optimiser: Adam(1e-3));
      cpuMs = math.min(cpuMs, sw.elapsedMicroseconds / 1000);
    }
    print(
      '\n$name $sizes, $rows rows: CPU f64 one network '
      '${cpuMs.toStringAsFixed(1)} ms '
      '(${(cpuMs * 1000 / rows).toStringAsFixed(1)} µs/sample)',
    );
    for (final e in memberCounts) {
      for (final (wg, ws) in [
        for (final wg in workgroups.isEmpty ? [null] : workgroups)
          for (final ws in stateModes) (wg, ws),
      ]) {
        final nets = [for (var i = 0; i < e; i++) network(sizes, i)];
        final GpuDeepEnsemble gpu;
        try {
          gpu = await GpuDeepEnsemble.create(
            backend,
            nets,
            workgroupSize: wg,
            workgroupState: ws.$1,
            workgroupParams: ws.$2,
          );
        } on ArgumentError catch (err) {
          print('  $e members: ${err.message}');
          continue;
        }
        var best = double.infinity;
        var nonFinite = 0;
        for (var r = 0; r < 5; r++) {
          final sw = Stopwatch()..start();
          final pred = (await gpu.fit([x], [y], optimiser: Adam(1e-3)))!;
          final ms = sw.elapsedMicroseconds / 1000;
          if (r > 0) best = math.min(best, ms);
          nonFinite = 0;
          for (final p in pred) {
            for (final v in p.data) {
              if (!v.isFinite) nonFinite++;
            }
          }
        }
        gpu.dispose();
        final perMemberSample = best * 1000 / (e * rows);
        print(
          '  ${'$e'.padLeft(5)} members (wg ${gpu.workgroupSize}'
          '${gpu.kernels.workgroupState ? ', ws' : ''}'
          '${gpu.kernels.workgroupParams ? '+wp' : ''}): '
          '${best.toStringAsFixed(1).padLeft(8)} ms, '
          '${perMemberSample.toStringAsFixed(3)} µs/member-sample, '
          '${(cpuMs * e / best).toStringAsFixed(1)}× one core'
          '${nonFinite > 0 ? ' ($nonFinite non-finite predictions)' : ''}',
        );
      }
    }
  }
}
