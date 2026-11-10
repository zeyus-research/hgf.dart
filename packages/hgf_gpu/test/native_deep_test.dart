/// hgf_deep's `DeepNetwork` on the GPU (`GpuDeepNetwork`, f32) against the
/// f64 CPU engine, on wgpu-native. Skipped if the library can't be loaded.
///
///     dart test --tags gpu test/native_deep_test.dart
@TestOn('vm')
@Tags(['gpu'])
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:hgf_deep/hgf_deep.dart';
import 'package:hgf_gpu/hgf_gpu.dart';
import 'package:hgf_gpu/native.dart';
import 'package:test/test.dart';

import 'src/deep_gpu_check.dart';

WgpuBackend? _backend() {
  try {
    return WgpuBackend.create();
  } on ArgumentError {
    return null;
  }
}

void main() {
  final backend = _backend();
  final skip = backend == null ? 'wgpu-native not available' : null;

  final files =
      Directory('../hgf_deep/test/fixtures')
          .listSync()
          .whereType<File>()
          .where((f) => f.path.endsWith('.json'))
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path));

  group('hgf_deep fixtures: batchUpdate and predict', skip: skip, () {
    for (final file in files) {
      final fixture =
          jsonDecode(file.readAsStringSync()) as Map<String, Object?>;
      test('${fixture['name']}', () async {
        final all = await checkFixture(backend!, fixture);
        for (final r in all.where((r) => r.$2.isNaN)) {
          print('${fixture['name']}: ${r.$1}');
        }
        final results = [
          for (final r in all)
            if (!r.$2.isNaN) r,
        ];
        if (results.isEmpty) return;
        final worst = results.reduce((a, b) => a.$2 >= b.$2 ? a : b);
        final lost = results.fold(0, (n, r) => n + r.$3);
        print(
          '${fixture['name']}: max err ${worst.$2.toStringAsExponential(2)} '
          '(${worst.$1})${lost > 0 ? ', $lost lost' : ''}',
        );
        final bad = failures(fixture['name']! as String, results);
        expect(bad, isEmpty, reason: bad.join('\n'));
      });
    }
  });

  group('random networks', skip: skip, () {
    test('split-K weight gradient: batch 300 of capacity 512', () async {
      final cpu = randomNetwork([3, 8, 6], 11);
      final gpu = await GpuDeepNetwork.create(backend!, cpu, capacity: 512);
      final splits = [
        for (final k in gpu.kernels.batchUpdate(
          errors: true,
          optimiser: DeepOptimiserKind.sgd,
          updatePrecisions: true,
        ))
          if (k.name.contains('apply')) k.name,
      ];
      expect(splits, hasLength(2), reason: 'both layers should split');
      final rng = math.Random(5);
      final x = randn(300, 6, rng), y = randn(300, 3, rng);
      final cpuErr = cpu.batchUpdate(x, y, optimiser: const Sgd(0.05));
      final gpuErr = await gpu.batchUpdate(x, y, optimiser: const Sgd(0.05));
      final results = await compareNetworks(
        gpu,
        cpu,
        randomNetwork([3, 8, 6], 11),
      );
      final e = compare(cpuErr.data, gpuErr!.data);
      results.add(('input errors', e.err, e.lost));
      gpu.dispose();
      final bad = [
        for (final r in results)
          if (r.$2 > tolerance || r.$3 > 0)
            '${r.$1}: ${r.$2.toStringAsExponential(2)}, ${r.$3} lost',
      ];
      expect(bad, isEmpty, reason: bad.join('\n'));
    });

    for (final update in DeepVolatilityUpdate.values) {
      test('10 Adam steps over partial tiles, ${update.name}', () async {
        final cpu = randomNetwork([5, 37, 70, 6], 7, update: update);
        final gpu = await GpuDeepNetwork.create(backend!, cpu, capacity: 100);
        final rng = math.Random(3);
        final x = randn(450, 6, rng);
        final y = randn(450, 5, rng);
        final adam = Adam(1e-3);
        // CPU: the same minibatches as trainEpoch.
        for (var s = 0; s < 450; s += 100) {
          final b = math.min(100, 450 - s);
          cpu.batchUpdate(
            Matrix(b, 6, x.data.sublist(s * 6, (s + b) * 6)),
            Matrix(b, 5, y.data.sublist(s * 5, (s + b) * 5)),
            optimiser: adam,
          );
        }
        for (var s = 0; s < 450; s += 100) {
          final b = math.min(100, 450 - s);
          cpu.batchUpdate(
            Matrix(b, 6, x.data.sublist(s * 6, (s + b) * 6)),
            Matrix(b, 5, y.data.sublist(s * 5, (s + b) * 5)),
            optimiser: adam,
          );
        }
        final gpuAdam = Adam(1e-3);
        await gpu.trainEpoch(x, y, batchSize: 100, optimiser: gpuAdam);
        await gpu.trainEpoch(x, y, batchSize: 100, optimiser: gpuAdam);
        final results = await compareNetworks(
          gpu,
          cpu,
          randomNetwork([5, 37, 70, 6], 7, update: update),
        );
        final pc = compare(cpu.predict(x).data, (await gpu.predict(x)).data);
        results.add(('predict', pc.err, pc.lost));
        final worst = results.reduce((a, b) => a.$2 >= b.$2 ? a : b);
        print(
          '${update.name}: max err ${worst.$2.toStringAsExponential(2)} '
          '(${worst.$1})',
        );
        final bad = [
          for (final r in results)
            if (r.$2 > tolerance || r.$3 > 0)
              '${r.$1}: ${r.$2.toStringAsExponential(2)}, ${r.$3} lost',
        ];
        gpu.dispose();
        expect(bad, isEmpty, reason: bad.join('\n'));
      });
    }
  });
}
