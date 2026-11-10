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

  group('hgf_deep fixtures: sequential fit (GpuDeepEnsemble)', skip: skip, () {
    for (final file in files) {
      final fixture =
          jsonDecode(file.readAsStringSync()) as Map<String, Object?>;
      if (!(fixture['ops'] as List).any((o) => (o as Map)['op'] == 'fit')) {
        continue;
      }
      test('${fixture['name']}', () async {
        final all = await checkFitOps(backend!, fixture);
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
          '${fixture['name']} fit: max err '
          '${worst.$2.toStringAsExponential(2)} (${worst.$1})'
          '${lost > 0 ? ', $lost lost' : ''}',
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

  group('GpuDeepEnsemble: random members vs DeepNetwork.fit', skip: skip, () {
    Future<void> check({
      required List<int> sizes,
      required int members,
      required int rows,
      required bool shared,
      required Optimiser Function() optimiser,
      DeepVolatilityUpdate update = DeepVolatilityUpdate.unbounded,
      bool updatePrecisions = true,
      bool weightUpdate = true,
      int rowsPerDispatch = 64,
      int? workgroupSize,
      bool carryPrecision = true,
      int minFull = 2,
    }) async {
      DeepNetwork net(int seed) => randomNetwork(
        sizes,
        seed,
        update: update,
        carryPrecision: carryPrecision,
      );
      final rng = math.Random(17);
      final nets = [for (var e = 0; e < members; e++) net(100 + e)];
      final xs = [
        for (var e = 0; e < (shared ? 1 : members); e++)
          randn(rows, sizes.last, rng, 0.5),
      ];
      final ys = [
        for (var e = 0; e < (shared ? 1 : members); e++)
          randn(rows, sizes.first, rng, 0.5),
      ];
      final gpu = await GpuDeepEnsemble.create(
        backend!,
        nets,
        workgroupSize: workgroupSize,
      );
      final cpuOpt = [for (var e = 0; e < members; e++) optimiser()];
      final gpuOpt = optimiser();
      // Two calls, so optimiser state and carried beliefs continue.
      final half = rows ~/ 2;
      Matrix part(Matrix m, int a, int b) =>
          Matrix(b - a, m.cols, m.data.sublist(a * m.cols, b * m.cols));
      final gpuPred = <List<Matrix>>[];
      for (final (a, b) in [(0, half), (half, rows)]) {
        gpuPred.add(
          (await gpu.fit(
            [for (final x in xs) part(x, a, b)],
            [for (final y in ys) part(y, a, b)],
            optimiser: gpuOpt,
            weightUpdate: weightUpdate,
            updatePrecisions: updatePrecisions,
            rowsPerDispatch: rowsPerDispatch,
          ))!,
        );
      }
      // Members whose f64 reference runs away (P18) are compared up to the
      // row where it leaves f32.
      final bad = <String>[];
      final runaways = <String>[];
      var worst = 0.0;
      for (var e = 0; e < members; e++) {
        final ref = net(100 + e);
        int? until;
        final x = xs[shared ? 0 : e], y = ys[shared ? 0 : e];
        for (final (i, (a, b)) in [(0, half), (half, rows)].indexed) {
          final p = ref.fit(
            part(x, a, b),
            part(y, a, b),
            optimiser: cpuOpt[e],
            weightUpdate: weightUpdate,
            updatePrecisions: updatePrecisions,
            record: LayerField.values.toSet(),
          );
          final leaves = until == null
              ? firstRowOutsideF32(ref.trajectories!)
              : (0, '');
          if (until == null && leaves != null) {
            until = a + leaves.$1;
            runaways.add('$e at row $until');
          }
          final n = until == null
              ? p.data.length
              : math.max(0, until - a) * p.cols;
          final c = compare(
            p.data.sublist(0, n),
            gpuPred[i][e].data.sublist(0, n),
          );
          worst = math.max(worst, c.err);
          if (c.err > tolerance || c.lost > 0) {
            bad.add('member $e call $i predictions: ${c.err}, ${c.lost} lost');
          }
        }
        if (until != null) continue;
        final scratch = net(0);
        await gpu.download(e, scratch);
        for (final r in compareStates(ref, scratch)) {
          worst = math.max(worst, r.$2);
          if (r.$2 > tolerance || r.$3 > 0) {
            bad.add('member $e ${r.$1}: ${r.$2}, ${r.$3} lost');
          }
        }
      }
      gpu.dispose();
      print(
        'max err ${worst.toStringAsExponential(2)}'
        '${runaways.isEmpty ? '' : '; f64 runs away: ${runaways.join(', ')}'}',
      );
      expect(bad, isEmpty, reason: bad.join('\n'));
      expect(
        members - runaways.length,
        greaterThanOrEqualTo(minFull),
        reason: 'too few members compared in full',
      );
    }

    for (final update in DeepVolatilityUpdate.values) {
      test('${update.name}, Adam, 6 members on their own data', () async {
        await check(
          sizes: [3, 12, 9, 4],
          members: 6,
          rows: 120,
          shared: false,
          optimiser: () => Adam(1e-3),
          update: update,
        );
      });
    }
    test('no carried precision, Adam, 400 rows', () async {
      await check(
        sizes: [3, 12, 9, 4],
        members: 4,
        rows: 400,
        shared: false,
        optimiser: () => Adam(1e-3),
        carryPrecision: false,
        minFull: 4,
      );
    });
    test('SGD, shared data, frozen precisions, one row per dispatch', () async {
      await check(
        sizes: [2, 5, 3],
        members: 3,
        rows: 40,
        shared: true,
        optimiser: () => const Sgd(0.02),
        updatePrecisions: false,
        rowsPerDispatch: 1,
      );
    });
    test('no weight update, layers wider than the workgroup', () async {
      await check(
        sizes: [4, 150, 70, 3],
        members: 2,
        rows: 30,
        shared: true,
        optimiser: () => const Sgd(0.0),
        weightUpdate: false,
        workgroupSize: 32,
        minFull: 1,
      );
    });
    test('without reading the predictions: same weights as with', () async {
      // The kernel declares its OUT binding either way.
      final rng = math.Random(23);
      final x = randn(40, 2, rng, 0.5), y = randn(40, 3, rng, 0.5);
      final a = [randomNetwork([3, 6, 2], 1)];
      final b = [randomNetwork([3, 6, 2], 1)];
      final ga = await GpuDeepEnsemble.create(backend!, a);
      final gb = await GpuDeepEnsemble.create(backend, b);
      await ga.fit([x], [y], optimiser: Adam(1e-2));
      expect(
        await gb.fit([x], [y], optimiser: Adam(1e-2), readPredictions: false),
        isNull,
      );
      await ga.downloadAll(a);
      await gb.downloadAll(b);
      for (var l = 1; l < 3; l++) {
        expect(b[0].weights[l]!.data, a[0].weights[l]!.data);
      }
    });
    test('Adam over layers wider than the workgroup', () async {
      await check(
        sizes: [5, 37, 70, 6],
        members: 3,
        rows: 60,
        shared: false,
        optimiser: () => Adam(1e-3),
        workgroupSize: 32,
      );
    });
  });
}
