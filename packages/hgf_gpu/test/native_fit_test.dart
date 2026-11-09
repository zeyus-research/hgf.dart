/// GPU fitting passes (values and exact gradients) against the f64 CPU
/// engine, on wgpu-native. Skipped if the library can't be loaded.
///
///     dart test --tags gpu test/native_fit_test.dart
@TestOn('vm')
@Tags(['gpu'])
library;

import 'dart:async';
import 'dart:math' as math;

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_gpu/hgf_gpu.dart';
import 'package:hgf_gpu/native.dart';
import 'package:test/test.dart';

import 'src/fit_check.dart';

WgpuBackend? _backend() {
  try {
    return WgpuBackend.create();
  } on ArgumentError {
    return null;
  }
}

/// pyhgf's parameter-recovery problem: binary 2-level, softmax decisions,
/// [n] simulated agents.
(HgfModel, List<Subject>, List<List<double>>) recovery(int n, int seed) {
  final model = HgfModel(
    network: Network()
      ..addNodes(kind: NodeKind.binaryState)
      ..addNodes(valueChildren: [0]),
    response: const BinarySoftmaxReadout(),
    parameters: [
      Parameter.node(
        'omega2',
        1,
        NodeField.tonicVolatility,
        prior: TruncatedNormal(-3, 5, lower: -8, upper: 2),
      ),
      Parameter.response('inverseTemperature', prior: const Uniform(0.2, 20)),
    ],
  );
  final rng = math.Random(seed);
  final u = binaryOutcomes(320, rng);
  final truth = [
    for (var s = 0; s < n; s++)
      [-5 + 3 * rng.nextDouble(), 1 + 4 * rng.nextDouble()],
  ];
  final r = model.simulate(
    [for (var s = 0; s < n; s++) Subject(inputs: u)],
    truth,
    rng,
  );
  return (
    model,
    [for (var s = 0; s < n; s++) Subject(inputs: u, responses: r[s])],
    truth,
  );
}

void main() {
  group('drivers on the GPU', () {
    final backend = _backend();
    final skip = backend == null ? 'wgpu-native not available' : null;

    test('fitMap and laplace match the CPU', skip: skip, () async {
      final (model, subjects, _) = recovery(20, 11);
      final gpu = GpuFitter(backend!);
      final cpuFit = fitMap(model, subjects);
      final gpuFit = await gpu.fitMap(model, subjects);
      expect(gpu.gpuPasses, greaterThan(5));
      for (var s = 0; s < subjects.length; s++) {
        final a = gpuFit.results[s], b = cpuFit.results[s];
        expect(a.converged, b.converged, reason: 'subject $s');
        for (var i = 0; i < model.dimension; i++) {
          expect(a.y[i], closeTo(b.y[i], 2e-3), reason: 'subject $s, $i');
        }
        // Reported in f64 at the GPU's optimum: as good as the CPU's.
        expect(a.negLogPosterior, closeTo(b.negLogPosterior, 1e-5));
      }
      // Laplace's Hessian pass is precise: it runs on the CPU.
      final before = gpu.gpuPasses;
      final lap = await gpu.laplace(model, gpuFit);
      expect(gpu.gpuPasses, before);
      final ref = laplace(model, gpuFit);
      expect(lap.logEvidence, ref.logEvidence);
    });

    test('sampleNuts: posteriors agree with the CPU', skip: skip, () async {
      final (model, subjects, _) = recovery(3, 5);
      final gpu = GpuFitter(backend!);
      final opts = (chains: 4, warmup: 300, draws: 300);
      final g = await gpu.sampleNuts(
        model,
        subjects,
        chains: opts.chains,
        warmup: opts.warmup,
        draws: opts.draws,
      );
      final c = sampleNuts(
        model,
        subjects,
        chains: opts.chains,
        warmup: opts.warmup,
        draws: opts.draws,
      );
      expect(g.divergences, lessThan(10));
      for (var s = 0; s < subjects.length; s++) {
        for (var i = 0; i < model.dimension; i++) {
          final a = g.results[s].mean(i), b = c.results[s].mean(i);
          final sd = c.results[s].sd(i);
          expect(a, closeTo(b, 0.25 * sd), reason: 'subject $s, $i');
        }
      }
    });

    test(
      'cancel stops a job; yieldInterval lets timers run',
      skip: skip,
      () async {
        final (model, subjects, _) = recovery(2, 3);
        final gpu = GpuFitter(backend!)
          ..yieldInterval = const Duration(milliseconds: 5);
        var ticks = 0;
        final timer = Timer.periodic(Duration.zero, (_) => ticks++);
        final job = gpu.sampleNuts(model, subjects, warmup: 2000, draws: 2000);
        await Future<void>.delayed(const Duration(milliseconds: 100));
        gpu.cancel();
        await expectLater(job, throwsA(isA<FitCancelled>()));
        timer.cancel();
        expect(ticks, greaterThan(3));
        // Still usable.
        final map = await gpu.fitMap(model, subjects);
        expect(map.results, hasLength(2));
        gpu.dispose();
      },
    );
  });

  test(
    'WgpuIsolateFitter: the same results as a GpuFitter, off-isolate',
    () async {
      final WgpuIsolateFitter iso;
      try {
        iso = await WgpuIsolateFitter.spawn();
      } on StateError catch (e) {
        markTestSkipped(e.message);
        return;
      }
      final local = GpuFitter(WgpuBackend.create()!);
      final (model, subjects, _) = recovery(6, 5);
      final a = await iso.fitMap(
        model,
        subjects,
        restarts: 1,
        random: math.Random(2),
      );
      final b = await local.fitMap(
        model,
        subjects,
        restarts: 1,
        random: math.Random(2),
      );
      for (var s = 0; s < subjects.length; s++) {
        expect(a.results[s].y, b.results[s].y, reason: 'subject $s');
      }
      expect(iso.gpuPasses, greaterThan(0));
      expect(iso.cpuPasses, greaterThan(0)); // the precise report pass
      final na = await iso.sampleNuts(model, subjects, warmup: 60, draws: 40);
      final nb = await local.sampleNuts(model, subjects, warmup: 60, draws: 40);
      expect(na.results[3].parameter(0), nb.results[3].parameter(0));
      // Cancel, then still usable; a new model replaces the old one.
      var ticks = 0;
      final timer = Timer.periodic(Duration.zero, (_) => ticks++);
      final job = iso.sampleNuts(model, subjects, warmup: 4000, draws: 4000);
      await Future<void>.delayed(const Duration(milliseconds: 100));
      iso.cancel();
      await expectLater(job, throwsA(isA<FitCancelled>()));
      timer.cancel();
      expect(ticks, greaterThan(10), reason: 'the calling isolate stays free');
      final (model2, subjects2, _) = recovery(3, 9);
      final c = await iso.fitMap(model2, subjects2);
      final d = await local.fitMap(model2, subjects2);
      expect(c.results[2].y, d.results[2].y);
      await iso.close();
      local.dispose();
    },
  );

  test('fit passes: f32 GPU values and gradients match f64 CPU', () async {
    final WgpuBackend? backend;
    try {
      backend = WgpuBackend.create();
    } on ArgumentError catch (e) {
      markTestSkipped('wgpu-native not available: ${e.message}');
      return;
    }
    if (backend == null) {
      markTestSkipped('No wgpu adapter');
      return;
    }
    final fitter = GpuFitter(backend);
    final failures = await checkFitCases(fitter, log: print);
    expect(fitter.cpuPasses, 0, reason: fitter.fallbackReason);
    expect(failures, isEmpty, reason: failures.take(20).join('\n'));
  });
}
