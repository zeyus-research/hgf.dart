/// GPU fitting passes (values and exact gradients) against the f64 CPU
/// engine, on wgpu-native. Skipped if the library can't be loaded.
///
///     dart test --tags gpu test/native_fit_test.dart
@TestOn('vm')
@Tags(['gpu'])
library;

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
  });

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
