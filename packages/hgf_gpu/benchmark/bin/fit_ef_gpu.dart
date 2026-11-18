// ef-state and categorical models' likelihood passes (value, and value +
// gradient) on the GPU (GpuFitter) against the CPU (HgfModel.negLogPosterior on one
// isolate, f64 BatchNetwork). From packages/hgf_gpu:
//
//   dart compile exe benchmark/bin/fit_ef_gpu.dart -o .dart_tool/fit_ef_gpu
//   .dart_tool/fit_ef_gpu [--elements 256,2000,16384] [--trials 320]
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_gpu/hgf_gpu.dart';
import 'package:hgf_gpu/native.dart';

Future<double> best(int reps, Future<void> Function() f) async {
  await f();
  var b = double.infinity;
  for (var r = 0; r < reps; r++) {
    final sw = Stopwatch()..start();
    await f();
    b = math.min(b, sw.elapsedMicroseconds / 1000);
  }
  return b;
}

double _normal(math.Random rng) =>
    math.sqrt(-2 * math.log(1 - rng.nextDouble())) *
    math.cos(2 * math.pi * rng.nextDouble());

/// [n] trials of [d] correlated random walks plus noise, row-major.
Float64List _series(int n, int d, math.Random rng) {
  final out = Float64List(n * d);
  final x = List.filled(d, 0.0);
  for (var t = 0; t < n; t++) {
    for (var j = 0; j < d; j++) {
      x[j] += 0.05 * _normal(rng);
      out[t * d + j] =
          0.5 * x[j] + 0.3 * _normal(rng) + (j > 0 ? 0.2 * x[0] : 0);
    }
  }
  return out;
}

/// [n] one-hot rows over [k] categories; the likely category moves every
/// 40 trials.
Float64List _categories(int n, int k, math.Random rng) {
  final out = Float64List(n * k);
  for (var t = 0; t < n; t++) {
    final c = rng.nextDouble() < 0.7 ? (t ~/ 40) % k : rng.nextInt(k);
    out[t * k + c] = 1;
  }
  return out;
}

typedef _Case = ({
  String name,
  HgfModel model,
  Float64List Function(int n, math.Random rng) data,
});

/// A categorical node over [k] categories at 0 (binary nodes 1..k, their
/// value parents k+1..2k, the volatility parent 2k+1), scored by the
/// categorical surprise.
_Case _categorical(int k) => (
  name: 'categorical, $k categories (${2 * k + 2} nodes)',
  data: (n, rng) => _categories(n, k, rng),
  model: HgfModel(
    network: Network()..addCategoricalState(nCategories: k),
    response: const ObservationSurprise(CategoricalSurprise()),
    parameters: [
      Parameter(
        'omega2',
        targets: [
          for (var i = k + 1; i <= 2 * k; i++)
            NodeTarget(i, NodeField.tonicVolatility),
        ],
        prior: const Normal(-4, 2),
      ),
      Parameter.node(
        'omega3',
        2 * k + 1,
        NodeField.tonicVolatility,
        prior: const Normal(-6, 2),
      ),
    ],
  ),
);

List<_Case> _cases() {
  // Statistic s of an HGF-k ef-state node at 0: its node, then its value
  // parent, then k − 1 volatility levels.
  List<ParameterTarget> omegas(int nSs, int stride, int offset) => [
    for (var s = 0; s < nSs; s++)
      NodeTarget(1 + stride * s + offset, NodeField.tonicVolatility),
  ];
  return [
    (
      name: '1-d ef-state, HGF-2 (7 nodes)',
      data: (n, rng) => _series(n, 1, rng),
      model: HgfModel(
        network: Network()..addEfState(learning: EfLearning.hgf2),
        response: const ObservationSurprise(TotalGaussianSurprise()),
        parameters: [
          Parameter(
            'omega2',
            targets: omegas(2, 3, 1),
            prior: const Normal(-4, 2),
          ),
          Parameter(
            'omega3',
            targets: omegas(2, 3, 2),
            prior: const Normal(-6, 2),
          ),
        ],
      ),
    ),
    (
      name: '2-d ef-state, HGF-2 (16 nodes)',
      data: (n, rng) => _series(n, 2, rng),
      model: HgfModel(
        network: Network()..addEfState(dimension: 2, learning: EfLearning.hgf2),
        response: const ObservationSurprise(TotalGaussianSurprise()),
        parameters: [
          Parameter(
            'omega2',
            targets: omegas(5, 3, 1),
            prior: const Normal(-4, 2),
          ),
          Parameter(
            'omega3',
            targets: omegas(5, 3, 2),
            prior: const Normal(-6, 2),
          ),
        ],
      ),
    ),
    _categorical(3),
    _categorical(8),
  ];
}

Future<void> main(List<String> args) async {
  var sizes = [256, 2000, 16384];
  var nT = 320, reps = 5;
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--elements':
        sizes = [for (final s in args[++i].split(',')) int.parse(s)];
      case '--trials':
        nT = int.parse(args[++i]);
      case '--reps':
        reps = int.parse(args[++i]);
    }
  }
  final backend = WgpuBackend.create();
  if (backend == null) {
    stderr.writeln('No wgpu adapter');
    exitCode = 1;
    return;
  }
  print('GPU: ${backend.description}; $nT trials, best of $reps');
  final fitter = GpuFitter(backend);
  for (final c in _cases()) {
    final rng = math.Random(5);
    final subjects = [
      for (var s = 0; s < 16; s++) Subject(inputs: c.data(nT, rng)),
    ];
    final model = c.model;
    final k = model.dimension;
    print('\n== ${c.name}');
    for (final n in sizes) {
      final subjectOf = Int32List.fromList([
        for (var e = 0; e < n; e++) e % subjects.length,
      ]);
      final y = Float64List(n * k);
      final y0 = model.toUnconstrained(model.start());
      for (var e = 0; e < n; e++) {
        for (var i = 0; i < k; i++) {
          y[e * k + i] = y0[i] + 0.2 * (rng.nextDouble() - 0.5);
        }
      }
      final out = Float64List(n), grad = Float64List(n * k);
      final line = StringBuffer('  B=${n.toString().padLeft(5)}:');
      for (final g in [false, true]) {
        final cpu = await best(
          reps,
          () async => model.negLogPosterior(
            subjects,
            subjectOf,
            y,
            out,
            gradient: g ? grad : null,
          ),
        );
        final gpu = await best(
          reps,
          () => fitter.evaluatePass(
            Pass(model, subjects, subjectOf, y, gradient: g),
          ),
        );
        line.write(
          '  ${g ? 'gradient' : 'value'}: CPU ${cpu.toStringAsFixed(1)} ms, '
          'GPU ${gpu.toStringAsFixed(2)} ms '
          '(${(cpu / gpu).toStringAsFixed(1)}×)',
        );
      }
      print(line);
    }
  }
  if (fitter.cpuPasses != 0) {
    stderr.writeln('GPU fell back to the CPU: ${fitter.fallbackReason}');
    exitCode = 1;
  }
  fitter.dispose();
}
