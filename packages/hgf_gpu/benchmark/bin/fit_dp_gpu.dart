// A dp-state model's likelihood pass (DpSurprise, value only) on the GPU
// (GpuFitter: one cooperative workgroup per element) against the CPU
// (HgfModel.negLogPosterior on one isolate, f64 BatchNetwork). The CPU is
// timed on up to 16 elements and scaled linearly (each costs the same).
// From packages/hgf_gpu:
//
//   dart compile exe benchmark/bin/fit_dp_gpu.dart -o .dart_tool/fit_dp_gpu
//   .dart_tool/fit_dp_gpu [--elements 1,16,256,2048] [--trials 200]
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

/// Draws from three normal clusters (centres −6, 0.5, 7) in random order.
Float64List _mixture(int n, math.Random rng) => Float64List.fromList([
  for (var t = 0; t < n; t++)
    const [-6.0, 0.5, 7.0][rng.nextInt(3)] +
        math.sqrt(-2 * math.log(1 - rng.nextDouble())) *
            math.cos(2 * math.pi * rng.nextDouble()),
]);

HgfModel _model(int clusters) => HgfModel(
  network: Network()
    ..addDpState(batchSize: clusters, alpha: 1.0)
    ..addEfState(nNodes: clusters, valueChildren: [0], nus: 15.0),
  response: const ObservationSurprise(DpSurprise()),
  parameters: [
    Parameter.node('alpha', 0, NodeField.alpha, prior: const LogNormal(-2, 2)),
    Parameter.node(
      'sensoryPrecision',
      0,
      NodeField.sensoryPrecision,
      prior: const LogNormal(0, 1),
    ),
  ],
);

Future<void> main(List<String> args) async {
  var sizes = [1, 16, 256, 2048];
  var nT = 200, reps = 3;
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
  for (final clusters in [3, 10]) {
    final model = _model(clusters);
    final rng = math.Random(5);
    final subjects = [
      for (var s = 0; s < 16; s++) Subject(inputs: _mixture(nT, rng)),
    ];
    print('\n== dp-state, $clusters clusters');
    for (final n in sizes) {
      final subjectOf = Int32List.fromList([
        for (var e = 0; e < n; e++) e % subjects.length,
      ]);
      final y = Float64List(n * 2);
      for (var e = 0; e < n; e++) {
        y[2 * e] = math.log(0.01 + 2 * rng.nextDouble());
        y[2 * e + 1] = math.log(0.5 + rng.nextDouble());
      }
      final m = math.min(n, 16);
      final cpu =
          await best(1, () async {
            model.negLogPosterior(
              subjects,
              Int32List.sublistView(subjectOf, 0, m),
              Float64List.sublistView(y, 0, 2 * m),
              Float64List(m),
            );
          }) *
          n /
          m;
      final gpu = await best(
        reps,
        () => fitter.evaluatePass(Pass(model, subjects, subjectOf, y)),
      );
      print(
        '  B=${n.toString().padLeft(5)}: CPU ${cpu.toStringAsFixed(0)} ms'
        '${m < n ? ' (scaled from $m)' : ''}, GPU ${gpu.toStringAsFixed(1)} '
        'ms (${(cpu / gpu).toStringAsFixed(0)}×)',
      );
    }
  }
  if (fitter.cpuPasses != 0) {
    stderr.writeln('GPU fell back to the CPU: ${fitter.fallbackReason}');
    exitCode = 1;
  }
  fitter.dispose();
}
