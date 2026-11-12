// GpuFitter's gradient passes in forward mode (tangents) against reverse
// mode (adjoints, √T checkpoints), as multiples of a pass without
// gradients, by number of parameters (the models of hgf_fit's
// reverse_cpu.dart). From packages/hgf_gpu:
//
//   dart compile exe benchmark/bin/fit_reverse_gpu.dart -o .dart_tool/fit_reverse_gpu
//   .dart_tool/fit_reverse_gpu [--elements 2000,16384] [--trials 320]
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_gpu/hgf_gpu.dart';
import 'package:hgf_gpu/native.dart';

import '../../../hgf_fit/benchmark/src/gradient_cases.dart';

Future<double> best(int reps, Future<void> Function() f) async {
  await f();
  await f();
  var b = double.infinity;
  for (var r = 0; r < reps; r++) {
    final sw = Stopwatch()..start();
    await f();
    b = math.min(b, sw.elapsedMicroseconds / 1000);
  }
  return b;
}

Future<void> main(List<String> args) async {
  var sizes = [2000, 16384];
  var nT = 320, reps = 9;
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
  final fitters = {
    for (final m in GpuGradientMode.values.where(
      (m) => m != GpuGradientMode.auto,
    ))
      m: GpuFitter(backend, gradientMode: m),
  };
  for (final c in cases) {
    final rng = math.Random(3);
    final subj = subjects(c, 16, nT, rng);
    print('\n== ${c.name}');
    final all = c.parameters.length;
    for (final k in {2, 5, 10, 20, all}) {
      if (k > all) continue;
      final model = HgfModel(
        network: c.network(),
        response: c.response,
        parameters: c.parameters.sublist(0, k),
      );
      final carried = FitKernel(model).tapeCarried;
      for (final n in sizes) {
        final subjectOf = Int32List.fromList([
          for (var e = 0; e < n; e++) e % subj.length,
        ]);
        final y = Float64List(n * k);
        final y0 = model.toUnconstrained(model.start());
        for (var e = 0; e < n; e++) {
          for (var i = 0; i < k; i++) {
            y[e * k + i] = y0[i] + 0.05 * (rng.nextDouble() - 0.5);
          }
        }
        Pass pass(bool g) => Pass(model, subj, subjectOf, y, gradient: g);
        final f = fitters[GpuGradientMode.forward]!;
        final tPrimal = await best(reps, () => f.evaluatePass(pass(false)));
        final line = StringBuffer(
          '  k=${k.toString().padLeft(2)} (k·slots ${k * carried}), '
          'B=${n.toString().padLeft(5)}: value ${tPrimal.toStringAsFixed(2)} ms',
        );
        for (final MapEntry(key: mode, value: fitter) in fitters.entries) {
          final t = await best(reps, () => fitter.evaluatePass(pass(true)));
          line.write(
            '  ${mode.name} ${t.toStringAsFixed(2)} ms '
            '(${(t / tPrimal).toStringAsFixed(1)}×)',
          );
          if (t > 2000) break;
        }
        print(line);
      }
    }
  }
  for (final f in fitters.values) {
    f.dispose();
  }
}
