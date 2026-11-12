// The cost of an exact gradient on the CPU: forward-mode tangents through
// BatchNetwork against reverse mode (IrReverse: the trial's IR and its
// adjoints, interpreted over chunks of elements), as multiples of the
// likelihood alone, by number of parameters. AOT:
//
//   dart compile exe benchmark/bin/reverse_cpu.dart -o .dart_tool/reverse_cpu
//   .dart_tool/reverse_cpu [--elements 256] [--trials 320] [--chunk 64]
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_fit/ir.dart';

import '../src/gradient_cases.dart';

double best(int reps, void Function() f) {
  f(); // warm-up
  var b = double.infinity;
  for (var r = 0; r < reps; r++) {
    final sw = Stopwatch()..start();
    f();
    b = math.min(b, sw.elapsedMicroseconds / 1000);
  }
  return b;
}

void main(List<String> args) {
  var elements = 256, nT = 320, chunk = 64, reps = 5;
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--elements':
        elements = int.parse(args[++i]);
      case '--trials':
        nT = int.parse(args[++i]);
      case '--chunk':
        chunk = int.parse(args[++i]);
      case '--reps':
        reps = int.parse(args[++i]);
    }
  }
  print('$elements elements × $nT trials, best of $reps; ns per element-trial');
  for (final c in cases) {
    final rng = math.Random(3);
    final subj = subjects(c, 16, nT, rng);
    final ir = IrReverse(c.network(), irResponseOf(c.response), chunk: chunk);
    print(
      '\n== ${c.name}: IR ${ir.instructions} instructions, ${ir.nRegs} '
      'registers, ${ir.carriedSlots.length} carried slots',
    );
    final all = c.parameters.length;
    for (final k in {2, 5, 10, 20, all}) {
      if (k > all) continue;
      final params = c.parameters.sublist(0, k);
      HgfModel model(GradientMode m) => HgfModel(
        network: c.network(),
        response: c.response,
        parameters: params,
        gradientMode: m,
      );
      final fwd = model(GradientMode.forward);
      final rev = model(GradientMode.reverse);
      final subjectOf = Int32List.fromList([
        for (var e = 0; e < elements; e++) e % subj.length,
      ]);
      final y = Float64List(elements * k);
      final y0 = fwd.toUnconstrained(fwd.start());
      for (var e = 0; e < elements; e++) {
        for (var i = 0; i < k; i++) {
          y[e * k + i] = y0[i] + 0.05 * (rng.nextDouble() - 0.5);
        }
      }
      final out = Float64List(elements), g = Float64List(elements * k);
      final g2 = Float64List(elements * k);
      final tPrimal = best(
        reps,
        () => fwd.negLogPosterior(subj, subjectOf, y, out),
      );
      final tFwd = best(
        reps,
        () => fwd.negLogPosterior(subj, subjectOf, y, out, gradient: g),
      );
      final tRev = best(
        reps,
        () => rev.negLogPosterior(subj, subjectOf, y, out, gradient: g2),
      );
      var diff = 0.0;
      for (var e = 0; e < elements; e++) {
        var scale = 1.0;
        for (var i = 0; i < k; i++) {
          scale = math.max(scale, g[e * k + i].abs());
        }
        for (var i = 0; i < k; i++) {
          diff = math.max(diff, (g[e * k + i] - g2[e * k + i]).abs() / scale);
        }
      }
      double ns(double ms) => ms * 1e6 / (elements * nT);
      print(
        '  k=${k.toString().padLeft(2)}: likelihood ${ns(tPrimal).toStringAsFixed(0).padLeft(5)}'
        '  forward ${ns(tFwd).toStringAsFixed(0).padLeft(6)} (${(tFwd / tPrimal).toStringAsFixed(1)}×)'
        '  reverse ${ns(tRev).toStringAsFixed(0).padLeft(6)} (${(tRev / tPrimal).toStringAsFixed(1)}×)'
        '  max diff ${diff.toStringAsExponential(1)}',
      );
    }
  }
}
