/// Cost of the dp-state step: one `Network` and a `BatchNetwork`, per
/// element-trial. `dart compile exe benchmark/bin/dp_cost.dart` for AOT.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

Float64List mixture(int nTime, {int seed = 5}) {
  final rng = math.Random(seed);
  const centres = [-6.0, 0.5, 7.0];
  return Float64List.fromList([
    for (var t = 0; t < nTime; t++)
      centres[rng.nextInt(3)] +
          math.sqrt(-2 * math.log(1 - rng.nextDouble())) *
              math.cos(2 * math.pi * rng.nextDouble()),
  ]);
}

Network dp(int k, double alpha) => Network()
  ..addDpState(batchSize: k, alpha: alpha)
  ..addEfState(nNodes: k, valueChildren: [0], nus: 15.0);

void main(List<String> args) {
  final nTime = args.isEmpty ? 200 : int.parse(args[0]);
  final x = mixture(nTime);
  for (final k in [3, 6]) {
    final net = dp(k, 0.1);
    for (var rep = 0; rep < 2; rep++) {
      final sw = Stopwatch()..start();
      net.surprise(x, const DpSurprise());
      final us = sw.elapsedMicroseconds / nTime;
      if (rep == 1) print('Network k=$k: ${us.toStringAsFixed(1)} µs/trial');
    }
    for (final b in [8, 32]) {
      final batch = BatchNetwork(dp(k, 0.1), b)
        ..setInitial(
          0,
          NodeField.alpha,
          Float64List.fromList([for (var e = 0; e < b; e++) 0.01 + 0.05 * e]),
        );
      for (var rep = 0; rep < 2; rep++) {
        final sw = Stopwatch()..start();
        batch.surprise(x, const DpSurprise());
        final us = sw.elapsedMicroseconds / (nTime * b);
        if (rep == 1) {
          print('Batch k=$k b=$b: ${us.toStringAsFixed(1)} µs/element-trial');
        }
      }
    }
  }
}
