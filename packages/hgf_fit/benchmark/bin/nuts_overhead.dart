// The NUTS driver's own cost per chain and leapfrog step, without a
// filter: sampleFunction on a 2-d Gaussian with many chains.
//
//   dart compile exe benchmark/bin/nuts_overhead.dart -o .dart_tool/nuts_oh
//   .dart_tool/nuts_oh 8000 30
import 'dart:typed_data';

import 'package:hgf_fit/hgf_fit.dart';

void main(List<String> args) {
  final chains = args.isNotEmpty ? int.parse(args[0]) : 8000;
  final n = args.length > 1 ? int.parse(args[1]) : 30;
  var evals = 0;
  final sw = Stopwatch()..start();
  final fit = sampleFunction(
    (y) {
      evals++;
      return 0.5 * (y[0] * y[0] + 4 * y[1] * y[1]);
    },
    (y) => Float64List.fromList([y[0], 4 * y[1]]),
    Float64List(2),
    chains: chains,
    warmup: n,
    draws: n,
  );
  final ms = sw.elapsedMicroseconds / 1e3;
  print(
    '$chains chains, $n+$n: ${ms.toStringAsFixed(0)} ms, $evals evals, '
    '${(ms * 1e3 / evals).toStringAsFixed(2)} µs/eval '
    '(mean ${fit.mean(0).toStringAsFixed(3)})',
  );
}
