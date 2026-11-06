/// Cost of forward-mode tangents (`BatchNetwork(tangents: k)`) per
/// element-trial, against a plain batch, by model and update mode
/// (diagnostic).
///
/// From `packages/hgf`:
///
///     dart compile exe benchmark/bin/tangent_cost.dart -o /tmp/tc && /tmp/tc
///     dart compile wasm -O4 benchmark/bin/tangent_cost.dart -o /tmp/tc.wasm
///     node ../../tool/web/run_wasm.mjs /tmp/tc.wasm
///
/// "fwd FD" is what a forward-difference gradient costs in plain filters
/// (`k + 1` per point), "ratio" the tangent pass over it: below 1, exact
/// gradients are cheaper.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

const int _elements = 256;
const int _trials = 1000;

void main() {
  final rng = math.Random(1);
  final binary = Float64List.fromList([
    for (var t = 0; t < _trials; t++)
      rng.nextDouble() < (t % 200 < 100 ? 0.8 : 0.2) ? 1.0 : 0.0,
  ]);
  var x = 0.0;
  final continuous = Float64List.fromList([
    for (var t = 0; t < _trials; t++) x += 0.1 * (rng.nextDouble() - 0.5),
  ]);

  Network binary2() => Network()
    ..addNodes(kind: NodeKind.binaryState)
    ..addNodes(valueChildren: [0], tonicVolatility: -2.0);
  Network continuous3(VolatilityUpdates mode) =>
      Network(volatilityUpdates: mode)
        ..addNodes(precision: 1e2)
        ..addNodes(valueChildren: [0], tonicVolatility: -4.0)
        ..addNodes(volatilityChildren: [1], tonicVolatility: -6.0);

  final cases =
      <(String, Network Function(), Float64List, ResponseFunction, int)>[
        (
          'binary 2-level',
          binary2,
          binary,
          const FirstLevelBinarySurprise(),
          1,
        ),
        for (final mode in VolatilityUpdates.values)
          (
            'continuous 3-level ${mode.pyhgfName}',
            () => continuous3(mode),
            continuous,
            const TotalGaussianSurprise(),
            2,
          ),
      ];
  print(
    'ns per element-trial, $_elements elements x $_trials trials; '
    'tangents seeded on ω of the hidden nodes',
  );
  print(
    '${'model'.padRight(30)}  plain   k=1   k=2   k=4  '
    '| fwd FD k=2  ratio',
  );
  for (final (name, make, data, response, nHidden) in cases) {
    final cols = <double>[];
    for (final k in [0, 1, 2, 4]) {
      final net = make();
      final batch = BatchNetwork(net, _elements, tangents: k);
      final omegas = Float64List.fromList([
        for (var e = 0; e < _elements; e++)
          net.getInitial(1, NodeField.tonicVolatility) - 0.002 * e,
      ]);
      batch.setInitial(1, NodeField.tonicVolatility, omegas);
      final ones = Float64List(_elements)..fillRange(0, _elements, 1.0);
      for (var d = 0; d < k; d++) {
        batch.setInitialTangent(
          1 + d % nHidden,
          NodeField.tonicVolatility,
          d,
          ones,
        );
      }
      final grad = k == 0 ? null : Float64List(_elements * k);
      void run() => batch.surprise(data, response, gradient: grad);
      for (var i = 0; i < 3; i++) {
        run();
      }
      var reps = 0;
      final sw = Stopwatch()..start();
      while (sw.elapsedMilliseconds < 600) {
        run();
        reps++;
      }
      cols.add(sw.elapsedMicroseconds * 1e3 / (reps * _elements * _trials));
    }
    String f(double v) => v.toStringAsFixed(0).padLeft(5);
    final fd = 3 * cols[0];
    print(
      '${name.padRight(30)} ${cols.map(f).join(' ')}  | ${f(fd)}      '
      '${(cols[2] / fd).toStringAsFixed(2)}',
    );
  }
}
