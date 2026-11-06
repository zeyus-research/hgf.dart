// Parameter recovery on pyhgf's "4-Parameter_recovery" notebook setup: 20
// simulated agents (binary 2-level HGF, sigmoid decisions with an inverse
// temperature), refitted by MAP with every agent in one batch, then the
// Laplace approximation for 95% intervals and the log evidence.
//
//   dart run example/parameter_recovery.dart
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';

void main() {
  final u = Float64List.fromList([
    for (final l in File('data/binary_input.txt').readAsLinesSync())
      if (l.trim().isNotEmpty) double.parse(l),
  ]);
  const n = 20;
  final omegas = [for (var i = 0; i < n; i++) -6.0 + 5.0 * i / (n - 1)];
  final betas = [for (var i = 0; i < n; i++) 0.5 + 5.5 * i / (n - 1)];

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
      Parameter.response(
        'inverseTemperature',
        prior: const Uniform(0.2, 20),
        initial: 1.0,
      ),
    ],
  );

  final agents = [for (var i = 0; i < n; i++) Subject(inputs: u, id: i)];
  final responses = model.simulate(agents, [
    for (var i = 0; i < n; i++) [omegas[i], betas[i]],
  ], math.Random(123));
  final subjects = [
    for (var i = 0; i < n; i++)
      Subject(inputs: u, responses: responses[i], id: i),
  ];

  for (final (name, opt) in [
    ('L-BFGS', const Lbfgs() as Optimizer),
    ('Nelder-Mead', const NelderMead()),
  ]) {
    final sw = Stopwatch()..start();
    final fit = fitMap(model, subjects, optimizer: opt);
    sw.stop();
    print(
      '$name: ${sw.elapsedMilliseconds} ms, ${fit.passes} passes, '
      '${fit.elements} filters, '
      '${fit.results.where((r) => r.converged).length}/$n converged',
    );
    final estO = [for (final r in fit.results) r.x[0]];
    final estB = [for (final r in fit.results) r.x[1]];
    print(
      '  r(omega2) = ${_corr(omegas, estO).toStringAsFixed(3)}, '
      'r(beta) = ${_corr(betas, estB).toStringAsFixed(3)}',
    );
    for (final r in fit.results.take(n)) {
      final i = r.subject.id! as int;
      print(
        '  ${i.toString().padLeft(2)}: ω ${omegas[i].toStringAsFixed(2)} → '
        '${r.x[0].toStringAsFixed(2)}, β ${betas[i].toStringAsFixed(2)} → '
        '${r.x[1].toStringAsFixed(2)}  ${r.optimization.iterations} it, '
        '${r.optimization.message}',
      );
    }
  }

  // Laplace approximation around the L-BFGS estimates: one more pass.
  final sw = Stopwatch()..start();
  final lap = laplace(model, fitMap(model, subjects));
  sw.stop();
  print(
    '\nLaplace (MAP + Hessian): ${sw.elapsedMilliseconds} ms, '
    'log evidence ${lap.logEvidence.toStringAsFixed(2)}',
  );
  var covered = 0;
  String fmt(double v) => v.toStringAsFixed(2);
  for (final r in lap.results) {
    final i = r.map.subject.id! as int;
    final (lo, hi) = r.interval();
    final truth = [omegas[i], betas[i]];
    for (var p = 0; p < 2; p++) {
      if (lo[p] <= truth[p] && truth[p] <= hi[p]) covered++;
    }
    print(
      '  ${i.toString().padLeft(2)}: ω ${fmt(truth[0])} in '
      '[${fmt(lo[0])}, ${fmt(hi[0])}], β ${fmt(truth[1])} in '
      '[${fmt(lo[1])}, ${fmt(hi[1])}]',
    );
  }
  print('  95% intervals cover $covered of ${2 * n} true values');
}

double _corr(List<double> a, List<double> b) {
  final n = a.length;
  final ma = a.reduce((x, y) => x + y) / n, mb = b.reduce((x, y) => x + y) / n;
  var sab = 0.0, saa = 0.0, sbb = 0.0;
  for (var i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) * (a[i] - ma);
    sbb += (b[i] - mb) * (b[i] - mb);
  }
  return sab / math.sqrt(saa * sbb);
}
