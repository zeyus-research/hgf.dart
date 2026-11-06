// Full posteriors by NUTS for a few simulated agents (binary 2-level HGF,
// sigmoid decisions; pyhgf's parameter-recovery setup), every chain of
// every agent advanced in lock step, one batched pass per leapfrog step.
// Prints ArviZ-style summaries and, with an argument, writes the draws as
// JSON for ArviZ:
//
//   dart run example/posterior_sampling.dart [draws.json]
//
//   # Python (ArviZ 1.x)
//   import json, numpy as np, arviz as az
//   d = json.load(open("draws.json"))
//   groups = {g: {k: np.asarray(v) for k, v in d[g].items()}
//             for g in ("posterior", "sample_stats")}
//   idata = az.from_dict(groups, dims=d["dims"], coords=d["coords"])
//   print(az.summary(idata))
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';

void main(List<String> args) {
  final u = Float64List.fromList([
    for (final l in File('data/binary_input.txt').readAsLinesSync())
      if (l.trim().isNotEmpty) double.parse(l),
  ]);
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

  // Three agents, from weakly to well identified.
  const truth = [
    [-6.0, 0.5],
    [-3.5, 3.0],
    [-2.0, 5.0],
  ];
  final responses = model.simulate(
    [for (final _ in truth) Subject(inputs: u)],
    truth,
    math.Random(123),
  );
  final subjects = [
    for (var i = 0; i < truth.length; i++)
      Subject(inputs: u, responses: responses[i], id: 'agent$i'),
  ];

  // Start the chains near the MAP estimates (optional).
  final map = fitMap(model, subjects);
  final sw = Stopwatch()..start();
  final fit = sampleNuts(
    model,
    subjects,
    chains: 4,
    warmup: 1000,
    draws: 1000,
    start: [for (final r in map.results) r.x],
  );
  print(
    'NUTS: ${sw.elapsedMilliseconds} ms, ${fit.passes} passes, '
    '${fit.elements} filters, ${fit.divergences} divergences\n',
  );
  for (var i = 0; i < truth.length; i++) {
    final r = fit.results[i];
    print(
      'truth: omega2 = ${truth[i][0]}, inverseTemperature = ${truth[i][1]}',
    );
    print(r);
    print(
      '  converged: ${r.converged()}, E-BFMI '
      '${r.eBfmi.map((e) => e.toStringAsFixed(2)).join(' ')}\n',
    );
  }
  if (args.isNotEmpty) {
    File(args.first).writeAsStringSync(fit.toJson());
    print('wrote ${args.first}');
  }
}
