// Group-level inference, as in pyhgf's multilevel notebook
// (3-Multilevel_HGF): ten agents see the same binary sequence; each has
// its own tonic volatility ω₂ and inverse temperature β, drawn from
// population distributions whose means and sds are estimated too.
//
// First empirical Bayes (Laplace-EM: point estimates of the populations,
// in milliseconds), then the full joint posterior by NUTS. With an
// argument, the draws are written as JSON for ArviZ:
//
//   dart run example/hierarchical.dart [draws.json]
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
  // The subject-level model. The priors of pooled parameters are replaced
  // by their populations; they still set each parameter's scale: identity
  // for ω₂ (a normal population), log for β (a log-normal one).
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
        prior: const Normal(-4, 1),
      ),
      Parameter.response(
        'inverseTemperature',
        prior: const LogNormal(0.5, 0.5),
      ),
    ],
  );
  final hier = HierarchicalModel(
    model,
    pooled: {
      // ω₂ ~ N(μ, σ), μ ~ N(−5, 5), σ ~ HalfNormal(10)
      'omega2': const Population(mean: Normal(-5, 5), sd: HalfNormal(10)),
      // log β ~ N(μ, σ), μ ~ N(0, 2), σ ~ HalfNormal(2)
      'inverseTemperature': const Population(
        mean: Normal(0, 2),
        sd: HalfNormal(2),
      ),
    },
    // Each agent sees 320 trials and is well identified: the centred
    // parametrisation samples better then. With little data per agent,
    // leave the default (non-centred).
    centered: true,
  );

  // Simulate the agents as the notebook does.
  final rng = math.Random(123);
  final truth = [
    for (var i = 0; i < 10; i++)
      [
        const Normal(-4, 1).sample(rng),
        math.exp(const Normal(0.5, 0.5).sample(rng)),
      ],
  ];
  final responses = model.simulate(
    [for (final _ in truth) Subject(inputs: u)],
    truth,
    rng,
  );
  final subjects = [
    for (var i = 0; i < truth.length; i++)
      Subject(inputs: u, responses: responses[i], id: 'agent$i'),
  ];

  var sw = Stopwatch()..start();
  final eb = fitEmpiricalBayes(hier, subjects);
  print('Empirical Bayes: ${sw.elapsedMilliseconds} ms\n$eb');

  sw = Stopwatch()..start();
  final fit = sampleHierarchical(hier, subjects, chains: 4);
  print(
    'NUTS: ${(sw.elapsedMilliseconds / 1000).toStringAsFixed(1)} s, '
    '${fit.passes} passes, ${fit.elements} filters, '
    '${fit.divergences} divergences\n',
  );
  print(fit);
  for (final name in ['omega2', 'inverseTemperature']) {
    final mu = fit.summary(name, group: 'mu');
    print(
      '$name group mean: ${mu.mean.toStringAsFixed(3)} '
      '[${mu.q5.toStringAsFixed(3)}, ${mu.q95.toStringAsFixed(3)}] '
      '(simulated from ${name == 'omega2' ? -4.0 : 0.5})',
    );
  }
  if (args.isNotEmpty) {
    File(args.first).writeAsStringSync(fit.toJson());
    print('wrote ${args.first}');
  }
}
