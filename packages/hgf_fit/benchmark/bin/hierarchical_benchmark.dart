// Hierarchical NUTS on pyhgf's multilevel notebook (3-Multilevel_HGF):
// a binary 2-level HGF per agent with sigmoid decisions, the agents' ω₂
// normal and inverse temperatures log-normal around group means with
// hyperpriors μ_ω ~ N(−5, 5), σ_ω ~ HalfNormal(10), μ_β ~ N(0, 2),
// σ_β ~ HalfNormal(2). Agents are simulated as in the notebook
// (ω₂ ~ N(−4, 1), log β ~ N(0.5, 0.5)).
//
//   dart compile exe benchmark/bin/hierarchical_benchmark.dart -o /tmp/hb
//   /tmp/hb [--agents 10] [--chains 4] [--centered] [--out problem.json]
//
// tool/parity/fit_hier_jax.py samples the same model with NumPyro or PyMC
// (the notebook's route) and compares the posteriors.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';

void main(List<String> args) {
  var agents = 10, chains = 4, warmup = 1000, draws = 1000, seed = 0;
  var centered = false, nuts = true;
  String? out;
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--agents':
        agents = int.parse(args[++i]);
      case '--chains':
        chains = int.parse(args[++i]);
      case '--warmup':
        warmup = int.parse(args[++i]);
      case '--draws':
        draws = int.parse(args[++i]);
      case '--seed':
        seed = int.parse(args[++i]);
      case '--centered':
        centered = true;
      case '--no-nuts':
        nuts = false;
      case '--out':
        out = args[++i];
    }
  }
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
      // The priors are replaced by the populations; they fix the scales
      // (identity for ω₂, log for β).
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
      'omega2': const Population(mean: Normal(-5, 5), sd: HalfNormal(10)),
      'inverseTemperature': const Population(
        mean: Normal(0, 2),
        sd: HalfNormal(2),
      ),
    },
    centered: centered,
  );

  final rng = math.Random(123);
  final truth = [
    for (var i = 0; i < agents; i++)
      [
        -4.0 + const Normal(0, 1).sample(rng),
        math.exp(const Normal(0.5, 0.5).sample(rng)),
      ],
  ];
  final responses = model.simulate(
    [for (var i = 0; i < agents; i++) Subject(inputs: u)],
    truth,
    rng,
  );
  final subjects = [
    for (var i = 0; i < agents; i++)
      Subject(inputs: u, responses: responses[i], id: i),
  ];

  // Empirical Bayes (Laplace-EM), timed over a few repeats.
  late EmpiricalBayesFit eb;
  final ebTimes = <double>[];
  for (var r = 0; r < 6; r++) {
    final sw = Stopwatch()..start();
    eb = fitEmpiricalBayes(hier, subjects);
    if (r > 0) ebTimes.add(sw.elapsedMicroseconds / 1000);
  }
  ebTimes.sort();
  print(
    'hgf_fit empirical Bayes: ${ebTimes[ebTimes.length ~/ 2].toStringAsFixed(1)}'
    ' ms (median of ${ebTimes.length}), ${eb.iterations} iterations, '
    '${eb.passes} passes, ${eb.elements} filters\n$eb',
  );
  if (!nuts) return;

  final sw = Stopwatch()..start();
  final fit = sampleHierarchical(
    hier,
    subjects,
    chains: chains,
    warmup: warmup,
    draws: draws,
    seed: seed,
  );
  final ms = sw.elapsedMicroseconds / 1000;
  final summary = fit.draws.summary();
  final ess = [for (final s in summary) s.essBulk]..sort();
  final minEss = ess.first, medianEss = ess[ess.length ~/ 2];
  final groupEss = [
    for (final s in summary)
      if (s.name.endsWith('_mu') || s.name.endsWith('_sigma')) s.essBulk,
  ].reduce(math.min);
  final maxRhat = summary.map((s) => s.rhat).reduce(math.max);
  print(
    'hgf_fit hierarchical NUTS (${centered ? 'centred' : 'non-centred'}), '
    '$agents agents, $chains chains x ($warmup + $draws): '
    '${(ms / 1000).toStringAsFixed(1)} s, ${fit.passes} passes, '
    '${fit.elements} filters, ${fit.draws.leapfrogSteps} leapfrog steps '
    'after warmup, ${fit.divergences} divergences, bulk ESS min '
    '${minEss.toStringAsFixed(0)} (${(minEss / ms * 1000).toStringAsFixed(1)}'
    '/s), median ${medianEss.toStringAsFixed(0)} '
    '(${(medianEss / ms * 1000).toStringAsFixed(1)}/s), group min '
    '${groupEss.toStringAsFixed(0)} '
    '(${(groupEss / ms * 1000).toStringAsFixed(1)}/s), max R̂ '
    '${maxRhat.toStringAsFixed(3)}',
  );
  print(fit);
  if (out != null) {
    File(out).writeAsStringSync(
      jsonEncode({
        'inputs': u.toList(),
        'responses': [for (final r in responses) r.toList()],
        'truth': truth,
        'dart': {
          'centered': centered,
          'chains': chains,
          'warmup': warmup,
          'draws': draws,
          'ms': ms,
          'passes': fit.passes,
          'filters': fit.elements,
          'divergences': fit.divergences,
          'leapfrog': fit.draws.leapfrogSteps,
          'names': [for (final s in summary) s.name],
          'mean': [for (final s in summary) s.mean],
          'sd': [for (final s in summary) s.sd],
          'mcse_mean': [for (final s in summary) s.mcseMean],
          'ess_bulk': [for (final s in summary) s.essBulk],
          'min_ess_bulk': minEss,
          'median_ess_bulk': medianEss,
          'group_min_ess_bulk': groupEss,
          'empirical_bayes': {
            'ms': ebTimes[ebTimes.length ~/ 2],
            'iterations': eb.iterations,
            for (final MapEntry(:key, :value) in eb.populations.entries)
              key: [value.mean, value.sd],
          },
        },
      }),
    );
    print('wrote $out');
  }
}
