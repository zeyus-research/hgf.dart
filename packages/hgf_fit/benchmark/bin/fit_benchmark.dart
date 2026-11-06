// MAP fitting cost on pyhgf's parameter-recovery setup (notebook 4): a
// binary 2-level HGF with sigmoid decisions, ω₂ and β free per agent.
//
//   dart compile exe benchmark/bin/fit_benchmark.dart -o /tmp/fb && /tmp/fb
//   /tmp/fb --agents 200 --out problem.json   # also write the problem and
//                                             # estimates for the JAX script
//   /tmp/fb --nuts --chains 4 --out problem.json   # also NUTS (sampleNuts)
//
// tool/parity/fit_map_jax.py fits the same problem with JAX gradients and
// SciPy's L-BFGS-B for comparison; tool/parity/fit_nuts_jax.py samples it
// with PyMC or NumPyro.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';

void main(List<String> args) {
  var agents = 20, repeat = 7, chains = 4;
  var nuts = false;
  String? out;
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--agents':
        agents = int.parse(args[++i]);
      case '--repeat':
        repeat = int.parse(args[++i]);
      case '--out':
        out = args[++i];
      case '--nuts':
        nuts = true;
      case '--chains':
        chains = int.parse(args[++i]);
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
  final n = agents;
  final omegas = [for (var i = 0; i < n; i++) -6.0 + 5.0 * i / (n - 1)];
  final betas = [for (var i = 0; i < n; i++) 0.5 + 5.5 * i / (n - 1)];
  final responses = model.simulate(
    [for (var i = 0; i < n; i++) Subject(inputs: u)],
    [
      for (var i = 0; i < n; i++) [omegas[i], betas[i]],
    ],
    math.Random(123),
  );
  final subjects = [
    for (var i = 0; i < n; i++)
      Subject(inputs: u, responses: responses[i], id: i),
  ];

  final report = <String, Object>{};
  for (final (name, opt) in [
    ('lbfgs', const Lbfgs() as Optimizer),
    ('nelder_mead', const NelderMead()),
  ]) {
    final times = <double>[];
    late MapFit fit;
    for (var r = 0; r < repeat + 1; r++) {
      final sw = Stopwatch()..start();
      fit = fitMap(model, subjects, optimizer: opt);
      sw.stop();
      if (r > 0) times.add(sw.elapsedMicroseconds / 1000); // first = warm-up
    }
    times.sort();
    final median = times[times.length ~/ 2];
    final converged = fit.results.where((r) => r.converged).length;
    final filtersPerMs = fit.elements / median;
    print(
      '${name.padRight(12)} $n agents: ${median.toStringAsFixed(1)} ms '
      '(median of $repeat), ${fit.passes} passes, ${fit.elements} filters '
      '(${(fit.elements * u.length / median / 1000).toStringAsFixed(1)} '
      'M element-trials/s), $converged/$n converged',
    );
    report[name] = {
      'ms': median,
      'passes': fit.passes,
      'filters': fit.elements,
      'filters_per_ms': filtersPerMs,
      'converged': converged,
      'estimates': [for (final r in fit.results) r.x.toList()],
      'neg_log_posterior': [for (final r in fit.results) r.negLogPosterior],
    };
  }
  if (nuts) {
    final sw = Stopwatch()..start();
    final fit = sampleNuts(model, subjects, chains: chains);
    final ms = sw.elapsedMicroseconds / 1000;
    final summaries = [for (final r in fit.results) r.summary()];
    final ess = [
      for (final s in summaries)
        for (final p in s) p.essBulk,
    ];
    final rhats = [
      for (final s in summaries)
        for (final p in s) p.rhat,
    ];
    final minEss = ess.reduce(math.min);
    final medianEss = (ess.toList()..sort())[ess.length ~/ 2];
    final leapfrog = fit.results.fold(0, (s, r) => s + r.leapfrogSteps);
    print(
      'nuts         $n agents x $chains chains x (1000 + 1000): '
      '${(ms / 1000).toStringAsFixed(1)} s, ${fit.passes} passes, '
      '${fit.elements} filters, $leapfrog leapfrog steps after warmup, '
      '${fit.divergences} divergences, bulk ESS min '
      '${minEss.toStringAsFixed(0)} '
      '(${(minEss / ms * 1000).toStringAsFixed(1)}/s), median '
      '${medianEss.toStringAsFixed(0)} '
      '(${(medianEss / ms * 1000).toStringAsFixed(1)}/s), '
      'max R̂ ${rhats.reduce(math.max).toStringAsFixed(3)}',
    );
    report['nuts'] = {
      'ms': ms,
      'passes': fit.passes,
      'filters': fit.elements,
      'divergences': fit.divergences,
      'min_ess_bulk': minEss,
      'median_ess_bulk': medianEss,
      'mean': [
        for (final s in summaries) [for (final p in s) p.mean],
      ],
      'sd': [
        for (final s in summaries) [for (final p in s) p.sd],
      ],
      'mcse_mean': [
        for (final s in summaries) [for (final p in s) p.mcseMean],
      ],
    };
  }
  if (out != null) {
    File(out).writeAsStringSync(
      jsonEncode({
        'inputs': u.toList(),
        'responses': [for (final r in responses) r.toList()],
        'omegas': omegas,
        'betas': betas,
        'dart': report,
      }),
    );
    print('wrote $out');
  }
}
