// The Iowa Gambling Task: simulated agents play 100 trials (each learns
// the four decks' payoffs with its own two-level HGF per deck, sees only
// the chosen deck's payoff, and chooses by a softmax over the predicted
// payoffs). All of them are refitted by MAP in one batched job; then one
// "recorded" participant, given as choices and payoffs in dollars, is
// fitted with a 95% Laplace interval.
//
//   dart run example/iowa_gambling_task.dart
import 'dart:math' as math;

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';

void main() {
  final model = HgfModel(
    network: IowaGamblingTask.network(),
    response: IowaGamblingTask.readout(),
    parameters: [
      // One volatility for the four decks.
      Parameter(
        'omega',
        targets: [
          for (final n in IowaGamblingTask.valueNodes)
            NodeTarget(n, NodeField.tonicVolatility),
        ],
        prior: const Normal(-2, 1.5),
      ),
      Parameter.response('inverseTemperature', prior: const LogNormal(0.5, 1)),
    ],
  );

  // Parameter recovery: 200 agents with parameters drawn from the priors.
  final rng = math.Random(1);
  const nAgents = 200, nTrials = 100;
  final truth = [
    for (var a = 0; a < nAgents; a++)
      [for (final p in model.parameters) p.prior.sample(rng)],
  ];
  final agents = model.simulateChoices(
    [
      for (var a = 0; a < nAgents; a++)
        Subject(inputs: IowaGamblingTask.outcomes(nTrials, rng), id: a),
    ],
    truth,
    rng,
    reveals: IowaGamblingTask.reveals,
  );
  final sw = Stopwatch()..start();
  final fit = fitMap(model, agents);
  print(
    'MAP of $nAgents agents x $nTrials trials: ${sw.elapsedMilliseconds} ms',
  );
  for (var i = 0; i < model.dimension; i++) {
    final log = i == 1; // β is positive: compare on the log scale
    double f(double v) => log ? math.log(v) : v;
    final r = _corr(
      [for (final t in truth) f(t[i])],
      [for (final e in fit.results) f(e.x[i])],
    );
    print('  ${model.names[i]}: r = ${r.toStringAsFixed(2)}');
  }

  // A recorded participant: deck choices (0-3 = A-D) and the net payoff of
  // each card in dollars, e.g. from a CSV.
  final one = agents.first;
  final choices = one.responses!.toList();
  final payoffs = [
    for (var t = 0; t < nTrials; t++)
      100 * one.inputs[t * 4 + choices[t].toInt()],
  ];
  final (:inputs, :observed) = IowaGamblingTask.data(choices, payoffs);
  final subject = Subject(
    inputs: inputs,
    observed: observed,
    responses: one.responses,
  );
  final lap = laplace(model, fitMap(model, [subject]));
  final r = lap.results.single;
  final (lower, upper) = r.interval();
  print(
    'Participant (truth ω = ${truth[0][0].toStringAsFixed(2)}, '
    'β = ${truth[0][1].toStringAsFixed(2)}):',
  );
  for (var i = 0; i < model.dimension; i++) {
    print(
      '  ${model.names[i]} = ${r.map.x[i].toStringAsFixed(2)} '
      '(95%: ${lower[i].toStringAsFixed(2)} to ${upper[i].toStringAsFixed(2)})',
    );
  }
}

double _corr(List<double> a, List<double> b) {
  final n = a.length;
  final ma = a.reduce((x, y) => x + y) / n;
  final mb = b.reduce((x, y) => x + y) / n;
  var sab = 0.0, saa = 0.0, sbb = 0.0;
  for (var i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) * (a[i] - ma);
    sbb += (b[i] - mb) * (b[i] - mb);
  }
  return sab / math.sqrt(saa * sbb);
}
