// A 3-level binary HGF on a simulated reversal-learning sequence: filtered
// as a whole series and trial by trial, scored by its surprise, and run
// forward in generative mode.
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

void main() {
  // Outcomes: P(u = 1) flips between 0.8 and 0.2 every 40 trials.
  final rng = math.Random(42);
  final outcomes = Float64List.fromList([
    for (var t = 0; t < 200; t++)
      rng.nextDouble() < ((t ~/ 40).isEven ? 0.8 : 0.2) ? 1.0 : 0.0,
  ]);

  final hgf = Network()
    ..addNodes(kind: NodeKind.binaryState)
    ..addNodes(valueChildren: [0], tonicVolatility: -3.0)
    ..addNodes(volatilityChildren: [1], tonicVolatility: -4.0, mean: 1.0);

  // Whole series.
  final traj = hgf.inputData(outcomes)!;
  final predicted = traj.series(0, NodeField.expectedMean);
  for (var t = 0; t < 200; t += 20) {
    print(
      't=$t  P(u=1)=${predicted[t].toStringAsFixed(3)}  '
      'μ3=${traj.at(2, NodeField.mean, t).toStringAsFixed(3)}',
    );
  }

  // Trial by trial (e.g. inside an experiment loop).
  hgf.reset();
  final obs = Float64List(1);
  for (final u in outcomes.take(5)) {
    hgf.predict(); // the prediction for *this* trial, before seeing u
    final prediction = hgf.get(0, NodeField.expectedMean);
    obs[0] = u;
    hgf.step(obs); // reuses the prediction, then updates on u
    print('predicted ${prediction.toStringAsFixed(3)}, observed $u');
  }

  // Surprise: how well does each tonic volatility explain the outcomes?
  for (final omega in [-6.0, -4.0, -2.0]) {
    hgf.setInitial(1, NodeField.tonicVolatility, omega);
    final nll = hgf.surprise(outcomes, const FirstLevelBinarySurprise());
    print('ω2 = $omega: surprise ${nll.toStringAsFixed(1)}');
  }
  hgf.setInitial(1, NodeField.tonicVolatility, -3.0);

  // Generative mode: what would this agent expect to see next?
  hgf.inputData(outcomes, record: false);
  final future = hgf.sample(10, random: math.Random(1), append: true)!;
  print('simulated: ${future.series(0, NodeField.mean)}');
}
