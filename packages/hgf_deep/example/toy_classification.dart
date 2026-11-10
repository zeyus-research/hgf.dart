// Classic 2-D toy classification problems learned by a deep HGF network:
// XOR, two moons, concentric circles (two classes, a binary output node)
// and three-arm spirals (a categorical output layer). Each epoch shuffles
// the training set and learns one sample at a time; the table shows the
// prequential training scores (each sample predicted before it is learned)
// and the held-out test scores. The map at the end shows the learned
// decision regions (background) under the test points (digits = true
// class).
//
//   dart run example/toy_classification.dart [xor|moons|circles|spirals|all] [epochs]
import 'package:hgf_deep/datasets.dart';
import 'package:hgf_deep/hgf_deep.dart';

import 'src/train.dart';

/// A task and the network and settings that learn it.
typedef Task = ({Dataset data, List<int> hidden, double lr, int epochs});

Task _task(String name) => switch (name) {
  'xor' => (data: xor(400, noise: 0.25), hidden: [8], lr: 0.01, epochs: 20),
  'moons' => (
    data: moons(600, noise: 0.12),
    hidden: [16, 16],
    lr: 0.01,
    epochs: 30,
  ),
  'circles' => (
    data: circles(600, noise: 0.08),
    hidden: [16, 16],
    lr: 0.01,
    epochs: 30,
  ),
  'spirals' => (
    data: spirals(900, noise: 0.1),
    hidden: [32, 32],
    lr: 0.01,
    epochs: 60,
  ),
  _ => throw ArgumentError('Unknown task "$name".'),
};

void main(List<String> args) {
  final which = args.isEmpty ? 'all' : args[0];
  final names = which == 'all'
      ? ['xor', 'moons', 'circles', 'spirals']
      : [which];
  final summary = <String>[];
  for (final name in names) {
    final task = _task(name);
    final epochs = args.length > 1 ? int.parse(args[1]) : task.epochs;
    final (:train, :test) = split(task.data, testFraction: 0.25, seed: 1);
    final k = task.data.classes;
    final net = DeepNetwork()
      ..addLayer(
        k == 2 ? 1 : k,
        kind: k == 2 ? LayerKind.binary : LayerKind.categorical,
      );
    // Hidden layers without a volatility level learn every task reliably;
    // see volatility_effect.dart for what the level does.
    for (final h in task.hidden.reversed) {
      net.addLayer(h, coupling: CouplingFn.tanh, volatilityParent: false);
    }
    net
      ..addLayer(2)
      ..weightInitialisation(const XavierInit(), seed: 1);

    print(
      '\n== $name: ${train.length} train / ${test.length} test, '
      '2 → ${task.hidden.join(' → ')} → ${k == 2 ? '1 binary' : '$k categorical'}',
    );
    final scores = trainEpochs(
      net,
      train,
      test,
      optimiser: Adam(task.lr),
      epochs: epochs,
      printEvery: epochs > 20 ? 5 : 1,
    );
    if (scores.isEmpty) continue;
    final p = net.predict(test.x);
    print('\nconfusion (test):');
    print(formatConfusion(confusionMatrix(p, test.labels!, k)));
    print(decisionMap(net, test));
    summary.add(
      '${name.padRight(8)} test accuracy ${scores.last.testScore.toStringAsFixed(3)}'
      '  (${scores.length} epochs)',
    );
  }
  print('\n== summary');
  summary.forEach(print);
}
