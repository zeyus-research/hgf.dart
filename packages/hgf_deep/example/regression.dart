// Regression with a deep HGF network: noisy y = sin(x) on [−π, π], learned
// by a 1 → 16 → 16 → 1 tanh network one sample at a time. Prints the test
// MSE and R² per epoch (the noise floor is MSE ≈ 0.01, the noise variance)
// and a few predictions against the noiseless curve.
//
//   dart run example/regression.dart [epochs]
import 'dart:math' as math;

import 'package:hgf_deep/datasets.dart';
import 'package:hgf_deep/hgf_deep.dart';

import 'src/train.dart';

void main(List<String> args) {
  final epochs = args.isEmpty ? 40 : int.parse(args[0]);
  final (:train, :test) = split(sine(800, noise: 0.1), seed: 1);
  final net = DeepNetwork()
    ..addLayer(1)
    ..addLayer(16, coupling: CouplingFn.tanh, volatilityParent: false)
    ..addLayer(16, coupling: CouplingFn.tanh, volatilityParent: false)
    ..addLayer(1)
    ..weightInitialisation(const XavierInit(), seed: 1);
  print('sine: ${train.length} train / ${test.length} test, 1 → 16 → 16 → 1');
  final scores = trainEpochs(
    net,
    train,
    test,
    optimiser: Adam(0.01),
    epochs: epochs,
    printEvery: 5,
  );
  if (scores.isEmpty) return;

  print('\n       x   sin(x)   predicted');
  final xs = [for (var i = 0; i <= 8; i++) -math.pi + i * math.pi / 4];
  final p = net.predict(Matrix.column(xs));
  for (var i = 0; i < xs.length; i++) {
    print(
      '${xs[i].toStringAsFixed(3).padLeft(8)} '
      '${math.sin(xs[i]).toStringAsFixed(3).padLeft(8)} '
      '${p.at(i, 0).toStringAsFixed(3).padLeft(11)}',
    );
  }
  print(
    '\ntest MSE ${scores.last.testLoss.toStringAsFixed(4)}, '
    'R² ${scores.last.testScore.toStringAsFixed(3)}',
  );
}
