// A complete workflow on your own data, step by step: load a CSV, split it
// into training, validation and test sets, standardise the inputs, train
// with early stopping on the validation set (sample by sample, or in
// minibatches), evaluate on the test set, save the model and load it back
// to predict new samples.
//
// The data: the UCI wine dataset (example/data/wine.csv, 178 wines, 13
// chemical measurements, three cultivars as text labels). Swap in your own
// file, target column and network.
//
//   dart run example/workflow.dart [--minibatch] [--epochs 60]
import 'dart:convert';
import 'dart:io';

import 'package:hgf_deep/datasets.dart';
import 'package:hgf_deep/hgf_deep.dart';

void main(List<String> args) {
  final minibatch = args.contains('--minibatch');
  final i = args.indexOf('--epochs');
  final maxEpochs = i < 0 ? 60 : int.parse(args[i + 1]);

  // 1. Load. Every column but the target is a feature (or name them with
  // `features:`). Text labels become classes 0, 1, … in sorted order; rows
  // with a missing value are dropped.
  final file = File('example/data/wine.csv');
  if (!file.existsSync()) {
    stderr.writeln('Run from packages/hgf_deep (${file.path} not found).');
    exit(1);
  }
  final data = Dataset.fromCsv(
    file.readAsStringSync(),
    target: ['cultivar'],
    name: 'wine',
  );
  print(
    '${data.name}: ${data.length} rows, ${data.x.cols} features, '
    'classes ${data.classNames}',
  );

  // 2. Split: 60 % training, 20 % validation (for early stopping), 20 %
  // test (touched once, at the end).
  final (train: rest, test: testRaw) = split(data, testFraction: 0.2, seed: 1);
  final (train: trainRaw, test: validRaw) = split(
    rest,
    testFraction: 0.25,
    seed: 2,
  );

  // 3. Standardise every feature with the training set's mean and sd. The
  // wine measurements range from ~0.1 to ~1500: unscaled, the largest
  // would dominate the first layer.
  final scale = Standardiser.fit(trainRaw.x);
  final train = trainRaw.withX(scale.apply(trainRaw.x));
  final valid = validRaw.withX(scale.apply(validRaw.x));
  final test = testRaw.withX(scale.apply(testRaw.x));
  print(
    '${train.length} train / ${valid.length} validation / '
    '${test.length} test',
  );

  // 4. The network, built bottom (output) first: a 3-way categorical output,
  // one hidden layer, the 13 inputs on top.
  final net = DeepNetwork()
    ..addLayer(data.classes, kind: LayerKind.categorical)
    ..addLayer(16, coupling: CouplingFn.tanh, volatilityParent: false)
    ..addLayer(data.x.cols)
    ..weightInitialisation(const XavierInit(), seed: 1);

  // 5. Train, keeping the network that did best on the validation set and
  // stopping once it hasn't improved for `patience` epochs.
  final optimiser = Adam(minibatch ? 2e-2 : 1e-3);
  const patience = 8;
  var best = double.infinity;
  var bestEpoch = 0;
  DeepSnapshot? bestNet;
  print(
    minibatch
        ? 'Minibatches of 16 (batchUpdate):'
        : 'One sample at a time (fit):',
  );
  print('epoch  valid loss  valid acc');
  for (var epoch = 1; epoch <= maxEpochs; epoch++) {
    if (minibatch) {
      for (final b in minibatches(train, 16, seed: epoch)) {
        net.batchUpdate(b.x, b.y, optimiser: optimiser);
      }
    } else {
      final epochData = shuffled(train, seed: epoch);
      net.fit(epochData.x, epochData.y, optimiser: optimiser);
    }
    final p = net.predict(valid.x);
    final loss = crossEntropy(p, valid.y);
    if (epoch % 5 == 0 || epoch == 1) {
      print(
        '${'$epoch'.padLeft(5)}  ${loss.toStringAsFixed(4).padLeft(10)}  '
        '${accuracy(p, valid.labels!).toStringAsFixed(3).padLeft(9)}',
      );
    }
    if (loss < best) {
      best = loss;
      bestEpoch = epoch;
      bestNet = net.snapshot();
    } else if (epoch - bestEpoch >= patience) {
      print('stopped at epoch $epoch: no better since epoch $bestEpoch');
      break;
    }
  }
  net.restore(bestNet!);
  print(
    'kept the network of epoch $bestEpoch (valid loss '
    '${best.toStringAsFixed(4)})',
  );

  // 6. Evaluate once on the test set.
  final p = net.predict(test.x);
  print(
    '\ntest accuracy ${accuracy(p, test.labels!).toStringAsFixed(3)}, '
    'cross-entropy ${crossEntropy(p, test.y).toStringAsFixed(4)}',
  );
  print(
    formatConfusion(
      confusionMatrix(p, test.labels!, data.classes),
      names: data.classNames,
    ),
  );

  // 7. Save the network with what it needs to read new data: the input
  // scaling and the class names. Load it back and predict.
  final saved = File('${Directory.systemTemp.path}/wine_model.json')
    ..writeAsStringSync(
      jsonEncode({
        'network': net.toJson(),
        'scale': scale.toJson(),
        'classes': data.classNames,
        'features': data.featureNames,
      }),
    );
  print('saved to ${saved.path} (${saved.lengthSync()} bytes)');

  final model = jsonDecode(saved.readAsStringSync()) as Map<String, Object?>;
  final loaded = DeepNetwork.fromJson(model['network'] as Map<String, Object?>);
  final loadedScale = Standardiser.fromJson(
    model['scale'] as Map<String, Object?>,
  );
  final classes = (model['classes'] as List<Object?>).cast<String>();

  // A new wine (raw measurements, in the order of `features`).
  final wine = Matrix.fromRows([
    [13.2, 2.8, 2.4, 19.5, 105, 2.4, 2.3, 0.33, 1.6, 4.8, 0.95, 2.9, 900],
  ]);
  final probs = loaded.predict(loadedScale.apply(wine));
  final k = predictedClasses(probs).single;
  print(
    'new wine: ${classes[k]} '
    '(p = ${[for (final v in probs.data) v.toStringAsFixed(3)].join(', ')})',
  );
  if (loaded.predict(test.x).data.toString() != p.data.toString()) {
    throw StateError('The loaded network predicts differently');
  }
}
