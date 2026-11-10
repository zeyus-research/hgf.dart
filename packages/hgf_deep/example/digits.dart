// Handwritten digit recognition (UCI optical digits: 1797 8×8 images, ten
// classes) with a deep HGF network: 64 standardised pixels → 32 tanh →
// 10-way categorical output, trained one sample at a time. Prints the
// per-epoch scores, the test confusion matrix and the accuracy per digit.
//
//   dart run example/digits.dart [epochs]
import 'dart:convert';
import 'dart:io';

import 'package:hgf_deep/datasets.dart';
import 'package:hgf_deep/hgf_deep.dart';

import 'src/train.dart';

/// The digits dataset from `example/data/digits.csv.gz`.
Dataset loadDigits() {
  final file = File('example/data/digits.csv.gz');
  if (!file.existsSync()) {
    throw StateError('Run from packages/hgf_deep (${file.path} not found).');
  }
  final lines = const LineSplitter()
      .convert(String.fromCharCodes(gzip.decode(file.readAsBytesSync())))
      .where((l) => l.trim().isNotEmpty)
      .toList();
  final x = Matrix(lines.length, 64);
  final labels = <int>[];
  for (var i = 0; i < lines.length; i++) {
    final cells = lines[i].split(',');
    for (var j = 0; j < 64; j++) {
      x.data[i * 64 + j] = double.parse(cells[j]);
    }
    labels.add(int.parse(cells[64]));
  }
  return Dataset(
    'digits',
    x,
    encodeLabels(labels, 10),
    labels: labels,
    classes: 10,
  );
}

void main(List<String> args) {
  final epochs = args.isEmpty ? 15 : int.parse(args[0]);
  final (:train, :test) = split(loadDigits(), testFraction: 0.25, seed: 1);
  // Standardise with the training set's statistics only.
  final scale = Standardiser.fit(train.x);
  final tr = train.withX(scale.apply(train.x));
  final te = test.withX(scale.apply(test.x));

  final net = DeepNetwork()
    ..addLayer(10, kind: LayerKind.categorical)
    ..addLayer(32, coupling: CouplingFn.tanh, volatilityParent: false)
    ..addLayer(64)
    ..weightInitialisation(const XavierInit(), seed: 1);
  print(
    'digits: ${tr.length} train / ${te.length} test, '
    '64 → 32 tanh → 10 categorical',
  );
  final scores = trainEpochs(
    net,
    tr,
    te,
    optimiser: Adam(0.005),
    epochs: epochs,
  );
  if (scores.isEmpty) return;

  final p = net.predict(te.x);
  final confusion = confusionMatrix(p, te.labels!, 10);
  print('\nconfusion (test, rows true, columns predicted):');
  print(formatConfusion(confusion));
  print('accuracy per digit:');
  for (var d = 0; d < 10; d++) {
    final row = confusion[d];
    final total = row.fold(0, (a, b) => a + b);
    print('  $d  ${(row[d] / total).toStringAsFixed(3)}  (${row[d]}/$total)');
  }
  print('\ntest accuracy ${scores.last.testScore.toStringAsFixed(3)}');
}
