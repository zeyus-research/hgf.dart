// Shared training loop and printing for the hgf_deep examples.
import 'package:hgf_deep/datasets.dart';
import 'package:hgf_deep/hgf_deep.dart';

/// One epoch's scores.
typedef EpochScores = ({
  int epoch,
  double trainLoss,
  double trainScore,
  double testLoss,
  double testScore,
  int ms,
});

/// Train [net] for [epochs] passes over [train] (reshuffled every epoch,
/// one sample at a time with [DeepNetwork.fit] and a persistent
/// [optimiser]) and score it on [test] after each. Accuracy for
/// classification, R² for regression. Prints a row per epoch and stops
/// early, with a message, if the network diverges (NaN).
///
/// The train scores are prequential: each sample is predicted before the
/// network learns from it.
List<EpochScores> trainEpochs(
  DeepNetwork net,
  Dataset train,
  Dataset test, {
  required Optimiser optimiser,
  required int epochs,
  bool updatePrecisions = true,
  int seed = 0,
  int printEvery = 1,
}) {
  final cls = train.isClassification;
  final score = cls ? 'acc' : 'R²';
  print(
    '${'epoch'.padLeft(5)}  ${'train loss'.padLeft(10)}  '
    '${'train $score'.padLeft(9)}  ${'test loss'.padLeft(9)}  '
    '${'test $score'.padLeft(8)}  ${'ms'.padLeft(6)}',
  );
  final out = <EpochScores>[];
  for (var e = 1; e <= epochs; e++) {
    final sw = Stopwatch()..start();
    final batch = shuffled(train, seed: seed * 7919 + e);
    final pre = net.fit(
      batch.x,
      batch.y,
      optimiser: optimiser,
      updatePrecisions: updatePrecisions,
    );
    sw.stop();
    final post = net.predict(test.x);
    // predict() reads only the weights, so a hidden layer whose beliefs went
    // NaN would go unnoticed there; check the beliefs themselves.
    final badLayer = firstNonFiniteLayer(net);
    if (hasNonFinite(pre) || hasNonFinite(post) || badLayer != null) {
      print(
        'diverged at epoch $e: '
        '${badLayer == null ? 'the predictions' : 'the beliefs of layer $badLayer'} '
        'went NaN (a volatility runaway, see docs/upstream/README.md#p18). '
        'Try hidden layers without a volatility level '
        '(volatilityParent: false), a tonic volatility or a smaller learning '
        'rate.',
      );
      break;
    }
    final s = (
      epoch: e,
      trainLoss: cls ? crossEntropy(pre, batch.y) : mse(pre, batch.y),
      trainScore: cls ? accuracy(pre, batch.labels!) : r2(pre, batch.y),
      testLoss: cls ? crossEntropy(post, test.y) : mse(post, test.y),
      testScore: cls ? accuracy(post, test.labels!) : r2(post, test.y),
      ms: sw.elapsedMilliseconds,
    );
    out.add(s);
    if (e % printEvery == 0 || e == epochs) {
      print(
        '${'$e'.padLeft(5)}  ${s.trainLoss.toStringAsFixed(4).padLeft(10)}  '
        '${s.trainScore.toStringAsFixed(3).padLeft(9)}  '
        '${s.testLoss.toStringAsFixed(4).padLeft(9)}  '
        '${s.testScore.toStringAsFixed(3).padLeft(8)}  '
        '${'${s.ms}'.padLeft(6)}',
      );
    }
  }
  return out;
}

/// The first layer with a non-finite belief, or `null`.
int? firstNonFiniteLayer(DeepNetwork net) {
  for (var l = 0; l < net.nLayers; l++) {
    final s = net.state(l);
    for (final f in LayerField.values) {
      final v = s.field(f);
      if (v != null && v.any((x) => x.isNaN)) return l;
    }
  }
  return null;
}

/// An ASCII map of the predicted class over `[lo, hi]²` (2-D inputs), with
/// the test points overlaid as their true class digit; the background
/// symbols are the predicted class regions.
String decisionMap(
  DeepNetwork net,
  Dataset test, {
  double lo = -1.6,
  double hi = 1.6,
  int width = 60,
  int height = 24,
  Standardiser? scale,
}) {
  final grid = Matrix(width * height, 2);
  for (var r = 0; r < height; r++) {
    for (var c = 0; c < width; c++) {
      final k = r * width + c;
      grid.data[2 * k] = lo + (hi - lo) * (c + 0.5) / width;
      grid.data[2 * k + 1] = hi - (hi - lo) * (r + 0.5) / height;
    }
  }
  final classes = predictedClasses(
    net.predict(scale == null ? grid : scale.apply(grid)),
  );
  const background = ['·', ':', '-', '~', '=', '+'];
  const point = ['0', '1', '2', '3', '4', '5'];
  final cells = [
    for (var k = 0; k < width * height; k++)
      classes[k] < 0 ? '?' : background[classes[k] % background.length],
  ];
  final xs = test.x;
  for (var i = 0; i < test.length; i++) {
    final c = ((xs.at(i, 0) - lo) / (hi - lo) * width).floor();
    final r = ((hi - xs.at(i, 1)) / (hi - lo) * height).floor();
    if (c < 0 || c >= width || r < 0 || r >= height) continue;
    cells[r * width + c] = point[test.labels![i] % point.length];
  }
  final b = StringBuffer();
  for (var r = 0; r < height; r++) {
    b.writeln(cells.sublist(r * width, (r + 1) * width).join());
  }
  return b.toString();
}
