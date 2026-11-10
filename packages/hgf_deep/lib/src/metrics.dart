import 'dart:math' as math;

import 'matrix.dart';

/// The predicted class of each row: `p ≥ 0.5` for a single (binary) column,
/// the largest column otherwise. A row with a NaN predicts −1 (never
/// correct).
List<int> predictedClasses(Matrix p) => [
  for (var i = 0; i < p.rows; i++) _classOf(p, i),
];

int _classOf(Matrix p, int i) {
  if (p.cols == 1) {
    final v = p.at(i, 0);
    return v.isNaN ? -1 : (v >= 0.5 ? 1 : 0);
  }
  var best = -1;
  var max = double.negativeInfinity;
  for (var j = 0; j < p.cols; j++) {
    final v = p.at(i, j);
    if (v.isNaN) return -1;
    if (v > max) {
      max = v;
      best = j;
    }
  }
  return best;
}

/// Fraction of rows of [p] whose predicted class is [labels]'.
double accuracy(Matrix p, List<int> labels) {
  final c = predictedClasses(p);
  var hit = 0;
  for (var i = 0; i < c.length; i++) {
    if (c[i] == labels[i]) hit++;
  }
  return hit / c.length;
}

/// `counts[true][predicted]`; rows with a NaN prediction are left out.
List<List<int>> confusionMatrix(Matrix p, List<int> labels, int classes) {
  final m = [for (var i = 0; i < classes; i++) List.filled(classes, 0)];
  final c = predictedClasses(p);
  for (var i = 0; i < c.length; i++) {
    if (c[i] >= 0) m[labels[i]][c[i]]++;
  }
  return m;
}

/// A confusion matrix as an aligned text table (rows true, columns
/// predicted).
String formatConfusion(List<List<int>> m, {List<String>? names}) {
  final k = m.length;
  final label = names ?? [for (var i = 0; i < k; i++) '$i'];
  var w = 4;
  for (final row in m) {
    for (final v in row) {
      w = math.max(w, '$v'.length + 1);
    }
  }
  for (final n in label) {
    w = math.max(w, n.length + 1);
  }
  final b = StringBuffer()
    ..write('true\\pred'.padRight(10))
    ..writeAll([for (final n in label) n.padLeft(w)])
    ..writeln();
  for (var i = 0; i < k; i++) {
    b
      ..write(label[i].padRight(10))
      ..writeAll([for (final v in m[i]) '$v'.padLeft(w)])
      ..writeln();
  }
  return b.toString();
}

/// Mean squared error over every entry (NaN if any prediction is).
double mse(Matrix p, Matrix y) {
  _same(p, y);
  var s = 0.0;
  for (var k = 0; k < p.data.length; k++) {
    final d = p.data[k] - y.data[k];
    s += d * d;
  }
  return s / p.data.length;
}

/// Coefficient of determination, pooled over columns.
double r2(Matrix p, Matrix y) {
  _same(p, y);
  var mean = 0.0;
  for (final v in y.data) {
    mean += v;
  }
  mean /= y.data.length;
  var ssTot = 0.0;
  for (final v in y.data) {
    ssTot += (v - mean) * (v - mean);
  }
  return 1 - mse(p, y) * y.data.length / ssTot;
}

/// Mean negative log-likelihood of [y] under the predicted probabilities
/// [p]: Bernoulli per column for one column, categorical over the columns
/// otherwise. Probabilities are clipped to `[eps, 1 − eps]`.
double crossEntropy(Matrix p, Matrix y, {double eps = 1e-12}) {
  _same(p, y);
  double lg(double v) => math.log(math.min(math.max(v, eps), 1 - eps));
  var s = 0.0;
  if (p.cols == 1) {
    for (var i = 0; i < p.rows; i++) {
      final v = p.data[i];
      if (v.isNaN) return double.nan;
      s -= y.data[i] * lg(v) + (1 - y.data[i]) * lg(1 - v);
    }
  } else {
    for (var k = 0; k < p.data.length; k++) {
      final v = p.data[k];
      if (v.isNaN) return double.nan;
      if (y.data[k] != 0) s -= y.data[k] * lg(v);
    }
  }
  return s / p.rows;
}

/// Whether any entry of [m] is NaN or infinite.
bool hasNonFinite(Matrix m) => m.data.any((v) => !v.isFinite);

void _same(Matrix a, Matrix b) {
  if (a.rows != b.rows || a.cols != b.cols) {
    throw ArgumentError(
      'Shapes differ: ${a.rows} × ${a.cols} and ${b.rows} × ${b.cols}.',
    );
  }
}
