import 'dart:math' as math;
import 'dart:typed_data';

import 'matrix.dart';

/// Inputs and targets, one row per sample.
///
/// For classification [labels] holds each sample's class and [y] its
/// target encoding: one 0/1 column for two classes (a binary output layer),
/// one-hot rows otherwise (a categorical output layer). For regression
/// [labels] is `null` and [classes] is 0.
final class Dataset {
  Dataset(this.name, this.x, this.y, {this.labels, this.classes = 0}) {
    if (x.rows != y.rows) {
      throw ArgumentError('x has ${x.rows} rows but y has ${y.rows}.');
    }
    if (labels != null && labels!.length != x.rows) {
      throw ArgumentError(
        'labels has ${labels!.length} entries, not ${x.rows}.',
      );
    }
  }

  final String name;
  final Matrix x;
  final Matrix y;
  final List<int>? labels;
  final int classes;

  int get length => x.rows;
  bool get isClassification => labels != null;

  /// The rows [rows] (in that order) as a new dataset.
  Dataset subset(List<int> rows) => Dataset(
    name,
    _rows(x, rows),
    _rows(y, rows),
    labels: labels == null ? null : [for (final r in rows) labels![r]],
    classes: classes,
  );

  /// A copy with inputs [x] (same rows).
  Dataset withX(Matrix x) =>
      Dataset(name, x, y, labels: labels, classes: classes);
}

Matrix _rows(Matrix m, List<int> rows) {
  final out = Matrix(rows.length, m.cols);
  for (var i = 0; i < rows.length; i++) {
    out.data.setAll(
      i * m.cols,
      Float64List.sublistView(m.data, rows[i] * m.cols, (rows[i] + 1) * m.cols),
    );
  }
  return out;
}

/// The target encoding of [labels] over [classes] classes: one 0/1 column
/// for two classes, one-hot rows for more.
Matrix encodeLabels(List<int> labels, int classes) => classes == 2
    ? Matrix.column([for (final l in labels) l.toDouble()])
    : oneHot(labels, classes);

/// One-hot rows (`labels.length × classes`).
Matrix oneHot(List<int> labels, int classes) {
  final out = Matrix(labels.length, classes);
  for (var i = 0; i < labels.length; i++) {
    final l = labels[i];
    if (l < 0 || l >= classes) {
      throw ArgumentError.value(l, 'labels[$i]', 'not in [0, $classes)');
    }
    out.data[i * classes + l] = 1;
  }
  return out;
}

/// A labelled classification set from 2-D points.
Dataset _classification(
  String name,
  List<double> xs,
  List<int> labels,
  int k,
) => Dataset(
  name,
  Matrix(labels.length, xs.length ~/ labels.length, Float64List.fromList(xs)),
  encodeLabels(labels, k),
  labels: labels,
  classes: k,
);

double _gauss(math.Random r) {
  // Box–Muller.
  final u = 1 - r.nextDouble();
  final v = r.nextDouble();
  return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v);
}

/// XOR: points around the four corners of `[-1, 1]²`; class 1 where the
/// signs differ. Not linearly separable.
Dataset xor(int n, {double noise = 0.2, int seed = 0}) {
  final r = math.Random(seed);
  final xs = <double>[];
  final labels = <int>[];
  for (var i = 0; i < n; i++) {
    final a = r.nextBool() ? 1.0 : -1.0;
    final b = r.nextBool() ? 1.0 : -1.0;
    xs
      ..add(a + noise * _gauss(r))
      ..add(b + noise * _gauss(r));
    labels.add(a * b < 0 ? 1 : 0);
  }
  return _classification('xor', xs, labels, 2);
}

/// Two interleaving half circles (scikit-learn's `make_moons`), centred on
/// the origin.
Dataset moons(int n, {double noise = 0.1, int seed = 0}) {
  final r = math.Random(seed);
  final xs = <double>[];
  final labels = <int>[];
  for (var i = 0; i < n; i++) {
    final c = i % 2;
    final t = math.pi * r.nextDouble();
    final px = c == 0 ? math.cos(t) : 1 - math.cos(t);
    final py = c == 0 ? math.sin(t) : 0.5 - math.sin(t);
    xs
      ..add(px - 0.5 + noise * _gauss(r))
      ..add(py - 0.25 + noise * _gauss(r));
    labels.add(c);
  }
  return _classification('moons', xs, labels, 2);
}

/// A small circle inside a large one (scikit-learn's `make_circles`);
/// [factor] is the inner radius relative to the outer.
Dataset circles(
  int n, {
  double noise = 0.08,
  double factor = 0.5,
  int seed = 0,
}) {
  final r = math.Random(seed);
  final xs = <double>[];
  final labels = <int>[];
  for (var i = 0; i < n; i++) {
    final c = i % 2;
    final t = 2 * math.pi * r.nextDouble();
    final radius = c == 0 ? 1.0 : factor;
    xs
      ..add(radius * math.cos(t) + noise * _gauss(r))
      ..add(radius * math.sin(t) + noise * _gauss(r));
    labels.add(c);
  }
  return _classification('circles', xs, labels, 2);
}

/// [classes] interleaved spiral arms, each turning [turns] times. The
/// classic hard toy problem: needs curved boundaries in every direction.
Dataset spirals(
  int n, {
  int classes = 3,
  double noise = 0.1,
  double turns = 0.75,
  int seed = 0,
}) {
  final r = math.Random(seed);
  final xs = <double>[];
  final labels = <int>[];
  for (var i = 0; i < n; i++) {
    final c = i % classes;
    final t = r.nextDouble();
    final angle = 2 * math.pi * (turns * t + c / classes);
    final radius = 0.1 + 0.9 * t;
    xs
      ..add(radius * math.cos(angle) + noise * _gauss(r) * t)
      ..add(radius * math.sin(angle) + noise * _gauss(r) * t);
    labels.add(c);
  }
  return _classification('spirals', xs, labels, classes);
}

/// Regression: `y = sin(x) + noise` with `x` uniform in `[-π, π]`.
Dataset sine(int n, {double noise = 0.1, int seed = 0}) {
  final r = math.Random(seed);
  final x = Matrix(n, 1);
  final y = Matrix(n, 1);
  for (var i = 0; i < n; i++) {
    final v = (2 * r.nextDouble() - 1) * math.pi;
    x.data[i] = v;
    y.data[i] = math.sin(v) + noise * _gauss(r);
  }
  return Dataset('sine', x, y);
}

/// A seeded permutation of `0..n-1`.
List<int> permutation(int n, {int seed = 0}) =>
    List<int>.generate(n, (i) => i)..shuffle(math.Random(seed));

/// [ds] in a seeded random order.
Dataset shuffled(Dataset ds, {int seed = 0}) =>
    ds.subset(permutation(ds.length, seed: seed));

/// A seeded random train/test split.
({Dataset train, Dataset test}) split(
  Dataset ds, {
  double testFraction = 0.25,
  int seed = 0,
}) {
  if (!(testFraction > 0 && testFraction < 1)) {
    throw ArgumentError.value(testFraction, 'testFraction', 'not in (0, 1)');
  }
  final p = permutation(ds.length, seed: seed);
  final nTest = (ds.length * testFraction).round();
  return (
    train: ds.subset(p.sublist(nTest)),
    test: ds.subset(p.sublist(0, nTest)),
  );
}

/// Per-column standardisation `(x − mean) / sd`, fitted on one set and
/// applied to others. Constant columns are centred only.
final class Standardiser {
  Standardiser.fit(Matrix x)
    : mean = Float64List(x.cols),
      sd = Float64List(x.cols) {
    for (var j = 0; j < x.cols; j++) {
      var s = 0.0;
      for (var i = 0; i < x.rows; i++) {
        s += x.at(i, j);
      }
      final m = s / x.rows;
      var v = 0.0;
      for (var i = 0; i < x.rows; i++) {
        final d = x.at(i, j) - m;
        v += d * d;
      }
      mean[j] = m;
      final d = math.sqrt(v / x.rows);
      sd[j] = d > 1e-12 ? d : 1;
    }
  }

  final Float64List mean;
  final Float64List sd;

  Matrix apply(Matrix x) {
    final out = Matrix(x.rows, x.cols);
    for (var i = 0; i < x.rows; i++) {
      for (var j = 0; j < x.cols; j++) {
        out.data[i * x.cols + j] = (x.at(i, j) - mean[j]) / sd[j];
      }
    }
    return out;
  }
}
