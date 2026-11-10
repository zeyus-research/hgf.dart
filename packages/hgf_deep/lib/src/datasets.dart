import 'dart:math' as math;
import 'dart:typed_data';

import 'csv.dart';
import 'matrix.dart';

/// Inputs and targets, one row per sample.
///
/// For classification [labels] holds each sample's class and [y] its
/// target encoding: one 0/1 column for two classes (a binary output layer),
/// one-hot rows otherwise (a categorical output layer). For regression
/// [labels] is `null` and [classes] is 0.
final class Dataset {
  Dataset(
    this.name,
    this.x,
    this.y, {
    this.labels,
    this.classes = 0,
    this.classNames,
    this.featureNames,
  }) {
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

  /// The name of each class (label `k` is `classNames[k]`), when known.
  final List<String>? classNames;

  /// The name of each column of [x], when known.
  final List<String>? featureNames;

  int get length => x.rows;
  bool get isClassification => labels != null;

  /// The rows [rows] (in that order) as a new dataset.
  Dataset subset(List<int> rows) => Dataset(
    name,
    _rows(x, rows),
    _rows(y, rows),
    labels: labels == null ? null : [for (final r in rows) labels![r]],
    classes: classes,
    classNames: classNames,
    featureNames: featureNames,
  );

  /// A copy with inputs [x] (same rows).
  Dataset withX(Matrix x) => Dataset(
    name,
    x,
    y,
    labels: labels,
    classes: classes,
    classNames: classNames,
    featureNames: featureNames,
  );

  /// Reads a dataset from CSV text (comma-, semicolon- or tab-separated,
  /// with a header row; see [CsvData.parse]): the [target] column(s) as
  /// `y`, the [features] columns (by default every other column) as `x`.
  ///
  /// [classification] (by default: when a target column holds text, or
  /// when there is one target holding at most 20 distinct whole numbers)
  /// turns the single
  /// target into class labels: its distinct values, sorted (numerically if
  /// they are numbers), become classes `0, 1, …` ([classNames]), encoded as
  /// [encodeLabels] does. Rows with a missing value (an empty cell, `NA`,
  /// `NaN`, …) in a column used are dropped; a feature that holds text is
  /// an error.
  ///
  /// ```dart
  /// final iris = Dataset.fromCsv(
  ///   File('iris.csv').readAsStringSync(),
  ///   target: ['species'],
  /// );
  /// ```
  factory Dataset.fromCsv(
    String text, {
    required List<String> target,
    List<String>? features,
    bool? classification,
    String name = 'csv',
  }) => _fromCsv(CsvData.parse(text), target, features, classification, name);
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

/// [ds] in minibatches of [batchSize] rows (the last one may be smaller),
/// in a seeded random order unless [shuffle] is false: one epoch of
/// `DeepNetwork.batchUpdate` calls.
///
/// ```dart
/// for (var epoch = 0; epoch < 20; epoch++) {
///   for (final b in minibatches(train, 64, seed: epoch)) {
///     net.batchUpdate(b.x, b.y, optimiser: adam);
///   }
/// }
/// ```
Iterable<Dataset> minibatches(
  Dataset ds,
  int batchSize, {
  bool shuffle = true,
  int seed = 0,
}) sync* {
  if (batchSize < 1) throw ArgumentError.value(batchSize, 'batchSize');
  final order = shuffle
      ? permutation(ds.length, seed: seed)
      : List<int>.generate(ds.length, (i) => i);
  for (var s = 0; s < ds.length; s += batchSize) {
    yield ds.subset(order.sublist(s, math.min(s + batchSize, ds.length)));
  }
}

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

  /// A standardiser with these column [mean]s and [sd]s.
  Standardiser(this.mean, this.sd) {
    if (mean.length != sd.length) {
      throw ArgumentError('mean and sd have different lengths');
    }
  }

  /// Reads [toJson]'s output.
  factory Standardiser.fromJson(Map<String, Object?> json) => Standardiser(
    Float64List.fromList([
      for (final v in json['mean'] as List<Object?>) (v as num).toDouble(),
    ]),
    Float64List.fromList([
      for (final v in json['sd'] as List<Object?>) (v as num).toDouble(),
    ]),
  );

  final Float64List mean;
  final Float64List sd;

  /// The means and standard deviations, to save with a trained network:
  /// new inputs must be scaled as the training data were.
  Map<String, Object?> toJson() => {'mean': mean, 'sd': sd};

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

Dataset _fromCsv(
  CsvData csv,
  List<String> target,
  List<String>? features,
  bool? classification,
  String name,
) {
  if (target.isEmpty) throw ArgumentError('No target column');
  int col(String n) {
    final c = csv.header.indexOf(n);
    if (c < 0) {
      throw ArgumentError(
        'No column "$n"; the columns are ${csv.header.join(', ')}',
      );
    }
    return c;
  }

  final tCols = [for (final n in target) col(n)];
  final fCols = features == null
      ? [
          for (var c = 0; c < csv.header.length; c++)
            if (!tCols.contains(c)) c,
        ]
      : [for (final n in features) col(n)];
  if (fCols.isEmpty) throw ArgumentError('No feature columns');
  for (final c in fCols) {
    final bad = csv.firstText(c);
    if (bad != null) {
      throw FormatException(
        'Feature "${csv.header[c]}" holds text on row ${bad + 1}: '
        '"${csv.rows[bad][c]}"',
      );
    }
  }
  final textTarget = tCols.any((c) => csv.firstText(c) != null);
  final rows = [
    for (var r = 0; r < csv.rows.length; r++)
      if ([...fCols, ...tCols].every((c) => !csv.isMissing(r, c))) r,
  ];
  final cls =
      classification ??
      (textTarget ||
          (tCols.length == 1 &&
              rows.every((r) {
                final v = csv.number(r, tCols.single)!;
                return v == v.roundToDouble();
              }) &&
              {for (final r in rows) csv.number(r, tCols.single)}.length <=
                  20));
  final x = Matrix(rows.length, fCols.length);
  for (var i = 0; i < rows.length; i++) {
    for (var j = 0; j < fCols.length; j++) {
      x.data[i * fCols.length + j] = csv.number(rows[i], fCols[j])!;
    }
  }
  final featureNames = [for (final c in fCols) csv.header[c]];
  if (!cls) {
    if (textTarget) {
      throw FormatException('A regression target holds text');
    }
    final y = Matrix(rows.length, tCols.length);
    for (var i = 0; i < rows.length; i++) {
      for (var j = 0; j < tCols.length; j++) {
        y.data[i * tCols.length + j] = csv.number(rows[i], tCols[j])!;
      }
    }
    return Dataset(name, x, y, featureNames: featureNames);
  }
  if (tCols.length != 1) {
    throw ArgumentError('Classification takes one target column');
  }
  final t = tCols.single;
  final values = {for (final r in rows) csv.rows[r][t].trim()}.toList();
  final numeric = values.every((v) => double.tryParse(v) != null);
  values.sort(
    numeric
        ? (a, b) => double.parse(a).compareTo(double.parse(b))
        : (a, b) => a.compareTo(b),
  );
  if (values.length < 2) {
    throw FormatException('The target has ${values.length} class(es)');
  }
  final index = {for (var k = 0; k < values.length; k++) values[k]: k};
  final labels = [for (final r in rows) index[csv.rows[r][t].trim()]!];
  return Dataset(
    name,
    x,
    encodeLabels(labels, values.length),
    labels: labels,
    classes: values.length,
    classNames: values,
    featureNames: featureNames,
  );
}
