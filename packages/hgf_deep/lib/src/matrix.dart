import 'dart:typed_data';

/// A dense row-major matrix of doubles.
///
/// Inputs and outputs of [DeepNetwork] use it with one row per sample
/// (`samples × features`), and weight matrices with one row per child node
/// (`child × parent[+1]`).
final class Matrix {
  /// A zero matrix, or one over [data] (row-major, `rows * cols` long).
  Matrix(this.rows, this.cols, [Float64List? data])
    : data = data ?? Float64List(rows * cols) {
    if (rows < 0 || cols < 0) {
      throw ArgumentError('Matrix dimensions must be non-negative.');
    }
    if (this.data.length != rows * cols) {
      throw ArgumentError(
        'data has ${this.data.length} values, expected $rows × $cols.',
      );
    }
  }

  /// A matrix from a list of equal-length rows.
  factory Matrix.fromRows(List<List<double>> rows) {
    final n = rows.length;
    final m = n == 0 ? 0 : rows.first.length;
    final out = Matrix(n, m);
    for (var i = 0; i < n; i++) {
      final row = rows[i];
      if (row.length != m) {
        throw ArgumentError('Row $i has ${row.length} values, expected $m.');
      }
      out.data.setRange(i * m, i * m + m, row);
    }
    return out;
  }

  /// A matrix whose every entry is [value].
  factory Matrix.filled(int rows, int cols, double value) => Matrix(
    rows,
    cols,
    Float64List(rows * cols)..fillRange(0, rows * cols, value),
  );

  /// The identity matrix (`n × n`).
  factory Matrix.identity(int n) {
    final out = Matrix(n, n);
    for (var i = 0; i < n; i++) {
      out.data[i * n + i] = 1.0;
    }
    return out;
  }

  /// A single row (one sample) as a `1 × values.length` matrix.
  factory Matrix.row(List<double> values) =>
      Matrix(1, values.length, Float64List.fromList(values));

  /// A column (one feature, one sample per row) as a `values.length × 1`
  /// matrix.
  factory Matrix.column(List<double> values) =>
      Matrix(values.length, 1, Float64List.fromList(values));

  final int rows;
  final int cols;

  /// Row-major values: entry `(i, j)` is `data[i * cols + j]`.
  final Float64List data;

  /// Entry `(i, j)`.
  double at(int i, int j) => data[i * cols + j];

  /// Set entry `(i, j)`.
  void set(int i, int j, double value) => data[i * cols + j] = value;

  /// Row [i] as a view (no copy).
  Float64List row(int i) =>
      Float64List.sublistView(data, i * cols, i * cols + cols);

  /// Rows as lists (a copy).
  List<List<double>> toRows() => [
    for (var i = 0; i < rows; i++) row(i).toList(),
  ];

  /// A deep copy.
  Matrix copy() => Matrix(rows, cols, Float64List.fromList(data));

  @override
  String toString() => 'Matrix($rows × $cols)';
}
