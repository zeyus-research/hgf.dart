import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'presets.dart';

/// Settings of the simulated observations.
class DataSettings {
  DataSettings({
    this.nTrials = 400,
    this.seed = 1,
    this.pHigh = 0.8,
    this.block = 50,
    this.sdLow = 0.05,
    this.sdHigh = 0.4,
    this.noise = 0.1,
    this.nSources = 3,
    this.sourceSd = 0.4,
  });

  int nTrials;
  int seed;

  /// Binary: the probability of the likely outcome.
  double pHigh;

  /// Binary: trials between reversals. Continuous: trials between switches
  /// of volatility.
  int block;

  /// Continuous: the hidden value's step size in calm and volatile phases.
  double sdLow, sdHigh;

  /// Continuous: observation noise (standard deviation).
  double noise;

  /// Mixture: the number of Gaussian sources (means spread over −2.5..2.5).
  int nSources;

  /// Mixture: each source's standard deviation.
  double sourceSd;
}

/// Simulated observations, and the hidden state that generated them.
class SimulatedData {
  SimulatedData(this.observations, this.hidden);

  /// One value per trial (a category index for categorical data).
  final Float64List observations;

  /// Binary: P(u = 1) per trial. Continuous: the hidden value. Categorical:
  /// P(category k), one series per category. Mixture: the mean of the
  /// source drawn from.
  final List<Float64List> hidden;

  int get length => observations.length;
}

double _gauss(math.Random r) =>
    math.sqrt(-2 * math.log(1 - r.nextDouble())) *
    math.cos(2 * math.pi * r.nextDouble());

/// A block's pick out of [n], different from the previous block's.
int _nextPick(math.Random rng, int n, int previous) {
  if (n < 2) return 0;
  final k = rng.nextInt(n - 1);
  return k >= previous ? k + 1 : k;
}

/// Simulate [model]; [nCategories] for [DataModel.categoricalReversal].
SimulatedData simulate(DataModel model, DataSettings s, {int nCategories = 3}) {
  final rng = math.Random(s.seed);
  final n = s.nTrials;
  final u = Float64List(n), hidden = Float64List(n);
  switch (model) {
    case DataModel.categoricalReversal:
      // One category has probability pHigh, the rest share the remainder;
      // which one changes every block.
      final m = nCategories;
      final p = [for (var k = 0; k < m; k++) Float64List(n)];
      final pHigh = math.max(s.pHigh, 1 / m);
      var likely = 0;
      for (var t = 0; t < n; t++) {
        if (t > 0 && t % s.block == 0) likely = _nextPick(rng, m, likely);
        final r = rng.nextDouble();
        var acc = 0.0, pick = m - 1;
        for (var k = m - 1; k >= 0; k--) {
          final pk = k == likely ? pHigh : (1 - pHigh) / (m - 1);
          p[k][t] = pk;
        }
        for (var k = 0; k < m; k++) {
          acc += p[k][t];
          if (r < acc) {
            pick = k;
            break;
          }
        }
        u[t] = pick.toDouble();
      }
      return SimulatedData(u, p);
    case DataModel.mixture:
      final m = math.max(1, s.nSources);
      var source = 0;
      for (var t = 0; t < n; t++) {
        // Every source once, then at random.
        if (t > 0 && t % s.block == 0) {
          final b = t ~/ s.block;
          source = b < m ? b : _nextPick(rng, m, source);
        }
        final mean = m == 1 ? 0.0 : -2.5 + 5.0 * source / (m - 1);
        hidden[t] = mean;
        u[t] = mean + s.sourceSd * _gauss(rng);
      }
    case DataModel.reversal:
      // Reversal learning: P(u = 1) flips between pHigh and 1 − pHigh.
      for (var t = 0; t < n; t++) {
        final p = (t ~/ s.block).isEven ? s.pHigh : 1 - s.pHigh;
        hidden[t] = p;
        u[t] = rng.nextDouble() < p ? 1 : 0;
      }
    case DataModel.randomWalk:
      // A random walk whose step size switches between calm and volatile.
      var x = 0.0;
      for (var t = 0; t < n; t++) {
        x += ((t ~/ s.block).isEven ? s.sdLow : s.sdHigh) * _gauss(rng);
        hidden[t] = x;
        u[t] = x + s.noise * _gauss(rng);
      }
  }
  return SimulatedData(u, [hidden]);
}

/// One-hot rows (`nTime × n`) for category indices; a missing (non-finite)
/// observation gets a row of zeros.
Float64List oneHot(Float64List categories, int n) {
  final out = Float64List(categories.length * n);
  for (var t = 0; t < categories.length; t++) {
    final c = categories[t];
    if (c.isFinite) out[t * n + c.toInt()] = 1;
  }
  return out;
}

/// A table of numbers read from CSV, TSV or whitespace-separated text
/// ([parseTable]).
class CsvTable {
  CsvTable(this.names, this.columns, this.textCells);

  /// Column names: the header's, or `column 1`, `column 2`, … without one.
  final List<String> names;

  /// One series per column; NaN marks a missing (or non-numeric) cell.
  final List<Float64List> columns;

  /// Per column, how many cells held text other than a missing-value
  /// marker (such a column is not numeric).
  final List<int> textCells;

  int get rows => columns.isEmpty ? 0 : columns.first.length;

  /// Indexes of the columns that hold numbers only (and missing values),
  /// and at least one number.
  List<int> get numericColumns => [
    for (var c = 0; c < columns.length; c++)
      if (textCells[c] == 0 && columns[c].any((v) => v.isFinite)) c,
  ];

  /// The column whose name matches one of [hints] best (case-insensitive:
  /// equal, then starts with, then contains), among [among]; null if none.
  int? guess(List<String> hints, Iterable<int> among) {
    for (final match in <bool Function(String, String)>[
      (n, h) => n == h,
      (n, h) => n.startsWith(h),
      (n, h) => n.contains(h),
    ]) {
      for (final h in hints) {
        for (final c in among) {
          if (match(names[c].toLowerCase().trim(), h)) return c;
        }
      }
    }
    return null;
  }
}

/// Cells read as a missing value.
const _missing = {'', 'na', 'nan', 'null', 'n/a', 'none', '.', '-'};

/// Parse a table of numbers: comma-, semicolon-, tab- or
/// whitespace-separated (detected from the first row), with an optional
/// header row (a first row with any non-numeric cell). Blank lines and
/// lines starting with `#` are skipped; double-quoted cells may contain the
/// separator. Empty cells and `NA`, `NaN`, `null`, … are missing values.
/// With semicolons as separators, a decimal comma is accepted.
CsvTable parseTable(String text) {
  final lines = [
    for (final raw in const LineSplitter().convert(text))
      if (raw.trim().isNotEmpty && !raw.trimLeft().startsWith('#')) raw,
  ];
  if (lines.isEmpty) return CsvTable(const [], const [], const []);
  final first = lines.first;
  final sep = first.contains('\t')
      ? '\t'
      : first.contains(';')
      ? ';'
      : first.contains(',')
      ? ','
      : null;
  final decimalComma = sep == ';';
  final rows = [for (final l in lines) _splitRow(l, sep)];
  double? number(String cell) {
    final c = cell.trim();
    return double.tryParse(decimalComma ? c.replaceAll(',', '.') : c);
  }

  bool isMissing(String cell) => _missing.contains(cell.trim().toLowerCase());
  final header = rows.first.any((c) => !isMissing(c) && number(c) == null);
  final width = rows.fold(0, (w, r) => math.max(w, r.length));
  final body = header ? rows.skip(1).toList() : rows;
  final names = [
    for (var c = 0; c < width; c++)
      header && c < rows.first.length && rows.first[c].trim().isNotEmpty
          ? rows.first[c].trim()
          : 'column ${c + 1}',
  ];
  final columns = [for (var c = 0; c < width; c++) Float64List(body.length)];
  final textCells = List.filled(width, 0);
  for (var r = 0; r < body.length; r++) {
    final row = body[r];
    for (var c = 0; c < width; c++) {
      final cell = c < row.length ? row[c] : '';
      final v = number(cell);
      if (v == null && !isMissing(cell)) textCells[c]++;
      columns[c][r] = v ?? double.nan;
    }
  }
  return CsvTable(names, columns, textCells);
}

/// Split one line at [sep] (null: runs of whitespace), keeping separators
/// inside double quotes, and unquoting (`""` is a literal quote).
List<String> _splitRow(String line, String? sep) {
  if (sep == null) return line.trim().split(RegExp(r'\s+'));
  final out = <String>[];
  final cell = StringBuffer();
  var quoted = false;
  for (var i = 0; i < line.length; i++) {
    final ch = line[i];
    if (quoted) {
      if (ch == '"') {
        if (i + 1 < line.length && line[i + 1] == '"') {
          cell.write('"');
          i++;
        } else {
          quoted = false;
        }
      } else {
        cell.write(ch);
      }
    } else if (ch == '"') {
      quoted = true;
    } else if (ch == sep) {
      out.add(cell.toString());
      cell.clear();
    } else {
      cell.write(ch);
    }
  }
  out.add(cell.toString());
  return out;
}

/// The first numeric column of a table ([parseTable]): one value per row,
/// NaN for a missing one.
Float64List parseColumn(String text) {
  final t = parseTable(text);
  final numeric = t.numericColumns;
  if (numeric.isEmpty) {
    final c = t.textCells.indexWhere((n) => n > 0);
    throw FormatException(
      c < 0 ? 'No numbers found' : 'Column "${t.names[c]}" holds text',
    );
  }
  return t.columns[numeric.first];
}
