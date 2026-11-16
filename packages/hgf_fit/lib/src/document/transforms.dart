import 'dart:typed_data';

import 'package:hgf/hgf.dart';

/// A data transform: a new column [name], computed from the data's columns
/// (and earlier transforms) by [expression], before filtering. It is
/// computed once, not differentiated, so it may use the data-only
/// functions: `lag(x, k)` and `diff(x)` (per participant), `mate(x, k)`
/// (the k-th other member of the participant's group, at the same time; see
/// [DocData]), `missing(x)` and `fill(x, v)`. A string compares with a
/// column's text: `press == 'left'`.
///
/// A missing cell (empty, `NA`, ...) is NaN and stays NaN through
/// arithmetic and comparisons; a NaN result is an empty cell, which an
/// input reads as unobserved. Text compares as written: `none` is a label
/// (`press == 'none'`), though CsvData reads it as missing in a numeric
/// column.
final class DocTransform {
  DocTransform(this.name, String expression)
    : expression = Expr.parse(expression);

  DocTransform.of(this.name, this.expression);

  final String name;
  final Expr expression;

  Map<String, Object?> toJson() => {'name': name, 'expr': '$expression'};

  factory DocTransform.fromJson(Map<String, Object?> json) =>
      DocTransform(json['name'] as String, json['expr'] as String);
}

/// [csv] with a column per transform appended, computed in order (see
/// [DocTransform]). [participantColumn] scopes `lag` and `diff`;
/// `mate` needs it and [groupColumn], and aligns rows by [timeColumn]'s
/// value (or, without one, by position among the participant's rows in
/// that group).
CsvData applyTransforms(
  CsvData csv,
  List<DocTransform> transforms, {
  String? participantColumn,
  String? groupColumn,
  String? timeColumn,
}) {
  if (transforms.isEmpty) return csv;
  return _Table(
    csv,
    participantColumn,
    groupColumn,
    timeColumn,
  ).run(transforms);
}

/// What `missing` counts as missing: CsvData's markers, except `none`
/// (which may be a label, e.g. a "nothing" press).
const _missing = {'', 'na', 'nan', 'null', 'n/a', '.'};

/// A column's values: numbers (NaN when missing or text) and, for a column
/// of the data or one copied from it, the cells' text.
final class _Col {
  _Col(this.v, [this.text]);

  final Float64List v;
  final List<String>? text;
}

final class _Table {
  _Table(this.csv, this.participantColumn, this.groupColumn, this.timeColumn)
    : n = csv.rows.length {
    int? col(String? name, String what) {
      if (name == null) return null;
      final c = csv.header.indexOf(name);
      if (c < 0) {
        throw FormatException('No column "$name" ($what) in the data');
      }
      return c;
    }

    final pc = col(participantColumn, 'participants');
    participantOf = [
      for (final row in csv.rows) pc == null ? '' : row[pc].trim(),
    ];
    // The previous row of the same participant.
    previous = Int32List(n);
    final last = <String, int>{};
    for (var r = 0; r < n; r++) {
      previous[r] = last[participantOf[r]] ?? -1;
      last[participantOf[r]] = r;
    }
    _groupCol = col(groupColumn, 'groups');
    _timeCol = col(timeColumn, 'time');
  }

  final CsvData csv;
  final String? participantColumn, groupColumn, timeColumn;
  final int n;
  late final List<String> participantOf;
  late final Int32List previous;
  int? _groupCol, _timeCol;
  final _columns = <String, _Col>{};

  CsvData run(List<DocTransform> transforms) {
    final header = [...csv.header];
    final rows = [
      for (final r in csv.rows) [...r],
    ];
    for (final t in transforms) {
      if (header.contains(t.name) || _columns.containsKey(t.name)) {
        throw FormatException(
          'Transform "${t.name}": the data already has a column of that name',
        );
      }
      final _Col out;
      try {
        out = eval(t.expression);
      } on FormatException catch (e) {
        throw FormatException('Transform "${t.name}": ${e.message}');
      }
      _columns[t.name] = out;
      header.add(t.name);
      for (var r = 0; r < n; r++) {
        rows[r].add(
          out.text?[r] ??
              (out.v[r].isNaN ? '' : ExprNum.formatNumber(out.v[r])),
        );
      }
    }
    return CsvData(header, rows, decimalComma: csv.decimalComma);
  }

  _Col column(String name) {
    if (_columns[name] case final c?) return c;
    final c = csv.header.indexOf(name);
    if (c < 0) {
      throw FormatException(
        'No column "$name"; the columns are '
        '${[...csv.header, ..._columns.keys].join(', ')}',
      );
    }
    final v = Float64List(n);
    final text = <String>[];
    for (var r = 0; r < n; r++) {
      v[r] = csv.number(r, c) ?? double.nan;
      text.add(csv.rows[r][c].trim());
    }
    return _columns[name] = _Col(v, text);
  }

  _Col constant(double x) => _Col(Float64List(n)..fillRange(0, n, x));

  /// A positive integer constant argument.
  int count(Expr e, String fn) {
    if (e case ExprNum(
      :final value,
    ) when value >= 1 && value == value.roundToDouble()) {
      return value.toInt();
    }
    throw FormatException('$fn: k must be a whole number ≥ 1, not $e');
  }

  _Col shifted(_Col x, int k) {
    final v = Float64List(n);
    final text = x.text == null ? null : List<String>.filled(n, '');
    for (var r = 0; r < n; r++) {
      var p = r;
      for (var i = 0; i < k && p >= 0; i++) {
        p = previous[p];
      }
      v[r] = p < 0 ? double.nan : x.v[p];
      if (text != null && p >= 0) text[r] = x.text![p];
    }
    return _Col(v, text);
  }

  late final _mateRows = _buildMates();

  /// Per mate rank k (from 1) a row index per row (−1: none).
  List<Int32List> _buildMates() {
    final gc = _groupCol;
    if (participantColumn == null || gc == null) {
      throw const FormatException(
        'mate() needs the data\'s participant and group columns',
      );
    }
    String key(int r) {
      if (_timeCol case final tc?) {
        final x = csv.number(r, tc);
        return x == null || x.isNaN ? csv.rows[r][tc].trim() : '$x';
      }
      return '';
    }

    final groupOf = [for (final row in csv.rows) row[gc].trim()];
    // Members per group, in order of first appearance.
    final members = <String, List<String>>{};
    // (participant, group, key) → row; without a time column the key is the
    // row's position among the participant's rows in the group.
    final at = <String, int>{};
    final ordinal = <String, int>{};
    final keys = List<String>.filled(n, '');
    for (var r = 0; r < n; r++) {
      final p = participantOf[r], g = groupOf[r];
      final m = members[g] ??= [];
      if (!m.contains(p)) m.add(p);
      final pg = '$p\u0000$g';
      final k = _timeCol == null
          ? '${ordinal[pg] = (ordinal[pg] ?? -1) + 1}'
          : key(r);
      keys[r] = k;
      at['$pg\u0000$k'] = r;
    }
    final most = members.values.fold(0, (a, m) => m.length > a ? m.length : a);
    return [
      for (var rank = 1; rank < most; rank++)
        () {
          final out = Int32List(n)..fillRange(0, n, -1);
          for (var r = 0; r < n; r++) {
            final g = groupOf[r];
            final others = [
              for (final q in members[g]!)
                if (q != participantOf[r]) q,
            ];
            if (rank > others.length) continue;
            out[r] = at['${others[rank - 1]}\u0000$g\u0000${keys[r]}'] ?? -1;
          }
          return out;
        }(),
    ];
  }

  _Col mate(_Col x, int k) {
    final rows = k <= _mateRows.length ? _mateRows[k - 1] : null;
    final v = Float64List(n)..fillRange(0, n, double.nan);
    final text = x.text == null ? null : List<String>.filled(n, '');
    if (rows != null) {
      for (var r = 0; r < n; r++) {
        final m = rows[r];
        if (m < 0) continue;
        v[r] = x.v[m];
        if (text != null) text[r] = x.text![m];
      }
    }
    return _Col(v, text);
  }

  _Col map(List<_Col> xs, double Function(List<double>) f) {
    final v = Float64List(n);
    final env = List<double>.filled(xs.length, 0);
    for (var r = 0; r < n; r++) {
      var missing = false;
      for (var i = 0; i < xs.length; i++) {
        final x = xs[i].v[r];
        if (x.isNaN) missing = true;
        env[i] = x;
      }
      v[r] = missing ? double.nan : f(env);
    }
    return _Col(v);
  }

  _Col eval(Expr e) {
    switch (e) {
      case ExprNum(:final value):
        return constant(value);
      case ExprStr(:final value):
        throw FormatException(
          'The string \'$value\' can only be compared with a column '
          '(== or !=)',
        );
      case ExprVar(:final name):
        return column(name);
      case ExprNeg(:final a):
        return map([eval(a)], (x) => -x[0]);
      case ExprBinary(:final op, :final a, :final b)
          when (op == '==' || op == '!=') && (a is ExprStr || b is ExprStr):
        final s = (a is ExprStr ? a : b as ExprStr).value;
        final other = eval(a is ExprStr ? b : a);
        final text =
            other.text ??
            (throw FormatException(
              'Compare \'$s\' with a column of the data (or a lag/mate of '
              'one), not ${a is ExprStr ? b : a}',
            ));
        final v = Float64List(n);
        for (var r = 0; r < n; r++) {
          final t = text[r];
          v[r] = _missing.contains(t.toLowerCase())
              ? double.nan
              : ((t == s) == (op == '==') ? 1 : 0);
        }
        return _Col(v);
      case ExprBinary(:final op, :final a, :final b):
        final f = ExprBinary(
          op,
          const ExprVar('a'),
          const ExprVar('b'),
        ).compile(const ['a', 'b']);
        return map([eval(a), eval(b)], f);
      case ExprCall(:final fn, :final args):
        switch (fn.name) {
          case 'lag':
            return shifted(
              eval(args[0]),
              args.length > 1 ? count(args[1], 'lag') : 1,
            );
          case 'diff':
            final x = eval(args[0]);
            return map([x, shifted(x, 1)], (v) => v[0] - v[1]);
          case 'mate':
            return mate(
              eval(args[0]),
              args.length > 1 ? count(args[1], 'mate') : 1,
            );
          case 'missing':
            final x = eval(args[0]);
            return _Col(
              Float64List.fromList([
                for (var r = 0; r < n; r++)
                  (x.text != null
                          ? _missing.contains(x.text![r].toLowerCase())
                          : x.v[r].isNaN)
                      ? 1.0
                      : 0.0,
              ]),
            );
          case 'fill':
            final x = eval(args[0]), y = eval(args[1]);
            return _Col(
              Float64List.fromList([
                for (var r = 0; r < n; r++) x.v[r].isNaN ? y.v[r] : x.v[r],
              ]),
            );
          case 'where':
            // NaN only where the chosen branch is.
            final c = eval(args[0]), t = eval(args[1]), f = eval(args[2]);
            return _Col(
              Float64List.fromList([
                for (var r = 0; r < n; r++)
                  c.v[r].isNaN ? double.nan : (c.v[r] != 0 ? t.v[r] : f.v[r]),
              ]),
            );
        }
        final names = [for (var i = 0; i < args.length; i++) 'a$i'];
        final f = ExprCall(fn, [
          for (final v in names) ExprVar(v),
        ]).compile(names);
        return map([for (final a in args) eval(a)], f);
    }
  }
}
