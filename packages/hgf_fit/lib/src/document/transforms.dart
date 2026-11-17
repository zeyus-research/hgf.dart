import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'clock.dart';
import 'deep_document.dart';
import 'deep_training.dart';
import 'steps.dart';

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
/// `mate` and the `mates_*` functions need it and [groupColumn], and align
/// rows by [timeColumn]'s value (or, without one, by position among the
/// participant's rows in that group).
CsvData applyTransforms(
  CsvData csv,
  List<DocTransform> transforms, {
  String? participantColumn,
  String? groupColumn,
  String? timeColumn,
}) {
  if (transforms.isEmpty) return csv;
  return (TransformTable(
    csv,
    transforms,
    participantColumn: participantColumn,
    groupColumn: groupColumn,
    timeColumn: timeColumn,
  )..evaluate()).table;
}

/// What `missing` counts as missing: CsvData's markers, except `none`
/// (which may be a label, e.g. a "nothing" press).
const _missing = {'', 'na', 'nan', 'null', 'n/a', '.'};

/// The data transforms over a table that may be filled in as it goes: a
/// simulation writes each time step's responses ([setCell]) and then
/// computes the transforms at that step's rows ([evaluate]), so a
/// participant's inputs can read their team-mates' simulated presses
/// (`mate`, `mates_sum`, …). Every function reads the row itself, earlier
/// rows of the same participant (`lag`, `diff`) or the group's rows at the
/// same time (`mate`, `mates_*`), so evaluating the rows of one time step
/// after another gives what [applyTransforms] gives for the whole table.
final class TransformTable {
  TransformTable(
    CsvData csv,
    List<DocTransform> transforms, {
    String? participantColumn,
    String? groupColumn,
    String? timeColumn,
  }) : this._(
         csv,
         transforms,
         _Index(
           csv,
           csv.header.length,
           participantColumn,
           groupColumn,
           timeColumn,
         ),
       );

  TransformTable._(CsvData csv, List<DocTransform> transforms, _Index index)
    : _transforms = List.unmodifiable(transforms),
      _base = csv.header.length {
    final header = [...csv.header];
    for (final t in transforms) {
      if (header.contains(t.name)) {
        throw FormatException(
          'Transform "${t.name}": the data already has a column of that name',
        );
      }
      header.add(t.name);
    }
    table = CsvData(header, [
      for (final r in csv.rows) [...r, for (final _ in transforms) ''],
    ], decimalComma: csv.decimalComma);
    _t = _Table(table, _base, index);
  }

  /// [previous]'s first [keep] transforms (already evaluated: their cells
  /// and columns are reused) followed by the rest of [transforms], which
  /// [evaluate] computes.
  TransformTable._extend(
    TransformTable previous,
    List<DocTransform> transforms,
    int keep,
  ) : _transforms = List.unmodifiable(transforms),
      _base = previous._base,
      _from = keep {
    final header = previous.table.header.sublist(0, _base + keep);
    for (final t in transforms.skip(keep)) {
      if (header.contains(t.name)) {
        throw FormatException(
          'Transform "${t.name}": the data already has a column of that name',
        );
      }
      header.add(t.name);
    }
    final width = header.length, upTo = _base + keep;
    table = CsvData(header, [
      for (final r in previous.table.rows)
        List<String>.filled(width, '')..setRange(0, upTo, r),
    ], decimalComma: previous.table.decimalComma);
    _t = _Table(table, _base, previous._t.index);
    for (final t in transforms.take(keep)) {
      _t.columns[t.name] = previous._t.columns[t.name]!;
    }
  }

  final List<DocTransform> _transforms;
  final int _base;
  late final _Table _t;

  /// The first transform [evaluate] computes (earlier ones were reused).
  int _from = 0;

  /// The data with a column per transform (empty until evaluated).
  late final CsvData table;

  /// Writes [text] into the data's [column] at [row] (before the
  /// transforms read it there).
  void setCell(int row, String column, String text) {
    final c = table.header.indexOf(column);
    if (c < 0 || c >= _base) {
      throw ArgumentError('No data column "$column"');
    }
    table.rows[row][c] = text;
  }

  /// Computes the transforms at [rows] (default: every row). Rows a `lag`
  /// reaches back to must have been evaluated before, and a group's rows
  /// at one time are evaluated together (`mate` reads them).
  void evaluate([List<int>? rows]) {
    _t.active = rows ?? [for (var r = 0; r < table.rows.length; r++) r];
    _t.pass++;
    for (var k = _from; k < _transforms.length; k++) {
      final t = _transforms[k];
      final _Col out;
      try {
        out = _t.eval(t.expression);
      } on FormatException catch (e) {
        throw FormatException('Transform "${t.name}": ${e.message}');
      }
      _t.columns[t.name] = out;
      final c = _base + k;
      for (final r in _t.active) {
        table.rows[r][c] =
            out.text?[r] ??
            (out.v[r].isNaN ? '' : ExprNum.formatNumber(out.v[r]));
      }
    }
  }
}

/// A column's values: numbers (NaN when missing or text) and, for a column
/// of the data or one copied from it, the cells' text.
final class _Col {
  _Col(int n, {bool text = false})
    : v = Float64List(n)..fillRange(0, n, double.nan),
      text = text ? List<String>.filled(n, '') : null;

  final Float64List v;
  final List<String>? text;
}

/// Evaluates expressions at the [active] rows. Each expression node keeps
/// its column across passes (by identity), so `lag` finds the earlier
/// rows' values of whatever it shifts.
final class _Index {
  _Index(
    this.csv,
    this.nData,
    this.participantColumn,
    this.groupColumn,
    this.timeColumn,
  ) : n = csv.rows.length {
    final pc = _col(participantColumn, 'participants');
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
    groupCol = _col(groupColumn, 'groups');
    timeCol = _col(timeColumn, 'time');
  }

  int? _col(String? name, String what) {
    if (name == null) return null;
    final c = csv.header.indexOf(name);
    if (c < 0 || c >= nData) {
      throw FormatException('No column "$name" ($what) in the data');
    }
    return c;
  }

  /// The table it was built on (its first [nData] columns are the data's,
  /// which is all it reads).
  final CsvData csv;
  final int nData;
  final String? participantColumn, groupColumn, timeColumn;
  final int n;
  late final List<String> participantOf;
  late final Int32List previous;
  late final int? groupCol, timeCol;

  /// Whether it indexes [data] as [csv] (the same rows and data columns).
  bool fits(
    CsvData data,
    String? participantColumn,
    String? groupColumn,
    String? timeColumn,
  ) =>
      data.rows.length == n &&
      participantColumn == this.participantColumn &&
      groupColumn == this.groupColumn &&
      timeColumn == this.timeColumn;

  late final mates = _buildMates();

  /// Per row, the rows of the other members of its group at the same time,
  /// in the order the members first appear in the group (−1: absent).
  List<Int32List> _buildMates() {
    final gc = groupCol;
    if (participantColumn == null || gc == null) {
      throw const FormatException(
        'mate() needs the data\'s participant and group columns',
      );
    }
    String key(int r) {
      if (timeCol case final tc?) {
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
      final k = timeCol == null
          ? '${ordinal[pg] = (ordinal[pg] ?? -1) + 1}'
          : key(r);
      keys[r] = k;
      at['$pg\u0000$k'] = r;
    }
    // An empty group cell: no team-mates (e.g. rounds played alone).
    final alone = Int32List(0);
    return [
      for (var r = 0; r < n; r++)
        if (_missing.contains(groupOf[r].toLowerCase()))
          alone
        else
          Int32List.fromList([
            for (final q in members[groupOf[r]]!)
              if (q != participantOf[r])
                at['$q\u0000${groupOf[r]}\u0000${keys[r]}'] ?? -1,
          ]),
    ];
  }
}

/// Evaluates expressions at the [active] rows. Each expression node keeps
/// its column across passes (by identity), so `lag` finds the earlier
/// rows' values of whatever it shifts.
final class _Table {
  _Table(this.csv, this.nData, this.index) : n = csv.rows.length;

  final CsvData csv;

  /// The data's columns come first in [csv]; the transforms' after.
  final int nData;

  /// The rows' participants, previous rows and team-mates (shared by the
  /// tables a [TransformCache] makes from the same data).
  final _Index index;
  final int n;
  List<String> get participantOf => index.participantOf;
  Int32List get previous => index.previous;
  List<int> active = const [];
  int pass = 0;

  /// The transforms' columns so far, by name.
  final columns = <String, _Col>{};
  final _data = <String, (_Col, int)>{};
  final _memo = Map<Expr, _Col>.identity();

  /// A column of the data, read at the active rows once per pass (a
  /// simulation fills it as it goes).
  _Col column(String name) {
    if (columns[name] case final c?) return c;
    final c = csv.header.indexOf(name);
    if (c < 0 || c >= nData) {
      throw FormatException(
        'No column "$name"; the columns are '
        '${[...csv.header.take(nData), ...columns.keys].join(', ')}',
      );
    }
    final (col, at) = _data[name] ?? (_Col(n, text: true), -1);
    if (at != pass) {
      for (final r in active) {
        col.v[r] = csv.number(r, c) ?? double.nan;
        col.text![r] = csv.rows[r][c].trim();
      }
      _data[name] = (col, pass);
    }
    return col;
  }

  _Col _out(Expr e, {bool text = false}) => _memo[e] ??= _Col(n, text: text);

  /// A positive integer constant argument.
  int count(Expr e, String fn) {
    if (e case ExprNum(
      :final value,
    ) when value >= 1 && value == value.roundToDouble()) {
      return value.toInt();
    }
    throw FormatException('$fn: k must be a whole number ≥ 1, not $e');
  }

  void shifted(_Col out, _Col x, int k) {
    for (final r in active) {
      var p = r;
      for (var i = 0; i < k && p >= 0; i++) {
        p = previous[p];
      }
      out.v[r] = p < 0 ? double.nan : x.v[p];
      if (out.text != null) out.text![r] = p < 0 ? '' : x.text![p];
    }
  }

  void mate(_Col out, _Col x, int k) {
    for (final r in active) {
      final others = index.mates[r];
      final m = k <= others.length ? others[k - 1] : -1;
      out.v[r] = m < 0 ? double.nan : x.v[m];
      if (out.text != null) out.text![r] = m < 0 ? '' : x.text![m];
    }
  }

  /// `mates_sum`, `mates_mean`, `mates_count`: over the team-mates present
  /// at the row's time whose x is not missing (for a column of text, a
  /// cell that is not empty: `mates_count(press)` counts presses).
  void mates(_Col out, _Col x, String fn) {
    final text = fn == 'mates_count' ? x.text : null;
    for (final r in active) {
      var sum = 0.0, count = 0;
      for (final m in index.mates[r]) {
        if (m < 0) continue;
        if (text != null) {
          if (!_missing.contains(text[m].toLowerCase())) count++;
          continue;
        }
        if (x.v[m].isNaN) continue;
        sum += x.v[m];
        count++;
      }
      out.v[r] = switch (fn) {
        'mates_count' => count.toDouble(),
        _ when count == 0 => double.nan,
        'mates_sum' => sum,
        _ => sum / count,
      };
    }
  }

  void map(_Col out, List<_Col> xs, double Function(List<double>) f) {
    final env = List<double>.filled(xs.length, 0);
    for (final r in active) {
      var missing = false;
      for (var i = 0; i < xs.length; i++) {
        final x = xs[i].v[r];
        if (x.isNaN) missing = true;
        env[i] = x;
      }
      out.v[r] = missing ? double.nan : f(env);
    }
  }

  /// Compiled closures per node, so a pass per time step stays cheap.
  final _fns = Map<Expr, double Function(List<double>)>.identity();

  _Col eval(Expr e) {
    switch (e) {
      case ExprNum(:final value):
        final out = _out(e);
        for (final r in active) {
          out.v[r] = value;
        }
        return out;
      case ExprStr(:final value):
        throw FormatException(
          'The string \'$value\' can only be compared with a column '
          '(== or !=)',
        );
      case ExprVar(:final name):
        return column(name);
      case ExprNeg(:final a):
        final out = _out(e);
        map(out, [eval(a)], (x) => -x[0]);
        return out;
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
        final out = _out(e);
        for (final r in active) {
          final t = text[r];
          out.v[r] = _missing.contains(t.toLowerCase())
              ? double.nan
              : ((t == s) == (op == '==') ? 1 : 0);
        }
        return out;
      case ExprBinary(:final op, :final a, :final b):
        final f = _fns[e] ??= ExprBinary(
          op,
          const ExprVar('a'),
          const ExprVar('b'),
        ).compile(const ['a', 'b']);
        final out = _out(e);
        map(out, [eval(a), eval(b)], f);
        return out;
      case ExprCall(:final fn, :final args):
        switch (fn.name) {
          case 'lag':
            final x = eval(args[0]);
            final out = _out(e, text: x.text != null);
            shifted(out, x, args.length > 1 ? count(args[1], 'lag') : 1);
            return out;
          case 'diff':
            final x = eval(args[0]);
            final out = _out(e);
            for (final r in active) {
              final p = previous[r];
              out.v[r] = p < 0 ? double.nan : x.v[r] - x.v[p];
            }
            return out;
          case 'mate':
            final x = eval(args[0]);
            final out = _out(e, text: x.text != null);
            mate(out, x, args.length > 1 ? count(args[1], 'mate') : 1);
            return out;
          case 'mates_sum' || 'mates_mean' || 'mates_count':
            final x = eval(args[0]);
            final out = _out(e);
            mates(out, x, fn.name);
            return out;
          case 'missing':
            final x = eval(args[0]);
            final out = _out(e);
            for (final r in active) {
              out.v[r] =
                  (x.text != null
                      ? _missing.contains(x.text![r].toLowerCase())
                      : x.v[r].isNaN)
                  ? 1.0
                  : 0.0;
            }
            return out;
          case 'fill':
            final x = eval(args[0]), y = eval(args[1]);
            final out = _out(e);
            for (final r in active) {
              out.v[r] = x.v[r].isNaN ? y.v[r] : x.v[r];
            }
            return out;
          case 'where':
            // NaN only where the chosen branch is.
            final c = eval(args[0]), t = eval(args[1]), f = eval(args[2]);
            final out = _out(e);
            for (final r in active) {
              out.v[r] = c.v[r].isNaN
                  ? double.nan
                  : (c.v[r] != 0 ? t.v[r] : f.v[r]);
            }
            return out;
        }
        final f = _fns[e] ??= () {
          final names = [for (var i = 0; i < args.length; i++) 'a$i'];
          return ExprCall(fn, [
            for (final v in names) ExprVar(v),
          ]).compile(names);
        }();
        final out = _out(e);
        map(out, [for (final a in args) eval(a)], f);
        return out;
    }
  }
}

/// Keeps the last grid and the last transformed table, so evaluating the
/// same data again after an edit that changed neither is free: the model
/// builder re-reads the data on every edit, and a full session (hundreds of
/// thousands of rows) takes a second or more to put on the clock and
/// transform.
///
/// Transforms appended to the last ones (or the last ones edited from some
/// point on) compute only from there; the rows' participants and
/// team-mates are kept while the data and the participant, group and time
/// columns are. The data must not change in place: a new table is new data
/// (it is matched by identity).
final class TransformCache {
  CsvData? _clockFrom;
  String? _clockKey;
  CsvData? _grid;

  CsvData? _stepsFrom;
  String? _stepsKey;
  CsvData? _stepped;

  CsvData? _from;
  _Index? _index;
  TransformTable? _last;

  Object? _memoFor;
  String? _memoKey;
  Object? _memo;

  /// What [make] gives for [table] read as [key] says, kept until another
  /// table or key comes (one slot: e.g. the subjects of the last
  /// transformed table, while the inputs read the same columns).
  T memo<T extends Object>(Object table, String key, T Function() make) {
    if (identical(table, _memoFor) && key == _memoKey && _memo is T) {
      return _memo as T;
    }
    final v = make();
    _memoFor = table;
    _memoKey = key;
    _memo = v;
    return v;
  }

  /// Whether [DataLayout.transform] trains a deep network it has no
  /// result for (at once), or leaves its columns empty and lists it in
  /// [pendingDeep] (the builder trains in the background, then
  /// [putDeep]s the result).
  bool trainDeep = true;

  final _deepResults = <String, DeepResult>{};

  /// The deep networks the last [withDeep] had no result for, with the
  /// table each reads and its key ([DeepTraining.keyOf]).
  List<({DocDeep deep, CsvData table, String key})> pendingDeep = const [];

  /// The trained deep networks the last [withDeep] used, by id.
  Map<String, DeepResult> usedDeep = const {};

  /// A trained deep network, used from now on wherever its key matches.
  void putDeep(DeepResult r) {
    _deepResults[r.key] = r;
    _deepSlots.clear();
  }

  /// The result for [key], if trained.
  DeepResult? deepResult(String key) => _deepResults[key];

  /// The last few [withDeep] calls, by what they appended.
  final _deepSlots =
      <
        String,
        ({
          CsvData from,
          CsvData out,
          List<({DocDeep deep, CsvData table, String key})> pending,
          Map<String, DeepResult> used,
        })
      >{};

  /// [table] with each of [deep]'s prediction columns appended, in order
  /// (see [DataLayout.transform]); made again only when the table or the
  /// networks changed, or a result came. Sets [pendingDeep] and
  /// [usedDeep].
  CsvData withDeep(CsvData table, List<DocDeep> deep, StepScope scope) {
    final docKey = jsonEncode([
      for (final d in deep) d.toJson(),
      scope.participantColumn,
      scope.groupColumn,
      scope.timeColumn,
    ]);
    if (_deepSlots[docKey] case final slot? when identical(slot.from, table)) {
      pendingDeep = slot.pending;
      usedDeep = slot.used;
      return slot.out;
    }
    final pending = <({DocDeep deep, CsvData table, String key})>[];
    final used = <String, DeepResult>{};
    var out = table;
    for (final d in deep) {
      final key = DeepTraining.keyOf(d, out, scope: scope);
      var r = _deepResults[key];
      if (r == null && trainDeep) {
        r = DeepTraining.run(d, out, scope: scope);
        _deepResults[key] = r;
      }
      if (r != null) {
        used[d.id] = r;
        out = r.appendTo(out);
      } else {
        pending.add((deep: d, table: out, key: key));
        out = _emptyColumns(out, deepPredictionColumns(d));
      }
    }
    // Keep the results in use and a few recent others (undo, redo).
    final inUse = {
      for (final s in _deepSlots.values) ...s.used.values,
      ...used.values,
    };
    final spare = [
      for (final MapEntry(:key, value: r) in _deepResults.entries)
        if (!inUse.any((u) => identical(u, r))) key,
    ];
    for (final k in spare.take(math.max(0, spare.length - 6))) {
      _deepResults.remove(k);
    }
    pendingDeep = pending;
    usedDeep = used;
    _deepSlots.remove(docKey);
    _deepSlots[docKey] = (from: table, out: out, pending: pending, used: used);
    while (_deepSlots.length > 4) {
      _deepSlots.remove(_deepSlots.keys.first);
    }
    return out;
  }

  static CsvData _emptyColumns(CsvData t, List<String> names) {
    for (final n in names) {
      if (t.header.contains(n)) {
        throw FormatException('The data already has a column "$n"');
      }
    }
    return CsvData(
      [...t.header, ...names],
      [
        for (final row in t.rows) [...row, for (final _ in names) ''],
      ],
      decimalComma: t.decimalComma,
    );
  }

  /// How many transforms the last [apply] computed (the rest were reused);
  /// -1 when it returned the cached table as it was.
  int lastComputed = 0;

  /// [csv] on [clock]'s grid (see [DocClock.resample]), made again only
  /// when the data, the clock or the columns changed.
  CsvData clocked(
    CsvData csv,
    DocClock clock, {
    String? participantColumn,
    required String timeColumn,
  }) {
    final key = jsonEncode([clock.toJson(), participantColumn, timeColumn]);
    if (!identical(csv, _clockFrom) || key != _clockKey || _grid == null) {
      _grid = clock.resample(
        csv,
        participantColumn: participantColumn,
        timeColumn: timeColumn,
      );
      _clockFrom = csv;
      _clockKey = key;
    }
    return _grid!;
  }

  /// [csv] through [steps] (see [applySteps]), made again only when the
  /// table, the steps or the scope changed.
  CsvData stepped(CsvData csv, List<DocStep> steps, StepScope scope) {
    final key = jsonEncode([
      for (final s in steps) s.toJson(),
      scope.participantColumn,
      scope.groupColumn,
      scope.timeColumn,
    ]);
    if (!identical(csv, _stepsFrom) || key != _stepsKey || _stepped == null) {
      _stepsFrom = null;
      final rows = <int>[];
      var t = csv;
      for (final (i, s) in steps.indexed) {
        try {
          t = s.apply(t, scope);
        } on FormatException catch (e) {
          stepRows = rows;
          throw FormatException('Data step ${i + 1} (${s.kind}): ${e.message}');
        }
        rows.add(t.rows.length);
      }
      stepRows = rows;
      _stepped = t;
      _stepsFrom = csv;
      _stepsKey = key;
    }
    return _stepped!;
  }

  /// How many rows each step of the last [stepped] call gave (up to the
  /// step that failed, if one did).
  List<int> stepRows = const [];

  /// What [applyTransforms] gives, reusing the last call's work where it
  /// can (see [TransformCache]).
  CsvData apply(
    CsvData csv,
    List<DocTransform> transforms, {
    String? participantColumn,
    String? groupColumn,
    String? timeColumn,
  }) {
    if (transforms.isEmpty) {
      lastComputed = 0;
      return csv;
    }
    final sameData =
        identical(csv, _from) &&
        _index != null &&
        _index!.fits(csv, participantColumn, groupColumn, timeColumn);
    if (!sameData) {
      _from = csv;
      _index = _Index(
        csv,
        csv.header.length,
        participantColumn,
        groupColumn,
        timeColumn,
      );
      _last = null;
    }
    final last = _last;
    var keep = 0;
    if (last != null) {
      final done = last._transforms;
      while (keep < done.length &&
          keep < transforms.length &&
          done[keep].name == transforms[keep].name &&
          '${done[keep].expression}' == '${transforms[keep].expression}') {
        keep++;
      }
      if (keep == done.length && keep == transforms.length) {
        lastComputed = -1;
        return last.table;
      }
    }
    final table = last != null && keep > 0
        ? TransformTable._extend(last, transforms, keep)
        : TransformTable._(csv, transforms, _index!);
    table.evaluate();
    _last = table;
    lastComputed = transforms.length - keep;
    return table.table;
  }
}
