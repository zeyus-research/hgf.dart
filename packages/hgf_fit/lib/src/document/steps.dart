import 'dart:math' as math;

import 'package:hgf/hgf.dart';

import 'transforms.dart';

/// Where a [DocStep] runs: the data layout's participant, group and time
/// columns. Steps work on each participant's rows (in file order; without
/// a participant column, every row is one participant's).
typedef StepScope = ({
  String? participantColumn,
  String? groupColumn,
  String? timeColumn,
});

/// A data-pipeline step: it changes the table's rows (or rewrites columns
/// in place) after the clock and before the transforms. Expressions in a
/// step see the table as it is at that step, with the data-only functions
/// of [DocTransform] (`lag`, `diff`, `mate`, …).
///
/// The kinds: [DecimateStep] (keep every k-th row, aggregating the rest),
/// [SegmentStep] (windows around events), [FilterStep] (keep the rows an
/// expression picks), [StandardiseStep] (z-score columns), [SplitStep]
/// (label rows or participants train / test).
sealed class DocStep {
  const DocStep();

  /// The JSON `kind`.
  String get kind;

  /// The columns the step reads (expressions' variables included).
  Set<String> get reads;

  /// The table's columns after the step, given its columns before.
  List<String> columnsAfter(List<String> header) => header;

  /// One line for the builder's card.
  String get summary;

  /// [csv] after the step.
  CsvData apply(CsvData csv, StepScope scope);

  Map<String, Object?> toJson();

  factory DocStep.fromJson(Map<String, Object?> json) {
    List<String> strings(Object? v) => [...?(v as List?)?.cast<String>()];
    Expr? expr(Object? v) => v == null ? null : Expr.parse(v as String);
    return switch (json['kind']) {
      'decimate' => DecimateStep(
        factor: (json['factor'] as num).toInt(),
        how: StepAggregate.parse(json['how'] as String? ?? 'first'),
        columns: {
          for (final MapEntry(:key, :value)
              in ((json['columns'] as Map?) ?? const {}).entries)
            key as String: StepAggregate.parse(value as String),
        },
        within: strings(json['within']),
      ),
      'segment' => SegmentStep(
        at: expr(json['at'])!,
        before: (json['before'] as num?)?.toInt() ?? 0,
        after: (json['after'] as num?)?.toInt(),
        column: json['column'] as String? ?? 'segment',
        offsetColumn: json['offset_column'] as String?,
      ),
      'filter' => FilterStep(expr(json['keep'])!),
      'standardise' => StandardiseStep(
        columns: strings(json['columns']),
        perParticipant: json['per_participant'] as bool? ?? true,
        fitOn: expr(json['fit_on']),
      ),
      'split' => SplitStep(
        column: json['column'] as String? ?? 'split',
        parts: [
          for (final p in (json['parts'] as List? ?? const []))
            (
              label: (p as Map)['label'] as String,
              fraction: (p['fraction'] as num).toDouble(),
            ),
        ],
        byParticipant: json['by_participant'] as bool? ?? false,
        seed: (json['seed'] as num?)?.toInt() ?? 0,
      ),
      final k => throw FormatException('Unknown data step "$k"'),
    };
  }
}

/// [csv] after each of [steps] in order.
CsvData applySteps(CsvData csv, List<DocStep> steps, StepScope scope) {
  for (final (i, s) in steps.indexed) {
    try {
      csv = s.apply(csv, scope);
    } on FormatException catch (e) {
      throw FormatException('Data step ${i + 1} (${s.kind}): ${e.message}');
    }
  }
  return csv;
}

/// The columns after [steps], given the data's [header].
List<String> stepColumns(List<String> header, List<DocStep> steps) {
  for (final s in steps) {
    header = s.columnsAfter(header);
  }
  return header;
}

/// The rows of each participant of [csv] (first appearance order; rows in
/// file order).
List<List<int>> _units(CsvData csv, String? participantColumn) {
  if (participantColumn == null) {
    return [List.generate(csv.rows.length, (r) => r)];
  }
  final c = _col(csv, participantColumn, 'participants');
  final groups = <String, List<int>>{};
  for (var r = 0; r < csv.rows.length; r++) {
    (groups[csv.rows[r][c].trim()] ??= []).add(r);
  }
  return groups.values.toList();
}

int _col(CsvData csv, String name, String what) {
  final c = csv.header.indexOf(name);
  if (c < 0) throw FormatException('No column "$name" ($what) in the data');
  return c;
}

/// [e] evaluated on every row of [csv] (NaN where missing).
List<double> _evaluate(CsvData csv, Expr e, StepScope scope) {
  final t = applyTransforms(
    csv,
    [DocTransform.of('\u0000step', e)],
    participantColumn: scope.participantColumn,
    groupColumn: scope.groupColumn,
    timeColumn: scope.timeColumn,
  );
  final c = t.header.length - 1;
  return [for (var r = 0; r < t.rows.length; r++) t.number(r, c) ?? double.nan];
}

bool _on(double v) => !v.isNaN && v != 0;

/// [csv] with [rows] (indices into it) in that order.
CsvData _pick(CsvData csv, Iterable<int> rows) => CsvData(csv.header, [
  for (final r in rows) csv.rows[r],
], decimalComma: csv.decimalComma);

/// What `missing` counts as missing (as the transforms do: `none` is a
/// label).
const _missing = {'', 'na', 'nan', 'null', 'n/a', '.'};

bool _isMissing(String cell) => _missing.contains(cell.trim().toLowerCase());

String _num(double v) => v.isNaN ? '' : ExprNum.formatNumber(v);

/// How a [DecimateStep] makes one cell of a block's rows.
enum StepAggregate {
  /// The block's first row.
  first,

  /// The block's last row.
  last,

  /// The mean of the numbers (missing cells skipped).
  mean,
  sum,
  min,
  max,

  /// 1 if any number is nonzero, 0 if none is (missing cells skipped).
  any,

  /// The most common non-missing text (ties: the first seen).
  mode;

  static StepAggregate parse(String name) => StepAggregate.values.firstWhere(
    (a) => a.name == name,
    orElse: () => throw FormatException('Unknown aggregate "$name"'),
  );

  /// Whether it reads numbers.
  bool get numeric => switch (this) {
    first || last || mode => false,
    _ => true,
  };
}

/// Decimation: each participant's rows in blocks of [factor] (a block also
/// ends where a [within] column changes, e.g. a segment), one row per
/// block. A column's cell is the block's first row's ([how]; [columns]
/// sets it per column), or aggregated over the block: `any` keeps a press
/// that falls between kept rows, `mean` averages a signal. The times are
/// the first row's, so a 120 Hz grid decimated by 12 is a 10 Hz grid on the
/// same clock.
final class DecimateStep extends DocStep {
  DecimateStep({
    required this.factor,
    this.how = StepAggregate.first,
    Map<String, StepAggregate> columns = const {},
    List<String> within = const [],
  }) : columns = Map.unmodifiable(columns),
       within = List.unmodifiable(within) {
    if (factor < 1) {
      throw ArgumentError.value(factor, 'factor', 'must be at least 1');
    }
  }

  final int factor;
  final StepAggregate how;
  final Map<String, StepAggregate> columns;
  final List<String> within;

  @override
  String get kind => 'decimate';

  @override
  Set<String> get reads => {...columns.keys, ...within};

  @override
  String get summary =>
      'every $factor${switch (factor % 10) {
        1 when factor % 100 != 11 => 'st',
        2 when factor % 100 != 12 => 'nd',
        3 when factor % 100 != 13 => 'rd',
        _ => 'th',
      }} row'
      '${how == StepAggregate.first ? '' : ', ${how.name}'}'
      '${columns.isEmpty ? '' : ' (${[for (final MapEntry(:key, :value) in columns.entries) '$key: ${value.name}'].join(', ')})'}';

  @override
  CsvData apply(CsvData csv, StepScope scope) {
    if (factor == 1 && columns.isEmpty && how == StepAggregate.first) {
      return csv;
    }
    final aggs = [for (final h in csv.header) columns[h] ?? how];
    // [how] aggregates numbers; label columns (text in a cell; the
    // participant, group and time columns) keep the block's first row.
    for (var k = 0; k < csv.header.length; k++) {
      final h = csv.header[k];
      if (columns.containsKey(h) || !aggs[k].numeric) continue;
      if (h == scope.participantColumn ||
          h == scope.groupColumn ||
          h == scope.timeColumn ||
          csv.firstText(k) != null) {
        aggs[k] = StepAggregate.first;
      }
    }
    for (final c in columns.keys) {
      _col(csv, c, 'to aggregate');
    }
    final wc = [
      for (final w in within) _col(csv, w, 'blocks end where it changes'),
    ];
    final rows = <List<String>>[];
    for (final unit in _units(csv, scope.participantColumn)) {
      var start = 0;
      while (start < unit.length) {
        var end = start + 1;
        while (end < unit.length &&
            end - start < factor &&
            wc.every(
              (c) =>
                  csv.rows[unit[end]][c].trim() ==
                  csv.rows[unit[start]][c].trim(),
            )) {
          end++;
        }
        final block = unit.sublist(start, end);
        rows.add([
          for (var c = 0; c < csv.header.length; c++)
            _aggregate(csv, block, c, aggs[c]),
        ]);
        start = end;
      }
    }
    return CsvData(csv.header, rows, decimalComma: csv.decimalComma);
  }

  String _aggregate(CsvData csv, List<int> block, int c, StepAggregate a) {
    switch (a) {
      case StepAggregate.first:
        return csv.rows[block.first][c];
      case StepAggregate.last:
        return csv.rows[block.last][c];
      case StepAggregate.mode:
        final counts = <String, int>{};
        for (final r in block) {
          final s = csv.rows[r][c].trim();
          if (!_isMissing(s)) counts[s] = (counts[s] ?? 0) + 1;
        }
        String? best;
        for (final MapEntry(:key, :value) in counts.entries) {
          if (best == null || value > counts[best]!) best = key;
        }
        return best ?? '';
      default:
        final xs = <double>[];
        for (final r in block) {
          final v = csv.number(r, c);
          if (v == null) {
            throw FormatException(
              'Column "${csv.header[c]}" has text ("${csv.rows[r][c]}"), '
              'which "${a.name}" can\'t aggregate',
            );
          }
          if (!v.isNaN) xs.add(v);
        }
        if (xs.isEmpty) return '';
        return _num(switch (a) {
          StepAggregate.mean => xs.fold(0.0, (s, x) => s + x) / xs.length,
          StepAggregate.sum => xs.fold(0.0, (s, x) => s + x),
          StepAggregate.min => xs.reduce(math.min),
          StepAggregate.max => xs.reduce(math.max),
          _ => xs.any((x) => x != 0) ? 1.0 : 0.0,
        });
    }
  }

  @override
  Map<String, Object?> toJson() => {
    'kind': kind,
    'factor': factor,
    if (how != StepAggregate.first) 'how': how.name,
    if (columns.isNotEmpty)
      'columns': {
        for (final MapEntry(:key, :value) in columns.entries) key: value.name,
      },
    if (within.isNotEmpty) 'within': within,
  };
}

/// Windows around events: each row where [at] is nonzero starts a segment
/// of the participant's rows from [before] rows before it to [after] rows
/// after it (null: up to the next segment). Where windows overlap, the
/// later event's segment takes the rows. Rows in no segment are dropped.
/// [column] numbers the participant's segments from 1; [offsetColumn], if
/// given, holds each row's position relative to its event (0 at the
/// event, negative before it).
final class SegmentStep extends DocStep {
  SegmentStep({
    required this.at,
    this.before = 0,
    this.after,
    this.column = 'segment',
    this.offsetColumn,
  }) {
    if (before < 0) throw ArgumentError.value(before, 'before', 'negative');
    if (after case final a? when a < 0) {
      throw ArgumentError.value(a, 'after', 'negative');
    }
  }

  final Expr at;
  final int before;
  final int? after;
  final String column;
  final String? offsetColumn;

  @override
  String get kind => 'segment';

  @override
  Set<String> get reads => at.variables;

  @override
  List<String> columnsAfter(List<String> header) => [
    ...header,
    if (!header.contains(column)) column,
    if (offsetColumn case final o? when !header.contains(o)) o,
  ];

  @override
  String get summary =>
      'at $at, ${before == 0 ? '' : '$before before, '}'
      '${after == null ? 'to the next' : '$after after'}';

  @override
  CsvData apply(CsvData csv, StepScope scope) {
    final v = _evaluate(csv, at, scope);
    final header = columnsAfter(csv.header);
    final sc = header.indexOf(column);
    final oc = offsetColumn == null ? -1 : header.indexOf(offsetColumn!);
    final rows = <List<String>>[];
    for (final unit in _units(csv, scope.participantColumn)) {
      final onsets = [
        for (var i = 0; i < unit.length; i++)
          if (_on(v[unit[i]])) i,
      ];
      // The segment (index into onsets) of each row, later ones winning.
      final seg = List.filled(unit.length, -1);
      for (final (k, o) in onsets.indexed) {
        final end = after == null
            ? (k + 1 < onsets.length
                  ? onsets[k + 1] - before - 1
                  : unit.length - 1)
            : o + after!;
        for (
          var i = math.max(0, o - before);
          i <= math.min(end, unit.length - 1);
          i++
        ) {
          seg[i] = k;
        }
      }
      for (var i = 0; i < unit.length; i++) {
        final k = seg[i];
        if (k < 0) continue;
        final row = [
          ...csv.rows[unit[i]],
          for (var c = csv.header.length; c < header.length; c++) '',
        ];
        row[sc] = '${k + 1}';
        if (oc >= 0) row[oc] = '${i - onsets[k]}';
        rows.add(row);
      }
    }
    return CsvData(header, rows, decimalComma: csv.decimalComma);
  }

  @override
  Map<String, Object?> toJson() => {
    'kind': kind,
    'at': '$at',
    if (before != 0) 'before': before,
    'after': ?after,
    if (column != 'segment') 'column': column,
    'offset_column': ?offsetColumn,
  };
}

/// Keeps the rows where [keep] is nonzero (a missing value drops the row),
/// e.g. `round > 0` or `segment_time >= 0`.
final class FilterStep extends DocStep {
  const FilterStep(this.keep);

  final Expr keep;

  @override
  String get kind => 'filter';

  @override
  Set<String> get reads => keep.variables;

  @override
  String get summary => 'keep $keep';

  @override
  CsvData apply(CsvData csv, StepScope scope) {
    final v = _evaluate(csv, keep, scope);
    return _pick(csv, [
      for (var r = 0; r < v.length; r++)
        if (_on(v[r])) r,
    ]);
  }

  @override
  Map<String, Object?> toJson() => {'kind': kind, 'keep': '$keep'};
}

/// Z-scores [columns] in place, `(x − mean) / sd` (population sd; a
/// constant column is only centred), per participant or over everyone.
/// The mean and sd come from the rows where [fitOn] is nonzero (e.g.
/// `split == 'train'`, so test rows don't leak into the scaling), or from
/// every row. Missing cells stay missing.
final class StandardiseStep extends DocStep {
  StandardiseStep({
    required List<String> columns,
    this.perParticipant = true,
    this.fitOn,
  }) : columns = List.unmodifiable(columns);

  final List<String> columns;
  final bool perParticipant;
  final Expr? fitOn;

  @override
  String get kind => 'standardise';

  @override
  Set<String> get reads => {...columns, ...?fitOn?.variables};

  @override
  String get summary =>
      '${columns.join(', ')}${perParticipant ? ' per participant' : ''}'
      '${fitOn == null ? '' : ', fitted where $fitOn'}';

  @override
  CsvData apply(CsvData csv, StepScope scope) {
    final cs = [for (final c in columns) _col(csv, c, 'to standardise')];
    final fit = fitOn == null ? null : _evaluate(csv, fitOn!, scope);
    final rows = [
      for (final r in csv.rows) [...r],
    ];
    final units = perParticipant
        ? _units(csv, scope.participantColumn)
        : [List.generate(csv.rows.length, (r) => r)];
    for (final c in cs) {
      for (final unit in units) {
        var n = 0, s = 0.0;
        final xs = List.filled(unit.length, double.nan);
        for (final (i, r) in unit.indexed) {
          final v = csv.number(r, c);
          if (v == null) {
            throw FormatException(
              'Column "${csv.header[c]}" has text ("${csv.rows[r][c]}")',
            );
          }
          xs[i] = v;
          if (!v.isNaN && (fit == null || _on(fit[r]))) {
            n++;
            s += v;
          }
        }
        if (n == 0) {
          throw FormatException(
            'No rows to fit "${csv.header[c]}"\'s mean and sd on',
          );
        }
        final mean = s / n;
        var ss = 0.0;
        for (final (i, r) in unit.indexed) {
          if (!xs[i].isNaN && (fit == null || _on(fit[r]))) {
            ss += (xs[i] - mean) * (xs[i] - mean);
          }
        }
        final sd = math.sqrt(ss / n);
        final scale = sd > 1e-12 ? sd : 1.0;
        for (final (i, r) in unit.indexed) {
          rows[r][c] = _num((xs[i] - mean) / scale);
        }
      }
    }
    return CsvData(csv.header, rows, decimalComma: csv.decimalComma);
  }

  @override
  Map<String, Object?> toJson() => {
    'kind': kind,
    'columns': columns,
    if (!perParticipant) 'per_participant': false,
    if (fitOn case final f?) 'fit_on': '$f',
  };
}

/// One part of a [SplitStep]: its label and share.
typedef SplitPart = ({String label, double fraction});

/// Labels rows for training and testing in [column]: with [byParticipant],
/// whole participants (shuffled with [seed], then cut by the [parts]'
/// fractions); otherwise each participant's rows in time order (the first
/// 80 % `train`, the rest `test`: no future rows in training). Every row
/// is kept.
final class SplitStep extends DocStep {
  SplitStep({
    this.column = 'split',
    List<SplitPart> parts = const [
      (label: 'train', fraction: 0.8),
      (label: 'test', fraction: 0.2),
    ],
    this.byParticipant = false,
    this.seed = 0,
  }) : parts = List.unmodifiable(parts) {
    if (this.parts.isEmpty) throw ArgumentError('A split needs a part');
    if (this.parts.any((p) => !(p.fraction >= 0))) {
      throw ArgumentError('A split\'s fractions must not be negative');
    }
    if (!(total > 0)) throw ArgumentError('A split\'s fractions sum to 0');
  }

  final String column;
  final List<SplitPart> parts;
  final bool byParticipant;
  final int seed;

  double get total => parts.fold(0.0, (s, p) => s + p.fraction);

  @override
  String get kind => 'split';

  @override
  Set<String> get reads => const {};

  @override
  List<String> columnsAfter(List<String> header) =>
      header.contains(column) ? header : [...header, column];

  @override
  String get summary =>
      '${parts.map((p) => '${p.label} ${_pct(p.fraction / total)}').join(' / ')}'
      ' by ${byParticipant ? 'participant' : 'time'}';

  static String _pct(double f) =>
      '${ExprNum.formatNumber((f * 1000).roundToDouble() / 10)}%';

  /// The label of item [i] of [n] (the parts cut [0, n) in order).
  String _label(int i, int n) {
    var cum = 0.0;
    for (final p in parts) {
      cum += p.fraction;
      if (i < (cum / total * n).round()) return p.label;
    }
    return parts.last.label;
  }

  @override
  CsvData apply(CsvData csv, StepScope scope) {
    final header = columnsAfter(csv.header);
    final c = header.indexOf(column);
    final rows = [
      for (final r in csv.rows)
        [...r, for (var k = r.length; k < header.length; k++) ''],
    ];
    final units = _units(csv, scope.participantColumn);
    if (byParticipant) {
      final order = shuffled(units.length, seed);
      for (final (i, u) in order.indexed) {
        final label = _label(i, units.length);
        for (final r in units[u]) {
          rows[r][c] = label;
        }
      }
    } else {
      for (final unit in units) {
        for (final (i, r) in unit.indexed) {
          rows[r][c] = _label(i, unit.length);
        }
      }
    }
    return CsvData(header, rows, decimalComma: csv.decimalComma);
  }

  /// 0..n-1 shuffled (Fisher–Yates) with mulberry32 seeded by [seed]: the
  /// same in Dart (native and web) and in the Python exports.
  static List<int> shuffled(int n, int seed) {
    var state = seed & 0xffffffff;
    double next() {
      state = (state + 0x6d2b79f5) & 0xffffffff;
      var t = _imul(state ^ (state >>> 15), 1 | state);
      t = ((t + _imul(t ^ (t >>> 7), 61 | t)) & 0xffffffff) ^ t;
      return ((t ^ (t >>> 14)) & 0xffffffff) / 4294967296;
    }

    final out = List.generate(n, (i) => i);
    for (var i = n - 1; i > 0; i--) {
      final j = (next() * (i + 1)).floor();
      final tmp = out[i];
      out[i] = out[j];
      out[j] = tmp;
    }
    return out;
  }

  /// 32-bit multiply (as JavaScript's `Math.imul`, unsigned), exact on the
  /// web too.
  static int _imul(int a, int b) {
    a &= 0xffffffff;
    b &= 0xffffffff;
    final lo = (a & 0xffff) * b;
    final hi = ((a >>> 16) * b) & 0xffff;
    return (lo + (hi << 16)) & 0xffffffff;
  }

  @override
  Map<String, Object?> toJson() => {
    'kind': kind,
    if (column != 'split') 'column': column,
    'parts': [
      for (final p in parts) {'label': p.label, 'fraction': p.fraction},
    ],
    if (byParticipant) 'by_participant': true,
    if (seed != 0) 'seed': seed,
  };
}
