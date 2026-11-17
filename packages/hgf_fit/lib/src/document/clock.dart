import 'dart:math' as math;

import 'package:hgf/hgf.dart';

import 'transforms.dart';

/// One kind of event in an event table: the rows [when] picks (nonzero;
/// null: every row) and the columns it [hold]s, each an expression over the
/// event's row (a column's own text is copied, so `value` keeps its labels).
/// Each held value lasts until the stream's next event for the same
/// participant: a press is held until the next press or release.
final class ClockStream {
  ClockStream({this.when, required Map<String, Expr> hold})
    : hold = Map.unmodifiable(hold);

  /// [hold] and [when] parsed from text.
  factory ClockStream.parse({
    String? when,
    required Map<String, String> hold,
  }) => ClockStream(
    when: when == null ? null : Expr.parse(when),
    hold: {
      for (final MapEntry(:key, :value) in hold.entries) key: Expr.parse(value),
    },
  );

  final Expr? when;
  final Map<String, Expr> hold;

  Map<String, Object?> toJson() => {
    if (when case final w?) 'when': '$w',
    'hold': {for (final MapEntry(:key, :value) in hold.entries) key: '$value'},
  };

  factory ClockStream.fromJson(Map<String, Object?> json) => ClockStream.parse(
    when: json['when'] as String?,
    hold: (json['hold'] as Map).cast<String, String>(),
  );
}

/// Events onto a shared clock: an event table (a row per event, with its
/// time in [onsetColumn] on a clock every participant shares, e.g. LSL
/// time) becomes one row per participant per step of [step] (in the
/// onset's units), the [streams]' columns held from each event until the
/// stream's next one (zero-order hold; empty before a participant's first
/// event). The grid's times are whole multiples of [step], so outputs line
/// up with other recordings on the same clock, and every participant of a
/// session (the [sessionColumn]; default: everyone) shares the session's
/// grid, from its first event to its last. Events shorter than a step
/// between two grid times are not seen.
///
/// [keep] (over the grid's columns) drops the rows where it is zero or
/// empty, e.g. `missing(round) == 0` to keep the game rounds.
///
/// The grid has the participant column, the [sessionColumn], the time
/// column (the document's time column) and the streams' columns.
final class DocClock {
  DocClock({
    required this.onsetColumn,
    required this.step,
    this.sessionColumn,
    List<ClockStream> streams = const [],
    this.keep,
  }) : streams = List.unmodifiable(streams) {
    if (!(step > 0) || !step.isFinite) {
      throw ArgumentError.value(step, 'step', 'must be positive');
    }
  }

  final String onsetColumn;
  final double step;
  final String? sessionColumn;
  final List<ClockStream> streams;
  final Expr? keep;

  /// The held columns' names, in order.
  List<String> get heldColumns => [for (final s in streams) ...s.hold.keys];

  /// The grid's columns, given the data layout's participant and time
  /// columns.
  List<String> columns({
    String? participantColumn,
    required String timeColumn,
  }) => [?participantColumn, ?sessionColumn, timeColumn, ...heldColumns];

  /// Whether [csv] is on the clock already: it has the [timeColumn] and no
  /// [onsetColumn] (simulated data, a saved grid).
  bool isOnClock(CsvData csv, String timeColumn) =>
      csv.header.contains(timeColumn) && !csv.header.contains(onsetColumn);

  /// The variables the clock reads from the event table.
  Set<String> get reads => {
    onsetColumn,
    ?sessionColumn,
    for (final s in streams) ...{
      ...?s.when?.variables,
      for (final e in s.hold.values) ...e.variables,
    },
  };

  Map<String, Object?> toJson() => {
    'onset_column': onsetColumn,
    'step': step,
    'session_column': ?sessionColumn,
    'streams': [for (final s in streams) s.toJson()],
    if (keep case final k?) 'keep': '$k',
  };

  factory DocClock.fromJson(Map<String, Object?> json) => DocClock(
    onsetColumn: json['onset_column'] as String,
    step: (json['step'] as num).toDouble(),
    sessionColumn: json['session_column'] as String?,
    streams: [
      for (final s in (json['streams'] as List? ?? const []))
        ClockStream.fromJson((s as Map).cast<String, Object?>()),
    ],
    keep: json['keep'] == null ? null : Expr.parse(json['keep'] as String),
  );

  /// [events] on the grid (see [DocClock]); [participantColumn] splits the
  /// events into participants (none: one), [timeColumn] names the grid's
  /// times.
  CsvData resample(
    CsvData events, {
    String? participantColumn,
    required String timeColumn,
  }) {
    final out = columns(
      participantColumn: participantColumn,
      timeColumn: timeColumn,
    );
    if (out.toSet().length != out.length) {
      throw FormatException(
        'The clock\'s columns must have different names: ${out.join(', ')}',
      );
    }
    // Every stream's selector and held values, evaluated per event row.
    final tmp = <DocTransform>[];
    for (final (i, s) in streams.indexed) {
      if (s.when case final w?) tmp.add(DocTransform.of('\u0000when$i', w));
      for (final MapEntry(:key, :value) in s.hold.entries) {
        tmp.add(DocTransform.of('\u0000$i\u0000$key', value));
      }
    }
    final CsvData t;
    try {
      t = applyTransforms(events, tmp, participantColumn: participantColumn);
    } on FormatException catch (e) {
      throw FormatException(
        'The clock: ${e.message.replaceAll('\u0000', ' ')}',
      );
    }
    int col(String name, String what) {
      final c = t.header.indexOf(name);
      if (c < 0) throw FormatException('No column "$name" ($what) in the data');
      return c;
    }

    final oc = col(onsetColumn, 'event times');
    final pc = participantColumn == null
        ? null
        : col(participantColumn, 'participants');
    final sc = sessionColumn == null ? null : col(sessionColumn!, 'sessions');
    final whenCols = [
      for (final (i, s) in streams.indexed)
        s.when == null ? null : t.header.indexOf('\u0000when$i'),
    ];
    final holdCols = [
      for (final (i, s) in streams.indexed)
        [for (final k in s.hold.keys) t.header.indexOf('\u0000$i\u0000$k')],
    ];

    // Per session, its participants (first appearance) and time range; per
    // participant and stream, the events in onset order.
    final sessions = <String, ({List<String> people, List<double> range})>{};
    final sessionOf = <String, String>{};
    final byStream = <String, List<List<(double, int, List<String>)>>>{};
    for (var r = 0; r < t.rows.length; r++) {
      final row = t.rows[r];
      final onset = t.number(r, oc);
      if (onset == null || onset.isNaN) continue;
      final p = pc == null ? '' : row[pc].trim();
      final ses = sc == null ? '' : row[sc].trim();
      var any = false;
      for (var i = 0; i < streams.length; i++) {
        final wc = whenCols[i];
        if (wc != null) {
          final v = t.number(r, wc);
          if (v == null || v.isNaN || v == 0) continue;
        }
        any = true;
        (byStream[p] ??= [
          for (final _ in streams) [],
        ])[i].add((onset, r, [for (final c in holdCols[i]) row[c]]));
      }
      if (!any) continue;
      final known = sessionOf[p];
      if (known != null && known != ses) {
        throw FormatException(
          'Participant "$p" is in sessions "$known" and "$ses"; give each '
          'session its own participant ids',
        );
      }
      sessionOf[p] = ses;
      final s = sessions[ses] ??= (people: [], range: [onset, onset]);
      if (!s.people.contains(p)) s.people.add(p);
      s.range[0] = math.min(s.range[0], onset);
      s.range[1] = math.max(s.range[1], onset);
    }

    final digits = _decimals(step);
    final rows = <List<String>>[];
    for (final MapEntry(key: ses, value: s) in sessions.entries) {
      final k0 = (s.range[0] / step).floor(), k1 = (s.range[1] / step).ceil();
      final times = [
        for (var k = k0; k <= k1; k++)
          double.parse((k * step).toStringAsFixed(digits)),
      ];
      for (final p in s.people) {
        final evs = [
          for (final e in byStream[p]!)
            // Stable: events at the same time keep their order.
            e..sort((a, b) {
              final c = a.$1.compareTo(b.$1);
              return c != 0 ? c : a.$2.compareTo(b.$2);
            }),
        ];
        final at = List.filled(streams.length, -1);
        for (final time in times) {
          final row = <String>[
            if (participantColumn != null) p,
            if (sessionColumn != null) ses,
            ExprNum.formatNumber(time),
          ];
          for (var i = 0; i < streams.length; i++) {
            final e = evs[i];
            while (at[i] + 1 < e.length &&
                e[at[i] + 1].$1 <= time + step * 1e-9) {
              at[i]++;
            }
            row.addAll(
              at[i] < 0 ? List.filled(holdCols[i].length, '') : e[at[i]].$3,
            );
          }
          rows.add(row);
        }
      }
    }
    var grid = CsvData(out, rows, decimalComma: events.decimalComma);
    if (keep case final k?) {
      final CsvData kept;
      try {
        kept = applyTransforms(grid, [DocTransform.of('\u0000keep', k)]);
      } on FormatException catch (e) {
        throw FormatException('The clock\'s keep: ${e.message}');
      }
      final kc = kept.header.length - 1;
      grid = CsvData(out, [
        for (var r = 0; r < kept.rows.length; r++)
          if (kept.number(r, kc) case final v? when !v.isNaN && v != 0)
            grid.rows[r],
      ], decimalComma: events.decimalComma);
    }
    return grid;
  }
}

/// Decimals to print multiples of [step] with (at most 9).
int _decimals(double step) {
  for (var d = 0; d < 9; d++) {
    final s = step * math.pow(10, d);
    if ((s - s.roundToDouble()).abs() < 1e-9 * math.max(1, s)) return d;
  }
  return 9;
}
