import 'dart:convert';
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import '../model.dart';
import 'clock.dart';
import 'steps.dart';
import 'transforms.dart';

/// The tables [DataLayout.prepared] gave, by its layouts' keys.
final _preparedBy = Expando<String>('prepared');

/// One input node's columns in a [DataLayout], in the network's input
/// order: [columns] (as many as the node takes data columns), or, with
/// [categories], one column of labels (or 0-based codes) read as the
/// category's index, or one-hot ([oneHot], a categorical node).
///
/// An empty cell is an unobserved input, unless [observedColumn] names a
/// column of 0/1 flags (see `DocInput`).
final class LayoutInput {
  LayoutInput(
    this.name, {
    required List<String> columns,
    List<String>? categories,
    this.oneHot = false,
    this.observedColumn,
  }) : columns = List.unmodifiable(columns),
       categories = categories == null ? null : List.unmodifiable(categories) {
    if (categories != null && this.columns.length != 1) {
      throw ArgumentError('Input "$name": labels come from one column');
    }
    if (oneHot && categories == null) {
      throw ArgumentError('Input "$name": one-hot needs categories');
    }
  }

  final String name;
  final List<String> columns;
  final List<String>? categories;
  final bool oneHot;
  final String? observedColumn;

  /// How many data columns the input node takes.
  int get width => oneHot ? categories!.length : columns.length;
}

/// How a long-format CSV (one row per time step) is read into one
/// `Subject` per participant: the [inputs] in the network's input order,
/// the [responseColumns] (with [options], cells hold option labels or
/// codes), the participant, time or time-step, and group columns, and the
/// [transforms] computed first. What `CompiledDocument.subjects` reads
/// with, and what the exported Dart scripts write out.
final class DataLayout {
  DataLayout({
    required List<LayoutInput> inputs,
    List<String> responseColumns = const [],
    List<String>? options,
    this.participantColumn,
    this.timeColumn,
    this.timeStepColumn,
    this.groupColumn,
    this.clock,
    List<DocStep> steps = const [],
    List<DocTransform> transforms = const [],
  }) : inputs = List.unmodifiable(inputs),
       steps = List.unmodifiable(steps),
       responseColumns = List.unmodifiable(responseColumns),
       options = options == null ? null : List.unmodifiable(options),
       transforms = List.unmodifiable(transforms) {
    if (timeColumn != null && timeStepColumn != null) {
      throw ArgumentError('Give a time column or a time-step column');
    }
    if (clock != null && timeColumn == null) {
      throw ArgumentError('A clock needs a time column (its grid\'s times)');
    }
  }

  final List<LayoutInput> inputs;
  final List<String> responseColumns;
  final List<String>? options;
  final String? participantColumn, timeColumn, timeStepColumn, groupColumn;
  final List<DocTransform> transforms;

  /// Puts event tables onto a shared clock before the [transforms].
  final DocClock? clock;

  /// The data-pipeline steps, after the [clock] and before the
  /// [transforms].
  final List<DocStep> steps;

  /// Where the [steps] run.
  StepScope get scope => (
    participantColumn: participantColumn,
    groupColumn: groupColumn,
    timeColumn: timeColumn,
  );

  /// The width of a row of `Subject.inputs`.
  int get width => inputs.fold(0, (w, i) => w + i.width);

  /// [csv] on the [clock]'s grid (unless it is on it already, or there is
  /// no clock); with a [cache], made again only when needed.
  CsvData clocked(CsvData csv, {TransformCache? cache}) {
    final c = clock;
    if (c == null || c.isOnClock(csv, timeColumn!)) return csv;
    if (cache != null) {
      return cache.clocked(
        csv,
        c,
        participantColumn: participantColumn,
        timeColumn: timeColumn!,
      );
    }
    return c.resample(
      csv,
      participantColumn: participantColumn,
      timeColumn: timeColumn!,
    );
  }

  /// [csv] on the [clock]'s grid and through the [steps]: what the
  /// [transforms] read. With a [cache], made again only when needed.
  ///
  /// A table this gave already (for this layout's clock, steps and
  /// columns) comes back as it is, so the readers ([subjects],
  /// [withColumns], …) take raw or prepared data alike.
  CsvData prepared(CsvData csv, {TransformCache? cache}) {
    if (steps.isEmpty) return clocked(csv, cache: cache);
    if (_preparedBy[csv] == _prepareKey) return csv;
    final grid = clocked(csv, cache: cache);
    final out = cache != null
        ? cache.stepped(grid, steps, scope)
        : applySteps(grid, steps, scope);
    _preparedBy[out] = _prepareKey;
    return out;
  }

  /// [table] (made from a [prepared] one, e.g. with a column filled in)
  /// marked as prepared: the readers don't run the steps on it again.
  CsvData asPrepared(CsvData table) {
    if (steps.isNotEmpty) _preparedBy[table] = _prepareKey;
    return table;
  }

  /// What [prepared] does, as text: tables it gave are tagged with it.
  late final String _prepareKey = jsonEncode([
    clock?.toJson(),
    for (final s in steps) s.toJson(),
    participantColumn,
    groupColumn,
    timeColumn,
  ]);

  /// [csv] on the [clock]'s grid, through the [steps], with the
  /// [transforms]' columns appended; a [cache] reuses what it can of the
  /// last call's work.
  CsvData transform(CsvData csv, {TransformCache? cache}) {
    final grid = prepared(csv, cache: cache);
    if (cache != null) {
      return cache.apply(
        grid,
        transforms,
        participantColumn: participantColumn,
        groupColumn: groupColumn,
        timeColumn: timeColumn,
      );
    }
    return applyTransforms(
      grid,
      transforms,
      participantColumn: participantColumn,
      groupColumn: groupColumn,
      timeColumn: timeColumn,
    );
  }

  /// The rows of each participant of [csv], in order of first appearance
  /// (rows in file order); without a participant column, one participant
  /// `''` with every row.
  Map<String, List<int>> participantRows(CsvData csv) {
    final groups = <String, List<int>>{};
    if (participantColumn case final p?) {
      final c = csv.header.indexOf(p);
      if (c < 0) {
        throw FormatException('No column "$p" (participants) in the data');
      }
      for (var r = 0; r < csv.rows.length; r++) {
        (groups[csv.rows[r][c].trim()] ??= []).add(r);
      }
    } else {
      groups[''] = [for (var r = 0; r < csv.rows.length; r++) r];
    }
    return groups;
  }

  /// The subjects in [data], one per participant (see [participantRows]),
  /// after the [transforms]. A label is matched before a missing marker
  /// (`none` may be an option), an empty cell is an unobserved input (read
  /// as 0, never NaN) or a missed response (NaN), and time steps come from
  /// [timeStepColumn], or the differences of [timeColumn] (the first row
  /// takes the second's).
  List<Subject> subjects(CsvData data, {TransformCache? cache}) {
    final csv = transform(data, cache: cache);
    if (cache != null) {
      return cache.memo(csv, 'subjects $_readKey', () => _subjects(csv));
    }
    return _subjects(csv);
  }

  /// What [subjects] reads of a table, as text: equal keys read a table
  /// alike.
  late final String _readKey = jsonEncode([
    for (final i in inputs)
      [i.name, i.columns, i.categories, i.oneHot, i.observedColumn],
    responseColumns,
    options,
    participantColumn,
    timeColumn,
    timeStepColumn,
  ]);

  List<Subject> _subjects(CsvData csv) {
    final read = reader(csv);
    final width = this.width, nIn = inputs.length;
    return [
      for (final MapEntry(key: id, value: rows) in participantRows(csv).entries)
        () {
          final nTime = rows.length;
          final u = Float64List(nTime * width);
          final observed = Float64List(nTime * nIn);
          var masked = false;
          for (var t = 0; t < nTime; t++) {
            if (read.inputs(rows[t], u, t * width, observed, t * nIn)) {
              masked = true;
            }
          }
          Float64List? timeSteps;
          if (read.dtCol case final dtCol?) {
            timeSteps = Float64List.fromList([
              for (final row in rows) read.number(row, dtCol),
            ]);
          } else if (read.tCol case final tCol?) {
            final times = [for (final row in rows) read.number(row, tCol)];
            timeSteps = Float64List(nTime);
            for (var t = 1; t < nTime; t++) {
              timeSteps[t] = times[t] - times[t - 1];
            }
            if (nTime > 0) timeSteps[0] = nTime > 1 ? timeSteps[1] : 1.0;
          }
          Float64List? responses;
          if (responseColumns.isNotEmpty) {
            final nr = responseColumns.length;
            responses = Float64List(nTime * nr);
            for (var t = 0; t < nTime; t++) {
              read.responses(rows[t], responses, t * nr);
            }
          }
          return Subject(
            inputs: u,
            observed: masked ? observed : null,
            timeSteps: timeSteps,
            responses: responses,
            id: participantColumn == null ? null : id,
          );
        }(),
    ];
  }

  /// Reads rows of [csv] (with the [transforms]' columns already there):
  /// what [subjects] reads with, one row at a time.
  LayoutReader reader(CsvData csv) => LayoutReader._(this, csv);

  /// [data] (after the [transforms]) with [columns] appended: one map
  /// per participant, in the order of [subjects] (see [participantRows]),
  /// of new column name to one value per row of that participant. Missing
  /// values (NaN, or a column a participant lacks) are empty cells. The
  /// data's own columns stay, so new columns line up with other
  /// recordings.
  CsvData withColumns(
    CsvData data,
    List<Map<String, List<double>>> columns, {
    TransformCache? cache,
  }) {
    final csv = transform(data, cache: cache);
    final names = <String>{for (final c in columns) ...c.keys}.toList();
    for (final n in names) {
      if (csv.header.contains(n)) {
        throw FormatException('The data already has a column "$n"');
      }
    }
    final rows = [
      for (final r in csv.rows) [...r, for (final _ in names) ''],
    ];
    final base = csv.header.length;
    final groups = participantRows(csv).values.toList();
    if (columns.length != groups.length) {
      throw ArgumentError(
        'The data has ${groups.length} participants, not ${columns.length}',
      );
    }
    for (final (k, rowIdx) in groups.indexed) {
      final cols = columns[k];
      for (final (j, n) in names.indexed) {
        final series = cols[n];
        if (series == null) continue;
        for (final (t, r) in rowIdx.indexed) {
          if (t < series.length && !series[t].isNaN) {
            rows[r][base + j] = _cell(series[t]);
          }
        }
      }
    }
    return CsvData([...csv.header, ...names], rows, decimalComma: false);
  }
}

/// Reads one row of a table at a time, as [DataLayout.subjects] does (see
/// [DataLayout.reader]).
final class LayoutReader {
  LayoutReader._(this.layout, this.csv) {
    final header = csv.header;
    int col(String name, String what) {
      final c = header.indexOf(name);
      if (c < 0) {
        throw FormatException(
          'No column "$name" ($what) in the data; its columns are '
          '${header.join(', ')}',
        );
      }
      return c;
    }

    inputCols = [
      for (final input in layout.inputs)
        [for (final name in input.columns) col(name, 'input ${input.name}')],
    ];
    observedCols = [
      for (final input in layout.inputs)
        input.observedColumn == null
            ? null
            : col(input.observedColumn!, 'observed flags of ${input.name}'),
    ];
    responseCols = [
      for (final name in layout.responseColumns) col(name, 'responses'),
    ];
    tCol = layout.timeColumn == null ? null : col(layout.timeColumn!, 'time');
    dtCol = layout.timeStepColumn == null
        ? null
        : col(layout.timeStepColumn!, 'time steps');
  }

  final DataLayout layout;
  final CsvData csv;
  late final List<List<int>> inputCols;
  late final List<int?> observedCols;
  late final List<int> responseCols;
  late final int? tCol, dtCol;

  double number(int row, int c) =>
      csv.number(row, c) ??
      (throw FormatException(
        'Row ${row + 2}, column "${csv.header[c]}": "${csv.rows[row][c]}" is '
        'not a number',
      ));

  // A label first: "none" is a label here, not a missing value.
  int? _code(int row, int c, List<String> labels) {
    final cell = csv.rows[row][c].trim();
    final i = labels.indexOf(cell);
    if (i >= 0) return i;
    if (csv.isMissing(row, c)) return null;
    final v = int.tryParse(cell);
    if (v != null && v >= 0 && v < labels.length) return v;
    throw FormatException(
      'Row ${row + 2}, column "${csv.header[c]}": "$cell" is not one of '
      '${labels.join(', ')}',
    );
  }

  /// [row]'s inputs into `u[uOff ..]` (the layout's width; zeros where
  /// unobserved) and their observed flags into `observed[oOff ..]` (one per
  /// input). True when an input is unobserved or has a flag column.
  bool inputs(
    int row,
    Float64List u,
    int uOff,
    Float64List observed,
    int oOff,
  ) {
    var masked = false;
    var off = uOff;
    for (var k = 0; k < layout.inputs.length; k++) {
      final input = layout.inputs[k];
      final w = input.width;
      final flagCol = observedCols[k];
      u.fillRange(off, off + w, 0);
      var seen = true;
      if (input.categories case final cats?) {
        final i = _code(row, inputCols[k].single, cats);
        if (i == null) {
          seen = false;
        } else if (input.oneHot) {
          u[off + i] = 1;
        } else {
          u[off] = i.toDouble();
        }
      } else {
        for (var j = 0; j < w; j++) {
          final v = number(row, inputCols[k][j]);
          if (v.isNaN) {
            seen = false;
          } else {
            u[off + j] = v;
          }
        }
      }
      if (flagCol != null) {
        final flag = number(row, flagCol) != 0;
        if (flag && !seen) {
          throw FormatException(
            'Row ${row + 2}: input "${input.name}" is empty but '
            'flagged observed',
          );
        }
        seen = flag;
        masked = true;
      }
      observed[oOff + k] = seen ? 1 : 0;
      if (!seen) {
        // Empty cells stay 0: a NaN at an unobserved trial would still
        // reach the parents' beliefs.
        masked = true;
        if (flagCol == null) u.fillRange(off, off + w, 0);
      }
      off += w;
    }
    return masked;
  }

  /// [row]'s responses into `out[off ..]` (option codes with options; NaN
  /// for a missed response).
  void responses(int row, Float64List out, int off) {
    final opts = layout.options;
    for (final (j, c) in responseCols.indexed) {
      out[off + j] = opts != null
          ? _code(row, c, opts)?.toDouble() ?? double.nan
          : number(row, c);
    }
  }
}

String _cell(double v) =>
    v == v.roundToDouble() && v.abs() < 1e15 ? '${v.toInt()}' : '$v';
