import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import '../model.dart';
import 'transforms.dart';

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
    List<DocTransform> transforms = const [],
  }) : inputs = List.unmodifiable(inputs),
       responseColumns = List.unmodifiable(responseColumns),
       options = options == null ? null : List.unmodifiable(options),
       transforms = List.unmodifiable(transforms) {
    if (timeColumn != null && timeStepColumn != null) {
      throw ArgumentError('Give a time column or a time-step column');
    }
  }

  final List<LayoutInput> inputs;
  final List<String> responseColumns;
  final List<String>? options;
  final String? participantColumn, timeColumn, timeStepColumn, groupColumn;
  final List<DocTransform> transforms;

  /// The width of a row of `Subject.inputs`.
  int get width => inputs.fold(0, (w, i) => w + i.width);

  /// [csv] with the [transforms]' columns appended.
  CsvData transform(CsvData csv) => applyTransforms(
    csv,
    transforms,
    participantColumn: participantColumn,
    groupColumn: groupColumn,
    timeColumn: timeColumn,
  );

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
  List<Subject> subjects(CsvData data) {
    final csv = transform(data);
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

    final width = this.width;
    final nIn = inputs.length;
    final inputCols = [
      for (final input in inputs)
        [for (final name in input.columns) col(name, 'input ${input.name}')],
    ];
    final observedCols = [
      for (final input in inputs)
        input.observedColumn == null
            ? null
            : col(input.observedColumn!, 'observed flags of ${input.name}'),
    ];
    final responseCols = [
      for (final name in responseColumns) col(name, 'responses'),
    ];
    final tCol = timeColumn == null ? null : col(timeColumn!, 'time');
    final dtCol = timeStepColumn == null
        ? null
        : col(timeStepColumn!, 'time steps');

    double number(int row, int c) =>
        csv.number(row, c) ??
        (throw FormatException(
          'Row ${row + 2}, column "${header[c]}": "${csv.rows[row][c]}" is '
          'not a number',
        ));
    // A label first: "none" is a label here, not a missing value.
    int? code(int row, int c, List<String> labels) {
      final cell = csv.rows[row][c].trim();
      final i = labels.indexOf(cell);
      if (i >= 0) return i;
      if (csv.isMissing(row, c)) return null;
      final v = int.tryParse(cell);
      if (v != null && v >= 0 && v < labels.length) return v;
      throw FormatException(
        'Row ${row + 2}, column "${header[c]}": "$cell" is not one of '
        '${labels.join(', ')}',
      );
    }

    return [
      for (final MapEntry(key: id, value: rows) in participantRows(csv).entries)
        () {
          final nTime = rows.length;
          final u = Float64List(nTime * width);
          final observed = Float64List(nTime * nIn)
            ..fillRange(0, nTime * nIn, 1);
          var masked = false;
          for (var t = 0; t < nTime; t++) {
            final row = rows[t];
            var off = 0;
            for (var k = 0; k < nIn; k++) {
              final input = inputs[k];
              final w = input.width;
              final flagCol = observedCols[k];
              var seen = true;
              if (input.categories case final cats?) {
                final i = code(row, inputCols[k].single, cats);
                if (i == null) {
                  seen = false;
                } else if (input.oneHot) {
                  u[t * width + off + i] = 1;
                } else {
                  u[t * width + off] = i.toDouble();
                }
              } else {
                for (var j = 0; j < w; j++) {
                  final v = number(row, inputCols[k][j]);
                  if (v.isNaN) {
                    seen = false;
                  } else {
                    u[t * width + off + j] = v;
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
              if (!seen) {
                // Empty cells stay 0: a NaN at an unobserved trial would
                // still reach the parents' beliefs.
                observed[t * nIn + k] = 0;
                masked = true;
                if (flagCol == null) {
                  u.fillRange(t * width + off, t * width + off + w, 0);
                }
              }
              off += w;
            }
          }
          Float64List? timeSteps;
          if (dtCol != null) {
            timeSteps = Float64List.fromList([
              for (final row in rows) number(row, dtCol),
            ]);
          } else if (tCol != null) {
            final times = [for (final row in rows) number(row, tCol)];
            timeSteps = Float64List(nTime);
            for (var t = 1; t < nTime; t++) {
              timeSteps[t] = times[t] - times[t - 1];
            }
            if (nTime > 0) timeSteps[0] = nTime > 1 ? timeSteps[1] : 1.0;
          }
          Float64List? responses;
          if (responseCols.isNotEmpty) {
            final nr = responseCols.length;
            final opts = options;
            responses = Float64List(nTime * nr);
            for (var t = 0; t < nTime; t++) {
              for (var j = 0; j < nr; j++) {
                final c = responseCols[j];
                final row = rows[t];
                responses[t * nr + j] = opts != null
                    ? code(row, c, opts)?.toDouble() ?? double.nan
                    : number(row, c);
              }
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

  /// [data] (after the [transforms]) with [columns] appended: one map
  /// per participant, in the order of [subjects] (see [participantRows]),
  /// of new column name to one value per row of that participant. Missing
  /// values (NaN, or a column a participant lacks) are empty cells. The
  /// data's own columns stay, so new columns line up with other
  /// recordings.
  CsvData withColumns(CsvData data, List<Map<String, List<double>>> columns) {
    final csv = transform(data);
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

String _cell(double v) =>
    v == v.roundToDouble() && v.abs() < 1e15 ? '${v.toInt()}' : '$v';
