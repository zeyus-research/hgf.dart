import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_deep/datasets.dart'
    show Standardiser, accuracy, permutation, r2;
import 'package:hgf_deep/hgf_deep.dart';

import 'deep_document.dart';
import 'steps.dart' show StepScope;
import 'transforms.dart';

/// The columns a deep network appends to the table: one per target node
/// (`<id>_prediction`, `<id>_prediction_k`), or per category of a
/// categorical target (`<id>_p_<label>`).
List<String> deepPredictionColumns(DocDeep d) {
  final n = d.targetLayer!.size;
  if (d.categories case final cats?) {
    return [for (final c in cats) '${d.id}_p_$c'];
  }
  if (n == 1) return ['${d.id}_prediction'];
  return [for (var k = 0; k < n; k++) '${d.id}_prediction_$k'];
}

/// What a deep network reads from a table (the data after the clock, the
/// steps and the transforms): the rows that hold every column it reads, in
/// table order, their features [x] (raw, not standardised) and targets [y]
/// (one-hot over [DocDeep.categories] for a label column), which of them
/// it trains on, and each row's participant.
final class DeepData {
  DeepData._(this.x, this.y, this.rows, this.train, this.participants);

  /// Reads [table] as [deep] says. [scope] gives the participant column
  /// (a continuous network filters each participant on its own) and what
  /// [DocDeepTraining.trainOn] may read (`lag`, `mate`, …).
  factory DeepData.read(DocDeep deep, CsvData table, {StepScope? scope}) =>
      DeepData.columns(
        table,
        features: deep.features,
        targets: deep.targets,
        categories: deep.categories,
        trainOn: deep.isContinuous ? null : deep.training.trainOn,
        scope: scope,
        label: 'Deep network "${deep.label}"',
      );

  /// Reads [features] and [targets] (a label column with [categories]) of
  /// [table], training where [trainOn] is nonzero (every row without it).
  factory DeepData.columns(
    CsvData table, {
    List<String> features = const [],
    required List<String> targets,
    List<String>? categories,
    Expr? trainOn,
    StepScope? scope,
    String label = 'The deep network',
  }) {
    int column(String name) {
      final c = table.header.indexOf(name);
      if (c < 0) {
        throw FormatException('$label reads "$name", which is not a column');
      }
      return c;
    }

    final fx = [for (final f in features) column(f)];
    final fy = [for (final t in targets) column(t)];
    final cats = categories;
    final on = trainOn == null ? null : evaluateOn(table, trainOn, scope);
    final pc = switch (scope?.participantColumn) {
      final p? => table.header.indexOf(p),
      null => -1,
    };
    final xs = <double>[], ys = <double>[], rows = <int>[];
    final train = <bool>[], who = <String>[];
    final x = List<double>.filled(fx.length, 0);
    final width = cats?.length ?? fy.length;
    final y = List<double>.filled(width, 0);
    rowLoop:
    for (var r = 0; r < table.rows.length; r++) {
      for (final (j, c) in fx.indexed) {
        final v = table.number(r, c);
        if (v == null || !v.isFinite) continue rowLoop;
        x[j] = v;
      }
      if (cats != null) {
        final k = cats.indexOf(table.rows[r][fy.single].trim());
        if (k < 0) continue;
        for (var j = 0; j < width; j++) {
          y[j] = j == k ? 1 : 0;
        }
      } else {
        for (final (j, c) in fy.indexed) {
          final v = table.number(r, c);
          if (v == null || !v.isFinite) continue rowLoop;
          y[j] = v;
        }
      }
      xs.addAll(x);
      ys.addAll(y);
      rows.add(r);
      train.add(on == null || (!on[r].isNaN && on[r] != 0));
      who.add(pc < 0 ? '' : table.rows[r][pc].trim());
    }
    return DeepData._(
      Matrix(rows.length, fx.length, Float64List.fromList(xs)),
      Matrix(rows.length, width, Float64List.fromList(ys)),
      rows,
      train,
      who,
    );
  }

  final Matrix x, y;

  /// The table rows read, in order (row `i` of [x] and [y] is table row
  /// `rows[i]`).
  final List<int> rows;

  /// Whether row `i` is a training row.
  final List<bool> train;

  /// Row `i`'s participant (`''` without a participant column).
  final List<String> participants;

  int get length => rows.length;

  /// The indices (into [rows]) of the training rows, and of the others.
  late final List<int> trainIndices = [
    for (var i = 0; i < length; i++)
      if (train[i]) i,
  ];
  late final List<int> testIndices = [
    for (var i = 0; i < length; i++)
      if (!train[i]) i,
  ];
}

/// [e] evaluated on every row of [table] (NaN where missing), with the
/// transforms' data functions.
List<double> evaluateOn(CsvData table, Expr e, StepScope? scope) {
  final t = applyTransforms(
    table,
    [DocTransform.of('\u0000on', e)],
    participantColumn: scope?.participantColumn,
    groupColumn: scope?.groupColumn,
    timeColumn: scope?.timeColumn,
  );
  final c = t.header.length - 1;
  return [for (var r = 0; r < t.rows.length; r++) t.number(r, c) ?? double.nan];
}

Matrix _pick(Matrix m, List<int> rows) {
  final out = Matrix(rows.length, m.cols);
  for (final (i, r) in rows.indexed) {
    out.data.setRange(i * m.cols, (i + 1) * m.cols, m.data, r * m.cols);
  }
  return out;
}

/// How well a deep network predicts: accuracy for binary and categorical
/// targets, R² for continuous ones.
typedef DeepScore = ({String name, double value});

/// A trained (or filtered) deep network and what it predicts for every row
/// it read.
final class DeepResult {
  DeepResult._({
    required this.deep,
    required this.network,
    required this.scale,
    required this.data,
    required this.predictions,
    required this.epochScores,
    required this.trainScore,
    required this.testScore,
    required this.key,
  });

  final DocDeep deep;

  /// The trained network (for a continuous network: the last participant's
  /// filter).
  final DeepNetwork network;

  /// The features' standardiser, fitted on the training rows.
  final Standardiser? scale;
  final DeepData data;

  /// One row per row of [data] (`data.length × predictionColumns`): after
  /// training, the trained network's prediction; for a continuous network,
  /// the prediction made before that row was seen.
  final Matrix predictions;

  /// The training rows' score after each pass.
  final List<DeepScore> epochScores;

  /// The score on the training rows and on the others (`null` when there
  /// are none, and for a continuous network, which does not train).
  final DeepScore? trainScore, testScore;

  /// What the network was trained on ([DeepTraining.keyOf]).
  final String key;

  List<String> get columns => deepPredictionColumns(deep);

  /// The prediction columns over a table of [tableRows] rows: NaN on the
  /// rows the network did not read.
  Map<String, Float64List> columnsFor(int tableRows) {
    final out = {
      for (final c in columns)
        c: Float64List(tableRows)..fillRange(0, tableRows, double.nan),
    };
    final names = columns;
    for (final (i, r) in data.rows.indexed) {
      for (final (j, n) in names.indexed) {
        out[n]![r] = predictions.at(i, j);
      }
    }
    return out;
  }

  /// [table] (the table the network read) with the prediction columns
  /// appended.
  CsvData appendTo(CsvData table) {
    final names = columns;
    for (final n in names) {
      if (table.header.contains(n)) {
        throw FormatException('The data already has a column "$n"');
      }
    }
    final at = {for (final (i, r) in data.rows.indexed) r: i};
    return CsvData(
      [...table.header, ...names],
      [
        for (final (r, row) in table.rows.indexed)
          [
            ...row,
            for (var j = 0; j < names.length; j++)
              switch (at[r]) {
                final i? => _cell(predictions.at(i, j)),
                null => '',
              },
          ],
      ],
      decimalComma: false,
    );
  }

  /// The trained network, the standardiser and the columns as JSON (what
  /// the exported Dart scripts save).
  Map<String, Object?> toJson() => {
    'network': network.toJson(),
    'scale': scale?.toJson(),
    'features': deep.features,
    'targets': deep.targets,
    'categories': deep.categories,
  };
}

String _cell(double v) => v.isNaN ? '' : '$v';

/// Trains a [DocDeep] on a table a piece at a time ([advance]), so a UI
/// can show progress and stay responsive; [run] does it all at once.
///
/// A layered network is built, its weights initialised
/// ([DocDeepTraining.init], seeded), the features standardised on the
/// training rows, and it is trained on the training rows for
/// [DocDeepTraining.epochs] passes, one row at a time (`fit`) or in
/// minibatches (`batchUpdate`), in a new order each pass when shuffling
/// (seeded by the pass number, from 1). Then it predicts every row it read.
/// This is what the exported scripts do, and gives the same numbers.
///
/// A continuous network filters each participant's rows, in order, with a
/// fresh network each (it does not learn).
final class DeepTraining {
  DeepTraining(this.deep, CsvData table, {StepScope? scope})
    : data = DeepData.read(deep, table, scope: scope),
      key = keyOf(deep, table, scope: scope) {
    final t = deep.training;
    final net = deep.build();
    _net = net;
    if (deep.isContinuous) {
      final groups = <String, List<int>>{};
      for (var i = 0; i < data.length; i++) {
        (groups[data.participants[i]] ??= []).add(i);
      }
      _groups = groups.values.toList();
      _total = data.length;
      _predictions = Matrix(data.length, deep.targetLayer!.size);
      return;
    }
    final train = data.trainIndices;
    if (train.isEmpty) {
      throw FormatException(
        'Deep network "${deep.label}" has no rows to train on',
      );
    }
    if (t.standardise) {
      _scale = Standardiser.fit(_pick(data.x, train));
      _x = _scale!.apply(data.x);
    } else {
      _x = data.x;
    }
    if (t.init.strategy case final s?) {
      net.weightInitialisation(s, seed: t.seed);
    }
    _optimiser = t.optimiserObject;
    _synaptic = t.learningKind == LearningKind.synapticUncertainty
        ? SynapticUncertainty(window: t.synapticWindow)
        : null;
    _total = t.epochs * train.length;
  }

  /// Trains [deep] on [table] at once.
  static DeepResult run(DocDeep deep, CsvData table, {StepScope? scope}) =>
      (DeepTraining(deep, table, scope: scope)..advance()).result!;

  /// What [deep] trained on [table] depends on: its document part and
  /// the cells it reads (equal keys train alike).
  static String keyOf(DocDeep deep, CsvData table, {StepScope? scope}) {
    final cols = <int>{
      for (final c in [
        ...deep.features,
        ...deep.targets,
        ...?deep.training.trainOn?.variables,
        ?scope?.participantColumn,
        if (deep.training.trainOn != null) ...[
          ?scope?.groupColumn,
          ?scope?.timeColumn,
        ],
      ])
        if (table.header.indexOf(c) case final i when i >= 0) i,
    }.toList()..sort();
    var h = 0;
    for (final row in table.rows) {
      for (final c in cols) {
        h = Object.hash(h, row[c]);
      }
    }
    return jsonEncode([deep.toJson(), table.rows.length, cols, h]);
  }

  final DocDeep deep;
  final DeepData data;
  final String key;

  late DeepNetwork _net;
  Standardiser? _scale;
  late final Matrix _x;
  Optimiser? _optimiser;
  SynapticUncertainty? _synaptic;
  late final List<List<int>> _groups;
  late Matrix _predictions;

  int _total = 0, _done = 0;
  int _epoch = 0;

  /// The current pass's rows (indices into [data]) and how far into them.
  List<int>? _order;
  int _at = 0;

  // Continuous: which participant and how far.
  int _group = 0;

  final _scores = <DeepScore>[];
  DeepResult? _result;

  /// Rows of work in all (passes × training rows, or rows filtered).
  int get total => _total;

  /// Rows of work done.
  int get done => _done;
  double get progress => _total == 0 ? 1 : _done / _total;

  /// The pass under way (from 1), or the last when done.
  int get epoch => _epoch;
  bool get isDone => _result != null;

  /// The training rows' score after each finished pass.
  List<DeepScore> get epochScores => List.unmodifiable(_scores);

  /// The result, once [isDone].
  DeepResult? get result => _result;

  DeepScore _score(Matrix p, Matrix y) =>
      deep.targetLayer!.kind == LayerKind.volatile
      ? (name: 'R²', value: r2(p, y))
      : (name: 'accuracy', value: accuracy(p, _labels(y)));

  static List<int> _labels(Matrix y) => [
    for (var i = 0; i < y.rows; i++)
      if (y.cols == 1)
        y.at(i, 0) > 0.5 ? 1 : 0
      else
        () {
          var best = 0;
          for (var j = 1; j < y.cols; j++) {
            if (y.at(i, j) > y.at(i, best)) best = j;
          }
          return best;
        }(),
  ];

  /// Works through about [rows] rows (at least one call's worth: a pass of
  /// a synaptic-uncertainty network, a minibatch, or a chunk of rows), and
  /// returns whether it is done.
  bool advance([int rows = 1 << 52]) {
    if (_result != null) return true;
    var budget = rows;
    if (deep.isContinuous) {
      while (budget > 0 && _group < _groups.length) {
        final g = _groups[_group];
        if (_at == 0 && _group > 0) _net = deep.build();
        final n = math.min(budget, g.length - _at);
        final part = g.sublist(_at, _at + n);
        final p = _net.inputData(_pick(data.y, part));
        for (final (k, i) in part.indexed) {
          _predictions.data.setRange(
            i * _predictions.cols,
            (i + 1) * _predictions.cols,
            p.data,
            k * p.cols,
          );
        }
        _at += n;
        _done += n;
        budget -= n;
        if (_at == g.length) {
          _group++;
          _at = 0;
        }
      }
      _epoch = 1;
      if (_group == _groups.length) _finish(_predictions);
      return isDone;
    }
    final t = deep.training;
    final train = data.trainIndices;
    while (budget > 0 && _epoch <= t.epochs) {
      if (_order == null) {
        _epoch++;
        if (_epoch > t.epochs) break;
        _order = t.shuffle
            ? [
                for (final k in permutation(train.length, seed: _epoch))
                  train[k],
              ]
            : train;
        _at = 0;
      }
      final order = _order!;
      final int n;
      if (t.batchSize case final b?) {
        // Whole minibatches (as `minibatches` cuts them).
        final batches = math.max(1, (budget / b).floor());
        n = math.min(batches * b, order.length - _at);
        for (var s = _at; s < _at + n; s += b) {
          final idx = order.sublist(s, math.min(s + b, _at + n));
          _net.batchUpdate(
            _pick(_x, idx),
            _pick(data.y, idx),
            optimiser: _optimiser,
            learningKind: t.learningKind,
            synapticUncertainty: _synaptic,
          );
        }
      } else {
        // Synaptic learning keeps its importance within one fit call: a
        // whole pass at once.
        n = _synaptic != null
            ? order.length - _at
            : math.min(budget, order.length - _at);
        final idx = order.sublist(_at, _at + n);
        _net.fit(
          _pick(_x, idx),
          _pick(data.y, idx),
          optimiser: _optimiser,
          learningKind: t.learningKind,
          synapticUncertainty: _synaptic,
        );
      }
      _at += n;
      _done += n;
      budget -= n;
      if (_at >= order.length) {
        _order = null;
        _scores.add(
          _score(_net.predict(_pick(_x, train)), _pick(data.y, train)),
        );
      }
    }
    if (_epoch > t.epochs) {
      _epoch = t.epochs;
      _finish(_net.predict(_x));
    }
    return isDone;
  }

  void _finish(Matrix p) {
    DeepScore? train, test;
    if (!deep.isContinuous) {
      final tr = data.trainIndices, te = data.testIndices;
      train = _score(_pick(p, tr), _pick(data.y, tr));
      if (te.isNotEmpty) test = _score(_pick(p, te), _pick(data.y, te));
    }
    _result = DeepResult._(
      deep: deep,
      network: _net,
      scale: _scale,
      data: data,
      predictions: p,
      epochScores: epochScores,
      trainScore: train,
      testScore: test,
      key: key,
    );
  }
}
