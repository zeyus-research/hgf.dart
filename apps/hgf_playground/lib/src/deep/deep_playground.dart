import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:hgf_deep/datasets.dart';
import 'package:hgf_deep/hgf_deep.dart';

import 'deep_spec.dart';

/// Scores after one epoch.
typedef EpochRecord = ({
  double trainLoss,
  double trainScore,
  double testLoss,
  double testScore,
});

/// The state of the deep network page: the task, the network built from a
/// [DeepSpec], training in time-budgeted chunks, the history of scores, the
/// decision map, and a traced sample.
///
/// Training runs on the UI thread in slices of [frameBudget] (each a few
/// rows of [DeepNetwork.fit]), so the page stays responsive at any network
/// size; the owner calls [tick] once per frame while [playing].
class DeepPlayground extends ChangeNotifier {
  DeepPlayground({
    DeepTask task = DeepTask.moons,
    DeepSpec spec = const DeepSpec(),
    this.frameBudget = const Duration(milliseconds: 6),
  }) : _task = task,
       _spec = spec {
    _loadData();
    _rebuild(keepWeights: false);
  }

  /// The training time spent per [tick].
  final Duration frameBudget;

  DeepTask get task => _task;
  DeepTask _task;

  DeepSpec get spec => _spec;
  DeepSpec _spec;

  /// The network being trained (its beliefs are the carried ones).
  DeepNetwork get network => _net;
  late DeepNetwork _net;

  late Optimiser _optimiser;

  late Dataset train;
  late Dataset test;

  /// Completed epochs.
  int get epoch => _history.length;

  /// Samples of the current epoch already learned.
  int get row => _row;
  int _row = 0;

  /// Scores of every completed epoch.
  List<EpochRecord> get history => List.unmodifiable(_history);
  final List<EpochRecord> _history = [];

  /// Bumped whenever the weights or the history change (for repaints).
  int get revision => _revision;
  int _revision = 0;

  bool get playing => _playing;
  bool _playing = false;

  /// Epochs to train before stopping on its own (`null`: until paused).
  int? stopAfter;

  /// Samples learned per frame while playing (`null`: as many as fit in
  /// [frameBudget]). One per frame lets you watch the traced sample's
  /// beliefs change as the network learns.
  int? get samplesPerFrame => _samplesPerFrame;
  int? _samplesPerFrame = 25;
  set samplesPerFrame(int? n) {
    if (n == _samplesPerFrame) return;
    _samplesPerFrame = n;
    notifyListeners();
  }

  /// The first layer (index in the network, 0 = output) whose beliefs went
  /// NaN, if any.
  int? get divergedLayer => _diverged;
  int? _diverged;

  /// The permutation of the training set for the current epoch.
  List<int> _order = const [];

  /// Prequential predictions of the current epoch (before each sample was
  /// learned), for its train scores.
  Matrix? _epochPredictions;
  Dataset? _epochData;

  // --- Task and spec ---------------------------------------------------------

  void setTask(DeepTask task) {
    if (task == _task) return;
    _task = task;
    _loadData();
    _rebuild(keepWeights: false);
  }

  /// Change the network. Learned weights carry over when the layer sizes
  /// are unchanged (a different activation, volatility level, update scheme
  /// or tonic volatility on the same weights); the beliefs, the optimiser
  /// and the history start afresh either way.
  void setSpec(DeepSpec spec, {bool keepWeights = true}) {
    final keep =
        keepWeights && spec.sameShapes(_spec) && spec.seed == _spec.seed;
    _spec = spec;
    _rebuild(keepWeights: keep);
  }

  /// Fresh weights and beliefs, empty history.
  void reset() => _rebuild(keepWeights: false);

  void _loadData() {
    final (:train, :test) = split(_task.data(), testFraction: 0.25, seed: 1);
    this.train = train;
    this.test = test;
    _traceSample = 0;
    _customInput = null;
  }

  void _rebuild({required bool keepWeights}) {
    final old = keepWeights ? _net.weights : null;
    _net = _spec.build(_task);
    if (old != null) {
      _net.setWeights([for (var l = 1; l < old.length; l++) old[l]!]);
    }
    _optimiser = _spec.makeOptimiser();
    _history.clear();
    _row = 0;
    _order = const [];
    _epochPredictions = null;
    _diverged = null;
    _playing = false;
    _gridStale = true;
    _revision++;
    _retrace();
    notifyListeners();
  }

  // --- Training --------------------------------------------------------------

  void play() {
    if (_playing) return;
    _playing = true;
    notifyListeners();
  }

  void pause() {
    if (!_playing) return;
    _playing = false;
    notifyListeners();
  }

  /// Train one full epoch now.
  void stepEpoch() {
    final target = epoch + 1;
    while (epoch < target) {
      _trainRows(train.length);
    }
    _afterTraining();
  }

  /// Train for up to [frameBudget]; call once per frame while [playing].
  void tick() {
    if (!_playing) return;
    final fixed = _samplesPerFrame;
    if (fixed != null) {
      _trainRows(fixed);
      if (stopAfter != null && epoch >= stopAfter!) _playing = false;
      _afterTraining();
      return;
    }
    final sw = Stopwatch()..start();
    var chunk = 1;
    while (sw.elapsed < frameBudget) {
      final before = sw.elapsedMicroseconds;
      _trainRows(chunk);
      final took = sw.elapsedMicroseconds - before;
      // Aim each chunk at ~a quarter of the budget.
      final perRow = math.max(1, took ~/ chunk);
      chunk = math.max(1, frameBudget.inMicroseconds ~/ (4 * perRow));
      if (stopAfter != null && epoch >= stopAfter!) {
        _playing = false;
        break;
      }
    }
    _afterTraining();
  }

  void _afterTraining() {
    _revision++;
    _retrace();
    notifyListeners();
  }

  /// Learn up to [n] rows of the current epoch (ending it if they run out).
  void _trainRows(int n) {
    if (_row == 0) {
      _order = permutation(train.length, seed: 1000 * _spec.seed + epoch);
      _epochData = train.subset(_order);
      _epochPredictions = Matrix(train.length, train.y.cols);
    }
    final data = _epochData!;
    final end = math.min(train.length, _row + n);
    final x = _slice(data.x, _row, end);
    final y = _slice(data.y, _row, end);
    final pred = _net.fit(x, y, optimiser: _optimiser);
    _epochPredictions!.data.setAll(_row * pred.cols, pred.data);
    _row = end;
    if (_row == train.length) _endEpoch();
  }

  static Matrix _slice(Matrix m, int a, int b) => Matrix(
    b - a,
    m.cols,
    Float64List.sublistView(m.data, a * m.cols, b * m.cols),
  );

  void _endEpoch() {
    final data = _epochData!;
    final pre = _epochPredictions!;
    final post = _net.predict(test.x);
    final cls = _task.classes > 0;
    _history.add((
      trainLoss: cls ? crossEntropy(pre, data.y) : mse(pre, data.y),
      trainScore: cls ? accuracy(pre, data.labels!) : r2(pre, data.y),
      testLoss: cls ? crossEntropy(post, test.y) : mse(post, test.y),
      testScore: cls ? accuracy(post, test.labels!) : r2(post, test.y),
    ));
    _row = 0;
    _gridStale = true;
    _diverged ??= _firstNaNLayer();
    if (_diverged != null) _playing = false;
  }

  int? _firstNaNLayer() {
    for (var l = 0; l < _net.nLayers; l++) {
      final s = _net.state(l);
      for (final f in LayerField.values) {
        final v = s.field(f);
        if (v != null && v.any((x) => x.isNaN)) return l;
      }
    }
    return null;
  }

  // --- Decision map ----------------------------------------------------------

  bool _gridStale = true;
  Matrix? _grid;

  /// Grid resolution of the decision map: coarser for big networks, whose
  /// predictions cost more.
  int get gridSize {
    final p = _spec.parameters(_task);
    return p > 200000 ? 24 : (p > 20000 ? 40 : 64);
  }

  /// The network's prediction over the input range: for 2-D tasks a
  /// `gridSize²` × outputs matrix (row-major, top row first); for
  /// regression `gridSize·4` × 1 along x.
  Matrix get grid {
    if (_gridStale || _grid == null) {
      _grid = _net.predict(_gridInputs());
      _gridStale = false;
    }
    return _grid!;
  }

  Matrix _gridInputs() {
    final (lo, hi) = _task.inputRange;
    if (_task.isRegression) {
      final n = gridSize * 4;
      return Matrix.column([
        for (var i = 0; i < n; i++) lo + (hi - lo) * i / (n - 1),
      ]);
    }
    final n = gridSize;
    final m = Matrix(n * n, 2);
    for (var r = 0; r < n; r++) {
      for (var c = 0; c < n; c++) {
        final k = r * n + c;
        m.data[2 * k] = lo + (hi - lo) * (c + 0.5) / n;
        m.data[2 * k + 1] = hi - (hi - lo) * (r + 0.5) / n;
      }
    }
    return m;
  }

  // --- Tracing one sample ----------------------------------------------------

  /// The test sample being traced (when no custom input is set).
  int get traceSample => _traceSample;
  int _traceSample = 0;
  List<double>? _customInput;

  /// The traced input: a custom point, or the test sample's.
  List<double> get traceInput => _customInput ?? test.x.row(_traceSample);

  /// The traced target: the test sample's (a custom input of a
  /// classification task takes the class the task's data has nearest it;
  /// of the sine task, sin(x)).
  List<double> get traceTarget {
    if (_customInput == null) return test.y.row(_traceSample);
    final x = _customInput!;
    if (_task.isRegression) return [math.sin(x[0])];
    var best = 0;
    var bestD = double.infinity;
    for (var i = 0; i < train.length; i++) {
      var d = 0.0;
      for (var j = 0; j < x.length; j++) {
        final e = train.x.at(i, j) - x[j];
        d += e * e;
      }
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return train.y.row(best);
  }

  /// Whether the custom input's target is borrowed from the nearest
  /// training point.
  bool get targetIsNearest => _customInput != null && !_task.isRegression;

  void traceTestSample(int i) {
    _traceSample = i.clamp(0, test.length - 1);
    _customInput = null;
    _retrace();
    notifyListeners();
  }

  void traceCustomInput(List<double> x) {
    _customInput = List.of(x);
    _retrace();
    notifyListeners();
  }

  /// The traced step of the network on [traceInput], from its current
  /// state, without changing it (or `null` before anything is traced).
  DeepTrace? get trace => _trace;
  DeepTrace? _trace;

  /// The phase shown (an index into [trace]'s steps).
  int get cursor => _cursor;
  int _cursor = 0;

  void setCursor(int i) {
    final t = _trace;
    if (t == null) return;
    _cursor = i.clamp(0, t.steps.length - 1);
    notifyListeners();
  }

  void stepCursor(int delta) => setCursor(_cursor + delta);

  /// Re-run the trace from the network's current state (a "peek": the
  /// network is restored afterwards, the optimiser copied).
  void _retrace() {
    final snap = _net.snapshot();
    final keep = _trace == null ? null : _cursor;
    _trace = _net.trace(
      traceInput,
      y: traceTarget,
      optimiser: _optimiser.copy(),
    );
    _net.restore(snap);
    _cursor = keep == null ? 0 : math.min(keep, _trace!.steps.length - 1);
  }

  /// Learn from the traced sample for real (one step of [DeepNetwork.fit]
  /// with the network's optimiser), then trace it again.
  void commitTrace() {
    _net.trace(traceInput, y: traceTarget, optimiser: _optimiser);
    _gridStale = true;
    _revision++;
    _diverged ??= _firstNaNLayer();
    _retrace();
    notifyListeners();
  }

  /// The states before the shown phase (the previous step's, or the
  /// network's own before the trace).
  List<LayerState>? get statesBefore {
    final t = _trace;
    if (t == null) return null;
    if (_cursor > 0) return t.steps[_cursor - 1].states;
    return [for (var l = 0; l < _net.nLayers; l++) _net.state(l)];
  }
}
