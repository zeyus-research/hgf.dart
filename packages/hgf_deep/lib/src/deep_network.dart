import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'config.dart';
import 'init.dart';
import 'kernels.dart';
import 'matrix.dart';
import 'optimiser.dart';
import 'state.dart';

/// A deep predictive-coding network of HGF layers with learned weights: a
/// port of pyhgf's vectorised `DeepNetwork`.
///
/// Layers are added bottom (output) first. The top layer holds the
/// predictors `x`, which are clamped on it; the bottom layer receives the
/// observations `y`. Each step predicts every layer top-down from the one
/// above (`W @ g(μ̂_parent)`), then propagates the prediction errors
/// bottom-up through the value- and volatility-level posteriors, and then
/// learns the weights from the settled beliefs.
///
/// ```dart
/// final net = DeepNetwork()
///   ..addLayer(4)                          // output
///   ..addLayer(6, coupling: CouplingFn.tanh)
///   ..addLayer(2)                          // input
///   ..weightInitialisation(const XavierInit(), seed: 1);
/// final predictions = net.fit(x, y, optimiser: Adam(1e-3));
/// ```
///
/// Everything runs in float64 and matches pyhgf's JAX backend under
/// `jax_enable_x64` (see `test/parity_test.dart`).
final class DeepNetwork {
  DeepNetwork({
    this.coupling = CouplingFn.linear,
    DeepVolatilityUpdate volatilityUpdates = DeepVolatilityUpdate.unbounded,
    double maxPosteriorPrecision = 1e10,
    double precisionClippingValue = 1e-6,
    bool predictPrecision = true,
    bool feedforwardUncertainty = false,
    this.tonicVolatility = false,
    bool meanFieldUpdates = false,
    bool carryPrecision = true,
  }) : settings = DeepSettings(
         volatilityUpdates: volatilityUpdates,
         maxPosteriorPrecision: maxPosteriorPrecision,
         precisionClippingValue: precisionClippingValue,
         predictPrecision: predictPrecision,
         feedforwardUncertainty: feedforwardUncertainty,
         carryPrecision: carryPrecision,
         meanFieldUpdates: meanFieldUpdates,
       ) {
    if (meanFieldUpdates && feedforwardUncertainty) {
      throw ArgumentError(
        'meanFieldUpdates is incompatible with feedforwardUncertainty: the '
        'mean-field prediction treats value parents as point estimates, so '
        'there is no value-coupling variance to propagate.',
      );
    }
    if (!(precisionClippingValue > 0 && precisionClippingValue < 0.5)) {
      throw ArgumentError.value(
        precisionClippingValue,
        'precisionClippingValue',
        'must be in (0, 0.5)',
      );
    }
  }

  /// The default coupling function of layers added without one.
  final Coupling coupling;

  /// Whether volatile layers carry a value-level tonic volatility ω
  /// (default −4, overridable per layer).
  final bool tonicVolatility;

  /// The network-wide update settings.
  final DeepSettings settings;

  final List<LayerConfig> _configs = [];
  List<Layer> _layers = const [];
  List<LayerState> _state = const [];
  final Workspace _ws1 = Workspace(1);
  Workspace? _wsBatch;
  List<LayerState>? _batchStates;

  /// The trajectories recorded by the last [fit] (`null` when it recorded
  /// nothing).
  DeepTrajectories? trajectories;

  /// Number of layers.
  int get nLayers => _configs.length;

  /// Node count of each layer, bottom to top.
  List<int> get layerSizes => [for (final c in _configs) c.size];

  /// The layer descriptions, bottom to top.
  List<LayerConfig> get configs => List.unmodifiable(_configs);

  /// The built layers (internal: for benchmarks and the GPU codegen).
  List<Layer> get debugLayers => _layers;

  /// The current beliefs of layer [layer] (width 1).
  LayerState state(int layer) => _state[layer];

  /// Add a layer above the existing ones (the first is the output layer).
  ///
  /// [addConstantInput] appends a bias column to this layer's weights (on
  /// by default). [fullyConnected] `false` connects one-to-one with the
  /// layer below (same size, no bias). [volatilityParent] defaults to
  /// [carryPrecision] for volatile layers and `true` otherwise.
  /// [initial] overrides initial beliefs, broadcast to every node (for
  /// example `{LayerField.precision: 2.0}`); volatility-level overrides on a
  /// layer without a volatility parent are dropped, as in pyhgf.
  ///
  /// Adding a layer rebuilds the network: weights go back to 1 (dense) or
  /// the identity (one-to-one).
  DeepNetwork addLayer(
    int size, {
    LayerKind kind = LayerKind.volatile,
    bool addConstantInput = true,
    bool fullyConnected = true,
    Coupling? coupling,
    bool? volatilityParent,
    double? tonicVolatility,
    double tonicVolatilityVol = -4.0,
    Map<LayerField, double> initial = const {},
  }) {
    if (size < 1) throw ArgumentError.value(size, 'size', 'must be positive');
    if (kind == LayerKind.categorical && _configs.isNotEmpty) {
      throw ArgumentError(
        'Categorical layers represent a single N-way choice and are only '
        'supported as the output (bottom) layer: add it first.',
      );
    }
    final isVolatile = kind == LayerKind.volatile;
    if (volatilityParent == true && isVolatile && !settings.carryPrecision) {
      throw ArgumentError(
        'A volatile layer with volatilityParent requires carryPrecision: '
        'without carried precision the layer starts every sample afresh, so '
        'there is no random walk for a volatility parent to modulate.',
      );
    }
    if (tonicVolatility != null && isVolatile && !this.tonicVolatility) {
      throw ArgumentError(
        'A tonicVolatility override on a volatile layer requires the network '
        'to carry the parameter: build it with tonicVolatility: true.',
      );
    }
    if (!fullyConnected) {
      if (addConstantInput) {
        throw ArgumentError(
          'One-to-one layers (fullyConnected: false) cannot use '
          'addConstantInput.',
        );
      }
      if (_configs.isNotEmpty && _configs.last.size != size) {
        throw ArgumentError(
          'One-to-one layers require the same size as the child layer '
          '(${_configs.last.size}), got $size.',
        );
      }
    }
    _configs.add(
      LayerConfig(
        size: size,
        kind: kind,
        addConstantInput: addConstantInput,
        fullyConnected: fullyConnected,
        coupling: coupling ?? this.coupling,
        volatilityParent:
            volatilityParent ?? (!isVolatile || settings.carryPrecision),
        tonicVolatility: isVolatile
            ? (tonicVolatility ?? (this.tonicVolatility ? -4.0 : null))
            : null,
        tonicVolatilityVol: tonicVolatilityVol,
        initial: Map.unmodifiable(initial),
      ),
    );
    _build();
    return this;
  }

  /// Rebuild every layer from its description: initial beliefs, unit
  /// weights.
  void _build() {
    final layers = <Layer>[];
    final state = <LayerState>[];
    for (var i = 0; i < _configs.length; i++) {
      final c = _configs[i];
      final s = LayerState.initial(c.size, volatility: c.volatilityParent);
      c.initial.forEach((field, value) {
        s.field(field)?.fillRange(0, c.size, value);
      });
      Float64List filled(double v) =>
          Float64List(c.size)..fillRange(0, c.size, v);
      Matrix? weights;
      if (i > 0) {
        final child = _configs[i - 1].size;
        weights = c.fullyConnected
            ? Matrix.filled(child, c.size + (c.addConstantInput ? 1 : 0), 1.0)
            : Matrix.identity(child);
      }
      layers.add(
        Layer(
          config: c,
          index: i,
          tonicVolatilityVol: filled(c.tonicVolatilityVol),
          tonicVolatility: c.tonicVolatility == null
              ? null
              : filled(c.tonicVolatility!),
          priorPrecision: Float64List.fromList(s.precision),
          weights: weights,
        ),
      );
      state.add(s);
    }
    _layers = layers;
    _state = state;
    _batchStates = null;
    trajectories = null;
  }

  /// Back to the initial beliefs and unit weights.
  void reset() => _build();

  void _requireLayers([int min = 1]) {
    if (_configs.length < min) {
      throw StateError(
        min == 1
            ? 'Add at least one layer first.'
            : 'Add at least $min layers first.',
      );
    }
  }

  /// The weight matrices, bottom to top (`null` for the bottom layer).
  /// Each is `childSize × (size[+1])`, the last column the bias. They are the
  /// network's own: changes take effect.
  List<Matrix?> get weights => [for (final l in _layers) l.weights];

  /// Replace the weights of layers `1..n` (same shapes as [weights]).
  void setWeights(List<Matrix> weights) {
    _requireLayers(2);
    if (weights.length != _layers.length - 1) {
      throw ArgumentError(
        'Expected ${_layers.length - 1} weight matrices (layers 1..n), got '
        '${weights.length}.',
      );
    }
    for (var l = 1; l < _layers.length; l++) {
      final w = _layers[l].weights!;
      final n = weights[l - 1];
      if (n.rows != w.rows || n.cols != w.cols) {
        throw ArgumentError(
          'Layer $l: expected ${w.rows} × ${w.cols} weights, got '
          '${n.rows} × ${n.cols}.',
        );
      }
    }
    for (var l = 1; l < _layers.length; l++) {
      _layers[l].weights = weights[l - 1].copy();
    }
  }

  /// Draw every weight matrix from [strategy]. The bias column is set to
  /// zero and left out of the fan-in. Each layer draws from its own stream,
  /// seeded from [seed] and its index, so adding layers above leaves the
  /// lower layers' draws unchanged. The same [seed] gives the same weights
  /// in every run.
  DeepNetwork weightInitialisation(WeightInit strategy, {int seed = 0}) {
    for (var l = 1; l < _layers.length; l++) {
      final layer = _layers[l];
      final w = layer.weights!;
      final nParents = w.cols - (layer.hasConstant ? 1 : 0);
      // Not Object.hash: its seed is randomised per process.
      final rng = math.Random(((seed & 0x7fffffff) * 31 + l) & 0x7fffffff);
      final flat = strategy.draw(nParents, w.rows, rng);
      final next = Matrix(w.rows, w.cols);
      for (var i = 0; i < w.rows; i++) {
        for (var j = 0; j < nParents; j++) {
          next.data[i * w.cols + j] = flat[i * nParents + j];
        }
      }
      layer.weights = next;
    }
    return this;
  }

  void _checkColumns(Matrix m, int expected, String name) {
    if (m.cols != expected) {
      throw ArgumentError(
        '$name has ${m.cols} column(s) but its layer has $expected node(s).',
      );
    }
  }

  /// Train on `(x, y)` one sample at a time, as pyhgf's `fit`: for each row,
  /// a prediction sweep (clamping `x` on top), an update sweep (clamping `y`
  /// at the bottom) and, with [weightUpdate], one [optimiser] step. Returns
  /// the output layer's prediction for every sample, made before it saw
  /// that sample's `y` (`samples × outputSize`).
  ///
  /// With [updatePrecisions] (default) the carried beliefs (the value-level
  /// precision and the volatility level) evolve from sample to sample;
  /// without it they go back to their values before the fit after every
  /// sample. [record] names the fields to keep at every step, in
  /// [trajectories].
  Matrix fit(
    Matrix x,
    Matrix y, {
    Optimiser? optimiser,
    LearningKind learningKind = LearningKind.precisionWeighted,
    bool weightUpdate = true,
    double timeStep = 1.0,
    bool updatePrecisions = true,
    Set<LayerField> record = const {},
  }) {
    _requireLayers(2);
    if (weightUpdate && optimiser == null) {
      throw ArgumentError('fit needs an optimiser (or weightUpdate: false).');
    }
    _checkColumns(x, _configs.last.size, 'x');
    _checkColumns(y, _configs.first.size, 'y');
    if (x.rows != y.rows) {
      throw ArgumentError(
        'x and y must have the same number of samples, got ${x.rows} and '
        '${y.rows}.',
      );
    }
    final n = x.rows;
    final out = Matrix(n, _configs.first.size);
    final template = updatePrecisions
        ? null
        : [for (final s in _state) s.copy()];
    final traj = record.isEmpty ? null : DeepTrajectories._(record, _state, n);
    final grads = _gradBuffers();
    final weights = this.weights;
    for (var t = 0; t < n; t++) {
      predictionSweep(_layers, _state, x, t, _ws1, timeStep, settings);
      updateSweep(_layers, _state, y, t, _ws1, timeStep, settings);
      if (weightUpdate) {
        for (final g in grads) {
          g?.fillRange(0, g.length, 0.0);
        }
        accumulateGradients(_layers, _state, _ws1, learningKind, grads);
        optimiser!.step(weights, grads);
      }
      out.data.setAll(t * out.cols, _state.first.expectedMean);
      if (template != null) _restoreCarried(template);
      traj?._record(t, _state);
      if (updatePrecisions) carryPrecisions(_layers, _state, settings);
    }
    trajectories = traj;
    return out;
  }

  List<Float64List?> _gradBuffers() => [
    for (final l in _layers)
      l.weights == null ? null : Float64List(l.weights!.data.length),
  ];

  /// Put the carried fields (precision, mean_vol, precision_vol) back to
  /// [template]'s.
  void _restoreCarried(List<LayerState> template) {
    for (var l = 0; l < _state.length; l++) {
      final s = _state[l];
      final t = template[l];
      s.precision.setAll(0, t.precision);
      s.meanVol?.setAll(0, t.meanVol!);
      s.precisionVol?.setAll(0, t.precisionVol!);
    }
  }

  /// The output layer's predicted mean for every row of [x], from the
  /// current network (the state is not modified). `samples × outputSize`.
  Matrix predict(Matrix x) {
    _requireLayers();
    _checkColumns(x, _configs.last.size, 'x');
    final b = x.rows;
    if (_layers.length == 1) return x.copy();
    // The expected means depend on nothing but the weights and x, so only
    // the mean chain is run.
    var a = Float64List(_configs.last.size * b);
    for (var i = 0; i < _configs.last.size; i++) {
      for (var s = 0; s < b; s++) {
        a[i * b + s] = x.data[s * x.cols + i];
      }
    }
    for (var l = _layers.length - 1; l > 0; l--) {
      final parent = _layers[l];
      final child = _layers[l - 1];
      final w = parent.weights!;
      final coupledRows = parent.size + (parent.hasConstant ? 1 : 0);
      final coupled = Float64List(coupledRows * b);
      for (var k = 0; k < parent.size * b; k++) {
        coupled[k] = parent.coupling.f(a[k]);
      }
      if (parent.hasConstant) {
        coupled.fillRange(parent.size * b, coupledRows * b, 1.0);
      }
      final next = Float64List(child.size * b);
      matMul(next, w.data, w.cols, 1, child.size, w.cols, coupled, b);
      switch (child.kind) {
        case LayerKind.volatile:
          break;
        case LayerKind.binary:
          final lo = settings.precisionClippingValue;
          for (var k = 0; k < next.length; k++) {
            next[k] = math.min(math.max(sigmoid(next[k]), lo), 1 - lo);
          }
        case LayerKind.categorical:
          for (var s = 0; s < b; s++) {
            var max = double.negativeInfinity;
            for (var i = 0; i < child.size; i++) {
              max = math.max(max, next[i * b + s]);
            }
            var sum = 0.0;
            for (var i = 0; i < child.size; i++) {
              final e = math.exp(next[i * b + s] - max);
              next[i * b + s] = e;
              sum += e;
            }
            for (var i = 0; i < child.size; i++) {
              next[i * b + s] = next[i * b + s] / sum;
            }
          }
      }
      a = next;
    }
    final out = Matrix(b, _configs.first.size);
    for (var i = 0; i < out.cols; i++) {
      for (var s = 0; s < b; s++) {
        out.data[s * out.cols + i] = a[i * b + s];
      }
    }
    return out;
  }

  Workspace _workspace(int width) {
    final ws = _wsBatch;
    if (ws != null && ws.width == width) return ws;
    return _wsBatch = Workspace(width);
  }

  /// One batch-synchronous learning step, as pyhgf's `batch_update`: every
  /// row of `(x, y)` is swept from the same state (same weights, same
  /// precisions), then the mean weight gradient drives one [optimiser] step
  /// (none when `null`) and, with [updatePrecisions], the mean change of the
  /// carried beliefs is added to the state. The batch counts as a single
  /// observation.
  ///
  /// Returns the per-sample error messages at the input layer
  /// (`samples × inputSize`), for whatever sits behind this network.
  Matrix batchUpdate(
    Matrix x,
    Matrix y, {
    Optimiser? optimiser,
    LearningKind learningKind = LearningKind.precisionWeighted,
    double timeStep = 1.0,
    bool updatePrecisions = true,
  }) {
    _requireLayers(2);
    _checkColumns(x, _configs.last.size, 'x');
    _checkColumns(y, _configs.first.size, 'y');
    if (x.rows != y.rows) {
      throw ArgumentError(
        'x and y must have the same number of samples, got ${x.rows} and '
        '${y.rows}.',
      );
    }
    final b = x.rows;
    final ws = _workspace(b);
    final states = _tiled(b);
    predictionSweep(_layers, states, x, 0, ws, timeStep, settings);
    updateSweep(_layers, states, y, 0, ws, timeStep, settings);
    final errors = Matrix(b, _configs.last.size);
    inputPredictionError(_layers, states, ws, settings, errors, 0);
    if (optimiser != null) {
      final grads = _gradBuffers();
      accumulateGradients(_layers, states, ws, learningKind, grads);
      optimiser.step(weights, grads);
    }
    if (updatePrecisions) {
      carryPrecisions(_layers, states, settings);
      for (var l = 0; l < _state.length; l++) {
        final t = _state[l];
        final s = states[l];
        _addMeanIncrement(t.precision, s.precision, b);
        if (t.meanVol != null) {
          _addMeanIncrement(t.meanVol!, s.meanVol!, b);
          _addMeanIncrement(t.precisionVol!, s.precisionVol!, b);
        }
      }
    }
    return errors;
  }

  /// `template += mean_s(batch[·][s] − template)`.
  static void _addMeanIncrement(
    Float64List template,
    Float64List batch,
    int b,
  ) {
    for (var i = 0; i < template.length; i++) {
      final t = template[i];
      var sum = 0.0;
      for (var s = 0; s < b; s++) {
        sum += batch[i * b + s] - t;
      }
      template[i] = t + sum / b;
    }
  }

  /// The batch states, re-armed from the network's state.
  List<LayerState> _tiled(int width) {
    final cached = _batchStates;
    if (cached != null && cached.first.width == width) {
      for (var l = 0; l < cached.length; l++) {
        cached[l].retile(_state[l]);
      }
      return cached;
    }
    return _batchStates = [for (final s in _state) s.tile(width)];
  }

  /// The error message the input layer received in the last step (width 1;
  /// after [fit]).
  Float64List inputPredictionErrorOfState() {
    _requireLayers(2);
    final out = Matrix(1, _configs.last.size);
    inputPredictionError(_layers, _state, _ws1, settings, out, 0);
    return out.data;
  }
}

/// Belief fields recorded at every step of a [DeepNetwork.fit].
final class DeepTrajectories {
  DeepTrajectories._(
    Set<LayerField> fields,
    List<LayerState> states,
    this.steps,
  ) : _data = {
        for (final f in fields)
          f: [
            for (final s in states)
              s.field(f) == null ? null : Float64List(steps * s.nodes),
          ],
      };

  /// Number of recorded steps.
  final int steps;
  final Map<LayerField, List<Float64List?>> _data;

  /// The recorded fields.
  Iterable<LayerField> get fields => _data.keys;

  /// Field [field] of layer [layer], `steps × nodes` row-major (`null` for
  /// a volatility field the layer lacks).
  Float64List? of(LayerField field, int layer) => _data[field]![layer];

  void _record(int t, List<LayerState> states) {
    _data.forEach((f, perLayer) {
      for (var l = 0; l < states.length; l++) {
        final dst = perLayer[l];
        if (dst == null) continue;
        final src = states[l].field(f)!;
        dst.setAll(t * src.length, src);
      }
    });
  }
}
