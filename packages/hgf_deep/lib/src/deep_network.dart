import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'config.dart';
import 'continuous.dart';
import 'init.dart';
import 'kernels.dart';
import 'matrix.dart';
import 'optimiser.dart';
import 'state.dart';
import 'trace.dart';

part 'deep_network_json.dart';

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
    bool updateInputLayer = false,
    DeepFixes? fixes,
  }) : settings = DeepSettings(
         volatilityUpdates: volatilityUpdates,
         maxPosteriorPrecision: maxPosteriorPrecision,
         precisionClippingValue: precisionClippingValue,
         predictPrecision: predictPrecision,
         feedforwardUncertainty: feedforwardUncertainty,
         carryPrecision: carryPrecision,
         meanFieldUpdates: meanFieldUpdates,
         updateInputLayer: updateInputLayer,
         fixes: fixes ?? defaultFixes,
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

  /// The corrections a network built without `fixes:` uses
  /// ([DeepFixes.recommended]). Set it to [DeepFixes.none] to reproduce
  /// pyhgf everywhere (the parity tests do), or start the program with
  /// `-DHGF_DEEP_UPSTREAM=true`.
  static DeepFixes defaultFixes =
      const bool.fromEnvironment('HGF_DEEP_UPSTREAM')
      ? DeepFixes.none
      : DeepFixes.recommended;

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
  ContinuousTopology? _topology;

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
  /// by default; never on continuous layers). [fullyConnected] `false`
  /// connects one-to-one with the layer below (same size, no bias).
  /// [volatilityParent] defaults to [carryPrecision] for volatile layers
  /// and `true` otherwise (continuous layers ignore it).
  /// [initial] overrides initial beliefs, broadcast to every node (for
  /// example `{LayerField.precision: 2.0}`); volatility-level overrides on a
  /// layer without a volatility parent are dropped, as in pyhgf.
  ///
  /// [LayerKind.continuous] layers form a filter of their own (see
  /// [inputData]) and take the continuous-only arguments:
  ///
  /// * [valueChildren] / [volatilityChildren]: the index of the existing
  ///   layer this one is the value / volatility parent of. Without either,
  ///   the layer is the value parent of the previous one.
  /// * [valueCoupling] (default 1) and [volatilityCoupling] κ₀ (default 1):
  ///   dense entries are the strength divided by this layer's size, so the
  ///   child receives the strength times the layer's average; one-to-one
  ///   entries are the strength. [volatilityFullyConnected] `false` makes κ
  ///   one-to-one (required by the unbounded update).
  /// * [tonicVolatility] ω (default −4), [tonicDrift] ρ (default 0) and
  ///   [autoconnectionStrength] λ (default 1). The observation layer (the
  ///   first) defaults to ω = 0 and λ = 0.
  ///
  /// Adding a layer rebuilds the network: weights go back to 1 (dense) or
  /// the identity (one-to-one).
  DeepNetwork addLayer(
    int size, {
    LayerKind kind = LayerKind.volatile,
    bool? addConstantInput,
    bool fullyConnected = true,
    Coupling? coupling,
    bool? volatilityParent,
    double? tonicVolatility,
    double? tonicVolatilityVol,
    Map<LayerField, double> initial = const {},
    int? valueChildren,
    int? volatilityChildren,
    double? valueCoupling,
    double? volatilityCoupling,
    bool? volatilityFullyConnected,
    double? tonicDrift,
    double? autoconnectionStrength,
  }) {
    if (size < 1) throw ArgumentError.value(size, 'size', 'must be positive');
    final continuous = kind == LayerKind.continuous;
    if (_configs.isNotEmpty && isContinuous != continuous) {
      throw ArgumentError(
        'Continuous layers cannot mix with volatile/binary/categorical layers '
        'in one network.',
      );
    }
    if (continuous) {
      return _addContinuousLayer(
        size,
        addConstantInput: addConstantInput,
        fullyConnected: fullyConnected,
        coupling: coupling,
        tonicVolatility: tonicVolatility,
        tonicVolatilityVol: tonicVolatilityVol,
        initial: initial,
        valueChildren: valueChildren,
        volatilityChildren: volatilityChildren,
        valueCoupling: valueCoupling ?? 1.0,
        volatilityCoupling: volatilityCoupling ?? 1.0,
        volatilityFullyConnected: volatilityFullyConnected ?? true,
        tonicDrift: tonicDrift,
        autoconnectionStrength: autoconnectionStrength,
      );
    }
    final misplaced = [
      if (valueChildren != null) 'valueChildren',
      if (volatilityChildren != null) 'volatilityChildren',
      if (valueCoupling != null) 'valueCoupling',
      if (volatilityCoupling != null) 'volatilityCoupling',
      if (volatilityFullyConnected != null) 'volatilityFullyConnected',
      if (tonicDrift != null) 'tonicDrift',
      if (autoconnectionStrength != null) 'autoconnectionStrength',
    ];
    if (misplaced.isNotEmpty) {
      throw ArgumentError(
        '${misplaced.join(', ')} only apply to continuous layers.',
      );
    }
    final hasConstant = addConstantInput ?? true;
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
      if (hasConstant) {
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
        addConstantInput: hasConstant,
        fullyConnected: fullyConnected,
        coupling: coupling ?? this.coupling,
        volatilityParent:
            volatilityParent ?? (!isVolatile || settings.carryPrecision),
        tonicVolatility: isVolatile
            ? (tonicVolatility ?? (this.tonicVolatility ? -4.0 : null))
            : null,
        tonicVolatilityVol: tonicVolatilityVol ?? -4.0,
        initial: Map.unmodifiable(initial),
      ),
    );
    _build();
    return this;
  }

  /// [addLayer] for a continuous layer (pyhgf's `add_layer` with
  /// `kind="continuous"` and `_resolve_continuous_children`).
  DeepNetwork _addContinuousLayer(
    int size, {
    required bool? addConstantInput,
    required bool fullyConnected,
    required Coupling? coupling,
    required double? tonicVolatility,
    required double? tonicVolatilityVol,
    required Map<LayerField, double> initial,
    required int? valueChildren,
    required int? volatilityChildren,
    required double valueCoupling,
    required double volatilityCoupling,
    required bool volatilityFullyConnected,
    required double? tonicDrift,
    required double? autoconnectionStrength,
  }) {
    if (addConstantInput ?? false) {
      throw ArgumentError(
        'Continuous layers cannot use addConstantInput: the per-node '
        'tonicDrift already provides a constant offset.',
      );
    }
    if (tonicVolatilityVol != null) {
      throw ArgumentError(
        'tonicVolatilityVol does not apply to continuous layers (they have '
        'no internal volatility level).',
      );
    }
    final existing = _configs.length;
    final valueChild = valueChildren == null && volatilityChildren == null
        ? (existing > 0 ? existing - 1 : null)
        : valueChildren;
    final volatilityChild = volatilityChildren;
    for (final (name, child) in [
      ('valueChildren', valueChild),
      ('volatilityChildren', volatilityChild),
    ]) {
      if (child != null && (child < 0 || child >= existing)) {
        throw RangeError(
          '$name=$child does not name an existing layer (the network has '
          '$existing layers).',
        );
      }
    }
    if (valueChild != null) {
      final owner = _configs.indexWhere((c) => c.valueChild == valueChild);
      if (owner >= 0) {
        throw ArgumentError(
          'Layer $valueChild already has a value parent (layer $owner).',
        );
      }
      if (!fullyConnected && _configs[valueChild].size != size) {
        throw ArgumentError(
          'One-to-one layers require the same size as the child layer '
          '(${_configs[valueChild].size}), got $size.',
        );
      }
    }
    if (volatilityChild != null) {
      final owner = _configs.indexWhere(
        (c) => c.volatilityChild == volatilityChild,
      );
      if (owner >= 0) {
        throw ArgumentError(
          'Layer $volatilityChild already has a volatility parent '
          '(layer $owner).',
        );
      }
      final childSize = _configs[volatilityChild].size;
      if (!volatilityFullyConnected && childSize != size) {
        throw ArgumentError(
          'A one-to-one volatility coupling requires equal sizes (child '
          'layer $volatilityChild has $childSize nodes, this layer $size).',
        );
      }
      if (settings.volatilityUpdates == DeepVolatilityUpdate.unbounded) {
        // The unbounded update is implemented for one volatility child per
        // node and no value child.
        if (volatilityFullyConnected && !(size == 1 && childSize == 1)) {
          throw ArgumentError(
            'The unbounded volatility update requires a one-to-one '
            'volatility coupling (volatilityFullyConnected: false). Use '
            'standard or eHGF for fully connected volatility parents.',
          );
        }
        if (valueChild != null) {
          throw ArgumentError(
            'Under the unbounded volatility update a volatility parent '
            'cannot also be a value parent. Use standard or eHGF for mixed '
            'parents.',
          );
        }
      }
    }
    // The observation layer performs no autoregression of its own and its
    // volatility exponent is zeroed (pyhgf's observation-leaf convention).
    final observation = existing == 0;
    final config = LayerConfig(
      size: size,
      kind: LayerKind.continuous,
      addConstantInput: false,
      fullyConnected: fullyConnected,
      coupling: coupling ?? this.coupling,
      volatilityParent: false,
      tonicVolatility: tonicVolatility ?? (observation ? 0.0 : -4.0),
      tonicVolatilityVol: -4.0,
      initial: Map.unmodifiable(initial),
      valueChild: valueChild,
      volatilityChild: volatilityChild,
      valueCoupling: valueCoupling,
      volatilityCoupling: volatilityCoupling,
      volatilityFullyConnected: volatilityFullyConnected,
      tonicDrift: tonicDrift ?? 0.0,
      autoconnectionStrength:
          autoconnectionStrength ?? (observation ? 0.0 : 1.0),
    );
    _configs.add(config);
    try {
      _build();
    } on ArgumentError {
      _configs.removeLast();
      _build();
      rethrow;
    }
    return this;
  }

  /// Whether this is a network of continuous layers (a filter run with
  /// [inputData]).
  bool get isContinuous =>
      _configs.isNotEmpty && _configs.first.kind == LayerKind.continuous;

  /// Reject the deep-network entry points on a continuous network.
  void _requireDeep(String method) {
    if (isContinuous) {
      throw StateError(
        '$method() is not available for continuous networks: they are '
        'filters, not trainable deep networks. Use inputData() to run '
        'observations through the network.',
      );
    }
  }

  /// One fixed coupling matrix of a continuous layer (`_fan_in_matrix`):
  /// dense entries `strength / size`, one-to-one `strength` on the
  /// diagonal.
  Matrix? _fanIn(int? child, int size, double strength, bool dense) {
    if (child == null) return null;
    final rows = _configs[child].size;
    if (dense) return Matrix.filled(rows, size, strength / size);
    final m = Matrix(rows, size);
    for (var i = 0; i < rows && i < size; i++) {
      m.data[i * size + i] = strength;
    }
    return m;
  }

  /// Rebuild every layer from its description: initial beliefs, unit
  /// weights.
  void _build() {
    if (isContinuous) {
      _buildContinuous();
      return;
    }
    _topology = null;
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

  /// [_build] for a continuous network (`_init_continuous_state`).
  void _buildContinuous() {
    final topology = ContinuousTopology(_configs);
    final layers = <Layer>[];
    final state = <LayerState>[];
    for (var i = 0; i < _configs.length; i++) {
      final c = _configs[i];
      final s = LayerState.continuous(
        c.size,
        hasVolatilityParent: topology.volatilityParentOf[i] != null,
      );
      c.initial.forEach((field, value) {
        s.field(field)?.fillRange(0, c.size, value);
      });
      Float64List filled(double v) =>
          Float64List(c.size)..fillRange(0, c.size, v);
      layers.add(
        Layer(
          config: c,
          index: i,
          tonicVolatilityVol: filled(c.tonicVolatilityVol),
          tonicVolatility: filled(c.tonicVolatility!),
          priorPrecision: Float64List.fromList(s.precision),
          weights: _fanIn(
            c.valueChild,
            c.size,
            c.valueCoupling,
            c.fullyConnected,
          ),
          volatilityWeights: _fanIn(
            c.volatilityChild,
            c.size,
            c.volatilityCoupling,
            c.volatilityFullyConnected,
          ),
          tonicDrift: filled(c.tonicDrift),
          autoconnectionStrength: filled(c.autoconnectionStrength),
        ),
      );
      state.add(s);
    }
    _topology = topology;
    _layers = layers;
    _state = state;
    _batchStates = null;
    trajectories = null;
  }

  /// Filter the observations [y] (`steps × size of layer 0`) through a
  /// continuous network, as pyhgf's `input_data`: for every row, a
  /// top-down prediction of every layer from its parents, then the row is
  /// clamped on layer 0 and the prediction errors propagate up through the
  /// value and volatility couplings. There is no learning: the couplings
  /// are parameters of the filter.
  ///
  /// [timeSteps] gives the time step of every row (default 1). [record]
  /// names the fields to keep at every step, in [trajectories]. Returns
  /// layer 0's expected mean at every step, the prediction made before
  /// that row was seen. The state carries over to the next call.
  Matrix inputData(
    Matrix y, {
    List<double>? timeSteps,
    Set<LayerField> record = const {},
  }) {
    if (!isContinuous) {
      throw StateError(
        'inputData() is only available for continuous networks; use fit() '
        'for volatile/binary/categorical networks.',
      );
    }
    _checkColumns(y, _configs.first.size, 'y');
    final n = y.rows;
    if (timeSteps != null && timeSteps.length != n) {
      throw ArgumentError(
        'timeSteps has ${timeSteps.length} entries for $n observations.',
      );
    }
    final unknown = [
      for (final f in record)
        if (_state.every((s) => s.field(f) == null)) f.pyhgfName,
    ];
    if (unknown.isNotEmpty) {
      throw ArgumentError(
        'Unknown record field(s) $unknown: no layer of this network has '
        'them.',
      );
    }
    final topo = _topology!;
    final out = Matrix(n, _configs.first.size);
    final traj = record.isEmpty ? null : DeepTrajectories._(record, _state, n);
    for (var t = 0; t < n; t++) {
      final dt = timeSteps?[t] ?? 1.0;
      continuousPredictionSweep(
        _layers,
        _state,
        topo,
        dt,
        settings.meanFieldUpdates,
      );
      continuousUpdateSweep(_layers, _state, topo, y, t, dt, settings);
      out.data.setAll(t * out.cols, _state.first.expectedMean);
      traj?._record(t, _state);
    }
    trajectories = traj;
    return out;
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

  /// Continuous networks: the fixed volatility couplings κ, bottom to top
  /// (`child × size`, `null` for a layer that is no volatility parent).
  /// For a continuous network [weights] holds the fixed value couplings W
  /// (`child × size`, `null` for a layer that is no value parent). Both are
  /// the network's own: edit them in place to set other couplings.
  List<Matrix?> get volatilityWeights => [
    for (final l in _layers) l.volatilityWeights,
  ];

  /// Replace the weights of layers `1..n` (same shapes as [weights]).
  void setWeights(List<Matrix> weights) {
    _requireDeep('setWeights');
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

  /// The accumulated precision of each weight's belief above the prior
  /// (pyhgf's `weights_precision_delta`), bottom to top: `null` for the
  /// bottom layer and for layers without a belief. Same shapes as
  /// [weights]; the network's own (changes take effect).
  List<Matrix?> get weightPrecisionDeltas => [
    for (final l in _layers)
      l.weightsPrecisionDelta == null
          ? null
          : Matrix(l.weights!.rows, l.weights!.cols, l.weightsPrecisionDelta!),
  ];

  /// Give the weights of [layers] (indices, 0 the bottom; default every
  /// layer holding weights) a belief: its mean is the weight, its precision
  /// above the prior starts at zero. Layers that already carry one keep it.
  /// [LearningKind.synapticUncertainty] installs it itself.
  DeepNetwork installWeightBelief({List<int>? layers}) {
    _requireDeep('installWeightBelief');
    _requireLayers();
    if (layers != null) {
      final unheld = [
        for (final i in layers)
          if (i < 0 || i >= _layers.length || _layers[i].weights == null) i,
      ];
      if (unheld.isNotEmpty) {
        throw ArgumentError(
          'Layers $unheld hold no weight matrix, so they cannot carry a '
          'weight belief.',
        );
      }
    }
    for (var i = 0; i < _layers.length; i++) {
      final l = _layers[i];
      if (l.weights == null || l.weightsPrecisionDelta != null) continue;
      if (layers != null && !layers.contains(i)) continue;
      l.weightsPrecisionDelta = Float64List(l.weights!.data.length);
    }
    return this;
  }

  /// Check a learning request (pyhgf's `_resolve_learning`): the
  /// weight-belief rule takes its settings instead of an optimiser.
  /// Installs the weight belief when [kind] asks for it.
  void _resolveLearning(
    LearningKind kind,
    SynapticUncertainty? synaptic,
    Optimiser? optimiser,
  ) {
    if (kind != LearningKind.synapticUncertainty) {
      if (synaptic != null) {
        throw ArgumentError(
          'synapticUncertainty settings are only used by '
          'LearningKind.synapticUncertainty, not by ${kind.name}.',
        );
      }
      return;
    }
    if (settings.meanFieldUpdates) {
      throw ArgumentError(
        'LearningKind.synapticUncertainty is derived from the relaxed '
        'updates and is not available with meanFieldUpdates.',
      );
    }
    if (synaptic == null) {
      throw ArgumentError(
        'LearningKind.synapticUncertainty needs its settings: pass '
        'synapticUncertainty: SynapticUncertainty(window: N).',
      );
    }
    if (optimiser != null) {
      throw ArgumentError(
        'LearningKind.synapticUncertainty takes its step size from the '
        'weight belief: pass no optimiser.',
      );
    }
    installWeightBelief();
  }

  /// Draw every dense weight matrix from [strategy] (one-to-one layers keep
  /// the identity). The bias column is set to
  /// zero and left out of the fan-in. Each layer draws from its own stream,
  /// seeded from [seed] and its index, so adding layers above leaves the
  /// lower layers' draws unchanged. The same [seed] gives the same weights
  /// in every run.
  DeepNetwork weightInitialisation(WeightInit strategy, {int seed = 0}) {
    _requireDeep('weightInitialisation');
    for (var l = 1; l < _layers.length; l++) {
      final layer = _layers[l];
      // One-to-one layers keep their identity weights: every strategy draws
      // at the scale of a dense fan-in (pyhgf #476).
      if (!layer.config.fullyConnected) continue;
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
  ///
  /// [LearningKind.synapticUncertainty] learns with the weight-belief rule
  /// instead of an optimiser: pass its settings as [synapticUncertainty].
  Matrix fit(
    Matrix x,
    Matrix y, {
    Optimiser? optimiser,
    LearningKind learningKind = LearningKind.precisionWeighted,
    SynapticUncertainty? synapticUncertainty,
    bool weightUpdate = true,
    double timeStep = 1.0,
    bool updatePrecisions = true,
    Set<LayerField> record = const {},
  }) {
    _requireDeep('fit');
    _requireLayers(2);
    _resolveLearning(learningKind, synapticUncertainty, optimiser);
    if (weightUpdate && optimiser == null && synapticUncertainty == null) {
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
    final learner = !weightUpdate
        ? null
        : synapticUncertainty == null
        ? _Learner.optimiser(optimiser!)
        : _Learner.synaptic(synapticUncertainty, _gradBuffers());
    for (var t = 0; t < n; t++) {
      _stepRow(
        x,
        y,
        t,
        learner,
        learningKind,
        timeStep,
        template,
        updatePrecisions,
        grads,
        weights,
        out,
        traj,
        null,
      );
    }
    trajectories = traj;
    return out;
  }

  /// One row of [fit]; shared with [trace] so both run the same arithmetic.
  void _stepRow(
    Matrix x,
    Matrix y,
    int t,
    _Learner? learner,
    LearningKind learningKind,
    double timeStep,
    List<LayerState>? template,
    bool updatePrecisions,
    List<Float64List?> grads,
    List<Matrix?> weights,
    Matrix out,
    DeepTrajectories? traj,
    TraceRecorder? rec,
  ) {
    final hook = rec == null
        ? null
        : (TracePhase phase, int layer) => rec.record(phase, layer, _state);
    predictionSweep(_layers, _state, x, t, _ws1, timeStep, settings, hook);
    updateSweep(_layers, _state, y, t, _ws1, timeStep, settings, hook);
    if (learner != null || rec != null) {
      for (final g in grads) {
        g?.fillRange(0, g.length, 0.0);
      }
      final imp = learner?.importance;
      if (imp != null) {
        for (final g in imp) {
          g?.fillRange(0, g.length, 0.0);
        }
      }
      accumulateGradients(
        _layers,
        _state,
        _ws1,
        learningKind,
        grads,
        importance: imp,
      );
      learner?.step(_layers, weights, grads);
      rec?.record(
        TracePhase.weightUpdate,
        -1,
        _state,
        weights: weights,
        gradients: [
          for (var l = 0; l < grads.length; l++)
            grads[l] == null
                ? null
                : Matrix(
                    weights[l]!.rows,
                    weights[l]!.cols,
                    Float64List.fromList(grads[l]!),
                  ),
        ],
      );
    }
    out.data.setAll(t * out.cols, _state.first.expectedMean);
    if (template != null) _restoreCarried(template);
    traj?._record(t, _state);
    if (updatePrecisions) carryPrecisions(_layers, _state, settings);
    rec?.record(TracePhase.carry, -1, _state);
  }

  /// Run one sample exactly as one row of [fit] and record the network
  /// after every phase: the input clamp, each layer's top-down prediction,
  /// the observation, the leaf error, each interior layer's posterior and
  /// error, the weight step and the carry. For teaching and inspection:
  /// the arithmetic is [fit]'s, so tracing rows one by one leaves the
  /// network exactly where [fit] would.
  ///
  /// With [y] `null` only the prediction sweep runs and nothing is learned
  /// (the beliefs still hold the predictions; [snapshot] and [restore] undo
  /// that). The descent gradients are recorded even when [optimiser] is
  /// `null` (no step is taken then).
  DeepTrace trace(
    List<double> x, {
    List<double>? y,
    Optimiser? optimiser,
    LearningKind learningKind = LearningKind.precisionWeighted,
    SynapticUncertainty? synapticUncertainty,
    double timeStep = 1.0,
    bool updatePrecisions = true,
  }) {
    _requireDeep('trace');
    _requireLayers(2);
    final xm = Matrix.row(x);
    _checkColumns(xm, _configs.last.size, 'x');
    final weights = this.weights;
    final rec = TraceRecorder(weights);
    if (y == null) {
      predictionSweep(
        _layers,
        _state,
        xm,
        0,
        _ws1,
        timeStep,
        settings,
        (phase, layer) => rec.record(phase, layer, _state),
      );
      return rec.finish(List.of(_state.first.expectedMean));
    }
    final ym = Matrix.row(y);
    _checkColumns(ym, _configs.first.size, 'y');
    _resolveLearning(learningKind, synapticUncertainty, optimiser);
    final out = Matrix(1, _configs.first.size);
    _stepRow(
      xm,
      ym,
      0,
      synapticUncertainty != null
          ? _Learner.synaptic(synapticUncertainty, _gradBuffers())
          : optimiser == null
          ? null
          : _Learner.optimiser(optimiser),
      learningKind,
      timeStep,
      updatePrecisions ? null : [for (final s in _state) s.copy()],
      updatePrecisions,
      _gradBuffers(),
      weights,
      out,
      null,
      rec,
    );
    return rec.finish(List.of(out.data));
  }

  /// Rebuilds a network from [toJson]'s output: the same layers, settings,
  /// weights, per-node parameters and current beliefs, so it predicts and
  /// learns bit for bit like the original. Custom couplings are looked up
  /// by name in [couplings].
  factory DeepNetwork.fromJson(
    Map<String, Object?> json, {
    Map<String, Coupling> couplings = const {},
  }) => _deepNetworkFromJson(json, couplings);

  /// The whole network as JSON (`jsonEncode` accepts it): its settings,
  /// layers, weights, per-node parameters and current beliefs. Save a
  /// trained network with it and load it back with [DeepNetwork.fromJson]:
  ///
  /// ```dart
  /// File('model.json').writeAsStringSync(jsonEncode(net.toJson()));
  /// final again = DeepNetwork.fromJson(
  ///   jsonDecode(File('model.json').readAsStringSync()) as Map<String, Object?>,
  /// );
  /// ```
  ///
  /// The optimiser's state (Adam's moments) is not part of the network;
  /// training a loaded network goes on with a fresh optimiser. Custom
  /// couplings are written by name only. Non-finite numbers are written as
  /// the strings `"NaN"`, `"Infinity"` and `"-Infinity"`.
  Map<String, Object?> toJson() => _deepNetworkToJson(this);

  /// A copy of the weights and beliefs, for [restore].
  DeepSnapshot snapshot() => DeepSnapshot._(
    [for (final l in _layers) l.weights?.copy()],
    [for (final s in _state) s.copy()],
    [
      for (final l in _layers)
        l.weightsPrecisionDelta == null
            ? null
            : Float64List.fromList(l.weightsPrecisionDelta!),
    ],
  );

  /// Put the weights and beliefs back to [snap] (taken from this network,
  /// with the same layers).
  void restore(DeepSnapshot snap) {
    if (snap._states.length != _state.length) {
      throw ArgumentError('The snapshot is of a different network.');
    }
    for (var l = 0; l < _layers.length; l++) {
      final w = snap._weights[l];
      if (w != null) _layers[l].weights!.data.setAll(0, w.data);
      final d = snap._deltas[l];
      _layers[l].weightsPrecisionDelta = d == null
          ? null
          : Float64List.fromList(d);
      _state[l].setFrom(snap._states[l]);
    }
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
    _requireDeep('predict');
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
        case LayerKind.continuous:
          throw UnsupportedError('Continuous layers run in inputData().');
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
  ///
  /// [sampleWeight] (one non-negative weight per row) makes every batch
  /// mean a weighted mean with denominator `sum(sampleWeight)`, so padded
  /// or masked rows (weight 0) do not dilute the step. An all-zero mask
  /// contributes nothing, but the optimiser still steps.
  /// [LearningKind.synapticUncertainty] takes [synapticUncertainty] instead
  /// of an optimiser.
  Matrix batchUpdate(
    Matrix x,
    Matrix y, {
    Optimiser? optimiser,
    LearningKind learningKind = LearningKind.precisionWeighted,
    SynapticUncertainty? synapticUncertainty,
    double timeStep = 1.0,
    bool updatePrecisions = true,
    List<double>? sampleWeight,
  }) {
    _requireDeep('batchUpdate');
    _requireLayers(2);
    _resolveLearning(learningKind, synapticUncertainty, optimiser);
    _checkColumns(x, _configs.last.size, 'x');
    _checkColumns(y, _configs.first.size, 'y');
    if (x.rows != y.rows) {
      throw ArgumentError(
        'x and y must have the same number of samples, got ${x.rows} and '
        '${y.rows}.',
      );
    }
    final b = x.rows;
    Float64List? weightsOfSamples;
    if (sampleWeight != null) {
      if (sampleWeight.length != b) {
        throw ArgumentError(
          'sampleWeight has ${sampleWeight.length} entries for $b samples.',
        );
      }
      var total = 0.0;
      for (final v in sampleWeight) {
        total += v;
      }
      final denom = total > 0 ? total : 1.0;
      weightsOfSamples = Float64List.fromList([
        for (final v in sampleWeight) v / denom,
      ]);
    }
    final ws = _workspace(b);
    final states = _tiled(b);
    predictionSweep(_layers, states, x, 0, ws, timeStep, settings);
    updateSweep(_layers, states, y, 0, ws, timeStep, settings);
    final errors = Matrix(b, _configs.last.size);
    inputPredictionError(_layers, states, ws, settings, errors, 0);
    final learner = synapticUncertainty != null
        ? _Learner.synaptic(synapticUncertainty, _gradBuffers())
        : optimiser == null
        ? null
        : _Learner.optimiser(optimiser);
    if (learner != null) {
      final grads = _gradBuffers();
      accumulateGradients(
        _layers,
        states,
        ws,
        learningKind,
        grads,
        sampleWeights: weightsOfSamples,
        importance: learner.importance,
      );
      learner.step(_layers, weights, grads);
    }
    if (updatePrecisions) {
      carryPrecisions(_layers, states, settings);
      for (var l = 0; l < _state.length; l++) {
        final t = _state[l];
        final s = states[l];
        _addMeanIncrement(t.precision, s.precision, b, weightsOfSamples);
        if (t.meanVol != null) {
          _addMeanIncrement(t.meanVol!, s.meanVol!, b, weightsOfSamples);
          _addMeanIncrement(
            t.precisionVol!,
            s.precisionVol!,
            b,
            weightsOfSamples,
          );
        }
      }
    }
    return errors;
  }

  /// `template += mean_s(batch[·][s] − template)`, or with [weights] (summing
  /// to 1, or all zero) `template += Σ_s weights[s]·(batch[·][s] − template)`.
  static void _addMeanIncrement(
    Float64List template,
    Float64List batch,
    int b, [
    Float64List? weights,
  ]) {
    for (var i = 0; i < template.length; i++) {
      final t = template[i];
      var sum = 0.0;
      if (weights == null) {
        for (var s = 0; s < b; s++) {
          sum += batch[i * b + s] - t;
        }
        template[i] = t + sum / b;
      } else {
        for (var s = 0; s < b; s++) {
          sum += weights[s] * (batch[i * b + s] - t);
        }
        template[i] = t + sum;
      }
    }
  }

  /// How much learning signal each weight matrix receives
  /// (`check_gradient_health`): one prediction and update sweep on the
  /// sample `(x, y)` from the current state (which is left unchanged),
  /// then the largest child-side gradient factor `|u|` per matrix.
  ///
  /// A matrix is [GradientHealth.dead] when that is exactly zero or below
  /// [relativeFloor] times the largest, or when its child's belief
  /// displacement is within a few float32 ulps of the belief's scale
  /// ([GradientHealth.quantised]: the factor is rounding noise at float32,
  /// the precision of the GPU and of pyhgf's default).
  GradientHealth checkGradientHealth(
    List<double> x,
    List<double> y, {
    LearningKind learningKind = LearningKind.precisionWeighted,
    double relativeFloor = 1e-9,
  }) {
    _requireDeep('checkGradientHealth');
    _requireLayers(2);
    final xm = Matrix.row(x), ym = Matrix.row(y);
    _checkColumns(xm, _configs.last.size, 'x');
    _checkColumns(ym, _configs.first.size, 'y');
    final states = [for (final st in _state) st.copy()];
    final ws = Workspace(1);
    predictionSweep(_layers, states, xm, 0, ws, 1.0, settings);
    updateSweep(_layers, states, ym, 0, ws, 1.0, settings);
    final factorMax = Float64List(_layers.length);
    accumulateGradients(
      _layers,
      states,
      ws,
      learningKind,
      _gradBuffers(),
      factorMax: factorMax,
    );
    final magnitudes = <double?>[
      null,
      for (var l = 1; l < _layers.length; l++) factorMax[l],
    ];
    var live = 0.0;
    for (final m in magnitudes) {
      if (m != null && m > live) live = m;
    }
    final dead = <int>{
      for (var l = 1; l < magnitudes.length; l++)
        if (magnitudes[l]! == 0.0 || magnitudes[l]! < live * relativeFloor) l,
    };
    // Displacements within a few f32 ulps of the belief's scale.
    const eps32 = 1.1920928955078125e-7;
    final quantised = <int>[];
    for (var l = 1; l < _layers.length; l++) {
      final c = states[l - 1];
      final ulps = [
        for (var k = 0; k < c.nodes; k++)
          c.valuePredictionError[k].abs() /
              math.max(
                math.max(c.mean[k].abs(), c.expectedMean[k].abs()) * eps32,
                1e-300,
              ),
      ]..sort();
      final n = ulps.length;
      final median = n.isOdd
          ? ulps[n ~/ 2]
          : (ulps[n ~/ 2 - 1] + ulps[n ~/ 2]) / 2;
      if (median < 8.0) quantised.add(l);
    }
    return GradientHealth._(
      magnitudes,
      (dead..addAll(quantised)).toList()..sort(),
      quantised,
    );
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
    _requireDeep('inputPredictionErrorOfState');
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

/// A copy of a [DeepNetwork]'s weights and beliefs ([DeepNetwork.snapshot]).
/// Optimiser state is not included (see [Optimiser.copy]).
final class DeepSnapshot {
  DeepSnapshot._(this._weights, this._states, this._deltas);

  final List<Matrix?> _weights;
  final List<LayerState> _states;
  final List<Float64List?> _deltas;
}

/// The result of [DeepNetwork.checkGradientHealth].
final class GradientHealth {
  GradientHealth._(this.magnitudes, this.dead, this.quantised);

  /// Per weight matrix (indexed like [DeepNetwork.weights]; `null` for the
  /// bottom layer), the largest child-side gradient factor `|u|`.
  final List<double?> magnitudes;

  /// The matrices receiving no usable signal, by either detector.
  final List<int> dead;

  /// The subset flagged because the child's belief displacement is within
  /// a few float32 ulps of its scale.
  final List<int> quantised;

  /// Whether part of the network learns while another part does not (the
  /// case pyhgf warns about: only a head is training).
  bool get partiallyDead =>
      dead.isNotEmpty &&
      [
        for (var l = 0; l < magnitudes.length; l++)
          if (magnitudes[l] case final m? when m > 0 && !dead.contains(l)) l,
      ].isNotEmpty;

  @override
  String toString() =>
      'GradientHealth(magnitudes: $magnitudes, dead: $dead, '
      'quantised: $quantised)';
}

/// What takes the learning step: an [Optimiser], or the weight-belief rule
/// of [LearningKind.synapticUncertainty] (which also needs the curvature
/// increments, in [importance]).
final class _Learner {
  _Learner.optimiser(Optimiser this.optimiser)
    : synaptic = null,
      importance = null;

  _Learner.synaptic(SynapticUncertainty this.synaptic, this.importance)
    : optimiser = null;

  final Optimiser? optimiser;
  final SynapticUncertainty? synaptic;
  final List<Float64List?>? importance;

  void step(List<Layer> layers, List<Matrix?> weights, List<Float64List?> g) {
    final o = optimiser;
    if (o != null) {
      o.step(weights, g);
    } else {
      applySynapticUncertainty(layers, g, importance!, synaptic!);
    }
  }
}
