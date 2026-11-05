part of '../network.dart';

/// [batchSize] copies of a [Network] run in lock step. The copies share one
/// graph and one update schedule; each **element** has its own parameters,
/// initial beliefs and coupling strengths, and optionally its own
/// observations.
///
/// This is the engine for fitting and simulation. One pass filters every
/// parameter set. Each update step runs stage by stage across a block of
/// elements, so the CPU overlaps the elements' independent exps and
/// divisions; a single filter is a chain of dependent ones. Per element-trial
/// that is 1.4-1.9x faster than looping a [Network] over the parameter sets
/// on AOT, and 1.2-1.5x on Wasm (`benchmark/bin/batch_blocks.dart`).
///
/// ```dart
/// final batch = BatchNetwork(hgf, omegas.length)
///   ..setInitial(1, NodeField.tonicVolatility, omegas);
/// final nll = batch.surprise(u, const TotalGaussianSurprise()); // one per ω
/// ```
///
/// Each element computes bit for bit what [Network] computes with the same
/// parameters and data. The batch is a snapshot of the network as compiled
/// when the batch was created, including its initial values and coupling
/// strengths. Later [Network.setInitial] calls don't reach it. Adding nodes
/// to the network makes the batch unusable.
///
/// Trajectories aren't recorded. Read the final state with [get] or [read],
/// and the per-trial surprise through [surprise]. To record one parameter
/// set, run it on a [Network].
///
/// State is stored `[field][node][element]`, so each stage reads the same
/// node of consecutive elements. The batch runs [blockSize] elements at a
/// time through the whole series.
final class BatchNetwork {
  /// A batch of [batchSize] copies of [network], every element starting
  /// from the network's initial values and coupling strengths.
  ///
  /// Throws [UnsupportedError] if the network has a node kind the batched
  /// engine doesn't support yet; run those on [Network].
  BatchNetwork(Network network, int batchSize, {int blockSize = 256})
    : this._(network..build(), _positive(batchSize, 'batchSize'), blockSize);

  BatchNetwork._(Network net, int b, int blockSize)
    : _template = net,
      _n = net._n,
      _b = b,
      blockSize = _positive(blockSize, 'blockSize'),
      _kind = net._kind,
      _couplingFn = net._couplingFn,
      _couplings = net._couplings,
      _isLeaf = net._isLeaf,
      _inputs = net._inputs,
      _vpStart = net._vpStart,
      _vpIdx = net._vpIdx,
      _vcStart = net._vcStart,
      _vcIdx = net._vcIdx,
      _volpStart = net._volpStart,
      _volpIdx = net._volpIdx,
      _volcStart = net._volcStart,
      _volcIdx = net._volcIdx,
      _vpK = _broadcast(net._vpK, b),
      _vcK = _broadcast(net._vcK, b),
      _volpK = _broadcast(net._volpK, b),
      _volcK = _broadcast(net._volcK, b),
      _initial = _broadcast(net._initial, b),
      _mean = Float64List(net._n * b),
      _expectedMean = Float64List(net._n * b),
      _precision = Float64List(net._n * b),
      _expectedPrecision = Float64List(net._n * b),
      _conditionalExpectedPrecision = Float64List(net._n * b),
      _observed = Float64List(net._n * b),
      _tonicVolatility = Float64List(net._n * b),
      _tonicDrift = Float64List(net._n * b),
      _autoconnectionStrength = Float64List(net._n * b),
      _currentVariance = Float64List(net._n * b),
      _effectivePrecision = Float64List(net._n * b),
      _valuePredictionError = Float64List(net._n * b),
      _volatilityPredictionError = Float64List(net._n * b),
      _nus = Float64List(net._n * b),
      _klDivergence = Float64List(net._n * b),
      _surprise = Float64List(net._n * b),
      _predNode = net._predNode,
      _predStep = net._predStep,
      _updNode = net._updNode,
      _updStep = net._updStep,
      _s = _BatchScratch(blockSize < b ? blockSize : b, net._u.capacity),
      meanFieldUpdates = net.meanFieldUpdates,
      ehgfMgfCorrection = net.ehgfMgfCorrection,
      maxPosteriorPrecision = net.maxPosteriorPrecision,
      precisionClippingValue = net.precisionClippingValue {
    for (final s in [..._predStep, ..._updStep]) {
      if (s > _kLastBatchedStep) {
        throw UnsupportedError(
          'BatchNetwork does not support ${UpdateStep.values[s].name} yet; '
          'use Network',
        );
      }
    }
    _checkInvariants();
    reset();
  }

  static int _positive(int v, String name) =>
      v >= 1 ? v : throw ArgumentError.value(v, name, 'Must be at least 1');

  /// [src] with each value repeated [b] times: `out[k * b + e] = src[k]`.
  static Float64List _broadcast(Float64List src, int b) {
    final out = Float64List(src.length * b);
    for (var k = 0; k < src.length; k++) {
      out.fillRange(k * b, (k + 1) * b, src[k]);
    }
    return out;
  }

  /// Highest step code the batched kernels implement. Step codes added for
  /// new node kinds (vector-valued nodes) are rejected until they get a
  /// batched kernel.
  static const int _kLastBatchedStep = _kPredictionErrorBinary;

  final Network _template;
  final int _n;
  final int _b;

  /// How many elements run through the series together. Larger blocks give
  /// the CPU more independent work per stage; smaller ones keep the block's
  /// state in cache. The default of 256 is best or close to it for every
  /// update mode on AOT and Wasm (`benchmark/bin/batch_blocks.dart`).
  final int blockSize;

  final Uint8List _kind, _couplingFn, _isLeaf;
  final List<Coupling> _couplings;
  final Int32List _inputs;
  final Int32List _vpStart, _vpIdx, _vcStart, _vcIdx;
  final Int32List _volpStart, _volpIdx, _volcStart, _volcIdx;

  // Per-element coupling strengths, `[edge][element]`.
  final Float64List _vpK, _vcK, _volpK, _volcK;

  // Initial state, `[field][node][element]`.
  final Float64List _initial;

  // Current state, one `[node][element]` column per field.
  final Float64List _mean,
      _expectedMean,
      _precision,
      _expectedPrecision,
      _conditionalExpectedPrecision,
      _observed,
      _tonicVolatility,
      _tonicDrift,
      _autoconnectionStrength,
      _currentVariance,
      _effectivePrecision,
      _valuePredictionError,
      _volatilityPredictionError,
      _nus,
      _klDivergence,
      _surprise;

  final Int32List _predNode, _updNode;
  final Uint8List _predStep, _updStep;

  final _BatchScratch _s;

  /// As [Network.meanFieldUpdates].
  final bool meanFieldUpdates;

  /// As [Network.ehgfMgfCorrection].
  final bool ehgfMgfCorrection;

  /// As [Network.maxPosteriorPrecision], for every element.
  double maxPosteriorPrecision;

  /// As [Network.precisionClippingValue], for every element.
  double precisionClippingValue;

  /// Number of elements.
  int get batchSize => _b;

  /// Number of nodes in each element.
  int get nNodes => _n;

  /// The kind of node [i].
  NodeKind kindOf(int i) => _template._defs[i].kind;

  /// As [Network.inputIdxs].
  List<int> get inputIdxs => _template._inputIdxsView;

  void _checkInvariants() {
    final n = _n, b = _b;
    for (final f in NodeField.values) {
      if (_column(f).length != n * b) throw StateError('Bad column ${f.name}');
    }
    if (_initial.length != NodeField.values.length * n * b) {
      throw StateError('Bad initial state');
    }
    if (_vpK.length != _vpIdx.length * b ||
        _vcK.length != _vcIdx.length * b ||
        _volpK.length != _volpIdx.length * b ||
        _volcK.length != _volcIdx.length * b) {
      throw StateError('Bad coupling strengths');
    }
    if (_s.stride < (blockSize < b ? blockSize : b) ||
        _s.coupling.length < _template._u.capacity * _s.stride) {
      throw StateError('Scratch buffers too small');
    }
    // The graph arrays are the network's, validated by its own build().
    if (_vpStart.length != n + 1 || _kind.length != n) {
      throw StateError('Bad graph');
    }
  }

  /// Throws if the network was changed (nodes added) after the batch was
  /// made from it.
  void _checkTemplate() {
    final t = _template;
    if (!t._compiled || !identical(t._vpStart, _vpStart)) {
      throw StateError(
        'The network changed after this BatchNetwork was made from it',
      );
    }
  }

  // ---------------------------------------------------------------------------
  // State access
  // ---------------------------------------------------------------------------

  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  Float64List _column(NodeField f) => switch (f) {
    NodeField.mean => _mean,
    NodeField.expectedMean => _expectedMean,
    NodeField.precision => _precision,
    NodeField.expectedPrecision => _expectedPrecision,
    NodeField.conditionalExpectedPrecision => _conditionalExpectedPrecision,
    NodeField.observed => _observed,
    NodeField.tonicVolatility => _tonicVolatility,
    NodeField.tonicDrift => _tonicDrift,
    NodeField.autoconnectionStrength => _autoconnectionStrength,
    NodeField.currentVariance => _currentVariance,
    NodeField.effectivePrecision => _effectivePrecision,
    NodeField.valuePredictionError => _valuePredictionError,
    NodeField.volatilityPredictionError => _volatilityPredictionError,
    NodeField.nus => _nus,
    NodeField.klDivergence => _klDivergence,
    NodeField.surprise => _surprise,
  };

  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  int _index(int element, int node) {
    if (element < 0 || element >= _b) {
      throw RangeError.range(element, 0, _b - 1, 'element');
    }
    if (node < 0 || node >= _n) throw RangeError.range(node, 0, _n - 1, 'node');
    return node * _b + element;
  }

  /// Current value of [field] on [node] of [element].
  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  double get(int element, int node, NodeField field) =>
      _column(field)[_index(element, node)];

  /// Set the current value of [field] on [node] of [element].
  void set(int element, int node, NodeField field, double value) =>
      _column(field)[_index(element, node)] = value;

  /// Current value of [field] on [node] for every element, copied into
  /// [into] (length [batchSize]) if given.
  Float64List read(int node, NodeField field, [Float64List? into]) {
    final start = _index(0, node);
    final out = into ?? Float64List(_b);
    if (out.length != _b) throw ArgumentError('into must have batchSize items');
    return out..setRange(0, _b, _column(field), start);
  }

  /// Initial value of [field] on [node] of [element].
  double getInitial(int element, int node, NodeField field) =>
      _initial[(field.index * _n) * _b + _index(element, node)];

  /// Set the initial value of [field] on [node] per element ([values] has
  /// one entry per element), e.g. one tonic volatility per element. Takes
  /// effect on the next [reset] / run.
  void setInitial(int node, NodeField field, Float64List values) {
    if (values.length != _b) {
      throw ArgumentError('values must have one entry per element');
    }
    _initial.setRange(
      (field.index * _n) * _b + _index(0, node),
      (field.index * _n + node + 1) * _b,
      values,
    );
  }

  /// Set the initial value of [field] on [node] of one [element].
  void setInitialAt(int element, int node, NodeField field, double value) =>
      _initial[(field.index * _n) * _b + _index(element, node)] = value;

  /// Set the value coupling strength from [parent] to [child] per element
  /// ([strengths] has one entry per element).
  void setValueCoupling(int parent, int child, Float64List strengths) =>
      _setCoupling(
        parent,
        child,
        strengths,
        _vcStart,
        _vcIdx,
        _vcK,
        _vpStart,
        _vpIdx,
        _vpK,
        'value',
      );

  /// Set the volatility coupling strength from [parent] to [child] per
  /// element ([strengths] has one entry per element).
  void setVolatilityCoupling(int parent, int child, Float64List strengths) =>
      _setCoupling(
        parent,
        child,
        strengths,
        _volcStart,
        _volcIdx,
        _volcK,
        _volpStart,
        _volpIdx,
        _volpK,
        'volatility',
      );

  void _setCoupling(
    int parent,
    int child,
    Float64List strengths,
    Int32List childStart,
    Int32List childIdx,
    Float64List childK,
    Int32List parentStart,
    Int32List parentIdx,
    Float64List parentK,
    String what,
  ) {
    _index(0, parent);
    _index(0, child);
    if (strengths.length != _b) {
      throw ArgumentError('strengths must have one entry per element');
    }
    final b = _b;
    var found = false;
    // The parent's children carry the parent-side strength, the child's
    // parents the child-side one; set both, as Network.addNodes does.
    for (var e = childStart[parent]; e < childStart[parent + 1]; e++) {
      if (childIdx[e] == child) {
        childK.setRange(e * b, (e + 1) * b, strengths);
        found = true;
      }
    }
    for (var e = parentStart[child]; e < parentStart[child + 1]; e++) {
      if (parentIdx[e] == parent) {
        parentK.setRange(e * b, (e + 1) * b, strengths);
      }
    }
    if (!found) {
      throw ArgumentError('Node $parent is not a $what parent of node $child');
    }
  }

  /// Restore every element to its initial state.
  void reset() {
    final nb = _n * _b;
    for (final f in NodeField.values) {
      _column(f).setRange(0, nb, _initial, f.index * nb);
    }
  }

  // ---------------------------------------------------------------------------
  // Filtering
  // ---------------------------------------------------------------------------

  /// Filter a series of observations through every element, starting from
  /// the initial state (or the current one with [append]).
  ///
  /// By default every element sees the same [data], laid out as for
  /// [Network.inputData] (row-major `nTime x nInputs`). With [perElement],
  /// each element has its own series: [data] is `nTime x nInputs x
  /// batchSize`, element fastest (`data[(t * nInputs + k) * batchSize + e]`).
  /// [observed] has the same layout as [data]; [timeSteps] (length `nTime`)
  /// is shared.
  void inputData(
    Float64List data, {
    Float64List? observed,
    Float64List? timeSteps,
    bool perElement = false,
    bool append = false,
  }) {
    final nTime = _prepareRun(data, observed, timeSteps, perElement, append);
    _run(data, observed, timeSteps, perElement, nTime, null, null, null);
  }

  /// Filter [data] (as in [inputData]) and return the total surprise of
  /// [response] for each element: the negative log-likelihood of each
  /// parameter set. [response] must support batches
  /// ([ResponseFunction.batchSurprise]); the built-in ones do.
  ///
  /// [into] (length [batchSize]) receives the totals if given. [perTrial]
  /// (length `nTime * batchSize`) receives each trial's surprise,
  /// `perTrial[t * batchSize + e]`.
  Float64List surprise(
    Float64List data,
    ResponseFunction response, {
    Float64List? observed,
    Float64List? timeSteps,
    bool perElement = false,
    bool append = false,
    Float64List? into,
    Float64List? perTrial,
  }) {
    final nTime = _prepareRun(data, observed, timeSteps, perElement, append);
    response.prepare(_template, nTime);
    final totals = into ?? Float64List(_b);
    if (totals.length != _b) {
      throw ArgumentError('into must have one entry per element');
    }
    if (perTrial != null && perTrial.length != nTime * _b) {
      throw ArgumentError('perTrial must have nTime x batchSize entries');
    }
    totals.fillRange(0, _b, 0.0);
    _run(
      data,
      observed,
      timeSteps,
      perElement,
      nTime,
      response,
      totals,
      perTrial,
    );
    return totals;
  }

  /// One belief-propagation step for every element from the current state.
  /// [observations] has one value per input node, or with [perElement] one
  /// per input node and element (`observations[k * batchSize + e]`).
  void step(
    Float64List observations, {
    Float64List? observed,
    double timeStep = 1.0,
    bool perElement = false,
  }) {
    _checkTemplate();
    final width = _inputs.length * (perElement ? _b : 1);
    if (observations.length != width) {
      throw ArgumentError('Expected $width observation(s)');
    }
    if (observed != null && observed.length != width) {
      throw ArgumentError('Expected $width observed flag(s)');
    }
    for (var lo = 0; lo < _b; lo += blockSize) {
      final hi = lo + blockSize < _b ? lo + blockSize : _b;
      _propagate(observations, observed, 0, timeStep, perElement, lo, hi);
    }
  }

  int _prepareRun(
    Float64List data,
    Float64List? observed,
    Float64List? timeSteps,
    bool perElement,
    bool append,
  ) {
    _checkTemplate();
    final width = _inputs.length * (perElement ? _b : 1);
    if (data.length % width != 0) {
      throw ArgumentError(
        'data length ${data.length} is not a multiple of $width '
        '(${_inputs.length} input(s)${perElement ? ' x $_b elements' : ''})',
      );
    }
    final nTime = data.length ~/ width;
    if (observed != null && observed.length != data.length) {
      throw ArgumentError('observed must have the same shape as data');
    }
    if (timeSteps != null && timeSteps.length != nTime) {
      throw ArgumentError('timeSteps must have one entry per time step');
    }
    if (!append) reset();
    return nTime;
  }

  void _run(
    Float64List data,
    Float64List? observed,
    Float64List? timeSteps,
    bool perElement,
    int nTime,
    ResponseFunction? response,
    Float64List? totals,
    Float64List? perTrial,
  ) {
    final b = _b, width = _inputs.length * (perElement ? b : 1);
    for (var lo = 0; lo < b; lo += blockSize) {
      final hi = lo + blockSize < b ? lo + blockSize : b;
      for (var t = 0; t < nTime; t++) {
        _propagate(
          data,
          observed,
          t * width,
          timeSteps == null ? 1.0 : timeSteps[t],
          perElement,
          lo,
          hi,
        );
        if (response != null) {
          for (var e = lo; e < hi; e++) {
            final s = response.batchSurprise(this, t, e);
            totals![e] += s;
            if (perTrial != null) perTrial[t * b + e] = s;
          }
        }
      }
    }
  }

  /// One time slice for elements `[lo, hi)`: predictions, observations,
  /// then the update sweep. Bounds-check free on the same grounds as
  /// [Network._propagate]; the callers validate the observation buffers.
  @pragma('vm:unsafe:no-bounds-checks')
  void _propagate(
    Float64List data,
    Float64List? observed,
    int offset,
    double timeStep,
    bool perElement,
    int lo,
    int hi,
  ) {
    final predNode = _predNode, predStep = _predStep;
    for (var k = 0; k < predNode.length; k++) {
      _dispatch(predStep[k], predNode[k], timeStep, lo, hi);
    }
    final inputs = _inputs, b = _b, mean = _mean, obs = _observed;
    for (var k = 0; k < inputs.length; k++) {
      final base = inputs[k] * b;
      if (perElement) {
        final src = offset + k * b;
        for (var e = lo; e < hi; e++) {
          mean[base + e] = data[src + e];
          obs[base + e] = observed == null ? 1.0 : observed[src + e];
        }
      } else {
        final x = data[offset + k];
        final o = observed == null ? 1.0 : observed[offset + k];
        for (var e = lo; e < hi; e++) {
          mean[base + e] = x;
          obs[base + e] = o;
        }
      }
    }
    final updNode = _updNode, updStep = _updStep;
    for (var k = 0; k < updNode.length; k++) {
      _dispatch(updStep[k], updNode[k], timeStep, lo, hi);
    }
  }

  @pragma('vm:prefer-inline')
  void _dispatch(int step, int node, double t, int lo, int hi) {
    switch (step) {
      case _kPredictionContinuous:
        _bPredictionContinuous(node, t, lo, hi);
      case _kPredictionContinuousMeanField:
        _bPredictionContinuousMeanField(node, t, lo, hi);
      case _kPredictionBinary:
        _bPredictionBinary(node, lo, hi);
      case _kPosteriorContinuous:
        _bPosteriorContinuous(node, lo, hi, false);
      case _kPosteriorContinuousMeanField:
        _bPosteriorContinuous(node, lo, hi, true);
      case _kPosteriorContinuousEhgf:
        _bPosteriorContinuousEhgf(node, t, lo, hi, false);
      case _kPosteriorContinuousEhgfMeanField:
        _bPosteriorContinuousEhgf(node, t, lo, hi, true);
      case _kPosteriorContinuousUnbounded:
        _bPosteriorContinuousUnbounded(node, t, lo, hi);
      case _kPredictionErrorContinuous:
        _bPredictionErrorContinuous(node, lo, hi);
      case _kPredictionErrorBinary:
        _bPredictionErrorBinary(node, lo, hi);
    }
  }
}
