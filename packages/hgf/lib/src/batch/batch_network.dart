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
///
/// ## Exact gradients
///
/// With [tangents] `k > 0`, every element also carries `k` forward-mode
/// tangents (directional derivatives) through the filter: seed how the
/// initial state and coupling strengths depend on `k` parameters
/// ([setInitialTangent], [setValueCouplingTangent],
/// [setVolatilityCouplingTangent]), run, and read the derivatives of the
/// state ([getTangent], [readTangent]) or of the total surprise
/// ([surprise] with `gradient:`). The primal values stay bit-identical to a
/// batch without tangents.
///
/// ```dart
/// final batch = BatchNetwork(hgf, omegas.length, tangents: 1)
///   ..setInitial(1, NodeField.tonicVolatility, omegas)
///   ..setInitialTangent(1, NodeField.tonicVolatility, 0, ones);
/// final grad = Float64List(omegas.length); // d nll / d ω, per element
/// batch.surprise(u, const TotalGaussianSurprise(), gradient: grad);
/// ```
///
/// Only the fields in [tangentFields] carry tangents (the others are
/// discrete or unused by the batched node kinds). The derivatives are those
/// of the computed function: through each update's branches as taken (a
/// clamp that is active passes no derivative) and, for the Lambert W in the
/// unbounded update, the derivative of `W` itself rather than of its
/// iterations.
final class BatchNetwork {
  /// A batch of [batchSize] copies of [network], every element starting
  /// from the network's initial values and coupling strengths, carrying
  /// [tangents] forward-mode tangents (none by default).
  ///
  /// Throws [UnsupportedError] if the network has a node kind the batched
  /// engine doesn't support yet; run those on [Network].
  BatchNetwork(
    Network network,
    int batchSize, {
    int blockSize = 256,
    int tangents = 0,
  }) : this._(
         network..build(),
         _positive(batchSize, 'batchSize'),
         blockSize,
         tangents >= 0
             ? tangents
             : throw ArgumentError.value(tangents, 'tangents', 'Negative'),
       );

  BatchNetwork._(Network net, int b, int blockSize, int k)
    : _template = net,
      _n = net._n,
      _b = b,
      _k = k,
      blockSize = _dpBlock(net, b, _positive(blockSize, 'blockSize')),
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
      _dpAlpha = Float64List(net._n * b),
      _sensoryPrecision = Float64List(net._n * b),
      _nTotal = Float64List(net._n * b),
      _nActiveCluster = Float64List(net._n * b),
      _inputCol = net._inputCol,
      _inputDim = net._inputDim,
      _nCols = net._nInputColumns,
      _hasVectorInputs = net._hasVectorInputs,
      _vMeanStart = net._vMeanStart,
      _ssStart = net._ssStart,
      _alphaStart = net._alphaStart,
      _vMean = Float64List(net._vMeanInitial.length * b),
      _xis = Float64List(net._xisInitial.length * b),
      _ss = Float64List(net._ssInitial.length * b),
      _alpha = Float64List(net._alphaInitial.length * b),
      _vMeanInitial = _broadcast(net._vMeanInitial, b),
      _xisInitial = _broadcast(net._xisInitial, b),
      _ssInitial = _broadcast(net._ssInitial, b),
      _alphaInitial = _broadcast(net._alphaInitial, b),
      _dpStart = net._dpStart,
      _dpN = Float64List(net._dpN.length * b),
      _dpExpectedMeans = Float64List(net._dpN.length * b),
      _dpExpectedSigmas = Float64List(net._dpN.length * b),
      _dpActivated = Float64List(net._dpN.length * b),
      _dpNInitial = _broadcast(net._dpNInitial, b),
      _dpExpectedMeansInitial = _broadcast(net._dpExpectedMeansInitial, b),
      _dpExpectedSigmasInitial = _broadcast(net._dpExpectedSigmasInitial, b),
      _dpActivatedInitial = _broadcast(net._dpActivatedInitial, b),
      _dpMeansE = Float64List(net._dpN.length),
      _dpSigmasE = Float64List(net._dpN.length),
      _dpScratch = net._dpN.isEmpty
          ? null
          : DpScratch(
              net._dpN.length,
              slots: _dpBlock(net, b, _positive(blockSize, 'blockSize')),
            ),
      _alphaCurrent = Float64List(net._alphaScratch.length),
      _alphaNext = Float64List(net._alphaScratch.length),
      _predNode = net._predNode,
      _predStep = net._predStep,
      _updNode = net._updNode,
      _updStep = net._updStep,
      _s = _BatchScratch(blockSize < b ? blockSize : b, net._u.capacity),
      _dInitial = Float64List(_kTangentFields * net._n * b * k),
      _dMean = Float64List(net._n * b * k),
      _dExpectedMean = Float64List(net._n * b * k),
      _dPrecision = Float64List(net._n * b * k),
      _dExpectedPrecision = Float64List(net._n * b * k),
      _dConditionalExpectedPrecision = Float64List(net._n * b * k),
      _dTonicVolatility = Float64List(net._n * b * k),
      _dTonicDrift = Float64List(net._n * b * k),
      _dAutoconnectionStrength = Float64List(net._n * b * k),
      _dCurrentVariance = Float64List(net._n * b * k),
      _dEffectivePrecision = Float64List(net._n * b * k),
      _dValuePredictionError = Float64List(net._n * b * k),
      _dVolatilityPredictionError = Float64List(net._n * b * k),
      _dVpK = Float64List(net._vpIdx.length * b * k),
      _dVcK = Float64List(net._vcIdx.length * b * k),
      _dVolpK = Float64List(net._volpIdx.length * b * k),
      _dVolcK = Float64List(net._volcIdx.length * b * k),
      _ts = _TangentScratch(blockSize < b ? blockSize : b, net._u.capacity, k),
      _activeDirs = Int32List(k),
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

  /// Highest step code the batched kernels implement: the scalar kinds and
  /// the vector kinds (`batch_vectors.dart`).
  static const int _kLastBatchedStep = _kPredictionErrorDp;

  /// Memory for the dp-state proposal row caches ([DpScratch]), one slot
  /// per element of a block.
  static const int _dpCacheBytes = 64 << 20;

  /// [blockSize] for [net]: with dp-state nodes, at most as many elements
  /// as have a row cache slot within [_dpCacheBytes] (and at least one).
  /// Their step costs ~1 ms per element-trial, so a small block loses
  /// nothing.
  static int _dpBlock(Network net, int b, int blockSize) {
    final block = blockSize < b ? blockSize : b;
    final nd = net._dpN.length;
    if (nd == 0) return blockSize;
    final fit = _dpCacheBytes ~/ (nd * DpScratch.bytesPerRow);
    return fit < 1 ? 1 : (fit < block ? fit : block);
  }

  final Network _template;
  final int _n;
  final int _b;

  /// Number of tangent directions (0: none).
  final int _k;

  /// How many elements run through the series together. Larger blocks give
  /// the CPU more independent work per stage; smaller ones keep the block's
  /// state in cache. The default of 256 is best or close to it for every
  /// update mode on AOT and Wasm (`benchmark/bin/batch_blocks.dart`).
  /// Networks with dp-state nodes cap it (at most the batch size, and as
  /// many elements as their proposal caches fit in 64 MB: 52 with 4
  /// clusters).
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
      _surprise,
      _dpAlpha,
      _sensoryPrecision,
      _nTotal,
      _nActiveCluster;

  // Input columns, as [Network]'s: input q reads `_inputDim[q]` columns
  // from column `_inputCol[q]` of each row of `_nCols`.
  final Int32List _inputCol, _inputDim;
  final int _nCols;
  final bool _hasVectorInputs;

  // Vector side store, `[offset][element]` (node i's entry j of element e
  // at `(start[i] + j) * B + e`), with its initial values; ef-state and
  // categorical nodes only.
  final Int32List _vMeanStart, _ssStart, _alphaStart;
  final Float64List _vMean, _xis, _ss, _alpha;
  final Float64List _vMeanInitial, _xisInitial, _ssInitial, _alphaInitial;

  // The categorical posterior's old and new α of one element.
  final Float64List _alphaCurrent, _alphaNext;

  // dp-state vectors, `[offset][element]` from `_dpStart`, with their
  // initial values; one element's cluster means and sigmas gathered for
  // [dpCandidate]; its row caches, one slot per element of a block.
  final Int32List _dpStart;
  final Float64List _dpN, _dpExpectedMeans, _dpExpectedSigmas, _dpActivated;
  final Float64List _dpNInitial,
      _dpExpectedMeansInitial,
      _dpExpectedSigmasInitial,
      _dpActivatedInitial;
  final Float64List _dpMeansE, _dpSigmasE;
  final DpScratch? _dpScratch;

  final Int32List _predNode, _updNode;
  final Uint8List _predStep, _updStep;

  final _BatchScratch _s;

  // Tangents, `[node][direction][element]` per field (index
  // `(node * k + d) * B + element`), and `[edge][direction][element]` per
  // coupling column; `_dInitial` is `[slot][node][direction][element]`.
  // Empty without tangents.
  final Float64List _dInitial;
  final Float64List _dMean,
      _dExpectedMean,
      _dPrecision,
      _dExpectedPrecision,
      _dConditionalExpectedPrecision,
      _dTonicVolatility,
      _dTonicDrift,
      _dAutoconnectionStrength,
      _dCurrentVariance,
      _dEffectivePrecision,
      _dValuePredictionError,
      _dVolatilityPredictionError;
  final Float64List _dVpK, _dVcK, _dVolpK, _dVolcK;
  final _TangentScratch _ts;

  // Set by [reset] from the seeds: the directions some seed moves (the
  // others stay zero through the filter, e.g. a response parameter's, and
  // the kernels skip them), and which constant tangent columns are all zero
  // (their terms are skipped).
  final Int32List _activeDirs;
  int _nActive = 0;
  bool _zeroAuto = true, _zeroVpK = true, _zeroVcK = true, _zeroVolpK = true;
  late final SurprisePartials _partials = SurprisePartials._(this);

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

  /// Number of forward-mode tangent directions each element carries (see
  /// the class documentation); 0 if none.
  int get tangents => _k;

  /// The fields that carry tangents, in the order of their tangent slots.
  static const List<NodeField> tangentFields = [
    NodeField.mean,
    NodeField.expectedMean,
    NodeField.precision,
    NodeField.expectedPrecision,
    NodeField.conditionalExpectedPrecision,
    NodeField.tonicVolatility,
    NodeField.tonicDrift,
    NodeField.autoconnectionStrength,
    NodeField.currentVariance,
    NodeField.effectivePrecision,
    NodeField.valuePredictionError,
    NodeField.volatilityPredictionError,
  ];

  /// Number of nodes in each element.
  int get nNodes => _n;

  /// The kind of node [i].
  NodeKind kindOf(int i) => _template._defs[i].kind;

  /// As [Network.inputIdxs].
  List<int> get inputIdxs => _template._inputIdxsView;

  /// How many value parents node [i] has (as [Network.valueParentsOf],
  /// without allocating; for response functions).
  int valueParentCount(int i) => _vpStart[i + 1] - _vpStart[i];

  /// Node [i]'s value parent number [k], in [Network.valueParentsOf] order.
  int valueParent(int i, int k) {
    RangeError.checkValidIndex(k, null, 'k', valueParentCount(i));
    return _vpIdx[_vpStart[i] + k];
  }

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
    final nd = _dpStart[n];
    for (final v in [
      _dpN,
      _dpExpectedMeans,
      _dpExpectedSigmas,
      _dpActivated,
      _dpNInitial,
      _dpExpectedMeansInitial,
      _dpExpectedSigmasInitial,
      _dpActivatedInitial,
    ]) {
      if (v.length != nd * b) throw StateError('Bad dp-state store');
    }
    final dps = _dpScratch;
    if (nd > 0 &&
        (_dpMeansE.length != nd ||
            _dpSigmasE.length != nd ||
            dps == null ||
            dps.clusters != nd ||
            dps.slots < (blockSize < b ? blockSize : b))) {
      throw StateError('Bad dp-state scratch');
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
    NodeField.alpha => _dpAlpha,
    NodeField.sensoryPrecision => _sensoryPrecision,
    NodeField.nTotal => _nTotal,
    NodeField.nActiveCluster => _nActiveCluster,
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

  /// [node]'s current [field] vector in [element] (ef-state, categorical
  /// and dp-state nodes; empty for nodes without that vector). As
  /// [Network.getVector].
  Float64List getVector(int element, int node, VectorField field) {
    final i = _index(element, node) - element; // validates both
    assert(i == node * _b);
    final (store, start) = _vectorStore(field);
    final len = _template.vectorLength(node, field);
    if (len == 0) return Float64List(0);
    return Float64List.fromList([
      for (var j = 0; j < len; j++) store[(start[node] + j) * _b + element],
    ]);
  }

  (Float64List, Int32List) _vectorStore(VectorField field) => switch (field) {
    VectorField.mean => (_vMean, _vMeanStart),
    VectorField.xis => (_xis, _ssStart),
    VectorField.observationSs => (_ss, _ssStart),
    VectorField.alpha => (_alpha, _alphaStart),
    VectorField.n => (_dpN, _dpStart),
    VectorField.expectedMeans => (_dpExpectedMeans, _dpStart),
    VectorField.expectedSigmas => (_dpExpectedSigmas, _dpStart),
    VectorField.activated => (_dpActivated, _dpStart),
  };

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

  // ---------------------------------------------------------------------------
  // Tangents
  // ---------------------------------------------------------------------------

  /// The tangent column of [f], or null if [f] carries none.
  Float64List? _tangentColumn(NodeField f) => switch (f) {
    NodeField.mean => _dMean,
    NodeField.expectedMean => _dExpectedMean,
    NodeField.precision => _dPrecision,
    NodeField.expectedPrecision => _dExpectedPrecision,
    NodeField.conditionalExpectedPrecision => _dConditionalExpectedPrecision,
    NodeField.tonicVolatility => _dTonicVolatility,
    NodeField.tonicDrift => _dTonicDrift,
    NodeField.autoconnectionStrength => _dAutoconnectionStrength,
    NodeField.currentVariance => _dCurrentVariance,
    NodeField.effectivePrecision => _dEffectivePrecision,
    NodeField.valuePredictionError => _dValuePredictionError,
    NodeField.volatilityPredictionError => _dVolatilityPredictionError,
    _ => null,
  };

  int _tangentSlot(NodeField field) {
    final slot = tangentFields.indexOf(field);
    if (slot < 0) {
      throw ArgumentError.value(
        field,
        'field',
        'Carries no tangent (see BatchNetwork.tangentFields)',
      );
    }
    return slot;
  }

  int _direction(int d) {
    if (d < 0 || d >= _k) {
      if (_k == 0) {
        throw StateError('This BatchNetwork carries no tangents');
      }
      throw RangeError.range(d, 0, _k - 1, 'direction');
    }
    return d;
  }

  /// Set how the initial value of [field] on [node] depends on parameter
  /// [direction]: `values[e]` is `d initial / d θ_direction` for element
  /// `e`. Takes effect on the next [reset] / run, like [setInitial].
  void setInitialTangent(
    int node,
    NodeField field,
    int direction,
    Float64List values,
  ) {
    final slot = _tangentSlot(field), d = _direction(direction);
    if (values.length != _b) {
      throw ArgumentError('values must have one entry per element');
    }
    _index(0, node);
    _dInitial.setRange(
      ((slot * _n + node) * _k + d) * _b,
      ((slot * _n + node) * _k + d + 1) * _b,
      values,
    );
  }

  /// Set how the value coupling strength from [parent] to [child] depends
  /// on parameter [direction], per element.
  void setValueCouplingTangent(
    int parent,
    int child,
    int direction,
    Float64List values,
  ) => _setCouplingTangent(
    parent,
    child,
    _direction(direction),
    values,
    _vcStart,
    _vcIdx,
    _dVcK,
    _vpStart,
    _vpIdx,
    _dVpK,
    'value',
  );

  /// Set how the volatility coupling strength from [parent] to [child]
  /// depends on parameter [direction], per element.
  void setVolatilityCouplingTangent(
    int parent,
    int child,
    int direction,
    Float64List values,
  ) => _setCouplingTangent(
    parent,
    child,
    _direction(direction),
    values,
    _volcStart,
    _volcIdx,
    _dVolcK,
    _volpStart,
    _volpIdx,
    _dVolpK,
    'volatility',
  );

  void _setCouplingTangent(
    int parent,
    int child,
    int d,
    Float64List values,
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
    if (values.length != _b) {
      throw ArgumentError('values must have one entry per element');
    }
    final b = _b, k = _k;
    var found = false;
    void set(Float64List col, int edge) =>
        col.setRange((edge * k + d) * b, (edge * k + d + 1) * b, values);

    for (var e = childStart[parent]; e < childStart[parent + 1]; e++) {
      if (childIdx[e] == child) {
        set(childK, e);
        found = true;
      }
    }
    for (var e = parentStart[child]; e < parentStart[child + 1]; e++) {
      if (parentIdx[e] == parent) set(parentK, e);
    }
    if (!found) {
      throw ArgumentError('Node $parent is not a $what parent of node $child');
    }
  }

  /// Zero every tangent seed (initial values and coupling strengths), so
  /// nothing depends on any parameter. Takes effect on the next run.
  void clearTangents() {
    _dInitial.fillRange(0, _dInitial.length, 0.0);
    for (final c in [_dVpK, _dVcK, _dVolpK, _dVolcK]) {
      c.fillRange(0, c.length, 0.0);
    }
  }

  /// The current derivative of [field] on [node] of [element] along
  /// [direction]. Throws for fields outside [tangentFields].
  double getTangent(int element, int node, NodeField field, int direction) {
    final col = _tangentColumn(field);
    if (col == null) _tangentSlot(field); // throws
    _index(element, node);
    return col![(node * _k + _direction(direction)) * _b + element];
  }

  /// [getTangent] for every element, copied into [into] (length
  /// [batchSize]) if given.
  Float64List readTangent(
    int node,
    NodeField field,
    int direction, [
    Float64List? into,
  ]) {
    final col = _tangentColumn(field);
    if (col == null) _tangentSlot(field); // throws
    final d = _direction(direction);
    _index(0, node);
    final out = into ?? Float64List(_b);
    if (out.length != _b) throw ArgumentError('into must have batchSize items');
    return out..setRange(0, _b, col!, (node * _k + d) * _b);
  }

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

  /// Restore every element to its initial state (and initial tangents).
  void reset() {
    _pendingPrediction = null;
    final nb = _n * _b;
    for (final f in NodeField.values) {
      _column(f).setRange(0, nb, _initial, f.index * nb);
    }
    _vMean.setAll(0, _vMeanInitial);
    _xis.setAll(0, _xisInitial);
    _ss.setAll(0, _ssInitial);
    _alpha.setAll(0, _alphaInitial);
    _dpN.setAll(0, _dpNInitial);
    _dpExpectedMeans.setAll(0, _dpExpectedMeansInitial);
    _dpExpectedSigmas.setAll(0, _dpExpectedSigmasInitial);
    _dpActivated.setAll(0, _dpActivatedInitial);
    if (_k == 0) return;
    final nbk = nb * _k;
    for (var slot = 0; slot < _kTangentFields; slot++) {
      _tangentColumn(
        tangentFields[slot],
      )!.setRange(0, nbk, _dInitial, slot * nbk);
    }
    _scanSeeds();
  }

  /// Which directions and constant tangent columns are non-zero.
  void _scanSeeds() {
    final k = _k, b = _b, n = _n;
    bool zero(Float64List a, int from, int to) {
      for (var q = from; q < to; q++) {
        if (a[q] != 0.0) return false; // NaN counts as non-zero
      }
      return true;
    }

    // Does direction d of a `[row][direction][element]` array have a
    // non-zero entry?
    bool anyIn(Float64List a, int rows, int d) {
      for (var r = 0; r < rows; r++) {
        final o = (r * k + d) * b;
        if (!zero(a, o, o + b)) return true;
      }
      return false;
    }

    _nActive = 0;
    for (var d = 0; d < k; d++) {
      if (anyIn(_dInitial, _kTangentFields * n, d) ||
          anyIn(_dVpK, _vpIdx.length, d) ||
          anyIn(_dVcK, _vcIdx.length, d) ||
          anyIn(_dVolpK, _volpIdx.length, d) ||
          anyIn(_dVolcK, _volcIdx.length, d)) {
        _activeDirs[_nActive++] = d;
      }
    }
    final nbk = n * b * k;
    int slotOf(NodeField f) => tangentFields.indexOf(f) * nbk;
    final auto = slotOf(NodeField.autoconnectionStrength);
    _zeroAuto = zero(_dInitial, auto, auto + nbk);
    _zeroVpK = zero(_dVpK, 0, _dVpK.length);
    _zeroVcK = zero(_dVcK, 0, _dVcK.length);
    _zeroVolpK = zero(_dVolpK, 0, _dVolpK.length);
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
    _run(data, observed, timeSteps, perElement, nTime, null, null, null, null);
  }

  /// Filter [data] (as in [inputData]) and return the total surprise of
  /// [response] for each element: the negative log-likelihood of each
  /// parameter set. [response] must support batches
  /// ([ResponseFunction.batchSurprise]); the built-in ones do.
  ///
  /// [into] (length [batchSize]) receives the totals if given. [perTrial]
  /// (length `nTime * batchSize`) receives each trial's surprise,
  /// `perTrial[t * batchSize + e]`.
  ///
  /// With [gradient] (length `batchSize * tangents`), the batch must carry
  /// tangents and [response] must support them
  /// ([ResponseFunction.batchSurprisePartials]): `gradient[e * tangents + d]`
  /// receives the derivative of element `e`'s total along direction `d`.
  Float64List surprise(
    Float64List data,
    ResponseFunction response, {
    Float64List? observed,
    Float64List? timeSteps,
    bool perElement = false,
    bool append = false,
    Float64List? into,
    Float64List? perTrial,
    Float64List? gradient,
  }) {
    if (gradient != null) {
      if (_k == 0) throw StateError('This BatchNetwork carries no tangents');
      if (gradient.length != _b * _k) {
        throw ArgumentError('gradient must have batchSize x tangents entries');
      }
    }
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
    gradient?.fillRange(0, gradient.length, 0.0);
    _run(
      data,
      observed,
      timeSteps,
      perElement,
      nTime,
      response,
      totals,
      perTrial,
      gradient,
    );
    return totals;
  }

  /// One belief-propagation step for every element from the current state.
  /// [observations] has one value per input node, or with [perElement] one
  /// per input node and element (`observations[k * batchSize + e]`).
  ///
  /// The step starts with the prediction sweep, unless [predict] already ran
  /// it for this trial (as in [Network.step]).
  void step(
    Float64List observations, {
    Float64List? observed,
    double timeStep = 1.0,
    bool perElement = false,
  }) {
    _checkTemplate();
    final per = perElement ? _b : 1;
    final width = _nCols * per, flags = _inputs.length * per;
    if (observations.length != width) {
      throw ArgumentError('Expected $width observation(s)');
    }
    if (observed != null && observed.length != flags) {
      throw ArgumentError('Expected $flags observed flag(s)');
    }
    if (_pendingPrediction != null && _pendingPrediction != timeStep) {
      throw StateError(
        'predict() ran with timeStep $_pendingPrediction, step() got $timeStep',
      );
    }
    final predicted = _pendingPrediction != null;
    _pendingPrediction = null;
    for (var lo = 0; lo < _b; lo += blockSize) {
      final hi = lo + blockSize < _b ? lo + blockSize : _b;
      _propagate(
        observations,
        observed,
        0,
        0,
        timeStep,
        perElement,
        lo,
        hi,
        predicted: predicted,
      );
    }
  }

  /// Run only the prediction sweep for the next trial, for every element, so
  /// that the predicted beliefs (`expectedMean`, `expectedPrecision`, ...)
  /// refer to the upcoming observation: e.g. to draw a choice that decides
  /// what the agent observes (`HgfModel.simulateChoices` in hgf_fit). The
  /// next [step] reuses it; calling [predict] twice without a [step] is a
  /// no-op. As [Network.predict].
  void predict({double timeStep = 1.0}) {
    _checkTemplate();
    if (_pendingPrediction != null) {
      if (_pendingPrediction != timeStep) {
        throw StateError(
          'A prediction for timeStep $_pendingPrediction is pending',
        );
      }
      return;
    }
    for (var lo = 0; lo < _b; lo += blockSize) {
      final hi = lo + blockSize < _b ? lo + blockSize : _b;
      if (_k != 0) {
        _tPredictSweep(timeStep, lo, hi);
      } else {
        final predNode = _predNode, predStep = _predStep;
        for (var k = 0; k < predNode.length; k++) {
          _dispatch(predStep[k], predNode[k], timeStep, lo, hi);
        }
      }
    }
    _pendingPrediction = timeStep;
  }

  /// Time step of a [predict] not yet consumed by [step], if any.
  double? _pendingPrediction;

  int _prepareRun(
    Float64List data,
    Float64List? observed,
    Float64List? timeSteps,
    bool perElement,
    bool append,
  ) {
    _checkTemplate();
    final per = perElement ? _b : 1;
    final width = _nCols * per;
    if (data.length % width != 0) {
      throw ArgumentError(
        'data length ${data.length} is not a multiple of $width '
        '($_nCols input column(s)${perElement ? ' x $_b elements' : ''})',
      );
    }
    final nTime = data.length ~/ width;
    if (observed != null && observed.length != nTime * _inputs.length * per) {
      throw ArgumentError(
        'observed must have one flag per time step and input node'
        '${perElement ? ' and element' : ''}',
      );
    }
    if (timeSteps != null && timeSteps.length != nTime) {
      throw ArgumentError('timeSteps must have one entry per time step');
    }
    if (!append) reset();
    if (_pendingPrediction != null) {
      throw StateError('predict() is pending; call step() or reset() first');
    }
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
    Float64List? gradient,
  ) {
    final b = _b, k = _k, per = perElement ? b : 1;
    final width = _nCols * per, flags = _inputs.length * per;
    final partials = _partials;
    for (var lo = 0; lo < b; lo += blockSize) {
      final hi = lo + blockSize < b ? lo + blockSize : b;
      for (var t = 0; t < nTime; t++) {
        _propagate(
          data,
          observed,
          t * width,
          t * flags,
          timeSteps == null ? 1.0 : timeSteps[t],
          perElement,
          lo,
          hi,
        );
        if (gradient != null) {
          for (var e = lo; e < hi; e++) {
            partials._bind(e, gradient, e * k);
            final s = response!.batchSurprisePartials(this, t, e, partials);
            totals![e] += s;
            if (perTrial != null) perTrial[t * b + e] = s;
          }
        } else if (response != null) {
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
    int observedOffset,
    double timeStep,
    bool perElement,
    int lo,
    int hi, {
    bool predicted = false,
  }) {
    if (_k != 0) {
      _tPropagate(
        data,
        observed,
        offset,
        observedOffset,
        timeStep,
        perElement,
        lo,
        hi,
        predicted: predicted,
      );
      return;
    }
    if (!predicted) {
      final predNode = _predNode, predStep = _predStep;
      for (var k = 0; k < predNode.length; k++) {
        _dispatch(predStep[k], predNode[k], timeStep, lo, hi);
      }
    }
    _observe(data, observed, offset, observedOffset, perElement, lo, hi);
    final updNode = _updNode, updStep = _updStep;
    for (var k = 0; k < updNode.length; k++) {
      _dispatch(updStep[k], updNode[k], timeStep, lo, hi);
    }
  }

  /// The observations of one time slice into the input nodes, for elements
  /// `[lo, hi)`.
  @pragma('vm:prefer-inline')
  @pragma('vm:unsafe:no-bounds-checks')
  void _observe(
    Float64List data,
    Float64List? observed,
    int offset,
    int observedOffset,
    bool perElement,
    int lo,
    int hi,
  ) {
    if (_hasVectorInputs) {
      _observeVectors(
        data,
        observed,
        offset,
        observedOffset,
        perElement,
        lo,
        hi,
      );
      return;
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
  }

  /// [_observe] with ef-state or categorical inputs: each takes its columns
  /// into its vector mean (and the first into the scalar mean), as
  /// [Network]'s `_observeVectors` does.
  @pragma('vm:never-inline')
  void _observeVectors(
    Float64List data,
    Float64List? observed,
    int offset,
    int observedOffset,
    bool perElement,
    int lo,
    int hi,
  ) {
    final inputs = _inputs, b = _b, mean = _mean, obs = _observed;
    final x = _vMean;
    for (var q = 0; q < inputs.length; q++) {
      final idx = inputs[q], base = idx * b, col = _inputCol[q];
      final kind = _kind[idx];
      final vector = kind == _kindEf || kind == _kindCategorical;
      final dim = _inputDim[q], v0 = vector ? _vMeanStart[idx] : 0;
      // Column c of element e is at `src + c * stride`.
      final stride = perElement ? b : 1;
      for (var e = lo; e < hi; e++) {
        final src = perElement ? offset + col * b + e : offset + col;
        mean[base + e] = data[src];
        if (vector) {
          for (var j = 0; j < dim; j++) {
            x[(v0 + j) * b + e] = data[src + j * stride];
          }
        }
        obs[base + e] = observed == null
            ? 1.0
            : perElement
            ? observed[observedOffset + q * b + e]
            : observed[observedOffset + q];
      }
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
      default:
        _bDispatchVector(step, node, lo, hi);
    }
  }
}

/// Where a response function reports the partial derivatives of one
/// trial's surprise ([ResponseFunction.batchSurprisePartials]); the
/// [BatchNetwork] combines them with the beliefs' tangents into the
/// gradient, so a response function never sees the tangent directions.
final class SurprisePartials {
  SurprisePartials._(this._net);

  final BatchNetwork _net;
  int _element = 0, _offset = 0;
  Float64List _gradient = _f64;

  void _bind(int element, Float64List gradient, int offset) {
    _element = element;
    _gradient = gradient;
    _offset = offset;
  }

  /// The surprise changes by [derivative] per unit of [field] on [node]
  /// (of the element being scored). [field] must carry tangents
  /// (`BatchNetwork.tangentFields`).
  @pragma('vm:unsafe:no-bounds-checks')
  void add(int node, NodeField field, double derivative) {
    final net = _net;
    final col = net._tangentColumn(field);
    if (col == null) net._tangentSlot(field); // throws
    if (node < 0 || node >= net._n) {
      throw RangeError.range(node, 0, net._n - 1, 'node');
    }
    if (derivative == 0.0) return;
    final k = net._k, b = net._b, act = net._activeDirs;
    final g = _gradient, o = _offset, base = node * k * b + _element;
    for (var a = 0; a < net._nActive; a++) {
      final d = act[a];
      g[o + d] += derivative * col![base + d * b];
    }
  }

  /// The surprise changes by [derivative] per unit of a quantity outside
  /// the network whose derivative along direction `d` is
  /// `tangents[offset + d]` (e.g. a response model's own parameter).
  void addTangents(double derivative, Float64List tangents, int offset) {
    if (derivative == 0.0) return;
    final g = _gradient, o = _offset;
    for (var d = 0; d < _net._k; d++) {
      g[o + d] += derivative * tangents[offset + d];
    }
  }
}
