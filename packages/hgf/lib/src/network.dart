import 'dart:math' as math;
import 'dart:typed_data';

import 'coupling.dart';
import 'math.dart';
import 'response.dart';
import 'trajectories.dart';
import 'types.dart';

part 'batch/batch_network.dart';
part 'batch/batch_updates.dart';
part 'update_sequence.dart';
part 'updates/categorical.dart';
part 'updates/exponential.dart';
part 'updates/prediction.dart';
part 'updates/prediction_error.dart';
part 'updates/posterior.dart';

// Integer step codes for the hot-loop switch (indexes of [UpdateStep]).
const int _kPredictionContinuous = 0;
const int _kPredictionContinuousMeanField = 1;
const int _kPredictionBinary = 2;
const int _kPosteriorContinuous = 3;
const int _kPosteriorContinuousMeanField = 4;
const int _kPosteriorContinuousEhgf = 5;
const int _kPosteriorContinuousEhgfMeanField = 6;
const int _kPosteriorContinuousUnbounded = 7;
const int _kPredictionErrorContinuous = 8;
const int _kPredictionErrorBinary = 9;
const int _kPredictionErrorEfFixed = 10;
const int _kPredictionErrorEfDynamic = 11;
const int _kPosteriorEfDynamic = 12;
const int _kPredictionErrorCategorical = 13;
const int _kPosteriorCategorical = 14;

const int _kindContinuous = 0;
const int _kindBinary = 1;
const int _kindConstant = 2;
const int _kindEf = 3;
const int _kindCategorical = 4;

/// `_couplingFn` code of a [CustomCoupling] (named ones use their index).
const int _kCustomCoupling = 255;

// Shared empty placeholders: non-late fields avoid an initialisation check on
// every access in the hot loop.
final Uint8List _u8 = Uint8List(0);
final Int32List _i32 = Int32List(0);
final Float64List _f64 = Float64List(0);
final _Unbounded _noUnbounded = _Unbounded(0);

/// A node as declared through [Network.addNodes], before compilation.
final class _NodeDef {
  _NodeDef(this.kind, this.coupling, this.params);

  final NodeKind kind;
  final Coupling coupling;
  final Map<NodeField, double> params;

  /// The number of input columns: an ef-state node's observation dimension,
  /// a categorical-state node's number of categories.
  int dimension = 1;

  /// ef-state only: the initial sufficient statistics (`null`: pyhgf's
  /// default).
  List<double>? xis;

  final List<int> valueParents = [];
  final List<double> valueParentStrengths = [];
  final List<int> valueChildren = [];
  final List<double> valueChildStrengths = [];
  final List<int> volatilityParents = [];
  final List<double> volatilityParentStrengths = [];
  final List<int> volatilityChildren = [];
  final List<double> volatilityChildStrengths = [];

  bool get hasChildren =>
      valueChildren.isNotEmpty || volatilityChildren.isNotEmpty;
}

/// A predictive-coding network of probabilistic nodes (the pyhgf `Network`).
///
/// Build it with [addNodes], then filter observations with [inputData] (a
/// whole series, starting from the initial beliefs) or [step] (one
/// observation at a time, from the current beliefs, for real-time use).
///
/// ```dart
/// final hgf = Network()
///   ..addNodes(precision: 1e4)                       // 0: input
///   ..addNodes(valueChildren: [0], mean: 1.04)       // 1: value parent
///   ..addNodes(volatilityChildren: [1]);             // 2: volatility parent
/// final traj = hgf.inputSeries(series);
/// final mu1 = traj.series(1, NodeField.mean);
/// ```
///
/// The network compiles on first use into flat typed arrays (one
/// [Float64List] per attribute, CSR edge lists, an int-coded update
/// sequence), so propagation allocates nothing per time step.
class Network {
  Network({
    this.volatilityUpdates = VolatilityUpdates.unbounded,
    this.meanFieldUpdates = false,
    this.ehgfMgfCorrection = false,
    this.maxPosteriorPrecision = 1e10,
    this.precisionClippingValue = 1e-6,
  });

  /// The posterior update for nodes with volatility children.
  final VolatilityUpdates volatilityUpdates;

  /// Use the mean-field value-coupling updates instead of the relaxed (rHGF)
  /// ones.
  final bool meanFieldUpdates;

  /// **Diverges from pyhgf (off by default).** Adds the moment-generating
  /// function terms of the relaxed prediction to the eHGF volatility
  /// re-prediction, which pyhgf (JAX and rshgf, as of 0.3.2) leaves out.
  ///
  /// The eHGF precision update re-predicts each volatility child's
  /// volatility from the parent's new posterior mean μ_b:
  /// `ν = t·exp(κ μ_b + ω̃)`, with `ω̃ = ω + Σ_{p≠b} κ_p μ̂_p`. The relaxed
  /// prediction of that child used
  /// `Ω = t·exp(ω + Σ_p κ_p μ̂_p + Σ_p κ_p² / (2 π̂_p))`. Without the
  /// `κ²/(2π̂)` terms, ν at μ_b = μ̂_b is smaller than Ω, so the
  /// re-predicted precision isn't the child's predicted precision (as the
  /// pyhgf docstring says it is). The increment is then floored at zero on
  /// most trials, the volatility parents never gain precision, and 3-level
  /// continuous HGFs diverge (NaN after 44 trials on the USD/CHF example).
  /// With this flag, ω̃ also gets `Σ_p κ_p² / (2 π̂_p)` over all of the child's
  /// volatility parents, so ν equals Ω at μ_b = μ̂_b.
  ///
  /// Only affects [VolatilityUpdates.eHGF] with relaxed updates (the
  /// mean-field prediction has no such terms). Checked against pyhgf's JAX
  /// backend patched with the same change (`*_mgf_correction` fixtures).
  final bool ehgfMgfCorrection;

  /// Upper bound applied to every posterior precision write.
  double maxPosteriorPrecision;

  /// Binary predicted means are clipped to `[v, 1 - v]`.
  double precisionClippingValue;

  final List<_NodeDef> _defs = [];
  bool _compiled = false;

  /// Number of nodes.
  int get nNodes => _defs.length;

  /// The kind of node [i].
  NodeKind kindOf(int i) => _defs[i].kind;

  // ---------------------------------------------------------------------------
  // Construction
  // ---------------------------------------------------------------------------

  /// Add [nNodes] node(s) of [kind].
  ///
  /// Edges are given as node indexes; the reciprocal edge is added to the
  /// other node. Optional `*Strengths` lists set the coupling strengths
  /// (default 1.0), in the same order as the indexes. [coupling] is the
  /// value-coupling function from the new node(s) to their value children:
  /// a named [CouplingFn] or a [CustomCoupling].
  ///
  /// The scalar parameters override the node kind's defaults. As in pyhgf,
  /// continuous nodes without children (inputs) always have
  /// `autoconnectionStrength = 0` and `tonicVolatility = 0`.
  Network addNodes({
    NodeKind kind = NodeKind.continuousState,
    int nNodes = 1,
    List<int>? valueParents,
    List<int>? valueChildren,
    List<int>? volatilityParents,
    List<int>? volatilityChildren,
    List<double>? valueParentsStrengths,
    List<double>? valueChildrenStrengths,
    List<double>? volatilityParentsStrengths,
    List<double>? volatilityChildrenStrengths,
    Coupling coupling = CouplingFn.linear,
    double? mean,
    double? expectedMean,
    double? precision,
    double? expectedPrecision,
    double? tonicVolatility,
    double? tonicDrift,
    double? autoconnectionStrength,
    Map<NodeField, double>? parameters,
  }) {
    if (nNodes < 1) throw ArgumentError.value(nNodes, 'nNodes');
    final params = <NodeField, double>{
      ...?parameters,
      NodeField.mean: ?mean,
      NodeField.expectedMean: ?expectedMean,
      NodeField.precision: ?precision,
      NodeField.expectedPrecision: ?expectedPrecision,
      NodeField.tonicVolatility: ?tonicVolatility,
      NodeField.tonicDrift: ?tonicDrift,
      NodeField.autoconnectionStrength: ?autoconnectionStrength,
    };
    List<double> strengths(List<int>? idxs, List<double>? s, String name) {
      if (idxs == null) return const [];
      if (s == null) return List.filled(idxs.length, 1.0);
      if (s.length != idxs.length) {
        throw ArgumentError('$name must have one strength per index');
      }
      return s;
    }

    final vpS = strengths(valueParents, valueParentsStrengths, 'valueParents');
    final vcS = strengths(
      valueChildren,
      valueChildrenStrengths,
      'valueChildren',
    );
    final volpS = strengths(
      volatilityParents,
      volatilityParentsStrengths,
      'volatilityParents',
    );
    final volcS = strengths(
      volatilityChildren,
      volatilityChildrenStrengths,
      'volatilityChildren',
    );
    if (kind == NodeKind.efState) {
      throw ArgumentError(
        'Add ef-state nodes with addEfState (they need a dimension and a '
        'learning mode)',
      );
    }
    if (kind == NodeKind.categoricalState) {
      throw ArgumentError(
        'Add categorical-state nodes with addCategoricalState (it also adds '
        'their binary HGFs)',
      );
    }
    if (kind == NodeKind.binaryState &&
        (volatilityParents != null || volatilityChildren != null)) {
      throw ArgumentError('Binary state nodes take value edges only');
    }
    if (kind == NodeKind.constantState) {
      if (valueParents != null ||
          volatilityParents != null ||
          volatilityChildren != null) {
        throw ArgumentError('Constant state nodes take value children only');
      }
      if (params.isNotEmpty) {
        // pyhgf ignores them too (the bias is 1 on every backend); scale the
        // bias with the coupling strength instead.
        throw ArgumentError(
          'Constant state nodes take no parameters; set the bias through the '
          'value coupling strength',
        );
      }
      // Always linear, as in pyhgf (the bias stays 1 on every backend).
      coupling = CouplingFn.linear;
    }

    for (var n = 0; n < nNodes; n++) {
      final id = _defs.length;
      final def = _NodeDef(kind, coupling, Map.of(params));
      _defs.add(def);
      void link(
        List<int>? idxs,
        List<double> s,
        List<int> mine,
        List<double> mineS,
        List<int> Function(_NodeDef) theirs,
        List<double> Function(_NodeDef) theirsS,
      ) {
        if (idxs == null) return;
        for (var k = 0; k < idxs.length; k++) {
          final other = idxs[k];
          if (other < 0 || other >= id) {
            throw ArgumentError(
              'Node $id references node $other, which does not exist yet',
            );
          }
          mine.add(other);
          mineS.add(s[k]);
          final o = _defs[other];
          if (o.kind == NodeKind.constantState &&
              !identical(theirs(o), o.valueChildren)) {
            throw ArgumentError(
              'Node $other is a constant state node, which can only be a '
              'value parent',
            );
          }
          if (o.kind == NodeKind.efState &&
              !identical(theirs(o), o.valueParents)) {
            throw ArgumentError(
              'Node $other is an ef-state node, which only takes value '
              'parents',
            );
          }
          if (o.kind == NodeKind.categoricalState) {
            throw ArgumentError(
              'Node $other is a categorical state node; its edges are set '
              'by addCategoricalState',
            );
          }
          if (!theirs(o).contains(id)) {
            theirs(o).add(id);
            theirsS(o).add(s[k]);
          }
        }
      }

      link(
        valueParents,
        vpS,
        def.valueParents,
        def.valueParentStrengths,
        (o) => o.valueChildren,
        (o) => o.valueChildStrengths,
      );
      link(
        valueChildren,
        vcS,
        def.valueChildren,
        def.valueChildStrengths,
        (o) => o.valueParents,
        (o) => o.valueParentStrengths,
      );
      link(
        volatilityParents,
        volpS,
        def.volatilityParents,
        def.volatilityParentStrengths,
        (o) => o.volatilityChildren,
        (o) => o.volatilityChildStrengths,
      );
      link(
        volatilityChildren,
        volcS,
        def.volatilityChildren,
        def.volatilityChildStrengths,
        (o) => o.volatilityParents,
        (o) => o.volatilityParentStrengths,
      );
    }
    _compiled = false;
    return this;
  }

  /// Add [nNodes] exponential-family state node(s) (pyhgf's `ef-state`):
  /// input nodes that track the expected sufficient statistics ξ
  /// ([VectorField.xis]) of a normal distribution of their observations, a
  /// multivariate one when [dimension] > 1. Each takes [dimension] columns
  /// of the input data.
  ///
  /// With [EfLearning.fixed], ξ moves toward each observation's sufficient
  /// statistics at the fixed rate `1 / (1 + nus)`. With the HGF learning
  /// modes, the node also gets one chain of continuous nodes per sufficient
  /// statistic, added right after it (as pyhgf does, in reverse node order
  /// when [nNodes] > 1), and the rate is implied by how much those chains
  /// learn; [nus] is then only the initial value.
  ///
  /// [xis] overrides the initial ξ (`d + d(d+1)/2` values; default: the
  /// sufficient statistics of N(0, I)). pyhgf ignores a `xis` parameter.
  Network addEfState({
    int nNodes = 1,
    int dimension = 1,
    EfLearning learning = EfLearning.fixed,
    double nus = 3.0,
    List<double>? xis,
  }) {
    if (nNodes < 1) throw ArgumentError.value(nNodes, 'nNodes');
    if (dimension < 1) throw ArgumentError.value(dimension, 'dimension');
    final nSs = _efSufficientStatisticsCount(dimension);
    if (xis != null && xis.length != nSs) {
      throw ArgumentError.value(
        xis,
        'xis',
        'Must have $nSs values for dimension $dimension',
      );
    }
    final initialXis = List<double>.unmodifiable(
      xis ?? _efDefaultXis(dimension),
    );
    final first = _defs.length;
    for (var k = 0; k < nNodes; k++) {
      _defs.add(
        _NodeDef(NodeKind.efState, CouplingFn.linear, {NodeField.nus: nus})
          ..dimension = dimension
          ..xis = initialXis,
      );
    }
    _compiled = false;
    if (learning.levels == 0) return this;
    for (var idx = first + nNodes - 1; idx >= first; idx--) {
      for (var i = 0; i < nSs; i++) {
        addNodes(valueChildren: [idx]);
        addNodes(valueChildren: [_defs.length - 1]);
        if (learning.levels >= 2) {
          addNodes(volatilityChildren: [_defs.length - 1]);
        }
        if (learning.levels >= 3) {
          addNodes(volatilityChildren: [_defs.length - 1]);
        }
      }
    }
    return this;
  }

  /// `d + d(d+1)/2`: the length of a d-dimensional normal's sufficient
  /// statistics (the mean, then the lower triangle of x xᵀ).
  static int _efSufficientStatisticsCount(int d) => d + d * (d + 1) ~/ 2;

  /// The sufficient statistics of N(0, I): zeros, then the lower triangle of
  /// the identity row by row.
  static List<double> _efDefaultXis(int d) => [
    for (var i = 0; i < d; i++) 0.0,
    for (var r = 0; r < d; r++)
      for (var c = 0; c <= r; c++) r == c ? 1.0 : 0.0,
  ];

  /// Add a categorical state node (pyhgf's `categorical-state`): an input
  /// node observing one of [nCategories] categories, as [nCategories] input
  /// columns (one-hot; `1` marks the category observed).
  ///
  /// As in pyhgf, the node comes with one binary HGF per category and a
  /// shared volatility parent, `2n + 2` nodes in all, in this order (with `c`
  /// the categorical node's index and `n` = [nCategories]):
  ///
  /// - `c`: the categorical node;
  /// - `c + 1 .. c + n`: binary nodes, its value parents, one per category;
  /// - `c + n + 1 .. c + 2n`: continuous nodes, the value parent of each
  ///   binary node in turn;
  /// - `c + 2n + 1`: a continuous node, the volatility parent of all of
  ///   those.
  ///
  /// Each trial, every binary node observes its category's indicator, the
  /// binary HGFs update as usual, and the categorical node then derives a
  /// Dirichlet distribution over the categories ([VectorField.alpha]) from
  /// how much each one learned; see [NodeField.klDivergence] and
  /// [NodeField.surprise] for what it records.
  ///
  /// The other parameters set the initial beliefs of the implied nodes;
  /// their defaults are pyhgf's: binary nodes `mean` [binaryMean] (`1/n`) and
  /// `precision` [binaryPrecision]; value parents `mean` [valueMean]
  /// (`−ln(n − 1)`, so that `σ(mean) = 1/n`), `precision` [valuePrecision],
  /// `tonicVolatility` [valueTonicVolatility]; the volatility parent likewise.
  /// pyhgf ignores its `binary_parameters` argument and always uses these
  /// defaults. Change any parameter later with [setInitial].
  ///
  /// Unlike pyhgf, several categorical nodes can share a network (call this
  /// once per node): pyhgf's `n_nodes > 1` builds a broken graph.
  Network addCategoricalState({
    int nCategories = 4,
    double? binaryMean,
    double binaryPrecision = 1.0,
    double? valueMean,
    double valuePrecision = 1.0,
    double valueTonicVolatility = -4.0,
    double volatilityMean = 0.0,
    double volatilityPrecision = 1.0,
    double volatilityTonicVolatility = -4.0,
  }) {
    if (nCategories < 2) {
      throw ArgumentError.value(
        nCategories,
        'nCategories',
        'Must be at least 2',
      );
    }
    final n = nCategories;
    final c = _defs.length;
    final cat = _NodeDef(NodeKind.categoricalState, CouplingFn.linear, {
      NodeField.mean: 1.0 / n,
    })..dimension = n;
    _defs.add(cat);
    for (var i = 0; i < n; i++) {
      // Wired here rather than through addNodes, which rejects edges to
      // categorical nodes.
      final b = _defs.length;
      _defs.add(
        _NodeDef(NodeKind.binaryState, CouplingFn.linear, {
            NodeField.mean: binaryMean ?? 1.0 / n,
            NodeField.precision: binaryPrecision,
          })
          ..valueChildren.add(c)
          ..valueChildStrengths.add(1.0),
      );
      cat
        ..valueParents.add(b)
        ..valueParentStrengths.add(1.0);
    }
    for (var i = 0; i < n; i++) {
      addNodes(
        valueChildren: [c + 1 + i],
        mean: valueMean ?? -math.log(n - 1),
        precision: valuePrecision,
        tonicVolatility: valueTonicVolatility,
      );
    }
    addNodes(
      volatilityChildren: [for (var i = 0; i < n; i++) c + n + 1 + i],
      mean: volatilityMean,
      precision: volatilityPrecision,
      tonicVolatility: volatilityTonicVolatility,
    );
    _compiled = false;
    return this;
  }

  // ---------------------------------------------------------------------------
  // Compiled representation
  // ---------------------------------------------------------------------------

  int _n = 0;
  Uint8List _kind = _u8;
  Uint8List _couplingFn = _u8; // CouplingFn.index or _kCustomCoupling
  List<Coupling> _couplings = const []; // per node, declared on the parent
  Uint8List _isLeaf = _u8; // no value and no volatility children
  Int32List _inputs = _i32;

  // Input columns: input k reads `_inputDim[k]` values from column
  // `_inputCol[k]` of each data row (more than one for a multivariate
  // ef-state input or a categorical one); `_nInputColumns` is the row width.
  Int32List _inputDim = _i32, _inputCol = _i32;
  int _nInputColumns = 0;
  bool _hasVectorInputs = false;

  // CSR edge lists. Strengths are stored on the side that reads them: a
  // child's value/volatility *parents* carry the child-side strengths
  // (pyhgf's `*_coupling_parents`), a parent's *children* the parent-side
  // ones (`*_coupling_children`).
  Int32List _vpStart = _i32, _vpIdx = _i32, _vcStart = _i32, _vcIdx = _i32;
  Int32List _volpStart = _i32,
      _volpIdx = _i32,
      _volcStart = _i32,
      _volcIdx = _i32;
  Float64List _vpK = _f64, _vcK = _f64, _volpK = _f64, _volcK = _f64;

  // Current state, one column per field.
  Float64List _mean = _f64,
      _expectedMean = _f64,
      _precision = _f64,
      _expectedPrecision = _f64,
      _conditionalExpectedPrecision = _f64,
      _observed = _f64,
      _tonicVolatility = _f64,
      _tonicDrift = _f64,
      _autoconnectionStrength = _f64,
      _currentVariance = _f64,
      _effectivePrecision = _f64,
      _valuePredictionError = _f64,
      _volatilityPredictionError = _f64,
      _nus = _f64,
      _klDivergence = _f64,
      _surprise = _f64;

  // Vector side store, outside the scalar hot loop: one CSR-like array per
  // [VectorField] (node i's vector is `[start[i], start[i + 1])`), read only
  // by kind-specific steps. `xis` and `observation_ss` share offsets.
  Int32List _vMeanStart = _i32, _ssStart = _i32, _alphaStart = _i32;
  Float64List _vMean = _f64, _xis = _f64, _ss = _f64, _alpha = _f64;
  Float64List _vMeanInitial = _f64,
      _xisInitial = _f64,
      _ssInitial = _f64,
      _alphaInitial = _f64;

  // Scratch for the categorical posterior (the new α, before the KL
  // divergence from the old one), sized to the most categories.
  Float64List _alphaScratch = _f64;

  // Initial state: all fields, field-major (`field.index * n + node`).
  Float64List _initial = _f64;

  Int32List _predNode = _i32, _updNode = _i32;
  Uint8List _predStep = _u8, _updStep = _u8;

  // Scratch for the unbounded update (sized to the widest volatility fan-out).
  _Unbounded _u = _noUnbounded;

  /// Compile the network. Called automatically by [inputData], [step] and
  /// the accessors; call it explicitly to front-load the cost.
  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  @pragma('dart2js:prefer-inline')
  void build() {
    if (!_compiled) _compile();
  }

  void _compile() {
    final n = _n = _defs.length;
    if (n == 0) throw StateError('The network has no nodes');

    _kind = Uint8List(n);
    _couplingFn = Uint8List(n);
    _couplings = List.unmodifiable([for (final d in _defs) d.coupling]);
    _isLeaf = Uint8List(n);
    final inputs = <int>[];
    for (var i = 0; i < n; i++) {
      final d = _defs[i];
      _kind[i] = switch (d.kind) {
        NodeKind.continuousState => _kindContinuous,
        NodeKind.binaryState => _kindBinary,
        NodeKind.constantState => _kindConstant,
        NodeKind.efState => _kindEf,
        NodeKind.categoricalState => _kindCategorical,
      };
      if (d.kind == NodeKind.constantState && d.valueChildren.isEmpty) {
        throw StateError('Constant state node $i has no value children');
      }
      final c = d.coupling;
      _couplingFn[i] = c is CouplingFn ? c.index : _kCustomCoupling;
      if (!d.hasChildren) {
        _isLeaf[i] = 1;
        inputs.add(i);
      }
    }
    _inputs = Int32List.fromList(inputs);
    _inputIdxsView = List.unmodifiable(inputs);
    _inputDim = Int32List(inputs.length);
    _inputCol = Int32List(inputs.length);
    var col = 0;
    _hasVectorInputs = false;
    for (var k = 0; k < inputs.length; k++) {
      final d = _defs[inputs[k]];
      final vector =
          d.kind == NodeKind.efState || d.kind == NodeKind.categoricalState;
      _inputCol[k] = col;
      _inputDim[k] = vector ? d.dimension : 1;
      col += _inputDim[k];
      if (vector) _hasVectorInputs = true;
    }
    _nInputColumns = col;
    _compileVectors();

    // CSR offsets, indexes and strengths. Every hot-path column is assigned
    // straight from a typed-data constructor (no records, no views):
    // dart2wasm's type-flow analysis then sees the exact list class and
    // inlines `[]`. Through a record it lost the type, and each strength read
    // became a virtual call returning a boxed double (W1).
    _vpStart = _csrStart(n, (d) => d.valueParents);
    _vpIdx = _csrIdx(_vpStart, (d) => d.valueParents);
    _vpK = _csrK(_vpStart, (d) => d.valueParentStrengths);
    _vcStart = _csrStart(n, (d) => d.valueChildren);
    _vcIdx = _csrIdx(_vcStart, (d) => d.valueChildren);
    _vcK = _csrK(_vcStart, (d) => d.valueChildStrengths);
    _volpStart = _csrStart(n, (d) => d.volatilityParents);
    _volpIdx = _csrIdx(_volpStart, (d) => d.volatilityParents);
    _volpK = _csrK(_volpStart, (d) => d.volatilityParentStrengths);
    _volcStart = _csrStart(n, (d) => d.volatilityChildren);
    _volcIdx = _csrIdx(_volcStart, (d) => d.volatilityChildren);
    _volcK = _csrK(_volcStart, (d) => d.volatilityChildStrengths);

    // Initial state: kind defaults, then user parameters, then (as pyhgf's
    // `input_idxs`) inputs get no autoconnection and no tonic volatility.
    final fieldCount = NodeField.values.length;
    _initial = Float64List(fieldCount * n);
    void init(NodeField f, int i, double v) => _initial[f.index * n + i] = v;
    for (var i = 0; i < n; i++) {
      final d = _defs[i];
      final defaults = _defaults(d.kind);
      for (final f in NodeField.values) {
        init(f, i, defaults[f] ?? 0.0);
      }
      d.params.forEach((f, v) => init(f, i, v));
      if (d.kind == NodeKind.continuousState && _isLeaf[i] == 1) {
        init(NodeField.autoconnectionStrength, i, 0.0);
        init(NodeField.tonicVolatility, i, 0.0);
      }
    }

    _mean = Float64List(n);
    _expectedMean = Float64List(n);
    _precision = Float64List(n);
    _expectedPrecision = Float64List(n);
    _conditionalExpectedPrecision = Float64List(n);
    _observed = Float64List(n);
    _tonicVolatility = Float64List(n);
    _tonicDrift = Float64List(n);
    _autoconnectionStrength = Float64List(n);
    _currentVariance = Float64List(n);
    _effectivePrecision = Float64List(n);
    _valuePredictionError = Float64List(n);
    _volatilityPredictionError = Float64List(n);
    _nus = Float64List(n);
    _klDivergence = Float64List(n);
    _surprise = Float64List(n);

    assert(UpdateStep.predictionContinuous.index == _kPredictionContinuous);
    assert(UpdateStep.predictionErrorBinary.index == _kPredictionErrorBinary);
    final seq = _buildUpdateSequence();
    _predNode = Int32List.fromList([for (final s in seq.predictions) s.$1]);
    _predStep = Uint8List.fromList([
      for (final s in seq.predictions) s.$2.index,
    ]);
    _updNode = Int32List.fromList([for (final s in seq.updates) s.$1]);
    _updStep = Uint8List.fromList([for (final s in seq.updates) s.$2.index]);
    _updateSequence = seq;

    var maxVolChildren = 0;
    for (var i = 0; i < n; i++) {
      maxVolChildren = math.max(
        maxVolChildren,
        _volcStart[i + 1] - _volcStart[i],
      );
    }
    _u = _Unbounded(maxVolChildren);

    _checkInvariants();
    _compiled = true;
    reset();
  }

  /// Offsets and initial values of the vector side store.
  void _compileVectors() {
    final n = _n;
    final mStart = Int32List(n + 1),
        sStart = Int32List(n + 1),
        aStart = Int32List(n + 1);
    var maxCategories = 0;
    for (var i = 0; i < n; i++) {
      final d = _defs[i];
      final ef = d.kind == NodeKind.efState;
      final cat = d.kind == NodeKind.categoricalState;
      mStart[i + 1] = mStart[i] + (ef || cat ? d.dimension : 0);
      sStart[i + 1] =
          sStart[i] + (ef ? _efSufficientStatisticsCount(d.dimension) : 0);
      aStart[i + 1] = aStart[i] + (cat ? d.dimension : 0);
      if (cat) {
        maxCategories = math.max(maxCategories, d.dimension);
        _checkCategoricalParents(i, d);
      }
      if (!ef) continue;
      final vp = d.valueParents;
      if (vp.isNotEmpty &&
          vp.length != _efSufficientStatisticsCount(d.dimension)) {
        throw StateError(
          'ef-state node $i has ${vp.length} value parent(s); with HGF '
          'learning it needs one per sufficient statistic '
          '(${_efSufficientStatisticsCount(d.dimension)})',
        );
      }
      for (final p in vp) {
        final pd = _defs[p];
        if (pd.kind != NodeKind.continuousState || pd.valueParents.isEmpty) {
          throw StateError(
            'Value parent $p of ef-state node $i must be a continuous node '
            'with a value parent of its own',
          );
        }
      }
    }
    _vMeanStart = mStart;
    _ssStart = sStart;
    _alphaStart = aStart;
    _vMean = Float64List(mStart[n]);
    _xis = Float64List(sStart[n]);
    _ss = Float64List(sStart[n]);
    _alpha = Float64List(aStart[n]);
    _vMeanInitial = Float64List(mStart[n]);
    _ssInitial = Float64List(sStart[n]);
    _xisInitial = Float64List(sStart[n]);
    // α starts at 1 (a flat Dirichlet), the mean at 1/n per category.
    _alphaInitial = Float64List(aStart[n])..fillRange(0, aStart[n], 1.0);
    _alphaScratch = Float64List(maxCategories);
    for (var i = 0; i < n; i++) {
      final d = _defs[i];
      final xis = d.xis;
      if (xis != null) _xisInitial.setAll(sStart[i], xis);
      if (d.kind == NodeKind.categoricalState) {
        _vMeanInitial.fillRange(mStart[i], mStart[i + 1], 1.0 / d.dimension);
      }
    }
  }

  /// A categorical node's value parents must be one binary node per
  /// category, each with a continuous (first) value parent: the posterior
  /// reads that parent's mean.
  void _checkCategoricalParents(int i, _NodeDef d) {
    if (d.valueParents.length != d.dimension) {
      throw StateError(
        'Categorical state node $i needs one binary value parent per '
        'category (${d.dimension}), has ${d.valueParents.length}',
      );
    }
    for (final b in d.valueParents) {
      final bd = _defs[b];
      if (bd.kind != NodeKind.binaryState ||
          bd.valueParents.isEmpty ||
          _defs[bd.valueParents.first].kind != NodeKind.continuousState) {
        throw StateError(
          'Value parent $b of categorical state node $i must be a binary '
          'node whose first value parent is a continuous node',
        );
      }
    }
  }

  Int32List _csrStart(int n, List<int> Function(_NodeDef) idx) {
    final start = Int32List(n + 1);
    for (var i = 0; i < n; i++) {
      start[i + 1] = start[i] + idx(_defs[i]).length;
    }
    return start;
  }

  Int32List _csrIdx(Int32List start, List<int> Function(_NodeDef) idx) {
    final out = Int32List(start.last);
    for (var i = 0; i < _defs.length; i++) {
      out.setAll(start[i], idx(_defs[i]));
    }
    return out;
  }

  Float64List _csrK(Int32List start, List<double> Function(_NodeDef) s) {
    final out = Float64List(start.last);
    for (var i = 0; i < _defs.length; i++) {
      out.setAll(start[i], s(_defs[i]));
    }
    return out;
  }

  /// Re-validate everything the bounds-check-free hot path relies on.
  void _checkInvariants() {
    final n = _n;
    void csr(Int32List start, Int32List idx, Float64List k) {
      if (start.length != n + 1 || start[0] != 0) {
        throw StateError('Corrupt edge offsets');
      }
      for (var i = 0; i < n; i++) {
        if (start[i + 1] < start[i]) throw StateError('Corrupt edge offsets');
      }
      if (start[n] != idx.length || idx.length != k.length) {
        throw StateError('Corrupt edge lists');
      }
      for (final j in idx) {
        if (j < 0 || j >= n) throw StateError('Edge to missing node $j');
      }
    }

    csr(_vpStart, _vpIdx, _vpK);
    csr(_vcStart, _vcIdx, _vcK);
    csr(_volpStart, _volpIdx, _volpK);
    csr(_volcStart, _volcIdx, _volcK);
    for (final f in NodeField.values) {
      if (_column(f).length != n) throw StateError('Bad column ${f.name}');
    }
    for (final a in [_kind, _couplingFn, _isLeaf]) {
      if (a.length != n) throw StateError('Bad node table');
    }
    if (_inputDim.length != _inputs.length ||
        _inputCol.length != _inputs.length) {
      throw StateError('Bad input table');
    }
    for (var k = 0, col = 0; k < _inputs.length; k++) {
      if (_inputCol[k] != col || _inputDim[k] < 1) {
        throw StateError('Bad input columns');
      }
      final i = _inputs[k];
      final vector = _kind[i] == _kindEf || _kind[i] == _kindCategorical;
      if (vector && _vMeanStart[i + 1] - _vMeanStart[i] != _inputDim[k]) {
        throw StateError('Bad vector input dimension');
      }
      if (!vector && _inputDim[k] != 1) {
        throw StateError('Bad input dimension');
      }
      col += _inputDim[k];
      if (col > _nInputColumns) throw StateError('Bad input columns');
    }
    for (final (start, len) in [
      (_vMeanStart, _vMean.length),
      (_ssStart, _ss.length),
      (_alphaStart, _alpha.length),
    ]) {
      if (start.length != n + 1 || start[0] != 0 || start[n] != len) {
        throw StateError('Corrupt vector offsets');
      }
      for (var i = 0; i < n; i++) {
        if (start[i + 1] < start[i]) throw StateError('Corrupt vector offsets');
      }
    }
    if (_xis.length != _ss.length ||
        _xisInitial.length != _ss.length ||
        _ssInitial.length != _ss.length ||
        _vMeanInitial.length != _vMean.length ||
        _alphaInitial.length != _alpha.length) {
      throw StateError('Bad vector store');
    }
    for (var i = 0; i < n; i++) {
      final len = _alphaStart[i + 1] - _alphaStart[i];
      if (_kind[i] != _kindCategorical) {
        if (len != 0) throw StateError('Bad categorical vector lengths');
        continue;
      }
      if (len != _vMeanStart[i + 1] - _vMeanStart[i] ||
          len != _vpStart[i + 1] - _vpStart[i] ||
          len > _alphaScratch.length) {
        throw StateError('Bad categorical vector lengths');
      }
      for (var e = _vpStart[i]; e < _vpStart[i + 1]; e++) {
        final b = _vpIdx[e];
        if (_kind[b] != _kindBinary || _vpStart[b + 1] == _vpStart[b]) {
          throw StateError('Bad categorical parents');
        }
      }
    }
    for (var i = 0; i < n; i++) {
      if (_kind[i] != _kindEf) continue;
      final d = _vMeanStart[i + 1] - _vMeanStart[i];
      if (_ssStart[i + 1] - _ssStart[i] != _efSufficientStatisticsCount(d)) {
        throw StateError('Bad ef-state vector lengths');
      }
      final nvp = _vpStart[i + 1] - _vpStart[i];
      if (nvp != 0 && nvp != _ssStart[i + 1] - _ssStart[i]) {
        throw StateError('Bad ef-state parents');
      }
      for (var e = _vpStart[i]; e < _vpStart[i + 1]; e++) {
        final p = _vpIdx[e];
        if (_vpStart[p + 1] == _vpStart[p]) {
          throw StateError('Bad ef-state parents');
        }
      }
    }
    for (final i in [..._predNode, ..._updNode, ..._inputs]) {
      if (i < 0 || i >= n) throw StateError('Bad node index $i in schedule');
    }
    if (_couplings.length != n) throw StateError('Bad coupling table');
    for (final c in _couplingFn) {
      if (c >= CouplingFn.values.length && c != _kCustomCoupling) {
        throw StateError('Bad coupling fn');
      }
    }
    var maxVol = 0;
    for (var i = 0; i < n; i++) {
      final w = _volcStart[i + 1] - _volcStart[i];
      if (w > maxVol) maxVol = w;
    }
    if (_u.capacity < maxVol) {
      throw StateError('Scratch buffers too small');
    }
  }

  static Map<NodeField, double> _defaults(NodeKind kind) => switch (kind) {
    NodeKind.continuousState => const {
      NodeField.mean: 0.0,
      NodeField.expectedMean: 0.0,
      NodeField.precision: 1.0,
      NodeField.expectedPrecision: 1.0,
      NodeField.conditionalExpectedPrecision: 1.0,
      NodeField.observed: 1.0,
      NodeField.tonicVolatility: -4.0,
      NodeField.tonicDrift: 0.0,
      NodeField.autoconnectionStrength: 1.0,
      NodeField.currentVariance: 1.0,
    },
    NodeKind.constantState => const {
      NodeField.mean: 1.0,
      NodeField.expectedMean: 1.0,
      NodeField.precision: 1.0,
      NodeField.expectedPrecision: double.infinity,
    },
    NodeKind.binaryState => const {
      NodeField.mean: 0.0,
      NodeField.expectedMean: 0.5,
      NodeField.precision: 1.0,
      NodeField.expectedPrecision: 1.0,
      NodeField.conditionalExpectedPrecision: 1.0,
      NodeField.observed: 1.0,
      NodeField.currentVariance: 1.0,
    },
    NodeKind.efState => const {
      NodeField.mean: 0.0,
      NodeField.observed: 1.0,
      NodeField.nus: 3.0,
    },
    NodeKind.categoricalState => const {NodeField.observed: 1.0},
  };

  UpdateSequence? _updateSequence;

  /// The belief-propagation schedule.
  UpdateSequence get updateSequence {
    build();
    return _updateSequence!;
  }

  /// Indexes of the input nodes (nodes without children), which receive the
  /// observation columns in order: one column each, except an ef-state
  /// input, which takes `dimension` consecutive columns, and a categorical
  /// one, which takes one per category.
  List<int> get inputIdxs {
    build();
    return _inputIdxsView;
  }

  List<int> _inputIdxsView = const [];

  /// Width of one row of input data: the number of [inputIdxs], plus
  /// `dimension - 1` per multivariate ef-state input and `n - 1` per
  /// categorical input with `n` categories.
  int get nInputColumns {
    build();
    return _nInputColumns;
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
  Int32List _vectorStart(VectorField f) => switch (f) {
    VectorField.mean => _vMeanStart,
    VectorField.xis || VectorField.observationSs => _ssStart,
    VectorField.alpha => _alphaStart,
  };

  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  Float64List _vector(VectorField f) => switch (f) {
    VectorField.mean => _vMean,
    VectorField.xis => _xis,
    VectorField.observationSs => _ss,
    VectorField.alpha => _alpha,
  };

  Float64List _vectorInitial(VectorField f) => switch (f) {
    VectorField.mean => _vMeanInitial,
    VectorField.xis => _xisInitial,
    VectorField.observationSs => _ssInitial,
    VectorField.alpha => _alphaInitial,
  };

  /// Length of [node]'s [field] vector (0 if the node has none).
  int vectorLength(int node, VectorField field) {
    build();
    RangeError.checkValidIndex(node, null, 'node', _n);
    final start = _vectorStart(field);
    return start[node + 1] - start[node];
  }

  /// A copy of [node]'s current [field] vector (empty if it has none).
  Float64List getVector(int node, VectorField field) {
    final len = vectorLength(node, field);
    final s = _vectorStart(field)[node];
    return Float64List.fromList(_vector(field).sublist(s, s + len));
  }

  /// Set [node]'s current [field] vector (the initial value is unchanged;
  /// see [setInitialVector]).
  void setVector(int node, VectorField field, List<double> values) {
    build();
    _writeVector(_vector(field), node, field, values);
  }

  /// A copy of [node]'s initial [field] vector, restored by [reset].
  Float64List getInitialVector(int node, VectorField field) {
    final len = vectorLength(node, field);
    final s = _vectorStart(field)[node];
    return Float64List.fromList(_vectorInitial(field).sublist(s, s + len));
  }

  /// Change [node]'s initial [field] vector (e.g. the prior `xis` of an
  /// ef-state node). Takes effect on the next [reset] / [inputData].
  void setInitialVector(int node, VectorField field, List<double> values) {
    build();
    _writeVector(_vectorInitial(field), node, field, values);
  }

  void _writeVector(
    Float64List dst,
    int node,
    VectorField field,
    List<double> values,
  ) {
    final len = vectorLength(node, field);
    if (values.length != len) {
      throw ArgumentError.value(
        values,
        'values',
        'Node $node has $len ${field.pyhgfName} value(s)',
      );
    }
    dst.setAll(_vectorStart(field)[node], values);
  }

  /// Current value of [field] on [node].
  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  @pragma('dart2js:prefer-inline')
  double get(int node, NodeField field) {
    build();
    return _column(field)[node];
  }

  /// Set the current value of [field] on [node] (the initial value is
  /// unchanged; see [setInitial]).
  void set(int node, NodeField field, double value) {
    build();
    _column(field)[node] = value;
  }

  /// Initial value of [field] on [node], restored by [reset].
  double getInitial(int node, NodeField field) {
    build();
    return _initial[field.index * _n + node];
  }

  /// Change the initial value of [field] on [node] (a parameter, e.g. the
  /// tonic volatility, or a prior mean/precision). Takes effect on the next
  /// [reset] / [inputData].
  void setInitial(int node, NodeField field, double value) {
    build();
    _initial[field.index * _n + node] = value;
  }

  /// Set the value coupling strength from [parent] to [child] (both sides
  /// of the edge, as [addNodes] does). Like the other parameters, it keeps
  /// its value across [reset]s.
  void setValueCoupling(int parent, int child, double strength) =>
      _setCoupling(parent, child, strength, volatility: false);

  /// Set the volatility coupling strength κ from [parent] to [child] (both
  /// sides of the edge, as [addNodes] does).
  void setVolatilityCoupling(int parent, int child, double strength) =>
      _setCoupling(parent, child, strength, volatility: true);

  void _setCoupling(
    int parent,
    int child,
    double strength, {
    required bool volatility,
  }) {
    if (parent < 0 || parent >= nNodes) {
      throw RangeError.range(parent, 0, nNodes - 1, 'parent');
    }
    if (child < 0 || child >= nNodes) {
      throw RangeError.range(child, 0, nNodes - 1, 'child');
    }
    final p = _defs[parent], c = _defs[child];
    final children = volatility ? p.volatilityChildren : p.valueChildren;
    final childS = volatility
        ? p.volatilityChildStrengths
        : p.valueChildStrengths;
    final parents = volatility ? c.volatilityParents : c.valueParents;
    final parentS = volatility
        ? c.volatilityParentStrengths
        : c.valueParentStrengths;
    if (!children.contains(child)) {
      throw ArgumentError(
        'Node $parent is not a ${volatility ? 'volatility' : 'value'} parent '
        'of node $child',
      );
    }
    for (var k = 0; k < children.length; k++) {
      if (children[k] == child) childS[k] = strength;
    }
    for (var k = 0; k < parents.length; k++) {
      if (parents[k] == parent) parentS[k] = strength;
    }
    if (!_compiled) return;
    final (cStart, cIdx, cK) = volatility
        ? (_volcStart, _volcIdx, _volcK)
        : (_vcStart, _vcIdx, _vcK);
    final (pStart, pIdx, pK) = volatility
        ? (_volpStart, _volpIdx, _volpK)
        : (_vpStart, _vpIdx, _vpK);
    for (var e = cStart[parent]; e < cStart[parent + 1]; e++) {
      if (cIdx[e] == child) cK[e] = strength;
    }
    for (var e = pStart[child]; e < pStart[child + 1]; e++) {
      if (pIdx[e] == parent) pK[e] = strength;
    }
  }

  /// Restore every node to its initial state.
  void reset() {
    build();
    _pendingPrediction = null;
    final n = _n;
    for (final f in NodeField.values) {
      final col = _column(f);
      final base = f.index * n;
      for (var i = 0; i < n; i++) {
        col[i] = _initial[base + i];
      }
    }
    for (final f in VectorField.values) {
      _vector(f).setAll(0, _vectorInitial(f));
    }
  }

  // ---------------------------------------------------------------------------
  // Filtering
  // ---------------------------------------------------------------------------

  /// Filter a whole series of observations, starting from the initial state
  /// (or from the current one with [append]).
  ///
  /// [data] is row-major `nTime x nInputColumns`: one column per input
  /// node, in [inputIdxs] order, except that an ef-state input takes
  /// `dimension` consecutive columns. [observed] is row-major
  /// `nTime x nInputs`, one flag per input node (`0.0` marks a missing
  /// observation; the same shape as [data] unless there are multivariate
  /// inputs). [timeSteps] has length `nTime` (default all `1.0`).
  ///
  /// Returns the recorded [fields] per node and time step, or `null` when
  /// [record] is false (the cheapest mode, e.g. for fitting; read the final
  /// state with [get]). With a [response] model, the recorded trajectories
  /// also carry its per-trial surprise ([Trajectories.responseSurprise]).
  ///
  /// To re-run often on same-length data (a playground re-filtering on
  /// every slider move), pass the previous result as [into]: its columns are
  /// overwritten and it is returned, instead of allocating new ones. Fresh
  /// columns cost about as much as the recording itself on first touch.
  /// [into] must come from this network's shape (same `nTime`, node count
  /// and kinds); its own fields are recorded, so [fields] must be left at
  /// its default. With a [response], [into] must carry
  /// [Trajectories.responseSurprise]; without one, that column is NaN-filled.
  ///
  /// ```dart
  /// Trajectories? traj;
  /// void rerun() => traj = hgf.inputData(u, into: traj);
  /// ```
  Trajectories? inputData(
    Float64List data, {
    Float64List? observed,
    Float64List? timeSteps,
    bool append = false,
    bool record = true,
    List<NodeField> fields = defaultRecordedFields,
    ResponseFunction? response,
    Trajectories? into,
  }) {
    final nTime = _prepareRun(data, observed, timeSteps, append, response);
    final Trajectories? traj;
    if (!record) {
      traj = null;
    } else if (into != null) {
      _checkReusable(into, nTime, fields, response);
      traj = into;
    } else {
      traj = Trajectories(
        nTime,
        _n,
        fields,
        kinds: [for (final d in _defs) d.kind],
        responseSurprise: response == null ? null : Float64List(nTime),
        vectorStarts: _recordedVectorStarts(),
      );
    }
    _run(
      data,
      observed,
      timeSteps,
      nTime,
      traj,
      response,
      traj?.responseSurprise,
    );
    return traj;
  }

  void _checkReusable(
    Trajectories into,
    int nTime,
    List<NodeField> fields,
    ResponseFunction? response,
  ) {
    if (!identical(fields, defaultRecordedFields)) {
      throw ArgumentError('Pass either fields or into, not both');
    }
    final kinds = into.kinds;
    var sameKinds = kinds != null && kinds.length == _n;
    for (var i = 0; sameKinds && i < _n; i++) {
      sameKinds = kinds![i] == _defs[i].kind;
    }
    final starts = _recordedVectorStarts();
    var sameVectors =
        into.vectorFields.length == starts.length &&
        starts.keys.every(into.vectorFields.contains);
    if (sameVectors) {
      starts.forEach((f, start) {
        for (var i = 0; i <= _n; i++) {
          if (into.vectorStart(f)[i] != start[i]) sameVectors = false;
        }
      });
    }
    if (into.nTime != nTime ||
        into.nNodes != _n ||
        !sameKinds ||
        !sameVectors) {
      throw ArgumentError.value(
        into,
        'into',
        'Recorded for a different shape (nTime ${into.nTime}, '
            '${into.nNodes} nodes); this run has nTime $nTime, $_n nodes',
      );
    }
    final rs = into.responseSurprise;
    if (response != null && rs == null) {
      throw ArgumentError.value(
        into,
        'into',
        'Has no responseSurprise column for the response model',
      );
    }
    if (response == null) rs?.fillRange(0, nTime, double.nan);
  }

  /// Filter [data] (as in [inputData], without recording) and return the
  /// total surprise of [response] over the trials: the negative
  /// log-likelihood to minimise when fitting. Pass [perTrial] (length
  /// `nTime`) to also get the surprise of each trial.
  ///
  /// ```dart
  /// final nll = hgf.surprise(u, const FirstLevelBinarySurprise());
  /// ```
  double surprise(
    Float64List data,
    ResponseFunction response, {
    Float64List? observed,
    Float64List? timeSteps,
    bool append = false,
    Float64List? perTrial,
  }) {
    final nTime = _prepareRun(data, observed, timeSteps, append, response);
    if (perTrial != null && perTrial.length != nTime) {
      throw ArgumentError('perTrial must have one entry per time step');
    }
    return _run(data, observed, timeSteps, nTime, null, response, perTrial);
  }

  /// The vector fields any node has, with their offsets (recorded by every
  /// run that records).
  Map<VectorField, Int32List> _recordedVectorStarts() => {
    for (final f in VectorField.values)
      if (_vectorStart(f)[_n] > 0) f: _vectorStart(f),
  };

  /// Validate a run's buffers, reset unless [append], and return `nTime`.
  int _prepareRun(
    Float64List data,
    Float64List? observed,
    Float64List? timeSteps,
    bool append,
    ResponseFunction? response,
  ) {
    build();
    final nCols = _nInputColumns;
    if (data.length % nCols != 0) {
      throw ArgumentError(
        'data length ${data.length} is not a multiple of the $nCols input '
        'column(s)',
      );
    }
    final nTime = data.length ~/ nCols;
    if (observed != null && observed.length != nTime * _inputs.length) {
      throw ArgumentError(
        'observed must have one flag per time step and input node '
        '($nTime x ${_inputs.length})',
      );
    }
    if (timeSteps != null && timeSteps.length != nTime) {
      throw ArgumentError('timeSteps must have one entry per time step');
    }
    if (!append) reset();
    if (_pendingPrediction != null) {
      throw StateError('predict() is pending; call step() or reset() first');
    }
    response?.prepare(this, nTime);
    return nTime;
  }

  /// The filtering loop shared by [inputData] and [surprise]. Returns the
  /// summed surprise of [response] (0 without one).
  double _run(
    Float64List data,
    Float64List? observed,
    Float64List? timeSteps,
    int nTime,
    Trajectories? traj,
    ResponseFunction? response,
    Float64List? perTrial,
  ) {
    final nIn = _inputs.length, nCols = _nInputColumns;
    final rec = traj == null ? null : _Recorder(this, traj);
    final recordVectors = rec != null && rec._hasVectors;
    var total = 0.0;
    for (var t = 0; t < nTime; t++) {
      _propagate(
        data,
        observed,
        t * nCols,
        t * nIn,
        timeSteps == null ? 1.0 : timeSteps[t],
      );
      if (recordVectors) rec.addVectors(t);
      rec?.add();
      if (response != null) {
        final s = response.surprise(this, t);
        total += s;
        if (perTrial != null) perTrial[t] = s;
      }
    }
    rec?.finish();
    return total;
  }

  /// Run [nTime] trials in generative mode (pyhgf's `sample`): at each
  /// trial, after the prediction sweep, every input node observes a draw from
  /// its own predictive distribution, and beliefs update as usual. A binary
  /// input draws `1` with probability `expectedMean`; a continuous input
  /// draws from `N(expectedMean, 1 / expectedPrecision)`.
  ///
  /// Starts from the initial state, or from the current one with [append]
  /// (pyhgf samples from the state after the last `input_data`). Pass a
  /// seeded [random] for reproducible draws. The draws are written to
  /// [observations] (row-major `nTime x nInputs`) if given, and are also the
  /// input nodes' recorded `mean`.
  ///
  /// For several independent samples from the same state:
  ///
  /// ```dart
  /// hgf.inputData(data, record: false);   // condition on the data so far
  /// final state = hgf.saveState();
  /// final rng = math.Random(42);
  /// final futures = [
  ///   for (var k = 0; k < 100; k++)
  ///     (hgf..restoreState(state)).sample(20, random: rng, append: true)!,
  /// ];
  /// ```
  Trajectories? sample(
    int nTime, {
    math.Random? random,
    Float64List? timeSteps,
    bool append = false,
    bool record = true,
    List<NodeField> fields = defaultRecordedFields,
    Float64List? observations,
  }) {
    build();
    if (nTime < 0) throw ArgumentError.value(nTime, 'nTime');
    if (_hasVectorInputs) {
      throw UnsupportedError(
        'sample() does not support ef-state or categorical inputs yet',
      );
    }
    final nIn = _inputs.length;
    if (timeSteps != null && timeSteps.length != nTime) {
      throw ArgumentError('timeSteps must have one entry per time step');
    }
    if (observations != null && observations.length != nTime * nIn) {
      throw ArgumentError('observations must have nTime x nInputs entries');
    }
    if (!append) reset();
    if (_pendingPrediction != null) {
      throw StateError('predict() is pending; call step() or reset() first');
    }
    final rng = random ?? math.Random();
    final traj = record
        ? Trajectories(
            nTime,
            _n,
            fields,
            kinds: [for (final d in _defs) d.kind],
          )
        : null;
    final rec = traj == null ? null : _Recorder(this, traj);
    final draws = Float64List(nIn);
    final inputs = _inputs;
    for (var t = 0; t < nTime; t++) {
      final dt = timeSteps == null ? 1.0 : timeSteps[t];
      _predictSweep(dt);
      for (var k = 0; k < nIn; k++) {
        final idx = inputs[k];
        final mu = _expectedMean[idx];
        final x = _kind[idx] == _kindBinary
            ? (rng.nextDouble() < mu ? 1.0 : 0.0)
            : mu + _standardNormal(rng) / math.sqrt(_expectedPrecision[idx]);
        draws[k] = x;
        if (observations != null) observations[t * nIn + k] = x;
      }
      _propagate(draws, null, 0, 0, dt, predicted: true);
      rec?.add();
    }
    rec?.finish();
    return traj;
  }

  /// A standard normal draw (Marsaglia's polar method).
  static double _standardNormal(math.Random rng) {
    double u, v, s;
    do {
      u = 2.0 * rng.nextDouble() - 1.0;
      v = 2.0 * rng.nextDouble() - 1.0;
      s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    return u * math.sqrt(-2.0 * math.log(s) / s);
  }

  /// A copy of every node's current state, to [restoreState] later (e.g. to
  /// draw several [sample]s from the same beliefs, or to branch a real-time
  /// session). The initial state and the parameters set with [setInitial]
  /// are not part of it.
  NetworkState saveState() {
    build();
    if (_pendingPrediction != null) {
      throw StateError('predict() is pending; call step() or reset() first');
    }
    final n = _n;
    final data = Float64List(_stateLength);
    for (final f in NodeField.values) {
      data.setRange(f.index * n, (f.index + 1) * n, _column(f));
    }
    var k = NodeField.values.length * n;
    for (final f in VectorField.values) {
      final v = _vector(f);
      data.setAll(k, v);
      k += v.length;
    }
    return NetworkState._(this, data);
  }

  /// Restore a state taken with [saveState] on this network.
  void restoreState(NetworkState state) {
    build();
    if (!identical(state._network, this) ||
        state._data.length != _stateLength) {
      throw ArgumentError('The state was saved from another network');
    }
    _pendingPrediction = null;
    final n = _n;
    for (final f in NodeField.values) {
      _column(f).setRange(0, n, state._data, f.index * n);
    }
    var k = NodeField.values.length * n;
    for (final f in VectorField.values) {
      final v = _vector(f);
      v.setRange(0, v.length, state._data, k);
      k += v.length;
    }
  }

  int get _stateLength {
    var len = NodeField.values.length * _n;
    for (final f in VectorField.values) {
      len += _vector(f).length;
    }
    return len;
  }

  /// Convenience for a single input node: filter [series].
  Trajectories inputSeries(
    List<double> series, {
    List<double>? observed,
    List<double>? timeSteps,
  }) => inputData(
    series is Float64List ? series : Float64List.fromList(series),
    observed: observed == null ? null : Float64List.fromList(observed),
    timeSteps: timeSteps == null ? null : Float64List.fromList(timeSteps),
  )!;

  /// One belief-propagation step from the current state (real-time use):
  /// [observations] has [nInputColumns] values (one per input node, or
  /// `dimension` for an ef-state input), [observed] one flag per input node.
  ///
  /// The step starts with the prediction sweep, unless [predict] already ran
  /// it for this trial; read the prediction for the upcoming observation
  /// with [predict] first, e.g.:
  ///
  /// ```dart
  /// hgf.predict();
  /// final p = hgf.get(0, NodeField.expectedMean); // P(u = 1) for this trial
  /// hgf.step(Float64List.fromList([u]));          // reuses that prediction
  /// ```
  void step(
    Float64List observations, {
    Float64List? observed,
    double timeStep = 1.0,
  }) {
    build();
    if (observations.length != _nInputColumns) {
      throw ArgumentError('Expected $_nInputColumns observation(s)');
    }
    if (observed != null && observed.length != _inputs.length) {
      throw ArgumentError('Expected ${_inputs.length} observed flag(s)');
    }
    if (_pendingPrediction != null && _pendingPrediction != timeStep) {
      throw StateError(
        'predict() ran with timeStep $_pendingPrediction, step() got $timeStep',
      );
    }
    final predicted = _pendingPrediction != null;
    _pendingPrediction = null;
    _propagate(observations, observed, 0, 0, timeStep, predicted: predicted);
  }

  /// Run only the prediction sweep for the next trial, so that predicted
  /// beliefs (`expectedMean`, `expectedPrecision`, ...) refer to the upcoming
  /// observation. The next [step] reuses it instead of predicting again;
  /// calling [predict] twice without a [step] is a no-op.
  void predict({double timeStep = 1.0}) {
    build();
    if (_pendingPrediction != null) {
      if (_pendingPrediction != timeStep) {
        throw StateError(
          'A prediction for timeStep $_pendingPrediction is pending',
        );
      }
      return;
    }
    _predictSweep(timeStep);
    _pendingPrediction = timeStep;
  }

  /// Time step of a [predict] not yet consumed by [step], if any.
  double? _pendingPrediction;

  void _predictSweep(double timeStep) {
    final predNode = _predNode, predStep = _predStep;
    for (var k = 0; k < predNode.length; k++) {
      _dispatch(predStep[k], predNode[k], timeStep);
    }
  }

  /// One time slice: predictions, observations, then the update sweep.
  ///
  /// The hot path (this and the update steps) runs without typed-data bounds
  /// checks (`vm:unsafe:no-bounds-checks`, 35-45% faster on the VM). That is
  /// sound because every index it uses comes from [build]: node indexes are
  /// validated in [addNodes], every state column has length `n`, the CSR
  /// offsets/indexes are built from those lists (and re-checked by
  /// [_checkInvariants]), and the callers ([inputData], [step]) validate the
  /// observation buffers' lengths.
  @pragma('vm:unsafe:no-bounds-checks')
  void _propagate(
    Float64List data,
    Float64List? observed,
    int dataOffset,
    int observedOffset,
    double timeStep, {
    bool predicted = false,
  }) {
    if (!predicted) {
      final predNode = _predNode, predStep = _predStep;
      for (var k = 0; k < predNode.length; k++) {
        _dispatch(predStep[k], predNode[k], timeStep);
      }
    }
    final inputs = _inputs;
    if (_hasVectorInputs) {
      _observeVectors(data, observed, dataOffset, observedOffset);
    } else {
      // One column per input, so observedOffset == dataOffset.
      for (var k = 0; k < inputs.length; k++) {
        final idx = inputs[k];
        _mean[idx] = data[dataOffset + k];
        _observed[idx] = observed == null ? 1.0 : observed[dataOffset + k];
      }
    }
    final updNode = _updNode, updStep = _updStep;
    for (var k = 0; k < updNode.length; k++) {
      _dispatch(updStep[k], updNode[k], timeStep);
    }
  }

  @pragma('vm:prefer-inline')
  void _dispatch(int step, int node, double t) {
    switch (step) {
      case _kPredictionContinuous:
        _predictionContinuous(node, t);
      case _kPredictionContinuousMeanField:
        _predictionContinuousMeanField(node, t);
      case _kPredictionBinary:
        _predictionBinary(node);
      case _kPosteriorContinuous:
        _posteriorContinuous(node, meanField: false);
      case _kPosteriorContinuousMeanField:
        _posteriorContinuous(node, meanField: true);
      case _kPosteriorContinuousEhgf:
        _posteriorContinuousEhgf(node, t, meanField: false);
      case _kPosteriorContinuousEhgfMeanField:
        _posteriorContinuousEhgf(node, t, meanField: true);
      case _kPosteriorContinuousUnbounded:
        _posteriorContinuousUnbounded(node, t);
      case _kPredictionErrorContinuous:
        _predictionErrorContinuous(node);
      case _kPredictionErrorBinary:
        _predictionErrorBinary(node);
      case _kPredictionErrorEfFixed:
        _predictionErrorEfFixed(node);
      case _kPredictionErrorEfDynamic:
        _predictionErrorEfDynamic(node);
      case _kPosteriorEfDynamic:
        _posteriorEfDynamic(node);
      case _kPredictionErrorCategorical:
        _predictionErrorCategorical(node);
      case _kPosteriorCategorical:
        _posteriorCategorical(node);
    }
  }

  /// The observation step when some inputs are ef-state or categorical
  /// nodes: those take their columns into their [VectorField.mean] (and the
  /// first into the scalar mean).
  @pragma('vm:unsafe:no-bounds-checks')
  void _observeVectors(
    Float64List data,
    Float64List? observed,
    int dataOffset,
    int observedOffset,
  ) {
    final inputs = _inputs, inputCol = _inputCol, inputDim = _inputDim;
    for (var k = 0; k < inputs.length; k++) {
      final idx = inputs[k];
      final c = dataOffset + inputCol[k];
      _mean[idx] = data[c];
      final kind = _kind[idx];
      if (kind == _kindEf || kind == _kindCategorical) {
        final s = _vMeanStart[idx];
        _vMean.setRange(s, s + inputDim[k], data, c);
      }
      _observed[idx] = observed == null ? 1.0 : observed[observedOffset + k];
    }
  }
}

extension on Network {
  /// The non-linear coupling function of [node] with code [cf] (`cf != 0`):
  /// a static enum lookup for the named ones, the node's own object for a
  /// custom one.
  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  Coupling _couplingOf(int cf, int node) =>
      cf == _kCustomCoupling ? _couplings[node] : CouplingFn.values[cf];
}

/// A snapshot of a [Network]'s state, from [Network.saveState].
final class NetworkState {
  NetworkState._(this._network, this._data);

  final Network _network;
  final Float64List _data;
}

/// `max(a, b)` that propagates a NaN in [a] (like `jnp.maximum`).
@pragma('vm:prefer-inline')
@pragma('wasm:prefer-inline')
double _maxNan(double a, double b) => (a.isNaN || a > b) ? a : b;

/// `min(a, b)` that propagates a NaN in [a] (like `jnp.minimum`).
@pragma('vm:prefer-inline')
@pragma('wasm:prefer-inline')
double _minNan(double a, double b) => (a.isNaN || a < b) ? a : b;

/// Copies a network's state into [Trajectories], one trial per [add].
///
/// Trials are staged time-major ([_block] at a time, one contiguous row of
/// `nodes × fields` values each, so the stage stays in L1) and then written
/// to each node-major (field, node) series a block at a time. Writing the
/// node-major layout trial by trial costs a strided store per value.
final class _Recorder {
  _Recorder(Network net, Trajectories traj)
    : _n = net._n,
      _nTime = traj.nTime,
      _src = [for (final f in traj.fields) net._column(f)],
      _dst = [for (final f in traj.fields) traj.raw(f)],
      _stage = Float64List(_block * net._n * traj.fields.length),
      _vSrc = [for (final f in traj.vectorFields) net._vector(f)],
      _vDst = [for (final f in traj.vectorFields) traj.rawVector(f)],
      _vStart = [for (final f in traj.vectorFields) traj.vectorStart(f)],
      _hasVectors = traj.vectorFields.isNotEmpty;

  // Vector fields (few values, written directly): node i's [field] at trial
  // t goes to `start[i] * nTime + t * len + j`.
  final List<Float64List> _vSrc;
  final List<Float64List> _vDst;
  final List<Int32List> _vStart;
  final bool _hasVectors;

  static const int _block = 64;

  final int _n;
  final int _nTime;
  final List<Float64List> _src;
  final List<Float64List> _dst;
  final Float64List _stage;
  var _staged = 0;
  var _t = 0;

  /// Record the current state as the next trial.
  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  @pragma('vm:unsafe:no-bounds-checks')
  void add() {
    final src = _src, stage = _stage, n = _n;
    var k = _staged * n * src.length;
    for (var c = 0; c < src.length; c++) {
      final col = src[c];
      if (isWasm) {
        // A bulk copy: dart2wasm's per-element reads of a column it can't
        // type exactly are dispatched calls. On AOT the plain loop is faster
        // for these short rows.
        stage.setRange(k, k + n, col);
        k += n;
      } else {
        for (var i = 0; i < n; i++) {
          stage[k++] = col[i];
        }
      }
    }
    if (++_staged == _block) _flush();
  }

  /// Record the vector fields of trial [t] (call before that trial's [add]).
  /// Not part of [add]: even untaken, it slowed recording on dart2wasm by
  /// 3-6%.
  @pragma('vm:never-inline')
  @pragma('wasm:never-inline')
  void addVectors(int t) {
    for (var c = 0; c < _vSrc.length; c++) {
      final src = _vSrc[c], dst = _vDst[c], start = _vStart[c];
      for (var i = 0; i < _n; i++) {
        final s = start[i], len = start[i + 1] - s;
        if (len == 0) continue;
        dst.setRange(s * _nTime + t * len, s * _nTime + (t + 1) * len, src, s);
      }
    }
  }

  /// Write any staged trials out.
  void finish() {
    if (_staged != 0) _flush();
  }

  @pragma('vm:unsafe:no-bounds-checks')
  void _flush() {
    final dst = _dst, stage = _stage, n = _n, count = _staged;
    final row = n * dst.length;
    for (var c = 0; c < dst.length; c++) {
      final out = dst[c];
      for (var i = 0; i < n; i++) {
        final base = i * _nTime + _t;
        var k = c * n + i;
        for (var j = 0; j < count; j++, k += row) {
          out[base + j] = stage[k];
        }
      }
    }
    _t += count;
    _staged = 0;
  }
}
