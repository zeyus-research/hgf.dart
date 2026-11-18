import 'dart:math' as math;

import 'package:hgf/hgf.dart' hide digamma, lambertW0, logAddExp;

import 'expr.dart';
import 'ir.dart';

/// The fields every node has a slot for, in slot order.
const List<NodeField> gpuNodeFields = [
  NodeField.mean,
  NodeField.expectedMean,
  NodeField.precision,
  NodeField.expectedPrecision,
  NodeField.conditionalExpectedPrecision,
  NodeField.currentVariance,
  NodeField.effectivePrecision,
  NodeField.valuePredictionError,
  NodeField.volatilityPredictionError,
  NodeField.observed,
  NodeField.tonicVolatility,
  NodeField.tonicDrift,
  NodeField.autoconnectionStrength,
];

const List<String> _fieldVar = [
  'm', 'em', 'p', 'ep', 'cep', 'cv', 'eff', 'vape', 'vope', 'ob', //
  'tv', 'td', 'ac',
];

/// What a kernel adds to each element's total after every trial.
///
/// The observation surprises read the network alone; the readouts score a
/// response per trial and have parameters, which live in slots (so they can
/// differ per element).
sealed class GpuResponse {
  const GpuResponse();

  /// `TotalGaussianSurprise`: the Gaussian surprise of every continuous
  /// node.
  static const totalGaussianSurprise = _ObservationResponse('total');

  /// `FirstLevelGaussianSurprise`: node 0's observation under node 1's
  /// prediction.
  static const firstLevelGaussianSurprise = _ObservationResponse('first');

  /// `FirstLevelBinarySurprise`: the binary surprise of the input nodes.
  static const firstLevelBinarySurprise = _ObservationResponse('binary');

  /// `CategoricalSurprise`: each categorical input's indicators under its
  /// binary nodes' predictions (0 where it is unobserved, with
  /// `NetworkFixes.categoricalMissing`).
  static const categoricalSurprise = _ObservationResponse('categorical');

  /// Parameter defaults, in slot order.
  List<double> get defaults => const [];

  /// Whether it scores a response per trial.
  bool get usesResponses => false;
}

final class _ObservationResponse extends GpuResponse {
  const _ObservationResponse(this.which);

  final String which;

  @override
  String toString() => 'GpuResponse.$which';
}

/// `GaussianReadout`: `response ~ N(belief, 1 / precision)`, the belief
/// being [field] of [node]. Parameter: `precision`.
final class GpuGaussianReadout extends GpuResponse {
  const GpuGaussianReadout({
    this.node = 0,
    this.field = NodeField.expectedMean,
    this.precision = 1.0,
  });

  final int node;
  final NodeField field;
  final double precision;

  @override
  List<double> get defaults => [precision];

  @override
  bool get usesResponses => true;
}

/// `BinarySoftmaxReadout`: binary responses with probability
/// `σ(β logit μ̂_node)`. Parameter: `inverseTemperature`.
final class GpuBinarySoftmax extends GpuResponse {
  const GpuBinarySoftmax({this.node = 0, this.inverseTemperature = 1.0});

  final int node;
  final double inverseTemperature;

  @override
  List<double> get defaults => [inverseTemperature];

  @override
  bool get usesResponses => true;
}

/// `CategoricalSoftmaxReadout`: category `k` of categorical [node] chosen
/// with probability `μ̂ₖ^β / Σⱼ μ̂ⱼ^β`, `μ̂ₖ` the prediction of its `k`-th
/// binary node. Parameter: `inverseTemperature`.
final class GpuCategoricalSoftmax extends GpuResponse {
  const GpuCategoricalSoftmax({this.node = 0, this.inverseTemperature = 1.0});

  final int node;
  final double inverseTemperature;

  @override
  List<double> get defaults => [inverseTemperature];

  @override
  bool get usesResponses => true;
}

/// `SoftmaxReadout`: a choice among options worth `vₖ = Σ w · belief`,
/// option `k` with utility `uₖ = β vₖ + bₖ + κ [k = previous]` and
/// probability `(1 − ε) softmax(u)ₖ + ε / K`. Parameters:
/// `inverseTemperature`, then `stickiness` (κ), `lapse` (ε) and `bias_k`
/// where they are set (null: left out).
final class GpuSoftmax extends GpuResponse {
  const GpuSoftmax({
    required this.options,
    this.field = NodeField.expectedMean,
    this.inverseTemperature = 1.0,
    this.stickiness,
    this.lapse,
    this.bias,
  });

  final List<Map<int, double>> options;
  final NodeField field;
  final double inverseTemperature;
  final double? stickiness;
  final double? lapse;
  final List<double>? bias;

  @override
  List<double> get defaults => [
    inverseTemperature,
    ?stickiness,
    ?lapse,
    ...?bias,
  ];

  @override
  bool get usesResponses => true;
}

/// `ExprReadout`: expressions of the beliefs and parameters; [kind] is
/// `choice` (softmax over the option values), `gaussian` (mean, precision)
/// or `surprise` (which also reads `response`). Parameters: [parameters],
/// in order.
final class GpuExprReadout extends GpuResponse {
  GpuExprReadout({
    required this.kind,
    required List<Expr> expressions,
    required Map<String, ({int node, NodeField field})> beliefs,
    required Map<String, double> parameters,
  }) : expressions = List.unmodifiable(expressions),
       beliefs = Map.unmodifiable(beliefs),
       parameters = Map.unmodifiable(parameters);

  final String kind;
  final List<Expr> expressions;
  final Map<String, ({int node, NodeField field})> beliefs;
  final Map<String, double> parameters;

  @override
  List<double> get defaults => parameters.values.toList();

  @override
  bool get usesResponses => true;
}

/// `f32` versions of the engine's `f64` constants (each [Lit.wide] is the
/// `f64` one, for the CPU evaluator).
abstract final class F32 {
  /// Smallest normal `f32`; the `f64` engine floors at 1e-128, below
  /// `f32`'s range (JAX in `float32` flushes it to 0).
  static const minNormal = Lit('1.17549435e-38f', 1.17549435e-38, 1e-128);

  /// `ln(f32::MAX)`, for `logMaxFinite`.
  static const logMaxFinite = Lit('88.7228391f', 88.7228391, 709.782712893384);

  /// Largest `logS` for which the unbounded update forms `exp(logS)`
  /// directly (700 in `f64`).
  static const linearLogSMax = Lit('80.0f', 80.0, 700.0);
}

/// Where a trial's observations and response come from (WGSL
/// expressions, valid inside the trial loop where `t` is the trial).
///
/// A trial's row has `Network.nInputColumns` columns: one per scalar input
/// node, `dimension` for an ef-state input. Observed flags have one per
/// input node.
abstract interface class TrialSource {
  /// Observation of input column [c] (`f32`).
  String input(int c);

  /// Whether input node [q] (the `q`-th of `Network.inputIdxs`) is
  /// observed (`f32`, 0 or 1).
  String observed(int q);

  /// Whether this trial has a response (`bool`).
  String responsePresent();

  /// The response (`f32`; only read where [responsePresent]).
  String response();

  /// The previous trial's response (`f32`), or `-1` where there is none (the
  /// first trial, or a missed response): no option's index.
  String previousResponse();
}

/// The result of [TrialCompiler.compile].
final class CompiledTrial {
  CompiledTrial._(
    this.body,
    this.carried,
    this.written,
    this.active,
    this.slotOfVar,
  );

  /// The trial's statements (inside the trial loop).
  final String body;

  /// Slot variables read before they are written in a trial: loaded
  /// before the loop.
  final Set<String> carried;

  /// Slot variables the trial writes.
  final Set<String> written;

  /// Slot variables that carry tangents.
  final Set<String> active;

  /// Every slot variable's slot.
  final Map<String, int> slotOfVar;
}

/// Compiles one trial of a [Network] (predictions, observations, updates,
/// response) into WGSL, with forward-mode tangents along [directions]
/// directions for the slot variables in `seeds` and everything they reach.
///
/// The slot layout: every node field of [gpuNodeFields] (`field * n +
/// node`), the value edges, the volatility edges, each ef-state node's
/// expected sufficient statistics ξ and learning rate ν ([vectorSlot],
/// [fieldSlot]), each categorical node's α, probabilities, KL divergence
/// and surprise, the gains of [gainNodes], the response's parameters
/// ([GpuResponse.defaults]), then the running total, its Kahan
/// compensation, the non-finite flag and the result (`resultSlot`).
///
/// With [vectorState] false, ef-state nodes keep ξ and ν at their initial
/// values and categorical nodes their α, probabilities, KL divergence and
/// surprise: no response reads them, so a likelihood doesn't depend on
/// them (with HGF learning, an ef node's sufficient statistics still reach
/// the parents that track them, and a categorical node's indicators its
/// binary nodes). The fitting kernels leave them out.
///
/// Throws [UnsupportedError] for node kinds and update steps without a GPU
/// kernel (dp-state nodes, custom couplings).
final class TrialCompiler {
  TrialCompiler(
    this.net,
    this.response, {
    this.directions = 0,
    this.vectorState = true,
  }) : n = net.nNodes {
    net.build();
    for (var i = 0; i < n; i++) {
      switch (net.kindOf(i)) {
        case NodeKind.continuousState ||
            NodeKind.binaryState ||
            NodeKind.constantState:
          break;
        case NodeKind.efState:
          // A dp-state cluster has its dp node as value child; dp-state has
          // no kernel (below), so only input ef-state nodes get here.
          _checkEfParents(i);
          efNodes.add(i);
          if (vectorState) {
            for (final p in net.valueParentsOf(i)) {
              _addGain(i, p, 'scales its learning rate');
            }
          }
        case NodeKind.categoricalState:
          _checkCategoricalParents(i);
          categoricalNodes.add(i);
          if (vectorState) {
            for (final b in net.valueParentsOf(i)) {
              _addGain(i, b, 'sets its α');
            }
          }
        default:
          throw UnsupportedError(
            '${net.kindOf(i).pyhgfName} has no GPU kernel yet',
          );
      }
    }
    var col = 0;
    for (final i in net.inputIdxs) {
      switch (net.kindOf(i)) {
        case NodeKind.continuousState || NodeKind.binaryState:
          inputColumns.add((col, 1));
          col++;
        case NodeKind.efState:
          final d = efDimension(i);
          inputColumns.add((col, d));
          col += d;
        case NodeKind.categoricalState:
          final d = categoryCount(i);
          inputColumns.add((col, d));
          col += d;
        default:
          throw UnsupportedError('Input node $i: ${net.kindOf(i).pyhgfName}');
      }
    }
    if (col != net.nInputColumns) {
      throw StateError(
        'Input columns: $col, the network has '
        '${net.nInputColumns}',
      );
    }
    for (var c = 0; c < n; c++) {
      for (final p in net.valueParentsOf(c)) {
        valueEdges.add((p, c));
      }
      for (final p in net.volatilityParentsOf(c)) {
        volatilityEdges.add((p, c));
      }
    }
    switch (response) {
      case GpuGaussianReadout(:final node, :final field):
        _checkNode(node);
        if (!gpuNodeFields.contains(field)) {
          throw UnsupportedError('No GPU slot for $field');
        }
      case GpuBinarySoftmax(:final node):
        _checkNode(node);
      case GpuSoftmax(:final options, :final field, :final bias):
        if (!gpuNodeFields.contains(field)) {
          throw UnsupportedError('No GPU slot for $field');
        }
        if (options.length < 2) {
          throw ArgumentError.value(options, 'options', 'Needs two or more');
        }
        if (bias != null && bias.length != options.length) {
          throw ArgumentError.value(bias, 'bias', 'Needs one per option');
        }
        for (final o in options) {
          o.keys.forEach(_checkNode);
        }
      case GpuExprReadout(:final beliefs):
        for (final b in beliefs.values) {
          _checkNode(b.node);
          if (!gpuNodeFields.contains(b.field)) {
            throw UnsupportedError('No GPU slot for ${b.field}');
          }
        }
      case GpuCategoricalSoftmax(:final node):
        _checkNode(node);
        if (net.kindOf(node) != NodeKind.categoricalState) {
          throw ArgumentError.value(node, 'node', 'Not a categorical node');
        }
      case _ObservationResponse(:final which):
        if (which == 'categorical' && categoricalNodes.isEmpty) {
          throw ArgumentError('CategoricalSurprise needs a categorical input');
        }
        if (which == 'first' &&
            (n < 2 ||
                net.kindOf(0) != NodeKind.continuousState ||
                net.kindOf(1) != NodeKind.continuousState)) {
          throw ArgumentError(
            'FirstLevelGaussianSurprise needs continuous nodes 0 and 1',
          );
        }
        if (which == 'binary' &&
            net.inputIdxs.any((i) => net.kindOf(i) != NodeKind.binaryState)) {
          throw ArgumentError('FirstLevelBinarySurprise needs binary inputs');
        }
    }
    var s = 0;
    for (var f = 0; f < gpuNodeFields.length; f++) {
      for (var i = 0; i < n; i++) {
        slotOfVar['${_fieldVar[f]}$i'] = s++;
      }
    }
    for (final (p, c) in valueEdges) {
      slotOfVar['a${p}_$c'] = s++;
    }
    for (final (p, c) in volatilityEdges) {
      slotOfVar['k${p}_$c'] = s++;
    }
    for (final i in efNodes) {
      for (var k = 0; k < efStatistics(i); k++) {
        slotOfVar['xi${i}_$k'] = s++;
      }
      slotOfVar['nu$i'] = s++;
    }
    for (final i in categoricalNodes) {
      for (var k = 0; k < categoryCount(i); k++) {
        slotOfVar['al${i}_$k'] = s++;
      }
      for (var k = 0; k < categoryCount(i); k++) {
        slotOfVar['cx${i}_$k'] = s++;
      }
      slotOfVar['kl$i'] = s++;
      slotOfVar['su$i'] = s++;
    }
    for (final i in gainNodes) {
      slotOfVar['gain$i'] = s++;
    }
    for (var j = 0; j < response.defaults.length; j++) {
      slotOfVar['rp$j'] = s++;
    }
    totalSlot = s;
  }

  void _checkNode(int node) {
    if (node < 0 || node >= n) {
      throw ArgumentError.value(node, 'node', 'No such node');
    }
  }

  /// Node [p], a value parent of [i], has a continuous value parent whose
  /// posterior also writes its gain ([gainNodes]); it must have [p] as its
  /// only child.
  void _addGain(int i, int p, String what) {
    final pp = net.valueParentsOf(p).first;
    if (net.valueChildrenOf(pp).length != 1 ||
        net.volatilityChildrenOf(pp).isNotEmpty) {
      throw UnsupportedError(
        '${net.kindOf(i).pyhgfName} node $i: node $pp, which $what, has '
        'children besides node $p',
      );
    }
    gainNodes.add(pp);
  }

  /// One binary value parent per category, each with one continuous value
  /// parent (as `addCategoricalState` builds them).
  void _checkCategoricalParents(int i) {
    for (final b in net.valueParentsOf(i)) {
      final parents = net.valueParentsOf(b);
      if (net.kindOf(b) != NodeKind.binaryState ||
          parents.length != 1 ||
          net.kindOf(parents.single) != NodeKind.continuousState) {
        throw UnsupportedError(
          'categorical-state node $i: parent $b must be a binary node with '
          'one continuous value parent',
        );
      }
    }
  }

  /// HGF learning: each sufficient statistic's continuous value parent,
  /// which has a value parent of its own (whose change scales ν).
  void _checkEfParents(int i) {
    final parents = net.valueParentsOf(i);
    if (parents.isEmpty) return;
    if (parents.length != efStatistics(i)) {
      throw StateError('ef-state node $i: one value parent per statistic');
    }
    for (final p in parents) {
      if (net.kindOf(p) != NodeKind.continuousState ||
          net.valueParentsOf(p).isEmpty) {
        throw UnsupportedError(
          'ef-state node $i: parent $p must be a continuous node with a '
          'value parent',
        );
      }
    }
  }

  final Network net;
  final GpuResponse response;

  /// Tangent directions (0: none).
  final int directions;
  final int n;

  /// Whether ef-state nodes update ξ and ν, and categorical nodes their α,
  /// probabilities, KL divergence and surprise (see the class
  /// documentation).
  final bool vectorState;

  final valueEdges = <(int, int)>[];
  final volatilityEdges = <(int, int)>[];
  final slotOfVar = <String, int>{};

  /// The ef-state nodes, in node order.
  final efNodes = <int>[];

  /// The categorical-state nodes, in node order.
  final categoricalNodes = <int>[];

  /// With [vectorState]: the nodes whose change of mean Δ sets an ef-state
  /// node's ν (the value parent of each statistic's node) or a categorical
  /// node's α (the value parent of each category's binary node). Each has
  /// that node as its only child, so its update is linear in the child's
  /// prediction error δ, Δ = K δ, and its posterior also writes the gain K
  /// (`gain`, 0 where the child is unobserved). The vector steps take Δ
  /// from K instead of subtracting means: in f32 an observation often
  /// equals its prediction (δ = Δ = 0, so δ/Δ would be NaN), and a small Δ
  /// cancels, where f64 still resolves both.
  final gainNodes = <int>{};

  /// Each input node's first data column and number of columns, in
  /// `Network.inputIdxs` order.
  final inputColumns = <(int, int)>[];

  /// The observation dimension of ef-state node [i].
  int efDimension(int i) => net.vectorLength(i, VectorField.mean);

  /// The number of sufficient statistics of ef-state node [i]
  /// (`d + d(d+1)/2`).
  int efStatistics(int i) => net.vectorLength(i, VectorField.xis);

  /// The number of categories of categorical node [i].
  int categoryCount(int i) => net.valueParentsOf(i).length;

  /// The length of [node]'s vector [field] on the GPU: an ef-state node's
  /// [VectorField.xis], a categorical node's [VectorField.alpha] and
  /// [VectorField.mean]; 0 for the others.
  int vectorLength(int node, VectorField field) =>
      switch ((net.kindOf(node), field)) {
        (NodeKind.efState, VectorField.xis) => efStatistics(node),
        (NodeKind.categoricalState, VectorField.alpha || VectorField.mean) =>
          categoryCount(node),
        _ => 0,
      };

  static const _vectorVar = {
    VectorField.xis: 'xi',
    VectorField.alpha: 'al',
    VectorField.mean: 'cx',
  };

  /// Slot of entry [k] of [node]'s vector [field] (see [vectorLength]).
  int vectorSlot(int node, VectorField field, int k) {
    final stem = _vectorVar[field];
    final v = stem == null ? null : slotOfVar['$stem${node}_$k'];
    if (v == null) throw ArgumentError('No GPU slot for $field[$k] of $node');
    return v;
  }

  /// Slot of ef-state node [node]'s expected sufficient statistic [k] (ξ).
  int xisSlot(int node, int k) => vectorSlot(node, VectorField.xis, k);

  /// Slot of ef-state node [node]'s learning rate ν.
  int nusSlot(int node) => fieldSlot(node, NodeField.nus);

  static const _extraVar = {
    NodeField.nus: 'nu',
    NodeField.klDivergence: 'kl',
    NodeField.surprise: 'su',
  };

  /// The slot variable of [field] of [node]: a node field of
  /// [gpuNodeFields], an ef-state node's [NodeField.nus], or a categorical
  /// node's [NodeField.klDivergence] or [NodeField.surprise]; null if it
  /// has none.
  String? varOf(int node, NodeField field) {
    if (gpuNodeFields.contains(field)) return fieldVar(node, field);
    final v = '${_extraVar[field]}$node';
    return slotOfVar.containsKey(v) ? v : null;
  }

  /// Slot of [field] of [node] (see [varOf]).
  int fieldSlot(int node, NodeField field) {
    final v = varOf(node, field);
    if (v == null) throw ArgumentError('No GPU slot for $field of $node');
    return slotOfVar[v]!;
  }

  /// Slot of the running total (then compensation, flag, result).
  late final int totalSlot;

  int get resultSlot => totalSlot + 3;

  /// Slots per element, including the result.
  int get nSlots => totalSlot + 4;

  /// Name of the slot variable of [field] of [node].
  static String fieldVar(int node, NodeField field) =>
      '${_fieldVar[gpuNodeFields.indexOf(field)]}$node';

  /// Slot of [field] of [node].
  int slot(int node, NodeField field) {
    final f = gpuNodeFields.indexOf(field);
    if (f < 0) throw ArgumentError.value(field, 'field', 'No GPU slot');
    return f * n + node;
  }

  int valueCouplingSlot(int parent, int child) {
    final v = slotOfVar['a${parent}_$child'];
    if (v == null) throw ArgumentError('No value edge $parent → $child');
    return v;
  }

  int volatilityCouplingSlot(int parent, int child) {
    final v = slotOfVar['k${parent}_$child'];
    if (v == null) throw ArgumentError('No volatility edge $parent → $child');
    return v;
  }

  /// Slot of response parameter [j].
  int responseSlot(int j) => totalSlot - response.defaults.length + j;

  /// The network's initial value of every slot, the response's defaults,
  /// and zero for the total and its bookkeeping.
  List<double> initialSlots() {
    final out = List<double>.filled(nSlots, 0.0);
    for (var f = 0; f < gpuNodeFields.length; f++) {
      for (var i = 0; i < n; i++) {
        out[f * n + i] = net.getInitial(i, gpuNodeFields[f]);
      }
    }
    var s = gpuNodeFields.length * n;
    for (var c = 0; c < n; c++) {
      for (final x in net.valueParentStrengthsOf(c)) {
        out[s++] = x;
      }
    }
    for (var c = 0; c < n; c++) {
      for (final x in net.volatilityParentStrengthsOf(c)) {
        out[s++] = x;
      }
    }
    for (final i in efNodes) {
      for (final x in net.getInitialVector(i, VectorField.xis)) {
        out[s++] = x;
      }
      out[s++] = net.getInitial(i, NodeField.nus);
    }
    for (final i in categoricalNodes) {
      for (final x in net.getInitialVector(i, VectorField.alpha)) {
        out[s++] = x;
      }
      for (final x in net.getInitialVector(i, VectorField.mean)) {
        out[s++] = x;
      }
      out[s++] = net.getInitial(i, NodeField.klDivergence);
      out[s++] = net.getInitial(i, NodeField.surprise);
    }
    s += gainNodes.length; // gain: written before read
    for (final d in response.defaults) {
      out[s++] = d;
    }
    return out;
  }

  /// Number of tangent chunks (vectors of up to 4 directions).
  int get chunks => (directions + 3) ~/ 4;

  /// Directions in chunk [c].
  int chunkWidth(int c) {
    final w = directions - 4 * c;
    return w > 4 ? 4 : w;
  }

  /// WGSL type of chunk [c].
  String chunkType(int c) {
    final w = chunkWidth(c);
    return w == 1 ? 'f32' : 'vec$w<f32>';
  }

  /// Compiles the trial, with [seeds] (slot variables) carrying tangents
  /// on entry; adds whatever they reach until the set is closed.
  CompiledTrial compile(TrialSource source, {Set<String> seeds = const {}}) {
    if (directions == 0 && seeds.isNotEmpty) {
      throw ArgumentError('Seeds without directions');
    }
    var active = {...seeds};
    for (;;) {
      final pass = _Pass(this, source, active);
      final needs = pass.run();
      if (needs.isEmpty) {
        final carried = {
          for (final e in pass.firstAccessRead.entries)
            if (e.value) e.key,
        };
        return CompiledTrial._(
          pass.body.toString(),
          carried,
          pass.written,
          active,
          slotOfVar,
        );
      }
      active = {...active, ...needs};
    }
  }

  /// Compiles the trial in SSA form, for reverse mode: one flat scope, every
  /// local and every write of a slot variable a fresh `let`, and the
  /// statements kept as expression trees ([SsaTrial.statements]) so their
  /// adjoints can be emitted in reverse order. No tangents, no `bad` flag
  /// and no running total (the trial's surprise is [SsaTrial.objective]).
  ///
  /// The unbounded update's expansions, guarded by `if` blocks in
  /// [compile], are computed unconditionally here and selected in.
  SsaTrial compileSsa(TrialSource source) {
    final pass = _Pass(this, source, const {}, ssa: true);
    pass.run();
    final carried = {
      for (final e in pass.firstAccessRead.entries)
        if (e.value) e.key,
    };
    return SsaTrial._(
      pass.body.toString(),
      List.unmodifiable(pass.statements),
      List.unmodifiable(pass.steps),
      pass.objective!,
      carried,
      pass.written,
      Map.unmodifiable(pass.current),
      slotOfVar,
    );
  }
}

/// A trial in SSA form ([TrialCompiler.compileSsa]).
final class SsaTrial {
  SsaTrial._(
    this.body,
    this.statements,
    this.steps,
    this.objective,
    this.carried,
    this.written,
    this.finalVersion,
    this.slotOfVar,
  );

  /// The primal statements (`let`s, flat), reading the carried slot
  /// variables by their own names.
  final String body;

  /// Every differentiable `let` of [body], in order: its name and value.
  final List<(String, Ex)> statements;

  /// Every `let` of [body] in order, the boolean ones included: its name
  /// and value (an [Ex] or a [Cond]). What the CPU evaluator runs.
  final List<(String, Object)> steps;

  /// The statement holding the trial's surprise.
  final String objective;

  /// Slot variables read before they are written.
  final Set<String> carried;

  /// Slot variables the trial writes.
  final Set<String> written;

  /// The name of each written slot variable's last version.
  final Map<String, String> finalVersion;

  final Map<String, int> slotOfVar;
}

/// One code-generation pass over the trial, for a fixed set of active slot
/// variables.
final class _Pass {
  _Pass(this.c, this.source, this.activeSlots, {this.ssa = false})
    : net = c.net,
      n = c.n,
      k = ssa ? 0 : c.directions;

  final TrialCompiler c;
  final TrialSource source;
  final Set<String> activeSlots;
  final Network net;
  final int n, k;

  /// SSA mode ([TrialCompiler.compileSsa]).
  final bool ssa;

  /// SSA mode: the differentiable statements, the current version of each
  /// written slot variable, and the surprise's statement.
  final statements = <(String, Ex)>[];
  final steps = <(String, Object)>[];
  final current = <String, String>{};
  String? objective;
  var _version = 0;
  String _fresh(String name) => '${name}_v${_version++}';

  /// `bad = bad || !(c)` (not in SSA mode, where the forward sweep has
  /// already flagged the element).
  void flag(Cond c) {
    if (!ssa) line('bad = bad || !${c.text};');
  }

  final body = StringBuffer();
  var _indent = 2;

  /// First access per slot variable: true if read before written.
  final firstAccessRead = <String, bool>{};
  final written = <String>{};

  /// Slot variables that would need tangents (to add, then recompile).
  final needs = <String>{};

  /// Local variables' activity, innermost scope last.
  final _scopes = <Map<String, bool>>[{}];

  late final tangents = Tangents(_isActive);

  bool _isActive(String name) {
    for (var s = _scopes.length - 1; s >= 0; s--) {
      final a = _scopes[s][name];
      if (a != null) return a;
    }
    if (c.slotOfVar.containsKey(name)) return activeSlots.contains(name);
    // Locals declared without tangents, and builtins.
    return false;
  }

  Set<String> run() {
    final seq = net.updateSequence;
    for (final (i, step) in seq.predictions) {
      _step(i, step);
    }
    line('// observations');
    final inputs = net.inputIdxs;
    for (var q = 0; q < inputs.length; q++) {
      final i = inputs[q];
      final (col, d) = c.inputColumns[q];
      // An ef-state or categorical input's first column is also its scalar
      // mean.
      w(i, NodeField.mean, Lit(source.input(col)));
      w(i, NodeField.observed, Lit(source.observed(q)));
      if (kind(i) == NodeKind.efState || kind(i) == NodeKind.categoricalState) {
        _observation[i] = [
          r(i, NodeField.mean),
          for (var j = 1; j < d; j++)
            let('x${i}_$j', Lit(source.input(col + j))),
        ];
      }
    }
    for (final (i, step) in seq.updates) {
      _step(i, step);
    }
    _response();
    return needs;
  }

  // ---------------------------------------------------------------------------
  // Emission
  // ---------------------------------------------------------------------------

  void line(String s) {
    body
      ..write('  ' * _indent)
      ..writeln(s);
  }

  void open(String s) {
    if (ssa && s.isNotEmpty) {
      throw UnsupportedError('Control flow has no SSA form: $s');
    }
    if (!ssa) line('$s {');
    _indent++;
    _scopes.add({});
  }

  void close([String s = '}']) {
    _indent--;
    _scopes.removeLast();
    if (ssa) {
      if (s != '}') throw UnsupportedError('Control flow has no SSA form: $s');
      return;
    }
    line(s);
    if (s.endsWith('{')) {
      _indent++;
      _scopes.add({});
    }
  }

  Ex? _d(Ex e, {Ex? self}) => k == 0 ? null : tangents.d(e, self: self);

  void _tangentLines(String prefix, String name, String op, Ex? dt) {
    for (var ch = 0; ch < c.chunks; ch++) {
      final type = c.chunkType(ch);
      final rhs = dt == null ? '$type()' : dt.emit(Emit(ch, type));
      line('$prefix${name}_t$ch $op $rhs;');
    }
  }

  /// `let name = e;` (and its tangent).
  Ex let(String name, Object e) {
    final x = ex(e);
    if (ssa) {
      final v = _fresh(name);
      line('let $v = ${x.w};');
      statements.add((v, x));
      steps.add((v, x));
      return Ref(v);
    }
    line('let $name = ${x.w};');
    final dt = _d(x, self: Ref(name));
    if (dt != null) _tangentLines('let ', name, '=', dt);
    _scopes.last[name] = dt != null;
    return Ref(name);
  }

  /// A boolean `let`.
  Cond letCond(String name, Cond c) {
    if (ssa) {
      name = _fresh(name);
      steps.add((name, c));
    }
    line('let $name = ${c.text};');
    return Cond(name);
  }

  /// `var name = e;`, with a tangent if [e] has one or [active].
  Ex variable(String name, Object e, {bool active = false}) {
    if (ssa) throw UnsupportedError('Mutable local $name has no SSA form');
    final x = ex(e);
    line('var $name = ${x.w};');
    final dt = _d(x);
    final has = k > 0 && (dt != null || active);
    if (has) _tangentLines('var ', name, '=', dt);
    _scopes.last[name] = has;
    return Ref(name);
  }

  /// `name = e;` for a local or slot variable.
  void set(String name, Object e) {
    final x = ex(e);
    if (ssa) {
      if (!c.slotOfVar.containsKey(name)) {
        throw UnsupportedError('Mutable local $name has no SSA form');
      }
      final v = _fresh(name);
      line('let $v = ${x.w};');
      statements.add((v, x));
      steps.add((v, x));
      current[name] = v;
      return;
    }
    final dt = _d(x);
    if (_isActive(name)) {
      _tangentLines('', name, '=', dt);
    } else if (dt != null) {
      if (c.slotOfVar.containsKey(name)) {
        needs.add(name);
      } else {
        throw StateError('Active value assigned to inactive local $name');
      }
    }
    line('$name = ${x.w};');
  }

  /// Whether any of [es] carries a tangent.
  bool anyActive(Iterable<Ex> es) => k > 0 && es.any((e) => _d(e) != null);

  var _tmp = 0;
  String tmp([String stem = 't']) => '$stem${_tmp++}';

  // ---------------------------------------------------------------------------
  // Variables
  // ---------------------------------------------------------------------------

  /// Reads node field [f] of [node].
  Ex r(int node, NodeField f) {
    final v = TrialCompiler.fieldVar(node, f);
    firstAccessRead.putIfAbsent(v, () => true);
    return Ref(current[v] ?? v);
  }

  /// Writes node field [f] of [node].
  void w(int node, NodeField f, Object e) {
    final x = ex(e); // reads recorded first
    final v = TrialCompiler.fieldVar(node, f);
    firstAccessRead.putIfAbsent(v, () => false);
    written.add(v);
    set(v, x);
  }

  Ex _slotRead(String v) {
    firstAccessRead.putIfAbsent(v, () => true);
    return Ref(current[v] ?? v);
  }

  /// Writes a slot variable other than a node field.
  void _slotWrite(String v, Object e) {
    final x = ex(e);
    firstAccessRead.putIfAbsent(v, () => false);
    written.add(v);
    set(v, x);
  }

  Ex a(int parent, int child) => _slotRead('a${parent}_$child');
  Ex kap(int parent, int child) => _slotRead('k${parent}_$child');
  Ex rp(int j) => _slotRead('rp$j');

  static const dt = Ref('dt');
  static const logT = Ref('logT');

  // ---------------------------------------------------------------------------
  // Graph helpers
  // ---------------------------------------------------------------------------

  NodeKind kind(int i) => net.kindOf(i);
  bool isLeaf(int i) =>
      net.valueChildrenOf(i).isEmpty && net.volatilityChildrenOf(i).isEmpty;

  /// An input node with noise parents (pyhgf's `is_noise_child`, #477): a
  /// continuous leaf with volatility parents.
  bool isNoiseChild(int i) =>
      kind(i) == NodeKind.continuousState &&
      isLeaf(i) &&
      net.volatilityParentsOf(i).isNotEmpty;
  bool get meanField => net.meanFieldUpdates;

  /// The value coupling of [parent]: a [CouplingFn] or an [ExprCoupling]
  /// (a [CustomCoupling]'s closures can't be lowered).
  Coupling couplingFn(int parent) {
    final cf = net.couplingOf(parent);
    if (cf is CouplingFn || cf is ExprCoupling) return cf;
    throw UnsupportedError(
      'Custom coupling functions have no GPU kernel (node $parent); write '
      'it as an ExprCoupling',
    );
  }

  /// Left-to-right sum of [terms] (0 if empty).
  static Ex sum(Iterable<Ex> terms) {
    Ex? out;
    for (final t in terms) {
      out = out == null ? t : out + t;
    }
    return out ?? zero;
  }

  // ---------------------------------------------------------------------------
  // Steps
  // ---------------------------------------------------------------------------

  void _step(int i, UpdateStep step) {
    line('// ${step.name} $i');
    open('');
    switch (step) {
      case UpdateStep.predictionContinuous:
        _predictionContinuous(i, meanField: false);
      case UpdateStep.predictionContinuousMeanField:
        _predictionContinuous(i, meanField: true);
      case UpdateStep.predictionBinary:
        _predictionBinary(i);
      case UpdateStep.predictionErrorContinuous:
        _predictionErrorContinuous(i);
      case UpdateStep.predictionErrorBinary:
        _predictionErrorBinary(i);
      case UpdateStep.noisePredictionErrorContinuous:
        _noisePredictionErrorContinuous(i);
      case UpdateStep.posteriorContinuous:
        _posteriorStandard(i, meanField: false);
      case UpdateStep.posteriorContinuousMeanField:
        _posteriorStandard(i, meanField: true);
      case UpdateStep.posteriorContinuousEhgf:
        _posteriorEhgf(i, meanField: false);
      case UpdateStep.posteriorContinuousEhgfMeanField:
        _posteriorEhgf(i, meanField: true);
      case UpdateStep.posteriorContinuousUnbounded:
        _posteriorUnbounded(i);
      case UpdateStep.predictionErrorEfFixed:
        _predictionErrorEfFixed(i);
      case UpdateStep.predictionErrorEfDynamic:
        _predictionErrorEfDynamic(i);
      case UpdateStep.posteriorEfDynamic:
        _posteriorEfDynamic(i);
      case UpdateStep.predictionErrorCategorical:
        _predictionErrorCategorical(i);
      case UpdateStep.posteriorCategorical:
        _posteriorCategorical(i);
      default:
        throw UnsupportedError('${step.name} has no GPU kernel yet');
    }
    close();
  }

  /// `g(x)`, `g'(x)`, `g''(x)` of [fn].
  Ex g(Coupling c, Ex x) => switch (c) {
    final ExprCoupling e => lowerExpr(e.g, (_) => x),
    final CouplingFn fn => _g(fn, x),
    _ => throw StateError('$c'),
  };

  Ex dg(Coupling c, Ex x) => switch (c) {
    final ExprCoupling e => lowerExpr(e.dg, (_) => x),
    final CouplingFn fn => _dg(fn, x),
    _ => throw StateError('$c'),
  };

  Ex d2g(Coupling c, Ex x) => switch (c) {
    final ExprCoupling e => lowerExpr(e.d2g, (_) => x),
    final CouplingFn fn => _d2g(fn, x),
    _ => throw StateError('$c'),
  };

  Ex _g(CouplingFn fn, Ex x) => switch (fn) {
    CouplingFn.linear => x,
    CouplingFn.relu => max(x, zero),
    CouplingFn.sigmoid => sigm(x),
    CouplingFn.tanh => tanh(x),
    CouplingFn.leakyRelu => sel(Lit('0.01f') * x, x, ge(x, zero)),
    CouplingFn.gelu => call('gelu', [x]),
  };

  Ex _dg(CouplingFn fn, Ex x) => switch (fn) {
    CouplingFn.linear => one,
    CouplingFn.relu => sel(zero, one, gt(x, zero)),
    CouplingFn.sigmoid => call('dsigm', [x]),
    CouplingFn.tanh => call('dtanh', [x]),
    CouplingFn.leakyRelu => sel(Lit('0.01f'), one, ge(x, zero)),
    CouplingFn.gelu => call('dgelu', [x]),
  };

  Ex _d2g(CouplingFn fn, Ex x) => switch (fn) {
    CouplingFn.linear || CouplingFn.relu || CouplingFn.leakyRelu => zero,
    CouplingFn.sigmoid => call('d2sigm', [x]),
    CouplingFn.tanh => call('d2tanh', [x]),
    CouplingFn.gelu => call('d2gelu', [x]),
  };

  void _predictionContinuous(int i, {required bool meanField}) {
    final input = isLeaf(i);
    final driftTerms = <Ex>[r(i, NodeField.tonicDrift)];
    final vcvTerms = <Ex>[];
    for (final p in net.valueParentsOf(i)) {
      final alpha = a(p, i);
      final pm = r(p, NodeField.expectedMean);
      if (kind(p) == NodeKind.constantState) {
        // μ̂ = 1, π̂ = ∞: drift only.
        driftTerms.add(alpha * pm);
        continue;
      }
      final fn = couplingFn(p);
      driftTerms.add(alpha * g(fn, pm));
      if (!meanField && !input) {
        final cc = let(tmp('c'), dt * alpha * dg(fn, pm));
        vcvTerms.add(cc * cc / r(p, NodeField.expectedPrecision));
      }
    }
    final volTerms = <Ex>[r(i, NodeField.tonicVolatility)];
    final volParents = net.volatilityParentsOf(i);
    for (final p in volParents) {
      final kappa = kap(p, i);
      volTerms.add(kappa * r(p, NodeField.expectedMean));
      // An input's volatility parents are noise parents, without MGF terms.
      if (!meanField && !input) {
        volTerms.add(
          (kappa * kappa) / (Lit('2.0f') * r(p, NodeField.expectedPrecision)),
        );
      }
    }
    final drift = let('drift', sum(driftTerms));
    final lambda = let('lambda', r(i, NodeField.autoconnectionStrength));
    final prec = let('prec', r(i, NodeField.precision));
    // An input is generated by its value parents, not drifted (pyhgf #478).
    final expMean = let(
      'expMean',
      lambda * r(i, NodeField.mean) + (input ? drift : dt * drift),
    );
    final carried = let('carried', (lambda * lambda) / prec);
    if (input) {
      // Observed exactly, no drift between observations: the precision of
      // the observation noise, π̂ = exp(−(ε_u + Σ κ μ̂_q)) with noise
      // parents q (γ = 1), else the prior π (γ = 0). pyhgf #477.
      final Ex ep;
      if (volParents.isEmpty) {
        ep = prec;
      } else {
        ep = let('noisePrec', exp(-sum(volTerms)));
      }
      w(i, NodeField.currentVariance, carried);
      w(i, NodeField.expectedMean, expMean);
      w(i, NodeField.effectivePrecision, volParents.isEmpty ? zero : one);
      w(i, NodeField.expectedPrecision, ep);
      w(i, NodeField.conditionalExpectedPrecision, ep);
      return;
    }
    final totvol = let('totvol', sum(volTerms));
    final pv = let('pv', dt * exp(totvol));
    // f64: pv ≤ 1e-128 becomes NaN; here pv ≤ 0 (underflow) flags the
    // element.
    flag(gt(pv, zero));
    Ex? cond;
    final Ex expected;
    if (!meanField) {
      final vcv = let('vcv', sum(vcvTerms));
      cond = let('cond', one / max(carried + pv, F32.minNormal));
      expected = let('expected', one / max(carried + pv + vcv, F32.minNormal));
    } else {
      expected = let('expected', one / max(carried + pv, F32.minNormal));
    }
    w(i, NodeField.currentVariance, carried);
    w(i, NodeField.expectedMean, expMean);
    w(i, NodeField.effectivePrecision, pv * expected);
    w(i, NodeField.expectedPrecision, expected);
    w(i, NodeField.conditionalExpectedPrecision, meanField ? expected : cond!);
  }

  void _predictionBinary(int i) {
    final parents = net.valueParentsOf(i);
    final s = sum([for (final p in parents) r(p, NodeField.expectedMean)]);
    final v = lit(net.precisionClippingValue);
    // 1 − μ̂ from σ(−s), not by subtraction, which loses f32 precision
    // where μ̂ is near 1; π̂ = μ̂ (1 − μ̂) keeps it (see [oneMinusMean]).
    final pr = let('pr', clamp(sigm(s), v, one - v));
    final qr = let('qr', clamp(sigm(-s), v, one - v));
    w(i, NodeField.expectedMean, pr);
    w(i, NodeField.expectedPrecision, pr * qr);
  }

  /// `1 − μ̂` of [node]: for a binary node `π̂ / μ̂` (exact to rounding,
  /// where `1 − μ̂` cancels catastrophically in f32 near 1).
  Ex oneMinusMean(int node) => kind(node) == NodeKind.binaryState
      ? r(node, NodeField.expectedPrecision) / r(node, NodeField.expectedMean)
      : one - r(node, NodeField.expectedMean);

  Cond anyObserved(int i) {
    final children = [
      ...net.valueChildrenOf(i),
      ...net.volatilityChildrenOf(i),
    ];
    if (children.isEmpty) return no;
    return children
        .map((ch) => ne(r(ch, NodeField.observed), zero))
        .reduce((x, y) => x | y);
  }

  void _predictionErrorContinuous(int i) {
    final leaf = isLeaf(i);
    final epx = let('epx', r(i, NodeField.expectedPrecision));
    final delta = let(
      'delta',
      r(i, NodeField.mean) - r(i, NodeField.expectedMean),
    );
    w(i, NodeField.valuePredictionError, delta);
    // A noise child sends its noise prediction error instead
    // ([_noisePredictionErrorContinuous]).
    if (!isNoiseChild(i)) {
      w(
        i,
        NodeField.volatilityPredictionError,
        (epx / r(i, NodeField.precision)) + epx * delta * delta - one,
      );
    }
    if (!leaf) {
      w(i, NodeField.observed, sel(zero, one, anyObserved(i)));
    }
  }

  /// The noise prediction error of input [i] (Eq. 72 of Weber et al.,
  /// 2026; pyhgf #477): ε = π̂_u (Σ_b (α_b g_b'(μ_b))² / π_b
  /// + (u − m_u)²) − 1, with m_u the prediction of u from the value
  /// parents' posteriors. No time step: the input is generated by its
  /// value parents, not drifted (pyhgf #478).
  void _noisePredictionErrorContinuous(int i) {
    final meanTerms = <Ex>[r(i, NodeField.expectedMean)];
    final varTerms = <Ex>[];
    for (final p in net.valueParentsOf(i)) {
      final alpha = a(p, i);
      final pm = r(p, NodeField.mean), pem = r(p, NodeField.expectedMean);
      if (kind(p) == NodeKind.constantState) {
        // Linear. A bias parent is known exactly; pyhgf reads its
        // placeholder `precision` (1) here, adding α² of variance
        // ([NetworkFixes.constantNoiseVariance], P21).
        meanTerms.add(alpha * (pm - pem));
        if (!net.fixes.constantNoiseVariance) {
          final cc = let(tmp('c'), alpha);
          varTerms.add(cc * cc / r(p, NodeField.precision));
        }
        continue;
      }
      final fn = couplingFn(p);
      meanTerms.add(alpha * (g(fn, pm) - g(fn, pem)));
      final cc = let(tmp('c'), alpha * dg(fn, pm));
      varTerms.add(cc * cc / r(p, NodeField.precision));
    }
    final pmean = let('npm', sum(meanTerms));
    final pvar = let('npv', sum(varTerms));
    final d = let('nd', r(i, NodeField.mean) - pmean);
    w(
      i,
      NodeField.volatilityPredictionError,
      r(i, NodeField.expectedPrecision) * (pvar + d * d) - one,
    );
  }

  void _predictionErrorBinary(int i) {
    final epx = let('epx', r(i, NodeField.expectedPrecision));
    // u − μ̂ as u (1 − μ̂) − (1 − u) μ̂: exact for u ∈ {0, 1}.
    final u = r(i, NodeField.mean);
    final q = let('q', oneMinusMean(i));
    w(
      i,
      NodeField.valuePredictionError,
      (u * q - (one - u) * r(i, NodeField.expectedMean)) *
          r(i, NodeField.observed) /
          epx,
    );
    w(i, NodeField.precision, epx);
  }

  // ---- posterior building blocks ----

  Ex relaxedGain(int ch, {required bool meanField}) {
    final cep = r(ch, NodeField.expectedPrecision);
    if (meanField || kind(ch) != NodeKind.continuousState) return cep;
    final cond = r(ch, NodeField.conditionalExpectedPrecision);
    if (isLeaf(ch)) return cond;
    final cp = r(ch, NodeField.precision);
    return cond * cp / (cond + (cp - cep));
  }

  Ex relaxedEffective(int ch, {required bool meanField}) {
    final cep = r(ch, NodeField.expectedPrecision);
    if (meanField || kind(ch) != NodeKind.continuousState) return cep;
    final cond = r(ch, NodeField.conditionalExpectedPrecision);
    if (isLeaf(ch)) return cond;
    final cp = r(ch, NodeField.precision);
    return cond * (cp - cep) / (cond + (cp - cep));
  }

  /// The coupling of value child [ch] to [i] as the child's prediction
  /// sees it: t·α for a state child, whose drift the time step scales, α
  /// for an input (pyhgf #478).
  Ex _vcAlpha(int i, int ch) =>
      kind(ch) == NodeKind.continuousState && !isLeaf(ch)
      ? a(i, ch) * dt
      : a(i, ch);

  /// The value-children precision increment of [i].
  Ex valuePrecisionWpe(int i, {required bool meanField}) {
    final children = net.valueChildrenOf(i);
    if (children.isEmpty) return zero;
    final fn = couplingFn(i);
    return sum([
      for (final ch in children)
        if (fn == CouplingFn.linear)
          relaxedEffective(ch, meanField: meanField) *
              _vcAlpha(i, ch) *
              _vcAlpha(i, ch) *
              r(ch, NodeField.observed)
        else
          (relaxedEffective(ch, meanField: meanField) *
                      _vcAlpha(i, ch) *
                      _vcAlpha(i, ch) *
                      dg(fn, r(i, NodeField.mean)) *
                      dg(fn, r(i, NodeField.mean)) -
                  relaxedGain(ch, meanField: meanField) *
                      _vcAlpha(i, ch) *
                      d2g(fn, r(i, NodeField.mean)) *
                      r(ch, NodeField.valuePredictionError)) *
              r(ch, NodeField.observed),
    ]);
  }

  /// The value-children mean increment of [i] at precision [prec].
  Ex valueMeanWpe(int i, Ex prec, {required bool meanField}) {
    final children = net.valueChildrenOf(i);
    if (children.isEmpty) return zero;
    return sum([
      for (final ch in children)
        valueMeanGain(i, ch, prec, meanField: meanField) *
            (r(ch, NodeField.valuePredictionError) * r(ch, NodeField.observed)),
    ]);
  }

  /// The mean increment of [i] per unit of value child [ch]'s prediction
  /// error, at precision [prec].
  Ex valueMeanGain(int i, int ch, Ex prec, {required bool meanField}) =>
      _vcAlpha(i, ch) *
      dg(couplingFn(i), r(i, NodeField.mean)) *
      relaxedGain(ch, meanField: meanField) /
      prec;

  /// The gain of [i], which sets an ef-state node's ν or a categorical
  /// node's α ([TrialCompiler.gainNodes]): its mean increment per unit of
  /// its only child's prediction error, 0 where the child is unobserved.
  void _gain(int i, Ex prec, {required bool meanField}) {
    final ch = net.valueChildrenOf(i).single;
    _slotWrite(
      'gain$i',
      valueMeanGain(i, ch, prec, meanField: meanField) *
          r(ch, NodeField.observed),
    );
  }

  /// The standard volatility precision increments of [i] (of its noise
  /// children only with [noiseOnly]).
  List<Ex> volatilityPrecisionWpe(int i, {bool noiseOnly = false}) => [
    for (final ch in net.volatilityChildrenOf(i))
      if (!noiseOnly || isNoiseChild(ch)) standardIncrement(i, ch),
  ];

  /// ½ (κγ)² + (κγ)² Δ − ½ κ² γ Δ of volatility child [ch]; for a noise
  /// child ½ κ² (1 + ε) (Eq. 78 of Weber et al., 2026).
  Ex standardIncrement(int i, int ch) => () {
    final kappa = kap(i, ch);
    final e = r(ch, NodeField.effectivePrecision);
    final v = r(ch, NodeField.volatilityPredictionError);
    final ke = let(tmp('ke'), kappa * e);
    return (Lit('0.5f') * ke * ke +
            ke * ke * v -
            Lit('0.5f') * kappa * kappa * e * v) *
        r(ch, NodeField.observed);
  }();

  Ex volatilityMeanWpe(int i, Ex prec, {bool noiseOnly = false}) => sum([
    for (final ch in net.volatilityChildrenOf(i))
      if (!noiseOnly || isNoiseChild(ch))
        (kap(i, ch) *
                r(ch, NodeField.effectivePrecision) *
                r(ch, NodeField.volatilityPredictionError)) /
            (Lit('2.0f') * prec) *
            r(ch, NodeField.observed),
  ]);

  Ex observedOrPredicted(int i, Cond hasObs, Ex wpe) {
    final ep = r(i, NodeField.expectedPrecision);
    return min(
      sel(ep, max(ep + wpe, F32.minNormal), hasObs),
      lit(net.maxPosteriorPrecision),
    );
  }

  /// ω̃ of volatility child [ch] seen from parent [i].
  Ex effectiveTonic(int i, int ch) => sum([
    r(ch, NodeField.tonicVolatility),
    for (final p in net.volatilityParentsOf(ch))
      if (p != i) kap(p, ch) * r(p, NodeField.expectedMean),
  ]);

  Ex mgfTerms(int ch) => sum([
    for (final p in net.volatilityParentsOf(ch))
      kap(p, ch) *
          kap(p, ch) /
          (Lit('2.0f') * r(p, NodeField.expectedPrecision)),
  ]);

  Ex valueCouplingVariance(int ch) => max(
    one / r(ch, NodeField.expectedPrecision) -
        one / r(ch, NodeField.conditionalExpectedPrecision),
    zero,
  );

  void _posteriorStandard(int i, {required bool meanField}) {
    final hasObs = letCond('hasObs', anyObserved(i));
    final wpe = let(
      'wpe',
      sum([
        valuePrecisionWpe(i, meanField: meanField),
        ...volatilityPrecisionWpe(i),
      ]),
    );
    final prec = let('prec', observedOrPredicted(i, hasObs, wpe));
    final vm = let('vm', valueMeanWpe(i, prec, meanField: meanField));
    final volm = let('volm', volatilityMeanWpe(i, prec));
    if (c.gainNodes.contains(i)) _gain(i, prec, meanField: meanField);
    w(i, NodeField.precision, prec);
    w(i, NodeField.mean, r(i, NodeField.expectedMean) + (vm + volm));
  }

  /// The eHGF precision increment of volatility child [ch] (lets named by
  /// [q]).
  Ex ehgfIncrement(int i, int ch, int q, Ex tonic) {
    final kappa = kap(i, ch);
    final prevVar = let(
      'pvv_$q',
      r(ch, NodeField.currentVariance) + valueCouplingVariance(ch),
    );
    final pv = let('pve_$q', dt * exp(kappa * r(i, NodeField.mean) + tonic));
    final ep = let('epe_$q', one / (prevVar + pv));
    final eff = let('effe_$q', pv * ep);
    final weight = let('wte_$q', (pv - prevVar) * ep);
    final d = r(ch, NodeField.mean) - r(ch, NodeField.expectedMean);
    final vope = let(
      'vpe_$q',
      (one / r(ch, NodeField.precision) + d * d) * ep - one,
    );
    final inc = let(
      'inc_$q',
      Lit('0.5f') * kappa * kappa * eff * (eff + weight * vope),
    );
    return max(inc, zero) * r(ch, NodeField.observed);
  }

  void _posteriorEhgf(int i, {required bool meanField}) {
    final epi = let('epi', r(i, NodeField.expectedPrecision));
    final vm = let('vm', valueMeanWpe(i, epi, meanField: meanField));
    final volm = let('volm', volatilityMeanWpe(i, epi));
    if (c.gainNodes.contains(i)) _gain(i, epi, meanField: meanField);
    w(i, NodeField.mean, r(i, NodeField.expectedMean) + vm + volm);
    final hasObs = letCond('hasObs', anyObserved(i));
    final mgf = !meanField;
    final children = net.volatilityChildrenOf(i);
    final incs = <Ex>[
      for (var q = 0; q < children.length; q++)
        if (isNoiseChild(children[q]))
          standardIncrement(i, children[q])
        else
          ehgfIncrement(
            i,
            children[q],
            q,
            mgf
                ? effectiveTonic(i, children[q]) + mgfTerms(children[q])
                : effectiveTonic(i, children[q]),
          ),
    ];
    final wpe = let(
      'wpe',
      sum([valuePrecisionWpe(i, meanField: meanField), ...incs]),
    );
    w(i, NodeField.precision, observedOrPredicted(i, hasObs, wpe));
  }

  void _posteriorUnbounded(int i) {
    final mf = meanField;
    // Noise children enter the effective prior (Eqs. 74 and 78 of Weber et
    // al., 2026), not the expansions (pyhgf #477).
    final all = net.volatilityChildrenOf(i);
    final children = [
      for (final ch in all)
        if (!isNoiseChild(ch)) ch,
    ];
    final hasNoise = children.length != all.length;
    final emi = let('emi', r(i, NodeField.expectedMean));
    final Ex pp, pm;
    if (net.valueChildrenOf(i).isEmpty && !hasNoise) {
      pp = let('pp', r(i, NodeField.expectedPrecision));
      pm = let('pm', emi);
    } else {
      final vpw = let('vpw', valuePrecisionWpe(i, meanField: mf));
      final npw = let('npw', sum(volatilityPrecisionWpe(i, noiseOnly: true)));
      pp = let(
        'pp',
        max(r(i, NodeField.expectedPrecision) + vpw + npw, F32.minNormal),
      );
      final vmw = let('vmw', valueMeanWpe(i, pp, meanField: mf));
      final nmw = let('nmw', volatilityMeanWpe(i, pp, noiseOnly: true));
      pm = let('pm', emi + vmw + nmw);
      // No volatility children, so one expansion: the mean is pm.
      if (c.gainNodes.contains(i)) _gain(i, pp, meanField: mf);
    }
    final nc = children.length;
    // Per child: observed, coupling, ω̃, previous variance, b, vope.
    final ob = <Cond>[], ck = <Ex>[], to = <Ex>[], pv = <Ex>[];
    final b = <Ex>[], vo = <Ex>[];
    for (var q = 0; q < nc; q++) {
      final ch = children[q];
      ob.add(letCond('ob_$q', ne(r(ch, NodeField.observed), zero)));
      ck.add(let('ck_$q', kap(i, ch)));
      to.add(let('to_$q', effectiveTonic(i, ch)));
      pv.add(
        let(
          'pv_$q',
          max(
            r(ch, NodeField.currentVariance) + valueCouplingVariance(ch),
            F32.minNormal,
          ),
        ),
      );
      final d = let(
        'd_$q',
        r(ch, NodeField.mean) - r(ch, NodeField.expectedMean),
      );
      b.add(let('b_$q', one / r(ch, NodeField.precision) + d * d));
      vo.add(let('vo_$q', r(ch, NodeField.expectedPrecision) * b[q] - one));
    }
    const half = Lit('0.5f');
    // logS of child q at x, and w = S / (S + pv) in the linear or the log
    // regime.
    (Ex logS, Cond linear, Ex wq) weightAt(int q, Ex x, String name) {
      final logS = let('logS_$name', logT + ck[q] * x + to[q]);
      final linear = letCond('lin_$name', le(logS, F32.linearLogSMax));
      final wq = let(
        'w_$name',
        sel(sigm(logS - log(pv[q])), exp(logS) / (exp(logS) + pv[q]), linear),
      );
      return (logS, linear, wq);
    }

    // Expansion 0: at the prediction, concave part only.
    final p0 = <Ex>[pp], s0 = <Ex>[pp * (pm - emi)];
    for (var q = 0; q < nc; q++) {
      final (_, _, wq) = weightAt(q, emi, '0_$q');
      p0.add(sel(zero, half * ck[q] * ck[q] * wq * (one - wq), ob[q]));
      s0.add(sel(zero, half * ck[q] * wq * vo[q], ob[q]));
    }
    final pList = <Ex>[let('P_0', sum(p0))];
    final mList = <Ex>[let('M_0', emi + let('S_0', sum(s0)) / pList[0])];
    final vList = <Cond>[yes];
    // Expansions 1..nc: at each child's Lambert W0 mode.
    final act = anyActive([pp, pm, emi, ...ck, ...to, ...pv, ...b]);
    for (var x = 1; x <= nc; x++) {
      final q = x - 1;
      // SSA (reverse mode) has no `if`: the expansion is computed always
      // and selected in, with κ replaced by 1 where it is off so that the
      // discarded values stay finite (their adjoints are zero, and a zero
      // adjoint times an infinite partial would be NaN).
      Ex? px, mx;
      Cond on = yes;
      Ex cq = ck[q];
      if (ssa) {
        on = letCond('on_$x', ob[q] & ne(ck[q], zero));
        cq = let('cq_$x', sel(one, ck[q], on));
      } else {
        px = variable('P_$x', zero, active: act);
        mx = variable('M_$x', zero, active: act);
        line('var V_$x = false;');
        open('if (${ob[q].text} && ${ne(ck[q], zero).text})');
      }
      final pihat = let('pihat', pp / (cq * cq));
      final gam = let('gam', logT + cq * pm + to[q]);
      final w0 = let(
        'w0',
        b[q] / (Lit('2.0f') * pihat) * exp(half / pihat - gam),
      );
      final logW = let(
        'logW',
        log(b[q]) - log(Lit('2.0f') * pihat) + half / pihat - gam,
      );
      final warg = let(
        'warg',
        sel(exp(min(logW, F32.logMaxFinite)), w0, gt(w0, zero) & isFinite(w0)),
      );
      final lw = let('lw', lambertW0(warg));
      final xs = let('xs', (gam + lw - half / pihat - logT - to[q]) / cq);
      final full = <Ex>[pp], conc = <Ex>[pp], grad = <Ex>[-pp * (xs - pm)];
      for (var r2 = 0; r2 < nc; r2++) {
        final (logS, linear, wr) = weightAt(r2, xs, 'm$r2');
        final invD = let(
          'invD_$r2',
          sel(
            exp(-logAddExp(log(pv[r2]), logS)),
            one / (exp(logS) + pv[r2]),
            linear,
          ),
        );
        final da = let('da_$r2', b[r2] * invD - one);
        full.add(
          sel(
            zero,
            half * ck[r2] * ck[r2] * wr * (wr + (Lit('2.0f') * wr - one) * da),
            ob[r2],
          ),
        );
        conc.add(sel(zero, half * ck[r2] * ck[r2] * wr * (one - wr), ob[r2]));
        grad.add(sel(zero, half * ck[r2] * wr * da, ob[r2]));
      }
      final fullV = let('full', sum(full));
      final concV = let('conc', sum(conc));
      final gradV = let('grad', sum(grad));
      if (ssa) {
        final pChosen = let('Pc_$x', sel(fullV, concV, le(fullV, zero)));
        px = let('P_$x', sel(zero, pChosen, on));
        mx = let('M_$x', sel(zero, xs + gradV / pChosen, on));
        vList.add(letCond('V_$x', on & isFinite(px) & isFinite(mx)));
      } else {
        set('P_$x', sel(fullV, concV, le(fullV, zero)));
        set('M_$x', xs + gradV / px!);
        line('V_$x = ${(isFinite(px) & isFinite(mx!)).text};');
        close();
        vList.add(Cond('V_$x'));
      }
      pList.add(px);
      mList.add(mx);
    }
    // Energies.
    final energies = <Ex>[];
    for (var x = 0; x <= nc; x++) {
      final mx = mList[x];
      final terms = <Ex>[-half * pp * (mx - pm) * (mx - pm)];
      for (var q = 0; q < nc; q++) {
        final logS = let('logS_e${x}_$q', logT + ck[q] * mx + to[q]);
        final den = exp(logS) + pv[q];
        final logD = logAddExp(log(pv[q]), logS);
        terms.add(
          sel(
            zero,
            sel(
              -half * logD - half * b[q] * exp(-logD),
              -half * log(den) - half * b[q] / den,
              le(logS, F32.linearLogSMax),
            ),
            ob[q],
          ),
        );
      }
      energies.add(let('E_$x', sum(terms)));
    }
    // Softmax and moment matching over the valid expansions.
    var maxE = energies[0];
    for (var x = 1; x <= nc; x++) {
      maxE = let(
        'maxE_$x',
        sel(maxE, energies[x], vList[x] & gt(energies[x], maxE)),
      );
    }
    final u = <Ex>[
      for (var x = 0; x <= nc; x++)
        let('u_$x', sel(zero, exp(energies[x] - maxE), vList[x])),
    ];
    final tot = let('tot', sum(u));
    final pmean = let(
      'pmean',
      sum([
        for (var x = 0; x <= nc; x++)
          sel(zero, (u[x] / tot) * mList[x], vList[x]),
      ]),
    );
    final pvar = let(
      'pvar',
      sum([
        for (var x = 0; x <= nc; x++)
          sel(
            zero,
            (u[x] / tot) *
                (one / pList[x] + (mList[x] - pmean) * (mList[x] - pmean)),
            vList[x],
          ),
      ]),
    );
    final hasObs = letCond('hasObs', anyObserved(i));
    w(
      i,
      NodeField.precision,
      min(
        sel(r(i, NodeField.expectedPrecision), one / pvar, hasObs),
        lit(net.maxPosteriorPrecision),
      ),
    );
    w(i, NodeField.mean, pmean);
  }

  // ---------------------------------------------------------------------------
  // ef-state
  // ---------------------------------------------------------------------------

  /// Each ef-state or categorical input's observation this trial (its
  /// columns).
  final _observation = <int, List<Ex>>{};

  /// t(x) of [i]'s observation: `x`, then the lower triangle of `x xᵀ` row
  /// by row (`[x, x²]` in one dimension).
  List<Ex> _efSufficientStatistics(int i) {
    final x = _observation[i]!;
    return [
      ...x,
      for (var row = 0; row < x.length; row++)
        for (var col = 0; col <= row; col++) x[row] * x[col],
    ];
  }

  /// ξ ← ξ + (t(x) − ξ) / (1 + ν) where [on] (the node is observed).
  void _efMoveXis(int i, Cond on) {
    final ss = _efSufficientStatistics(i);
    final rate = let('rate', one / (one + _slotRead('nu$i')));
    for (var k = 0; k < ss.length; k++) {
      final xi = _slotRead('xi${i}_$k');
      _slotWrite('xi${i}_$k', sel(xi, xi + rate * (ss[k] - xi), on));
    }
  }

  /// A fixed learning rate (no parents).
  void _predictionErrorEfFixed(int i) {
    if (!c.vectorState) return;
    _efMoveXis(i, letCond('on', ne(r(i, NodeField.observed), zero)));
  }

  /// HGF learning: each sufficient statistic becomes the observation of
  /// the value parent tracking it.
  void _predictionErrorEfDynamic(int i) {
    final ss = _efSufficientStatistics(i);
    final ob = let('ob', r(i, NodeField.observed));
    final parents = net.valueParentsOf(i);
    for (var k = 0; k < parents.length; k++) {
      w(parents[k], NodeField.observed, ob);
      w(parents[k], NodeField.mean, ss[k]);
    }
  }

  /// HGF learning, at the end of the sweep: ν is the mean over the
  /// statistics of δ/Δ, each parent's prediction error over the change in
  /// its own value parent's expectation; then ξ moves as with a fixed rate.
  ///
  /// Each δ/Δ is 1/K, K the gain of the parent's update, which is linear
  /// in δ (see [TrialCompiler.gainNodes]): the same value where δ ≠ 0;
  /// where δ = 0 the CPU engines get 0/0.
  ///
  /// Unobserved, both changes are 0 and the CPU engines store ν = 0/0
  /// (NaN), which nothing reads before the next observed trial sets it
  /// again; the kernel keeps the last ν instead (and divides by 1 there,
  /// so that no derivative is NaN). ξ and the likelihood are the same.
  void _posteriorEfDynamic(int i) {
    if (!c.vectorState) return;
    final on = letCond('on', ne(r(i, NodeField.observed), zero));
    final parents = net.valueParentsOf(i);
    final ratios = <Ex>[];
    for (final p in parents) {
      // δ/Δ = 1/K (TrialCompiler.gainNodes).
      final gain = _slotRead('gain${net.valueParentsOf(p).first}');
      ratios.add(one / sel(one, gain, on));
    }
    final nu = let('nu', sum(ratios) / lit(parents.length.toDouble()));
    _slotWrite('nu$i', sel(_slotRead('nu$i'), nu, on));
    _efMoveXis(i, on);
  }

  // ---------------------------------------------------------------------------
  // categorical-state
  // ---------------------------------------------------------------------------

  /// Each category's indicator, and the node's observed flag, to that
  /// category's binary node.
  void _predictionErrorCategorical(int i) {
    final x = _observation[i]!;
    final ob = let('ob', r(i, NodeField.observed));
    final parents = net.valueParentsOf(i);
    for (var k = 0; k < parents.length; k++) {
      w(parents[k], NodeField.mean, x[k]);
      w(parents[k], NodeField.observed, ob);
    }
  }

  /// The binary surprise of category [k]'s indicator under its binary
  /// node [b]'s prediction.
  Ex _categorySurprise(int b) => binarySurprise(
    ne(r(b, NodeField.mean), zero),
    r(b, NodeField.expectedMean),
    oneMinusMean(b),
  );

  /// After the binary HGFs' updates. The node's mean is the first
  /// category's new probability σ(μ_p) (`μ_p` the binary node's value
  /// parent's mean), which a readout may read.
  ///
  /// With [TrialCompiler.vectorState], per category: `α = ν μ̂_b + 1`,
  /// `ν = δ/Δ − 1`, with the binary node's prediction error
  /// `δ = x − μ̂_b` and the change of its expectation
  /// `Δ = σ(μ_p) − μ̂_b`; then the KL divergence from the previous α, the
  /// summed binary surprise, and the probabilities σ(μ_p).
  ///
  /// Δ is `σ(μ_p) (1 − μ̂_b) (1 − e^{−ι})`, with `μ̂_b = σ(μ̂_p)` and
  /// `ι = μ_p − μ̂_p = K δ / π̂_b` from the gain K of `p`'s update
  /// ([TrialCompiler.gainNodes]): subtracting means cancels in f32 where
  /// `p` moves little. Where μ̂_b was clipped (the identity fails) it is
  /// the difference.
  ///
  /// Unobserved, the CPU engines without [NetworkFixes.categoricalMissing]
  /// get Δ = 0 and α = ±∞ (P12); the kernel keeps α and records a KL
  /// divergence of 0 either way (WGSL has no infinities), and the surprise
  /// follows the fix.
  void _posteriorCategorical(int i) {
    final parents = net.valueParentsOf(i);
    Ex probability(int b) =>
        sigm(r(net.valueParentsOf(b).single, NodeField.mean));
    if (c.vectorState) {
      final on = letCond('on', ne(r(i, NodeField.observed), zero));
      final v = lit(net.precisionClippingValue);
      final cur = <Ex>[], next = <Ex>[], up = <Ex>[], terms = <Ex>[];
      for (var k = 0; k < parents.length; k++) {
        final b = parents[k], p = net.valueParentsOf(b).single;
        final em = let('emb$k', r(b, NodeField.expectedMean));
        final q = let('qb$k', oneMinusMean(b));
        final u = r(b, NodeField.mean);
        // x − μ̂ as x (1 − μ̂) − (1 − x) μ̂, as the binary node's.
        final delta = let('dl$k', u * q - (one - u) * em);
        up.add(let('up$k', probability(b)));
        final iota = let(
          'io$k',
          _slotRead('gain$p') * r(b, NodeField.valuePredictionError),
        );
        final clipped = letCond('cl$k', le(em, v) | ge(em, one - v));
        final change = let(
          'ch$k',
          sel(up[k] * q * -expm1(-iota), up[k] - em, clipped),
        );
        final ratio = let('ra$k', delta / sel(one, change, on));
        next.add(let('an$k', (ratio - one) * em + one));
        cur.add(_slotRead('al${i}_$k'));
        terms.add(_categorySurprise(b));
      }
      final kl = let('kl', _dirichletKl(cur, next));
      for (var k = 0; k < parents.length; k++) {
        _slotWrite('al${i}_$k', sel(cur[k], next[k], on));
        _slotWrite('cx${i}_$k', up[k]);
      }
      _slotWrite('kl$i', sel(zero, kl, on));
      final s = sum(terms);
      _slotWrite('su$i', net.fixes.categoricalMissing ? sel(zero, s, on) : s);
    }
    w(i, NodeField.mean, probability(parents.first));
  }

  /// `KL[Dir(a) ‖ Dir(b)]`, as `dirichletKullbackLeibler` sums it (for
  /// positive concentrations).
  Ex _dirichletKl(List<Ex> a, List<Ex> b) {
    final sa = let('klsa', sum(a)), sb = let('klsb', sum(b));
    final psi = let('klpsi', digamma(sa));
    final gammas = sum([
      for (var k = 0; k < a.length; k++) lgamma(b[k]) - lgamma(a[k]),
    ]);
    final digammas = sum([
      for (var k = 0; k < a.length; k++) (a[k] - b[k]) * (digamma(a[k]) - psi),
    ]);
    return (lgamma(sa) - lgamma(sb)) + gammas + digammas;
  }

  // ---------------------------------------------------------------------------
  // Response
  // ---------------------------------------------------------------------------

  static Ex gaussianSurprise(Ex x, Ex mu, Ex pi) {
    final d = x - mu;
    return Lit('0.5f') *
        (Lit('1.83787707f', 1.83787707, 1.8378770664093453) -
            log(pi) +
            pi * d * d);
  }

  /// `binarySurprise` with the probability [p] of a 1 and [q] = 1 − p
  /// given separately (each clipped to [1e-6, 1 − 1e-6]).
  static Ex binarySurprise(Cond isOne, Ex p, Ex q) {
    const lo = Lit('1e-6f'), hi = Lit('0.999999f');
    return sel(-log(clamp(q, lo, hi)), -log(clamp(p, lo, hi)), isOne);
  }

  void _response() {
    line('// response');
    open('');
    final Ex s;
    switch (c.response) {
      case _ObservationResponse(:final which) when which == 'total':
        s = sum([
          for (var i = 0; i < n; i++)
            if (kind(i) == NodeKind.continuousState)
              gaussianSurprise(
                r(i, NodeField.mean),
                r(i, NodeField.expectedMean),
                r(i, NodeField.expectedPrecision),
              ),
        ]);
      case _ObservationResponse(:final which) when which == 'first':
        s = gaussianSurprise(
          r(0, NodeField.mean),
          r(1, NodeField.expectedMean),
          r(1, NodeField.expectedPrecision),
        );
      case _ObservationResponse(:final which) when which == 'categorical':
        // As the categorical posterior sums the node's surprise.
        s = sum([
          for (final i in c.categoricalNodes)
            () {
              final terms = sum([
                for (final b in net.valueParentsOf(i)) _categorySurprise(b),
              ]);
              return net.fixes.categoricalMissing
                  ? sel(zero, terms, ne(r(i, NodeField.observed), zero))
                  : terms;
            }(),
        ]);
      case _ObservationResponse():
        s = sum([
          for (final i in net.inputIdxs)
            binarySurprise(
              ne(r(i, NodeField.mean), zero),
              r(i, NodeField.expectedMean),
              oneMinusMean(i),
            ),
        ]);
      case GpuGaussianReadout(:final node, :final field):
        final present = letCond('present', Cond(source.responsePresent()));
        final resp = let('resp', Lit(source.response()));
        s = sel(zero, gaussianSurprise(resp, r(node, field), rp(0)), present);
      case GpuBinarySoftmax(:final node):
        final present = letCond('present', Cond(source.responsePresent()));
        final resp = let('resp', Lit(source.response()));
        final m = r(node, NodeField.expectedMean);
        final z = let('z', rp(0) * log(m / oneMinusMean(node)));
        final p = let('prob', one / (one + exp(-z)));
        final q = let('qprob', one / (one + exp(z)));
        s = sel(zero, binarySurprise(ne(resp, zero), p, q), present);
      case GpuCategoricalSoftmax(:final node):
        // log Σ μ̂ₖ^β − β log μ̂_resp; a response that is no category's index
        // picks none (the CPU engine rejects it).
        final present = letCond('present', Cond(source.responsePresent()));
        final resp = let('resp', Lit(source.response()));
        final parents = net.valueParentsOf(node);
        final z = [
          for (var k = 0; k < parents.length; k++)
            let('z$k', rp(0) * log(r(parents[k], NodeField.expectedMean))),
        ];
        var lse = z.first;
        for (var k = 1; k < z.length; k++) {
          lse = let('lse$k', logAddExp(lse, z[k]));
        }
        final chosen = sum([
          for (var k = 0; k < z.length; k++)
            sel(zero, z[k], eq(resp, lit(k.toDouble()))),
        ]);
        s = sel(zero, lse - chosen, present);
      case GpuSoftmax(
        :final options,
        :final field,
        :final stickiness,
        :final lapse,
        :final bias,
      ):
        // log Σ exp(uₖ) − u_resp, uₖ = β vₖ + bₖ + κ [k = previous]; a
        // response that is no option's index picks none (the CPU engine
        // rejects it). With a lapse ε: −log((1 − ε) q + ε / K).
        final present = letCond('present', Cond(source.responsePresent()));
        final resp = let('resp', Lit(source.response()));
        var j = 1;
        final kappa = stickiness == null ? null : rp(j++);
        final eps = lapse == null ? null : rp(j++);
        final b0 = j;
        final prev = kappa == null
            ? null
            : let('prev', Lit(source.previousResponse()));
        final z = [
          for (var k = 0; k < options.length; k++)
            let(
              'z$k',
              sum([
                if (options[k].isNotEmpty)
                  rp(0) *
                      sum([
                        for (final MapEntry(:key, :value) in options[k].entries)
                          value == 1.0
                              ? r(key, field)
                              : lit(value) * r(key, field),
                      ]),
                if (bias != null) rp(b0 + k),
                if (kappa != null)
                  sel(zero, kappa, eq(prev!, lit(k.toDouble()))),
              ]),
            ),
        ];
        var lse = z.first;
        for (var k = 1; k < z.length; k++) {
          lse = let('lse$k', logAddExp(lse, z[k]));
        }
        final chosen = sum([
          for (var k = 0; k < z.length; k++)
            sel(zero, z[k], eq(resp, lit(k.toDouble()))),
        ]);
        final Ex nll;
        if (eps == null) {
          nll = lse - chosen;
        } else {
          final logq = let('logq', chosen - lse);
          nll = -logAddExp(
            log(one - eps) + logq,
            log(eps) - lit(math.log(options.length)),
          );
        }
        s = sel(zero, nll, present);
      case GpuExprReadout(
        :final kind,
        :final expressions,
        :final beliefs,
        :final parameters,
      ):
        final present = letCond('present', Cond(source.responsePresent()));
        final resp = let('resp', Lit(source.response()));
        final params = parameters.keys.toList();
        Ex? prev;
        Ex variable(String name) {
          if (beliefs[name] case final b?) return r(b.node, b.field);
          final j = params.indexOf(name);
          if (j >= 0) return rp(j);
          if (name == 'response') return resp;
          if (name == 'previous') {
            return prev ??= let('prev', Lit(source.previousResponse()));
          }
          throw StateError('Unbound variable $name');
        }

        final values = [
          for (var k = 0; k < expressions.length; k++)
            let('ev$k', lowerExpr(expressions[k], variable)),
        ];
        switch (kind) {
          case 'choice':
            var lse = values.first;
            for (var k = 1; k < values.length; k++) {
              lse = let('lse$k', logAddExp(lse, values[k]));
            }
            final chosen = sum([
              for (var k = 0; k < values.length; k++)
                sel(zero, values[k], eq(resp, lit(k.toDouble()))),
            ]);
            s = sel(zero, lse - chosen, present);
          case 'gaussian':
            s = sel(
              zero,
              gaussianSurprise(resp, values[0], values[1]),
              present,
            );
          default:
            s = sel(zero, values[0], present);
        }
    }
    final sv = let('s', s);
    if (ssa) {
      objective = (sv as Ref).name;
      close();
      return;
    }
    flag(isFinite(sv));
    // Kahan summation (the tangent is summed plainly).
    final ds = _d(sv);
    if (ds != null) _tangentLines('', 'total', '+=', ds);
    line('let yk = s - comp;');
    line('let tk = total + yk;');
    line('comp = (tk - total) - yk;');
    line('total = tk;');
    close();
  }
}

/// Functions the generated kernels call.
const String prelude = r'''
fn isFinite(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}
fn sigm(x: f32) -> f32 { return 1.0f / (1.0f + exp(-x)); }
fn dsigm(x: f32) -> f32 { let s = sigm(x); return s * (1.0f - s); }
fn d2sigm(x: f32) -> f32 {
  let s = sigm(x);
  return s * (1.0f - s) * (1.0f - 2.0f * s);
}
fn d3sigm(x: f32) -> f32 {
  let s = sigm(x);
  return s * (1.0f - s) * (1.0f - 6.0f * s + 6.0f * s * s);
}
fn dtanh(x: f32) -> f32 { let t = tanh(x); return 1.0f - t * t; }
fn d2tanh(x: f32) -> f32 { let t = tanh(x); return -2.0f * t * (1.0f - t * t); }
fn d3tanh(x: f32) -> f32 {
  let t = tanh(x);
  return -2.0f * (1.0f - t * t) * (1.0f - 3.0f * t * t);
}
fn erfcApprox(x: f32) -> f32 {
  let t = 1.0f / (1.0f + 0.3275911f * abs(x));
  let poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f +
      t * (-1.453152027f + t * 1.061405429f))));
  let approx = poly * exp(-(x * x));
  return select(2.0f - approx, approx, x >= 0.0f);
}
fn gelu(x: f32) -> f32 { return x * 0.5f * erfcApprox(-x / 1.41421356f); }
fn dgelu(x: f32) -> f32 {
  let phi = 0.5f * erfcApprox(-x / 1.41421356f);
  let pdf = exp(-(x * x) / 2.0f) / 2.50662827f;
  return phi + x * pdf;
}
fn d2gelu(x: f32) -> f32 {
  let pdf = exp(-(x * x) / 2.0f) / 2.50662827f;
  return pdf * (2.0f - x * x);
}
fn d3gelu(x: f32) -> f32 {
  let pdf = exp(-(x * x) / 2.0f) / 2.50662827f;
  return pdf * (x * x * x - 4.0f * x);
}
fn logAddExp(a: f32, b: f32) -> f32 {
  let m = max(a, b);
  return m + log(exp(a - m) + exp(b - m));
}
fn expm1(x: f32) -> f32 {
  // eˣ − 1 without cancellation: Taylor to x¹¹/11! for |x| < 1 (error
  // below 4e-8 relative), else exp(x) − 1, which loses at most a bit there.
  if (abs(x) < 1.0f) {
    var p = 2.5052108e-8f;
    p = 2.7557319e-7f + x * p;
    p = 2.7557319e-6f + x * p;
    p = 2.4801587e-5f + x * p;
    p = 1.9841270e-4f + x * p;
    p = 1.3888889e-3f + x * p;
    p = 8.3333333e-3f + x * p;
    p = 4.1666667e-2f + x * p;
    p = 1.6666667e-1f + x * p;
    p = 0.5f + x * p;
    return x * (1.0f + x * p);
  }
  return exp(x) - 1.0f;
}
fn lgamma(x0: f32) -> f32 {
  // ln Γ(x) for x > 0: Γ(x) = Γ(x + 1) / x up to x ≥ 8, then Stirling's
  // series.
  var x = x0;
  var prod = 1.0f;
  for (var i = 0; i < 8 && x < 8.0f; i++) {
    prod *= x;
    x += 1.0f;
  }
  let inv = 1.0f / x;
  let inv2 = inv * inv;
  let series = inv * (0.083333333f - inv2 * (0.0027777778f -
      inv2 * (0.00079365079f - inv2 * 0.00059523810f)));
  return (x - 0.5f) * log(x) - x + 0.91893853f + series - log(prod);
}
fn digamma(x0: f32) -> f32 {
  // ψ(x) for x > 0: ψ(x) = ψ(x + 1) − 1/x up to x ≥ 8, then the asymptotic
  // series.
  var x = x0;
  var acc = 0.0f;
  for (var i = 0; i < 8 && x < 8.0f; i++) {
    acc -= 1.0f / x;
    x += 1.0f;
  }
  let inv2 = 1.0f / (x * x);
  return acc + log(x) - 0.5f / x - inv2 * (0.083333333f - inv2 *
      (0.0083333333f - inv2 * (0.0039682540f - inv2 * 0.0041666667f)));
}
fn lambertW0(z: f32) -> f32 {
  // Halley on (w eʷ − z) / eʷ, as pyhgf #475: no overflow for finite z.
  var w = log(z + 1.0f);
  for (var i = 0; i < 6; i++) {
    let f = w - z * exp(-w);
    let w1 = w + 1.0f;
    let st = (2.0f * f * w1) / (2.0f * w1 * w1 - f * (w + 2.0f));
    w -= st;
    // 1e-12 in f64.
    if (abs(st) <= 1e-6f * abs(w)) { break; }
  }
  return w;
}
''';
