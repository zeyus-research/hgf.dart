import 'dart:convert';
import 'dart:math' as math;

import 'package:hgf/hgf.dart';

/// One builder call of a [NetworkPlan], in the terms of `package:hgf`'s
/// API (and pyhgf's, which takes the same calls).
sealed class BuildCall {
  const BuildCall();

  /// Applies this call to [network], with [couplings] for custom couplings.
  void apply(Network network, Map<String, Coupling> couplings);
}

/// An edge declared by a new node, to an earlier node, with its strength.
typedef PlanEdge = ({int node, double strength});

/// `Network.addNodes`: [nNodes] continuous, binary or constant nodes, the
/// first numbered [first].
final class AddNodes extends BuildCall {
  AddNodes({
    required this.first,
    required this.kind,
    this.nNodes = 1,
    this.valueParents = const [],
    this.valueChildren = const [],
    this.volatilityParents = const [],
    this.volatilityChildren = const [],
    this.coupling = CouplingFn.linear,
    this.customCoupling,
    this.parameters = const {},
  });

  final int first;
  final NodeKind kind;
  final int nNodes;
  final List<PlanEdge> valueParents,
      valueChildren,
      volatilityParents,
      volatilityChildren;

  /// The value coupling to the new nodes' value children, unless
  /// [customCoupling] names a custom one.
  final CouplingFn coupling;
  final String? customCoupling;

  /// Initial values that differ from the kind's defaults.
  final Map<NodeField, double> parameters;

  AddNodes _withCount(int n) => AddNodes(
    first: first,
    kind: kind,
    nNodes: n,
    valueParents: valueParents,
    valueChildren: valueChildren,
    volatilityParents: volatilityParents,
    volatilityChildren: volatilityChildren,
    coupling: coupling,
    customCoupling: customCoupling,
    parameters: parameters,
  );

  /// Whether a node declared like [other] can join this call.
  bool _sameDeclaration(AddNodes other) {
    bool edges(List<PlanEdge> a, List<PlanEdge> b) =>
        a.length == b.length &&
        [
          for (var i = 0; i < a.length; i++)
            a[i].node == b[i].node && _same(a[i].strength, b[i].strength),
        ].every((x) => x);
    return kind == other.kind &&
        coupling == other.coupling &&
        customCoupling == other.customCoupling &&
        edges(valueParents, other.valueParents) &&
        edges(valueChildren, other.valueChildren) &&
        edges(volatilityParents, other.volatilityParents) &&
        edges(volatilityChildren, other.volatilityChildren) &&
        parameters.length == other.parameters.length &&
        parameters.entries.every(
          (e) =>
              other.parameters.containsKey(e.key) &&
              _same(e.value, other.parameters[e.key]!),
        );
  }

  @override
  void apply(Network network, Map<String, Coupling> couplings) {
    List<int>? idx(List<PlanEdge> e) =>
        e.isEmpty ? null : [for (final x in e) x.node];
    List<double>? s(List<PlanEdge> e) =>
        e.isEmpty ? null : [for (final x in e) x.strength];
    network.addNodes(
      kind: kind,
      nNodes: nNodes,
      valueParents: idx(valueParents),
      valueParentsStrengths: s(valueParents),
      valueChildren: idx(valueChildren),
      valueChildrenStrengths: s(valueChildren),
      volatilityParents: idx(volatilityParents),
      volatilityParentsStrengths: s(volatilityParents),
      volatilityChildren: idx(volatilityChildren),
      volatilityChildrenStrengths: s(volatilityChildren),
      coupling: customCoupling == null ? coupling : couplings[customCoupling]!,
      parameters: parameters.isEmpty ? null : parameters,
    );
  }
}

/// `Network.addEfState`, the first node numbered [first]. With HGF
/// [learning], the call also adds [impliedNodes] continuous nodes after
/// the ef-state nodes.
final class AddEfState extends BuildCall {
  AddEfState({
    required this.first,
    this.nNodes = 1,
    this.dimension = 1,
    this.learning = EfLearning.fixed,
    this.nus = 3.0,
    this.xis,
    this.valueChildren,
  });

  final int first, nNodes, dimension;
  final EfLearning learning;
  final double nus;

  /// The initial sufficient statistics, when they aren't N(0, I)'s.
  final List<double>? xis;

  /// The dp-state node(s) these nodes are clusters of.
  final List<int>? valueChildren;

  int get impliedNodes => learning.levels == 0
      ? 0
      : nNodes * efStatisticsCount(dimension) * (learning.levels + 1);

  @override
  void apply(Network network, Map<String, Coupling> couplings) =>
      network.addEfState(
        nNodes: nNodes,
        dimension: dimension,
        learning: learning,
        nus: nus,
        xis: xis,
        valueChildren: valueChildren,
      );
}

/// `Network.addCategoricalState`, the categorical node numbered [node],
/// followed by its `2 n + 1` implied nodes. Null parameters keep their
/// defaults.
final class AddCategoricalState extends BuildCall {
  AddCategoricalState({
    required this.node,
    required this.nCategories,
    this.binaryMean,
    this.binaryPrecision,
    this.valueMean,
    this.valuePrecision,
    this.valueTonicVolatility,
    this.volatilityMean,
    this.volatilityPrecision,
    this.volatilityTonicVolatility,
  });

  final int node, nCategories;
  final double? binaryMean,
      binaryPrecision,
      valueMean,
      valuePrecision,
      valueTonicVolatility,
      volatilityMean,
      volatilityPrecision,
      volatilityTonicVolatility;

  int get impliedNodes => 2 * nCategories + 1;

  /// The non-default arguments by their Dart name (pyhgf's
  /// `binary_parameters` keys in [pyhgfKeys]).
  Map<String, double> get arguments => {
    'binaryMean': ?binaryMean,
    'binaryPrecision': ?binaryPrecision,
    'valueMean': ?valueMean,
    'valuePrecision': ?valuePrecision,
    'valueTonicVolatility': ?valueTonicVolatility,
    'volatilityMean': ?volatilityMean,
    'volatilityPrecision': ?volatilityPrecision,
    'volatilityTonicVolatility': ?volatilityTonicVolatility,
  };

  /// pyhgf's `binary_parameters` key for each of [arguments]'s names.
  static const pyhgfKeys = {
    'binaryMean': 'mean_1',
    'binaryPrecision': 'precision_1',
    'valueMean': 'mean_2',
    'valuePrecision': 'precision_2',
    'valueTonicVolatility': 'tonic_volatility_2',
    'volatilityMean': 'mean_3',
    'volatilityPrecision': 'precision_3',
    'volatilityTonicVolatility': 'tonic_volatility_3',
  };

  @override
  void apply(Network network, Map<String, Coupling> couplings) =>
      network.addCategoricalState(
        nCategories: nCategories,
        binaryMean: binaryMean,
        binaryPrecision: binaryPrecision ?? 1.0,
        valueMean: valueMean,
        valuePrecision: valuePrecision ?? 1.0,
        valueTonicVolatility: valueTonicVolatility ?? -4.0,
        volatilityMean: volatilityMean ?? 0.0,
        volatilityPrecision: volatilityPrecision ?? 1.0,
        volatilityTonicVolatility: volatilityTonicVolatility ?? -4.0,
      );
}

/// `Network.addDpState`, numbered [node].
final class AddDpState extends BuildCall {
  AddDpState({
    required this.node,
    this.batchSize = 10,
    this.alpha = 1.0,
    this.sensoryPrecision = 1.0,
  });

  final int node, batchSize;
  final double alpha, sensoryPrecision;

  @override
  void apply(Network network, Map<String, Coupling> couplings) =>
      network.addDpState(
        batchSize: batchSize,
        alpha: alpha,
        sensoryPrecision: sensoryPrecision,
      );
}

/// `Network.setInitial`: a value the builder calls don't set.
final class SetInitial extends BuildCall {
  SetInitial(this.node, this.field, this.value);

  final int node;
  final NodeField field;
  final double value;

  @override
  void apply(Network network, Map<String, Coupling> couplings) =>
      network.setInitial(node, field, value);
}

/// `Network.setValueCoupling` / `setVolatilityCoupling`: a strength the
/// builder calls don't set (on an implied node's edge).
final class SetCoupling extends BuildCall {
  SetCoupling(
    this.parent,
    this.child,
    this.strength, {
    required this.volatility,
  });

  final int parent, child;
  final double strength;
  final bool volatility;

  @override
  void apply(Network network, Map<String, Coupling> couplings) => volatility
      ? network.setVolatilityCoupling(parent, child, strength)
      : network.setValueCoupling(parent, child, strength);
}

/// `Network.setInitialVector`.
final class SetInitialVector extends BuildCall {
  SetInitialVector(this.node, this.field, this.values);

  final int node;
  final VectorField field;
  final List<double> values;

  @override
  void apply(Network network, Map<String, Coupling> couplings) =>
      network.setInitialVector(node, field, values);
}

/// A network as the builder calls that make it: the input of the code
/// generators.
///
/// [NetworkPlan.fromJson] reads a `Network.toJson()` snapshot (after
/// expansion: a categorical node's implied nodes, an ef-state node's
/// learning chains, are nodes of their own) and recovers the calls:
/// `addNodes` for continuous, binary and constant nodes (each declares its
/// edges to earlier nodes, as they were built; consecutive nodes declared
/// alike are one call), `addEfState`, `addCategoricalState` and
/// `addDpState` for the vector kinds (their implied nodes folded back in),
/// then `setInitial` / `setInitialVector` for whatever those calls don't
/// set (values changed after building, e.g. by the playground's sliders).
/// Parameters equal to the defaults are left out.
///
/// The plan is checked: replayed on a fresh `Network`, it gives the same
/// snapshot (structure, edge order, strengths, settings and every initial
/// value, compared bit for bit), so the generated code filters exactly
/// like the original. A network the calls can't reproduce throws an
/// [UnsupportedError].
final class NetworkPlan {
  NetworkPlan._({
    required this.volatilityUpdates,
    required this.meanFieldUpdates,
    required this.ehgfMgfCorrection,
    required this.maxPosteriorPrecision,
    required this.precisionClippingValue,
    required this.calls,
    required this.customCouplings,
    required this.nNodes,
    required this.inputs,
    required this.inputDimensions,
    required this.kinds,
  });

  ///
  /// [categoricalArguments] passes the implied nodes' non-default values
  /// to `addCategoricalState` and [efXis] the initial ξ to `addEfState`;
  /// without them (for pyhgf, which ignores both), they become
  /// `setInitial` / `setInitialVector` calls.
  factory NetworkPlan.fromJson(
    Map<String, Object?> json, {
    bool categoricalArguments = true,
    bool efXis = true,
  }) => _plan(json, categoricalArguments: categoricalArguments, efXis: efXis);

  final VolatilityUpdates volatilityUpdates;
  final bool meanFieldUpdates, ehgfMgfCorrection;
  final double maxPosteriorPrecision, precisionClippingValue;
  final List<BuildCall> calls;

  /// The names of the custom couplings the network uses, which code can't
  /// be generated for.
  final List<String> customCouplings;

  final int nNodes;

  /// The input nodes, and how many data columns each one takes.
  final List<int> inputs, inputDimensions;

  /// Every node's kind.
  final List<NodeKind> kinds;

  /// The width of one row of input data.
  int get nInputColumns => inputDimensions.fold(0, (a, b) => a + b);

  /// Placeholders for the custom couplings (never evaluated): enough to
  /// build and compare the network.
  static Map<String, Coupling> placeholderCouplings(Iterable<String> names) => {
    for (final name in names)
      name: CustomCoupling(
        name,
        g: _placeholder,
        dg: _placeholder,
        d2g: _placeholder,
      ),
  };

  static double _placeholder(double x) =>
      throw StateError('Placeholder coupling');

  /// Replays the calls on a new network, with [couplings] for the custom
  /// ones (default: placeholders that can build but not run).
  Network build({Map<String, Coupling>? couplings}) {
    final c = couplings ?? placeholderCouplings(customCouplings);
    final network = Network(
      volatilityUpdates: volatilityUpdates,
      meanFieldUpdates: meanFieldUpdates,
      ehgfMgfCorrection: ehgfMgfCorrection,
      maxPosteriorPrecision: maxPosteriorPrecision,
      precisionClippingValue: precisionClippingValue,
    );
    for (final call in calls) {
      call.apply(network, c);
    }
    return network..build();
  }
}

/// The ef-state sufficient-statistics count of a `d`-dimensional normal.
int efStatisticsCount(int d) => d + d * (d + 1) ~/ 2;

/// `addNodes`' defaults (`Network._defaults`); a different value is passed
/// as a parameter. (If these drift from the engine's, the plan stays
/// correct: the difference becomes a `setInitial`.)
Map<NodeField, double> _defaults(NodeKind kind) => switch (kind) {
  NodeKind.continuousState => const {
    NodeField.precision: 1.0,
    NodeField.expectedPrecision: 1.0,
    NodeField.conditionalExpectedPrecision: 1.0,
    NodeField.observed: 1.0,
    NodeField.tonicVolatility: -4.0,
    NodeField.autoconnectionStrength: 1.0,
    NodeField.currentVariance: 1.0,
  },
  NodeKind.binaryState => const {
    NodeField.expectedMean: 0.5,
    NodeField.precision: 1.0,
    NodeField.expectedPrecision: 1.0,
    NodeField.conditionalExpectedPrecision: 1.0,
    NodeField.observed: 1.0,
    NodeField.currentVariance: 1.0,
  },
  _ => const {},
};

bool _same(double a, double b) =>
    a == b ? (a != 0 || (1 / a) == (1 / b)) : (a.isNaN && b.isNaN);

bool _sameList(List<double> a, List<double> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (!_same(a[i], b[i])) return false;
  }
  return true;
}

/// The snapshot without what only the builder calls see (the declared
/// parameters and xis, which the compiled initial values already carry).
String _comparable(Map<String, Object?> json) => jsonEncode({
  ...json,
  'nodes': [
    for (final n in json['nodes'] as List<Object?>)
      {
        for (final MapEntry(:key, :value)
            in (n as Map<String, Object?>).entries)
          if (key != 'params' && key != 'xis') key: value,
      },
  ],
});

NetworkPlan _plan(
  Map<String, Object?> json, {
  required bool categoricalArguments,
  required bool efXis,
}) {
  final nodes = [
    for (final n in json['nodes'] as List<Object?>) n as Map<String, Object?>,
  ];
  final customs = <String>{
    for (final n in nodes)
      if (n['custom_coupling'] == true) n['coupling_fn'] as String,
  };
  final placeholders = NetworkPlan.placeholderCouplings(customs);
  // The source, compiled: its initial values, inputs and canonical JSON.
  final source = Network.fromJson(json, couplings: placeholders);
  final n = source.nNodes;
  List<PlanEdge> edges(Map<String, Object?> node, String key) {
    final e = node[key] as Map<String, Object?>;
    final idxs = e['idxs'] as List<Object?>;
    final s = e['strengths'] as List<Object?>;
    return [
      for (var k = 0; k < idxs.length; k++)
        (node: (idxs[k] as num).toInt(), strength: _decode(s[k])),
    ];
  }

  NodeKind kind(int i) => source.kindOf(i);
  bool isLeaf(int i) =>
      source.valueChildrenOf(i).isEmpty &&
      source.volatilityChildrenOf(i).isEmpty;
  double initial(int i, NodeField f) => source.getInitial(i, f);

  final calls = <BuildCall>[];
  var i = 0;
  while (i < n) {
    final node = nodes[i];
    switch (kind(i)) {
      case NodeKind.categoricalState:
        final k = (node['dimension'] as num?)?.toInt() ?? 1;
        double? arg(int at, NodeField f, double byDefault) {
          final v = initial(at, f);
          return !categoricalArguments || _same(v, byDefault) ? null : v;
        }

        final c = AddCategoricalState(
          node: i,
          nCategories: k,
          binaryMean: arg(i + 1, NodeField.mean, 1.0 / k),
          binaryPrecision: arg(i + 1, NodeField.precision, 1.0),
          valueMean: arg(i + k + 1, NodeField.mean, -math.log(k - 1)),
          valuePrecision: arg(i + k + 1, NodeField.precision, 1.0),
          valueTonicVolatility: arg(i + k + 1, NodeField.tonicVolatility, -4.0),
          volatilityMean: arg(i + 2 * k + 1, NodeField.mean, 0.0),
          volatilityPrecision: arg(i + 2 * k + 1, NodeField.precision, 1.0),
          volatilityTonicVolatility: arg(
            i + 2 * k + 1,
            NodeField.tonicVolatility,
            -4.0,
          ),
        );
        calls.add(c);
        i += 1 + c.impliedNodes;
      case NodeKind.dpState:
        calls.add(
          AddDpState(
            node: i,
            batchSize: (node['batch_size'] as num?)?.toInt() ?? 0,
            alpha: initial(i, NodeField.alpha),
            sensoryPrecision: initial(i, NodeField.sensoryPrecision),
          ),
        );
        i++;
      case NodeKind.efState:
        i = _efCalls(i, nodes, source, calls, xisArgument: efXis);
      case NodeKind.continuousState ||
          NodeKind.binaryState ||
          NodeKind.constantState:
        final k = kind(i);
        List<PlanEdge> earlier(String key) => [
          for (final e in edges(node, key))
            if (e.node < i) e,
        ];
        final params = <NodeField, double>{};
        if (k != NodeKind.constantState) {
          final defaults = _defaults(k);
          final leaf = k == NodeKind.continuousState && isLeaf(i);
          for (final f in NodeField.values) {
            if (leaf &&
                (f == NodeField.tonicVolatility ||
                    f == NodeField.autoconnectionStrength)) {
              continue; // forced to 0 on inputs: a setInitial if not
            }
            final v = initial(i, f);
            if (!_same(v, defaults[f] ?? 0.0)) params[f] = v;
          }
        }
        final custom = node['custom_coupling'] == true;
        final call = AddNodes(
          first: i,
          kind: k,
          valueParents: earlier('value_parents'),
          valueChildren: earlier('value_children'),
          volatilityParents: earlier('volatility_parents'),
          volatilityChildren: earlier('volatility_children'),
          coupling: custom
              ? CouplingFn.linear
              : CouplingFn.parse(node['coupling_fn'] as String),
          customCoupling: custom ? node['coupling_fn'] as String : null,
          parameters: params,
        );
        // Joins the previous call if declared alike (and unconnected to
        // it: an edge to the previous node would differ anyway).
        if (calls.lastOrNull case final AddNodes prev
            when prev.first + prev.nNodes == i && prev._sameDeclaration(call)) {
          calls[calls.length - 1] = prev._withCount(prev.nNodes + 1);
        } else {
          calls.add(call);
        }
        i++;
    }
  }

  final settings = (
    volatilityUpdates: VolatilityUpdates.parse(
      json['volatility_updates'] as String,
    ),
    meanFieldUpdates: json['mean_field_updates'] as bool,
    ehgfMgfCorrection: json['ehgf_mgf_correction'] as bool,
    maxPosteriorPrecision: _decode(json['max_posterior_precision']),
    precisionClippingValue: _decode(json['precision_clipping_value']),
  );
  NetworkPlan make(List<BuildCall> calls) => NetworkPlan._(
    volatilityUpdates: settings.volatilityUpdates,
    meanFieldUpdates: settings.meanFieldUpdates,
    ehgfMgfCorrection: settings.ehgfMgfCorrection,
    maxPosteriorPrecision: settings.maxPosteriorPrecision,
    precisionClippingValue: settings.precisionClippingValue,
    calls: List.unmodifiable(calls),
    customCouplings: List.unmodifiable(customs),
    nNodes: n,
    inputs: List.unmodifiable(source.inputIdxs),
    inputDimensions: List.unmodifiable([
      for (final input in source.inputIdxs)
        switch (source.kindOf(input)) {
          NodeKind.efState || NodeKind.categoricalState =>
            (nodes[input]['dimension'] as num?)?.toInt() ?? 1,
          _ => 1,
        },
    ]),
    kinds: List.unmodifiable([for (var j = 0; j < n; j++) source.kindOf(j)]),
  );

  // Whatever the calls leave different becomes setInitial(Vector) or
  // set*Coupling.
  final built = make(calls).build(couplings: placeholders);
  if (built.nNodes != n) {
    throw UnsupportedError(
      'Cannot express this network as builder calls: they make '
      '${built.nNodes} nodes, not $n',
    );
  }
  final builtNodes = built.toJson()['nodes'] as List<Object?>;
  for (var j = 0; j < n; j++) {
    for (final volatility in [false, true]) {
      final key = volatility ? 'volatility_children' : 'value_children';
      final want = edges(nodes[j], key);
      final got = edges(builtNodes[j] as Map<String, Object?>, key);
      if (want.length != got.length) continue; // fails the check below
      for (var k = 0; k < want.length; k++) {
        if (want[k].node == got[k].node &&
            !_same(want[k].strength, got[k].strength)) {
          calls.add(
            SetCoupling(
              j,
              want[k].node,
              want[k].strength,
              volatility: volatility,
            ),
          );
        }
      }
    }
  }
  for (var j = 0; j < n; j++) {
    for (final f in NodeField.values) {
      final v = source.getInitial(j, f);
      if (!_same(built.getInitial(j, f), v)) calls.add(SetInitial(j, f, v));
    }
  }
  for (var j = 0; j < n; j++) {
    for (final f in VectorField.values) {
      final v = source.getInitialVector(j, f);
      if (v.isEmpty) continue;
      if (!_sameList(built.getInitialVector(j, f), v)) {
        calls.add(SetInitialVector(j, f, v));
      }
    }
  }
  final plan = make(calls);
  {
    final want = _comparable(source.toJson());
    final got = _comparable(plan.build(couplings: placeholders).toJson());
    if (want != got) {
      throw UnsupportedError(
        'Cannot express this network as builder calls: the replayed network '
        'differs from the snapshot at '
        '${_firstDifference(jsonDecode(want), jsonDecode(got), r'$')}',
      );
    }
  }
  return plan;
}

/// Where [a] and [b] (decoded JSON) first differ, as a path.
String _firstDifference(Object? a, Object? b, String path) {
  if (a is Map && b is Map) {
    for (final k in {...a.keys, ...b.keys}) {
      if (jsonEncode(a[k]) != jsonEncode(b[k])) {
        return _firstDifference(a[k], b[k], '$path.$k');
      }
    }
  } else if (a is List && b is List) {
    if (a.length != b.length) {
      return '$path (${a.length} vs ${b.length} entries)';
    }
    for (var i = 0; i < a.length; i++) {
      if (jsonEncode(a[i]) != jsonEncode(b[i])) {
        return _firstDifference(a[i], b[i], '$path[$i]');
      }
    }
  }
  return '$path ($a vs $b)';
}

/// The calls for the ef-state nodes from [start] on; returns the next
/// node to read.
///
/// Consecutive ef-state nodes (not clusters) with the same dimension can
/// come from one call. With HGF learning, each node's chains (one per
/// sufficient statistic, `levels + 1` nodes each) follow all of the call's
/// nodes, in reverse node order. The nodes with chains are therefore the
/// last call of a run, the ones before it calls with a fixed rate.
int _efCalls(
  int start,
  List<Map<String, Object?>> nodes,
  Network source,
  List<BuildCall> calls, {
  required bool xisArgument,
}) {
  int dim(int i) => (nodes[i]['dimension'] as num?)?.toInt() ?? 1;
  List<int> children(int i) => source.valueChildrenOf(i);
  final n = source.nNodes;
  final d = dim(start);
  final defaultXis = [
    for (var r = 0; r < d; r++) 0.0,
    for (var r = 0; r < d; r++)
      for (var c = 0; c <= r; c++) r == c ? 1.0 : 0.0,
  ];
  List<double>? xis(int i) {
    final v = source.getInitialVector(i, VectorField.xis);
    return !xisArgument || _sameList(v, defaultXis) ? null : v;
  }

  // Clusters of a dp-state node: one call per run with the same children.
  final clusterOf = children(start);
  if (clusterOf.isNotEmpty) {
    var end = start + 1;
    while (end < n &&
        source.kindOf(end) == NodeKind.efState &&
        _sameInts(children(end), clusterOf) &&
        _same(
          source.getInitial(end, NodeField.nus),
          source.getInitial(start, NodeField.nus),
        ) &&
        _sameNullable(xis(end), xis(start))) {
      end++;
    }
    calls.add(
      AddEfState(
        first: start,
        nNodes: end - start,
        nus: source.getInitial(start, NodeField.nus),
        xis: xis(start),
        valueChildren: List.of(clusterOf),
      ),
    );
    return end;
  }

  var end = start + 1;
  while (end < n &&
      source.kindOf(end) == NodeKind.efState &&
      children(end).isEmpty &&
      dim(end) == d) {
    end++;
  }
  // The run's nodes with learning chains: a suffix of it.
  var learnFrom = end;
  while (learnFrom > start && source.valueParentsOf(learnFrom - 1).isNotEmpty) {
    learnFrom--;
  }
  // Fixed-rate nodes, merged while declared alike.
  var j = start;
  while (j < learnFrom) {
    var k = j + 1;
    while (k < learnFrom &&
        _same(
          source.getInitial(k, NodeField.nus),
          source.getInitial(j, NodeField.nus),
        ) &&
        _sameNullable(xis(k), xis(j))) {
      k++;
    }
    calls.add(
      AddEfState(
        first: j,
        nNodes: k - j,
        dimension: d,
        nus: source.getInitial(j, NodeField.nus),
        xis: xis(j),
      ),
    );
    j = k;
  }
  if (learnFrom == end) return end;
  // The learning level from the first chain (that of the run's last
  // node): its root, the root's value parent, then volatility parents.
  final root = end;
  var levels = 1;
  bool volParentOf(int p, int child) =>
      p < n &&
      source.kindOf(p) == NodeKind.continuousState &&
      _sameInts(source.volatilityChildrenOf(p), [child]);
  if (volParentOf(root + 2, root + 1)) {
    levels = 2;
    if (volParentOf(root + 3, root + 2)) levels = 3;
  }
  final call = AddEfState(
    first: learnFrom,
    nNodes: end - learnFrom,
    dimension: d,
    learning: EfLearning.values.firstWhere((l) => l.levels == levels),
    nus: source.getInitial(learnFrom, NodeField.nus),
    xis: xis(learnFrom),
  );
  calls.add(call);
  return end + call.impliedNodes;
}

bool _sameInts(List<int> a, List<int> b) =>
    a.length == b.length &&
    [for (var i = 0; i < a.length; i++) a[i] == b[i]].every((x) => x);

bool _sameNullable(List<double>? a, List<double>? b) =>
    a == null ? b == null : b != null && _sameList(a, b);

double _decode(Object? v) => switch (v) {
  num() => v.toDouble(),
  'NaN' => double.nan,
  'Infinity' => double.infinity,
  '-Infinity' => double.negativeInfinity,
  _ => throw FormatException('Not a number: $v'),
};
