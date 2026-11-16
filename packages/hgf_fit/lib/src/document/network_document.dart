import 'package:hgf/hgf.dart';

import '../codec.dart' show jsonDouble, readJsonDouble;
import '../codegen/plan.dart';

/// The kind of a [DocEdge]: a value or a volatility coupling.
enum EdgeType {
  value,
  volatility;

  static EdgeType parse(String name) => switch (name) {
    'value' => value,
    'volatility' => volatility,
    _ => throw FormatException('Unknown edge type "$name"'),
  };
}

/// A node of a [NetworkDocument], named by its [id].
///
/// A categorical, ef-state or dp-state node is one document node, as the
/// builder shows it; the engine nodes it implies (a categorical node's
/// binary and continuous parents, an ef-state node's learning chains) are
/// addressed as `id.k`, the `k`-th engine node of the block (`id` itself
/// is `id.0`). An ef-state node with [EfStateNode.nNodes] above 1 is a
/// group, as one `addEfState` call makes it.
sealed class DocNode {
  const DocNode(this.id, {this.label});

  /// Unique in the document; no dots (they address implied nodes).
  final String id;

  /// A display name, if not [id].
  final String? label;

  /// The engine nodes this node makes: itself and its implied nodes.
  int get size;

  Map<String, Object?> toJson();

  static DocNode fromJson(Map<String, Object?> json) {
    final id = json['id'] as String;
    final label = json['label'] as String?;
    final kind = NodeKind.parse(json['kind'] as String);
    Map<String, Object?> params() =>
        (json['params'] as Map<String, Object?>?) ?? const {};
    return switch (kind) {
      NodeKind.continuousState ||
      NodeKind.binaryState ||
      NodeKind.constantState => ScalarNode(
        id,
        kind: kind,
        label: label,
        coupling: (json['coupling_fn'] as String?) ?? 'linear',
        customCoupling: (json['custom_coupling'] as bool?) ?? false,
        params: {
          for (final MapEntry(:key, :value) in params().entries)
            NodeField.parse(key): readJsonDouble(value),
        },
      ),
      NodeKind.categoricalState => CategoricalNode(
        id,
        label: label,
        nCategories: (json['n_categories'] as num).toInt(),
        params: {
          for (final MapEntry(:key, :value) in params().entries)
            key: readJsonDouble(value),
        },
      ),
      NodeKind.efState => EfStateNode(
        id,
        label: label,
        nNodes: (json['n_nodes'] as num?)?.toInt() ?? 1,
        dimension: (json['dimension'] as num?)?.toInt() ?? 1,
        learning: EfLearning.parse(
          (json['learning'] as String?) ?? EfLearning.fixed.pyhgfName,
        ),
        nus: json['nus'] == null ? 3.0 : readJsonDouble(json['nus']),
        xis: switch (json['xis']) {
          final List<Object?> xs => [for (final x in xs) readJsonDouble(x)],
          _ => null,
        },
      ),
      NodeKind.dpState => DpStateNode(
        id,
        label: label,
        batchSize: (json['batch_size'] as num?)?.toInt() ?? 10,
        alpha: json['alpha'] == null ? 1.0 : readJsonDouble(json['alpha']),
        sensoryPrecision: json['sensory_precision'] == null
            ? 1.0
            : readJsonDouble(json['sensory_precision']),
      ),
    };
  }

  Map<String, Object?> _head(NodeKind kind) => {
    'id': id,
    'kind': kind.pyhgfName,
    'label': ?label,
  };
}

/// A continuous, binary or constant node (`Network.addNodes`).
final class ScalarNode extends DocNode {
  ScalarNode(
    super.id, {
    required this.kind,
    super.label,
    this.coupling = 'linear',
    this.customCoupling = false,
    Map<NodeField, double> params = const {},
  }) : params = Map.unmodifiable(params) {
    if (kind != NodeKind.continuousState &&
        kind != NodeKind.binaryState &&
        kind != NodeKind.constantState) {
      throw ArgumentError.value(kind, 'kind', 'Not a scalar node kind');
    }
  }

  final NodeKind kind;

  /// The value coupling to this node's value children: a [CouplingFn]'s
  /// pyhgf name, an expression in `x` ([ExprCoupling], e.g.
  /// `2 * tanh(x)`), or a custom coupling's name when [customCoupling].
  final String coupling;
  final bool customCoupling;

  /// Initial values given to `addNodes` (those that differ from the kind's
  /// defaults).
  final Map<NodeField, double> params;

  @override
  int get size => 1;

  @override
  Map<String, Object?> toJson() => {
    ..._head(kind),
    if (coupling != 'linear' || customCoupling) 'coupling_fn': coupling,
    if (customCoupling) 'custom_coupling': true,
    if (params.isNotEmpty)
      'params': {
        for (final MapEntry(:key, :value) in params.entries)
          key.pyhgfName: jsonDouble(value),
      },
  };
}

/// A categorical node and its `2 n + 1` implied nodes
/// (`Network.addCategoricalState`).
final class CategoricalNode extends DocNode {
  CategoricalNode(
    super.id, {
    required this.nCategories,
    super.label,
    Map<String, double> params = const {},
  }) : params = Map.unmodifiable(params) {
    for (final k in params.keys) {
      if (!AddCategoricalState.pyhgfKeys.containsValue(k)) {
        throw ArgumentError.value(k, 'params', 'Not a categorical parameter');
      }
    }
  }

  final int nCategories;

  /// Non-default arguments, by pyhgf's `binary_parameters` keys (`mean_1`,
  /// `precision_1`, `mean_2`, …; see [AddCategoricalState.pyhgfKeys]).
  final Map<String, double> params;

  @override
  int get size => 2 * nCategories + 2;

  @override
  Map<String, Object?> toJson() => {
    ..._head(NodeKind.categoricalState),
    'n_categories': nCategories,
    if (params.isNotEmpty)
      'params': {
        for (final MapEntry(:key, :value) in params.entries)
          key: jsonDouble(value),
      },
  };

  AddCategoricalState _call(int node) {
    double? p(String dartName) =>
        params[AddCategoricalState.pyhgfKeys[dartName]!];
    return AddCategoricalState(
      node: node,
      nCategories: nCategories,
      binaryMean: p('binaryMean'),
      binaryPrecision: p('binaryPrecision'),
      valueMean: p('valueMean'),
      valuePrecision: p('valuePrecision'),
      valueTonicVolatility: p('valueTonicVolatility'),
      volatilityMean: p('volatilityMean'),
      volatilityPrecision: p('volatilityPrecision'),
      volatilityTonicVolatility: p('volatilityTonicVolatility'),
    );
  }
}

/// [nNodes] ef-state nodes and their learning chains
/// (`Network.addEfState`). As a cluster of a dp-state node, it has one
/// value edge to it (strength 1).
final class EfStateNode extends DocNode {
  EfStateNode(
    super.id, {
    super.label,
    this.nNodes = 1,
    this.dimension = 1,
    this.learning = EfLearning.fixed,
    this.nus = 3.0,
    List<double>? xis,
  }) : xis = xis == null ? null : List.unmodifiable(xis);

  final int nNodes, dimension;
  final EfLearning learning;
  final double nus;

  /// The initial sufficient statistics, when they aren't N(0, I)'s.
  final List<double>? xis;

  @override
  int get size =>
      nNodes +
      (learning.levels == 0
          ? 0
          : nNodes * efStatisticsCount(dimension) * (learning.levels + 1));

  @override
  Map<String, Object?> toJson() => {
    ..._head(NodeKind.efState),
    if (nNodes != 1) 'n_nodes': nNodes,
    if (dimension != 1) 'dimension': dimension,
    if (learning != EfLearning.fixed) 'learning': learning.pyhgfName,
    'nus': jsonDouble(nus),
    if (xis case final xs?) 'xis': [for (final x in xs) jsonDouble(x)],
  };
}

/// A Dirichlet process node (`Network.addDpState`); its clusters are
/// [EfStateNode]s with a value edge to it.
final class DpStateNode extends DocNode {
  const DpStateNode(
    super.id, {
    super.label,
    this.batchSize = 10,
    this.alpha = 1.0,
    this.sensoryPrecision = 1.0,
  });

  final int batchSize;
  final double alpha, sensoryPrecision;

  @override
  int get size => 1;

  @override
  Map<String, Object?> toJson() => {
    ..._head(NodeKind.dpState),
    'batch_size': batchSize,
    'alpha': jsonDouble(alpha),
    'sensory_precision': jsonDouble(sensoryPrecision),
  };
}

/// A coupling from [parent] to [child] (node addresses: an id, or `id.k`
/// for an implied node).
final class DocEdge {
  const DocEdge(this.type, this.parent, this.child, {this.strength = 1.0});

  final EdgeType type;
  final String parent, child;
  final double strength;

  Map<String, Object?> toJson() => {
    'type': type.name,
    'parent': parent,
    'child': child,
    if (strength != 1.0) 'strength': jsonDouble(strength),
  };

  factory DocEdge.fromJson(Map<String, Object?> json) => DocEdge(
    EdgeType.parse(json['type'] as String),
    json['parent'] as String,
    json['child'] as String,
    strength: json['strength'] == null ? 1.0 : readJsonDouble(json['strength']),
  );
}

/// A value set after the nodes and edges: what their declarations don't
/// say (an input node's value set after building, an implied node's
/// parameter, a strength on an implied edge).
sealed class DocOverride {
  const DocOverride();

  Map<String, Object?> toJson();

  static DocOverride fromJson(Map<String, Object?> json) {
    if (json['coupling'] case final String type) {
      return CouplingOverride(
        EdgeType.parse(type),
        json['parent'] as String,
        json['child'] as String,
        readJsonDouble(json['strength']),
      );
    }
    if (json['vector_field'] case final String name) {
      return VectorOverride(
        json['node'] as String,
        VectorField.values.firstWhere(
          (f) => f.pyhgfName == name,
          orElse: () => throw FormatException('Unknown vector field "$name"'),
        ),
        [for (final x in json['values'] as List<Object?>) readJsonDouble(x)],
      );
    }
    return InitialOverride(
      json['node'] as String,
      NodeField.parse(json['field'] as String),
      readJsonDouble(json['value']),
    );
  }
}

/// `Network.setInitial`.
final class InitialOverride extends DocOverride {
  const InitialOverride(this.node, this.field, this.value);

  final String node;
  final NodeField field;
  final double value;

  @override
  Map<String, Object?> toJson() => {
    'node': node,
    'field': field.pyhgfName,
    'value': jsonDouble(value),
  };
}

/// `Network.setValueCoupling` / `setVolatilityCoupling`.
final class CouplingOverride extends DocOverride {
  const CouplingOverride(this.type, this.parent, this.child, this.strength);

  final EdgeType type;
  final String parent, child;
  final double strength;

  @override
  Map<String, Object?> toJson() => {
    'coupling': type.name,
    'parent': parent,
    'child': child,
    'strength': jsonDouble(strength),
  };
}

/// `Network.setInitialVector`.
final class VectorOverride extends DocOverride {
  VectorOverride(this.node, this.field, List<double> values)
    : values = List.unmodifiable(values);

  final String node;
  final VectorField field;
  final List<double> values;

  @override
  Map<String, Object?> toJson() => {
    'node': node,
    'vector_field': field.pyhgfName,
    'values': [for (final x in values) jsonDouble(x)],
  };
}

/// An HGF network as the model builder edits it: nodes by id, a flat edge
/// list, the update scheme and settings.
///
/// It compiles to a `Network` through the builder calls (`addNodes` and
/// friends) in node order: each edge is declared by its later endpoint,
/// which must be a continuous, binary or constant node (or an ef-state
/// node, for its value edge to an earlier dp-state node). A node's edges
/// keep their order in [edges]. [NetworkDocument.fromNetwork] recovers a
/// document from any network the calls can build (via [NetworkPlan]), and
/// the document builds it again bit for bit.
final class NetworkDocument {
  NetworkDocument({
    this.volatilityUpdates = VolatilityUpdates.unbounded,
    this.meanFieldUpdates = false,
    this.ehgfMgfCorrection = false,
    NetworkFixes? fixes,
    this.maxPosteriorPrecision = 1e10,
    this.precisionClippingValue = 1e-6,
    required List<DocNode> nodes,
    List<DocEdge> edges = const [],
    List<DocOverride> overrides = const [],
  }) : fixes = fixes ?? Network.defaultFixes,
       nodes = List.unmodifiable(nodes),
       edges = List.unmodifiable(edges),
       overrides = List.unmodifiable(overrides) {
    final seen = <String>{};
    for (final n in nodes) {
      if (n.id.isEmpty || n.id.contains('.')) {
        throw ArgumentError.value(n.id, 'id', 'Empty, or contains a dot');
      }
      if (!seen.add(n.id)) {
        throw ArgumentError.value(n.id, 'id', 'Duplicate node id');
      }
    }
  }

  final VolatilityUpdates volatilityUpdates;
  final bool meanFieldUpdates, ehgfMgfCorrection;
  final NetworkFixes fixes;
  final double maxPosteriorPrecision, precisionClippingValue;
  final List<DocNode> nodes;
  final List<DocEdge> edges;
  final List<DocOverride> overrides;

  /// The document of [network]: one node per builder call's node (merged
  /// `addNodes` calls split up), ids from [id] (default `n<index>`, the
  /// node's index in [network]).
  factory NetworkDocument.fromNetwork(
    Network network, {
    String Function(int index)? id,
  }) {
    final plan = NetworkPlan.fromJson(network.toJson());
    final idOf = id ?? (i) => 'n$i';
    final address = List<String?>.filled(plan.nNodes, null);
    void block(int first, int size) {
      final base = idOf(first);
      for (var k = 0; k < size; k++) {
        address[first + k] = k == 0 ? base : '$base.$k';
      }
    }

    for (final call in plan.calls) {
      switch (call) {
        case AddNodes(:final first, :final nNodes):
          for (var j = 0; j < nNodes; j++) {
            address[first + j] = idOf(first + j);
          }
        case AddEfState(:final first, :final nNodes):
          block(first, nNodes + call.impliedNodes);
        case AddCategoricalState(:final node):
          block(node, call.impliedNodes + 1);
        case AddDpState(:final node):
          block(node, 1);
        case SetInitial() || SetCoupling() || SetInitialVector():
          break;
      }
    }
    String at(int i) => address[i]!;

    final nodes = <DocNode>[];
    final edges = <DocEdge>[];
    final overrides = <DocOverride>[];
    for (final call in plan.calls) {
      switch (call) {
        case AddNodes():
          for (var j = 0; j < call.nNodes; j++) {
            final me = at(call.first + j);
            nodes.add(
              ScalarNode(
                me,
                kind: call.kind,
                coupling: call.customCoupling ?? couplingText(call.coupling),
                customCoupling: call.customCoupling != null,
                params: call.parameters,
              ),
            );
            for (final e in call.valueParents) {
              edges.add(
                DocEdge(EdgeType.value, at(e.node), me, strength: e.strength),
              );
            }
            for (final e in call.valueChildren) {
              edges.add(
                DocEdge(EdgeType.value, me, at(e.node), strength: e.strength),
              );
            }
            for (final e in call.volatilityParents) {
              edges.add(
                DocEdge(
                  EdgeType.volatility,
                  at(e.node),
                  me,
                  strength: e.strength,
                ),
              );
            }
            for (final e in call.volatilityChildren) {
              edges.add(
                DocEdge(
                  EdgeType.volatility,
                  me,
                  at(e.node),
                  strength: e.strength,
                ),
              );
            }
          }
        case AddEfState():
          final me = at(call.first);
          nodes.add(
            EfStateNode(
              me,
              nNodes: call.nNodes,
              dimension: call.dimension,
              learning: call.learning,
              nus: call.nus,
              xis: call.xis,
            ),
          );
          for (final c in call.valueChildren ?? const <int>[]) {
            edges.add(DocEdge(EdgeType.value, me, at(c)));
          }
        case AddCategoricalState():
          nodes.add(
            CategoricalNode(
              at(call.node),
              nCategories: call.nCategories,
              params: {
                for (final MapEntry(:key, :value) in call.arguments.entries)
                  AddCategoricalState.pyhgfKeys[key]!: value,
              },
            ),
          );
        case AddDpState():
          nodes.add(
            DpStateNode(
              at(call.node),
              batchSize: call.batchSize,
              alpha: call.alpha,
              sensoryPrecision: call.sensoryPrecision,
            ),
          );
        case SetInitial(:final node, :final field, :final value):
          overrides.add(InitialOverride(at(node), field, value));
        case SetCoupling(:final parent, :final child, :final strength):
          overrides.add(
            CouplingOverride(
              call.volatility ? EdgeType.volatility : EdgeType.value,
              at(parent),
              at(child),
              strength,
            ),
          );
        case SetInitialVector(:final node, :final field, :final values):
          overrides.add(VectorOverride(at(node), field, values));
      }
    }
    return NetworkDocument(
      volatilityUpdates: plan.volatilityUpdates,
      meanFieldUpdates: plan.meanFieldUpdates,
      ehgfMgfCorrection: plan.ehgfMgfCorrection,
      fixes: plan.fixes,
      maxPosteriorPrecision: plan.maxPosteriorPrecision,
      precisionClippingValue: plan.precisionClippingValue,
      nodes: nodes,
      edges: edges,
      overrides: overrides,
    );
  }

  factory NetworkDocument.fromJson(Map<String, Object?> json) =>
      NetworkDocument(
        volatilityUpdates: VolatilityUpdates.parse(
          (json['volatility_updates'] as String?) ?? 'unbounded',
        ),
        meanFieldUpdates: (json['mean_field_updates'] as bool?) ?? false,
        ehgfMgfCorrection: (json['ehgf_mgf_correction'] as bool?) ?? false,
        fixes: switch (json['fixes']) {
          final Map<String, Object?> f => NetworkFixes.fromJson(f),
          _ => null,
        },
        maxPosteriorPrecision: json['max_posterior_precision'] == null
            ? 1e10
            : readJsonDouble(json['max_posterior_precision']),
        precisionClippingValue: json['precision_clipping_value'] == null
            ? 1e-6
            : readJsonDouble(json['precision_clipping_value']),
        nodes: [
          for (final n in json['nodes'] as List<Object?>)
            DocNode.fromJson(n as Map<String, Object?>),
        ],
        edges: [
          for (final e in (json['edges'] as List<Object?>?) ?? const [])
            DocEdge.fromJson(e as Map<String, Object?>),
        ],
        overrides: [
          for (final o in (json['overrides'] as List<Object?>?) ?? const [])
            DocOverride.fromJson(o as Map<String, Object?>),
        ],
      );

  Map<String, Object?> toJson() => {
    'volatility_updates': volatilityUpdates.pyhgfName,
    'mean_field_updates': meanFieldUpdates,
    'ehgf_mgf_correction': ehgfMgfCorrection,
    'fixes': fixes.toJson(),
    'max_posterior_precision': jsonDouble(maxPosteriorPrecision),
    'precision_clipping_value': jsonDouble(precisionClippingValue),
    'nodes': [for (final n in nodes) n.toJson()],
    'edges': [for (final e in edges) e.toJson()],
    if (overrides.isNotEmpty)
      'overrides': [for (final o in overrides) o.toJson()],
  };

  /// The names of the custom couplings the nodes use.
  List<String> get customCouplings => [
    ...{
      for (final n in nodes)
        if (n case ScalarNode(customCoupling: true, :final coupling)) coupling,
    },
  ];

  /// Where each node lands in the built network.
  late final NodeAddresses addresses = NodeAddresses._(nodes);

  /// The builder calls that make this network, in order: what [build]
  /// replays.
  List<BuildCall> calls() {
    final a = addresses;
    final resolved = [
      for (final e in edges) (e, a.indexOf(e.parent), a.indexOf(e.child)),
    ];
    // Each edge is declared by its later endpoint.
    for (final (e, p, c) in resolved) {
      if (p == c) {
        throw FormatException(
          'Edge ${_show(e)}: a node cannot be its own parent',
        );
      }
      final later = p > c ? p : c;
      final owner = a.nodeAt(later);
      final ok = switch (owner) {
        ScalarNode() => true,
        EfStateNode() =>
          later == a.firstOf(owner.id) &&
              later == p &&
              e.type == EdgeType.value &&
              a.nodeAt(c) is DpStateNode &&
              c == a.firstOf(a.nodeAt(c).id) &&
              e.strength == 1.0,
        _ => false,
      };
      if (!ok) {
        throw FormatException(
          'Edge ${_show(e)}: "${a.addressOf(later)}" comes after '
          '"${a.addressOf(p + c - later)}" but cannot declare this edge '
          '(only continuous, binary and constant nodes declare edges, and '
          'an ef-state node its value edge to a dp-state node). Move '
          '"${owner.id}" before it, or the other node after it.',
        );
      }
    }

    final calls = <BuildCall>[];
    for (final node in nodes) {
      final f = a.firstOf(node.id);
      switch (node) {
        case ScalarNode():
          final vp = <PlanEdge>[], vc = <PlanEdge>[];
          final volp = <PlanEdge>[], volc = <PlanEdge>[];
          for (final (e, p, c) in resolved) {
            final value = e.type == EdgeType.value;
            if (c == f && p < f) {
              (value ? vp : volp).add((node: p, strength: e.strength));
            } else if (p == f && c < f) {
              (value ? vc : volc).add((node: c, strength: e.strength));
            }
          }
          final named = node.customCoupling
              ? null
              : parseCoupling(node.coupling);
          calls.add(
            AddNodes(
              first: f,
              kind: node.kind,
              valueParents: vp,
              valueChildren: vc,
              volatilityParents: volp,
              volatilityChildren: volc,
              coupling: named ?? CouplingFn.linear,
              customCoupling: node.customCoupling ? node.coupling : null,
              parameters: node.params,
            ),
          );
        case EfStateNode():
          final children = [
            for (final (_, p, c) in resolved)
              if (p == f && c < f) c,
          ];
          calls.add(
            AddEfState(
              first: f,
              nNodes: node.nNodes,
              dimension: node.dimension,
              learning: node.learning,
              nus: node.nus,
              xis: node.xis,
              valueChildren: children.isEmpty ? null : children,
            ),
          );
        case CategoricalNode():
          calls.add(node._call(f));
        case DpStateNode():
          calls.add(
            AddDpState(
              node: f,
              batchSize: node.batchSize,
              alpha: node.alpha,
              sensoryPrecision: node.sensoryPrecision,
            ),
          );
      }
    }
    for (final o in overrides) {
      calls.add(switch (o) {
        InitialOverride(:final node, :final field, :final value) => SetInitial(
          a.indexOf(node),
          field,
          value,
        ),
        CouplingOverride(:final type, :final parent, :final child) =>
          SetCoupling(
            a.indexOf(parent),
            a.indexOf(child),
            o.strength,
            volatility: type == EdgeType.volatility,
          ),
        VectorOverride(:final node, :final field, :final values) =>
          SetInitialVector(a.indexOf(node), field, values),
      });
    }
    return calls;
  }

  /// The network, with [couplings] for the custom couplings (by name;
  /// default: placeholders that build but cannot run).
  Network build({Map<String, Coupling>? couplings}) {
    final c = couplings ?? NetworkPlan.placeholderCouplings(customCouplings);
    for (final name in customCouplings) {
      if (!c.containsKey(name)) {
        throw ArgumentError(
          'The network uses the custom coupling "$name"; pass it in '
          '`couplings`',
        );
      }
    }
    final network = Network(
      volatilityUpdates: volatilityUpdates,
      meanFieldUpdates: meanFieldUpdates,
      ehgfMgfCorrection: ehgfMgfCorrection,
      fixes: fixes,
      maxPosteriorPrecision: maxPosteriorPrecision,
      precisionClippingValue: precisionClippingValue,
    );
    for (final call in calls()) {
      call.apply(network, c);
    }
    return network..build();
  }

  static String _show(DocEdge e) => '${e.parent} -${e.type.name}-> ${e.child}';
}

/// The engine indices of a [NetworkDocument]'s nodes.
final class NodeAddresses {
  NodeAddresses._(List<DocNode> nodes) {
    var i = 0;
    for (final n in nodes) {
      _first[n.id] = i;
      for (var k = 0; k < n.size; k++) {
        _owner.add(n);
      }
      i += n.size;
    }
  }

  final _first = <String, int>{};
  final _owner = <DocNode>[];

  /// The number of engine nodes.
  int get length => _owner.length;

  /// The index of the first engine node of the node [id].
  int firstOf(String id) =>
      _first[id] ?? (throw FormatException('No node "$id"'));

  /// The document node an engine index belongs to.
  DocNode nodeAt(int index) => _owner[index];

  /// The engine index of [address] (`id` or `id.k`).
  int indexOf(String address) {
    final dot = address.indexOf('.');
    if (dot < 0) return firstOf(address);
    final id = address.substring(0, dot);
    final k = int.tryParse(address.substring(dot + 1));
    final first = firstOf(id);
    final size = _owner[first].size;
    if (k == null || k < 0 || k >= size) {
      throw FormatException(
        'Bad address "$address": "$id" has implied nodes $id.1 to '
        '$id.${size - 1}',
      );
    }
    return first + k;
  }

  /// The address of engine node [index].
  String addressOf(int index) {
    final node = _owner[index];
    final k = index - _first[node.id]!;
    return k == 0 ? node.id : '${node.id}.$k';
  }
}
