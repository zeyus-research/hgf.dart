import 'package:hgf/hgf.dart';

import 'network_document.dart';

/// Edits of a [NetworkDocument], as the model builder makes them. Each
/// returns a new document; nodes stay in an order that compiles
/// ([ordered]), so the builder never has to think about it.
extension NetworkDocumentEdits on NetworkDocument {
  /// This document with the given parts replaced.
  NetworkDocument copyWith({
    VolatilityUpdates? volatilityUpdates,
    bool? meanFieldUpdates,
    NetworkFixes? fixes,
    List<DocNode>? nodes,
    List<DocEdge>? edges,
    List<DocOverride>? overrides,
  }) => NetworkDocument(
    volatilityUpdates: volatilityUpdates ?? this.volatilityUpdates,
    meanFieldUpdates: meanFieldUpdates ?? this.meanFieldUpdates,
    fixes: fixes ?? this.fixes,
    maxPosteriorPrecision: maxPosteriorPrecision,
    precisionClippingValue: precisionClippingValue,
    nodes: nodes ?? this.nodes,
    edges: edges ?? this.edges,
    overrides: overrides ?? this.overrides,
  );

  /// The node [id], or null.
  DocNode? node(String id) {
    for (final n in nodes) {
      if (n.id == id) return n;
    }
    return null;
  }

  /// The first id `<prefix><k>` (k = 0, 1, …) no node has.
  String freshId([String prefix = 'x']) {
    final used = {for (final n in nodes) n.id};
    for (var k = 0; ; k++) {
      if (!used.contains('$prefix$k')) return '$prefix$k';
    }
  }

  /// With [node] added: a continuous, binary or constant node last, any
  /// other kind after the last node of a kind other than those (see
  /// [ordered]).
  NetworkDocument withNode(DocNode node) {
    if (this.node(node.id) != null) {
      throw ArgumentError.value(node.id, 'id', 'Duplicate node id');
    }
    final list = [...nodes];
    if (node is ScalarNode) {
      list.add(node);
    } else {
      final at = list.lastIndexWhere((n) => n is! ScalarNode) + 1;
      list.insert(at, node);
    }
    return copyWith(nodes: list).ordered();
  }

  /// Without the node [id], its edges and its overrides (implied nodes'
  /// included).
  NetworkDocument withoutNode(String id) {
    bool mine(String address) => _ownerId(address) == id;
    return copyWith(
      nodes: [
        for (final n in nodes)
          if (n.id != id) n,
      ],
      edges: [
        for (final e in edges)
          if (!mine(e.parent) && !mine(e.child)) e,
      ],
      overrides: [
        for (final o in overrides)
          if (!_overrideAddresses(o).any(mine)) o,
      ],
    );
  }

  /// With the node [id] replaced by [node]. A new id renames every
  /// reference to it; edges and overrides that address implied nodes
  /// [node] no longer has (fewer categories, say) are dropped.
  NetworkDocument replaceNode(String id, DocNode node) {
    final i = nodes.indexWhere((n) => n.id == id);
    if (i < 0) throw ArgumentError.value(id, 'id', 'No such node');
    if (node.id != id && this.node(node.id) != null) {
      throw ArgumentError.value(node.id, 'id', 'Duplicate node id');
    }
    String? map(String address) {
      final (owner, k) = _split(address);
      if (owner != id) return address;
      if (k >= node.size) return null;
      return k == 0 ? node.id : '${node.id}.$k';
    }

    final list = [...nodes]..[i] = node;
    final newEdges = <DocEdge>[];
    for (final e in edges) {
      final p = map(e.parent), c = map(e.child);
      if (p == null || c == null) continue;
      newEdges.add(DocEdge(e.type, p, c, strength: e.strength));
    }
    final newOverrides = <DocOverride>[];
    for (final o in overrides) {
      switch (o) {
        case InitialOverride(:final field, :final value):
          if (map(o.node) case final a?) {
            newOverrides.add(InitialOverride(a, field, value));
          }
        case CouplingOverride(:final type, :final strength):
          final p = map(o.parent), c = map(o.child);
          if (p != null && c != null) {
            newOverrides.add(CouplingOverride(type, p, c, strength));
          }
        case VectorOverride(:final field, :final values):
          if (map(o.node) case final a?) {
            newOverrides.add(VectorOverride(a, field, values));
          }
      }
    }
    return copyWith(
      nodes: list,
      edges: newEdges,
      overrides: newOverrides,
    ).ordered();
  }

  /// With [edge] appended (no checks: see [checkNetwork]).
  NetworkDocument withEdge(DocEdge edge) =>
      copyWith(edges: [...edges, edge]).ordered();

  /// Without the edge at [index].
  NetworkDocument withoutEdge(int index) =>
      copyWith(edges: [...edges]..removeAt(index));

  /// With the edge at [index] replaced by [edge]. A coupling override of
  /// the same edge is dropped when the strength changes (it would win).
  NetworkDocument replaceEdge(int index, DocEdge edge) {
    final old = edges[index];
    return copyWith(
      edges: [...edges]..[index] = edge,
      overrides: old.strength == edge.strength
          ? null
          : [
              for (final o in overrides)
                if (!(o is CouplingOverride &&
                    o.type == old.type &&
                    o.parent == old.parent &&
                    o.child == old.child))
                  o,
            ],
    ).ordered();
  }

  /// With [field] of the node at [address] set to [value]: in a scalar
  /// node's own parameters (dropping any override of the same field), or
  /// as an [InitialOverride] (an implied node, an input's value).
  NetworkDocument withParam(String address, NodeField field, double value) {
    final own = node(address);
    final rest = [
      for (final o in overrides)
        if (!(o is InitialOverride && o.node == address && o.field == field)) o,
    ];
    if (own is ScalarNode && own.kind != NodeKind.constantState) {
      return copyWith(
        nodes: [
          for (final n in nodes)
            n.id == address
                ? ScalarNode(
                    own.id,
                    kind: own.kind,
                    label: own.label,
                    coupling: own.coupling,
                    customCoupling: own.customCoupling,
                    params: {...own.params, field: value},
                  )
                : n,
        ],
        overrides: rest,
      );
    }
    return copyWith(
      overrides: [...rest, InitialOverride(address, field, value)],
    );
  }

  /// The nodes in an order that compiles: this document if it does, else
  /// the kinds that declare no edges first (dp-state, categorical, then
  /// ef-state, so clusters follow their dp-state node), then the
  /// continuous, binary and constant nodes, each group in its order here.
  NetworkDocument ordered() {
    if (_compiles(this)) return this;
    int rank(DocNode n) => switch (n) {
      DpStateNode() || CategoricalNode() => 0,
      EfStateNode() => 1,
      ScalarNode() => 2,
    };
    final sorted = [...nodes];
    _stableSort(sorted, (a, b) => rank(a) - rank(b));
    return copyWith(nodes: sorted);
  }

  /// The document node an address (`id` or `id.k`) belongs to, or null.
  DocNode? ownerOf(String address) => node(_ownerId(address));
}

bool _compiles(NetworkDocument doc) {
  try {
    doc.calls();
    return true;
  } on Object {
    return false;
  }
}

void _stableSort<T>(List<T> list, int Function(T, T) compare) {
  final indexed = [for (final (i, x) in list.indexed) (i, x)];
  indexed.sort((a, b) {
    final c = compare(a.$2, b.$2);
    return c != 0 ? c : a.$1 - b.$1;
  });
  for (var i = 0; i < list.length; i++) {
    list[i] = indexed[i].$2;
  }
}

String _ownerId(String address) {
  final dot = address.indexOf('.');
  return dot < 0 ? address : address.substring(0, dot);
}

/// `(id, k)` of an address; k = -1 when it doesn't parse.
(String, int) _split(String address) {
  final dot = address.indexOf('.');
  if (dot < 0) return (address, 0);
  return (
    address.substring(0, dot),
    int.tryParse(address.substring(dot + 1)) ?? -1,
  );
}

Iterable<String> _overrideAddresses(DocOverride o) => switch (o) {
  InitialOverride(:final node) => [node],
  CouplingOverride(:final parent, :final child) => [parent, child],
  VectorOverride(:final node) => [node],
};

/// How bad a [DocIssue] is.
enum IssueSeverity {
  /// The network can't be built (or would loop).
  error,

  /// It builds, but probably isn't what was meant.
  warning,
}

/// A problem with a [NetworkDocument], for the builder to show next to the
/// nodes and edges concerned.
final class DocIssue {
  const DocIssue(
    this.severity,
    this.message, {
    this.nodes = const [],
    this.edges = const [],
    this.parts = const [],
  });

  final IssueSeverity severity;
  final String message;

  /// The ids of the nodes concerned.
  final List<String> nodes;

  /// The indices (in [NetworkDocument.edges]) of the edges concerned.
  final List<int> edges;

  /// The parts of a model document concerned (see [DocPart]): its data,
  /// an input, a transform, the response, an output, or the network.
  final List<String> parts;

  bool get isError => severity == IssueSeverity.error;

  /// This issue, also naming [more] parts.
  DocIssue inParts(List<String> more) => DocIssue(
    severity,
    message,
    nodes: nodes,
    edges: edges,
    parts: [...parts, ...more],
  );

  @override
  String toString() => '${severity.name}: $message';
}

/// What is wrong with [doc], errors first: edges the engine can't build
/// (a binary node's volatility edge, a parent of a constant node, an edge
/// of a categorical node, …), cycles, couplings that don't parse, a
/// dp-state node whose clusters don't add up, and, if all that passes,
/// whatever building the network throws. Warnings: a binary node without
/// a continuous value parent, a coupling function on a node without value
/// children, the same pair coupled twice.
///
/// Without [build], the last step (building the network) is left to the
/// caller, who may want the network anyway.
List<DocIssue> checkNetwork(NetworkDocument doc, {bool build = true}) {
  final issues = <DocIssue>[];
  void error(
    String m, {
    List<String> nodes = const [],
    List<int> edges = const [],
  }) =>
      issues.add(DocIssue(IssueSeverity.error, m, nodes: nodes, edges: edges));
  void warn(
    String m, {
    List<String> nodes = const [],
    List<int> edges = const [],
  }) => issues.add(
    DocIssue(IssueSeverity.warning, m, nodes: nodes, edges: edges),
  );

  if (doc.nodes.isEmpty) {
    error('The network has no nodes: add one.');
    return issues;
  }
  final byId = {for (final n in doc.nodes) n.id: n};
  String name(DocNode n) => n.label ?? n.id;

  // Resolve each edge's endpoints: (node, k), k = 0 for the node itself.
  (DocNode, int)? resolve(String address) {
    final (id, k) = _split(address);
    final n = byId[id];
    if (n == null || k < 0 || k >= n.size) return null;
    return (n, k);
  }

  final resolved = <int, ((DocNode, int), (DocNode, int))>{};
  final seen = <(EdgeType, String, String), int>{};
  final pairs = <(String, String), int>{};
  for (final (i, e) in doc.edges.indexed) {
    final p = resolve(e.parent), c = resolve(e.child);
    if (p == null || c == null) {
      final bad = p == null ? e.parent : e.child;
      error(
        'Edge ${e.parent} → ${e.child}: there is no node "$bad".',
        edges: [i],
      );
      continue;
    }
    if (e.parent == e.child) {
      error(
        '${e.parent} cannot be its own parent.',
        nodes: [p.$1.id],
        edges: [i],
      );
      continue;
    }
    if (seen[(e.type, e.parent, e.child)] case final first?) {
      error(
        '${e.parent} → ${e.child}: this ${e.type.name} edge is there twice.',
        edges: [first, i],
      );
      continue;
    }
    seen[(e.type, e.parent, e.child)] = i;
    if (pairs[(e.parent, e.child)] case final other?) {
      warn(
        '${e.parent} is both a value and a volatility parent of ${e.child}.',
        edges: [other, i],
      );
    }
    pairs[(e.parent, e.child)] = i;
    resolved[i] = (p, c);
  }

  // Kind rules, per edge.
  for (final MapEntry(key: i, value: (p, c)) in resolved.entries) {
    final e = doc.edges[i];
    final (pn, pk) = p;
    final (cn, ck) = c;
    final vol = e.type == EdgeType.volatility;
    final ids = [pn.id, cn.id];
    String? why;
    // The head of a block (k == 0) is the node itself; implied nodes of
    // categorical and ef-state blocks are continuous or binary nodes and
    // may take edges from scalar nodes.
    if (pk == 0 && pn is ScalarNode) {
      if (pn.kind == NodeKind.binaryState && vol) {
        why = 'binary nodes take value edges only';
      } else if (pn.kind == NodeKind.constantState && vol) {
        why = 'a constant node (a bias) can only be a value parent';
      }
    }
    if (why == null && ck == 0 && cn is ScalarNode) {
      if (cn.kind == NodeKind.binaryState && vol) {
        why = 'binary nodes take value edges only';
      } else if (cn.kind == NodeKind.constantState) {
        why = 'a constant node (a bias) takes no parents';
      }
    }
    if (why == null) {
      if (pk == 0 && pn is CategoricalNode ||
          ck == 0 && cn is CategoricalNode) {
        why =
            'a categorical node\'s edges are its own (connect to its implied '
            'nodes, e.g. ${(pn is CategoricalNode ? pn : cn).id}.1, instead)';
      } else if (ck == 0 && cn is DpStateNode) {
        if (!(pk == 0 && pn is EfStateNode && !vol)) {
          why =
              'a dp-state node\'s only parents are its clusters (ef-state '
              'nodes, by value edges)';
        } else if (e.strength != 1.0) {
          why = 'a cluster\'s edge to its dp-state node has strength 1';
        }
      } else if (pk == 0 && pn is DpStateNode) {
        why = 'a dp-state node has no children';
      } else if (pk == 0 && pn is EfStateNode) {
        why =
            'an ef-state node can only be a cluster of a dp-state node (a '
            'value parent)';
      } else if (ck == 0 && cn is EfStateNode) {
        why =
            'an ef-state node takes no parents here (with HGF learning, its '
            'learning chains are implied: ${cn.id}.1, …)';
      }
    }
    if (why != null) {
      error('${e.parent} → ${e.child}: $why.', nodes: ids, edges: [i]);
    }
  }

  // Cycles, over addresses.
  final children = <String, List<(String, int)>>{};
  for (final i in resolved.keys) {
    final e = doc.edges[i];
    (children[e.parent] ??= []).add((e.child, i));
  }
  final state = <String, int>{}; // 1: on the stack, 2: done
  final stack = <(String, int)>[]; // (address, edge into it)
  List<int>? cycle;
  void visit(String a) {
    if (cycle != null) return;
    state[a] = 1;
    for (final (c, i) in children[a] ?? const <(String, int)>[]) {
      if (cycle != null) return;
      if (state[c] == 1) {
        final at = stack.lastIndexWhere((s) => s.$1 == c);
        cycle = [for (final s in stack.skip(at + 1)) s.$2, i];
        return;
      }
      if (state[c] == null) {
        stack.add((c, i));
        visit(c);
        stack.removeLast();
      }
    }
    state[a] = 2;
  }

  for (final a in children.keys) {
    if (state[a] == null) {
      stack.add((a, -1));
      visit(a);
      stack.removeLast();
    }
  }
  if (cycle case final c?) {
    error(
      'The edges ${c.map((i) => '${doc.edges[i].parent} → ${doc.edges[i].child}').join(', ')} '
      'make a cycle: a node cannot be its own ancestor.',
      nodes: {for (final i in c) _ownerId(doc.edges[i].parent)}.toList(),
      edges: c,
    );
  }

  // Per node.
  final valueParents = <String, List<String>>{};
  final valueChildren = <String, int>{};
  for (final i in resolved.keys) {
    final e = doc.edges[i];
    if (e.type != EdgeType.value) continue;
    (valueParents[e.child] ??= []).add(e.parent);
    valueChildren[e.parent] = (valueChildren[e.parent] ?? 0) + 1;
  }
  for (final n in doc.nodes) {
    switch (n) {
      case ScalarNode(:final kind, :final coupling, :final customCoupling):
        if (kind == NodeKind.constantState) {
          if ((valueChildren[n.id] ?? 0) == 0) {
            error(
              '${name(n)} is a constant node (a bias) without a value child.',
              nodes: [n.id],
            );
          }
          continue;
        }
        if (!customCoupling) {
          try {
            parseCoupling(coupling);
          } on ArgumentError catch (e) {
            error('${name(n)}: ${e.message}', nodes: [n.id]);
          }
        }
        if ((coupling != 'linear' && coupling != 'identity') &&
            (valueChildren[n.id] ?? 0) == 0) {
          warn(
            '${name(n)} has the coupling "$coupling", which only applies to '
            'value children, and has none.',
            nodes: [n.id],
          );
        }
        if (kind == NodeKind.binaryState) {
          final ps = valueParents[n.id] ?? const [];
          final continuous = ps.any((a) {
            final o = resolve(a);
            if (o == null) return false;
            final (on, ok) = o;
            // Implied nodes of a block may be continuous; trust them.
            return ok > 0 ||
                on is ScalarNode && on.kind == NodeKind.continuousState;
          });
          if (!continuous) {
            warn(
              '${name(n)} is binary without a continuous value parent, so '
              'its prediction never learns: add one above it.',
              nodes: [n.id],
            );
          }
        }
      case DpStateNode(:final batchSize):
        final clusters = [
          for (final a in valueParents[n.id] ?? const <String>[])
            if (byId[a] case final EfStateNode ef) ef,
        ];
        final count = clusters.fold(0, (s, c) => s + c.nNodes);
        if (count != batchSize) {
          error(
            '${name(n)} has $batchSize clusters (its batch size) but '
            '$count ef-state node${count == 1 ? '' : 's'} above it: add '
            'an ef-state group of ${batchSize - count > 0 ? batchSize - count : batchSize} '
            'with a value edge to it, or change the batch size.',
            nodes: [n.id, for (final c in clusters) c.id],
          );
        }
        for (final c in clusters) {
          if (c.dimension != 1 || c.learning != EfLearning.fixed) {
            error(
              '${name(c)}: a dp-state cluster is univariate with a fixed '
              'learning rate.',
              nodes: [c.id],
            );
          }
        }
      case EfStateNode() || CategoricalNode():
        break;
    }
  }

  // Overrides that address nothing.
  for (final o in doc.overrides) {
    for (final a in _overrideAddresses(o)) {
      if (resolve(a) == null) {
        error('A set value addresses "$a", which is not a node.');
      }
    }
  }

  if (issues.any((i) => i.isError)) {
    issues.sort((a, b) => a.severity.index - b.severity.index);
    return issues;
  }
  try {
    doc.build();
  } on Object catch (e) {
    error(switch (e) {
      ArgumentError(:final message) => '$message',
      StateError(:final message) => message,
      FormatException(:final message) => message,
      _ => '$e',
    });
  }
  issues.sort((a, b) => a.severity.index - b.severity.index);
  return issues;
}

/// The message of what building a network threw.
String buildErrorMessage(Object e) => switch (e) {
  ArgumentError(:final message) => '$message',
  StateError(:final message) => message,
  FormatException(:final message) => message,
  _ => '$e',
};
