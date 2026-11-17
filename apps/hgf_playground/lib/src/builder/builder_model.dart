import 'dart:convert';
import 'dart:math' as math;
import 'dart:ui' show Offset;

import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';

import '../model/data.dart';
import '../model/playground.dart';
import '../model/presets.dart';
import 'doc_layout.dart';

/// What is selected on the builder's canvas.
sealed class BuilderSelection {
  const BuilderSelection();
}

/// A node, by id.
final class NodeSelection extends BuilderSelection {
  const NodeSelection(this.id);
  final String id;

  @override
  bool operator ==(Object other) => other is NodeSelection && other.id == id;

  @override
  int get hashCode => id.hashCode;
}

/// An edge, by its index in [NetworkDocument.edges].
final class EdgeSelection extends BuilderSelection {
  const EdgeSelection(this.index);
  final int index;

  @override
  bool operator ==(Object other) =>
      other is EdgeSelection && other.index == index;

  @override
  int get hashCode => index.hashCode;
}

/// One undo step: the document and where its nodes are drawn.
typedef _Snapshot = ({NetworkDocument doc, Map<String, Offset> positions});

/// Whether an edge may be drawn, and why not.
typedef ConnectionCheck = ({DocEdge edge, String? problem});

/// The model builder's state: a [NetworkDocument] edited on a canvas, the
/// nodes' positions, the selection, undo/redo, the checks
/// ([checkNetwork]) and a live preview: the network built from the
/// document and filtered on simulated data after every edit.
class BuilderModel extends ChangeNotifier {
  BuilderModel({NetworkDocument? doc, String? name, this.coalesce = false}) {
    if (doc != null) {
      open(doc, name: name);
    } else {
      openPreset(presets.first);
    }
  }

  /// Refresh once per frame instead of on every edit (needs a running
  /// Flutter binding).
  final bool coalesce;
  bool _refreshScheduled = false;

  late NetworkDocument _doc;
  NetworkDocument get doc => _doc;

  Map<String, Offset> _positions = {};

  /// Each node's position on the canvas (y grows downwards; parents above).
  Map<String, Offset> get positions => _positions;

  /// What the document is called (a preset's name, a file's).
  String name = 'Untitled';

  /// Bumped when a new document was opened (the canvas fits it to view).
  int opened = 0;

  BuilderSelection? _selection;
  BuilderSelection? get selection => _selection;

  String? get selectedNode => switch (_selection) {
    NodeSelection(:final id) => id,
    _ => null,
  };

  int? get selectedEdge => switch (_selection) {
    EdgeSelection(:final index) => index,
    _ => null,
  };

  List<DocIssue> issues = const [];
  bool get hasErrors => issues.any((i) => i.isError);

  /// Issues per node id and edge index, worst first.
  Map<String, IssueSeverity> nodeIssues = const {};
  Map<int, IssueSeverity> edgeIssues = const {};

  final _undo = <_Snapshot>[], _redo = <_Snapshot>[];
  static const _maxHistory = 200;

  bool get canUndo => _undo.isNotEmpty;
  bool get canRedo => _redo.isNotEmpty;

  _Snapshot get _snapshot => (doc: _doc, positions: _positions);

  // ---------------------------------------------------------------------------
  // Opening
  // ---------------------------------------------------------------------------

  /// Open [doc] (laid out anew unless [positions] are given); clears the
  /// history.
  void open(
    NetworkDocument doc, {
    String? name,
    Map<String, Offset>? positions,
  }) {
    _doc = doc.ordered();
    this.name = name ?? 'Untitled';
    final auto = layeredLayout(_doc);
    _positions = {
      for (final n in _doc.nodes) n.id: positions?[n.id] ?? auto[n.id]!,
    };
    _selection = null;
    _undo.clear();
    _redo.clear();
    opened++;
    _refreshNow();
  }

  /// Open [network] (ids `x<index>`, as the playground's labels).
  void openNetwork(Network network, {String? name}) =>
      open(NetworkDocument.fromNetwork(network, id: (i) => 'x$i'), name: name);

  void openPreset(Preset preset) =>
      openNetwork(preset.build(), name: preset.name);

  /// Open a saved builder file: a model document (`hgf.document/1`, with
  /// the canvas positions in its layout), a bare network document, or a
  /// network's JSON (`Network.toJson`).
  void openJson(String text, {String? name}) {
    final json = jsonDecode(text);
    if (json is! Map<String, Object?>) {
      throw const FormatException('Not a JSON object');
    }
    if (json['format'] == ModelDocument.format) {
      final m = ModelDocument.fromJson(json);
      final layout = m.layout['nodes'];
      open(
        m.network,
        name: m.name ?? name,
        positions: layout is Map<String, Object?>
            ? {
                for (final MapEntry(:key, :value) in layout.entries)
                  if (value case [final num x, final num y, ...])
                    key: Offset(x.toDouble(), y.toDouble()),
              }
            : null,
      );
    } else if (json['nodes'] case [Map(containsKey: _hasId), ...]) {
      open(NetworkDocument.fromJson(json), name: name);
    } else {
      openNetwork(Network.fromJson(json), name: name);
    }
  }

  static bool _hasId(Object? key) => key == 'id';

  /// The document as a builder file: a model document whose layout holds
  /// the canvas positions.
  String toJsonText() {
    final m = ModelDocument(
      network: _doc,
      name: name,
      layout: {
        'nodes': {
          for (final MapEntry(:key, :value) in _positions.entries)
            key: [_round(value.dx), _round(value.dy)],
        },
      },
    );
    return const JsonEncoder.withIndent('  ').convert(m.toJson());
  }

  static double _round(double v) => (v * 10).roundToDouble() / 10;

  // ---------------------------------------------------------------------------
  // History
  // ---------------------------------------------------------------------------

  /// Record the current state as an undo step, before a continuous edit
  /// (a drag, a slider) whose updates pass `record: false`.
  void checkpoint() {
    _undo.add(_snapshot);
    if (_undo.length > _maxHistory) _undo.removeAt(0);
    _redo.clear();
    notifyListeners();
  }

  void undo() {
    if (_undo.isEmpty) return;
    _redo.add(_snapshot);
    _restore(_undo.removeLast());
  }

  void redo() {
    if (_redo.isEmpty) return;
    _undo.add(_snapshot);
    _restore(_redo.removeLast());
  }

  void _restore(_Snapshot s) {
    _doc = s.doc;
    _positions = s.positions;
    _selection = switch (_selection) {
      NodeSelection(:final id) when _doc.node(id) == null => null,
      EdgeSelection(:final index) when index >= _doc.edges.length => null,
      final s => s,
    };
    _refreshNow();
  }

  /// Make [doc] (and [positions]) current; [record] makes it an undo step.
  void _set(
    NetworkDocument doc, {
    Map<String, Offset>? positions,
    bool record = true,
    bool coalesced = false,
  }) {
    if (record) {
      _undo.add(_snapshot);
      if (_undo.length > _maxHistory) _undo.removeAt(0);
      _redo.clear();
    }
    _doc = doc;
    if (positions != null) _positions = positions;
    if (coalesced) {
      _requestRefresh();
    } else {
      _refreshNow();
    }
  }

  // ---------------------------------------------------------------------------
  // Selection and positions
  // ---------------------------------------------------------------------------

  void select(BuilderSelection? s) {
    if (s == _selection) return;
    _selection = s;
    notifyListeners();
  }

  /// Move node [id] to [to] (no undo step: call [checkpoint] when the drag
  /// starts).
  void moveNode(String id, Offset to) {
    _positions = {..._positions, id: to};
    notifyListeners();
  }

  /// Lay the network out again (an undo step).
  void autoLayout() {
    _set(_doc, positions: layeredLayout(_doc));
    opened++; // fit the view to it
    notifyListeners();
  }

  // ---------------------------------------------------------------------------
  // Structure
  // ---------------------------------------------------------------------------

  /// Add a node of [kind] at [at] (default: above the selected node, or
  /// next to the others) and select it. Returns its id.
  String addNode(NodeKind kind, {Offset? at}) {
    final id = _doc.freshId();
    final DocNode node = switch (kind) {
      NodeKind.continuousState ||
      NodeKind.binaryState ||
      NodeKind.constantState => ScalarNode(id, kind: kind),
      NodeKind.categoricalState => CategoricalNode(id, nCategories: 3),
      NodeKind.efState => EfStateNode(id),
      NodeKind.dpState => DpStateNode(id, batchSize: 4),
    };
    final near =
        at ??
        switch (selectedNode) {
          final s? => _positions[s]! + const Offset(0, -layoutDy),
          null => () {
            final b = boundsOf(_positions.values);
            return b == null
                ? Offset.zero
                : Offset(b.right + layoutDx, b.bottom);
          }(),
        };
    final spot = at ?? freeSpot(_positions.values, near);
    _selection = NodeSelection(id);
    _set(_doc.withNode(node), positions: {..._positions, id: spot});
    return id;
  }

  /// Add a continuous node above (as a [type] parent of) [child], or
  /// below it ([below]: [child] becomes its parent), with the edge, and
  /// select the new node. Returns its id.
  String addLinked(String address, EdgeType type, {bool below = false}) {
    final owner = ownerIdOf(address);
    final id = _doc.freshId();
    final at = freeSpot(
      _positions.values,
      _positions[owner]! + Offset(0, below ? layoutDy : -layoutDy),
    );
    final edge = below
        ? DocEdge(type, address, id)
        : DocEdge(type, id, address);
    _selection = NodeSelection(id);
    _set(
      _doc
          .withNode(ScalarNode(id, kind: NodeKind.continuousState))
          .withEdge(edge),
      positions: {..._positions, id: at},
    );
    return id;
  }

  /// Delete the selected node or edge.
  void deleteSelection() {
    switch (_selection) {
      case NodeSelection(:final id):
        removeNode(id);
      case EdgeSelection(:final index):
        removeEdge(index);
      case null:
    }
  }

  void removeNode(String id) {
    if (_doc.node(id) == null) return;
    _selection = null;
    _set(_doc.withoutNode(id), positions: {..._positions}..remove(id));
  }

  /// Replace node [id] by [node] (a new id renames it).
  void replaceNode(String id, DocNode node, {bool record = true}) {
    var pos = _positions;
    if (node.id != id) {
      pos = {
        for (final MapEntry(:key, :value) in _positions.entries)
          key == id ? node.id : key: value,
      };
      if (_selection case NodeSelection(id: final s) when s == id) {
        _selection = NodeSelection(node.id);
      }
    }
    _set(_doc.replaceNode(id, node), positions: pos, record: record);
  }

  /// Whether a [type] edge from [parent] to [child] (addresses) may be
  /// drawn: the problem the checks find with it, or null.
  ConnectionCheck checkConnection(String parent, String child, EdgeType type) {
    final edge = DocEdge(type, parent, child);
    if (parent == child) {
      return (edge: edge, problem: 'A node cannot be its own parent');
    }
    final i = _doc.edges.length;
    final next = _doc.withEdge(edge);
    for (final issue in checkNetwork(next, build: false)) {
      if (issue.isError && issue.edges.contains(i)) {
        return (edge: edge, problem: issue.message);
      }
    }
    return (edge: edge, problem: null);
  }

  /// Add [edge] and select it.
  void addEdge(DocEdge edge) {
    _selection = EdgeSelection(_doc.edges.length);
    _set(_doc.withEdge(edge));
  }

  void removeEdge(int index) {
    if (index >= _doc.edges.length) return;
    _selection = null;
    _set(_doc.withoutEdge(index));
  }

  void replaceEdge(int index, DocEdge edge, {bool record = true}) =>
      _set(_doc.replaceEdge(index, edge), record: record, coalesced: !record);

  /// Set a node's (or an implied node's) [field]; `record: false` during a
  /// slider drag (call [checkpoint] when it starts).
  void setParam(
    String address,
    NodeField field,
    double value, {
    bool record = true,
  }) => _set(
    _doc.withParam(address, field, value),
    record: record,
    coalesced: !record,
  );

  void setSettings({
    VolatilityUpdates? volatilityUpdates,
    bool? meanFieldUpdates,
    bool? ehgfMgfCorrection,
  }) => _set(
    _doc.copyWith(
      volatilityUpdates: volatilityUpdates,
      meanFieldUpdates: meanFieldUpdates,
      ehgfMgfCorrection: ehgfMgfCorrection,
    ),
  );

  // ---------------------------------------------------------------------------
  // Checks and preview
  // ---------------------------------------------------------------------------

  /// The network built from [doc] (null while it has errors).
  Network? network;

  /// The preview's trajectories (null while there is no network, or it
  /// could not be filtered).
  Trajectories? trajectories;

  /// Bumped whenever [trajectories] changed.
  int revision = 0;

  /// Why there is no preview (or it diverged).
  String? previewError;

  /// Wall time of the last build and filter, in microseconds.
  int previewMicros = 0;

  /// The simulated data's settings: one series per input, kind by kind
  /// (reversals for binary inputs, random walks for continuous ones, …),
  /// each with its own seed.
  final DataSettings data = DataSettings(nTrials: 300);

  /// The input table (`nTrials × nInputColumns`) and what generated it,
  /// per input node index.
  Float64List _input = Float64List(0);
  Map<int, List<Float64List>> hidden = const {};
  Object? _inputKey;

  /// The trials simulated: [DataSettings.nTrials], at most 150 with a
  /// dp-state node (20 000 proposals per trial: slow).
  int get trials => _doc.nodes.any((n) => n is DpStateNode)
      ? math.min(data.nTrials, 150)
      : data.nTrials;

  void setTrials(int n) {
    data.nTrials = n.clamp(10, 10000);
    _inputKey = null;
    _requestRefresh();
  }

  void reseed() {
    data.seed = math.Random().nextInt(1 << 30);
    _inputKey = null;
    _refreshNow();
  }

  void _requestRefresh() {
    if (!coalesce) {
      _refreshNow();
      return;
    }
    if (_refreshScheduled) return;
    _refreshScheduled = true;
    SchedulerBinding.instance
      ..scheduleFrameCallback((_) {
        _refreshScheduled = false;
        _refreshNow();
      })
      ..scheduleFrame();
  }

  void _refreshNow() {
    final sw = Stopwatch()..start();
    var found = checkNetwork(_doc, build: false);
    network = null;
    trajectories = null;
    previewError = null;
    if (found.any((i) => i.isError)) {
      previewError = 'Fix the errors to see the preview.';
    } else {
      try {
        network = _doc.build();
      } on Object catch (e) {
        found = [DocIssue(IssueSeverity.error, buildErrorMessage(e)), ...found];
        previewError = 'Fix the errors to see the preview.';
      }
    }
    if (network case final net?) {
      try {
        _ensureInput(net);
        trajectories = net.inputData(_input, fields: Playground.recordedFields);
        previewError = _divergence(net, trajectories!);
      } on Object catch (e) {
        trajectories = null;
        previewError = '$e';
      }
    }
    issues = found;
    final ni = <String, IssueSeverity>{};
    final ei = <int, IssueSeverity>{};
    for (final i in found) {
      for (final n in i.nodes) {
        ni[n] = _worse(ni[n], i.severity);
      }
      for (final e in i.edges) {
        ei[e] = _worse(ei[e], i.severity);
      }
    }
    nodeIssues = ni;
    edgeIssues = ei;
    previewMicros = sw.elapsedMicroseconds;
    revision++;
    notifyListeners();
  }

  static IssueSeverity _worse(IssueSeverity? a, IssueSeverity b) =>
      a == null || b.index < a.index ? b : a;

  /// Simulated inputs for [net]'s input nodes, made again only when the
  /// inputs or the data settings changed.
  void _ensureInput(Network net) {
    final a = _doc.addresses;
    final inputs = net.inputIdxs;
    final shape = [
      for (final i in inputs) (net.kindOf(i), _width(a.nodeAt(i), i, a)),
    ];
    final key = Object.hashAll([...shape, trials, data.seed]);
    if (key == _inputKey) return;
    final n = trials, width = net.nInputColumns;
    final table = Float64List(n * width);
    final hid = <int, List<Float64List>>{};
    var col = 0;
    for (final (k, i) in inputs.indexed) {
      final (kind, w) = shape[k];
      final s = DataSettings(nTrials: n, seed: data.seed + 7919 * k);
      List<Float64List> columns;
      switch (kind) {
        case NodeKind.binaryState:
          final sim = simulate(DataModel.reversal, s);
          columns = [sim.observations];
          hid[i] = sim.hidden;
        case NodeKind.categoricalState:
          final sim = simulate(
            DataModel.categoricalReversal,
            s,
            nCategories: w,
          );
          final hot = oneHot(sim.observations, w);
          columns = [
            for (var c = 0; c < w; c++)
              Float64List.fromList([
                for (var t = 0; t < n; t++) hot[t * w + c],
              ]),
          ];
          hid[i] = sim.hidden;
        case NodeKind.dpState:
          final sim = simulate(DataModel.mixture, s);
          columns = [sim.observations];
          hid[i] = sim.hidden;
        default:
          columns = [];
          for (var c = 0; c < w; c++) {
            final sim = simulate(
              DataModel.randomWalk,
              DataSettings(nTrials: n, seed: s.seed + 31 * c),
            );
            columns.add(sim.observations);
            if (c == 0) hid[i] = sim.hidden;
          }
      }
      for (var c = 0; c < w; c++) {
        for (var t = 0; t < n; t++) {
          table[t * width + col + c] = columns[c][t];
        }
      }
      col += w;
    }
    if (col != width) {
      throw StateError('Input columns: expected $width, made $col');
    }
    _input = table;
    hidden = hid;
    _inputKey = key;
  }

  static int _width(DocNode node, int index, NodeAddresses a) => switch (node) {
    CategoricalNode(:final nCategories) when a.firstOf(node.id) == index =>
      nCategories,
    EfStateNode(:final dimension) => dimension,
    _ => 1,
  };

  static String? _divergence(Network net, Trajectories traj) {
    for (var i = 0; i < traj.nNodes; i++) {
      if (!traj.fields.contains(NodeField.mean)) break;
      final m = traj.series(i, NodeField.mean);
      for (var t = 0; t < traj.nTime; t++) {
        final v = m[t];
        if (v.abs() <= 1e6) continue;
        return 'Node $i ${v.isFinite ? 'ran away' : 'became ${v.isNaN ? 'NaN' : 'infinite'}'} '
            'at trial $t. Try the standard volatility update, or a lower ω.';
      }
    }
    return null;
  }

  /// The last filtered mean of the node at [address], for its caption.
  double? lastMean(String address) {
    final traj = trajectories;
    if (traj == null || traj.nTime == 0) return null;
    try {
      final i = _doc.addresses.indexOf(address);
      if (i >= traj.nNodes) return null;
      final v = traj.at(i, NodeField.mean, traj.nTime - 1);
      return v.isFinite ? v : null;
    } on FormatException {
      return null;
    }
  }

  /// The initial value of [field] at [address]: from the built network,
  /// else from the node's own parameters or the kind's default.
  double initial(String address, NodeField field) {
    final net = network;
    if (net != null) {
      try {
        return net.getInitial(_doc.addresses.indexOf(address), field);
      } on Object {
        // fall through
      }
    }
    final node = _doc.node(address);
    if (node is ScalarNode) {
      return node.params[field] ?? defaultInitial(node.kind, field);
    }
    return double.nan;
  }

  static final _defaults = <NodeKind, Network>{};

  /// The initial value a fresh node of [kind] has for [field].
  static double defaultInitial(NodeKind kind, NodeField field) {
    final net = _defaults[kind] ??= () {
      final n = Network();
      if (kind == NodeKind.constantState) {
        n
          ..addNodes(kind: NodeKind.continuousState)
          ..addNodes(kind: kind, valueChildren: [0]);
      } else {
        n.addNodes(kind: kind);
      }
      return n..build();
    }();
    return net.getInitial(kind == NodeKind.constantState ? 1 : 0, field);
  }
}

/// A short name for a node kind.
String kindName(NodeKind kind) => switch (kind) {
  NodeKind.continuousState => 'continuous',
  NodeKind.binaryState => 'binary',
  NodeKind.constantState => 'constant (bias)',
  NodeKind.efState => 'exponential family',
  NodeKind.categoricalState => 'categorical',
  NodeKind.dpState => 'Dirichlet process',
};

/// The kind of a document node.
NodeKind docKind(DocNode n) => switch (n) {
  ScalarNode(:final kind) => kind,
  CategoricalNode() => NodeKind.categoricalState,
  EfStateNode() => NodeKind.efState,
  DpStateNode() => NodeKind.dpState,
};

/// What the [k]-th engine node of [node]'s block is (k ≥ 1), for menus.
String impliedNodeName(DocNode node, int k) {
  if (node case CategoricalNode(nCategories: final c)) {
    if (k <= c) return 'binary node, category ${k - 1}';
    if (k <= 2 * c) return 'value (logit) of category ${k - c - 1}';
    return 'shared volatility parent';
  }
  if (node case EfStateNode(:final nNodes)) {
    if (k < nNodes) return 'ef-state node ${k + 1} of $nNodes';
    return 'learning chain node ${k - nNodes + 1}';
  }
  return 'implied node $k';
}

/// `x12` → `x₁₂`; other ids as they are. Implied addresses keep their
/// suffix (`x0.7` → `x₀.7`).
String prettyId(String address) {
  final m = RegExp(r'^([a-zA-Z])(\d+)(\..*)?$').firstMatch(address);
  if (m == null) return address;
  const sub = '₀₁₂₃₄₅₆₇₈₉';
  final digits = m[2]!.split('').map((d) => sub[int.parse(d)]).join();
  return '${m[1]}$digits${m[3] ?? ''}';
}
