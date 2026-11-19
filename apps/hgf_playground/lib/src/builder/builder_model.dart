import 'dart:convert';
import 'dart:math' as math;
import 'dart:ui' show Offset;

import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_deep/hgf_deep.dart' show LayerKind;
import 'package:hgf_fit/hgf_fit.dart' show HgfModel, Parameter, Subject;

import '../model/data.dart';
import '../model/playground.dart';
import '../model/presets.dart';
import 'builder_fit.dart';
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

/// A layer of the open deep network, by index (0: the target layer).
final class LayerSelection extends BuilderSelection {
  const LayerSelection(this.index);
  final int index;

  @override
  bool operator ==(Object other) =>
      other is LayerSelection && other.index == index;

  @override
  int get hashCode => index.hashCode;
}

/// A part of the model's pipeline, by its [DocPart] key: the data, a
/// transform, an input (drawn as the wire from its columns), the network,
/// the response or an output.
final class PartSelection extends BuilderSelection {
  const PartSelection(this.part);
  final String part;

  @override
  bool operator ==(Object other) =>
      other is PartSelection && other.part == part;

  @override
  int get hashCode => part.hashCode;
}

/// Which canvas the builder shows: the model's pipeline (data, transforms,
/// the network as one node, the response and outputs), the HGF network
/// inside it, or one of its deep networks ([BuilderModel.openDeep]).
enum BuilderView { pipeline, network, deep }

/// One undo step: the document and where its nodes are drawn.
typedef _Snapshot = ({
  ModelDocument doc,
  Map<String, Offset> positions,
  Map<String, Offset> pipeline,
});

/// Whether an edge may be drawn, and why not.
typedef ConnectionCheck = ({DocEdge edge, String? problem});

/// The model builder's state: a [ModelDocument] edited on two canvases
/// (the pipeline, and the HGF network inside it), the nodes' positions, the
/// selection, undo/redo, the checks ([checkModel]) and a live preview: the
/// network built from the document and filtered after every edit, on the
/// loaded data's selected participant or on simulated data (with simulated
/// responses when there is a response).
class BuilderModel extends ChangeNotifier {
  BuilderModel({
    NetworkDocument? doc,
    ModelDocument? document,
    String? name,
    this.coalesce = false,
  }) {
    if (document != null) {
      open(document, name: name);
    } else if (doc != null) {
      openNetworkDocument(doc, name: name);
    } else {
      openPreset(presets.first);
    }
  }

  /// Refresh once per frame instead of on every edit (needs a running
  /// Flutter binding).
  final bool coalesce;
  bool _refreshScheduled = false;

  late ModelDocument _model;

  /// The whole model.
  ModelDocument get document => _model;

  /// The network part of [document] (what the network canvas edits).
  NetworkDocument get doc => _model.network;

  Map<String, Offset> _positions = {};

  /// Each node's position on the canvas (y grows downwards; parents above).
  Map<String, Offset> get positions => _positions;

  Map<String, Offset> _pipeline = {};

  /// Where each pipeline part ([DocPart] key) is drawn: the stored
  /// position, else a column per stage (data, transforms, network,
  /// response and outputs).
  Map<String, Offset> get pipelinePositions {
    final out = <String, Offset>{};
    for (final (k, part) in pipelineParts.indexed) {
      out[part] = _pipeline[part] ?? _autoPipeline(part, k);
    }
    return out;
  }

  /// The parts drawn on the pipeline canvas, in drawing order.
  List<String> get pipelineParts => [
    for (var i = 0; i < _model.data.steps.length; i++) DocPart.step(i),
    DocPart.data,
    for (final t in _model.transforms) DocPart.transform(t.name),
    if (_model.hasNetwork) DocPart.network,
    for (final d in _model.deep) DocPart.deep(d.id),
    if (_model.response != null) DocPart.response,
    for (final o in _model.outputs) DocPart.output(o.name),
    if (_model.hasNetwork) DocPart.fit,
  ];

  /// The automatic layout: a column per stage that the document has (data;
  /// transforms and deep networks; the HGF network; the response and
  /// outputs; the fit), 280 apart, so a missing stage leaves no gap.
  Offset _autoPipeline(String part, int k) {
    if (part == DocPart.data) return Offset.zero;
    if (part.startsWith('step:')) {
      // A chain into the data card, from the left.
      final i = int.parse(part.substring('step:'.length));
      return Offset(-240.0 * (_model.data.steps.length - i), 0);
    }
    final (:between, :network, :readouts) = _stages(_model);
    final hgf = 1 + (between ? 1 : 0);
    final readout = hgf + (network ? 1 : 0);
    final fit = readout + (readouts ? 1 : 0);
    Offset at(int column, double y) => Offset(280.0 * column, y);
    if (part == DocPart.network) return at(hgf, 0);
    if (part.startsWith('transform:')) {
      final i = _model.transforms.indexWhere(
        (t) => DocPart.transform(t.name) == part,
      );
      return at(1, i * 90.0);
    }
    if (part.startsWith('deep:')) {
      final i = _model.deep.indexWhere((d) => DocPart.deep(d.id) == part);
      // Between the data and the HGF network (its inputs may read the
      // predictions), under the transforms.
      final top = _model.transforms.isEmpty
          ? 0.0
          : _model.transforms.length * 90.0 + 20;
      return at(1, top + i * 220.0);
    }
    if (part == DocPart.response) return at(readout, 0);
    if (part == DocPart.fit) return at(fit, 0);
    final i = _model.outputs.indexWhere((o) => DocPart.output(o.name) == part);
    return at(readout, (_model.response == null ? 0 : 130) + i * 80.0);
  }

  /// Which of the automatic layout's middle columns [d] has.
  static ({bool between, bool network, bool readouts}) _stages(
    ModelDocument d,
  ) => (
    between: d.transforms.isNotEmpty || d.deep.isNotEmpty,
    network: d.hasNetwork,
    readouts: d.response != null || d.outputs.isNotEmpty,
  );

  /// What the document is called (a preset's name, a file's).
  String name = 'Untitled';

  void rename(String to) {
    if (to == name) return;
    name = to;
    notifyListeners();
  }

  /// Bumped when a new document was opened (the canvases fit it to view).
  int opened = 0;

  BuilderView _view = BuilderView.pipeline;
  BuilderView get view => _view;

  /// Show [v]; the selection is per canvas, so it is cleared.
  void showView(BuilderView v) {
    if (v == _view) return;
    _view = v;
    _selection = null;
    notifyListeners();
  }

  String? _deepId;

  /// The id of the deep network the deep canvas shows.
  String? get openDeep => _deepId;

  /// The deep network the deep canvas shows, if it still exists.
  DocDeep? get deep => _deepId == null ? null : _model.deepById(_deepId!);

  /// Show deep network [id] on its canvas.
  void showDeep(String id) {
    if (_model.deepById(id) == null) return;
    _deepId = id;
    _view = BuilderView.deep;
    _selection = null;
    notifyListeners();
  }

  /// Show the HGF network's canvas (adding a network first if there is
  /// none).
  void showNetwork() {
    if (!_model.hasNetwork) addNetwork();
    showView(BuilderView.network);
  }

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

  int? get selectedLayer => switch (_selection) {
    LayerSelection(:final index) => index,
    _ => null,
  };

  String? get selectedPart => switch (_selection) {
    PartSelection(:final part) => part,
    _ => null,
  };

  List<DocIssue> issues = const [];
  bool get hasErrors => issues.any((i) => i.isError);

  /// Issues per node id, edge index and pipeline part, worst first.
  Map<String, IssueSeverity> nodeIssues = const {};
  Map<int, IssueSeverity> edgeIssues = const {};
  Map<String, IssueSeverity> partIssues = const {};

  final _undo = <_Snapshot>[], _redo = <_Snapshot>[];
  static const _maxHistory = 200;

  bool get canUndo => _undo.isNotEmpty;
  bool get canRedo => _redo.isNotEmpty;

  _Snapshot get _snapshot =>
      (doc: _model, positions: _positions, pipeline: _pipeline);

  // ---------------------------------------------------------------------------
  // Opening
  // ---------------------------------------------------------------------------

  /// Open [document] (laid out anew unless [positions] / [pipeline] are
  /// given); clears the history and shows the pipeline.
  void open(
    ModelDocument document, {
    String? name,
    Map<String, Offset>? positions,
    Map<String, Offset>? pipeline,
  }) {
    _model = document.copyWith(network: document.network.ordered());
    this.name = name ?? document.name ?? 'Untitled';
    final auto = layeredLayout(_model.network);
    _positions = {
      for (final n in _model.network.nodes)
        n.id: positions?[n.id] ?? auto[n.id]!,
    };
    _pipeline = {...?pipeline};
    _selection = null;
    _view = BuilderView.pipeline;
    _undo.clear();
    _redo.clear();
    _participant = 0;
    opened++;
    _refreshNow();
  }

  /// Open a network alone (no response, default input names).
  void openNetworkDocument(
    NetworkDocument doc, {
    String? name,
    Map<String, Offset>? positions,
  }) {
    Network? net;
    try {
      net = doc.build();
    } on Object {
      net = null;
    }
    open(
      ModelDocument(
        network: doc,
        inputs: net == null ? const [] : ModelDocument.defaultInputs(doc, net),
      ),
      name: name,
      positions: positions,
    );
  }

  /// Open [network] (ids `x<index>`, as the playground's labels), with the
  /// inputs the code exports name (`observation`, `input_<i>`).
  void openNetwork(Network network, {String? name}) => open(
    ModelDocument.fromNetwork(network, id: (i) => 'x$i', name: name),
    name: name,
  );

  /// Open [preset]: its network, and for a choice task its response
  /// (a softmax over the options, which reveal their outcomes).
  void openPreset(Preset preset) {
    final net = preset.build();
    final task = preset.task;
    if (task == null) {
      openNetwork(net, name: preset.name);
      return;
    }
    final m = ModelDocument.fromModel(
      HgfModel(
        network: net,
        response: task.readout(task.beta),
        parameters: const <Parameter>[],
      ),
      id: (i) => 'x$i',
      name: preset.name,
    );
    final inputNames = [for (final i in m.inputs) i.name];
    final labels = [
      for (final o in task.options) o.toLowerCase().replaceAll(' ', '_'),
    ];
    open(
      m.copyWith(
        response: DocResponse(
          'choice',
          readout: m.response!.readout,
          options: labels,
          reveals: {
            for (final (k, o) in labels.indexed)
              o: [for (final i in task.reveals[k]) inputNames[i]],
          },
        ),
      ),
      name: preset.name,
    );
  }

  /// Open a saved builder file: a model document (`hgf.document/1`, with
  /// the canvas positions in its layout), a bare network document, or a
  /// network's JSON (`Network.toJson`).
  void openJson(String text, {String? name}) {
    final json = jsonDecode(text);
    if (json is! Map<String, Object?>) {
      throw const FormatException('Not a JSON object');
    }
    Map<String, Offset>? offsets(Object? layout) =>
        layout is Map<String, Object?>
        ? {
            for (final MapEntry(:key, :value) in layout.entries)
              if (value case [final num x, final num y, ...])
                key: Offset(x.toDouble(), y.toDouble()),
          }
        : null;
    if (json['format'] == ModelDocument.format) {
      final m = ModelDocument.fromJson(json);
      open(
        m.copyWith(layout: const {}),
        name: m.name ?? name,
        positions: offsets(m.layout['nodes']),
        pipeline: offsets(m.layout['pipeline']),
      );
    } else if (json['nodes'] case [Map(containsKey: _hasId), ...]) {
      openNetworkDocument(NetworkDocument.fromJson(json), name: name);
    } else {
      openNetwork(Network.fromJson(json), name: name);
    }
  }

  static bool _hasId(Object? key) => key == 'id';

  /// The document as a builder file: the model document, whose layout
  /// holds both canvases' positions.
  String toJsonText() {
    List<double> xy(Offset o) => [_round(o.dx), _round(o.dy)];
    final m = _model.copyWith(
      name: name,
      layout: {
        'nodes': {
          for (final MapEntry(:key, :value) in _positions.entries)
            key: xy(value),
        },
        if (_pipeline.isNotEmpty)
          'pipeline': {
            for (final MapEntry(:key, :value) in _pipeline.entries)
              key: xy(value),
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
    _model = s.doc;
    _positions = s.positions;
    _pipeline = s.pipeline;
    _selection = switch (_selection) {
      NodeSelection(:final id) when doc.node(id) == null => null,
      EdgeSelection(:final index) when index >= doc.edges.length => null,
      PartSelection(:final part) when !_partExists(part) => null,
      final s => s,
    };
    _refreshNow();
  }

  bool _partExists(String part) =>
      pipelineParts.contains(part) ||
      _model.inputs.any((i) => DocPart.input(i.name) == part);

  /// Make [next] the network; [renamedIds] are node renames the rest of the
  /// document follows. [record] makes it an undo step.
  void _set(
    NetworkDocument next, {
    Map<String, Offset>? positions,
    Map<String, String> renamedIds = const {},
    bool record = true,
    bool coalesced = false,
  }) => _setModel(
    _model.withNetwork(next, renamedIds: renamedIds),
    positions: positions,
    record: record,
    coalesced: coalesced,
  );

  /// Make [next] the document; [record] makes it an undo step.
  void _setModel(
    ModelDocument next, {
    Map<String, Offset>? positions,
    Map<String, Offset>? pipeline,
    bool record = true,
    bool coalesced = false,
  }) {
    if (record) {
      _undo.add(_snapshot);
      if (_undo.length > _maxHistory) _undo.removeAt(0);
      _redo.clear();
    }
    // A column of the automatic layout appears or goes: the cards there
    // stay where they were drawn (the steps chain into the data card).
    final pinned = _stages(_model) != _stages(next) ? pipelinePositions : null;
    _model = next;
    if (positions != null) _positions = positions;
    if (pipeline != null) _pipeline = pipeline;
    if (pinned != null) {
      final parts = pipelineParts.toSet();
      _pipeline = {
        for (final MapEntry(:key, :value) in pinned.entries)
          if (parts.contains(key) && !key.startsWith('step:')) key: value,
        ..._pipeline,
      };
    }
    if (coalesced) {
      _requestRefresh();
    } else {
      _refreshNow();
    }
  }

  /// Make [next] the document (an edit of a pipeline part); `record: false`
  /// while typing or dragging (call [checkpoint] first).
  void update(ModelDocument next, {bool record = true}) =>
      _setModel(next, record: record, coalesced: !record);

  // ---------------------------------------------------------------------------
  // Selection and positions
  // ---------------------------------------------------------------------------

  void select(BuilderSelection? s) {
    if (s == _selection) return;
    _selection = s;
    notifyListeners();
  }

  final _layout = _Layout();

  /// Notified when a node or a card moves ([moveNode], [movePart]): only
  /// the canvases draw positions, so a drag repaints them without
  /// rebuilding the inspector and the preview on every pointer move. The
  /// canvases listen to this and to the model.
  Listenable get layoutChanges => _layout;

  /// Move node [id] to [to] (no undo step: call [checkpoint] when the drag
  /// starts). Notifies [layoutChanges] only.
  void moveNode(String id, Offset to) {
    _positions = {..._positions, id: to};
    _layout.changed();
  }

  /// Lay the network out again (an undo step).
  void autoLayout() {
    _set(doc, positions: layeredLayout(doc));
    opened++; // fit the view to it
    notifyListeners();
  }

  // ---------------------------------------------------------------------------
  // Structure
  // ---------------------------------------------------------------------------

  /// Add a node of [kind] at [at] (default: above the selected node, or
  /// next to the others) and select it. Returns its id.
  String addNode(NodeKind kind, {Offset? at}) {
    final id = doc.freshId();
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
    _set(doc.withNode(node), positions: {..._positions, id: spot});
    return id;
  }

  /// Add a continuous node above (as a [type] parent of) [child], or
  /// below it ([below]: [child] becomes its parent), with the edge, and
  /// select the new node. Returns its id.
  String addLinked(String address, EdgeType type, {bool below = false}) {
    final owner = ownerIdOf(address);
    final id = doc.freshId();
    final at = freeSpot(
      _positions.values,
      _positions[owner]! + Offset(0, below ? layoutDy : -layoutDy),
    );
    final edge = below
        ? DocEdge(type, address, id)
        : DocEdge(type, id, address);
    _selection = NodeSelection(id);
    _set(
      doc
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
      case PartSelection(:final part):
        removePart(part);
      case LayerSelection(:final index):
        removeLayer(index);
      case null:
    }
  }

  void removeNode(String id) {
    if (doc.node(id) == null) return;
    _selection = null;
    _set(doc.withoutNode(id), positions: {..._positions}..remove(id));
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
    _set(
      doc.replaceNode(id, node),
      positions: pos,
      renamedIds: node.id != id ? {id: node.id} : const {},
      record: record,
    );
  }

  /// Whether a [type] edge from [parent] to [child] (addresses) may be
  /// drawn: the problem the checks find with it, or null.
  ConnectionCheck checkConnection(String parent, String child, EdgeType type) {
    final edge = DocEdge(type, parent, child);
    if (parent == child) {
      return (edge: edge, problem: 'A node cannot be its own parent');
    }
    final i = doc.edges.length;
    final next = doc.withEdge(edge);
    for (final issue in checkNetwork(next, build: false)) {
      if (issue.isError && issue.edges.contains(i)) {
        return (edge: edge, problem: issue.message);
      }
    }
    return (edge: edge, problem: null);
  }

  /// Add [edge] and select it.
  void addEdge(DocEdge edge) {
    _selection = EdgeSelection(doc.edges.length);
    _set(doc.withEdge(edge));
  }

  void removeEdge(int index) {
    if (index >= doc.edges.length) return;
    _selection = null;
    _set(doc.withoutEdge(index));
  }

  void replaceEdge(int index, DocEdge edge, {bool record = true}) =>
      _set(doc.replaceEdge(index, edge), record: record, coalesced: !record);

  /// Set a node's (or an implied node's) [field]; `record: false` during a
  /// slider drag (call [checkpoint] when it starts).
  void setParam(
    String address,
    NodeField field,
    double value, {
    bool record = true,
  }) => _set(
    doc.withParam(address, field, value),
    record: record,
    coalesced: !record,
  );

  void setSettings({
    VolatilityUpdates? volatilityUpdates,
    bool? meanFieldUpdates,
  }) => _set(
    doc.copyWith(
      volatilityUpdates: volatilityUpdates,
      meanFieldUpdates: meanFieldUpdates,
    ),
  );

  // ---------------------------------------------------------------------------
  // Free parameters
  // ---------------------------------------------------------------------------

  /// Whether a free parameter sets [target].
  bool isFree(DocTarget target) => _model.parameterFor(target) != null;

  /// Free [target] (with a prior around [current]; see
  /// `ModelDocumentParameters.freeTarget`) or fix it again.
  void setFree(DocTarget target, bool free, {required double current}) =>
      _setModel(
        free ? _model.freeTarget(target, current) : _model.fixTarget(target),
      );

  /// The response model's parameters and their values (none without a
  /// compiled response).
  Map<String, double> get responseParameters =>
      compiled?.model?.response.parameters ?? const {};

  // ---------------------------------------------------------------------------
  // Pipeline
  // ---------------------------------------------------------------------------

  /// Move pipeline part [part] to [to] (no undo step: call [checkpoint]
  /// when the drag starts). Notifies [layoutChanges] only.
  void movePart(String part, Offset to) {
    _pipeline = {..._pipeline, part: to};
    _layout.changed();
  }

  /// Lay the pipeline out again (an undo step).
  void autoLayoutPipeline() {
    _setModel(_model, pipeline: const {});
    opened++;
    notifyListeners();
  }

  /// Add an HGF network (a continuous input with a volatility parent)
  /// when the document has none, and select its card.
  void addNetwork({Offset? at}) {
    if (_model.hasNetwork) return;
    final start = ModelDocument.fromNetwork(
      Network()
        ..addNodes()
        ..addNodes(volatilityChildren: [0]),
    );
    _positions = layeredLayout(start.network);
    _selection = const PartSelection(DocPart.network);
    _setModel(
      _model.copyWith(network: start.network, inputs: start.inputs),
      pipeline: {..._pipeline, DocPart.network: ?at},
    );
  }

  /// Add a deep network (a small classifier to start from) at [at], and
  /// select it; returns its id.
  String addDeep({Offset? at}) {
    final ids = {for (final d in _model.deep) d.id};
    var id = 'deep';
    for (var k = 2; ids.contains(id); k++) {
      id = 'deep$k';
    }
    final part = DocPart.deep(id);
    _selection = PartSelection(part);
    _setModel(
      _model.copyWith(deep: [..._model.deep, DocDeep.starter(id)]),
      pipeline: {..._pipeline, part: ?at},
    );
    return id;
  }

  /// Replace deep network [id] by [next] (an undo step unless [record] is
  /// false: call [checkpoint] first while typing). A new id keeps its card
  /// where it was.
  void setDeep(String id, DocDeep next, {bool record = true}) {
    if (next.id != id && _model.deepById(next.id) != null) return;
    final pipeline = {..._pipeline};
    if (next.id != id) {
      if (pipeline.remove(DocPart.deep(id)) case final at?) {
        pipeline[DocPart.deep(next.id)] = at;
      }
    }
    if (_deepId == id) _deepId = next.id;
    if (_selection == PartSelection(DocPart.deep(id))) {
      _selection = PartSelection(DocPart.deep(next.id));
    }
    _setModel(
      _model.copyWith(
        deep: [for (final d in _model.deep) d.id == id ? next : d],
      ),
      pipeline: pipeline,
      record: record,
      coalesced: !record,
    );
  }

  /// Replace layer [index] of the open deep network.
  void setLayer(int index, DocLayer layer, {bool record = true}) {
    final d = deep;
    if (d == null || index < 0 || index >= d.layers.length) return;
    final layers = [...d.layers];
    layers[index] = layer;
    setDeep(d.id, syncDeepColumns(d.copyWith(layers: layers)), record: record);
  }

  /// Insert a hidden layer above layer [below] of the open deep network
  /// (a copy of its nearest hidden layer, else 8 tanh nodes), and select
  /// it. In a continuous network the new level (as wide as [below]) may
  /// also go on top, and slots into the chain: it becomes the value parent
  /// of [below] and takes over [below]'s value parent
  /// ([DocDeep.insertLevel]).
  void insertLayer(int below) {
    final d = deep;
    if (d == null || below < 0) return;
    final n = d.layers.length;
    if (d.isContinuous) {
      if (below >= n) return;
      _selection = LayerSelection(below + 1);
      final level = DocLayer(
        size: d.layers[below].size,
        kind: LayerKind.continuous,
      );
      setDeep(d.id, d.insertLevel(below + 1, level));
      return;
    }
    if (below >= n - 1) return;
    final like = d.layers.skip(1).take(n - 2).firstOrNull;
    final layer =
        like?.copyWith(name: () => null) ??
        const DocLayer(size: 8, coupling: 'tanh');
    _selection = LayerSelection(below + 1);
    setDeep(d.id, d.copyWith(layers: [...d.layers]..insert(below + 1, layer)));
  }

  /// Whether layer [index] of the open deep network can be removed: a
  /// hidden layer (the target and feature layers stay), or in a
  /// continuous network any level above the observations while two
  /// layers remain.
  bool canRemoveLayer(int index) {
    final d = deep;
    if (d == null || index <= 0) return false;
    final n = d.layers.length;
    return d.isContinuous ? index < n && n > 2 : index < n - 1;
  }

  /// Remove layer [index] of the open deep network, if [canRemoveLayer]
  /// (a continuous network rewires around it: [DocDeep.removeLevel]).
  void removeLayer(int index) {
    final d = deep;
    if (d == null || !canRemoveLayer(index)) return;
    _selection = null;
    setDeep(
      d.id,
      d.isContinuous
          ? d.removeLevel(index)
          : d.copyWith(layers: [...d.layers]..removeAt(index)),
    );
  }

  /// Deep network [id] reading [column] as a feature (or, with [target], a
  /// target): the layer grows to match. A text column read as a target
  /// becomes a categorical target over its labels.
  void bindDeepColumn(String id, String column, {required bool target}) {
    final d = _model.deepById(id);
    if (d == null) return;
    if (!target) {
      if (d.features.contains(column)) return;
      setDeep(
        id,
        syncDeepColumns(d.copyWith(features: [...d.features, column])),
      );
      return;
    }
    final labels = _labelsOf(column);
    if (labels != null) {
      setDeep(
        id,
        syncDeepColumns(
          d.copyWith(targets: [column], categories: () => labels),
        ),
      );
      return;
    }
    if (d.targets.contains(column)) return;
    setDeep(
      id,
      syncDeepColumns(
        d.copyWith(
          targets: d.categories == null ? [...d.targets, column] : [column],
          categories: () => null,
        ),
      ),
    );
  }

  /// Deep network [id] without feature (or target) [column].
  void unbindDeepColumn(String id, String column, {required bool target}) {
    final d = _model.deepById(id);
    if (d == null) return;
    setDeep(
      id,
      syncDeepColumns(
        target
            ? d.copyWith(
                targets: [
                  for (final c in d.targets)
                    if (c != column) c,
                ],
                categories: () => null,
              )
            : d.copyWith(
                features: [
                  for (final c in d.features)
                    if (c != column) c,
                ],
              ),
      ),
    );
  }

  /// The labels of [column] in the loaded data when it holds text (in
  /// order of first appearance), else null.
  List<String>? _labelsOf(String column) {
    final data = csv;
    if (data == null) return null;
    final c = data.header.indexOf(column);
    if (c < 0 || data.firstText(c) == null) return null;
    final seen = <String>{};
    for (final row in data.rows) {
      final v = row[c].trim();
      if (v.isNotEmpty) seen.add(v);
      if (seen.length > 64) return null;
    }
    return seen.toList();
  }

  /// [d] with its feature and target layers sized to its columns: a
  /// feature layer as wide as its features; a target layer as wide as its
  /// targets, or categorical over its categories.
  static DocDeep syncDeepColumns(DocDeep d) {
    if (d.layers.length < 2 || d.isContinuous) {
      if (d.isContinuous && d.targets.isNotEmpty) {
        final layers = [...d.layers];
        layers[0] = layers[0].copyWith(size: d.targets.length);
        return d.copyWith(layers: layers);
      }
      return d;
    }
    final layers = [...d.layers];
    if (d.features.isNotEmpty) {
      layers[layers.length - 1] = layers.last.copyWith(size: d.features.length);
    }
    final t = layers.first;
    if (d.categories case final cats?) {
      layers[0] = t.copyWith(size: cats.length, kind: LayerKind.categorical);
    } else if (d.targets.isNotEmpty) {
      layers[0] = t.copyWith(size: d.targets.length);
    }
    return d.copyWith(layers: layers);
  }

  /// Add a transform reading [column] (or `0`) at [at], and select it.
  String addTransform({String? column, Offset? at}) {
    final base = column == null
        ? 'derived'
        : '${column.replaceAll(RegExp('[^A-Za-z0-9_]'), '_')}_2';
    final name = _model.freshName(base, csv?.header ?? const []);
    final source = column == null
        ? '0'
        : isPlainName(column)
        ? column
        : '`$column`';
    final part = DocPart.transform(name);
    _selection = PartSelection(part);
    _setModel(
      _model.withTransform(DocTransform(name, source)),
      pipeline: {..._pipeline, part: ?at},
    );
    return name;
  }

  /// The data-pipeline step kinds, as the builder names them.
  static const stepKinds = {
    'decimate': 'Decimate',
    'segment': 'Segment by events',
    'filter': 'Filter rows',
    'standardise': 'Standardise',
    'split': 'Split train / test',
  };

  /// The columns step [index] reads: the data's (on the clock) and those
  /// the steps before it add.
  List<String> columnsBeforeStep(int index) {
    final c = csv;
    if (c == null) return const [];
    final d = _model.data;
    return stepColumns(
      d.clockColumnsOf(c.header),
      d.steps.sublist(0, math.min(index, d.steps.length)),
    );
  }

  /// A new step of [kind] with defaults that keep the preview running.
  DocStep newStep(String kind, int index) {
    final d = _model.data;
    final cols = columnsBeforeStep(index);
    final roles = {
      d.participantColumn,
      d.timeColumn,
      d.timeStepColumn,
      d.groupColumn,
      d.clock?.sessionColumn,
    };
    final plain = [
      for (final c in cols)
        if (!roles.contains(c)) c,
    ];
    String q(String c) => isPlainName(c) ? c : '`$c`';
    final marker = const [
      'trial',
      'round',
      'block',
      'segment',
    ].where(plain.contains).firstOrNull;
    final first = marker ?? plain.firstOrNull;
    return switch (kind) {
      'decimate' => DecimateStep(factor: 2),
      'segment' => SegmentStep(
        at: Expr.parse(first == null ? '1' : 'diff(${q(first)}) != 0'),
        column: _model.freshName('segment', cols),
      ),
      'filter' => FilterStep(
        Expr.parse(first == null ? '1' : 'missing(${q(first)}) == 0'),
      ),
      'standardise' => StandardiseStep(columns: const []),
      'split' => SplitStep(column: _model.freshName('split', cols)),
      _ => throw ArgumentError.value(kind, 'kind'),
    };
  }

  /// Add a data-pipeline step of [kind] at [index] (default: last), and
  /// select it.
  void addStep(String kind, {int? index}) {
    final steps = [..._model.data.steps];
    final at = index ?? steps.length;
    steps.insert(at, newStep(kind, at));
    _selection = PartSelection(DocPart.step(at));
    _setModel(
      _model.copyWith(data: _model.data.copyWith(steps: steps)),
      pipeline: _withoutStepPositions(),
    );
  }

  /// Replace step [index] with [step].
  void replaceStep(int index, DocStep step) {
    final steps = [..._model.data.steps]..[index] = step;
    update(_model.copyWith(data: _model.data.copyWith(steps: steps)));
  }

  /// Move step [index] by [by] places (−1: earlier), keeping it selected.
  void moveStep(int index, int by) {
    final steps = [..._model.data.steps];
    final to = index + by;
    if (to < 0 || to >= steps.length) return;
    steps.insert(to, steps.removeAt(index));
    _selection = PartSelection(DocPart.step(to));
    _setModel(
      _model.copyWith(data: _model.data.copyWith(steps: steps)),
      pipeline: _withoutStepPositions(),
    );
  }

  /// The pipeline positions without the steps' (they are laid out again
  /// when steps come, go or move: they are addressed by index).
  Map<String, Offset> _withoutStepPositions() => {
    for (final MapEntry(:key, :value) in _pipeline.entries)
      if (!key.startsWith('step:')) key: value,
  };

  /// The loaded data on the clock and through the steps, and how many
  /// rows each step gave; null without data (or when the clock or a step
  /// fails: the checks say why).
  ({CsvData table, List<int> stepRows})? get prepared {
    final c = csv;
    if (c == null) return null;
    final d = _model.data;
    try {
      final t = DataLayout(
        inputs: const [],
        participantColumn: d.participantColumn,
        timeColumn: d.timeColumn,
        timeStepColumn: d.timeStepColumn,
        groupColumn: d.groupColumn,
        clock: d.clock,
        steps: d.steps,
      ).prepared(c, cache: _dataCache);
      return (
        table: t,
        stepRows: d.steps.isEmpty ? const [] : _dataCache.stepRows,
      );
    } on Object {
      return null;
    }
  }

  /// Add a trajectory output (a node's mean, or [node]'s [field]) at [at],
  /// and select it.
  String addOutput({String? node, String field = 'mean', Offset? at}) {
    final target = node ?? (doc.nodes.isEmpty ? 'x0' : doc.nodes.first.id);
    final name = _model.freshName('${target.replaceAll('.', '_')}_$field');
    final part = DocPart.output(name);
    _selection = PartSelection(part);
    _setModel(
      _model.withOutput(DocOutput(name, node: target, field: field)),
      pipeline: {..._pipeline, part: ?at},
    );
    return name;
  }

  /// Add a response that suits the first input: a softmax over a binary or
  /// categorical input's prediction, else a Gaussian readout of it.
  void addResponse({Offset? at}) {
    if (_model.response != null) return;
    final first = _model.inputs.isEmpty ? null : _model.inputs.first;
    final node = first?.node ?? (doc.nodes.isEmpty ? 'x0' : doc.nodes.first.id);
    final kind = switch (doc.node(ownerIdOf(node))) {
      final n? => docKind(n),
      null => NodeKind.continuousState,
    };
    final readout = switch (kind) {
      NodeKind.binaryState => <String, Object?>{
        'type': 'BinarySoftmaxReadout',
        'node': node,
        'inverseTemperature': 1.0,
      },
      NodeKind.categoricalState => {
        'type': 'CategoricalSoftmaxReadout',
        'node': node,
        'inverseTemperature': 1.0,
      },
      _ => {
        'type': 'GaussianReadout',
        'node': node,
        'field': 'expected_mean',
        'precision': 1.0,
      },
    };
    _selection = const PartSelection(DocPart.response);
    _setModel(
      _model.withResponse(
        DocResponse(_model.freshName('response'), readout: readout),
      ),
      pipeline: {..._pipeline, DocPart.response: ?at},
    );
  }

  /// Remove a transform, an output or the response; an input goes back to
  /// its default column (its name).
  void removePart(String part) {
    final colon = part.indexOf(':');
    final kind = colon < 0 ? part : part.substring(0, colon);
    final key = colon < 0 ? '' : part.substring(colon + 1);
    final next = switch (kind) {
      'transform' => _model.withoutTransform(key),
      'step' => _model.copyWith(
        data: _model.data.copyWith(
          steps: [..._model.data.steps]..removeAt(int.parse(key)),
        ),
      ),
      'output' => _model.withoutOutput(key),
      DocPart.response => _model.withResponse(null),
      'deep' => _model.copyWith(
        deep: [
          for (final d in _model.deep)
            if (d.id != key) d,
        ],
      ),
      DocPart.network => _model.copyWith(
        network: NetworkDocument(nodes: const []),
        inputs: const [],
        clearResponse: true,
        outputs: const [],
        parameters: const [],
        group: DocGroup.empty,
      ),
      'input' => switch (_model.input(key)) {
        final i? => _model.replaceInput(
          key,
          DocInput(i.name, node: i.node, categories: i.categories),
        ),
        null => null,
      },
      _ => null,
    };
    if (next == null) return;
    _selection = null;
    _setModel(
      next,
      pipeline: kind == 'step'
          ? _withoutStepPositions()
          : ({..._pipeline}..remove(part)),
    );
  }

  /// Read input [inputName] from [column].
  void bindColumn(String column, String inputName) {
    final i = _model.input(inputName);
    if (i == null) return;
    _selection = PartSelection(DocPart.input(inputName));
    _setModel(
      _model.replaceInput(
        inputName,
        DocInput(
          i.name,
          node: i.node,
          columns: [column],
          categories: i.categories,
          observedColumn: i.observedColumn,
        ),
      ),
    );
  }

  /// Read the recorded responses from [column].
  void bindResponseColumn(String column) {
    final r = _model.response;
    if (r == null) return;
    _selection = const PartSelection(DocPart.response);
    _setModel(
      _model.withResponse(
        DocResponse(
          r.name,
          readout: r.readout,
          columns: [column],
          options: r.options,
          reveals: r.reveals,
        ),
      ),
    );
  }

  /// The columns the pipeline can read: the loaded data's, or (simulated)
  /// the columns the document reads from its data.
  List<String> get dataColumns {
    if (csv case final c?) return _model.data.columnsOf(c.header);
    final derived = _model.transformNames;
    final out = <String>{};
    void add(String? c) {
      if (c != null && !derived.contains(c)) out.add(c);
    }

    final d = _model.data;
    add(d.participantColumn);
    add(d.timeColumn);
    add(d.timeStepColumn);
    add(d.groupColumn);
    final net = network;
    for (final i in _model.inputs) {
      var w = 1;
      if (net != null) {
        try {
          w = inputWidth(net, doc, doc.addresses.indexOf(i.node));
        } on Object {
          w = 1;
        }
      }
      i.columnsFor(w).forEach(add);
      add(i.observedColumn);
    }
    for (final t in _model.transforms) {
      t.expression.variables.forEach(add);
    }
    _model.response?.dataColumns.forEach(add);
    return out.toList();
  }

  // ---------------------------------------------------------------------------
  // Deep networks: trained on the loaded data, in the background
  // ---------------------------------------------------------------------------

  /// How long an edit may train deep networks before the preview shows
  /// (small ones then finish at once); the rest trains between frames.
  int deepBudgetMs = 15;

  /// Trainings under way, by [DeepTraining.key].
  final _deepRuns = <String, DeepTraining>{};
  Map<String, DeepTraining> _deepRunById = const {};
  Map<String, DeepResult> _deepDone = const {};
  Map<String, String> _deepErrors = const {};
  Set<String> _deepWaiting = const {};
  bool _pumping = false, _disposed = false;

  /// Deep network [id] trained on the loaded data (as the exports train
  /// it), once it is.
  DeepResult? deepResult(String id) => _deepDone[id];

  /// Deep network [id]'s training, while it runs.
  DeepTraining? deepRun(String id) => _deepRunById[id];

  /// Why deep network [id] could not be trained.
  String? deepError(String id) => _deepErrors[id];

  /// Whether deep network [id] waits for another's predictions.
  bool deepWaiting(String id) => _deepWaiting.contains(id);

  /// Deep network [id]'s training in a few words (its card and panel).
  String deepStatus(String id) {
    final d = _model.deepById(id);
    if (d == null) return '';
    if (csv == null) return 'load data to train';
    if (_deepErrors[id] case final e?) return e;
    if (_deepRunById[id] case final r?) {
      final pct = (r.progress * 100).floor();
      return d.isContinuous
          ? 'filtering… $pct%'
          : 'training… $pct% · pass ${math.max(1, r.epoch)} of '
                '${d.training.epochs}';
    }
    if (_deepWaiting.contains(id)) return 'waiting for the networks it reads';
    final r = _deepDone[id];
    if (r == null) {
      return partIssues[DocPart.deep(id)] == IssueSeverity.error
          ? 'fix the errors to train'
          : 'bind its columns to train';
    }
    if (d.isContinuous) return 'filtered ${r.data.length} rows';
    String f(DeepScore s) =>
        '${s.name} ${s.value.toStringAsFixed(s.name == 'R²' ? 3 : 2)}';
    return [
      'trained on ${r.data.trainIndices.length} rows',
      if (r.trainScore case final s?) f(s),
      if (r.testScore case final s?) 'held out ${f(s)}',
    ].join(' · ');
  }

  /// Every deep network's prediction columns, in document order (the HGF
  /// inputs and later deep networks may read them).
  List<String> get deepColumns => [
    for (final d in _model.deep)
      if (d.layers.isNotEmpty) ...deepPredictionColumns(d),
  ];

  /// The deep networks whose predictions the preview still waits for.
  List<String> get deepPending => [
    for (final d in compiled?.layout.deep ?? const <DocDeep>[])
      if (!_deepDone.containsKey(d.id)) d.label,
  ];

  /// Starts (or keeps) the training of every deep network that can train
  /// on the loaded data, and trains for up to [budget] ms at once; returns
  /// whether a network finished (its predictions are new columns).
  bool _syncDeep({int budget = 0}) {
    final data = csv;
    if (data == null || _model.deep.isEmpty) {
      _deepRuns.clear();
      _deepRunById = const {};
      _deepDone = const {};
      _deepErrors = const {};
      _deepWaiting = const {};
      return false;
    }
    final errors = <String, String>{};
    final ok = <DocDeep>[];
    for (final d in _model.deep) {
      final part = DocPart.deep(d.id);
      final bad = issues.where((i) => i.isError && i.parts.contains(part));
      if (bad.isNotEmpty) continue;
      if (d.layers.length < 2 ||
          d.targets.isEmpty ||
          (!d.isContinuous && d.features.isEmpty)) {
        continue;
      }
      ok.add(d);
    }
    final dd = _model.data;
    final layout = DataLayout(
      inputs: const [],
      participantColumn: dd.participantColumn,
      timeColumn: dd.timeColumn,
      timeStepColumn: dd.timeStepColumn,
      groupColumn: dd.groupColumn,
      clock: dd.clock,
      steps: dd.steps,
      transforms: _model.transforms,
      deep: ok,
    );
    try {
      layout.transform(data, cache: _dataCache);
    } on Object catch (e) {
      final m = e is FormatException ? e.message : '$e';
      _deepErrors = {for (final d in ok) d.id: m};
      _deepRuns.clear();
      _deepRunById = const {};
      _deepWaiting = const {};
      return false;
    }
    _deepDone = {..._dataCache.usedDeep};
    final notReady = <String>{};
    final runs = <String, DeepTraining>{};
    final byId = <String, DeepTraining>{};
    final waiting = <String>{};
    for (final p in _dataCache.pendingDeep) {
      final d = p.deep;
      final reads = {
        ...d.features,
        ...d.targets,
        ...?d.training.trainOn?.variables,
      };
      final outs = deepPredictionColumns(d);
      notReady.addAll(outs);
      if (reads.any(notReady.contains)) {
        waiting.add(d.id);
        continue;
      }
      var run = _deepRuns[p.key];
      if (run == null) {
        try {
          run = DeepTraining(d, p.table, scope: layout.scope);
        } on Object catch (e) {
          errors[d.id] = e is FormatException ? e.message : '$e';
          continue;
        }
      }
      runs[p.key] = run;
      byId[d.id] = run;
    }
    if (!_model.hasNetwork) {
      // The preview bar's participants: those the deep networks read.
      participants = {
        for (final r in _deepDone.values) ...r.data.participants,
        for (final r in byId.values) ...r.data.participants,
      }.where((p) => p.isNotEmpty).toList();
    }
    _deepRuns
      ..clear()
      ..addAll(runs);
    _deepRunById = byId;
    _deepErrors = errors;
    _deepWaiting = waiting;
    if (runs.isEmpty) return false;
    final finished = _train(budget);
    if (_deepRuns.isNotEmpty) _pump();
    return finished;
  }

  /// Trains the runs for up to [ms] milliseconds; whether one finished.
  bool _train(int ms) {
    if (ms <= 0) return false;
    final sw = Stopwatch()..start();
    var finished = false;
    for (final run in [..._deepRuns.values]) {
      try {
        while (sw.elapsedMilliseconds < ms && !run.advance(64)) {}
      } on Object catch (e) {
        _deepErrors = {..._deepErrors, run.deep.id: '$e'};
        _deepRuns.remove(run.key);
        _deepRunById = {
          for (final MapEntry(:key, :value) in _deepRunById.entries)
            if (key != run.deep.id) key: value,
        };
        continue;
      }
      if (run.isDone) {
        _dataCache.putDeep(run.result!);
        _deepRuns.remove(run.key);
        finished = true;
      }
      if (sw.elapsedMilliseconds >= ms) break;
    }
    return finished;
  }

  /// Trains in slices between frames until every run is done; a network
  /// that finishes refreshes the preview (its predictions are columns).
  Future<void> _pump() async {
    if (_pumping) return;
    _pumping = true;
    try {
      while (!_disposed && _deepRuns.isNotEmpty) {
        await Future<void>.delayed(const Duration(milliseconds: 4));
        if (_disposed || _deepRuns.isEmpty) break;
        if (_train(12)) {
          _refreshNow();
        } else {
          notifyListeners();
        }
      }
    } finally {
      _pumping = false;
    }
  }

  @override
  void dispose() {
    _disposed = true;
    _layout.dispose();
    super.dispose();
  }

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

  /// The document compiled (null while it has errors).
  CompiledDocument? compiled;

  /// The previewed subject (with its responses, recorded or simulated),
  /// when the whole document compiles.
  Subject? subject;

  /// The named outputs of the previewed subject.
  Map<String, Float64List> outputs = const {};

  /// The loaded data (null: simulated data), and its file's name.
  CsvData? csv;
  String? csvName;

  /// Why the loaded data could not be read (null when it could).
  String? dataError;

  /// Fitting from the builder, when the app wires it up.
  BuilderFitting? fitting;

  /// The fitting's state changed.
  void fitChanged() => notifyListeners();

  /// Filter the preview again (e.g. at new estimates).
  void refreshPreview() => _refreshNow();

  /// The participants (their ids: the loaded data's, or `1`…`n` for
  /// simulated ones; empty for a single simulated participant), and which
  /// one the preview shows.
  List<String> participants = const [];
  int _participant = 0;
  int get participant => _participant;

  void showParticipant(int k) {
    if (k == _participant) return;
    _participant = k;
    _refreshNow();
  }

  /// Use [data] (a CSV file named [name]) for the preview; null goes back
  /// to simulated data.
  void loadData(CsvData? data, {String? name, String? sessionHint}) {
    csv = data;
    csvName = data == null ? null : name;
    this.sessionHint = data == null ? null : sessionHint;
    _participant = 0;
    _refreshNow();
  }

  /// The loaded data's column that groups the participants recorded
  /// together (a BIDS dataset's `group`, say): what the clock proposes as
  /// its session column.
  String? sessionHint;

  /// Use the participants of a BIDS dataset ([merged], see
  /// `mergeBidsEvents`); a document without a participant column gets
  /// `participant_id` (an undo step).
  void loadBids(BidsMerge merged, {required String name, String? sessionHint}) {
    loadData(merged.data, name: name, sessionHint: sessionHint);
    final d = _model.data;
    if (d.participantColumn == null) {
      update(
        _model.copyWith(
          data: d.copyWith(participantColumn: () => 'participant_id'),
        ),
      );
    }
  }

  /// How many participants are simulated without loaded data. Each sees
  /// inputs of its own and, when the model has free parameters, has its
  /// own values of them ([simulatedTruth]).
  int simParticipants = 1;

  /// Simulate [n] participants; several need a participant column, which
  /// is named `participant` if the document has none (an undo step).
  void setSimParticipants(int n) {
    n = n.clamp(1, 200);
    if (n == simParticipants) return;
    simParticipants = n;
    _participant = 0;
    if (n > 1 && _model.data.participantColumn == null) {
      final d = _model.data;
      _setModel(
        _model.copyWith(
          data: DocData(
            participantColumn: _model.freshName('participant', dataColumns),
            timeColumn: d.timeColumn,
            timeStepColumn: d.timeStepColumn,
            groupColumn: d.groupColumn,
          ),
        ),
      );
      return;
    }
    _refreshNow();
  }

  /// Simulated participants play in teams of [simTeamSize] (the document's
  /// group column; 1: each alone), reshuffled every [simRegroupEvery] steps
  /// (0: never). Models whose inputs read responses (`mates_sum(press ==
  /// 'left')`, …) are simulated together, step by step
  /// ([CompiledDocument.simulateTogether]).
  int simTeamSize = 3;
  int simRegroupEvery = 0;

  /// Set the teams (see [simTeamSize]); teams of two or more need a group
  /// column, which is named `team` if the document has none (an undo step).
  void setSimTeams({int? size, int? regroupEvery}) {
    final s = (size ?? simTeamSize).clamp(1, 50);
    final e = math.max(0, regroupEvery ?? simRegroupEvery);
    if (s == simTeamSize && e == simRegroupEvery) return;
    simTeamSize = s;
    simRegroupEvery = e;
    if (s > 1 && _model.data.groupColumn == null) {
      final d = _model.data;
      _setModel(
        _model.copyWith(
          data: DocData(
            participantColumn: d.participantColumn,
            timeColumn: d.timeColumn,
            timeStepColumn: d.timeStepColumn,
            groupColumn: _model.freshName('team', dataColumns),
          ),
        ),
      );
      return;
    }
    _refreshNow();
  }

  /// Simulated participant [k]'s team at step [t] (see [simTeamSize]):
  /// consecutive participants at first, a shuffle per block after.
  String simTeam(int k, int t) {
    final size = simTeamSize;
    if (size <= 1) return _simId(k);
    final block = simRegroupEvery > 0 ? t ~/ simRegroupEvery : 0;
    var pos = k;
    if (block > 0) {
      final order = [for (var i = 0; i < math.max(1, simParticipants); i++) i]
        ..shuffle(math.Random(data.seed * 13 + 7 * block));
      pos = order.indexOf(k);
    }
    final team = pos ~/ size + 1;
    return simRegroupEvery > 0 ? 'b${block + 1}_$team' : '$team';
  }

  /// Identifies the data the fit sees: the loaded file, or the simulation's
  /// settings.
  String get dataKey => csv != null
      ? 'csv ${identityHashCode(csv)}'
      : 'sim ${data.seed} $trials $simParticipants $simTeamSize '
            '$simRegroupEvery $_inputKey';

  /// Each simulated participant's values of the free parameters (by id):
  /// the document's own values (the inspectors' sliders) for a single
  /// participant; for several, each drawn around them (±0.3 sd on the
  /// parameter's unconstrained scale). Empty for loaded data or a model
  /// without free parameters.
  Map<String, Float64List> get simulatedTruth {
    final c = compiled, m = c?.model;
    if (csv != null || c == null || m == null || m.dimension == 0) {
      return const {};
    }
    final base = c.currentValues();
    return {
      for (var k = 0; k < math.max(1, simParticipants); k++)
        _simId(k): k == 0 && simParticipants == 1
            ? Float64List.fromList(base)
            : _jitter(m, base, k),
    };
  }

  String _simId(int k) => '${k + 1}';

  /// How the builder and its fit name the [k]-th participant [s]: its id,
  /// or its position from 1.
  static String participantKey(Subject s, int k) => '${s.id ?? k + 1}';

  Float64List _jitter(HgfModel m, List<double> base, int k) {
    final rng = math.Random(data.seed * 31 + 17 * k + 5);
    double normal() {
      final u = 1 - rng.nextDouble(), v = rng.nextDouble();
      return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v);
    }

    return Float64List.fromList([
      for (final (i, p) in m.parameters.indexed)
        () {
          final y = p.transform.toUnconstrained(base[i]);
          if (!y.isFinite) return base[i];
          return p.transform.toConstrained(y + 0.3 * normal());
        }(),
    ]);
  }

  /// Every participant, as the fit sees them: the loaded data's, or the
  /// simulated participants with their simulated responses.
  List<Subject> allSubjects() {
    final c =
        compiled ?? (throw StateError('The document has errors: no subjects'));
    if (csv != null) return c.subjects(_onClock(c)!, cache: _dataCache);
    if (_together(c) case final sim?) return sim.subjects;
    final net = network!;
    _ensureInput(net);
    final truth = simulatedTruth;
    return [
      for (var k = 0; k < math.max(1, simParticipants); k++)
        _simSubject(c, k, truth),
    ];
  }

  /// The loaded data's grid, transforms and subjects, kept between edits
  /// that don't change them (a full session takes seconds to put on the
  /// clock and transform).
  final _dataCache = TransformCache()..trainDeep = false;

  /// The loaded data on [c]'s clock (see `DocClock`); the data itself
  /// without a clock; null for simulated data.
  CsvData? _onClock(CompiledDocument c) {
    final data = csv;
    if (data == null) return null;
    return c.layout.prepared(data, cache: _dataCache);
  }

  /// Whether the simulated participants are simulated together (their
  /// inputs read responses; see [simTeamSize]).
  bool get simulatesTogether =>
      csv == null && (compiled?.inputsReadResponses ?? false);

  ({CsvData data, List<Subject> subjects})? _togetherCache;
  CompiledDocument? _togetherFor;

  /// The participants simulated together, when [c]'s inputs read
  /// responses (once per refresh): the raw data (participant, team, time
  /// and the inputs' own columns, then the responses step by step) and
  /// the subjects it reads as.
  ({CsvData data, List<Subject> subjects})? _together(CompiledDocument c) {
    if (csv != null || !c.inputsReadResponses) return null;
    if (identical(_togetherFor, c)) return _togetherCache;
    _ensureInput(network!);
    final n = math.max(1, simParticipants);
    final ids = [for (var k = 0; k < n; k++) n > 1 ? _simId(k) : null];
    final plain = c.toCsv([
      for (var k = 0; k < n; k++) Subject(inputs: _inputs[k], id: ids[k]),
    ]);
    // The derived columns come from the transforms, step by step.
    final derived = {for (final t in c.document.transforms) t.name};
    final keep = [
      for (final (i, h) in plain.header.indexed)
        if (!derived.contains(h)) i,
    ];
    final group = c.document.data.groupColumn;
    final perParticipant = plain.rows.length ~/ n;
    final skeleton = CsvData(
      [for (final i in keep) plain.header[i], ?group],
      [
        for (final (r, row) in plain.rows.indexed)
          [
            for (final i in keep) row[i],
            if (group != null) simTeam(r ~/ perParticipant, r % perParticipant),
          ],
      ],
    );
    final truth = simulatedTruth;
    _togetherCache = c.simulateTogether(
      skeleton,
      math.Random(data.seed + 1),
      x: {for (var k = 0; k < n; k++) ids[k] ?? '': ?truth[_simId(k)]},
    );
    _togetherFor = c;
    return _togetherCache;
  }

  /// The simulated data as a table the document reads back: the raw data
  /// when participants are simulated together, else [CompiledDocument.toCsv].
  CsvData simulatedCsv() {
    final c =
        compiled ?? (throw StateError('The document has errors: no data'));
    return _together(c)?.data ?? c.toCsv(allSubjects());
  }

  /// Simulated participant [k]: its inputs and, with a response, its
  /// responses at its [truth] values (the document's starting values
  /// without free parameters).
  Subject _simSubject(
    CompiledDocument c,
    int k,
    Map<String, Float64List> truth,
  ) {
    final id = simParticipants > 1 ? _simId(k) : null;
    var s = Subject(inputs: _inputs[k], id: id);
    if (c.model != null && c.document.response!.readsData) {
      final x = truth[_simId(k)];
      s = c
          .simulate(
            [s],
            math.Random(data.seed + 1 + 7919 * k),
            x: x == null ? null : [x],
          )
          .single;
    }
    return s;
  }

  /// The preview's parameter values for [participant]: its estimates (when
  /// the fit says so), a simulated participant's own values, else the
  /// document's.
  List<double>? _previewX(CompiledDocument c, String participant) {
    final m = c.model;
    if (m == null || m.dimension == 0) return null;
    return fitting?.estimateFor(participant, m.names) ??
        simulatedTruth[participant] ??
        c.currentValues();
  }

  /// The previewed data as CSV, in the document's columns (simulated data,
  /// or the loaded participant read back).
  String? dataCsv() {
    final c = compiled;
    if (c == null || subject == null) return null;
    return csvText(csv == null ? simulatedCsv() : c.toCsv([subject!]));
  }

  /// The loaded data (or the simulated subject) with the named outputs
  /// appended per row, as CSV; null while the document has errors.
  ///
  /// Each participant's outputs are filtered at its estimates when there is
  /// a fit (and the preview uses it), else as the preview filters it.
  String? outputsCsv() {
    final c = compiled;
    if (c == null || subject == null) return null;
    final subs = csv == null
        ? allSubjects()
        : c.subjects(_onClock(c)!, cache: _dataCache);
    final data = _onClock(c) ?? simulatedCsv();
    // withOutputs names participants by their column ('' without one).
    final x = {
      for (final (k, s) in subs.indexed)
        '${s.id ?? ''}': ?_previewX(c, participantKey(s, k)),
    };
    return csvText(
      c.withOutputs(data, x: x, cache: csv == null ? null : _dataCache),
    );
  }

  /// The simulated data's settings: one series per input, kind by kind
  /// (reversals for binary inputs, random walks for continuous ones, …),
  /// each with its own seed.
  final DataSettings data = DataSettings(nTrials: 300);

  /// Per simulated participant, the input table (`nTrials ×
  /// nInputColumns`) and what generated it, per input node index.
  List<Float64List> _inputs = const [];
  List<Map<int, List<Float64List>>> _hiddens = const [];

  /// What generated the simulated inputs, per input node index (empty for
  /// loaded data).
  Map<int, List<Float64List>> hidden = const {};
  Object? _inputKey;

  /// The trials simulated: [DataSettings.nTrials], at most 150 with a
  /// dp-state node (20 000 proposals per trial: slow).
  int get trials => doc.nodes.any((n) => n is DpStateNode)
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
    _refreshBody();
    // Deep networks: small ones finish at once (then the preview reads
    // their predictions); the rest train in the background.
    for (
      var round = 0;
      _syncDeep(budget: round < 3 ? deepBudgetMs : 0);
      round++
    ) {
      _refreshBody();
    }
    previewMicros = sw.elapsedMicroseconds;
    revision++;
    notifyListeners();
  }

  void _refreshBody() {
    network = null;
    trajectories = null;
    previewError = null;
    compiled = null;
    subject = null;
    outputs = const {};
    hidden = const {};
    dataError = null;
    var found = checkNetwork(doc, build: false);
    if (!_model.hasNetwork && _model.deep.isNotEmpty) {
      // Deep networks only: their checks, no HGF preview.
      found = checkModel(_model, columns: csv?.header);
      previewError = csv == null
          ? 'Load data to train the deep networks (Data panel), or export '
                'them (File menu).'
          : null;
    } else if (found.any((i) => i.isError)) {
      previewError = 'Fix the errors to see the preview.';
    } else {
      try {
        network = doc.build();
      } on Object catch (e) {
        found = [DocIssue(IssueSeverity.error, buildErrorMessage(e)), ...found];
        previewError = 'Fix the errors to see the preview.';
      }
    }
    if (network case final net?) {
      // Every input node reads data: new ones get an input of their own.
      _model = _model.withInputsFor(net);
      found = checkModel(_model, columns: csv?.header, built: net);
      if (!found.any((i) => i.isError)) {
        try {
          compiled = _model.compile();
        } on Object catch (e) {
          found = [
            DocIssue(IssueSeverity.error, buildErrorMessage(e)),
            ...found,
          ];
        }
      }
      try {
        _preview(net);
      } on Object catch (e) {
        trajectories = null;
        previewError = '$e';
      }
    }
    issues = found;
    final ni = <String, IssueSeverity>{};
    final ei = <int, IssueSeverity>{};
    final pi = <String, IssueSeverity>{};
    for (final i in found) {
      for (final n in i.nodes) {
        ni[n] = _worse(ni[n], i.severity);
      }
      for (final e in i.edges) {
        ei[e] = _worse(ei[e], i.severity);
      }
      for (final p in i.parts) {
        pi[p] = _worse(pi[p], i.severity);
      }
    }
    nodeIssues = ni;
    edgeIssues = ei;
    partIssues = pi;
  }

  /// Filters the preview: the loaded data's participant through the
  /// compiled document, or simulated inputs (and, with a response,
  /// simulated responses); the network alone while the rest of the
  /// document has errors.
  void _preview(Network net) {
    final c = compiled;
    if (csv != null && c != null) {
      final List<Subject> subs;
      try {
        subs = c.subjects(_onClock(c)!, cache: _dataCache);
      } on FormatException catch (e) {
        dataError = e.message;
        previewError = 'The data: ${e.message}';
        return;
      }
      participants = [for (final (k, s) in subs.indexed) '${s.id ?? k + 1}'];
      if (subs.isEmpty) {
        previewError = 'The data has no rows.';
        return;
      }
      final k = _participant.clamp(0, subs.length - 1);
      final s = subs[k];
      subject = s;
      trajectories = c.filter(s, _previewX(c, participantKey(s, k)));
      outputs = c.outputs(trajectories!);
      previewError = _divergence(net, trajectories!);
      return;
    }
    final n = math.max(1, simParticipants);
    participants = n == 1 ? const [] : [for (var k = 0; k < n; k++) _simId(k)];
    _ensureInput(net);
    final k = _participant.clamp(0, n - 1);
    if (c == null) {
      trajectories = net.inputData(
        _inputs[k],
        fields: Playground.recordedFields,
      );
      hidden = _hiddens[k];
      previewError = _divergence(net, trajectories!);
      return;
    }
    final s = _together(c)?.subjects[k] ?? _simSubject(c, k, simulatedTruth);
    subject = s;
    hidden = _hiddens[k];
    trajectories = c.filter(s, _previewX(c, _simId(k)));
    outputs = c.outputs(trajectories!);
    previewError = _divergence(net, trajectories!);
  }

  static IssueSeverity _worse(IssueSeverity? a, IssueSeverity b) =>
      a == null || b.index < a.index ? b : a;

  /// Simulated inputs for [net]'s input nodes, made again only when the
  /// inputs or the data settings changed.
  void _ensureInput(Network net) {
    final a = doc.addresses;
    final inputs = net.inputIdxs;
    final shape = [
      for (final i in inputs) (net.kindOf(i), _width(a.nodeAt(i), i, a)),
    ];
    final count = math.max(1, simParticipants);
    final key = Object.hashAll([...shape, trials, data.seed, count]);
    if (key == _inputKey) return;
    final tables = <Float64List>[];
    final hiddens = <Map<int, List<Float64List>>>[];
    for (var p = 0; p < count; p++) {
      final (t, h) = _simulateInputs(net, shape, data.seed + 104729 * p);
      tables.add(t);
      hiddens.add(h);
    }
    _inputs = tables;
    _hiddens = hiddens;
    _inputKey = key;
  }

  /// One participant's simulated inputs (see [_ensureInput]), from [seed].
  (Float64List, Map<int, List<Float64List>>) _simulateInputs(
    Network net,
    List<(NodeKind, int)> shape,
    int seed,
  ) {
    final inputs = net.inputIdxs;
    final n = trials, width = net.nInputColumns;
    final table = Float64List(n * width);
    final hid = <int, List<Float64List>>{};
    var col = 0;
    for (final (k, i) in inputs.indexed) {
      final (kind, w) = shape[k];
      final s = DataSettings(nTrials: n, seed: seed + 7919 * k);
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
    return (table, hid);
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
      final i = doc.addresses.indexOf(address);
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
        return net.getInitial(doc.addresses.indexOf(address), field);
      } on Object {
        // fall through
      }
    }
    final node = doc.node(address);
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

/// [data] as CSV text (cells with a comma, quote or newline quoted).
String csvText(CsvData data) => data.toText();

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

/// [BuilderModel.layoutChanges].
class _Layout extends ChangeNotifier {
  void changed() => notifyListeners();
}
