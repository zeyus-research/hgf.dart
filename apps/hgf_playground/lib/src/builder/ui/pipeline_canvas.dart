import 'dart:math' as math;

import 'package:flutter/gestures.dart';
import 'package:flutter/services.dart';
import 'package:hgf_fit/document.dart';
import 'package:material_ui/material_ui.dart';

import '../builder_model.dart';

/// The width of a pipeline card, the height of its header and of a row.
const double cardWidth = 196, cardHeader = 40, cardRow = 22;

/// How many of the data's columns a data card lists.
const int maxDataRows = 40;

/// A row of a pipeline card: its label and, if it takes or gives a wire,
/// its port.
typedef CardRow = ({String label, String? tag, RowPort? port});

/// What a row's port is for.
sealed class RowPort {
  const RowPort();
}

/// A data column (an output port on the right).
final class ColumnOut extends RowPort {
  const ColumnOut(this.column);
  final String column;
}

/// An input of the network (an input port on the left).
final class InputIn extends RowPort {
  const InputIn(this.input);
  final String input;
}

/// The response's recorded column (an input port on the left).
final class RecordedIn extends RowPort {
  const RecordedIn();
}

/// A deep network's features or targets: a column dropped here is read
/// into its feature (or target) layer.
final class DeepIn extends RowPort {
  const DeepIn(this.id, {required this.target});
  final String id;
  final bool target;
}

/// A pipeline card: a part of the model and what it shows.
class PipelineCard {
  PipelineCard({
    required this.part,
    required this.title,
    required this.subtitle,
    required this.rows,
    required this.at,
    this.headerIn = false,
    this.headerOut,
    this.chainOut = false,
  });

  /// The [DocPart] key.
  final String part;
  final String title, subtitle;
  final List<CardRow> rows;

  /// The top-left corner, in canvas units.
  final Offset at;

  /// An input port on the header's left (transforms read columns; the
  /// response and outputs read beliefs).
  final bool headerIn;

  /// An output port on the header's right: a transform's column, or the
  /// network's beliefs (`null` column).
  final ({String? column})? headerOut;

  /// A data step's output (the table, into the next step or the data
  /// card): drawn on the header's right, not a wire to drag.
  final bool chainOut;

  double get height => cardHeader + rows.length * cardRow + 6;
  Rect get rect => at & Size(cardWidth, height);

  Offset get headerInPort => at + const Offset(0, cardHeader / 2);
  Offset get headerOutPort => at + const Offset(cardWidth, cardHeader / 2);
  Offset rowCentre(int k) => at + Offset(0, cardHeader + (k + 0.5) * cardRow);
  Offset rowPort(int k) {
    final c = rowCentre(k);
    return rows[k].port is ColumnOut ? c + const Offset(cardWidth, 0) : c;
  }
}

/// A wire between two ports, and what selecting it selects.
typedef PipelineWire = ({Offset from, Offset to, String part, bool ok});

/// The cards and wires of [m]'s pipeline.
({List<PipelineCard> cards, List<PipelineWire> wires}) pipelineGeometry(
  BuilderModel m,
) {
  final doc = m.document;
  final pos = m.pipelinePositions;
  final cards = <PipelineCard>[];
  final columns = m.dataColumns;
  final d = doc.data;
  String? role(String c) => c == d.participantColumn
      ? 'participant'
      : c == d.timeColumn
      ? 'time'
      : c == d.timeStepColumn
      ? 'time step'
      : c == d.groupColumn
      ? 'group'
      : null;
  final shown = columns.take(maxDataRows).toList();
  final steps = d.steps;
  final prepared = steps.isEmpty && d.clock == null ? null : m.prepared;
  for (final (i, st) in steps.indexed) {
    final rows = prepared == null || i >= prepared.stepRows.length
        ? null
        : prepared.stepRows[i];
    cards.add(
      PipelineCard(
        part: DocPart.step(i),
        title: BuilderModel.stepKinds[st.kind] ?? st.kind,
        subtitle: rows == null
            ? 'data step ${i + 1}'
            : 'data step ${i + 1} · $rows rows',
        rows: [(label: st.summary, tag: null, port: null)],
        at: pos[DocPart.step(i)]!,
        headerIn: true,
        chainOut: true,
      ),
    );
  }
  cards.add(
    PipelineCard(
      part: DocPart.data,
      title: 'Data',
      subtitle: m.csv == null
          ? 'simulated · ${m.trials} trials'
          : prepared != null
          ? '${m.csvName ?? 'loaded'} · ${m.csv!.rows.length} → '
                '${prepared.table.rows.length} rows'
          : '${m.csvName ?? 'loaded'} · ${m.csv!.rows.length} rows',
      headerIn: steps.isNotEmpty,
      rows: [
        for (final c in shown) (label: c, tag: role(c), port: ColumnOut(c)),
        if (columns.length > shown.length)
          (
            label: '+${columns.length - shown.length} more',
            tag: null,
            port: null,
          ),
      ],
      at: pos[DocPart.data]!,
    ),
  );
  for (final t in doc.transforms) {
    cards.add(
      PipelineCard(
        part: DocPart.transform(t.name),
        title: t.name,
        subtitle: 'transform',
        rows: [(label: '= ${t.expression}', tag: null, port: null)],
        at: pos[DocPart.transform(t.name)]!,
        headerIn: true,
        headerOut: (column: t.name),
      ),
    );
  }
  final nNodes = doc.network.nodes.length;
  if (doc.hasNetwork) {
    cards.add(
      PipelineCard(
        part: DocPart.network,
        title: 'HGF network',
        subtitle:
            '$nNodes node${nNodes == 1 ? '' : 's'} · double-click to edit',
        rows: [
          for (final i in doc.inputs)
            (label: i.name, tag: prettyId(i.node), port: InputIn(i.name)),
        ],
        at: pos[DocPart.network]!,
        headerOut: (column: null),
      ),
    );
  }
  for (final d in doc.deep) {
    cards.add(
      PipelineCard(
        part: DocPart.deep(d.id),
        title: d.label,
        subtitle: '${deepShape(d)} · double-click to edit',
        rows: deepRows(d, status: m.deepStatus(d.id)),
        at: pos[DocPart.deep(d.id)]!,
      ),
    );
  }
  if (doc.response case final r?) {
    cards.add(
      PipelineCard(
        part: DocPart.response,
        title: r.name,
        subtitle: 'response · ${readoutName(r.readout)}',
        rows: [
          (
            label: 'recorded: ${r.dataColumns.join(', ')}',
            tag: r.options == null ? null : '${r.options!.length} options',
            port: const RecordedIn(),
          ),
        ],
        at: pos[DocPart.response]!,
        headerIn: true,
      ),
    );
  }
  for (final o in doc.outputs) {
    cards.add(
      PipelineCard(
        part: DocPart.output(o.name),
        title: o.name,
        subtitle: 'output',
        rows: [
          (label: '${prettyId(o.node)}.${o.field}', tag: null, port: null),
        ],
        at: pos[DocPart.output(o.name)]!,
        headerIn: true,
      ),
    );
  }

  if (doc.hasNetwork) {
    cards.add(
      PipelineCard(
        part: DocPart.fit,
        title: 'Fit',
        subtitle: fitSubtitle(m),
        rows: _fitRows(m),
        at: pos[DocPart.fit]!,
      ),
    );
  }

  // Where each column comes from.
  final dataCard = cards[steps.length];
  final sources = <String, Offset>{
    for (final (k, row) in dataCard.rows.indexed)
      if (row.port case ColumnOut(:final column)) column: dataCard.rowPort(k),
    for (final c in cards)
      if (c.part.startsWith('deep:'))
        for (final (k, row) in c.rows.indexed)
          if (row.port case ColumnOut(:final column)) column: c.rowPort(k),
    for (final c in cards)
      if (c.headerOut case (column: final String col)) col: c.headerOutPort,
  };
  final byPart = {for (final c in cards) c.part: c};
  final wires = <PipelineWire>[
    // The data steps' chain into the data card.
    for (var i = 0; i < steps.length; i++)
      (
        from: cards[i].headerOutPort,
        to: cards[i + 1].headerInPort,
        part: DocPart.step(i),
        ok: true,
      ),
  ];
  for (final t in doc.transforms) {
    final card = byPart[DocPart.transform(t.name)]!;
    for (final v in t.expression.variables) {
      if (sources[v] case final from?) {
        wires.add((
          from: from,
          to: card.headerInPort,
          part: card.part,
          ok: true,
        ));
      }
    }
  }
  for (final d in doc.deep) {
    final card = byPart[DocPart.deep(d.id)]!;
    for (final (k, row) in card.rows.indexed) {
      if (row.port is! DeepIn || row.tag == null) continue;
      if (sources[row.label] case final from?) {
        wires.add((from: from, to: card.rowPort(k), part: card.part, ok: true));
      }
    }
  }
  final net = byPart[DocPart.network];
  if (net == null) return (cards: cards, wires: wires);
  final widths = <String, int>{};
  if (m.network case final built?) {
    for (final i in doc.inputs) {
      try {
        widths[i.name] = inputWidth(
          built,
          doc.network,
          doc.network.addresses.indexOf(i.node),
        );
      } on Object {
        // stale until the next refresh
      }
    }
  }
  for (final (k, i) in doc.inputs.indexed) {
    for (final c in i.columnsFor(widths[i.name] ?? 1)) {
      if (sources[c] case final from?) {
        wires.add((
          from: from,
          to: net.rowPort(k),
          part: DocPart.input(i.name),
          ok: true,
        ));
      }
    }
  }
  if (byPart[DocPart.response] case final card?) {
    wires.add((
      from: net.headerOutPort,
      to: card.headerInPort,
      part: DocPart.response,
      ok: true,
    ));
    for (final c in doc.response!.dataColumns) {
      if (sources[c] case final from?) {
        wires.add((
          from: from,
          to: card.rowPort(0),
          part: DocPart.response,
          ok: true,
        ));
      }
    }
  }
  for (final o in doc.outputs) {
    final card = byPart[DocPart.output(o.name)]!;
    wires.add((
      from: net.headerOutPort,
      to: card.headerInPort,
      part: card.part,
      ok: true,
    ));
  }
  return (cards: cards, wires: wires);
}

/// A deep network's layer sizes, features to targets (`2 → 8 → 1`).
String deepShape(DocDeep d) =>
    [for (final l in d.layers.reversed) '${l.size}'].join(' → ');

/// A deep card's rows: its feature columns, then its target columns, each
/// a drop target for more (a placeholder row while there are none).
List<CardRow> deepRows(DocDeep d, {String? status}) {
  final cats = d.categories;
  return [
    if (status != null && status.isNotEmpty)
      (label: status, tag: null, port: null),
    if (!d.isContinuous) ...[
      for (final c in d.features)
        (label: c, tag: 'feature', port: DeepIn(d.id, target: false)),
      if (d.features.isEmpty)
        (
          label: 'drop feature columns here',
          tag: null,
          port: DeepIn(d.id, target: false),
        ),
    ],
    for (final c in d.targets)
      (
        label: c,
        tag: d.isContinuous
            ? 'observed'
            : cats == null
            ? 'target'
            : '${cats.length} labels',
        port: DeepIn(d.id, target: true),
      ),
    if (d.targets.isEmpty)
      (
        label: d.isContinuous
            ? 'drop observed columns here'
            : 'drop target columns here',
        tag: null,
        port: DeepIn(d.id, target: true),
      ),
    // Its predictions: columns for the HGF inputs and later networks.
    if (d.layers.isNotEmpty && (d.targets.isNotEmpty || d.isContinuous))
      for (final c in deepPredictionColumns(d))
        (label: c, tag: 'prediction', port: ColumnOut(c)),
  ];
}

/// The fit card's subtitle: the method, and the last result.
String fitSubtitle(BuilderModel m) {
  final doc = m.document;
  final f = m.fitting;
  final grouped = !doc.group.isEmpty;
  final method = switch (doc.fit.method) {
    'nuts' => grouped ? 'hierarchical NUTS' : 'NUTS',
    'none' => 'none',
    _ => grouped ? 'empirical Bayes' : 'MAP',
  };
  final status = f == null
      ? ''
      : f.running
      ? ' · running'
      : f.result == null
      ? ''
      : f.stale
      ? ' · fitted (stale)'
      : ' · fitted';
  return '$method$status';
}

/// The fit card's rows: the free parameters and their roles.
List<CardRow> _fitRows(BuilderModel m) {
  final doc = m.document;
  final names = doc.parameterNames;
  if (names.isEmpty) {
    return [(label: 'no free parameters', tag: null, port: null)];
  }
  const most = 6;
  return [
    for (final n in names.take(most))
      (
        label: n,
        tag: switch (doc.group.roleOf(n)) {
          ParameterRole.pooled => 'pooled',
          ParameterRole.shared => 'shared',
          ParameterRole.individual => null,
        },
        port: null,
      ),
    if (names.length > most)
      (label: '+${names.length - most} more', tag: null, port: null),
  ];
}

/// A short name for a readout's type.
String readoutName(Map<String, Object?> readout) => switch (readout['type']) {
  'GaussianReadout' => 'Gaussian',
  'BinarySoftmaxReadout' => 'binary softmax',
  'CategoricalSoftmaxReadout' => 'categorical softmax',
  'SoftmaxReadout' => 'softmax over options',
  'ExprReadout' => 'expression',
  'ObservationSurprise' => 'observations',
  final t => '$t',
};

/// The cubic of a wire, left to right.
Path wirePath(Offset a, Offset b) {
  final k = math.max(40.0, (b.dx - a.dx).abs() / 2);
  return Path()
    ..moveTo(a.dx, a.dy)
    ..cubicTo(a.dx + k, a.dy, b.dx - k, b.dy, b.dx, b.dy);
}

Offset _wireAt(Offset a, Offset b, double t) {
  final k = math.max(40.0, (b.dx - a.dx).abs() / 2);
  final p1 = a + Offset(k, 0), p2 = b - Offset(k, 0);
  final u = 1 - t;
  return a * (u * u * u) +
      p1 * (3 * u * u * t) +
      p2 * (3 * u * t * t) +
      b * (t * t * t);
}

/// The model's pipeline on a canvas, left to right: the data (its
/// columns), transforms, the HGF network (one card; double-click it to edit
/// the network), the response and the trajectory outputs. Drag a column's
/// port onto one of the network's inputs to read it, onto the response to
/// record it, or onto the background to make a transform of it; drag the
/// network's beliefs port onto the background to add an output.
class PipelineCanvas extends StatefulWidget {
  const PipelineCanvas({super.key, required this.model});

  final BuilderModel model;

  @override
  State<PipelineCanvas> createState() => _PipelineCanvasState();
}

sealed class _Drag {}

final class _PanDrag extends _Drag {
  _PanDrag(this.pan, this.scale, this.focal);
  final Offset pan, focal;
  final double scale;
}

final class _CardDrag extends _Drag {
  _CardDrag(this.part, this.grab);
  final String part;
  final Offset grab;
}

final class _WireDrag extends _Drag {
  _WireDrag(this.from, this.column, this.at);
  final Offset from;

  /// The column dragged (null: the network's beliefs).
  final String? column;
  Offset at;

  /// What a drop here would do, and whether it can.
  String? tip;
  bool ok = true;
}

class _PipelineCanvasState extends State<PipelineCanvas> {
  final _focus = FocusNode(debugLabel: 'pipeline canvas');
  Offset _pan = Offset.zero;
  double _scale = 1;
  int _fitted = -1;
  Size _size = Size.zero;
  _Drag? _drag;
  Offset? _downAt;
  (Duration, Offset)? _lastTap;
  String? _hoverPart;
  static final _clock = Stopwatch()..start();

  BuilderModel get m => widget.model;

  @override
  void initState() {
    super.initState();
    m.addListener(_changed);
  }

  @override
  void didUpdateWidget(PipelineCanvas old) {
    super.didUpdateWidget(old);
    if (old.model != m) {
      old.model.removeListener(_changed);
      m.addListener(_changed);
    }
  }

  @override
  void dispose() {
    m.removeListener(_changed);
    _focus.dispose();
    super.dispose();
  }

  void _changed() => setState(() {});

  Offset _toCanvas(Offset screen) => (screen - _pan) / _scale;

  void _fit(List<PipelineCard> cards) {
    if (_size.isEmpty || cards.isEmpty) return;
    var r = cards.first.rect;
    for (final c in cards.skip(1)) {
      r = r.expandToInclude(c.rect);
    }
    final box = r.inflate(40);
    _scale = math
        .min(_size.width / box.width, _size.height / box.height)
        .clamp(0.3, 1.2);
    _pan = _size.center(Offset.zero) - box.center * _scale;
  }

  void _zoom(double factor, Offset focal) {
    final s = (_scale * factor).clamp(0.2, 3.0);
    final c = _toCanvas(focal);
    setState(() {
      _scale = s;
      _pan = focal - c * s;
    });
  }

  // ---------------------------------------------------------------------------
  // Hit tests
  // ---------------------------------------------------------------------------

  List<PipelineCard> get _cards => pipelineGeometry(m).cards;

  PipelineCard? _cardAt(Offset screen) {
    final p = _toCanvas(screen);
    for (final c in _cards.reversed) {
      if (c.rect.inflate(2).contains(p)) return c;
    }
    return null;
  }

  /// An output port near [screen]: a column (data row or transform) or the
  /// network's beliefs.
  ({Offset at, String? column})? _outPortAt(Offset screen) {
    final p = _toCanvas(screen);
    final reach = math.max(9.0, 12 / _scale);
    ({Offset at, String? column})? best;
    var bestD = reach * reach;
    void consider(Offset at, String? column) {
      final d = (at - p).distanceSquared;
      if (d < bestD) {
        bestD = d;
        best = (at: at, column: column);
      }
    }

    for (final c in _cards) {
      if (c.headerOut case (column: final col)) consider(c.headerOutPort, col);
      for (final (k, row) in c.rows.indexed) {
        if (row.port case ColumnOut(:final column)) {
          consider(c.rowPort(k), column);
        }
      }
    }
    return best;
  }

  /// The row of [card] under [screen], if any.
  int? _rowAt(PipelineCard card, Offset screen) {
    final p = _toCanvas(screen) - card.at;
    if (p.dy < cardHeader) return null;
    final k = ((p.dy - cardHeader) / cardRow).floor();
    return k >= 0 && k < card.rows.length ? k : null;
  }

  String? _wireAtPart(Offset screen) {
    final p = _toCanvas(screen);
    final reach = math.max(5.0, 8 / _scale);
    String? best;
    var bestD = reach;
    for (final w in pipelineGeometry(m).wires) {
      Offset? prev;
      for (var k = 0; k <= 24; k++) {
        final q = _wireAt(w.from, w.to, k / 24);
        if (prev != null) {
          final d = _segmentDistance(p, prev, q);
          if (d < bestD) {
            bestD = d;
            best = w.part;
          }
        }
        prev = q;
      }
    }
    return best;
  }

  static double _segmentDistance(Offset p, Offset a, Offset b) {
    final ab = b - a;
    final len2 = ab.distanceSquared;
    if (len2 == 0) return (p - a).distance;
    final t = (((p - a).dx * ab.dx + (p - a).dy * ab.dy) / len2).clamp(
      0.0,
      1.0,
    );
    return (p - (a + ab * t)).distance;
  }

  // ---------------------------------------------------------------------------
  // Gestures
  // ---------------------------------------------------------------------------

  void _onScaleStart(ScaleStartDetails d) {
    _focus.requestFocus();
    final at = d.pointerCount == 1
        ? (_downAt ?? d.localFocalPoint)
        : d.localFocalPoint;
    if (d.pointerCount == 1) {
      if (_outPortAt(at) case final p?) {
        setState(() => _drag = _WireDrag(p.at, p.column, _toCanvas(at)));
        return;
      }
      if (_cardAt(at) case final c?) {
        m.checkpoint();
        m.select(PartSelection(c.part));
        _drag = _CardDrag(c.part, _toCanvas(at) - c.at);
        return;
      }
    }
    _drag = _PanDrag(_pan, _scale, at);
  }

  void _onScaleUpdate(ScaleUpdateDetails d) {
    final at = d.localFocalPoint;
    switch (_drag) {
      case _PanDrag(:final pan, :final scale, :final focal):
        final s = (scale * d.scale).clamp(0.2, 3.0);
        final c = (focal - pan) / scale;
        setState(() {
          _scale = s;
          _pan = at - c * s;
        });
      case _CardDrag(:final part, :final grab):
        m.movePart(part, _toCanvas(at) - grab);
      case final _WireDrag w:
        w.at = _toCanvas(at);
        final (tip, ok) = _dropTip(w, at);
        w
          ..tip = tip
          ..ok = ok;
        setState(() {});
      case null:
    }
  }

  /// What dropping [w] at [screen] would do.
  (String, bool) _dropTip(_WireDrag w, Offset screen) {
    final card = _cardAt(screen);
    final column = w.column;
    if (column == null) {
      return card == null
          ? ('Release to add an output of a belief', true)
          : ('Drop on the background to add an output', false);
    }
    if (card == null) {
      return m.deepColumns.contains(column)
          ? (
              'Drop a prediction on an HGF input or a later deep network '
                  '(transforms run before the deep networks)',
              false,
            )
          : ('Release to make a transform of "$column"', true);
    }
    if (card.part == DocPart.network) {
      final k = _rowAt(card, screen);
      if (k == null) return ('Drop on one of the inputs', false);
      final port = card.rows[k].port;
      if (port case InputIn(:final input)) {
        return ('Release: input "$input" reads "$column"', true);
      }
    }
    if (card.part == DocPart.response) {
      return ('Release: the responses are in "$column"', true);
    }
    if (card.part.startsWith('deep:')) {
      final port = switch (_rowAt(card, screen)) {
        final k? => card.rows[k].port,
        null => null,
      };
      if (port case DeepIn(:final target)) {
        return (
          target
              ? 'Release: "$column" is a target (the target layer follows)'
              : 'Release: "$column" is a feature (the feature layer follows)',
          true,
        );
      }
      return ('Drop on the features or the targets', false);
    }
    if (card.part.startsWith('transform:')) {
      return ('Write "$column" into the expression to read it', false);
    }
    return ('Nothing to connect here', false);
  }

  void _onScaleEnd(ScaleEndDetails d) {
    final drag = _drag;
    _drag = null;
    if (drag is! _WireDrag) return;
    setState(() {});
    final screen = drag.at * _scale + _pan;
    final card = _cardAt(screen);
    final column = drag.column;
    if (column == null) {
      if (card == null) m.addOutput(at: drag.at);
      return;
    }
    if (card == null) {
      if (m.deepColumns.contains(column)) {
        if (drag.tip != null) _tell(drag.tip!);
        return;
      }
      m.addTransform(column: column, at: drag.at - const Offset(0, 20));
      return;
    }
    if (card.part == DocPart.network) {
      final port = switch (_rowAt(card, screen)) {
        final k? => card.rows[k].port,
        null => null,
      };
      if (port case InputIn(:final input)) m.bindColumn(column, input);
      return;
    }
    if (card.part == DocPart.response) {
      m.bindResponseColumn(column);
      return;
    }
    if (card.part.startsWith('deep:')) {
      final port = switch (_rowAt(card, screen)) {
        final k? => card.rows[k].port,
        null => null,
      };
      if (port case DeepIn(:final id, :final target)) {
        m.bindDeepColumn(id, column, target: target);
        return;
      }
    }
    if (drag.tip != null && !drag.ok) _tell(drag.tip!);
  }

  void _tell(String message) {
    ScaffoldMessenger.maybeOf(context)
      ?..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(content: Text(message), duration: const Duration(seconds: 4)),
      );
  }

  void _onTapUp(TapUpDetails d) {
    _focus.requestFocus();
    final at = d.localPosition;
    final now = _clock.elapsed;
    final double = switch (_lastTap) {
      (final t, final p) =>
        now - t < kDoubleTapTimeout && (p - at).distance < kDoubleTapSlop,
      null => false,
    };
    _lastTap = (now, at);
    if (_cardAt(at) case final c?) {
      if (double && c.part == DocPart.network) {
        _lastTap = null;
        m.showView(BuilderView.network);
        return;
      }
      if (double && c.part.startsWith('deep:')) {
        _lastTap = null;
        m.showDeep(c.part.substring('deep:'.length));
        return;
      }
      final port = switch (_rowAt(c, at)) {
        final k? => c.rows[k].port,
        null => null,
      };
      if (port case InputIn(:final input)) {
        m.select(PartSelection(DocPart.input(input)));
      } else {
        m.select(PartSelection(c.part));
      }
    } else if (_wireAtPart(at) case final part?) {
      m.select(PartSelection(part));
    } else {
      if (double) {
        _lastTap = null;
        m.addTransform(at: _toCanvas(at));
        return;
      }
      m.select(null);
    }
  }

  /// The data step kinds, at [at]; the pick is added at [index] (default:
  /// last).
  Future<void> _stepMenu(Offset at, {int? index}) async {
    final kind = await _popup<String>(at, [
      for (final MapEntry(:key, :value) in BuilderModel.stepKinds.entries)
        PopupMenuItem(value: key, child: Text(value)),
    ]);
    if (kind != null) m.addStep(kind, index: index);
  }

  Future<T?> _popup<T>(Offset at, List<PopupMenuEntry<T>> items) {
    final box = context.findRenderObject()! as RenderBox;
    final g = box.localToGlobal(at);
    final overlay =
        Overlay.of(context).context.findRenderObject()! as RenderBox;
    return showMenu<T>(
      context: context,
      position: RelativeRect.fromRect(
        Rect.fromLTWH(g.dx, g.dy, 1, 1),
        Offset.zero & overlay.size,
      ),
      items: items,
    );
  }

  Future<void> _contextMenu(Offset at) async {
    _focus.requestFocus();
    final card = _cardAt(at);
    final items = <PopupMenuEntry<VoidCallback>>[];
    final c = _toCanvas(at);
    if (card != null) {
      m.select(PartSelection(card.part));
      if (card.part == DocPart.network) {
        items.add(
          PopupMenuItem(
            value: () => m.showView(BuilderView.network),
            child: const Text('Edit the network'),
          ),
        );
        if (m.document.response == null) {
          items.add(
            PopupMenuItem(
              value: () => m.addResponse(at: card.at + const Offset(280, 0)),
              child: const Text('Add a response'),
            ),
          );
        }
        items.addAll([
          PopupMenuItem(
            value: () => m.addOutput(at: card.at + const Offset(280, 160)),
            child: const Text('Add an output'),
          ),
          PopupMenuItem(
            value: () => m.removePart(DocPart.network),
            child: const Text('Delete the HGF network'),
          ),
        ]);
      } else if (card.part.startsWith('deep:')) {
        final id = card.part.substring('deep:'.length);
        items.addAll([
          PopupMenuItem(
            value: () => m.showDeep(id),
            child: const Text('Edit the layers'),
          ),
          PopupMenuItem(
            value: () => m.removePart(card.part),
            child: const Text('Delete'),
          ),
        ]);
      } else if (card.part.startsWith('step:')) {
        final i = int.parse(card.part.substring('step:'.length));
        final n = m.document.data.steps.length;
        items.addAll([
          if (i > 0)
            PopupMenuItem(
              value: () => m.moveStep(i, -1),
              child: const Text('Move earlier'),
            ),
          if (i + 1 < n)
            PopupMenuItem(
              value: () => m.moveStep(i, 1),
              child: const Text('Move later'),
            ),
          PopupMenuItem(
            value: () => _stepMenu(at, index: i + 1),
            child: const Text('Add a data step after…'),
          ),
          PopupMenuItem(
            value: () => m.removePart(card.part),
            child: const Text('Delete'),
          ),
        ]);
      } else if (card.part == DocPart.data && m.csv != null) {
        items.add(
          PopupMenuItem(
            value: () => _stepMenu(at),
            child: const Text('Add a data step…'),
          ),
        );
      } else if (card.part != DocPart.data && card.part != DocPart.fit) {
        items.add(
          PopupMenuItem(
            value: () => m.removePart(card.part),
            child: const Text('Delete'),
          ),
        );
      }
    } else {
      items.addAll([
        if (m.csv != null)
          PopupMenuItem(
            value: () => _stepMenu(at),
            child: const Text('Add a data step…'),
          ),
        PopupMenuItem(
          value: () => m.addTransform(at: c),
          child: const Text('Add a transform'),
        ),
        if (m.document.hasNetwork)
          PopupMenuItem(
            value: () => m.addOutput(at: c),
            child: const Text('Add an output'),
          ),
        if (m.document.response == null && m.document.hasNetwork)
          PopupMenuItem(
            value: () => m.addResponse(at: c),
            child: const Text('Add a response'),
          ),
        const PopupMenuDivider(),
        if (!m.document.hasNetwork)
          PopupMenuItem(
            value: () => m.addNetwork(at: c),
            child: const Text('Add an HGF network'),
          ),
        PopupMenuItem(
          value: () => m.addDeep(at: c),
          child: const Text('Add a deep network'),
        ),
      ]);
    }
    if (items.isEmpty) return;
    final pick = await _popup(at, items);
    pick?.call();
  }

  KeyEventResult _onKey(FocusNode node, KeyEvent e) {
    if (e is! KeyDownEvent && e is! KeyRepeatEvent) {
      return KeyEventResult.ignored;
    }
    final k = e.logicalKey;
    final hw = HardwareKeyboard.instance;
    final mod = hw.isControlPressed || hw.isMetaPressed;
    if (mod && k == LogicalKeyboardKey.keyZ) {
      hw.isShiftPressed ? m.redo() : m.undo();
      return KeyEventResult.handled;
    }
    if (mod && k == LogicalKeyboardKey.keyY) {
      m.redo();
      return KeyEventResult.handled;
    }
    if (mod) return KeyEventResult.ignored;
    if (k == LogicalKeyboardKey.delete || k == LogicalKeyboardKey.backspace) {
      m.deleteSelection();
      return KeyEventResult.handled;
    }
    if (k == LogicalKeyboardKey.escape) {
      setState(() => _drag = null);
      m.select(null);
      return KeyEventResult.handled;
    }
    if (k == LogicalKeyboardKey.keyF) {
      setState(() => _fit(_cards));
      return KeyEventResult.handled;
    }
    if (k == LogicalKeyboardKey.enter && m.selectedPart == DocPart.network) {
      m.showView(BuilderView.network);
      return KeyEventResult.handled;
    }
    if (k == LogicalKeyboardKey.enter &&
        (m.selectedPart?.startsWith('deep:') ?? false)) {
      m.showDeep(m.selectedPart!.substring('deep:'.length));
      return KeyEventResult.handled;
    }
    return KeyEventResult.ignored;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return LayoutBuilder(
      builder: (context, box) {
        _size = box.biggest;
        final geometry = pipelineGeometry(m);
        if (_fitted != m.opened) {
          _fitted = m.opened;
          _fit(geometry.cards);
        }
        final wire = _drag is _WireDrag ? _drag! as _WireDrag : null;
        return Focus(
          focusNode: _focus,
          onKeyEvent: _onKey,
          child: ClipRect(
            child: Stack(
              children: [
                Positioned.fill(
                  child: Listener(
                    onPointerDown: (e) => _downAt = e.localPosition,
                    onPointerSignal: (e) {
                      if (e is PointerScrollEvent) {
                        _zoom(
                          math.exp(-e.scrollDelta.dy / 400),
                          e.localPosition,
                        );
                      }
                    },
                    child: MouseRegion(
                      cursor: wire != null
                          ? SystemMouseCursors.precise
                          : _hoverPart != null
                          ? SystemMouseCursors.move
                          : SystemMouseCursors.basic,
                      onHover: (e) {
                        final p = _outPortAt(e.localPosition) != null
                            ? null
                            : _cardAt(e.localPosition)?.part;
                        if (p != _hoverPart) setState(() => _hoverPart = p);
                      },
                      onExit: (_) => setState(() => _hoverPart = null),
                      child: GestureDetector(
                        behavior: HitTestBehavior.opaque,
                        onTapUp: _onTapUp,
                        onSecondaryTapUp: (d) => _contextMenu(d.localPosition),
                        onLongPressStart: (d) => _contextMenu(d.localPosition),
                        onScaleStart: _onScaleStart,
                        onScaleUpdate: _onScaleUpdate,
                        onScaleEnd: _onScaleEnd,
                        child: CustomPaint(
                          size: Size.infinite,
                          painter: _PipelinePainter(
                            model: m,
                            cards: geometry.cards,
                            wires: geometry.wires,
                            pan: _pan,
                            scale: _scale,
                            wire: wire,
                            scheme: scheme,
                            text: theme.textTheme,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
                if (wire?.tip case final tip?)
                  Positioned(
                    left: (wire!.at * _scale + _pan).dx + 14,
                    top: (wire.at * _scale + _pan).dy + 10,
                    child: IgnorePointer(
                      child: Material(
                        elevation: 2,
                        borderRadius: BorderRadius.circular(6),
                        color: wire.ok
                            ? scheme.inverseSurface
                            : scheme.errorContainer,
                        child: Padding(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 8,
                            vertical: 5,
                          ),
                          child: Text(
                            tip,
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: wire.ok
                                  ? scheme.onInverseSurface
                                  : scheme.onErrorContainer,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                Positioned(
                  right: 8,
                  bottom: 8,
                  child: Material(
                    color: scheme.surfaceContainerHigh,
                    borderRadius: BorderRadius.circular(8),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        IconButton(
                          tooltip: 'Zoom out',
                          icon: const Icon(Icons.remove, size: 18),
                          onPressed: () =>
                              _zoom(0.8, _size.center(Offset.zero)),
                        ),
                        IconButton(
                          tooltip: 'Fit to view (F)',
                          icon: const Icon(Icons.fit_screen_outlined, size: 18),
                          onPressed: () => setState(() => _fit(geometry.cards)),
                        ),
                        IconButton(
                          tooltip: 'Zoom in',
                          icon: const Icon(Icons.add, size: 18),
                          onPressed: () =>
                              _zoom(1.25, _size.center(Offset.zero)),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _PipelinePainter extends CustomPainter {
  _PipelinePainter({
    required this.model,
    required this.cards,
    required this.wires,
    required this.pan,
    required this.scale,
    required this.wire,
    required this.scheme,
    required this.text,
  });

  final BuilderModel model;
  final List<PipelineCard> cards;
  final List<PipelineWire> wires;
  final Offset pan;
  final double scale;
  final _WireDrag? wire;
  final ColorScheme scheme;
  final TextTheme text;

  static const warning = Color(0xFFD08A00);

  Color? _issueColor(IssueSeverity? s) => switch (s) {
    IssueSeverity.error => scheme.error,
    IssueSeverity.warning => warning,
    null => null,
  };

  Color _accent(String part) => part == DocPart.data
      ? scheme.secondary
      : part == DocPart.network
      ? scheme.primary
      : part.startsWith('transform:')
      ? scheme.tertiary
      : part == DocPart.response
      ? const Color(0xFF1BAF7A)
      : part == DocPart.fit
      ? const Color(0xFF8E6AC8)
      : part.startsWith('deep:')
      ? const Color(0xFFD0743C)
      : scheme.outline;

  @override
  void paint(Canvas canvas, Size size) {
    final step = 40 * scale;
    if (step >= 12) {
      final dot = Paint()
        ..color = scheme.outlineVariant.withValues(alpha: 0.35);
      for (var x = pan.dx % step; x < size.width; x += step) {
        for (var y = pan.dy % step; y < size.height; y += step) {
          canvas.drawCircle(Offset(x, y), 0.8, dot);
        }
      }
    }
    canvas
      ..save()
      ..translate(pan.dx, pan.dy)
      ..scale(scale);
    final selected = model.selectedPart;
    for (final w in wires) {
      final hot = w.part == selected;
      final color = hot
          ? scheme.primary
          : _issueColor(model.partIssues[w.part]) ?? scheme.onSurfaceVariant;
      canvas.drawPath(
        wirePath(w.from, w.to),
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = hot ? 3 : 1.6
          ..color = color.withValues(alpha: hot ? 1 : 0.8),
      );
    }
    for (final c in cards) {
      _card(canvas, c, selected);
    }
    if (wire case final w?) {
      canvas.drawPath(
        wirePath(w.from, w.at),
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.2
          ..color = w.ok ? scheme.primary : scheme.error,
      );
    }
    canvas.restore();
  }

  void _card(Canvas canvas, PipelineCard c, String? selected) {
    final rect = c.rect;
    final rr = RRect.fromRectAndRadius(rect, const Radius.circular(8));
    final isSelected = c.part == selected;
    final issue = _issueColor(model.partIssues[c.part]);
    final accent = _accent(c.part);
    canvas
      ..drawRRect(
        rr.shift(const Offset(0, 1.5)),
        Paint()..color = scheme.shadow.withValues(alpha: 0.12),
      )
      ..drawRRect(rr, Paint()..color = scheme.surfaceContainerLow);
    // Header band.
    canvas
      ..save()
      ..clipRRect(rr)
      ..drawRect(
        Rect.fromLTWH(rect.left, rect.top, rect.width, cardHeader),
        Paint()..color = accent.withValues(alpha: 0.16),
      )
      ..drawRect(
        Rect.fromLTWH(rect.left, rect.top, 4, rect.height),
        Paint()..color = accent,
      )
      ..restore();
    canvas.drawRRect(
      rr,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = isSelected ? 2.6 : (issue != null ? 2.2 : 1)
        ..color = isSelected ? scheme.primary : issue ?? scheme.outlineVariant,
    );
    _text(
      canvas,
      c.at + const Offset(12, 5),
      c.title,
      text.labelLarge!.copyWith(
        color: scheme.onSurface,
        fontWeight: FontWeight.w600,
      ),
    );
    _text(
      canvas,
      c.at + const Offset(12, 22),
      c.subtitle,
      text.labelSmall!.copyWith(color: scheme.onSurfaceVariant),
    );
    for (final (k, row) in c.rows.indexed) {
      final centre = c.rowCentre(k);
      final rowPart = switch (row.port) {
        InputIn(:final input) => DocPart.input(input),
        _ => null,
      };
      final rowIssue = rowPart == null
          ? null
          : _issueColor(model.partIssues[rowPart]);
      if (rowPart != null && rowPart == selected) {
        canvas.drawRect(
          Rect.fromLTWH(
            c.at.dx + 4,
            centre.dy - cardRow / 2,
            cardWidth - 4,
            cardRow,
          ),
          Paint()..color = scheme.primary.withValues(alpha: 0.12),
        );
      }
      final style = text.bodySmall!.copyWith(
        color: rowIssue ?? scheme.onSurface,
      );
      _text(
        canvas,
        Offset(c.at.dx + 12, centre.dy - 8),
        row.label,
        style,
        maxWidth: row.tag == null ? cardWidth - 22 : cardWidth * 0.55,
      );
      if (row.tag case final tag?) {
        _text(
          canvas,
          Offset(c.at.dx + cardWidth - 10, centre.dy - 7),
          tag,
          text.labelSmall!.copyWith(color: scheme.onSurfaceVariant),
          maxWidth: cardWidth * 0.4,
          right: true,
        );
      }
      if (row.port != null) _port(canvas, c.rowPort(k), accent);
    }
    if (c.headerIn) _port(canvas, c.headerInPort, accent);
    if (c.headerOut != null || c.chainOut) {
      _port(canvas, c.headerOutPort, accent);
    }
  }

  void _port(Canvas canvas, Offset p, Color color) {
    canvas
      ..drawCircle(p, 5, Paint()..color = scheme.surface)
      ..drawCircle(
        p,
        5,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.8
          ..color = color,
      )
      ..drawCircle(p, 2.2, Paint()..color = color);
  }

  void _text(
    Canvas canvas,
    Offset at,
    String s,
    TextStyle style, {
    double maxWidth = cardWidth - 20,
    bool right = false,
  }) {
    final tp = TextPainter(
      text: TextSpan(text: s, style: style),
      textDirection: TextDirection.ltr,
      maxLines: 1,
      ellipsis: '…',
    )..layout(maxWidth: maxWidth);
    tp.paint(canvas, right ? at - Offset(tp.width, 0) : at);
  }

  @override
  bool shouldRepaint(_PipelinePainter old) => true;
}
