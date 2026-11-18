import 'dart:math' as math;

import 'package:flutter/gestures.dart';
import 'package:flutter/services.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:material_ui/material_ui.dart';

import '../../ui/panels.dart' show fmt;
import '../builder_model.dart';
import '../doc_layout.dart';
import 'canvas_support.dart';
import 'search_picker.dart';

/// A port of a node: where edges of one type attach. Top ports take
/// parents, bottom ports children (parents sit above their children).
enum Port {
  topValue(true, EdgeType.value),
  topVolatility(true, EdgeType.volatility),
  bottomValue(false, EdgeType.value),
  bottomVolatility(false, EdgeType.volatility);

  const Port(this.top, this.type);

  /// A parent port (else a child port).
  final bool top;
  final EdgeType type;
}

/// The ports a node offers: value and volatility parents and children for
/// a continuous node; value edges only for a binary one; a constant node
/// (a bias) only has value children, an ef-state node (a dp-state
/// cluster) one value child, a dp-state node only its clusters. A
/// categorical node's ports connect to its implied nodes.
List<Port> portsOf(DocNode n) => switch (n) {
  ScalarNode(kind: NodeKind.binaryState) => const [
    Port.topValue,
    Port.bottomValue,
  ],
  ScalarNode(kind: NodeKind.constantState) => const [Port.bottomValue],
  EfStateNode() => const [Port.bottomValue],
  DpStateNode() => const [Port.topValue],
  _ => Port.values,
};

/// The radius of a node's shape, in canvas units.
double nodeRadius(DocNode n) => switch (n) {
  ScalarNode(kind: NodeKind.constantState) => 13,
  ScalarNode() => 22,
  _ => 27,
};

/// Where [port] of [n] (centred at [c]) sits, in canvas units.
Offset portPosition(DocNode n, Offset c, Port port) {
  final r = nodeRadius(n);
  final ports = portsOf(n);
  final pair = ports.where((p) => p.top == port.top).length > 1;
  final dx = !pair ? 0.0 : (port.type == EdgeType.value ? -11.0 : 11.0);
  return c + Offset(dx, port.top ? -r : r);
}

/// The canvas of the model builder: the HGF graph, parents above their
/// children. Value edges are solid, volatility edges dashed (and tinted);
/// each node has typed ports (filled: value, hollow: volatility) on top
/// for parents and below for children. Drag from a port to another node
/// to connect them (green: allowed; red: the reason why not), drag a node
/// to move it, drag the background to pan, scroll or pinch to zoom.
/// Double-click the background to add a continuous node, right-click (or
/// long-press) for more. Delete removes the selection, Ctrl+Z / Ctrl+Shift+Z
/// undo and redo, F fits the view.
class BuilderCanvas extends StatefulWidget {
  const BuilderCanvas({super.key, required this.model});

  final BuilderModel model;

  @override
  State<BuilderCanvas> createState() => _BuilderCanvasState();
}

sealed class _Drag {}

final class _PanDrag extends _Drag {
  _PanDrag(this.pan, this.scale, this.focal);
  final Offset pan, focal;
  final double scale;
}

final class _NodeDrag extends _Drag {
  _NodeDrag(this.id, this.grab);
  final String id;

  /// Pointer minus node centre, in canvas units.
  final Offset grab;
  bool moved = false;
}

final class _WireDrag extends _Drag {
  _WireDrag(this.id, this.port, this.at);
  final String id;
  final Port port;

  /// The pointer, in canvas units.
  Offset at;

  /// The node under the pointer, and whether the edge to it is allowed.
  String? target;
  ConnectionCheck? check;

  /// The target offers a choice of implied nodes.
  bool choose = false;
}

class _BuilderCanvasState extends State<BuilderCanvas> {
  final _focus = FocusNode(debugLabel: 'builder canvas');
  Offset _pan = Offset.zero;
  double _scale = 1;
  int _fitted = -1;
  Size _size = Size.zero;
  _Drag? _drag;

  /// Where the last pointer went down: the scale recogniser starts only
  /// after the touch slop, too late to hit a port.
  Offset? _downAt;
  final _right = RightClicks();

  /// The last tap on the background (time and place), for double-clicks.
  (Duration, Offset)? _lastTap;
  ({String? node, Port? port, int? edge})? _hover;

  BuilderModel get m => widget.model;

  /// The model and its positions (moves notify only the latter).
  late Listenable _source = Listenable.merge([m, m.layoutChanges]);

  @override
  void initState() {
    super.initState();
    _source.addListener(_changed);
  }

  @override
  void didUpdateWidget(BuilderCanvas old) {
    super.didUpdateWidget(old);
    if (old.model != m) {
      _source.removeListener(_changed);
      _source = Listenable.merge([m, m.layoutChanges])..addListener(_changed);
    }
  }

  @override
  void dispose() {
    _source.removeListener(_changed);
    _focus.dispose();
    super.dispose();
  }

  void _changed() => setState(() {});

  Offset _toCanvas(Offset screen) => (screen - _pan) / _scale;
  Offset _toScreen(Offset canvas) => canvas * _scale + _pan;

  void _fit() {
    if (_size.isEmpty) return;
    final b = boundsOf(m.positions.values);
    if (b == null) {
      _pan = _size.center(Offset.zero);
      _scale = 1;
      return;
    }
    final box = b.inflate(70);
    _scale = math
        .min(_size.width / box.width, _size.height / box.height)
        .clamp(0.3, 1.4);
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
  // Hit tests (screen coordinates in, canvas coordinates inside)
  // ---------------------------------------------------------------------------

  ({String id, Port port})? _portAt(Offset screen) {
    final p = _toCanvas(screen);
    final reach = math.max(9.0, 12 / _scale);
    ({String id, Port port})? best;
    var bestD = reach * reach;
    for (final n in m.doc.nodes) {
      final c = m.positions[n.id];
      if (c == null) continue;
      for (final port in portsOf(n)) {
        final d = (portPosition(n, c, port) - p).distanceSquared;
        if (d < bestD) {
          bestD = d;
          best = (id: n.id, port: port);
        }
      }
    }
    return best;
  }

  String? _nodeAt(Offset screen, {String? except}) {
    final p = _toCanvas(screen);
    for (final n in m.doc.nodes.reversed) {
      if (n.id == except) continue;
      final c = m.positions[n.id];
      if (c == null) continue;
      final r = nodeRadius(n) + 4;
      if ((c - p).distanceSquared <= r * r) return n.id;
    }
    return null;
  }

  int? _edgeAt(Offset screen) {
    final p = _toCanvas(screen);
    int? best;
    var bestD = math.max(5.0, 8 / _scale);
    for (final (i, e) in m.doc.edges.indexed) {
      final g = edgeGeometry(m, e);
      if (g == null) continue;
      final d = curveDistance(p, g.at);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  // ---------------------------------------------------------------------------
  // Gestures
  // ---------------------------------------------------------------------------

  void _onScaleStart(ScaleStartDetails d) {
    _focus.requestFocus();
    if (_right.active) {
      _drag = null; // a right click's drag: neither a pan nor a move
      return;
    }
    final at = d.pointerCount == 1
        ? (_downAt ?? d.localFocalPoint)
        : d.localFocalPoint;
    if (d.pointerCount == 1) {
      if (_portAt(at) case final p?) {
        setState(() => _drag = _WireDrag(p.id, p.port, _toCanvas(at)));
        return;
      }
      if (_nodeAt(at) case final id?) {
        m.checkpoint();
        m.select(NodeSelection(id));
        _drag = _NodeDrag(id, _toCanvas(at) - m.positions[id]!);
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
        // Keep the canvas point under the starting focal point under the
        // fingers.
        final c = (focal - pan) / scale;
        setState(() {
          _scale = s;
          _pan = at - c * s;
        });
      case final _NodeDrag drag:
        drag.moved = true;
        m.moveNode(drag.id, _toCanvas(at) - drag.grab);
      case final _WireDrag w:
        w.at = _toCanvas(at);
        final target = _nodeAt(at, except: w.id);
        if (target != w.target) {
          w.target = target;
          w.check = null;
          w.choose = false;
          if (target != null) {
            final candidates = _candidates(w, target);
            if (candidates.length > 1 || _isBlock(w.id) || _isBlock(target)) {
              final ok = candidates.where((c) => c.problem == null);
              w.choose = ok.isNotEmpty;
              w.check = ok.isNotEmpty ? ok.first : candidates.first;
            } else {
              w.check = candidates.single;
            }
          }
        }
        setState(() {});
      case null:
    }
  }

  void _onScaleEnd(ScaleEndDetails d) {
    final drag = _drag;
    _drag = null;
    if (drag is _WireDrag) {
      setState(() {});
      final target = drag.target;
      if (target == null) return;
      final candidates = _candidates(drag, target);
      final ok = [
        for (final c in candidates)
          if (c.problem == null) c,
      ];
      if (ok.length == 1 && !_isBlock(drag.id) && !_isBlock(target)) {
        m.addEdge(ok.single.edge);
      } else if (ok.isNotEmpty) {
        _chooseEdge(ok, _toScreen(drag.at));
      } else {
        tellCanvas(context, candidates.first.problem!);
      }
    }
  }

  bool _isBlock(String id) => (m.doc.node(id)?.size ?? 1) > 1;

  List<String> _addressesOf(String id) {
    final n = m.doc.node(id)!;
    return [id, for (var k = 1; k < n.size; k++) '$id.$k'];
  }

  /// The edges a wire from [w] to [target] could make, each checked.
  List<ConnectionCheck> _candidates(_WireDrag w, String target) {
    final out = <ConnectionCheck>[];
    for (final a in _addressesOf(w.id)) {
      for (final b in _addressesOf(target)) {
        final (parent, child) = w.port.top ? (b, a) : (a, b);
        out.add(m.checkConnection(parent, child, w.port.type));
      }
    }
    // Allowed first, then by how close to the heads they are.
    out.sort((x, y) {
      final ax = x.problem == null ? 0 : 1, ay = y.problem == null ? 0 : 1;
      return ax - ay;
    });
    return out;
  }

  Future<void> _chooseEdge(List<ConnectionCheck> ok, Offset screen) async {
    String describe(String address) {
      final dot = address.indexOf('.');
      if (dot < 0) return prettyId(address);
      final node = m.doc.node(address.substring(0, dot))!;
      final k = int.parse(address.substring(dot + 1));
      return '${prettyId(address)} (${impliedNodeName(node, k)})';
    }

    String label(DocEdge e) => '${describe(e.parent)} → ${describe(e.child)}';
    if (ok.length > searchPickerAbove) {
      final pick = await showSearchPicker<DocEdge>(
        context,
        title: 'Connect which nodes?',
        items: {for (final c in ok) c.edge: label(c.edge)},
      );
      if (pick case (final e,)) m.addEdge(e);
      return;
    }
    final pick = await showMenuAt<DocEdge>(context, screen, [
      for (final c in ok)
        PopupMenuItem(value: c.edge, child: Text(label(c.edge))),
    ]);
    if (pick != null) m.addEdge(pick);
  }

  void _onTapUp(TapUpDetails d) {
    _focus.requestFocus();
    final at = d.localPosition;
    if (_nodeAt(at) case final id?) {
      _lastTap = null;
      m.select(NodeSelection(id));
    } else if (_edgeAt(at) case final i?) {
      _lastTap = null;
      m.select(EdgeSelection(i));
    } else {
      // A double-click on the background adds a node (detected here: a
      // double-tap recogniser would hold every tap back).
      final now = _clock.elapsed;
      if (_lastTap case (final t, final p)
          when now - t < kDoubleTapTimeout &&
              (p - at).distance < kDoubleTapSlop) {
        _lastTap = null;
        m.addNode(NodeKind.continuousState, at: _toCanvas(at));
        return;
      }
      _lastTap = (now, at);
      m.select(null);
    }
  }

  static final _clock = Stopwatch()..start();

  Future<void> _contextMenu(Offset at) async {
    _focus.requestFocus();
    final node = _nodeAt(at);
    final edge = node == null ? _edgeAt(at) : null;
    final items = <PopupMenuEntry<VoidCallback>>[];
    if (node != null) {
      m.select(NodeSelection(node));
      final n = m.doc.node(node)!;
      final ports = portsOf(n);
      if (n is ScalarNode) {
        for (final (port, label) in [
          (Port.topValue, 'Add a value parent'),
          (Port.topVolatility, 'Add a volatility parent'),
          (Port.bottomValue, 'Add a value child'),
          (Port.bottomVolatility, 'Add a volatility child'),
        ]) {
          if (!ports.contains(port)) continue;
          items.add(
            PopupMenuItem(
              value: () => m.addLinked(node, port.type, below: !port.top),
              child: Text(label),
            ),
          );
        }
        items.add(const PopupMenuDivider());
      }
      items.add(
        PopupMenuItem(
          value: () => m.removeNode(node),
          child: const Text('Delete node'),
        ),
      );
    } else if (edge != null) {
      m.select(EdgeSelection(edge));
      final e = m.doc.edges[edge];
      final other = e.type == EdgeType.value
          ? EdgeType.volatility
          : EdgeType.value;
      items
        ..add(
          PopupMenuItem(
            value: () => m.replaceEdge(
              edge,
              DocEdge(other, e.parent, e.child, strength: e.strength),
            ),
            child: Text('Make it a ${other.name} edge'),
          ),
        )
        ..add(
          PopupMenuItem(
            value: () => m.removeEdge(edge),
            child: const Text('Delete edge'),
          ),
        );
    } else {
      final c = _toCanvas(at);
      for (final kind in NodeKind.values) {
        items.add(
          PopupMenuItem(
            value: () => m.addNode(kind, at: c),
            child: Text('Add a ${kindName(kind)} node'),
          ),
        );
      }
    }
    final pick = await showMenuAt(context, at, items);
    pick?.call();
  }

  void _onHover(PointerHoverEvent e) {
    final at = e.localPosition;
    final port = _portAt(at);
    final node = port == null ? _nodeAt(at) : null;
    final edge = port == null && node == null ? _edgeAt(at) : null;
    final h = (node: port?.id ?? node, port: port?.port, edge: edge);
    if (h != _hover) setState(() => _hover = h);
  }

  KeyEventResult _onKey(FocusNode node, KeyEvent e) =>
      canvasKey(
        m,
        e,
        onEscape: () => setState(() => _drag = null),
        onFit: () => setState(_fit),
      ) ??
      KeyEventResult.ignored;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return LayoutBuilder(
      builder: (context, box) {
        _size = box.biggest;
        if (_fitted != m.opened) {
          _fitted = m.opened;
          _fit();
        }
        final wire = _drag is _WireDrag ? _drag! as _WireDrag : null;
        final cursor = wire != null
            ? SystemMouseCursors.precise
            : _hover?.port != null
            ? SystemMouseCursors.precise
            : _hover?.node != null
            ? SystemMouseCursors.move
            : _hover?.edge != null
            ? SystemMouseCursors.click
            : SystemMouseCursors.basic;
        return Focus(
          focusNode: _focus,
          onKeyEvent: _onKey,
          child: ClipRect(
            child: Stack(
              children: [
                Positioned.fill(
                  child: NoBrowserContextMenu(
                    child: Listener(
                      onPointerDown: (e) {
                        _downAt = e.localPosition;
                        _right.down(e);
                      },
                      onPointerUp: (e) => _right.up(e, _contextMenu),
                      onPointerCancel: (_) => _right.cancel(),
                      onPointerSignal: (e) {
                        if (e is PointerScrollEvent) {
                          _zoom(
                            math.exp(-e.scrollDelta.dy / 400),
                            e.localPosition,
                          );
                        }
                      },
                      child: MouseRegion(
                        cursor: cursor,
                        onHover: _onHover,
                        onExit: (_) => setState(() => _hover = null),
                        child: GestureDetector(
                          behavior: HitTestBehavior.opaque,
                          onTapUp: _onTapUp,
                          onLongPressStart: (d) =>
                              _contextMenu(d.localPosition),
                          onScaleStart: _onScaleStart,
                          onScaleUpdate: _onScaleUpdate,
                          onScaleEnd: _onScaleEnd,
                          child: RepaintBoundary(
                            child: CustomPaint(
                              size: Size.infinite,
                              painter: _CanvasPainter(
                                model: m,
                                pan: _pan,
                                scale: _scale,
                                hover: _hover,
                                wire: wire,
                                scheme: scheme,
                                text: theme.textTheme,
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
                if (wire?.check case final c? when wire!.target != null)
                  Positioned(
                    left: _toScreen(wire.at).dx + 14,
                    top: _toScreen(wire.at).dy + 10,
                    child: _WireTip(
                      text: wire.choose
                          ? 'Release to choose which node'
                          : c.problem ?? 'Release to connect',
                      ok: c.problem == null,
                    ),
                  ),
                Positioned(left: 8, bottom: 8, child: _Legend(scheme: scheme)),
                Positioned(
                  right: 8,
                  bottom: 8,
                  child: CanvasZoomButtons(
                    onIn: () => _zoom(1.25, _size.center(Offset.zero)),
                    onOut: () => _zoom(0.8, _size.center(Offset.zero)),
                    onFit: () => setState(_fit),
                  ),
                ),
                if (m.doc.nodes.isEmpty)
                  Center(
                    child: Text(
                      'Double-click to add a node, or right-click (long-press) for more',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: scheme.onSurfaceVariant,
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

/// A cubic from a parent's child port to a child's parent port.
class EdgeGeometry {
  EdgeGeometry(this.p0, this.p1, this.p2, this.p3);
  final Offset p0, p1, p2, p3;

  Offset at(double t) {
    final u = 1 - t;
    return p0 * (u * u * u) +
        p1 * (3 * u * u * t) +
        p2 * (3 * u * t * t) +
        p3 * (t * t * t);
  }

  Path get path => Path()
    ..moveTo(p0.dx, p0.dy)
    ..cubicTo(p1.dx, p1.dy, p2.dx, p2.dy, p3.dx, p3.dy);

  /// [path] in dashes (7 on, 5 off), as one path; kept for the curves
  /// drawn last, since measuring a curve every frame is slow on the web.
  Path get dashedPath {
    final key = (p0, p1, p2, p3);
    if (_dashes[key] case final cached?) return cached;
    final out = Path();
    for (final metric in path.computeMetrics()) {
      for (var d = 0.0; d < metric.length; d += 12) {
        out.addPath(
          metric.extractPath(d, math.min(d + 7, metric.length)),
          Offset.zero,
        );
      }
    }
    if (_dashes.length > 256) _dashes.clear();
    return _dashes[key] = out;
  }

  static final _dashes = <(Offset, Offset, Offset, Offset), Path>{};

  /// The direction the curve arrives at [p3].
  Offset get endTangent {
    final d = p3 - p2;
    return d.distance == 0 ? const Offset(0, 1) : d / d.distance;
  }
}

/// Where [e] is drawn (null if an endpoint is missing).
EdgeGeometry? edgeGeometry(BuilderModel m, DocEdge e) {
  final pn = m.doc.node(ownerIdOf(e.parent)),
      cn = m.doc.node(ownerIdOf(e.child));
  if (pn == null || cn == null) return null;
  final pc = m.positions[pn.id], cc = m.positions[cn.id];
  if (pc == null || cc == null) return null;
  final value = e.type == EdgeType.value;
  Offset end(DocNode n, Offset c, bool top) {
    final want = top
        ? (value ? Port.topValue : Port.topVolatility)
        : (value ? Port.bottomValue : Port.bottomVolatility);
    if (portsOf(n).contains(want)) return portPosition(n, c, want);
    return c + Offset(0, top ? -nodeRadius(n) : nodeRadius(n));
  }

  final a = end(pn, pc, false), b = end(cn, cc, true);
  final k = math.max(30.0, (b.dy - a.dy).abs() / 2);
  return EdgeGeometry(a, a + Offset(0, k), b - Offset(0, k), b);
}

class _CanvasPainter extends CustomPainter {
  _CanvasPainter({
    required this.model,
    required this.pan,
    required this.scale,
    required this.hover,
    required this.wire,
    required this.scheme,
    required this.text,
  });

  final BuilderModel model;
  final Offset pan;
  final double scale;
  final ({String? node, Port? port, int? edge})? hover;
  final _WireDrag? wire;
  final ColorScheme scheme;
  final TextTheme text;

  static const warning = Color(0xFFD08A00);

  Color get valueColor => scheme.onSurfaceVariant;
  Color get volatilityColor => scheme.tertiary;

  Color? _issueColor(IssueSeverity? s) => switch (s) {
    IssueSeverity.error => scheme.error,
    IssueSeverity.warning => warning,
    null => null,
  };

  @override
  void paint(Canvas canvas, Size size) {
    _grid(canvas, size);
    canvas.save();
    canvas.translate(pan.dx, pan.dy);
    canvas.scale(scale);
    final m = model;
    // Inputs: nodes without children.
    final hasChildren = <String>{
      for (final e in m.doc.edges) ownerIdOf(e.parent),
    };
    for (final (i, e) in m.doc.edges.indexed) {
      _edge(canvas, i, e);
    }
    for (final n in m.doc.nodes) {
      _node(canvas, n, input: !hasChildren.contains(n.id));
    }
    if (wire case final w?) _wire(canvas, w);
    canvas.restore();
  }

  void _grid(Canvas canvas, Size size) => paintDotGrid(
    canvas,
    size,
    pan,
    scale,
    scheme.outlineVariant.withValues(alpha: 0.35),
    radius: math.max(1.0, 1.2 * scale.clamp(0.5, 1.5)) / 1.5,
  );

  void _edge(Canvas canvas, int i, DocEdge e) {
    final g = edgeGeometry(model, e);
    if (g == null) return;
    final value = e.type == EdgeType.value;
    final selected = model.selectedEdge == i;
    final hovered = hover?.edge == i;
    final color = selected
        ? scheme.primary
        : _issueColor(model.edgeIssues[i]) ??
              (value ? valueColor : volatilityColor);
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = selected || hovered ? 3 : 1.8;
    if (value) {
      canvas.drawPath(g.path, paint);
    } else {
      _dashed(canvas, g, paint);
    }
    _arrow(canvas, g.p3, g.endTangent, color);
    final mid = g.at(0.5);
    final labels = [if (e.strength != 1) '×${fmt(e.strength)}'];
    if (labels.isNotEmpty) {
      _label(canvas, mid + const Offset(10, 0), labels.join(' '), color);
    }
    // Implied endpoints: say which.
    for (final (address, at) in [
      (e.parent, g.at(0.12)),
      (e.child, g.at(0.88)),
    ]) {
      final dot = address.indexOf('.');
      if (dot < 0) continue;
      _label(canvas, at + const Offset(8, 0), address.substring(dot), color);
    }
  }

  void _dashed(Canvas canvas, EdgeGeometry g, Paint paint) =>
      canvas.drawPath(g.dashedPath, paint);

  void _arrow(Canvas canvas, Offset tip, Offset u, Color color) {
    final n = Offset(-u.dy, u.dx);
    final path = Path()
      ..moveTo(tip.dx, tip.dy)
      ..lineTo((tip - u * 9 + n * 4.5).dx, (tip - u * 9 + n * 4.5).dy)
      ..lineTo((tip - u * 9 - n * 4.5).dx, (tip - u * 9 - n * 4.5).dy)
      ..close();
    canvas.drawPath(path, Paint()..color = color);
  }

  void _node(Canvas canvas, DocNode n, {required bool input}) {
    final c = model.positions[n.id];
    if (c == null) return;
    final r = nodeRadius(n);
    final selected = model.selectedNode == n.id;
    final issue = _issueColor(model.nodeIssues[n.id]);
    final target = wire?.target == n.id ? wire : null;
    final shape = _shape(n, c, r);
    final fill = input ? scheme.primaryContainer : scheme.surface;
    canvas.drawPath(shape, Paint()..color = fill);
    if (target != null) {
      final ok = target.check?.problem == null;
      canvas.drawPath(
        shape,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 6
          ..color = (ok ? const Color(0xFF1BAF7A) : scheme.error).withValues(
            alpha: 0.45,
          ),
      );
    }
    canvas.drawPath(
      shape,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = selected ? 3 : (issue != null ? 2.4 : 1.6)
        ..color = selected ? scheme.primary : issue ?? scheme.onSurface,
    );
    final style = text.labelLarge!.copyWith(
      color: input ? scheme.onPrimaryContainer : scheme.onSurface,
      fontWeight: FontWeight.w600,
    );
    _text(canvas, c, prettyId(n.label ?? n.id), style, maxWidth: 2 * r + 20);
    // Caption: the live belief, or what a block holds.
    final caption = switch (n) {
      CategoricalNode(:final nCategories) => '$nCategories categories',
      DpStateNode(:final batchSize) => 'dp · $batchSize clusters',
      EfStateNode(:final nNodes) => nNodes > 1 ? 'ef ×$nNodes' : 'ef',
      ScalarNode(kind: NodeKind.constantState) => 'bias',
      _ => switch (model.lastMean(n.id)) {
        final v? => 'μ ${fmt(v)}',
        null => null,
      },
    };
    if (caption != null) {
      // Beside the node: edges and ports leave its top and bottom.
      _text(
        canvas,
        c + Offset(r + 8, 0),
        caption,
        text.labelSmall!.copyWith(color: scheme.onSurfaceVariant),
        leftAligned: true,
      );
    }
    // Ports: bigger when hovered, or when wiring could use them.
    final show =
        selected || hover?.node == n.id || (wire != null && wire!.id == n.id);
    for (final port in portsOf(n)) {
      final p = portPosition(n, c, port);
      final hot =
          hover?.node == n.id && hover?.port == port ||
          wire?.id == n.id && wire?.port == port;
      final rad = hot ? 6.0 : (show ? 4.5 : 3.0);
      final col = port.type == EdgeType.value ? valueColor : volatilityColor;
      if (port.type == EdgeType.value) {
        canvas.drawCircle(p, rad, Paint()..color = col);
      } else {
        canvas.drawCircle(p, rad, Paint()..color = scheme.surface);
        canvas.drawCircle(
          p,
          rad,
          Paint()
            ..style = PaintingStyle.stroke
            ..strokeWidth = 1.8
            ..color = col,
        );
      }
    }
  }

  void _wire(Canvas canvas, _WireDrag w) {
    final n = model.doc.node(w.id);
    final c = model.positions[w.id];
    if (n == null || c == null) return;
    final from = portPosition(n, c, w.port);
    final to = w.at;
    final ok = w.target == null || w.check?.problem == null;
    final color = w.target == null
        ? scheme.primary
        : ok
        ? const Color(0xFF1BAF7A)
        : scheme.error;
    final dir = w.port.top ? -1.0 : 1.0;
    final k = math.max(30.0, (to.dy - from.dy).abs() / 2);
    final g = EdgeGeometry(
      from,
      from + Offset(0, dir * k),
      to - Offset(0, dir * k),
      to,
    );
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.2;
    if (w.port.type == EdgeType.value) {
      canvas.drawPath(g.path, paint);
    } else {
      _dashed(canvas, g, paint);
    }
  }

  Path _shape(DocNode n, Offset c, double r) {
    switch (n) {
      case ScalarNode(kind: NodeKind.binaryState):
        return Path()..addRRect(
          RRect.fromRectAndRadius(
            Rect.fromCircle(center: c, radius: r * 0.9),
            const Radius.circular(4),
          ),
        );
      case EfStateNode():
        return Path()
          ..moveTo(c.dx, c.dy - r)
          ..lineTo(c.dx + r, c.dy)
          ..lineTo(c.dx, c.dy + r)
          ..lineTo(c.dx - r, c.dy)
          ..close();
      case CategoricalNode() || DpStateNode():
        final path = Path();
        for (var k = 0; k < 6; k++) {
          final a = math.pi / 6 + k * math.pi / 3;
          final p = c + Offset(math.cos(a), math.sin(a)) * r;
          k == 0 ? path.moveTo(p.dx, p.dy) : path.lineTo(p.dx, p.dy);
        }
        return path..close();
      case ScalarNode():
        return Path()..addOval(Rect.fromCircle(center: c, radius: r));
    }
  }

  void _label(Canvas canvas, Offset at, String s, Color color) {
    final tp = canvasText(s, text.labelSmall!.copyWith(color: color));
    final rect = Rect.fromLTWH(
      at.dx - 2,
      at.dy - tp.height / 2 - 1,
      tp.width + 4,
      tp.height + 2,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(3)),
      Paint()..color = scheme.surface.withValues(alpha: 0.85),
    );
    tp.paint(canvas, Offset(at.dx, at.dy - tp.height / 2));
  }

  void _text(
    Canvas canvas,
    Offset c,
    String s,
    TextStyle style, {
    double maxWidth = 160,
    bool leftAligned = false,
  }) {
    final tp = canvasText(s, style, maxWidth: maxWidth);
    tp.paint(canvas, c - Offset(leftAligned ? 0 : tp.width / 2, tp.height / 2));
  }

  @override
  bool shouldRepaint(_CanvasPainter old) => true;
}

class _WireTip extends StatelessWidget {
  const _WireTip({required this.text, required this.ok});

  final String text;
  final bool ok;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return IgnorePointer(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 300),
        child: Material(
          elevation: 2,
          borderRadius: BorderRadius.circular(6),
          color: ok ? scheme.inverseSurface : scheme.errorContainer,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
            child: Text(
              text,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: ok ? scheme.onInverseSurface : scheme.onErrorContainer,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _Legend extends StatelessWidget {
  const _Legend({required this.scheme});

  final ColorScheme scheme;

  @override
  Widget build(BuildContext context) {
    final style = Theme.of(
      context,
    ).textTheme.labelSmall?.copyWith(color: scheme.onSurfaceVariant);
    Widget line(bool dashed, Color c, String label) => Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        CustomPaint(size: const Size(26, 10), painter: _LinePainter(dashed, c)),
        const SizedBox(width: 6),
        Text(label, style: style),
      ],
    );
    return IgnorePointer(
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: scheme.surface.withValues(alpha: 0.85),
          borderRadius: BorderRadius.circular(6),
        ),
        child: Padding(
          padding: const EdgeInsets.all(6),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              line(false, scheme.onSurfaceVariant, 'value'),
              line(true, scheme.tertiary, 'volatility'),
            ],
          ),
        ),
      ),
    );
  }
}

class _LinePainter extends CustomPainter {
  _LinePainter(this.dashed, this.color);
  final bool dashed;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final p = Paint()
      ..color = color
      ..strokeWidth = 1.8;
    final y = size.height / 2;
    if (!dashed) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), p);
      return;
    }
    for (var x = 0.0; x < size.width; x += 8) {
      canvas.drawLine(Offset(x, y), Offset(math.min(x + 5, size.width), y), p);
    }
  }

  @override
  bool shouldRepaint(_LinePainter old) =>
      old.dashed != dashed || old.color != color;
}
