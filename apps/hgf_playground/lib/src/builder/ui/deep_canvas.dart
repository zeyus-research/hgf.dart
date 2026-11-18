import 'dart:math' as math;

import 'package:hgf_deep/hgf_deep.dart' show LayerKind;
import 'package:hgf_fit/document.dart';
import 'package:material_ui/material_ui.dart';

import '../builder_model.dart';
import 'canvas_support.dart';

/// At most this many nodes are drawn per layer (the rest as "+n").
const int _maxDots = 24;
const double _dot = 7, _dotGap = 18, _barHeight = 62, _plus = 12;

/// Where the deep canvas draws each layer and each "insert" button.
final class DeepGeometry {
  DeepGeometry(DocDeep d, Size size) {
    final n = d.layers.length;
    final gap = n <= 1
        ? 0.0
        : math.max(36.0, (size.height - 24 - n * _barHeight) / (n - 1));
    final cx = size.width / 2;
    final maxW = math.max(160.0, size.width - 32);
    // Layer n - 1 (the features) on top, layer 0 (the targets) at the bottom.
    for (var i = 0; i < n; i++) {
      final shown = math.min(d.layers[i].size, _maxDots);
      final w = math.min(maxW, math.max(220.0, shown * _dotGap + 80));
      final top = 12 + (n - 1 - i) * (_barHeight + gap);
      bars.add(Rect.fromLTWH(cx - w / 2, top, w, _barHeight));
    }
    for (var i = 0; i + 1 < n; i++) {
      plus.add(Offset(cx, (bars[i + 1].bottom + bars[i].top) / 2));
    }
    height = n == 0 ? 0 : bars.first.bottom + 12;
  }

  /// One per layer, by index.
  final bars = <Rect>[];

  /// The insert buttons: `plus[i]` inserts a layer above layer `i`.
  final plus = <Offset>[];
  late final double height;

  /// The centres of layer [i]'s drawn nodes.
  List<Offset> dots(DocDeep d, int i) {
    final bar = bars[i];
    final shown = math.min(d.layers[i].size, _maxDots);
    final y = bar.bottom - 16;
    final width = (shown - 1) * _dotGap;
    final x0 = bar.center.dx - width / 2;
    return [for (var k = 0; k < shown; k++) Offset(x0 + k * _dotGap, y)];
  }

  int? layerAt(Offset p) {
    for (var i = 0; i < bars.length; i++) {
      if (bars[i].inflate(4).contains(p)) return i;
    }
    return null;
  }

  int? plusAt(Offset p) {
    for (var i = 0; i < plus.length; i++) {
      if ((plus[i] - p).distance <= _plus + 6) return i;
    }
    return null;
  }
}

/// The layers of the open deep network, as pyhgf stacks them: the
/// features on top, the targets at the bottom. Tap a layer to edit it;
/// "+" between two layers inserts a hidden layer; Delete removes the
/// selected hidden layer.
class DeepCanvas extends StatefulWidget {
  const DeepCanvas({super.key, required this.model});

  final BuilderModel model;

  @override
  State<DeepCanvas> createState() => _DeepCanvasState();
}

class _DeepCanvasState extends State<DeepCanvas> {
  final _focus = FocusNode();
  final _scroll = ScrollController();
  int? _hoverPlus;

  BuilderModel get m => widget.model;

  @override
  void dispose() {
    _focus.dispose();
    _scroll.dispose();
    super.dispose();
  }

  KeyEventResult _onKey(FocusNode node, KeyEvent e) =>
      canvasKey(m, e) ?? KeyEventResult.ignored;

  @override
  Widget build(BuildContext context) {
    final d = m.deep;
    final theme = Theme.of(context);
    if (d == null) {
      return const Center(child: Text('This deep network was removed.'));
    }
    return Focus(
      focusNode: _focus,
      onKeyEvent: _onKey,
      child: LayoutBuilder(
        builder: (context, box) {
          final g = DeepGeometry(
            d,
            Size(box.maxWidth, math.max(box.maxHeight, 0)),
          );
          final size = Size(box.maxWidth, math.max(g.height, box.maxHeight));
          return Scrollbar(
            controller: _scroll,
            child: SingleChildScrollView(
              controller: _scroll,
              child: MouseRegion(
                onHover: (e) {
                  final p = g.plusAt(e.localPosition);
                  if (p != _hoverPlus) setState(() => _hoverPlus = p);
                },
                onExit: (_) => setState(() => _hoverPlus = null),
                cursor: _hoverPlus != null
                    ? SystemMouseCursors.click
                    : SystemMouseCursors.basic,
                child: GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onTapUp: (t) {
                    _focus.requestFocus();
                    final p = t.localPosition;
                    if (g.plusAt(p) case final i?) {
                      m.insertLayer(i);
                    } else if (g.layerAt(p) case final i?) {
                      m.select(LayerSelection(i));
                    } else {
                      m.select(null);
                    }
                  },
                  child: CustomPaint(
                    size: size,
                    painter: _DeepPainter(
                      deep: d,
                      geometry: g,
                      selected: m.selectedLayer,
                      hoverPlus: _hoverPlus,
                      scheme: theme.colorScheme,
                      text: theme.textTheme,
                      revision: m.revision,
                    ),
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

/// A short description of layer [i] of [d]: its size, kind and
/// activation, and what it carries.
String layerSummary(DocDeep d, int i) {
  final l = d.layers[i];
  final parts = <String>[
    '${l.size} node${l.size == 1 ? '' : 's'}',
    switch (l.kind) {
      LayerKind.binary => 'binary',
      LayerKind.categorical => 'categorical',
      LayerKind.continuous => 'continuous',
      LayerKind.volatile => i == 0 ? 'continuous target' : 'volatile',
    },
  ];
  if (i > 0 && !d.isContinuous) {
    parts.add(l.coupling ?? d.settings.coupling);
  }
  if (!l.fullyConnected) parts.add('one-to-one');
  if (l.volatilityParent == false && l.kind == LayerKind.volatile) {
    parts.add('no volatility level');
  }
  return parts.join(' · ');
}

class _DeepPainter extends CustomPainter {
  _DeepPainter({
    required this.deep,
    required this.geometry,
    required this.selected,
    required this.hoverPlus,
    required this.scheme,
    required this.text,
    required this.revision,
  });

  final DocDeep deep;
  final DeepGeometry geometry;
  final int? selected, hoverPlus;
  final ColorScheme scheme;
  final TextTheme text;
  final int revision;

  Color _accent(int i) => i == 0
      ? scheme.tertiary
      : i == deep.layers.length - 1 && !deep.isContinuous
      ? scheme.secondary
      : scheme.primary;

  void _text(Canvas c, Offset at, String s, TextStyle style, {double? max}) =>
      canvasText(s, style, maxWidth: max ?? double.infinity).paint(c, at);

  @override
  void paint(Canvas canvas, Size size) {
    final g = geometry;
    final n = deep.layers.length;
    // Connections: parent layer i + 1 predicts layer i.
    final line = Paint()
      ..color = scheme.outline.withValues(alpha: 0.35)
      ..strokeWidth = 0.8;
    for (var i = 0; i + 1 < n; i++) {
      final child = g.dots(deep, i), parent = g.dots(deep, i + 1);
      final down = Offset(0, -_dot - 34), up = Offset(0, _dot + 2);
      if (!deep.layers[i + 1].fullyConnected) {
        for (var k = 0; k < math.min(child.length, parent.length); k++) {
          canvas.drawLine(parent[k] + up, child[k] + down, line);
        }
      } else {
        final fan = Paint()
          ..color = scheme.outline.withValues(
            alpha: (1.2 / math.sqrt(child.length * parent.length)).clamp(
              0.14,
              0.45,
            ),
          )
          ..strokeWidth = 0.8;
        for (final p in parent) {
          for (final c in child) {
            canvas.drawLine(p + up, c + down, fan);
          }
        }
      }
    }
    // Layers.
    for (var i = 0; i < n; i++) {
      final bar = g.bars[i];
      final accent = _accent(i);
      final isSel = selected == i;
      final rrect = RRect.fromRectAndRadius(bar, const Radius.circular(10));
      canvas.drawRRect(
        rrect,
        Paint()
          ..color = Color.alphaBlend(
            accent.withValues(alpha: isSel ? 0.16 : 0.07),
            scheme.surface,
          ),
      );
      canvas.drawRRect(
        rrect,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = isSel ? 2.2 : 1.2
          ..color = isSel ? accent : accent.withValues(alpha: 0.6),
      );
      _text(
        canvas,
        bar.topLeft + const Offset(12, 6),
        deep.layerName(i),
        text.labelLarge!.copyWith(
          color: scheme.onSurface,
          fontWeight: FontWeight.w600,
        ),
        max: bar.width * 0.45,
      );
      final summary = layerSummary(deep, i);
      final sp = canvasText(
        summary,
        text.labelSmall!.copyWith(color: scheme.onSurfaceVariant),
        maxWidth: bar.width * 0.55 - 12,
      );
      sp.paint(canvas, Offset(bar.right - 12 - sp.width, bar.top + 8));
      final kind = deep.layers[i].kind;
      final fill = Paint()..color = accent;
      for (final c in g.dots(deep, i)) {
        if (kind == LayerKind.binary || kind == LayerKind.categorical) {
          canvas.drawRect(
            Rect.fromCenter(center: c, width: _dot * 1.7, height: _dot * 1.7),
            fill,
          );
        } else {
          canvas.drawCircle(c, _dot, fill);
        }
      }
      if (kind == LayerKind.categorical) {
        // One joint choice: the nodes are bracketed together.
        final ds = g.dots(deep, i);
        if (ds.isNotEmpty) {
          canvas.drawRRect(
            RRect.fromRectAndRadius(
              Rect.fromLTRB(
                ds.first.dx - _dot * 1.6,
                ds.first.dy - _dot * 1.6,
                ds.last.dx + _dot * 1.6,
                ds.last.dy + _dot * 1.6,
              ),
              const Radius.circular(4),
            ),
            Paint()
              ..style = PaintingStyle.stroke
              ..color = accent.withValues(alpha: 0.6),
          );
        }
      }
      final extra = deep.layers[i].size - _maxDots;
      if (extra > 0) {
        _text(
          canvas,
          Offset(g.dots(deep, i).last.dx + 12, bar.bottom - 23),
          '+$extra',
          text.labelSmall!.copyWith(color: scheme.onSurfaceVariant),
        );
      }
    }
    // Insert buttons.
    for (var i = 0; i < g.plus.length; i++) {
      final c = g.plus[i];
      final hot = hoverPlus == i;
      canvas.drawCircle(
        c,
        _plus,
        Paint()..color = hot ? scheme.primaryContainer : scheme.surface,
      );
      canvas.drawCircle(
        c,
        _plus,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.2
          ..color = hot ? scheme.primary : scheme.outline,
      );
      final cross = Paint()
        ..color = hot ? scheme.primary : scheme.onSurfaceVariant
        ..strokeWidth = 1.6
        ..strokeCap = StrokeCap.round;
      canvas
        ..drawLine(c - const Offset(5, 0), c + const Offset(5, 0), cross)
        ..drawLine(c - const Offset(0, 5), c + const Offset(0, 5), cross);
    }
  }

  @override
  bool shouldRepaint(_DeepPainter old) =>
      old.revision != revision ||
      old.deep != deep ||
      old.selected != selected ||
      old.hoverPlus != hoverPlus ||
      old.scheme != scheme;
}
