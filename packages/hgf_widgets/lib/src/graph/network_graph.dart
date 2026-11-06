import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:hgf/hgf.dart';

import 'layout.dart';

/// The default node label: `x₀`, `x₁`, ... as in the HGF literature (by
/// node index, as pyhgf numbers them).
String defaultNodeLabel(int node) {
  const sub = '₀₁₂₃₄₅₆₇₈₉';
  return 'x${node.toString().split('').map((d) => sub[int.parse(d)]).join()}';
}

/// A drawing of a [Network]'s graph: inputs at the bottom, value coupling
/// as solid arrows, volatility coupling as dashed ones (as in pyhgf's
/// `plot_network`). Tap a node to select it.
///
/// Shapes by kind: continuous ○, binary □, constant (bias) small ○,
/// ef-state ◇, categorical and dp-state ⬡. Input nodes are filled.
class NetworkGraph extends StatelessWidget {
  const NetworkGraph({
    super.key,
    required this.network,
    this.layout,
    this.selected,
    this.onNodeTap,
    this.label = defaultNodeLabel,
    this.caption,
    this.nodeColor,
    this.revision = 0,
  });

  final Network network;

  /// A precomputed layout (default: [NetworkLayout.of] the network).
  final NetworkLayout? layout;

  final int? selected;
  final ValueChanged<int>? onNodeTap;

  /// The text inside each node.
  final String Function(int node) label;

  /// Optional text under each node (e.g. its current belief).
  final String? Function(int node)? caption;

  /// Per-node colour (default: the theme's primary).
  final Color? Function(int node)? nodeColor;

  /// Bump when [caption]s changed without the network changing.
  final int revision;

  static const double nodeRadius = 18;

  @override
  Widget build(BuildContext context) {
    final l = layout ?? NetworkLayout.of(network);
    final theme = Theme.of(context);
    return LayoutBuilder(
      builder: (context, box) {
        final size = box.biggest;
        final centres = _centres(l, size);
        return GestureDetector(
          behavior: HitTestBehavior.opaque,
          onTapUp: onNodeTap == null
              ? null
              : (d) {
                  var best = -1;
                  var bestD = (nodeRadius + 8) * (nodeRadius + 8);
                  for (var i = 0; i < centres.length; i++) {
                    final dd = (centres[i] - d.localPosition).distanceSquared;
                    if (dd < bestD) {
                      best = i;
                      bestD = dd;
                    }
                  }
                  if (best >= 0) onNodeTap!(best);
                },
          child: CustomPaint(
            size: size,
            painter: _GraphPainter(
              network: network,
              layout: l,
              centres: centres,
              selected: selected,
              label: label,
              caption: caption,
              nodeColor: nodeColor,
              revision: revision,
              scheme: theme.colorScheme,
              textStyle: theme.textTheme.labelMedium ?? const TextStyle(),
            ),
          ),
        );
      },
    );
  }

  static List<Offset> _centres(NetworkLayout l, Size size) {
    const pad = nodeRadius + 14;
    final w = math.max(1.0, size.width - 2 * pad);
    final h = math.max(1.0, size.height - 2 * pad - 12);
    return [for (final p in l.position) Offset(pad + p.dx * w, pad + p.dy * h)];
  }
}

class _GraphPainter extends CustomPainter {
  _GraphPainter({
    required this.network,
    required this.layout,
    required this.centres,
    required this.selected,
    required this.label,
    required this.caption,
    required this.nodeColor,
    required this.revision,
    required this.scheme,
    required this.textStyle,
  });

  final Network network;
  final NetworkLayout layout;
  final List<Offset> centres;
  final int? selected;
  final String Function(int) label;
  final String? Function(int)? caption;
  final Color? Function(int)? nodeColor;
  final int revision;
  final ColorScheme scheme;
  final TextStyle textStyle;

  static const double r = NetworkGraph.nodeRadius;

  double _radius(int i) =>
      network.kindOf(i) == NodeKind.constantState ? r * 0.6 : r;

  @override
  void paint(Canvas canvas, Size size) {
    final edgePaint = Paint()
      ..color = scheme.onSurfaceVariant
      ..strokeWidth = 1.4
      ..style = PaintingStyle.stroke;
    for (final e in layout.edges) {
      final a = centres[e.parent], b = centres[e.child];
      final d = b - a;
      final len = d.distance;
      if (len < 1e-6) continue;
      final u = d / len;
      final from = a + u * _radius(e.parent);
      final to = b - u * (_radius(e.child) + 2);
      if (e.kind == EdgeKind.volatility) {
        _dashed(canvas, from, to, edgePaint);
      } else {
        canvas.drawLine(from, to, edgePaint);
      }
      _arrowHead(canvas, to, u, edgePaint.color);
      if (e.strength != 1.0) {
        final mid = (from + to) / 2 + Offset(-u.dy, u.dx) * 9;
        _paintText(
          canvas,
          _fmtStrength(e.strength),
          mid,
          textStyle.copyWith(
            color: scheme.onSurfaceVariant,
            fontSize: (textStyle.fontSize ?? 12) * 0.85,
          ),
        );
      }
    }

    for (var i = 0; i < centres.length; i++) {
      final c = centres[i];
      final kind = network.kindOf(i);
      final color = nodeColor?.call(i) ?? scheme.primary;
      final input = layout.level[i] == 0;
      final fill = Paint()
        ..color = input
            ? Color.alphaBlend(color.withValues(alpha: 0.35), scheme.surface)
            : scheme.surface;
      final stroke = Paint()
        ..color = color
        ..strokeWidth = i == selected ? 3.5 : 1.8
        ..style = PaintingStyle.stroke;
      final path = _shape(kind, c, _radius(i));
      if (i == selected) {
        canvas.drawPath(
          _shape(kind, c, _radius(i) + 5),
          Paint()..color = color.withValues(alpha: 0.18),
        );
      }
      canvas.drawPath(path, fill);
      canvas.drawPath(path, stroke);
      if (kind != NodeKind.constantState) {
        _paintText(
          canvas,
          label(i),
          c,
          textStyle.copyWith(color: scheme.onSurface),
        );
      } else {
        _paintText(
          canvas,
          '1',
          c,
          textStyle.copyWith(
            color: scheme.onSurface,
            fontSize: (textStyle.fontSize ?? 12) * 0.85,
          ),
        );
      }
      final cap = caption?.call(i);
      if (cap != null) {
        _paintText(
          canvas,
          cap,
          // Beside the node, clear of the vertical edges.
          c + Offset(_radius(i) + 6, 0),
          textStyle.copyWith(
            color: scheme.onSurfaceVariant,
            fontSize: (textStyle.fontSize ?? 12) * 0.85,
            fontFeatures: const [FontFeature.tabularFigures()],
          ),
          leftAligned: true,
        );
      }
    }
  }

  static String _fmtStrength(double k) =>
      k == k.roundToDouble() ? k.toStringAsFixed(0) : k.toStringAsFixed(2);

  Path _shape(NodeKind kind, Offset c, double rad) {
    switch (kind) {
      case NodeKind.binaryState:
        return Path()..addRRect(
          RRect.fromRectAndRadius(
            Rect.fromCircle(center: c, radius: rad * 0.9),
            Radius.circular(rad * 0.2),
          ),
        );
      case NodeKind.efState:
        return Path()
          ..moveTo(c.dx, c.dy - rad * 1.15)
          ..lineTo(c.dx + rad * 1.15, c.dy)
          ..lineTo(c.dx, c.dy + rad * 1.15)
          ..lineTo(c.dx - rad * 1.15, c.dy)
          ..close();
      case NodeKind.categoricalState:
      case NodeKind.dpState:
        final p = Path();
        for (var k = 0; k < 6; k++) {
          final a = math.pi / 6 + k * math.pi / 3;
          final v = c + Offset(math.cos(a), math.sin(a)) * rad * 1.08;
          k == 0 ? p.moveTo(v.dx, v.dy) : p.lineTo(v.dx, v.dy);
        }
        return p..close();
      case NodeKind.continuousState:
      case NodeKind.constantState:
        return Path()..addOval(Rect.fromCircle(center: c, radius: rad));
    }
  }

  static void _dashed(Canvas canvas, Offset a, Offset b, Paint paint) {
    const dash = 5.0, gap = 4.0;
    final d = b - a;
    final len = d.distance;
    final u = d / len;
    for (var s = 0.0; s < len; s += dash + gap) {
      canvas.drawLine(a + u * s, a + u * math.min(len, s + dash), paint);
    }
  }

  static void _arrowHead(Canvas canvas, Offset tip, Offset u, Color color) {
    const size = 7.0;
    final n = Offset(-u.dy, u.dx);
    final base = tip - u * size;
    canvas.drawPath(
      Path()
        ..moveTo(tip.dx, tip.dy)
        ..lineTo(base.dx + n.dx * size * 0.5, base.dy + n.dy * size * 0.5)
        ..lineTo(base.dx - n.dx * size * 0.5, base.dy - n.dy * size * 0.5)
        ..close(),
      Paint()..color = color,
    );
  }

  /// Text centred on [c], or starting at [c] when [leftAligned].
  static void _paintText(
    Canvas canvas,
    String s,
    Offset c,
    TextStyle style, {
    bool leftAligned = false,
  }) {
    final tp = TextPainter(
      text: TextSpan(text: s, style: style),
      textDirection: TextDirection.ltr,
      maxLines: 1,
    )..layout();
    tp.paint(canvas, c - Offset(leftAligned ? 0 : tp.width / 2, tp.height / 2));
  }

  @override
  bool shouldRepaint(_GraphPainter old) =>
      !identical(network, old.network) ||
      !identical(layout, old.layout) ||
      selected != old.selected ||
      revision != old.revision ||
      scheme != old.scheme ||
      textStyle != old.textStyle ||
      centres.length != old.centres.length ||
      (centres.isNotEmpty && centres.last != old.centres.last);
}
