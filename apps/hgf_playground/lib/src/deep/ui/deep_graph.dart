import 'dart:math' as math;

import 'package:material_ui/material_ui.dart';
import 'package:hgf_deep/hgf_deep.dart';
import 'package:hgf_widgets/hgf_widgets.dart';

import '../deep_view.dart';

/// A node of the network view: network layer and index.
typedef NodeRef = ({int layer, int node});

/// Colours of the network view.
@immutable
class DeepGraphColors {
  const DeepGraphColors({
    required this.positive,
    required this.negative,
    required this.neutral,
    required this.precise,
    required this.ink,
    required this.faint,
    required this.highlight,
    required this.surface,
  });

  factory DeepGraphColors.of(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final c = HgfColors.of(context);
    return DeepGraphColors(
      positive: c.belief,
      negative: c.observation,
      neutral: scheme.surfaceContainerHighest,
      precise: c.surprise,
      ink: scheme.onSurface,
      faint: scheme.outlineVariant,
      highlight: scheme.primary,
      surface: scheme.surface,
    );
  }

  final Color positive, negative, neutral, precise, ink, faint, highlight;
  final Color surface;

  /// A diverging colour: [negative] below 0, [positive] above, [neutral]
  /// at 0; saturates smoothly at ±[scale].
  Color diverging(double v, double scale) {
    if (!v.isFinite) return ink;
    final t = _tanh(v / scale);
    return Color.lerp(neutral, t >= 0 ? positive : negative, t.abs())!;
  }

  /// A precision on a log scale from 10⁻³ (neutral) to 10³ ([precise]).
  Color precision(double v) {
    if (!v.isFinite || v <= 0) return ink;
    final t = ((math.log(v) / math.ln10 + 3) / 6).clamp(0.0, 1.0);
    return Color.lerp(neutral, precise, t)!;
  }

  static double _tanh(double x) {
    final e = math.exp(-2 * x.abs());
    final t = (1 - e) / (1 + e);
    return x < 0 ? -t : t;
  }
}

/// The scale at which a diverging field saturates.
double fieldScale(NodeField f) => switch (f) {
  NodeField.valuePredictionError => 1.0,
  _ => 1.5,
};

/// The network, layer by layer from the input (left) to the output (right):
/// node colours show [field] after the traced phase, edges the weights (at
/// rest) or the messages of the phase, and a diamond above a node its
/// volatility level μ_vol. Tap a node to select it.
class DeepGraphView extends StatelessWidget {
  const DeepGraphView({
    super.key,
    required this.model,
    required this.field,
    this.selected,
    this.onSelect,
    this.revision = 0,
  });

  final DeepViewModel model;
  final NodeField field;
  final NodeRef? selected;
  final ValueChanged<NodeRef?>? onSelect;

  /// Bump to repaint when the model's contents change in place.
  final int revision;

  @override
  Widget build(BuildContext context) {
    final colors = DeepGraphColors.of(context);
    final text = Theme.of(context).textTheme;
    return LayoutBuilder(
      builder: (context, box) {
        final size = Size(box.maxWidth, box.maxHeight);
        final geometry = _Geometry(model, size);
        return GestureDetector(
          behavior: HitTestBehavior.opaque,
          onTapUp: onSelect == null
              ? null
              : (d) => onSelect!(geometry.hit(d.localPosition)),
          child: CustomPaint(
            size: size,
            painter: _GraphPainter(
              model: model,
              geometry: geometry,
              field: field,
              selected: selected,
              colors: colors,
              label: text.labelSmall!.copyWith(color: colors.ink),
              revision: revision,
            ),
          ),
        );
      },
    );
  }
}

/// Where everything goes.
class _Geometry {
  _Geometry(this.model, this.size) {
    final n = model.nLayers;
    const top = 34.0, bottom = 14.0;
    final left = 40.0, right = size.width - 40.0;
    xs = [
      for (var d = 0; d < n; d++)
        n == 1 ? size.width / 2 : left + (right - left) * d / (n - 1),
    ];
    final usable = math.max(10.0, size.height - top - bottom);
    final maxNodes = model.network.layerSizes.reduce(math.max);
    final spacing = usable / math.max(1, maxNodes);
    final colSpacing = n > 1 ? (right - left) / (n - 1) : size.width;
    radius = math.min(14.0, math.min(spacing * 0.36, colSpacing * 0.14));
    radius = math.max(radius, 2.5);
    positions = [
      for (var l = 0; l < n; l++)
        () {
          final count = model.network.layerSizes[l];
          final d = model.columnOf(l);
          final h = spacing * count;
          final y0 = top + (usable - h) / 2 + spacing / 2;
          return [
            for (var i = 0; i < count; i++) Offset(xs[d], y0 + i * spacing),
          ];
        }(),
    ];
    this.top = top;
  }

  final DeepViewModel model;
  final Size size;
  late final List<double> xs;
  late double radius;
  late final double top;

  /// positions[l][i]: node i of network layer l.
  late final List<List<Offset>> positions;

  /// Where a layer's bias sits: above its first node.
  Offset bias(int l) {
    final p = positions[l];
    final y = p.isEmpty ? top : p.first.dy - math.max(radius * 2.6, 12);
    return Offset(p.first.dx, math.max(top - 6, y));
  }

  Offset satellite(int l, int i) =>
      positions[l][i] + Offset(radius * 0.9, -radius * 0.9);

  NodeRef? hit(Offset p) {
    if (model.mode == ViewMode.summary) return null;
    NodeRef? best;
    var bestD = math.max(radius + 6, 10.0);
    for (var l = 0; l < positions.length; l++) {
      for (var i = 0; i < positions[l].length; i++) {
        final d = (positions[l][i] - p).distance;
        if (d < bestD) {
          bestD = d;
          best = (layer: l, node: i);
        }
      }
    }
    return best;
  }
}

class _GraphPainter extends CustomPainter {
  _GraphPainter({
    required this.model,
    required this.geometry,
    required this.field,
    required this.selected,
    required this.colors,
    required this.label,
    required this.revision,
  });

  final DeepViewModel model;
  final _Geometry geometry;
  final NodeField field;
  final NodeRef? selected;
  final DeepGraphColors colors;
  final TextStyle label;
  final int revision;

  DeepNetwork get net => model.network;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.clipRect(Offset.zero & size);
    _activeBackdrop(canvas, size);
    _headers(canvas);
    if (model.mode == ViewMode.summary) {
      _summary(canvas, size);
      return;
    }
    _edges(canvas);
    _nodes(canvas);
  }

  /// A tint behind the column the phase acts on.
  void _activeBackdrop(Canvas canvas, Size size) {
    final step = model.step;
    if (step == null) return;
    final cols = <int>{
      switch (step.phase) {
        TracePhase.clampInput => model.columnOf(net.nLayers - 1),
        TracePhase.observe || TracePhase.leafError => model.columnOf(0),
        TracePhase.predict ||
        TracePhase.posterior ||
        TracePhase.layerError => model.columnOf(step.layer),
        TracePhase.weightUpdate || TracePhase.carry => -1,
      },
    }..remove(-1);
    final w = net.nLayers > 1
        ? (geometry.xs.last - geometry.xs.first) / (net.nLayers - 1) * 0.42
        : 60.0;
    for (final d in cols) {
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTRB(
            geometry.xs[d] - w,
            4,
            geometry.xs[d] + w,
            size.height - 4,
          ),
          const Radius.circular(10),
        ),
        Paint()..color = colors.highlight.withValues(alpha: 0.08),
      );
    }
  }

  void _headers(Canvas canvas) {
    for (var l = 0; l < net.nLayers; l++) {
      final c = net.configs[l];
      final d = model.columnOf(l);
      final kind = switch (c.kind) {
        LayerKind.binary => 'binary',
        LayerKind.categorical => 'categorical',
        LayerKind.volatile =>
          l == 0 || l == net.nLayers - 1
              ? ''
              : (c.coupling is CouplingFn
                    ? (c.coupling as CouplingFn).name
                    : 'custom'),
      };
      final vol = l > 0 && l < net.nLayers - 1 && c.volatilityParent
          ? ' · vol'
          : '';
      _text(
        canvas,
        '${layerName(net, l)} (${c.size}${kind.isEmpty ? '' : ' $kind'}$vol)',
        Offset(geometry.xs[d], 10),
        label,
      );
    }
  }

  void _edges(Canvas canvas) {
    final step = model.step;
    final maxMsg = step == null ? 0.0 : model.maxMessage();
    final maxW = math.max(1e-9, model.maxWeight());
    final edges = <(double, Offset, Offset, Color, double)>[];
    for (var l = 1; l < net.nLayers; l++) {
      final w = model.weights[l]!;
      final parent = geometry.positions[l];
      final child = geometry.positions[l - 1];
      final cfg = net.configs[l];
      for (var i = 0; i < w.rows; i++) {
        for (var j = 0; j < w.cols; j++) {
          if (!cfg.fullyConnected && i != j) continue;
          final isBias = j == parent.length;
          final from = isBias ? geometry.bias(l) : parent[j];
          final msg = model.message(l, i, j);
          final double strength;
          final Color color;
          if (msg != null && maxMsg > 0) {
            // Phase messages; the weight step moves against the gradient.
            final v = step!.phase == TracePhase.weightUpdate ? -msg : msg;
            strength = (v.abs() / maxMsg).clamp(0.0, 1.0);
            color = (v >= 0 ? colors.positive : colors.negative).withValues(
              alpha: 0.15 + 0.85 * strength,
            );
          } else {
            final v = w.at(i, j);
            strength = (v.abs() / maxW).clamp(0.0, 1.0);
            final dim = step != null && _phaseHasMessages(step) ? 0.25 : 1.0;
            color = (v >= 0 ? colors.positive : colors.negative).withValues(
              alpha: (0.06 + 0.5 * strength) * dim,
            );
          }
          edges.add((strength, from, child[i], color, 0.4 + 2.6 * strength));
        }
      }
    }
    if (model.mode == ViewMode.topEdges &&
        edges.length > DeepViewModel.maxEdges) {
      edges.sort((a, b) => b.$1.compareTo(a.$1));
      edges.length = DeepViewModel.maxEdges;
    }
    // Weak first, so strong edges stay on top.
    edges.sort((a, b) => a.$1.compareTo(b.$1));
    final paint = Paint()..strokeCap = StrokeCap.round;
    for (final (_, a, b, color, width) in edges) {
      canvas.drawLine(
        a,
        b,
        paint
          ..color = color
          ..strokeWidth = width,
      );
    }
    // Direction of the phase's messages.
    if (step != null && _phaseHasMessages(step) && maxMsg > 0) {
      final up = step.phase == TracePhase.posterior;
      for (var l = 1; l < net.nLayers; l++) {
        final active = switch (step.phase) {
          TracePhase.predict => step.layer == l - 1,
          TracePhase.posterior => step.layer == l,
          _ => false,
        };
        if (!active) continue;
        final xa = geometry.xs[model.columnOf(l)];
        final xb = geometry.xs[model.columnOf(l - 1)];
        final y = geometry.top + 2;
        final from = Offset(up ? xb : xa, y), to = Offset(up ? xa : xb, y);
        final mid = Offset((xa + xb) / 2, y);
        final dir = (to - from) / (to - from).distance;
        canvas.drawLine(
          mid - dir * 14,
          mid + dir * 10,
          Paint()
            ..color = colors.highlight
            ..strokeWidth = 2,
        );
        _arrowHead(canvas, mid + dir * 14, dir, colors.highlight);
      }
    }
  }

  static bool _phaseHasMessages(TraceStep s) =>
      s.phase == TracePhase.predict ||
      s.phase == TracePhase.posterior ||
      s.phase == TracePhase.weightUpdate;

  void _nodes(Canvas canvas) {
    final r = geometry.radius;
    final step = model.step;
    for (var l = 0; l < net.nLayers; l++) {
      final hasConstant = l > 0 && net.configs[l].addConstantInput;
      if (hasConstant) {
        final b = geometry.bias(l);
        final rect = Rect.fromCenter(
          center: b,
          width: r * 1.3,
          height: r * 1.3,
        );
        canvas.drawRect(rect, Paint()..color = colors.surface);
        canvas.drawRect(
          rect,
          Paint()
            ..color = colors.faint
            ..style = PaintingStyle.stroke,
        );
        if (r >= 6) _text(canvas, '1', b, label.copyWith(fontSize: 9));
      }
      for (var i = 0; i < geometry.positions[l].length; i++) {
        final p = geometry.positions[l][i];
        final v = model.value(field, l, i);
        final fill = field.isPrecision
            ? colors.precision(v)
            : colors.diverging(v, fieldScale(field));
        // Ring thickness: posterior precision (log scale).
        final pi = model.states[l].precision[i];
        final ring = pi.isFinite && pi > 0
            ? (1.0 + (math.log(pi) / math.ln10 + 2).clamp(0.0, 4.0) * 0.6)
            : 1.0;
        canvas.drawCircle(p, r, Paint()..color = fill);
        canvas.drawCircle(
          p,
          r,
          Paint()
            ..color = colors.ink.withValues(alpha: 0.55)
            ..style = PaintingStyle.stroke
            ..strokeWidth = math.min(ring, r * 0.5),
        );
        if (_emphasised(step, l)) {
          canvas.drawCircle(
            p,
            r + 3,
            Paint()
              ..color = colors.highlight
              ..style = PaintingStyle.stroke
              ..strokeWidth = 1.5,
          );
        }
        if (selected?.layer == l && selected?.node == i) {
          canvas.drawCircle(
            p,
            r + 5,
            Paint()
              ..color = colors.ink
              ..style = PaintingStyle.stroke
              ..strokeWidth = 2,
          );
        }
        if (r >= 11) {
          _text(
            canvas,
            _short(v),
            p,
            label.copyWith(fontSize: 9, color: _onFill(fill)),
          );
        }
        _satellite(canvas, l, i, step);
      }
    }
  }

  /// The volatility level of node [i] of layer [l]: a diamond coloured by
  /// μ_vol, joined to its node by a dashed line.
  void _satellite(Canvas canvas, int l, int i, TraceStep? step) {
    final mv = model.meanVol(l, i);
    // The input layer's level never acts (the layer is clamped), nor does a
    // categorical one.
    if (mv == null ||
        l == net.nLayers - 1 ||
        net.configs[l].kind != LayerKind.volatile) {
      return;
    }
    final r = math.max(2.5, geometry.radius * 0.5);
    final c = geometry.satellite(l, i);
    _dashed(
      canvas,
      geometry.positions[l][i],
      c,
      Paint()
        ..color = colors.ink.withValues(alpha: 0.5)
        ..strokeWidth = 1,
    );
    final path = Path()
      ..moveTo(c.dx, c.dy - r)
      ..lineTo(c.dx + r, c.dy)
      ..lineTo(c.dx, c.dy + r)
      ..lineTo(c.dx - r, c.dy)
      ..close();
    canvas.drawPath(path, Paint()..color = colors.diverging(mv, 3));
    final active =
        step != null &&
        (step.phase == TracePhase.carry ||
            (step.phase == TracePhase.layerError && step.layer == l) ||
            (step.phase == TracePhase.leafError && l == 0));
    canvas.drawPath(
      path,
      Paint()
        ..color = active ? colors.highlight : colors.ink.withValues(alpha: 0.6)
        ..style = PaintingStyle.stroke
        ..strokeWidth = active ? 1.8 : 0.8,
    );
  }

  bool _emphasised(TraceStep? s, int l) {
    if (s == null) return false;
    return switch (s.phase) {
      TracePhase.clampInput => l == net.nLayers - 1,
      TracePhase.observe || TracePhase.leafError => l == 0,
      TracePhase.predict ||
      TracePhase.posterior ||
      TracePhase.layerError => l == s.layer,
      _ => false,
    };
  }

  /// Too many nodes to draw: each layer is a strip of its node values
  /// (averaged in bins), with the layer mean ± sd below.
  void _summary(Canvas canvas, Size size) {
    final top = geometry.top + 6, bottom = size.height - 30;
    final h = math.max(10.0, bottom - top);
    final colW = net.nLayers > 1
        ? math.min(
            36.0,
            (geometry.xs.last - geometry.xs.first) / (net.nLayers - 1) * 0.3,
          )
        : 36.0;
    for (var l = 0; l < net.nLayers; l++) {
      final x = geometry.xs[model.columnOf(l)];
      final n = net.layerSizes[l];
      final bins = math.min(n, math.max(1, h ~/ 3));
      var sum = 0.0, sq = 0.0, finite = 0;
      for (var b = 0; b < bins; b++) {
        final a = b * n ~/ bins, z = (b + 1) * n ~/ bins;
        var s = 0.0, k = 0;
        for (var i = a; i < z; i++) {
          final v = model.value(field, l, i);
          if (!v.isFinite) continue;
          s += v;
          k++;
        }
        final v = k == 0 ? double.nan : s / k;
        final fill = field.isPrecision
            ? colors.precision(v)
            : colors.diverging(v, fieldScale(field));
        canvas.drawRect(
          Rect.fromLTWH(x - colW / 2, top + h * b / bins, colW, h / bins + 0.5),
          Paint()..color = fill,
        );
      }
      for (var i = 0; i < n; i++) {
        final v = model.value(field, l, i);
        if (!v.isFinite) continue;
        sum += v;
        sq += v * v;
        finite++;
      }
      canvas.drawRect(
        Rect.fromLTWH(x - colW / 2, top, colW, h),
        Paint()
          ..color = _emphasised(model.step, l)
              ? colors.highlight
              : colors.ink.withValues(alpha: 0.4)
          ..style = PaintingStyle.stroke
          ..strokeWidth = _emphasised(model.step, l) ? 2 : 1,
      );
      final mean = finite == 0 ? double.nan : sum / finite;
      final sd = finite == 0
          ? double.nan
          : math.sqrt(math.max(0, sq / finite - mean * mean));
      _text(
        canvas,
        '${_short(mean)} ± ${_short(sd)}',
        Offset(x, bottom + 14),
        label,
      );
    }
  }

  static String _short(double v) {
    if (v.isNaN) return 'NaN';
    if (v.isInfinite) return v > 0 ? '∞' : '−∞';
    final a = v.abs();
    if (a >= 1000 || (a < 0.01 && a > 0)) return v.toStringAsExponential(0);
    return v.toStringAsFixed(a >= 10 ? 0 : (a >= 1 ? 1 : 2));
  }

  Color _onFill(Color fill) =>
      fill.computeLuminance() > 0.45 ? Colors.black87 : Colors.white;

  static void _dashed(Canvas canvas, Offset a, Offset b, Paint paint) {
    const dash = 3.0, gap = 2.0;
    final d = b - a;
    final len = d.distance;
    if (len == 0) return;
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

  static void _text(Canvas canvas, String s, Offset c, TextStyle style) {
    final tp = TextPainter(
      text: TextSpan(text: s, style: style),
      textDirection: TextDirection.ltr,
      maxLines: 1,
      ellipsis: '…',
    )..layout(maxWidth: 160);
    tp.paint(canvas, c - Offset(tp.width / 2, tp.height / 2));
  }

  @override
  bool shouldRepaint(_GraphPainter old) =>
      old.model != model ||
      old.field != field ||
      old.selected != selected ||
      old.colors != colors ||
      old.revision != revision ||
      old.geometry.size != geometry.size;
}
