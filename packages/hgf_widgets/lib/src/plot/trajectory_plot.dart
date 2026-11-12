import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:material_ui/material_ui.dart';

import 'layers.dart';
import 'reduce.dart';

/// Colours and sizes of a [TrajectoryPlot]; [PlotStyle.of] derives them
/// from the theme.
@immutable
class PlotStyle {
  const PlotStyle({
    required this.foreground,
    required this.grid,
    required this.accent,
    required this.background,
    this.fontSize = 10,
    this.fontFamily,
  });

  factory PlotStyle.of(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return PlotStyle(
      foreground: scheme.onSurfaceVariant,
      grid: scheme.outlineVariant.withValues(alpha: 0.5),
      accent: scheme.primary,
      background: scheme.surfaceContainerHighest,
      fontFamily: DefaultTextStyle.of(context).style.fontFamily,
    );
  }

  /// Axis labels and titles.
  final Color foreground;

  /// Grid lines and the frame.
  final Color grid;

  /// The default layer colour.
  final Color accent;

  /// The hover readout's background.
  final Color background;

  final double fontSize;
  final String? fontFamily;

  @override
  bool operator ==(Object other) =>
      other is PlotStyle &&
      other.foreground == foreground &&
      other.grid == grid &&
      other.accent == accent &&
      other.background == background &&
      other.fontSize == fontSize &&
      other.fontFamily == fontFamily;

  @override
  int get hashCode =>
      Object.hash(foreground, grid, accent, background, fontSize, fontFamily);
}

/// Where the plot area sits in the widget.
@immutable
class PlotGeometry {
  const PlotGeometry(this.size, {this.xAxis = true});

  static const double leftAxis = 46;
  static const double rightPad = 8;
  static const double topPad = 4;
  static const double bottomAxis = 18;

  final Size size;
  final bool xAxis;

  Rect get plot => Rect.fromLTRB(
    leftAxis,
    topPad,
    math.max(leftAxis + 1, size.width - rightPad),
    math.max(topPad + 1, size.height - (xAxis ? bottomAxis : topPad)),
  );
}

/// A time-series plot of one or more [PlotLayer]s over trials, built to
/// re-draw thousands of samples every frame: each layer is reduced to at
/// most two points per pixel column before drawing (see [SeriesReducer]).
///
/// Several plots can share one [hover] notifier to get a linked cursor; it
/// holds the hovered trial and only repaints the cursor layer.
class TrajectoryPlot extends StatelessWidget {
  const TrajectoryPlot({
    super.key,
    required this.layers,
    this.title,
    this.xRange,
    this.yRange,
    this.limit,
    this.revision = 0,
    this.hover,
    this.showXAxis = true,
    this.style,
  });

  /// Drawn in order (the first at the back).
  final List<PlotLayer> layers;

  /// Shown at the top left of the plot area.
  final String? title;

  /// The visible trials, as `(first, last)` x coordinates (default: all).
  final (double, double)? xRange;

  /// The y range (default: fit the layers that [PlotLayer.affectsRange]).
  final (double, double)? yRange;

  /// Draw only trials before [limit] (trial-by-trial playback); a cursor
  /// marks the last one.
  final int? limit;

  /// Bump when the layers' data changed in place.
  final int revision;

  /// The hovered trial, shared between plots.
  final ValueNotifier<int?>? hover;

  final bool showXAxis;
  final PlotStyle? style;

  int get _length {
    var n = 0;
    for (final l in layers) {
      n = math.max(n, l.length);
    }
    return n;
  }

  @override
  Widget build(BuildContext context) {
    final s = style ?? PlotStyle.of(context);
    final n = _length;
    final x = xRange ?? (0.0, math.max(1.0, n - 1.0));
    final painter = _DataPainter(
      layers: layers,
      title: title,
      xRange: x,
      yRange: yRange,
      limit: limit,
      revision: revision,
      showXAxis: showXAxis,
      style: s,
    );
    Widget plot = CustomPaint(
      painter: painter,
      foregroundPainter: hover == null
          ? null
          : _HoverPainter(
              hover: hover!,
              data: painter,
              layers: layers,
              style: s,
              showXAxis: showXAxis,
            ),
      size: Size.infinite,
    );
    final h = hover;
    if (h != null) {
      int? trialAt(Offset p, Size size) {
        final r = PlotGeometry(size, xAxis: showXAxis).plot;
        if (n == 0 || p.dx < r.left || p.dx > r.right) return null;
        final t = x.$1 + (p.dx - r.left) / r.width * (x.$2 - x.$1);
        return t.round().clamp(0, n - 1);
      }

      final painted = plot;
      plot = LayoutBuilder(
        builder: (context, box) {
          final size = box.biggest;
          return MouseRegion(
            onHover: (e) => h.value = trialAt(e.localPosition, size),
            onExit: (_) => h.value = null,
            // Horizontal drags only, so a plot in a vertical list still
            // scrolls on touch screens.
            child: GestureDetector(
              onTapDown: (d) => h.value = trialAt(d.localPosition, size),
              onHorizontalDragStart: (d) =>
                  h.value = trialAt(d.localPosition, size),
              onHorizontalDragUpdate: (d) =>
                  h.value = trialAt(d.localPosition, size),
              child: painted,
            ),
          );
        },
      );
    }
    return RepaintBoundary(child: plot);
  }
}

/// Text laid out once and cached across frames.
final Map<String, TextPainter> _labels = {};

/// Repaints plots when fonts change: on the web, glyphs such as subscripts
/// come from fallback fonts that load after the first frame, and text laid
/// out before that shows missing glyphs until it is laid out again.
final Listenable _fontsChanged = () {
  final fonts = PaintingBinding.instance.systemFonts;
  fonts.addListener(_labels.clear); // registered first: runs before repaints
  return fonts;
}();

TextPainter _text(
  String s,
  PlotStyle style, {
  Color? color,
  bool bold = false,
}) {
  final c = color ?? style.foreground;
  final key = '$s|${c.toARGB32()}|${style.fontSize}|${style.fontFamily}|$bold';
  return _labels.putIfAbsent(key, () {
    if (_labels.length > 2000) _labels.clear();
    return TextPainter(
      text: TextSpan(
        text: s,
        style: TextStyle(
          color: c,
          fontSize: style.fontSize,
          fontFamily: style.fontFamily,
          fontWeight: bold ? FontWeight.w600 : null,
        ),
      ),
      textDirection: TextDirection.ltr,
      maxLines: 1,
    )..layout();
  });
}

/// A tick label with as many digits as the tick [step] needs, so that
/// neighbouring ticks never print the same.
String formatTick(double v, double step) {
  if (v == 0) return '0';
  final a = v.abs();
  if (a >= 1e5 || a < 1e-3) {
    // Significant digits down to the step's magnitude.
    final digits =
        (math.log(a) / math.ln10).floor() -
        (math.log(step) / math.ln10).floor() +
        1;
    return v.toStringAsExponential((digits - 1).clamp(0, 6));
  }
  final decimals = step >= 1 ? 0 : (-math.log(step) / math.ln10).ceil();
  return v.toStringAsFixed(decimals.clamp(0, 6));
}

class _DataPainter extends CustomPainter {
  _DataPainter({
    required this.layers,
    required this.title,
    required this.xRange,
    required this.yRange,
    required this.limit,
    required this.revision,
    required this.showXAxis,
    required this.style,
  }) : super(repaint: _fontsChanged);

  final List<PlotLayer> layers;
  final String? title;
  final (double, double) xRange;
  final (double, double)? yRange;
  final int? limit;
  final int revision;
  final bool showXAxis;
  final PlotStyle style;

  /// The y range used by the last paint (for the hover painter).
  (double, double) lastY = (0, 1);

  static final SeriesReducer _reducer = SeriesReducer();
  static Float32List _px = Float32List(512);

  static Float32List _pixels(int n) {
    if (_px.length < n) _px = Float32List(math.max(n, _px.length * 2));
    return _px;
  }

  (int, int) _visible(int length) {
    final first = math.max(0, xRange.$1.floor());
    var last = math.min(length - 1, xRange.$2.ceil());
    if (limit != null) last = math.min(last, limit! - 1);
    return (first, last);
  }

  (double, double) _autoY() {
    var lo = double.infinity, hi = double.negativeInfinity;
    for (final l in layers) {
      if (!l.affectsRange) continue;
      final (first, last) = _visible(l.length);
      if (last < first) continue;
      final r = l.rangeOver(first, last);
      if (r == null) continue;
      lo = math.min(lo, r.$1);
      hi = math.max(hi, r.$2);
    }
    if (!(lo <= hi)) return (0, 1);
    if (hi - lo < 1e-12 * math.max(1, hi.abs())) {
      final pad = math.max(0.5, hi.abs() * 0.1);
      return (lo - pad, hi + pad);
    }
    final pad = (hi - lo) * 0.06;
    return (lo - pad, hi + pad);
  }

  @override
  void paint(Canvas canvas, Size size) {
    final r = PlotGeometry(size, xAxis: showXAxis).plot;
    final (y0, y1) = lastY = yRange ?? _autoY();
    final (x0, x1) = xRange;
    final sx = r.width / (x1 - x0), sy = r.height / (y1 - y0);
    double px(double t) => r.left + (t - x0) * sx;
    // Keep pixel coordinates within float32 range for wild values.
    double py(double v) =>
        (r.bottom - (v - y0) * sy).clamp(r.top - 1e4, r.bottom + 1e4);

    _axes(canvas, r, x0, x1, y0, y1, px, py);

    canvas.save();
    canvas.clipRect(r.inflate(1));
    final columns = r.width.ceil();
    for (final layer in layers) {
      final color = layer.color ?? style.accent;
      switch (layer) {
        case LineLayer():
          final (first, last) = _visible(layer.length);
          _reducer.line(layer.values, first, last, columns);
          _drawRuns(canvas, px, py, ui.PointMode.polygon, layer.width, color);
        case PointLayer():
          final (first, last) = _visible(layer.length);
          _reducer.line(layer.values, first, last, columns);
          _drawRuns(
            canvas,
            px,
            py,
            ui.PointMode.points,
            layer.radius * 2,
            color,
          );
        case BandLayer():
          final (first, last) = _visible(layer.length);
          _reducer.envelope(first, last, columns, layer.lower, layer.upper);
          _drawBand(canvas, px, py, color);
        case ReferenceLine():
          final y = py(layer.value);
          canvas.drawLine(
            Offset(r.left, y),
            Offset(r.right, y),
            Paint()
              ..color = color
              ..strokeWidth = 1,
          );
      }
    }
    final l = limit;
    if (l != null && l > 0) {
      final x = px(l - 1.0);
      canvas.drawLine(
        Offset(x, r.top),
        Offset(x, r.bottom),
        Paint()
          ..color = style.accent.withValues(alpha: 0.6)
          ..strokeWidth = 1,
      );
    }
    canvas.restore();

    final t = title;
    if (t != null) {
      // On a backdrop, so the title stays legible over the data.
      final tp = _text(t, style, bold: true);
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTWH(r.left + 1, r.top + 1, tp.width + 8, tp.height + 2),
          const Radius.circular(3),
        ),
        Paint()..color = style.background.withValues(alpha: 0.85),
      );
      tp.paint(canvas, Offset(r.left + 5, r.top + 2));
    }
  }

  void _axes(
    Canvas canvas,
    Rect r,
    double x0,
    double x1,
    double y0,
    double y1,
    double Function(double) px,
    double Function(double) py,
  ) {
    final grid = Paint()
      ..color = style.grid
      ..strokeWidth = 1;
    canvas.drawRect(r, grid..style = PaintingStyle.stroke);
    grid.style = PaintingStyle.fill;
    final yTicks = niceTicks(y0, y1, target: math.max(2, r.height ~/ 28));
    final yStep = yTicks.length > 1 ? yTicks[1] - yTicks[0] : 1.0;
    for (final v in yTicks) {
      final y = py(v);
      canvas.drawLine(Offset(r.left, y), Offset(r.right, y), grid);
      final tp = _text(formatTick(v, yStep), style);
      tp.paint(canvas, Offset(r.left - tp.width - 4, y - tp.height / 2));
    }
    if (!showXAxis) return;
    final xTicks = niceTicks(x0, x1, target: math.max(2, r.width ~/ 90));
    for (final v in xTicks) {
      final x = px(v);
      canvas.drawLine(Offset(x, r.top), Offset(x, r.bottom), grid);
      final tp = _text(formatTick(v, 1), style);
      tp.paint(canvas, Offset(x - tp.width / 2, r.bottom + 3));
    }
  }

  void _drawRuns(
    Canvas canvas,
    double Function(double) px,
    double Function(double) py,
    ui.PointMode mode,
    double width,
    Color color,
  ) {
    final red = _reducer;
    if (red.length == 0) return;
    final buf = _pixels(red.length * 2);
    for (var i = 0; i < red.length; i++) {
      buf[2 * i] = px(red.x[i]);
      buf[2 * i + 1] = py(red.y[i]);
    }
    final paint = Paint()
      ..color = color
      ..strokeWidth = width
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..style = PaintingStyle.stroke
      ..isAntiAlias = true;
    var start = 0;
    for (final end in red.runEnds) {
      final m = end - start;
      final pts = Float32List.sublistView(buf, 2 * start, 2 * end);
      if (m == 1 || mode == ui.PointMode.points) {
        canvas.drawRawPoints(
          ui.PointMode.points,
          pts,
          mode == ui.PointMode.points
              ? paint
              : (paint..strokeWidth = width * 2),
        );
      } else {
        canvas.drawRawPoints(mode, pts, paint);
      }
      start = end;
    }
  }

  void _drawBand(
    Canvas canvas,
    double Function(double) px,
    double Function(double) py,
    Color color,
  ) {
    final red = _reducer;
    if (red.length == 0) return;
    final buf = _pixels(red.length * 4);
    final paint = Paint()..color = color.withValues(alpha: color.a * 0.22);
    var start = 0;
    for (final end in red.runEnds) {
      final m = end - start;
      if (m == 1) {
        // A single column: a thin bar.
        final x = px(red.x[start]);
        canvas.drawRect(
          Rect.fromLTRB(x - 1, py(red.y2[start]), x + 1, py(red.y[start])),
          paint,
        );
      } else {
        var k = 0;
        for (var i = start; i < end; i++) {
          final x = px(red.x[i]);
          buf[k++] = x;
          buf[k++] = py(red.y2[i]);
          buf[k++] = x;
          buf[k++] = py(red.y[i]);
        }
        canvas.drawVertices(
          ui.Vertices.raw(
            ui.VertexMode.triangleStrip,
            Float32List.fromList(Float32List.sublistView(buf, 0, k)),
          ),
          BlendMode.srcOver,
          paint,
        );
      }
      start = end;
    }
  }

  @override
  bool shouldRepaint(_DataPainter old) =>
      revision != old.revision ||
      limit != old.limit ||
      xRange != old.xRange ||
      yRange != old.yRange ||
      title != old.title ||
      style != old.style ||
      showXAxis != old.showXAxis ||
      !listEquals(layers, old.layers);
}

class _HoverPainter extends CustomPainter {
  _HoverPainter({
    required this.hover,
    required this.data,
    required this.layers,
    required this.style,
    required this.showXAxis,
  }) : super(repaint: hover);

  final ValueNotifier<int?> hover;
  final _DataPainter data;
  final List<PlotLayer> layers;
  final PlotStyle style;
  final bool showXAxis;

  @override
  void paint(Canvas canvas, Size size) {
    final t = hover.value;
    if (t == null) return;
    final r = PlotGeometry(size, xAxis: showXAxis).plot;
    final (x0, x1) = data.xRange;
    final x = r.left + (t - x0) / (x1 - x0) * r.width;
    if (x < r.left || x > r.right) return;
    canvas.drawLine(
      Offset(x, r.top),
      Offset(x, r.bottom),
      Paint()
        ..color = style.foreground.withValues(alpha: 0.5)
        ..strokeWidth = 1,
    );
    // Text in the text colour; a swatch beside each line carries the
    // series' identity.
    final lines = <(TextPainter, Color?)>[
      (_text('t = $t', style, bold: true), null),
    ];
    for (final l in layers) {
      final label = l.label;
      if (label == null || l is ReferenceLine || t >= l.length) continue;
      final v = l.valueAt(t);
      if (!v.isFinite && !l.readoutGaps) continue;
      lines.add((
        _text('$label ${v.isFinite ? _fmt(v) : '—'}', style),
        l.color ?? style.accent,
      ));
    }
    const swatch = 10.0;
    var w = 0.0, h = 0.0;
    for (final (tp, _) in lines) {
      w = math.max(w, tp.width + swatch + 4);
      h += tp.height;
    }
    final right = x + 8 + w + 8 > r.right;
    var left = right ? x - 8 - w - 8 : x + 8;
    left = math.max(r.left, left);
    final box = Rect.fromLTWH(left, r.top + 2, w + 8, h + 6);
    canvas.drawRRect(
      RRect.fromRectAndRadius(box, const Radius.circular(4)),
      Paint()..color = style.background.withValues(alpha: 0.92),
    );
    var y = box.top + 3;
    for (final (tp, color) in lines) {
      if (color != null) {
        canvas.drawRRect(
          RRect.fromRectAndRadius(
            Rect.fromLTWH(
              box.left + 4,
              y + (tp.height - swatch) / 2,
              swatch,
              swatch,
            ),
            const Radius.circular(2),
          ),
          Paint()..color = color,
        );
      }
      tp.paint(canvas, Offset(box.left + 4 + swatch + 4, y));
      y += tp.height;
    }
  }

  static String _fmt(double v) {
    final a = v.abs();
    if (a != 0 && (a >= 1e4 || a < 1e-3)) return v.toStringAsExponential(2);
    return v.toStringAsFixed(3);
  }

  @override
  bool shouldRepaint(_HoverPainter old) =>
      hover != old.hover ||
      data != old.data ||
      style != old.style ||
      showXAxis != old.showXAxis;
}
