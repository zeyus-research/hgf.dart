import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/services.dart';
import 'package:material_ui/material_ui.dart';

import 'layers.dart';
import 'reduce.dart';
import 'viewport.dart';

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
/// holds the hovered trial and only repaints the cursor layer. Likewise a
/// shared [viewport] zooms and pans them together:
///
/// - zoom about the pointer with Ctrl/⌘ + scroll, a trackpad pinch or a
///   two-finger touch pinch; plain scrolling zooms too unless the plot sits
///   in a scrollable (which keeps the scroll);
/// - pan by dragging with the mouse, scrolling sideways (or Shift +
///   scroll), or the pinch's two fingers;
/// - double-click (or double-tap) to show every trial again.
///
/// The y range fits the visible trials unless [yRange] is set.
class TrajectoryPlot extends StatefulWidget {
  const TrajectoryPlot({
    super.key,
    required this.layers,
    this.title,
    this.xRange,
    this.yRange,
    this.limit,
    this.revision = 0,
    this.hover,
    this.viewport,
    this.showXAxis = true,
    this.style,
  });

  /// Drawn in order (the first at the back).
  final List<PlotLayer> layers;

  /// Shown at the top left of the plot area.
  final String? title;

  /// The visible trials, as `(first, last)` x coordinates (default: all,
  /// or the [viewport]'s). Fixes the range: the plot doesn't zoom.
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

  /// The visible trials, shared between plots; the plot zooms and pans it.
  final PlotViewport? viewport;

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
  State<TrajectoryPlot> createState() => _TrajectoryPlotState();
}

class _TrajectoryPlotState extends State<TrajectoryPlot> {
  /// Touch pointers down on the plot, for a two-finger pinch.
  final Map<int, Offset> _touches = {};

  /// The pinch's start: the window and the two fingers' trials.
  (double, double)? _pinchTrials;

  /// Whether the current horizontal drag pans (a mouse) rather than
  /// scrubbing the hover cursor (touch).
  bool _panning = false;

  /// The last trackpad scale of a pan-zoom gesture.
  double _lastScale = 1;

  bool _zoomed = false;

  PlotViewport? get _viewport => widget.xRange == null ? widget.viewport : null;

  @override
  void initState() {
    super.initState();
    _listen(null, _viewport);
  }

  @override
  void didUpdateWidget(TrajectoryPlot old) {
    super.didUpdateWidget(old);
    final before = old.xRange == null ? old.viewport : null;
    if (!identical(before, _viewport)) _listen(before, _viewport);
  }

  @override
  void dispose() {
    _viewport?.removeListener(_onViewport);
    super.dispose();
  }

  void _listen(PlotViewport? old, PlotViewport? now) {
    old?.removeListener(_onViewport);
    now?.addListener(_onViewport);
    _zoomed = now?.zoomed ?? false;
  }

  /// Rebuilds only when zooming starts or ends (for the cursor); the
  /// painters repaint on every change by themselves.
  void _onViewport() {
    final z = _viewport?.zoomed ?? false;
    if (z != _zoomed && mounted) setState(() => _zoomed = z);
  }

  (double, double) _window(int n) =>
      widget.xRange ?? PlotViewport.resolve(_viewport?.value, n);

  @override
  Widget build(BuildContext context) {
    final s = widget.style ?? PlotStyle.of(context);
    final n = widget._length;
    final vp = _viewport;
    final painter = _DataPainter(
      layers: widget.layers,
      title: widget.title,
      fixedX: widget.xRange,
      viewport: vp,
      length: n,
      yRange: widget.yRange,
      limit: widget.limit,
      revision: widget.revision,
      showXAxis: widget.showXAxis,
      style: s,
    );
    final h = widget.hover;
    final painted = CustomPaint(
      painter: painter,
      foregroundPainter: h == null
          ? null
          : _HoverPainter(
              hover: h,
              viewport: vp,
              data: painter,
              layers: widget.layers,
              style: s,
              showXAxis: widget.showXAxis,
            ),
      size: Size.infinite,
    );
    if (h == null && vp == null) return RepaintBoundary(child: painted);
    final wheelZooms = Scrollable.maybeOf(context) == null;
    return RepaintBoundary(
      child: LayoutBuilder(
        builder: (context, box) {
          final size = box.biggest;
          Rect plotRect() => PlotGeometry(size, xAxis: widget.showXAxis).plot;
          double xAt(double dx) {
            final r = plotRect();
            final (x0, x1) = _window(n);
            return x0 + (dx - r.left) / r.width * (x1 - x0);
          }

          int? trialAt(Offset p) {
            final r = plotRect();
            if (n == 0 || p.dx < r.left || p.dx > r.right) return null;
            return xAt(p.dx).round().clamp(0, n - 1);
          }

          /// Trials per pixel.
          double scale() {
            final (x0, x1) = _window(n);
            return (x1 - x0) / plotRect().width;
          }

          void setHover(Offset p) => h?.value = trialAt(p);

          Widget child = GestureDetector(
            onTapDown: h == null ? null : (d) => setHover(d.localPosition),
            onDoubleTap: vp?.reset,
            // Horizontal drags only, so a plot in a vertical list still
            // scrolls on touch screens.
            onHorizontalDragStart: (d) {
              _panning =
                  vp != null && vp.zoomed && d.kind == PointerDeviceKind.mouse;
              if (!_panning) setHover(d.localPosition);
            },
            onHorizontalDragUpdate: (d) {
              if (_panning) {
                vp!.pan(-d.delta.dx * scale(), n);
              } else if (_touches.length < 2) {
                setHover(d.localPosition);
              }
            },
            onHorizontalDragEnd: (_) => _panning = false,
            child: painted,
          );
          child = MouseRegion(
            cursor: _zoomed ? SystemMouseCursors.grab : MouseCursor.defer,
            onHover: h == null ? null : (e) => setHover(e.localPosition),
            onExit: h == null ? null : (_) => h.value = null,
            child: child,
          );
          if (vp == null || n < 2) return child;
          return Listener(
            onPointerSignal: (e) => _signal(e, vp, n, xAt, scale, wheelZooms),
            onPointerPanZoomStart: (_) => _lastScale = 1,
            onPointerPanZoomUpdate: (e) {
              final dx = e.localPanDelta.dx;
              if (dx != 0) vp.pan(-dx * scale(), n);
              if (e.scale > 0 && e.scale != _lastScale) {
                vp.zoom(_lastScale / e.scale, xAt(e.localPosition.dx), n);
                _lastScale = e.scale;
              }
            },
            onPointerDown: (e) {
              if (e.kind != PointerDeviceKind.touch) return;
              _touches[e.pointer] = e.localPosition;
              if (_touches.length == 2) {
                final [p, q] = _touches.values.toList();
                _pinchTrials = (xAt(p.dx), xAt(q.dx));
                h?.value = null;
              }
            },
            onPointerMove: (e) {
              if (!_touches.containsKey(e.pointer)) return;
              _touches[e.pointer] = e.localPosition;
              final start = _pinchTrials;
              if (_touches.length != 2 || start == null) return;
              final [p, q] = _touches.values.toList();
              // Keep each finger on the trial it started on.
              final apart = (q.dx - p.dx).abs();
              if (apart < 12) return;
              final r = plotRect();
              final span = (start.$2 - start.$1).abs() / apart * r.width;
              final first = start.$1 - (p.dx - r.left) / r.width * span;
              vp.show(first, first + span, n);
            },
            onPointerUp: (e) => _lift(e.pointer),
            onPointerCancel: (e) => _lift(e.pointer),
            child: child,
          );
        },
      ),
    );
  }

  void _lift(int pointer) {
    if (_touches.remove(pointer) != null && _touches.length < 2) {
      _pinchTrials = null;
    }
  }

  /// Wheels and web trackpads: zoom (vertical scroll with Ctrl/⌘, or
  /// without where [wheelZooms]; a pinch) and pan (sideways scroll, or
  /// Shift + scroll). Claims the event only when it acts, so a list around
  /// the plot still scrolls.
  void _signal(
    PointerSignalEvent e,
    PlotViewport vp,
    int n,
    double Function(double dx) xAt,
    double Function() scale,
    bool wheelZooms,
  ) {
    final resolver = GestureBinding.instance.pointerSignalResolver;
    switch (e) {
      case PointerScrollEvent(:final scrollDelta, :final localPosition):
        final keys = HardwareKeyboard.instance;
        final modified = keys.isControlPressed || keys.isMetaPressed;
        var dx = scrollDelta.dx, dy = scrollDelta.dy;
        if (keys.isShiftPressed && dx == 0) (dx, dy) = (dy, 0);
        if (dy != 0 && (modified || wheelZooms)) {
          resolver.register(e, (_) {
            vp.zoom(math.exp(dy * 0.0015), xAt(localPosition.dx), n);
          });
        } else if (dx != 0 && vp.zoomed && dy == 0) {
          resolver.register(e, (_) => vp.pan(dx * scale(), n));
        }
      case PointerScaleEvent(:final scale, :final localPosition) when scale > 0:
        resolver.register(e, (_) {
          vp.zoom(1 / scale, xAt(localPosition.dx), n);
        });
      default:
    }
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
    required this.fixedX,
    required this.viewport,
    required this.length,
    required this.yRange,
    required this.limit,
    required this.revision,
    required this.showXAxis,
    required this.style,
  }) : super(
         repaint: viewport == null
             ? _fontsChanged
             : Listenable.merge([_fontsChanged, viewport]),
       );

  final List<PlotLayer> layers;
  final String? title;
  final (double, double)? fixedX;
  final PlotViewport? viewport;
  final int length;

  /// The visible trials now.
  (double, double) get xRange =>
      fixedX ?? PlotViewport.resolve(viewport?.value, length);
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

  (int, int) _visible(int length, (double, double) x) {
    final first = math.max(0, x.$1.floor());
    var last = math.min(length - 1, x.$2.ceil());
    if (limit != null) last = math.min(last, limit! - 1);
    return (first, last);
  }

  (double, double) _autoY((double, double) x) {
    var lo = double.infinity, hi = double.negativeInfinity;
    for (final l in layers) {
      if (!l.affectsRange) continue;
      final (first, last) = _visible(l.length, x);
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
    final x = xRange;
    final (y0, y1) = lastY = yRange ?? _autoY(x);
    final (x0, x1) = x;
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
          final (first, last) = _visible(layer.length, x);
          _reducer.line(layer.values, first, last, columns);
          _drawRuns(canvas, px, py, ui.PointMode.polygon, layer.width, color);
        case PointLayer():
          final (first, last) = _visible(layer.length, x);
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
          final (first, last) = _visible(layer.length, x);
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
      final at = px(l - 1.0);
      canvas.drawLine(
        Offset(at, r.top),
        Offset(at, r.bottom),
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
      // Trials are whole numbers: no ticks between them when zoomed in.
      if (v != v.roundToDouble()) continue;
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
      fixedX != old.fixedX ||
      !identical(viewport, old.viewport) ||
      length != old.length ||
      yRange != old.yRange ||
      title != old.title ||
      style != old.style ||
      showXAxis != old.showXAxis ||
      !listEquals(layers, old.layers);
}

class _HoverPainter extends CustomPainter {
  _HoverPainter({
    required this.hover,
    required this.viewport,
    required this.data,
    required this.layers,
    required this.style,
    required this.showXAxis,
  }) : super(
         repaint: viewport == null
             ? hover
             : Listenable.merge([hover, viewport]),
       );

  final ValueNotifier<int?> hover;
  final PlotViewport? viewport;
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
      !identical(viewport, old.viewport) ||
      data != old.data ||
      style != old.style ||
      showXAxis != old.showXAxis;
}
