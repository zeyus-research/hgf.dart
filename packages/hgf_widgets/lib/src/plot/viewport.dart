import 'dart:math' as math;

import 'package:material_ui/material_ui.dart';

/// The visible x window (trials) of one or more linked [TrajectoryPlot]s,
/// as `(first, last)` x coordinates; `null` shows every trial.
///
/// Share one between plots, as with their hover notifier, so that zooming
/// or panning one moves them all. The window is kept inside the series
/// (see [window]); each method takes the series' [length] for that.
///
/// Changing it repaints the plots without rebuilding them.
class PlotViewport extends ValueNotifier<(double, double)?> {
  PlotViewport([super.value]);

  /// The narrowest window, in trials (or the whole series if shorter).
  static const double minSpan = 4;

  /// Every trial of a series of [length].
  static (double, double) all(int length) => (0.0, math.max(1.0, length - 1.0));

  /// [w] moved inside a series of [length] (the whole series when [w] is
  /// null or at least as wide).
  static (double, double) resolve((double, double)? w, int length) {
    final full = all(length);
    if (w == null) return full;
    final fullSpan = full.$2;
    final span = math.min(w.$2 - w.$1, fullSpan);
    if (!(span > 0) || span >= fullSpan - 1e-9) return full;
    final first = w.$1.clamp(0.0, fullSpan - span);
    return (first, first + span);
  }

  /// Whether a part of the series is shown.
  bool get zoomed => value != null;

  /// The visible window of a series of [length].
  (double, double) window(int length) => resolve(value, length);

  /// Show `first..last`, widened to [minSpan] around its centre and moved
  /// inside the series; the whole series resets to null.
  void show(double first, double last, int length) {
    final fullSpan = all(length).$2;
    var span = last - first;
    final min = math.min(minSpan, fullSpan);
    if (span < min) {
      final c = (first + last) / 2;
      span = min;
      first = c - span / 2;
    }
    if (!span.isFinite || span >= fullSpan - 1e-9) {
      value = null;
      return;
    }
    first = first.clamp(0.0, fullSpan - span);
    value = (first, first + span);
  }

  /// Scale the window by [factor] (< 1 zooms in) about the trial [anchor],
  /// which stays where it is on screen.
  void zoom(double factor, double anchor, int length) {
    final (a, b) = window(length);
    final span = b - a;
    final next = span * factor;
    final first = anchor - (anchor - a) * next / span;
    show(first, first + next, length);
  }

  /// Move the window by [dx] trials (when zoomed).
  void pan(double dx, int length) {
    if (!zoomed) return;
    final (a, b) = window(length);
    show(a + dx, b + dx, length);
  }

  /// Move the window so that trial [x] is visible (when zoomed), e.g. to
  /// follow playback.
  void follow(double x, int length) {
    if (!zoomed) return;
    final (a, b) = window(length);
    final span = b - a;
    if (x > b) {
      show(x - span * 0.9, x + span * 0.1, length);
    } else if (x < a) {
      show(x - span * 0.1, x + span * 0.9, length);
    }
  }

  /// Show every trial.
  void reset() => value = null;
}

/// An overview of a [PlotViewport]: the visible window as a thumb on a
/// track over the whole series (drag it, or tap the track, to pan), zoom
/// buttons, a reset, and the visible trials.
class PlotViewportBar extends StatelessWidget {
  const PlotViewportBar({
    super.key,
    required this.viewport,
    required this.length,
    this.unit = 'trials',
  });

  final PlotViewport viewport;

  /// The series' length.
  final int length;

  /// What the x axis counts.
  final String unit;

  /// The bar's height.
  static const double height = 32;

  static const String help =
      'Zoom: Ctrl/⌘ + scroll or pinch over a plot (plain scroll where the '
      'plots don\'t scroll). Pan: drag a zoomed plot with the mouse, scroll '
      'sideways, or drag the bar. Double-click a plot to show everything.';

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final small = theme.textTheme.bodySmall?.copyWith(
      color: scheme.onSurfaceVariant,
      fontFeatures: const [FontFeature.tabularFigures()],
    );
    return ValueListenableBuilder(
      valueListenable: viewport,
      builder: (context, _, _) {
        final (a, b) = viewport.window(length);
        final zoomed = viewport.zoomed && length > 1;
        final centre = (a + b) / 2;
        Widget button(String tip, IconData icon, VoidCallback? onPressed) =>
            IconButton(
              tooltip: tip,
              icon: Icon(icon, size: 18),
              visualDensity: VisualDensity.compact,
              onPressed: length > 1 ? onPressed : null,
            );
        return SizedBox(
          height: height,
          child: Row(
            children: [
              button(
                'Zoom out',
                Icons.zoom_out,
                zoomed ? () => viewport.zoom(2, centre, length) : null,
              ),
              Expanded(
                child: Tooltip(
                  message: help,
                  waitDuration: const Duration(milliseconds: 600),
                  child: _Track(
                    viewport: viewport,
                    length: length,
                    window: (a, b),
                    zoomed: zoomed,
                  ),
                ),
              ),
              button(
                'Zoom in',
                Icons.zoom_in,
                () => viewport.zoom(0.5, centre, length),
              ),
              button(
                'Show everything',
                Icons.zoom_out_map,
                zoomed ? viewport.reset : null,
              ),
              const SizedBox(width: 4),
              Text(
                zoomed
                    ? '$unit ${a.round()}–${b.round()} of $length'
                    : 'all $length $unit',
                style: small,
              ),
            ],
          ),
        );
      },
    );
  }
}

class _Track extends StatelessWidget {
  const _Track({
    required this.viewport,
    required this.length,
    required this.window,
    required this.zoomed,
  });

  final PlotViewport viewport;
  final int length;
  final (double, double) window;
  final bool zoomed;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return LayoutBuilder(
      builder: (context, box) {
        final width = box.maxWidth;
        final full = PlotViewport.all(length).$2;
        double trialAt(double x) => x / width * full;
        return MouseRegion(
          cursor: zoomed ? SystemMouseCursors.grab : MouseCursor.defer,
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            onTapDown: zoomed
                ? (d) {
                    final span = window.$2 - window.$1;
                    final c = trialAt(d.localPosition.dx);
                    viewport.show(c - span / 2, c + span / 2, length);
                  }
                : null,
            onHorizontalDragUpdate: zoomed
                ? (d) => viewport.pan(d.delta.dx / width * full, length)
                : null,
            child: CustomPaint(
              size: Size(width, 32),
              painter: _TrackPainter(
                window: window,
                full: full,
                track: scheme.surfaceContainerHighest,
                thumb: zoomed
                    ? scheme.primary.withValues(alpha: 0.55)
                    : scheme.outlineVariant,
              ),
            ),
          ),
        );
      },
    );
  }
}

class _TrackPainter extends CustomPainter {
  _TrackPainter({
    required this.window,
    required this.full,
    required this.track,
    required this.thumb,
  });

  final (double, double) window;
  final double full;
  final Color track, thumb;

  @override
  void paint(Canvas canvas, Size size) {
    const h = 8.0;
    final top = (size.height - h) / 2;
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(0, top, size.width, h),
        const Radius.circular(h / 2),
      ),
      Paint()..color = track,
    );
    final l = window.$1 / full * size.width;
    final r = math.max(l + 6, window.$2 / full * size.width);
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTRB(l, top, math.min(size.width, r), top + h),
        const Radius.circular(h / 2),
      ),
      Paint()..color = thumb,
    );
  }

  @override
  bool shouldRepaint(_TrackPainter old) =>
      window != old.window ||
      full != old.full ||
      track != old.track ||
      thumb != old.thumb;
}
