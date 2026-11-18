import 'dart:math' as math;

import 'package:flutter/gestures.dart';
import 'package:material_ui/material_ui.dart';

import 'canvas_support.dart';

/// How "fit to view" frames a canvas's content.
class ViewFit {
  const ViewFit({
    required this.margin,
    required this.maxScale,
    this.minScale = 0.3,
    double? narrowMinScale,
    this.showStart = false,
  }) : narrowMinScale = narrowMinScale ?? minScale;

  /// Space around the content, in canvas units.
  final double margin;

  /// The zoom range of a fit (pinch and the buttons go further).
  final double maxScale, minScale;

  /// The smallest fit on a narrow canvas ([narrowWidth]): on a phone, a
  /// pipeline at 0.3× is unreadable, so the fit stops at a zoom that can be
  /// read and lets the rest overflow.
  final double narrowMinScale;

  /// Where the content overflows: show its start (left, top) instead of its
  /// centre. A pipeline reads from the left.
  final bool showStart;

  /// Canvases narrower than this use [narrowMinScale].
  static const narrowWidth = 600.0;
}

/// What the builder's canvases share about where they look and how a
/// pointer moves them: pan and zoom, fit to view (again when a document is
/// opened), the scroll wheel, pinch, right clicks and long presses, the
/// zoom buttons, and telling a drag of the background (a pan) from a drag
/// of the canvas's own (a node, a card, a wire).
///
/// The canvas provides its content's bounds, the drag and tap hooks, and
/// paints through [viewport].
mixin CanvasViewport<T extends StatefulWidget> on State<T> {
  /// The smallest and largest zoom.
  static const zoomMin = 0.2, zoomMax = 3.0;

  late final canvasFocus = FocusNode(debugLabel: '$runtimeType');

  /// Screen = canvas × [scale] + [pan].
  Offset pan = Offset.zero;
  double scale = 1;
  Size viewSize = Size.zero;

  int _fitted = -1;

  /// Where the last pointer went down: the scale recogniser starts only
  /// after the touch slop, too late to hit a port.
  Offset? _downAt;
  final _right = RightClicks();
  ({Offset pan, double scale, Offset focal})? _panning;
  bool _dragging = false;

  /// The last tap (time and place), for double taps: a double-tap
  /// recogniser would hold every single tap back.
  (Duration, Offset)? _lastTap;
  static final _clock = Stopwatch()..start();

  // ---------------------------------------------------------------------------
  // What the canvas provides
  // ---------------------------------------------------------------------------

  /// The content, in canvas units (null: nothing to show).
  Rect? get contentBounds;

  ViewFit get viewFit;

  /// Bumped when a new document opens ([BuilderModel.opened]): the view
  /// fits again.
  int get openedVersion;

  /// One pointer starts dragging at [at] (screen): start a drag of the
  /// canvas's own and return true, or return false to pan.
  bool startDrag(Offset at);

  /// The drag started by [startDrag] moved to [at] (screen).
  void updateDrag(Offset at);

  /// The drag started by [startDrag] ended.
  void endDrag();

  void onTap(Offset at);

  /// A right click or a long press at [at].
  void openMenu(Offset at);

  // ---------------------------------------------------------------------------
  // The view
  // ---------------------------------------------------------------------------

  Offset toCanvas(Offset screen) => (screen - pan) / scale;
  Offset toScreen(Offset canvas) => canvas * scale + pan;

  /// Frame the content ([viewFit]): each axis that fits is centred.
  void fitView() {
    if (viewSize.isEmpty) return;
    final bounds = contentBounds;
    if (bounds == null) {
      pan = viewSize.center(Offset.zero);
      scale = 1;
      return;
    }
    final fit = viewFit;
    final box = bounds.inflate(fit.margin);
    final least = viewSize.width < ViewFit.narrowWidth
        ? fit.narrowMinScale
        : fit.minScale;
    final s = math
        .min(viewSize.width / box.width, viewSize.height / box.height)
        .clamp(least, fit.maxScale);
    double axis(double view, double lo, double hi) {
      final extent = (hi - lo) * s;
      return extent > view && fit.showStart
          ? -lo * s
          : (view - extent) / 2 - lo * s;
    }

    scale = s;
    pan = Offset(
      axis(viewSize.width, box.left, box.right),
      axis(viewSize.height, box.top, box.bottom),
    );
  }

  /// Zoom by [factor], keeping the canvas point under [focal] in place.
  void zoomBy(double factor, Offset focal) {
    final s = (scale * factor).clamp(zoomMin, zoomMax);
    final c = toCanvas(focal);
    setState(() {
      scale = s;
      pan = focal - c * s;
    });
  }

  /// Whether a tap at [at] is the second of a double tap (which it then
  /// consumes).
  bool doubleTap(Offset at) {
    final now = _clock.elapsed;
    if (_lastTap case (
      final t,
      final p,
    ) when now - t < kDoubleTapTimeout && (p - at).distance < kDoubleTapSlop) {
      _lastTap = null;
      return true;
    }
    _lastTap = (now, at);
    return false;
  }

  /// The last tap does not start a double tap (it hit something).
  void forgetTap() => _lastTap = null;

  @override
  void dispose() {
    canvasFocus.dispose();
    super.dispose();
  }

  // ---------------------------------------------------------------------------
  // Gestures
  // ---------------------------------------------------------------------------

  void _onScaleStart(ScaleStartDetails d) {
    canvasFocus.requestFocus();
    _panning = null;
    _dragging = false;
    if (_right.active) return; // a right click's drag: neither pan nor move
    final at = d.pointerCount == 1
        ? (_downAt ?? d.localFocalPoint)
        : d.localFocalPoint;
    if (d.pointerCount == 1 && startDrag(at)) {
      _dragging = true;
      return;
    }
    _panning = (pan: pan, scale: scale, focal: at);
  }

  void _onScaleUpdate(ScaleUpdateDetails d) {
    final at = d.localFocalPoint;
    if (_panning case (:final pan, :final scale, :final focal)) {
      final s = (scale * d.scale).clamp(zoomMin, zoomMax);
      // Keep the canvas point under the starting focal point under the
      // fingers.
      final c = (focal - pan) / scale;
      setState(() {
        this.scale = s;
        this.pan = at - c * s;
      });
    } else if (_dragging) {
      updateDrag(at);
    }
  }

  void _onScaleEnd(ScaleEndDetails d) {
    _panning = null;
    if (_dragging) {
      _dragging = false;
      endDrag();
    }
  }

  void _onTapUp(TapUpDetails d) {
    canvasFocus.requestFocus();
    onTap(d.localPosition);
  }

  void _openMenu(Offset at) {
    canvasFocus.requestFocus();
    openMenu(at);
  }

  /// The canvas: [painter]'s painting under the pointer handling, then
  /// [overlays] (tips, a legend) and the zoom buttons. Fits the view when
  /// the size is first known and when [openedVersion] changes; [painter]
  /// and [overlays] are called after that, so they see the fitted [pan]
  /// and [scale].
  Widget viewport({
    required CustomPainter Function() painter,
    required KeyEventResult Function(FocusNode, KeyEvent) onKey,
    MouseCursor cursor = SystemMouseCursors.basic,
    void Function(Offset at)? onHover,
    VoidCallback? onExit,
    List<Widget> Function()? overlays,
  }) => LayoutBuilder(
    builder: (context, box) {
      viewSize = box.biggest;
      // Not before the canvas has a size (a first layout can be empty, as
      // the page settles): the fit would be skipped and never come back.
      if (_fitted != openedVersion && !viewSize.isEmpty) {
        _fitted = openedVersion;
        fitView();
      }
      return Focus(
        focusNode: canvasFocus,
        onKeyEvent: onKey,
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
                    onPointerUp: (e) => _right.up(e, _openMenu),
                    onPointerCancel: (_) => _right.cancel(),
                    onPointerSignal: (e) {
                      if (e is PointerScrollEvent) {
                        zoomBy(
                          math.exp(-e.scrollDelta.dy / 400),
                          e.localPosition,
                        );
                      }
                    },
                    child: MouseRegion(
                      cursor: cursor,
                      onHover: onHover == null
                          ? null
                          : (e) => onHover(e.localPosition),
                      onExit: onExit == null ? null : (_) => onExit(),
                      child: GestureDetector(
                        behavior: HitTestBehavior.opaque,
                        onTapUp: _onTapUp,
                        onLongPressStart: (d) => _openMenu(d.localPosition),
                        onScaleStart: _onScaleStart,
                        onScaleUpdate: _onScaleUpdate,
                        onScaleEnd: _onScaleEnd,
                        child: RepaintBoundary(
                          child: CustomPaint(
                            size: Size.infinite,
                            painter: painter(),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
              ...?overlays?.call(),
              Positioned(
                right: 8,
                bottom: 8,
                child: CanvasZoomButtons(
                  onIn: () => zoomBy(1.25, viewSize.center(Offset.zero)),
                  onOut: () => zoomBy(0.8, viewSize.center(Offset.zero)),
                  onFit: () => setState(fitView),
                ),
              ),
            ],
          ),
        ),
      );
    },
  );
}
