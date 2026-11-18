import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/services.dart';
import 'package:material_ui/material_ui.dart';

import '../builder_model.dart';

// What the builder's canvases (network, pipeline, deep) share: text laid
// out once, the dotted grid, hit-test geometry, menus at the pointer, the
// zoom buttons, the keyboard shortcuts, and keeping the browser's own
// context menu out of the way of theirs.

// -----------------------------------------------------------------------------
// Text
// -----------------------------------------------------------------------------

typedef _TextKey = (String, TextStyle, double);

final Map<_TextKey, TextPainter> _texts = {};

/// Laid out text is dropped when fonts change: on the web, glyphs such as
/// subscripts come from fallback fonts that load after the first frame.
bool _fontsListened = false;

/// [s] in [style] on one line (ellipsised past [maxWidth]), laid out once
/// and reused across frames: the canvases repaint on every pointer move
/// while dragging, and laying out every label each time is most of a
/// frame on the web.
TextPainter canvasText(
  String s,
  TextStyle style, {
  double maxWidth = double.infinity,
}) {
  if (!_fontsListened) {
    _fontsListened = true;
    PaintingBinding.instance.systemFonts.addListener(_clearTexts);
  }
  final key = (s, style, maxWidth);
  return _texts.putIfAbsent(key, () {
    if (_texts.length > 4000) _clearTexts();
    return TextPainter(
      text: TextSpan(text: s, style: style),
      textDirection: TextDirection.ltr,
      maxLines: 1,
      ellipsis: '…',
    )..layout(maxWidth: maxWidth);
  });
}

void _clearTexts() {
  for (final t in _texts.values) {
    t.dispose();
  }
  _texts.clear();
}

// -----------------------------------------------------------------------------
// Grid and geometry
// -----------------------------------------------------------------------------

/// The canvases' dotted background: a dot every 40 canvas units (none when
/// they would be closer than 12 pixels), in one draw call.
void paintDotGrid(
  Canvas canvas,
  Size size,
  Offset pan,
  double scale,
  Color color, {
  double radius = 0.8,
}) {
  final step = 40 * scale;
  if (step < 12) return;
  final x0 = pan.dx % step, y0 = pan.dy % step;
  final nx = ((size.width - x0) / step).ceil(),
      ny = ((size.height - y0) / step).ceil();
  if (nx <= 0 || ny <= 0) return;
  final points = Float32List(2 * nx * ny);
  var k = 0;
  for (var i = 0; i < nx; i++) {
    for (var j = 0; j < ny; j++) {
      points[k++] = x0 + i * step;
      points[k++] = y0 + j * step;
    }
  }
  canvas.drawRawPoints(
    ui.PointMode.points,
    points,
    Paint()
      ..color = color
      ..strokeWidth = 2 * radius
      ..strokeCap = StrokeCap.round,
  );
}

/// The distance from [p] to the segment from [a] to [b].
double segmentDistance(Offset p, Offset a, Offset b) {
  final ab = b - a;
  final len2 = ab.distanceSquared;
  if (len2 == 0) return (p - a).distance;
  final t = (((p - a).dx * ab.dx + (p - a).dy * ab.dy) / len2).clamp(0.0, 1.0);
  return (p - (a + ab * t)).distance;
}

/// The distance from [p] to the curve [at] (`t` in 0..1), sampled at 24
/// segments: what hovering or tapping an edge tests.
double curveDistance(Offset p, Offset Function(double t) at) {
  var best = double.infinity;
  var prev = at(0);
  for (var k = 1; k <= 24; k++) {
    final q = at(k / 24);
    final d = segmentDistance(p, prev, q);
    if (d < best) best = d;
    prev = q;
  }
  return best;
}

// -----------------------------------------------------------------------------
// Menus, messages, buttons
// -----------------------------------------------------------------------------

/// A popup menu at [local] (in [context]'s render box).
Future<T?> showMenuAt<T>(
  BuildContext context,
  Offset local,
  List<PopupMenuEntry<T>> items,
) {
  final box = context.findRenderObject()! as RenderBox;
  final g = box.localToGlobal(local);
  final overlay = Overlay.of(context).context.findRenderObject()! as RenderBox;
  return showMenu<T>(
    context: context,
    position: RelativeRect.fromRect(
      Rect.fromLTWH(g.dx, g.dy, 1, 1),
      Offset.zero & overlay.size,
    ),
    items: items,
  );
}

/// [message] in a snack bar, replacing the one shown.
void tellCanvas(BuildContext context, String message) {
  ScaffoldMessenger.maybeOf(context)
    ?..hideCurrentSnackBar()
    ..showSnackBar(
      SnackBar(content: Text(message), duration: const Duration(seconds: 4)),
    );
}

/// Zoom out, fit to view (F) and zoom in, in the canvas's corner.
class CanvasZoomButtons extends StatelessWidget {
  const CanvasZoomButtons({
    super.key,
    required this.onIn,
    required this.onOut,
    required this.onFit,
  });

  final VoidCallback onIn, onOut, onFit;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Material(
      color: scheme.surfaceContainerHigh,
      borderRadius: BorderRadius.circular(8),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          IconButton(
            tooltip: 'Zoom out',
            icon: const Icon(Icons.remove, size: 18),
            onPressed: onOut,
          ),
          IconButton(
            tooltip: 'Fit to view (F)',
            icon: const Icon(Icons.fit_screen_outlined, size: 18),
            onPressed: onFit,
          ),
          IconButton(
            tooltip: 'Zoom in',
            icon: const Icon(Icons.add, size: 18),
            onPressed: onIn,
          ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// Keys
// -----------------------------------------------------------------------------

/// The shortcuts every canvas shares: Ctrl/Cmd+Z undo, Ctrl/Cmd+Shift+Z
/// and Ctrl/Cmd+Y redo, Delete or Backspace delete the selection, Escape
/// clears it (after [onEscape]), F fits the view ([onFit]). Returns null
/// for a key it leaves to the canvas.
KeyEventResult? canvasKey(
  BuilderModel m,
  KeyEvent e, {
  VoidCallback? onEscape,
  VoidCallback? onFit,
}) {
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
    onEscape?.call();
    m.select(null);
    return KeyEventResult.handled;
  }
  if (onFit != null && k == LogicalKeyboardKey.keyF) {
    onFit();
    return KeyEventResult.handled;
  }
  return null;
}

// -----------------------------------------------------------------------------
// Right clicks
// -----------------------------------------------------------------------------

/// Right clicks, from a canvas's raw pointer events. A secondary tap
/// recogniser loses its arena to the canvas's pan (which takes every
/// button) as soon as a mouse or trackpad moves a pixel or two between
/// press and release, so the canvases open their menus from here instead
/// and ignore pans started with the secondary button ([active]).
class RightClicks {
  Offset? _down;

  /// The pointer that is down pressed the secondary button.
  bool get active => _down != null;

  void down(PointerDownEvent e) =>
      _down = e.buttons & kSecondaryButton != 0 ? e.localPosition : null;

  /// Calls [open] where the button went down, if this ends a right click
  /// (not a drag).
  void up(PointerUpEvent e, void Function(Offset at) open) {
    final at = _down;
    _down = null;
    if (at != null && (e.localPosition - at).distance <= 12) open(at);
  }

  void cancel() => _down = null;
}

// -----------------------------------------------------------------------------
// The browser's context menu
// -----------------------------------------------------------------------------

/// Keeps the browser's context menu off [child] on the web, so that a right
/// click reaches the canvas's own menu: left on, Chrome on macOS opens its
/// menu on the button press and keeps the release, so the canvas never
/// sees a secondary tap. Text fields elsewhere keep the browser's menu.
class NoBrowserContextMenu extends StatefulWidget {
  const NoBrowserContextMenu({super.key, required this.child});

  final Widget child;

  @override
  State<NoBrowserContextMenu> createState() => _NoBrowserContextMenuState();
}

class _NoBrowserContextMenuState extends State<NoBrowserContextMenu> {
  /// How many of these the pointer is over (they may nest or overlap while
  /// the canvases swap).
  static int _inside = 0;
  bool _in = false;

  void _enter() {
    if (_in) return;
    _in = true;
    if (_inside++ == 0) BrowserContextMenu.disableContextMenu();
  }

  void _exit() {
    if (!_in) return;
    _in = false;
    if (--_inside == 0) BrowserContextMenu.enableContextMenu();
  }

  @override
  void dispose() {
    _exit();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!kIsWeb) return widget.child;
    return MouseRegion(
      opaque: false,
      onEnter: (_) => _enter(),
      onExit: (_) => _exit(),
      child: widget.child,
    );
  }
}
