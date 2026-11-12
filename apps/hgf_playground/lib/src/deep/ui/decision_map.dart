import 'dart:math' as math;
import 'dart:typed_data';

import 'package:material_ui/material_ui.dart';
import 'package:hgf_deep/datasets.dart';
import 'package:hgf_widgets/hgf_widgets.dart';

import '../deep_playground.dart';

/// The task's input space with the network's prediction over it: class
/// regions (2-D tasks) or the predicted curve (sine), the data, and the
/// traced input. Tap or drag to trace another input.
class DecisionMap extends StatelessWidget {
  const DecisionMap({super.key, required this.deep});

  final DeepPlayground deep;

  @override
  Widget build(BuildContext context) {
    final colors = HgfColors.of(context);
    final scheme = Theme.of(context).colorScheme;
    return LayoutBuilder(
      builder: (context, box) {
        final size = Size(box.maxWidth, box.maxHeight);
        void pick(Offset p) {
          final (lo, hi) = deep.task.inputRange;
          final rect = _plotRect(size);
          final fx = ((p.dx - rect.left) / rect.width).clamp(0.0, 1.0);
          final fy = ((p.dy - rect.top) / rect.height).clamp(0.0, 1.0);
          final x = lo + (hi - lo) * fx;
          deep.traceCustomInput(
            deep.task.isRegression ? [x] : [x, hi - (hi - lo) * fy],
          );
        }

        return GestureDetector(
          onTapUp: (d) => pick(d.localPosition),
          onPanUpdate: (d) => pick(d.localPosition),
          child: CustomPaint(
            size: size,
            painter: _MapPainter(
              deep: deep,
              revision: deep.revision,
              input: deep.traceInput,
              palette: colors.categories,
              ink: scheme.onSurface,
              faint: scheme.outlineVariant,
              surface: scheme.surface,
              belief: colors.belief,
              observation: colors.observation,
              label: Theme.of(
                context,
              ).textTheme.labelSmall!.copyWith(color: scheme.onSurfaceVariant),
            ),
          ),
        );
      },
    );
  }
}

Rect _plotRect(Size size) {
  final side = math.min(size.width - 8, size.height - 22);
  return Rect.fromLTWH(
    (size.width - side) / 2,
    4,
    math.max(10, side),
    math.max(10, side),
  );
}

class _MapPainter extends CustomPainter {
  _MapPainter({
    required this.deep,
    required this.revision,
    required this.input,
    required this.palette,
    required this.ink,
    required this.faint,
    required this.surface,
    required this.belief,
    required this.observation,
    required this.label,
  });

  final DeepPlayground deep;
  final int revision;
  final List<double> input;
  final List<Color> palette;
  final Color ink, faint, surface, belief, observation;
  final TextStyle label;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = _plotRect(size);
    final (lo, hi) = deep.task.inputRange;
    canvas.drawRect(rect, Paint()..color = surface);
    if (deep.task.isRegression) {
      _curve(canvas, rect, lo, hi);
    } else {
      _regions(canvas, rect);
      _points(canvas, rect, lo, hi);
      final p = Offset(
        rect.left + (input[0] - lo) / (hi - lo) * rect.width,
        rect.top + (hi - input[1]) / (hi - lo) * rect.height,
      );
      _crosshair(canvas, p, rect);
    }
    canvas.drawRect(
      rect,
      Paint()
        ..color = faint
        ..style = PaintingStyle.stroke,
    );
    final tp = TextPainter(
      text: TextSpan(
        text: deep.task.isRegression
            ? 'prediction (line) and data; tap to trace an x'
            : 'predicted class regions; tap or drag to trace a point',
        style: label,
      ),
      textDirection: TextDirection.ltr,
      maxLines: 1,
      ellipsis: '…',
    )..layout(maxWidth: size.width);
    tp.paint(canvas, Offset((size.width - tp.width) / 2, rect.bottom + 4));
  }

  void _regions(Canvas canvas, Rect rect) {
    final g = deep.grid;
    final n = deep.gridSize;
    final cell = Size(rect.width / n, rect.height / n);
    final binary = g.cols == 1;
    final paint = Paint()..isAntiAlias = false;
    for (var r = 0; r < n; r++) {
      for (var c = 0; c < n; c++) {
        final k = r * n + c;
        Color color;
        if (binary) {
          final p = g.data[k];
          if (!p.isFinite) {
            color = ink;
          } else {
            // Blend the two class colours by the probability; confident
            // regions are stronger.
            final conf = (p - 0.5).abs() * 2;
            color = Color.lerp(
              surface,
              p >= 0.5 ? palette[1] : palette[0],
              0.15 + 0.45 * conf,
            )!;
          }
        } else {
          var best = 0;
          var max = -1.0;
          for (var j = 0; j < g.cols; j++) {
            final v = g.data[k * g.cols + j];
            if (v > max) {
              max = v;
              best = j;
            }
          }
          color = Color.lerp(
            surface,
            palette[best % palette.length],
            0.15 +
                0.45 *
                    ((max - 1 / g.cols) * g.cols / (g.cols - 1)).clamp(
                      0.0,
                      1.0,
                    ),
          )!;
        }
        canvas.drawRect(
          Rect.fromLTWH(
            rect.left + c * cell.width,
            rect.top + r * cell.height,
            cell.width + 0.5,
            cell.height + 0.5,
          ),
          paint..color = color,
        );
      }
    }
  }

  void _points(Canvas canvas, Rect rect, double lo, double hi) {
    void draw(Dataset ds, bool hollow) {
      for (var i = 0; i < ds.length; i++) {
        final p = Offset(
          rect.left + (ds.x.at(i, 0) - lo) / (hi - lo) * rect.width,
          rect.top + (hi - ds.x.at(i, 1)) / (hi - lo) * rect.height,
        );
        if (!rect.contains(p)) continue;
        final color = palette[ds.labels![i] % palette.length];
        canvas.drawCircle(
          p,
          2.2,
          Paint()
            ..color = color
            ..style = hollow ? PaintingStyle.stroke : PaintingStyle.fill
            ..strokeWidth = 1,
        );
      }
    }

    draw(deep.train, false);
    draw(deep.test, true);
  }

  void _curve(Canvas canvas, Rect rect, double lo, double hi) {
    const yLo = -1.8, yHi = 1.8;
    Offset at(double x, double y) => Offset(
      rect.left + (x - lo) / (hi - lo) * rect.width,
      rect.top + (yHi - y) / (yHi - yLo) * rect.height,
    );
    for (final ds in [deep.train, deep.test]) {
      for (var i = 0; i < ds.length; i++) {
        final p = at(ds.x.data[i], ds.y.data[i]);
        if (rect.contains(p)) {
          canvas.drawCircle(
            p,
            1.8,
            Paint()..color = observation.withValues(alpha: 0.6),
          );
        }
      }
    }
    final g = deep.grid;
    final n = g.rows;
    final path = Path();
    var started = false;
    for (var i = 0; i < n; i++) {
      final x = lo + (hi - lo) * i / (n - 1);
      final y = g.data[i];
      if (!y.isFinite) {
        started = false;
        continue;
      }
      final p = at(x, y.clamp(yLo - 1, yHi + 1));
      started ? path.lineTo(p.dx, p.dy) : path.moveTo(p.dx, p.dy);
      started = true;
    }
    canvas.save();
    canvas.clipRect(rect);
    canvas.drawPath(
      path,
      Paint()
        ..color = belief
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );
    final tx = rect.left + (input[0] - lo) / (hi - lo) * rect.width;
    canvas.drawLine(
      Offset(tx, rect.top),
      Offset(tx, rect.bottom),
      Paint()
        ..color = ink
        ..strokeWidth = 1.2,
    );
    canvas.restore();
  }

  void _crosshair(Canvas canvas, Offset p, Rect rect) {
    if (!rect.inflate(1).contains(p)) return;
    final paint = Paint()
      ..color = ink
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;
    canvas.drawCircle(p, 7, paint);
    canvas.drawLine(p - const Offset(11, 0), p - const Offset(4, 0), paint);
    canvas.drawLine(p + const Offset(4, 0), p + const Offset(11, 0), paint);
    canvas.drawLine(p - const Offset(0, 11), p - const Offset(0, 4), paint);
    canvas.drawLine(p + const Offset(0, 4), p + const Offset(0, 11), paint);
  }

  @override
  bool shouldRepaint(_MapPainter old) =>
      old.revision != revision ||
      old.deep != deep ||
      !_sameInput(old.input, input) ||
      old.surface != surface;

  static bool _sameInput(List<double> a, List<double> b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}

/// Train and test scores per epoch.
class ScoresPlot extends StatelessWidget {
  const ScoresPlot({super.key, required this.deep, this.loss = false});

  final DeepPlayground deep;

  /// Losses instead of scores.
  final bool loss;

  @override
  Widget build(BuildContext context) {
    final c = HgfColors.of(context);
    final h = deep.history;
    Float64List series(double Function(EpochRecord) f) =>
        Float64List.fromList([for (final r in h) f(r)]);
    final cls = deep.task.classes > 0;
    final name = loss
        ? (cls ? 'cross-entropy' : 'MSE')
        : (cls ? 'accuracy' : 'R²');
    return TrajectoryPlot(
      title: '$name per epoch',
      revision: deep.revision,
      yRange: loss ? null : (cls ? (0.0, 1.0) : (-0.2, 1.0)),
      layers: [
        LineLayer(
          series((r) => loss ? r.trainLoss : r.trainScore),
          color: c.observation,
          label: 'train (before learning each sample)',
        ),
        LineLayer(
          series((r) => loss ? r.testLoss : r.testScore),
          color: c.belief,
          label: 'test',
        ),
        if (!loss && cls)
          ReferenceLine(
            1 / deep.task.classes,
            color: c.overflow,
            label: 'chance',
          ),
      ],
    );
  }
}
