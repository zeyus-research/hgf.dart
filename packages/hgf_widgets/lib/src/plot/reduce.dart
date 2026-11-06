import 'dart:typed_data';

/// Reduces a series to what can be seen at a given pixel width, so drawing
/// costs O(pixels), not O(samples).
///
/// Below two samples per pixel column every sample is kept. Above that,
/// each column keeps its finite minimum and maximum, in the order they
/// occur, which preserves the series' envelope (spikes stay visible, unlike
/// with stride sampling). Non-finite samples break the series into runs.
///
/// The buffers are reused between calls; read [x], [y] (and [y2] for an
/// envelope) up to [length], split at [runEnds].
final class SeriesReducer {
  /// Trial coordinate of each output point.
  Float64List x = Float64List(256);

  /// Value of each output point (an envelope's lower bound).
  Float64List y = Float64List(256);

  /// An envelope's upper bound ([envelope] only).
  Float64List y2 = Float64List(256);

  /// Number of output points.
  int length = 0;

  /// The exclusive end of each run of finite points, in order.
  final List<int> runEnds = [];

  void _reset(int capacity) {
    length = 0;
    runEnds.clear();
    if (x.length < capacity) {
      x = Float64List(capacity);
      y = Float64List(capacity);
      y2 = Float64List(capacity);
    }
  }

  void _endRun() {
    if (runEnds.isEmpty ? length > 0 : runEnds.last < length) {
      runEnds.add(length);
    }
  }

  /// The samples `values[first..last]` (inclusive), drawn [columns] pixels
  /// wide.
  void line(Float64List values, int first, int last, int columns) {
    final count = last - first + 1;
    if (count <= 0 || columns <= 0) {
      _reset(0);
      return;
    }
    if (count <= 2 * columns) {
      _reset(count);
      for (var t = first; t <= last; t++) {
        final v = values[t];
        if (v.isFinite) {
          x[length] = t.toDouble();
          y[length++] = v;
        } else {
          _endRun();
        }
      }
      _endRun();
      return;
    }
    _reset(2 * columns);
    var start = first;
    for (var c = 0; c < columns; c++) {
      final end = first + ((c + 1) * count) ~/ columns; // exclusive
      var lo = double.infinity, hi = double.negativeInfinity;
      var loAt = -1, hiAt = -1;
      var gap = false;
      for (var t = start; t < end; t++) {
        final v = values[t];
        if (!v.isFinite) {
          gap = true;
          continue;
        }
        if (v < lo) {
          lo = v;
          loAt = t;
        }
        if (v > hi) {
          hi = v;
          hiAt = t;
        }
      }
      if (loAt < 0) {
        _endRun();
      } else {
        // A column with a gap in it breaks the run before its points, so
        // the line doesn't bridge the missing stretch.
        if (gap && !values[start].isFinite) _endRun();
        final xc = (start + end - 1) * 0.5;
        if (loAt == hiAt) {
          x[length] = xc;
          y[length++] = lo;
        } else {
          final (a, b) = loAt < hiAt ? (lo, hi) : (hi, lo);
          x[length] = xc;
          y[length++] = a;
          x[length] = xc;
          y[length++] = b;
        }
        if (gap && !values[end - 1].isFinite) _endRun();
      }
      start = end;
    }
    _endRun();
  }

  /// An envelope between the bounds `lower(t)` and `upper(t)` for
  /// `t in first..last` (inclusive), drawn [columns] pixels wide. One point
  /// per sample, or per column once there are more samples than columns:
  /// [y] holds the column's lowest lower bound, [y2] its highest upper one.
  void envelope(
    int first,
    int last,
    int columns,
    double Function(int t) lower,
    double Function(int t) upper,
  ) {
    final count = last - first + 1;
    if (count <= 0 || columns <= 0) {
      _reset(0);
      return;
    }
    final dense = count > columns;
    _reset(dense ? columns : count);
    var start = first;
    final buckets = dense ? columns : count;
    for (var c = 0; c < buckets; c++) {
      final end = dense ? first + ((c + 1) * count) ~/ columns : start + 1;
      var lo = double.infinity, hi = double.negativeInfinity;
      var any = false;
      for (var t = start; t < end; t++) {
        final a = lower(t), b = upper(t);
        if (!a.isFinite || !b.isFinite) continue;
        any = true;
        if (a < lo) lo = a;
        if (b > hi) hi = b;
      }
      if (any) {
        x[length] = dense ? (start + end - 1) * 0.5 : start.toDouble();
        y[length] = lo;
        y2[length++] = hi;
      } else {
        _endRun();
      }
      start = end;
    }
    _endRun();
  }
}

/// The finite minimum and maximum of `values[first..last]`, or null if
/// there are none.
(double, double)? finiteRange(Float64List values, int first, int last) {
  var lo = double.infinity, hi = double.negativeInfinity;
  for (var t = first; t <= last; t++) {
    final v = values[t];
    if (!v.isFinite) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? (lo, hi) : null;
}

/// About [target] "nice" tick values (steps of 1, 2 or 5 × 10ⁿ) covering
/// `[lo, hi]`.
List<double> niceTicks(double lo, double hi, {int target = 4}) {
  if (!(hi > lo) || !lo.isFinite || !hi.isFinite) return const [];
  final raw = (hi - lo) / target;
  var mag = 1.0;
  while (mag * 10 <= raw) {
    mag *= 10;
  }
  while (mag > raw) {
    mag /= 10;
  }
  final f = raw / mag;
  final step =
      (f < 1.5
          ? 1
          : f < 3.5
          ? 2
          : f < 7.5
          ? 5
          : 10) *
      mag;
  // Ranges too narrow for doubles to step through (or underflowing) get no
  // ticks rather than an endless loop.
  if (!(step > 0) || (hi - lo) / step > 100 || lo / step == hi / step) {
    return const [];
  }
  final out = <double>[];
  for (var k = (lo / step).ceil(); k * step <= hi + step * 1e-9; k++) {
    out.add(k == 0 ? 0.0 : k * step);
  }
  return out;
}
