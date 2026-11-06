import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui';

/// Something drawn in a [TrajectoryPlot], indexed by trial (sample `t` is at
/// x = t). Non-finite values are gaps.
///
/// Layers hold the series by reference: to redraw after the data changed in
/// place (e.g. `Network.inputData(into: ...)`), bump the plot's `revision`.
sealed class PlotLayer {
  const PlotLayer({this.color, this.label});

  /// Defaults to the theme's primary colour.
  final Color? color;

  /// Shown in the hover readout; layers without one are not listed.
  final String? label;

  /// Whether the automatic y range should cover this layer.
  bool get affectsRange;

  /// The finite range of the layer over trials `first..last`, or null.
  (double, double)? rangeOver(int first, int last);

  /// The layer's value at trial [t], for the hover readout.
  double valueAt(int t);

  /// Number of samples.
  int get length;
}

/// A line through the samples.
final class LineLayer extends PlotLayer {
  const LineLayer(
    this.values, {
    super.color,
    super.label,
    this.width = 1.5,
    this.affectsRange = true,
  });

  final Float64List values;
  final double width;

  @override
  final bool affectsRange;

  @override
  int get length => values.length;

  @override
  double valueAt(int t) => values[t];

  @override
  (double, double)? rangeOver(int first, int last) {
    var lo = double.infinity, hi = double.negativeInfinity;
    for (var t = first; t <= last; t++) {
      final v = values[t];
      if (!v.isFinite) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    return lo <= hi ? (lo, hi) : null;
  }
}

/// A dot per sample (observations).
final class PointLayer extends PlotLayer {
  const PointLayer(
    this.values, {
    super.color,
    super.label,
    this.radius = 2.0,
    this.affectsRange = true,
  });

  final Float64List values;
  final double radius;

  @override
  final bool affectsRange;

  @override
  int get length => values.length;

  @override
  double valueAt(int t) => values[t];

  @override
  (double, double)? rangeOver(int first, int last) =>
      LineLayer(values).rangeOver(first, last);
}

/// A shaded band, either `center ± scale / √precision` (a belief and its
/// uncertainty) or between explicit bounds.
///
/// By default the band does not widen the automatic y range: early in a
/// series beliefs are vague, and their bands would flatten everything else.
final class BandLayer extends PlotLayer {
  /// `center ± scale / √precision`.
  const BandLayer.precision(
    Float64List center,
    Float64List precision, {
    this.scale = 1.0,
    super.color,
    super.label,
    this.affectsRange = false,
  }) : _a = center,
       _b = precision,
       _fromPrecision = true;

  /// Between [lower] and [upper].
  const BandLayer.bounds(
    Float64List lower,
    Float64List upper, {
    super.color,
    super.label,
    this.affectsRange = false,
  }) : _a = lower,
       _b = upper,
       scale = 1.0,
       _fromPrecision = false;

  final Float64List _a, _b;
  final bool _fromPrecision;

  /// The band's half-width in standard deviations ([BandLayer.precision]).
  final double scale;

  @override
  final bool affectsRange;

  @override
  int get length => _a.length;

  /// The band's lower bound at trial [t].
  double lower(int t) =>
      _fromPrecision ? _a[t] - scale / math.sqrt(_b[t]) : _a[t];

  /// The band's upper bound at trial [t].
  double upper(int t) =>
      _fromPrecision ? _a[t] + scale / math.sqrt(_b[t]) : _b[t];

  /// The half-width at trial [t] for [BandLayer.precision], else the
  /// midpoint.
  @override
  double valueAt(int t) => _fromPrecision ? scale / math.sqrt(_b[t]) : _a[t];

  @override
  (double, double)? rangeOver(int first, int last) {
    var lo = double.infinity, hi = double.negativeInfinity;
    for (var t = first; t <= last; t++) {
      final a = lower(t), b = upper(t);
      if (!a.isFinite || !b.isFinite) continue;
      if (a < lo) lo = a;
      if (b > hi) hi = b;
    }
    return lo <= hi ? (lo, hi) : null;
  }
}

/// A horizontal reference line at [value] (e.g. chance level).
final class ReferenceLine extends PlotLayer {
  const ReferenceLine(this.value, {super.color, super.label});

  final double value;

  @override
  bool get affectsRange => false;

  @override
  int get length => 0;

  @override
  double valueAt(int t) => value;

  @override
  (double, double)? rangeOver(int first, int last) => (value, value);
}
