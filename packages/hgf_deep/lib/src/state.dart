import 'dart:typed_data';

import 'config.dart';

/// The beliefs of one layer for [width] samples at once.
///
/// Every field is a `Float64List` of `nodes * width` values laid out
/// `[node][sample]`, so a weight contraction over the layer's nodes runs over
/// contiguous sample rows. The network's own state has width 1; a batched
/// step tiles it to the batch width. The six volatility-level fields are
/// `null` on a layer without a volatility parent (pyhgf's `None` leaves).
final class LayerState {
  LayerState._(
    this.nodes,
    this.width,
    this.mean,
    this.precision,
    this.expectedMean,
    this.expectedPrecision,
    this.conditionalExpectedPrecision,
    this.effectivePrecision,
    this.valuePredictionError,
    this.meanVol,
    this.precisionVol,
    this.expectedMeanVol,
    this.expectedPrecisionVol,
    this.effectivePrecisionVol,
    this.volatilityPredictionError,
    this.knownVariance,
  );

  /// pyhgf's defaults: means 0, precisions 1, effective precisions and
  /// prediction errors 0.
  factory LayerState.initial(int nodes, {required bool volatility}) {
    Float64List filled(double v) => Float64List(nodes)..fillRange(0, nodes, v);
    return LayerState._(
      nodes,
      1,
      filled(0),
      filled(1),
      filled(0),
      filled(1),
      filled(1),
      filled(0),
      filled(0),
      volatility ? filled(0) : null,
      volatility ? filled(1) : null,
      volatility ? filled(0) : null,
      volatility ? filled(1) : null,
      volatility ? filled(0) : null,
      volatility ? filled(0) : null,
      volatility ? filled(1) : null,
    );
  }

  /// A width-[width] state with every sample a copy of this width-1 state.
  LayerState tile(int width) {
    assert(this.width == 1);
    Float64List t(Float64List src) {
      final out = Float64List(nodes * width);
      for (var i = 0; i < nodes; i++) {
        out.fillRange(i * width, i * width + width, src[i]);
      }
      return out;
    }

    Float64List? tn(Float64List? src) => src == null ? null : t(src);
    return LayerState._(
      nodes,
      width,
      t(mean),
      t(precision),
      t(expectedMean),
      t(expectedPrecision),
      t(conditionalExpectedPrecision),
      t(effectivePrecision),
      t(valuePredictionError),
      tn(meanVol),
      tn(precisionVol),
      tn(expectedMeanVol),
      tn(expectedPrecisionVol),
      tn(effectivePrecisionVol),
      tn(volatilityPredictionError),
      tn(knownVariance),
    );
  }

  /// Overwrite this state's samples with copies of the width-1 [template].
  void retile(LayerState template) {
    assert(template.width == 1 && template.nodes == nodes);
    void t(Float64List dst, Float64List src) {
      for (var i = 0; i < nodes; i++) {
        dst.fillRange(i * width, i * width + width, src[i]);
      }
    }

    for (final f in LayerField.values) {
      final dst = field(f);
      if (dst != null) t(dst, template.field(f)!);
    }
    if (knownVariance != null) t(knownVariance!, template.knownVariance!);
  }

  /// A deep copy.
  LayerState copy() {
    Float64List c(Float64List s) => Float64List.fromList(s);
    Float64List? cn(Float64List? s) => s == null ? null : c(s);
    return LayerState._(
      nodes,
      width,
      c(mean),
      c(precision),
      c(expectedMean),
      c(expectedPrecision),
      c(conditionalExpectedPrecision),
      c(effectivePrecision),
      c(valuePredictionError),
      cn(meanVol),
      cn(precisionVol),
      cn(expectedMeanVol),
      cn(expectedPrecisionVol),
      cn(effectivePrecisionVol),
      cn(volatilityPredictionError),
      cn(knownVariance),
    );
  }

  /// Copy every field of [other] (same shape) into this state.
  void setFrom(LayerState other) {
    assert(other.nodes == nodes && other.width == width);
    for (final f in LayerField.values) {
      final dst = field(f);
      if (dst != null) dst.setAll(0, other.field(f)!);
    }
    knownVariance?.setAll(0, other.knownVariance!);
  }

  /// Number of nodes.
  final int nodes;

  /// Number of samples held side by side.
  final int width;

  /// Whether the layer has an internal volatility level.
  bool get hasVolatility => meanVol != null;

  // Value level.
  final Float64List mean;
  final Float64List precision;
  final Float64List expectedMean;

  /// Marginal predicted precision π̂.
  final Float64List expectedPrecision;

  /// Conditional predicted precision π̄ (given the value parents).
  final Float64List conditionalExpectedPrecision;

  /// Effective precision γ = Ω·π̂ of the value-level prediction.
  final Float64List effectivePrecision;
  final Float64List valuePredictionError;

  // Volatility level (internal).
  final Float64List? meanVol;
  final Float64List? precisionVol;
  final Float64List? expectedMeanVol;
  final Float64List? expectedPrecisionVol;
  final Float64List? effectivePrecisionVol;
  final Float64List? volatilityPredictionError;

  /// The variance of the value-level prediction that does not depend on the
  /// volatility level (the carried `1/π`), stored at prediction time for
  /// the eHGF and unbounded volatility posteriors under
  /// [DeepFixes.logSpaceVolatility]. Internal: not a [LayerField].
  final Float64List? knownVariance;

  /// The field's values (`null` for a volatility field the layer lacks).
  Float64List? field(LayerField f) => switch (f) {
    LayerField.mean => mean,
    LayerField.precision => precision,
    LayerField.expectedMean => expectedMean,
    LayerField.expectedPrecision => expectedPrecision,
    LayerField.conditionalExpectedPrecision => conditionalExpectedPrecision,
    LayerField.effectivePrecision => effectivePrecision,
    LayerField.valuePredictionError => valuePredictionError,
    LayerField.meanVol => meanVol,
    LayerField.precisionVol => precisionVol,
    LayerField.expectedMeanVol => expectedMeanVol,
    LayerField.expectedPrecisionVol => expectedPrecisionVol,
    LayerField.effectivePrecisionVol => effectivePrecisionVol,
    LayerField.volatilityPredictionError => volatilityPredictionError,
  };
}
