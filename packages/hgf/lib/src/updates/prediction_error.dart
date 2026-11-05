part of '../network.dart';

/// Prediction-error steps. Ports of rshgf
/// `src/updates/nodalised/prediction_error/*`.
extension on Network {
  /// δ = μ - μ̂ and Δ = π̂/π + π̂·δ² - 1. A node with children is observed
  /// when at least one of its children was.
  @pragma('vm:unsafe:no-bounds-checks')
  void _predictionErrorContinuous(int i) {
    final mean = _mean[i];
    final expectedMean = _expectedMean[i];
    final expectedPrecision = _expectedPrecision[i];
    final delta = mean - expectedMean;
    _valuePredictionError[i] = delta;
    _volatilityPredictionError[i] =
        (expectedPrecision / _precision[i]) +
        expectedPrecision * delta * delta -
        1.0;
    if (_isLeaf[i] == 0) {
      _observed[i] = _hasObservations(i) ? 1.0 : 0.0;
    }
  }

  /// δ = (u - μ̂)·observed / π̂; the binary node's precision becomes π̂.
  @pragma('vm:unsafe:no-bounds-checks')
  void _predictionErrorBinary(int i) {
    final expectedPrecision = _expectedPrecision[i];
    _valuePredictionError[i] =
        (_mean[i] - _expectedMean[i]) * _observed[i] / expectedPrecision;
    _precision[i] = expectedPrecision;
  }

  /// Whether any value or volatility child of [i] reported an observation.
  @pragma('vm:unsafe:no-bounds-checks')
  bool _hasObservations(int i) {
    for (var e = _vcStart[i], end = _vcStart[i + 1]; e < end; e++) {
      if (_observed[_vcIdx[e]] != 0.0) return true;
    }
    for (var e = _volcStart[i], end = _volcStart[i + 1]; e < end; e++) {
      if (_observed[_volcIdx[e]] != 0.0) return true;
    }
    return false;
  }
}
