part of '../network.dart';

/// Prediction steps. Ports of rshgf `src/updates/nodalised/prediction/*`
/// (which mirror pyhgf `updates/prediction/*`).
extension on Network {
  /// Relaxed (rHGF) prediction of a continuous node.
  ///
  /// * μ̂ = λ·μ + Δt·(ρ + Σ_b α_b g(μ̂_b))
  /// * π̄ = 1 / (λ²/π + Ω)            (conditional; no parent uncertainty)
  /// * π̂ = 1 / (1/π̄ + Σ_b (Δt α_b g'(μ̂_b))² / π̂_b)   (marginal)
  /// * Ω = Δt·exp(ω + Σ_j κ_j μ̂_j + Σ_j κ_j² / (2 π̂_j))
  @pragma('vm:unsafe:no-bounds-checks')
  void _predictionContinuous(int i, double t) {
    final mean = _mean[i];
    final lambda = _autoconnectionStrength[i];
    final precision = _precision[i];

    var driftrate = _tonicDrift[i];
    var valueCouplingVariance = 0.0;
    for (var e = _vpStart[i], end = _vpStart[i + 1]; e < end; e++) {
      final p = _vpIdx[e];
      final pMean = _expectedMean[p];
      final alpha = _vpK[e];
      final cf = _couplingFn[p];
      double value, gPrime;
      if (cf == 0) {
        value = pMean;
        gPrime = 1.0;
      } else {
        final fn = _couplingOf(cf, p);
        value = fn.f(pMean);
        gPrime = fn.df(pMean);
      }
      driftrate += alpha * value;
      final coeff = t * alpha * gPrime;
      valueCouplingVariance += coeff * coeff / _expectedPrecision[p];
    }
    final expectedMean = lambda * mean + t * driftrate;

    var totalVolatility = _tonicVolatility[i];
    final volStart = _volpStart[i], volEnd = _volpStart[i + 1];
    for (var e = volStart; e < volEnd; e++) {
      final p = _volpIdx[e];
      final kappa = _volpK[e];
      totalVolatility += kappa * _expectedMean[p];
      totalVolatility += (kappa * kappa) / (2.0 * _expectedPrecision[p]);
    }

    final pvRaw = t * fastExp(totalVolatility);
    final predictedVolatility = pvRaw > 1e-128 ? pvRaw : double.nan;
    final isInput = _isLeaf[i] == 1;
    final carriedVariance = isInput
        ? 1.0 / precision
        : (lambda * lambda) / precision;
    final conditional =
        1.0 / floorPredictedVariance(carriedVariance + predictedVolatility);
    final expected =
        1.0 /
        floorPredictedVariance(
          carriedVariance + predictedVolatility + valueCouplingVariance,
        );

    _currentVariance[i] = carriedVariance;
    _expectedMean[i] = expectedMean;
    _effectivePrecision[i] = predictedVolatility * expected;
    if (!(isInput && volStart == volEnd)) {
      _expectedPrecision[i] = expected;
      _conditionalExpectedPrecision[i] = conditional;
    } else {
      // Leaf without random walk: conditional == marginal == prior precision.
      _expectedPrecision[i] = precision;
      _conditionalExpectedPrecision[i] = precision;
    }
  }

  /// Mean-field prediction of a continuous node (no Taylor / MGF terms).
  @pragma('vm:unsafe:no-bounds-checks')
  void _predictionContinuousMeanField(int i, double t) {
    final mean = _mean[i];
    final lambda = _autoconnectionStrength[i];
    final precision = _precision[i];

    var driftrate = _tonicDrift[i];
    for (var e = _vpStart[i], end = _vpStart[i + 1]; e < end; e++) {
      final p = _vpIdx[e];
      final pMean = _expectedMean[p];
      final cf = _couplingFn[p];
      driftrate += _vpK[e] * (cf == 0 ? pMean : _couplingOf(cf, p).f(pMean));
    }
    final expectedMean = lambda * mean + t * driftrate;

    var totalVolatility = _tonicVolatility[i];
    final volStart = _volpStart[i], volEnd = _volpStart[i + 1];
    for (var e = volStart; e < volEnd; e++) {
      totalVolatility += _volpK[e] * _expectedMean[_volpIdx[e]];
    }

    final pvRaw = t * fastExp(totalVolatility);
    final predictedVolatility = pvRaw > 1e-128 ? pvRaw : double.nan;
    final isInput = _isLeaf[i] == 1;
    final carriedVariance = isInput
        ? 1.0 / precision
        : (lambda * lambda) / precision;
    final expected =
        1.0 / floorPredictedVariance(carriedVariance + predictedVolatility);

    _currentVariance[i] = carriedVariance;
    _expectedMean[i] = expectedMean;
    _effectivePrecision[i] = predictedVolatility * expected;
    if (!(isInput && volStart == volEnd)) {
      _expectedPrecision[i] = expected;
      _conditionalExpectedPrecision[i] = expected;
    } else {
      _expectedPrecision[i] = precision;
      _conditionalExpectedPrecision[i] = precision;
    }
  }

  /// Prediction of a binary node: μ̂ = s(Σ_b μ̂_b), clipped to [v, 1 - v];
  /// π̂ = μ̂(1 - μ̂).
  @pragma('vm:unsafe:no-bounds-checks')
  void _predictionBinary(int i) {
    var x = 0.0;
    for (var e = _vpStart[i], end = _vpStart[i + 1]; e < end; e++) {
      x += _expectedMean[_vpIdx[e]];
    }
    var m = sigmoid(x);
    final v = precisionClippingValue;
    if (m < v) {
      m = v;
    } else if (m > 1.0 - v) {
      m = 1.0 - v;
    }
    _expectedMean[i] = m;
    _expectedPrecision[i] = m * (1.0 - m);
  }
}
