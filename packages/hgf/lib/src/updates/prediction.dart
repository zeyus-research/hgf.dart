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
  ///
  /// An input node (a leaf) takes the precision of its observation noise
  /// instead ([_storeInputPrediction]).
  @pragma('vm:unsafe:no-bounds-checks')
  void _predictionContinuous(int i, double t) {
    final mean = _mean[i];
    final lambda = _autoconnectionStrength[i];
    final precision = _precision[i];
    final isInput = _isLeaf[i] == 1;

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
    // Carried variance: scaling the belief by λ scales its variance by λ².
    final carriedVariance = (lambda * lambda) / precision;
    _currentVariance[i] = carriedVariance;
    _expectedMean[i] = expectedMean;
    if (isInput) {
      _storeInputPrediction(i);
      return;
    }

    var totalVolatility = _tonicVolatility[i];
    for (var e = _volpStart[i], end = _volpStart[i + 1]; e < end; e++) {
      final p = _volpIdx[e];
      final kappa = _volpK[e];
      totalVolatility += kappa * _expectedMean[p];
      totalVolatility += (kappa * kappa) / (2.0 * _expectedPrecision[p]);
    }

    final pvRaw = t * fastExp(totalVolatility);
    final predictedVolatility = pvRaw > 1e-128 ? pvRaw : double.nan;
    final conditional =
        1.0 / floorPredictedVariance(carriedVariance + predictedVolatility);
    final expected =
        1.0 /
        floorPredictedVariance(
          carriedVariance + predictedVolatility + valueCouplingVariance,
        );

    _effectivePrecision[i] = predictedVolatility * expected;
    _expectedPrecision[i] = expected;
    _conditionalExpectedPrecision[i] = conditional;
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
    final carriedVariance = (lambda * lambda) / precision;
    _currentVariance[i] = carriedVariance;
    _expectedMean[i] = expectedMean;
    if (_isLeaf[i] == 1) {
      _storeInputPrediction(i);
      return;
    }

    var totalVolatility = _tonicVolatility[i];
    for (var e = _volpStart[i], end = _volpStart[i + 1]; e < end; e++) {
      totalVolatility += _volpK[e] * _expectedMean[_volpIdx[e]];
    }

    final pvRaw = t * fastExp(totalVolatility);
    final predictedVolatility = pvRaw > 1e-128 ? pvRaw : double.nan;
    final expected =
        1.0 / floorPredictedVariance(carriedVariance + predictedVolatility);

    _effectivePrecision[i] = predictedVolatility * expected;
    _expectedPrecision[i] = expected;
    _conditionalExpectedPrecision[i] = expected;
  }

  /// The predicted precisions of input node [i], which is observed exactly
  /// and does not drift between observations (pyhgf #477, Weber et al.,
  /// 2026, appendix on noise parents): without volatility parent, both are
  /// its `precision`, the precision of the observation noise, and γ = 0.
  /// Its volatility parents are noise parents, which set that precision,
  /// π̂_u = exp(−(ε_u + Σ_q κ_q μ̂_q)) (Eq. 66, ε_u its tonic volatility),
  /// and γ = 1 (Eq. 73). The same under the relaxed and the mean-field
  /// predictions.
  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  @pragma('vm:unsafe:no-bounds-checks')
  void _storeInputPrediction(int i) {
    final volStart = _volpStart[i], volEnd = _volpStart[i + 1];
    if (volStart == volEnd) {
      final precision = _precision[i];
      _expectedPrecision[i] = precision;
      _conditionalExpectedPrecision[i] = precision;
      _effectivePrecision[i] = 0.0;
      return;
    }
    var logNoiseVariance = _tonicVolatility[i];
    for (var e = volStart; e < volEnd; e++) {
      logNoiseVariance += _volpK[e] * _expectedMean[_volpIdx[e]];
    }
    final noisePrecision = fastExp(-logNoiseVariance);
    _expectedPrecision[i] = noisePrecision;
    _conditionalExpectedPrecision[i] = noisePrecision;
    _effectivePrecision[i] = 1.0;
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
