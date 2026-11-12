part of '../network.dart';

/// Prediction-error steps. Ports of rshgf
/// `src/updates/nodalised/prediction_error/*`.
extension on Network {
  /// δ = μ - μ̂ and Δ = π̂/π + π̂·δ² - 1. A node with children is observed
  /// when at least one of its children was. An input node with noise
  /// parents sends them the noise prediction error of
  /// [_noisePredictionErrorContinuous] instead of Δ.
  @pragma('vm:unsafe:no-bounds-checks')
  void _predictionErrorContinuous(int i) {
    final mean = _mean[i];
    final expectedMean = _expectedMean[i];
    final expectedPrecision = _expectedPrecision[i];
    final delta = mean - expectedMean;
    _valuePredictionError[i] = delta;
    if (_isLeaf[i] == 0) {
      _volatilityPredictionError[i] =
          (expectedPrecision / _precision[i]) +
          expectedPrecision * delta * delta -
          1.0;
      _observed[i] = _hasObservations(i) ? 1.0 : 0.0;
    } else if (!_isNoiseChild(i)) {
      _volatilityPredictionError[i] =
          (expectedPrecision / _precision[i]) +
          expectedPrecision * delta * delta -
          1.0;
    }
  }

  /// Whether [i] is an input node with noise parents: a continuous leaf with
  /// volatility parents (pyhgf's `is_noise_child`).
  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  @pragma('vm:unsafe:no-bounds-checks')
  bool _isNoiseChild(int i) => _noiseChild[i] == 1;

  /// The noise prediction error of input node [i] (Eq. 72 of Weber et al.,
  /// 2026; pyhgf #477), into its volatility prediction error:
  ///
  /// ε = π̂_u (Σ_b (t α_b g_b'(μ_b))² / π_b + (u − m_u)²) − 1,
  ///
  /// with m_u the predicted mean of u with every value parent b moved from
  /// μ̂_b to its posterior μ_b. With one linear value parent this is
  /// π̂_u/π_b + π̂_u (u − μ_b)² − 1, the expected squared observation noise
  /// under the parent's posterior. Runs after the value parents' updates.
  /// With [NetworkFixes.constantNoiseVariance], bias parents add no
  /// variance.
  @pragma('vm:unsafe:no-bounds-checks')
  void _noisePredictionErrorContinuous(int i, double t) {
    var posteriorMean = _expectedMean[i];
    var posteriorVariance = 0.0;
    final fixConstant = fixes.constantNoiseVariance;
    for (var e = _vpStart[i], end = _vpStart[i + 1]; e < end; e++) {
      final p = _vpIdx[e];
      final alpha = _vpK[e];
      final cf = _couplingFn[p];
      double shift, gPrime;
      if (cf == 0) {
        shift = _mean[p] - _expectedMean[p];
        gPrime = 1.0;
      } else {
        final fn = _couplingOf(cf, p);
        shift = fn.f(_mean[p]) - fn.f(_expectedMean[p]);
        gPrime = fn.df(_mean[p]);
      }
      posteriorMean += t * alpha * shift;
      // A bias parent is known exactly (P21; pyhgf reads its placeholder
      // precision 1 here).
      if (fixConstant && _kind[p] == _kindConstant) continue;
      final coeff = t * alpha * gPrime;
      posteriorVariance += coeff * coeff / _precision[p];
    }
    final d = _mean[i] - posteriorMean;
    _volatilityPredictionError[i] =
        _expectedPrecision[i] * (posteriorVariance + d * d) - 1.0;
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
