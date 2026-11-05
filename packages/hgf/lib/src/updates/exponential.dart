part of '../network.dart';

/// Exponential-family state node steps. Ports of pyhgf's JAX
/// `updates/prediction_error/exponential.py` and
/// `updates/posterior/exponential.py` (Mathys & Weber, 2020).
///
/// These nodes are few per network and sit outside the scalar hot loop, so
/// they keep their bounds checks.
extension on Network {
  /// t(x) of [i]'s observation into its `observation_ss`: `x`, then the
  /// lower triangle of `x xᵀ` row by row (`[x, x²]` in one dimension).
  void _efSufficientStatistics(int i) {
    final x = _vMean, ss = _ss;
    final m = _vMeanStart[i], d = _vMeanStart[i + 1] - m;
    var k = _ssStart[i];
    for (var j = 0; j < d; j++) {
      ss[k++] = x[m + j];
    }
    for (var r = 0; r < d; r++) {
      final xr = x[m + r];
      for (var c = 0; c <= r; c++) {
        ss[k++] = xr * x[m + c];
      }
    }
  }

  /// ξ ← ξ + (t(x) − ξ) / (1 + ν), skipped when [i] is unobserved.
  void _efMoveXis(int i) {
    if (_observed[i] == 0.0) return;
    final xis = _xis, ss = _ss;
    final w = 1.0 / (1.0 + _nus[i]);
    for (var k = _ssStart[i], end = _ssStart[i + 1]; k < end; k++) {
      xis[k] = xis[k] + w * (ss[k] - xis[k]);
    }
  }

  /// Fixed learning rate (no parents).
  void _predictionErrorEfFixed(int i) {
    _efSufficientStatistics(i);
    _efMoveXis(i);
  }

  /// HGF learning: each sufficient statistic becomes the observation of the
  /// value parent tracking it (validated in [_compileVectors]: one parent
  /// per statistic, in order).
  void _predictionErrorEfDynamic(int i) {
    _efSufficientStatistics(i);
    final observed = _observed[i];
    var k = _ssStart[i];
    for (var e = _vpStart[i], end = _vpStart[i + 1]; e < end; e++) {
      final p = _vpIdx[e];
      _observed[p] = observed;
      _mean[p] = _ss[k++];
    }
  }

  /// HGF learning, after the parents' updates: ν is the mean over the
  /// sufficient statistics of δ/Δ, the parent's prediction error over its
  /// own value parent's change in expectation; then ξ moves as with a fixed
  /// rate.
  void _posteriorEfDynamic(int i) {
    var sum = 0.0;
    final start = _vpStart[i], end = _vpStart[i + 1];
    for (var e = start; e < end; e++) {
      final p = _vpIdx[e];
      final pe = _mean[p] - _expectedMean[p];
      final pp = _vpIdx[_vpStart[p]];
      final ed = _mean[pp] - _expectedMean[pp];
      sum += pe / ed;
    }
    _nus[i] = sum / (end - start);
    _efMoveXis(i);
  }
}
