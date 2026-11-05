part of '../network.dart';

/// Categorical state node steps. Ports of pyhgf's JAX
/// `updates/prediction_error/categorical.py` and
/// `updates/posterior/categorical.py`.
///
/// A categorical node's value parents are its binary nodes, one per
/// category in order, each with a continuous value parent (validated in
/// [Network._compileVectors]). Like the ef-state steps, these sit outside
/// the scalar hot loop and keep their bounds checks.
extension on Network {
  /// Pass each category's indicator, and the node's `observed` flag, to that
  /// category's binary node (whose own prediction error step runs next).
  void _predictionErrorCategorical(int c) {
    final x = _vMean, observed = _observed[c];
    var m = _vMeanStart[c];
    for (var e = _vpStart[c], end = _vpStart[c + 1]; e < end; e++) {
      final b = _vpIdx[e];
      _mean[b] = x[m++];
      _observed[b] = observed;
    }
  }

  /// After the binary HGFs' updates: for each category, the prediction error
  /// `δ = x − μ̂_b` of the binary node and the change in its expectation
  /// `Δ = σ(μ_p) − μ̂_b` (`μ_p` the binary node's value parent's new mean)
  /// imply a precision `ν = δ/Δ − 1`, so `α = ν μ̂_b + 1`. Records the KL
  /// divergence from the previous α, the summed binary surprise, and
  /// replaces the node's mean with the updated probabilities `σ(μ_p)`.
  ///
  /// As in pyhgf, nothing checks `observed`: on a missing observation
  /// `Δ = 0` and α becomes ±∞ (KL NaN for that trial and the next).
  void _posteriorCategorical(int c) {
    final x = _vMean, alpha = _alpha, next = _alphaScratch;
    final m0 = _vMeanStart[c], a0 = _alphaStart[c];
    final vp0 = _vpStart[c], n = _vpStart[c + 1] - vp0;
    var surprise = 0.0;
    for (var i = 0; i < n; i++) {
      final b = _vpIdx[vp0 + i];
      final expected = _expectedMean[b];
      final updated = sigmoid(_mean[_vpIdx[_vpStart[b]]]);
      final observation = x[m0 + i];
      final nu = (observation - expected) / (updated - expected) - 1.0;
      next[i] = nu * expected + 1.0;
      surprise += binarySurprise(observation, expected);
      x[m0 + i] = updated;
    }
    _klDivergence[c] = dirichletKlAt(alpha, a0, next, 0, n);
    _surprise[c] = surprise;
    alpha.setRange(a0, a0 + n, next);
    _mean[c] = x[m0];
  }
}
