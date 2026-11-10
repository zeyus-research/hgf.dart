part of '../network.dart';

/// Batched ef-state and categorical-state steps: [Network]'s
/// `updates/exponential.dart` and `updates/categorical.dart`, element by
/// element over `[lo, hi)`, with the same arithmetic in the same order (so
/// each element matches [Network] bit for bit).
///
/// The vector side store is `[offset][element]`: entry `j` of node `i`'s
/// vector for element `e` is at `(start[i] + j) * batchSize + e`. These
/// steps are few per network and sit outside the scalar dispatch.
extension on BatchNetwork {
  @pragma('vm:never-inline')
  @pragma('wasm:never-inline')
  void _bDispatchVector(int step, int node, int lo, int hi) {
    switch (step) {
      case _kPredictionErrorEfFixed:
        _bEfSufficientStatistics(node, lo, hi);
        _bEfMoveXis(node, lo, hi);
      case _kPredictionErrorEfDynamic:
        _bPredictionErrorEfDynamic(node, lo, hi);
      case _kPosteriorEfDynamic:
        _bPosteriorEfDynamic(node, lo, hi);
      case _kPredictionErrorCategorical:
        _bPredictionErrorCategorical(node, lo, hi);
      case _kPosteriorCategorical:
        _bPosteriorCategorical(node, lo, hi);
    }
  }

  /// The tangents of the vector steps. They carry none of their own (the
  /// vector fields, ν, the KL divergence and the surprise are not tangent
  /// fields); what they write into tangent fields is either data (the
  /// observations they pass on, tangent zero) or a categorical node's
  /// mean, `σ(μ_p)`.
  void _tDispatchVector(int step, int node, int lo, int hi) {
    _bDispatchVector(step, node, lo, hi);
    switch (step) {
      case _kPredictionErrorEfDynamic || _kPredictionErrorCategorical:
        // The parents' (binary nodes') new means are observations.
        for (var e = _vpStart[node], end = _vpStart[node + 1]; e < end; e++) {
          _tZeroMean(_vpIdx[e], lo, hi);
        }
      case _kPosteriorCategorical:
        // μ_c = σ(μ_p) of the first category's binary node's parent p.
        final b = _b, k = _k, mean = _mean, dMean = _dMean;
        final p = _vpIdx[_vpStart[_vpIdx[_vpStart[node]]]];
        for (var ad = 0; ad < _nActive; ad++) {
          final d = _activeDirs[ad];
          final tc = (node * k + d) * b, tp = (p * k + d) * b;
          for (var e = lo; e < hi; e++) {
            final s = mean[node * b + e];
            dMean[tc + e] = s * (1.0 - s) * dMean[tp + e];
          }
        }
    }
  }

  void _tZeroMean(int node, int lo, int hi) {
    final b = _b, k = _k;
    for (var ad = 0; ad < _nActive; ad++) {
      final t0 = (node * k + _activeDirs[ad]) * b;
      _dMean.fillRange(t0 + lo, t0 + hi, 0.0);
    }
  }

  /// t(x) of [i]'s observation into its `observation_ss`: `x`, then the
  /// lower triangle of `x xᵀ` row by row.
  void _bEfSufficientStatistics(int i, int lo, int hi) {
    final b = _b, x = _vMean, ss = _ss;
    final m = _vMeanStart[i], d = _vMeanStart[i + 1] - m;
    final s0 = _ssStart[i];
    for (var e = lo; e < hi; e++) {
      var k = s0;
      for (var j = 0; j < d; j++) {
        ss[k++ * b + e] = x[(m + j) * b + e];
      }
      for (var r = 0; r < d; r++) {
        final xr = x[(m + r) * b + e];
        for (var c = 0; c <= r; c++) {
          ss[k++ * b + e] = xr * x[(m + c) * b + e];
        }
      }
    }
  }

  /// ξ ← ξ + (t(x) − ξ) / (1 + ν), skipped where [i] is unobserved.
  void _bEfMoveXis(int i, int lo, int hi) {
    final b = _b, xis = _xis, ss = _ss;
    final start = _ssStart[i], end = _ssStart[i + 1];
    for (var e = lo; e < hi; e++) {
      if (_observed[i * b + e] == 0.0) continue;
      final w = 1.0 / (1.0 + _nus[i * b + e]);
      for (var k = start; k < end; k++) {
        final q = k * b + e;
        xis[q] = xis[q] + w * (ss[q] - xis[q]);
      }
    }
  }

  void _bPredictionErrorEfDynamic(int i, int lo, int hi) {
    _bEfSufficientStatistics(i, lo, hi);
    final b = _b, mean = _mean, obs = _observed, ss = _ss;
    var k = _ssStart[i];
    for (var q = _vpStart[i], end = _vpStart[i + 1]; q < end; q++, k++) {
      final p = _vpIdx[q];
      for (var e = lo; e < hi; e++) {
        obs[p * b + e] = obs[i * b + e];
        mean[p * b + e] = ss[k * b + e];
      }
    }
  }

  void _bPosteriorEfDynamic(int i, int lo, int hi) {
    final b = _b, mean = _mean, expected = _expectedMean;
    final start = _vpStart[i], end = _vpStart[i + 1];
    for (var e = lo; e < hi; e++) {
      var sum = 0.0;
      for (var q = start; q < end; q++) {
        final p = _vpIdx[q];
        final pe = mean[p * b + e] - expected[p * b + e];
        final pp = _vpIdx[_vpStart[p]];
        final ed = mean[pp * b + e] - expected[pp * b + e];
        sum += pe / ed;
      }
      _nus[i * b + e] = sum / (end - start);
    }
    _bEfMoveXis(i, lo, hi);
  }

  void _bPredictionErrorCategorical(int c, int lo, int hi) {
    final b = _b, x = _vMean, mean = _mean, obs = _observed;
    var m = _vMeanStart[c];
    for (var q = _vpStart[c], end = _vpStart[c + 1]; q < end; q++, m++) {
      final bn = _vpIdx[q];
      for (var e = lo; e < hi; e++) {
        mean[bn * b + e] = x[m * b + e];
        obs[bn * b + e] = obs[c * b + e];
      }
    }
  }

  void _bPosteriorCategorical(int c, int lo, int hi) {
    final b = _b, x = _vMean, alpha = _alpha;
    final cur = _alphaCurrent, next = _alphaNext;
    final m0 = _vMeanStart[c], a0 = _alphaStart[c];
    final vp0 = _vpStart[c], n = _vpStart[c + 1] - vp0;
    for (var e = lo; e < hi; e++) {
      var surprise = 0.0;
      for (var i = 0; i < n; i++) {
        final bn = _vpIdx[vp0 + i];
        final expected = _expectedMean[bn * b + e];
        final updated = sigmoid(_mean[_vpIdx[_vpStart[bn]] * b + e]);
        final observation = x[(m0 + i) * b + e];
        final nu = (observation - expected) / (updated - expected) - 1.0;
        next[i] = nu * expected + 1.0;
        surprise += binarySurprise(observation, expected);
        x[(m0 + i) * b + e] = updated;
        cur[i] = alpha[(a0 + i) * b + e];
      }
      _klDivergence[c * b + e] = dirichletKlAt(cur, 0, next, 0, n);
      _surprise[c * b + e] = surprise;
      for (var i = 0; i < n; i++) {
        alpha[(a0 + i) * b + e] = next[i];
      }
      _mean[c * b + e] = x[m0 * b + e];
    }
  }
}
