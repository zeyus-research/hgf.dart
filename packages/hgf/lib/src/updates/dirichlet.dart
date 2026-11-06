part of '../network.dart';

/// Dirichlet process node steps. Ports of pyhgf's JAX
/// `updates/prediction/dirichlet.py` and
/// `updates/prediction_error/dirichlet.py`.
///
/// A dp-state node's value parents are its clusters: univariate ef-state
/// nodes with a fixed learning rate, one per cluster, in order (validated in
/// [Network._compileVectors]). Like the other vector-kind steps, these sit
/// outside the scalar hot loop and keep their bounds checks.
extension on Network {
  /// Each cluster's expected mean and standard deviation, from its ef-state
  /// parent's sufficient statistics `ξ = [E x, E x²]`.
  void _predictionDp(int c) {
    final em = _dpExpectedMeans, es = _dpExpectedSigmas, xis = _xis;
    var k = _dpStart[c];
    for (var e = _vpStart[c], end = _vpStart[c + 1]; e < end; e++, k++) {
      final s = _ssStart[_vpIdx[e]];
      final m = xis[s];
      em[k] = m;
      es[k] = math.sqrt(xis[s + 1] - m * m);
    }
  }

  /// Assign the observation to a cluster: each open cluster `i` explains it
  /// with probability `n_i / (α + N) · N(x; μ_i, σ_i)`, a new one with
  /// `α / (α + N) · N(x; μ*, σ*)` for the candidate `(μ*, σ*)` of
  /// [dpCandidate]. The chosen cluster's ef-state node gets `observed = 1`
  /// (a new one is initialised at the candidate instead, and learns from the
  /// next observation on); every cluster node gets the value as its mean.
  ///
  /// When every cluster is open, opening another one does nothing but count
  /// in [NodeField.nActiveCluster] (pyhgf's out-of-bounds writes are
  /// dropped).
  ///
  /// **Diverges from pyhgf on missing observations:** pyhgf never reads the
  /// node's `observed` flag, so a missing observation is clustered like any
  /// other (a NaN one opens a NaN cluster;
  /// [P14](../../../../docs/upstream/README.md#p14)). Here it is skipped:
  /// no cluster observes it and the counts don't change.
  void _predictionErrorDp(int c) {
    final vp0 = _vpStart[c], k = _vpStart[c + 1] - vp0, off = _dpStart[c];
    for (var i = 0; i < k; i++) {
      _observed[_vpIdx[vp0 + i]] = 0.0;
    }
    if (_observed[c] == 0.0) return;

    final value = _mean[c], alpha = _dpAlpha[c], nTotal = _nTotal[c];
    final em = _dpExpectedMeans, es = _dpExpectedSigmas, n = _dpN;
    final act = _dpActivated;

    // The most likely open cluster (jnp.max / jnp.argmax: NaN wins, ties go
    // to the first).
    var best = double.negativeInfinity;
    var bestIdx = 0;
    for (var i = 0; i < k; i++) {
      var ll = jaxNormPdf(value, em[off + i], es[off + i]) * act[off + i];
      ll *= n[off + i] / (alpha + nTotal);
      if (best.isNaN) continue;
      if (ll.isNaN || ll > best) {
        best = ll;
        bestIdx = i;
      }
    }

    final (cMean, cSigma) = dpCandidate(
      value,
      _sensoryPrecision[c],
      em,
      es,
      off,
      k,
      _dpScratch!,
    );
    final candidate =
        jaxNormPdf(value, cMean, cSigma) * (alpha / (alpha + nTotal));

    for (var i = 0; i < k; i++) {
      final p = _vpIdx[vp0 + i];
      _mean[p] = value;
      _vMean[_vMeanStart[p]] = value;
    }
    if (best >= candidate) {
      _observed[_vpIdx[vp0 + bestIdx]] = 1.0;
      n[off + bestIdx] += 1.0;
    } else {
      final idx = _nActiveCluster[c].toInt();
      if (idx < k) {
        act[off + idx] = 1.0;
        final s = _ssStart[_vpIdx[vp0 + idx]];
        _xis[s] = cMean;
        _xis[s + 1] = cMean * cMean + cSigma * cSigma;
        n[off + idx] = 1.0;
      }
      _nActiveCluster[c] += 1.0;
    }
    _nTotal[c] = nTotal + 1.0;
  }
}
