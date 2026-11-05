part of '../network.dart';

/// Posterior updates of continuous nodes. Port of rshgf
/// `src/updates/nodalised/posterior/continuous.rs` (mirroring pyhgf
/// `updates/posterior/continuous/*`).
extension on Network {
  // -------------------------------------------------------------------------
  // Value-coupling building blocks
  // -------------------------------------------------------------------------

  /// Relaxed gain g_a of value child [c] (see [_relaxedEffective] for π_eff).
  ///
  /// π_eff = π̄_a·π_y / (π̄_a + π_y),  g_a = π̄_a·π_a / (π̄_a + π_y),
  /// π_y = π_a - π̂_a. Gaussian leaves take the fully observed limit
  /// (π_eff = g_a = π̄_a); non-Gaussian children, and every child under
  /// mean-field, use π̂_a for both.
  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  @pragma('vm:unsafe:no-bounds-checks')
  double _relaxedGain(int c, bool meanField) {
    final cep = _expectedPrecision[c];
    if (meanField || _kind[c] != _kindContinuous) return cep;
    final cond = _conditionalExpectedPrecision[c];
    if (_isLeaf[c] == 1) return cond;
    final cp = _precision[c];
    return cond * cp / (cond + (cp - cep));
  }

  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  @pragma('vm:unsafe:no-bounds-checks')
  double _relaxedEffective(int c, bool meanField) {
    final cep = _expectedPrecision[c];
    if (meanField || _kind[c] != _kindContinuous) return cep;
    final cond = _conditionalExpectedPrecision[c];
    if (_isLeaf[c] == 1) return cond;
    final piY = _precision[c] - cep;
    return cond * piY / (cond + piY);
  }

  /// Precision increment from the value children of [i].
  @pragma('vm:unsafe:no-bounds-checks')
  double _valueCouplingPrecisionWpe(int i, bool meanField) {
    var wpe = 0.0;
    final start = _vcStart[i], end = _vcStart[i + 1];
    if (start == end) return wpe;
    final cf = _couplingFn[i];
    final parentMean = _mean[i];
    for (var e = start; e < end; e++) {
      final c = _vcIdx[e];
      final alpha = _vcK[e];
      double gPrimeSq, secondTerm;
      if (cf == 0) {
        gPrimeSq = 1.0;
        secondTerm = 0.0;
      } else {
        final fn = _couplingOf(cf, i);
        final gp = fn.df(parentMean);
        gPrimeSq = gp * gp;
        secondTerm = alpha * fn.d2f(parentMean) * _valuePredictionError[c];
      }
      wpe +=
          (_relaxedEffective(c, meanField) * alpha * alpha * gPrimeSq -
              _relaxedGain(c, meanField) * secondTerm) *
          _observed[c];
    }
    return wpe;
  }

  /// Mean increment from the value children of [i], given the node's
  /// posterior precision.
  @pragma('vm:unsafe:no-bounds-checks')
  double _valueCouplingMeanWpe(int i, double nodePrecision, bool meanField) {
    var wpe = 0.0;
    final start = _vcStart[i], end = _vcStart[i + 1];
    if (start == end) return wpe;
    final cf = _couplingFn[i];
    final gPrime = cf == 0 ? 1.0 : _couplingOf(cf, i).df(_mean[i]);
    for (var e = start; e < end; e++) {
      final c = _vcIdx[e];
      final childVape = _valuePredictionError[c] * _observed[c];
      wpe +=
          (_vcK[e] * gPrime * _relaxedGain(c, meanField) / nodePrecision) *
          childVape;
    }
    return wpe;
  }

  /// Standard volatility-coupling precision increment from the volatility
  /// children of [i], accumulated onto [wpe] (same summation order as rshgf).
  @pragma('vm:unsafe:no-bounds-checks')
  double _volatilityPrecisionWpe(int i, double wpe) {
    for (var e = _volcStart[i], end = _volcStart[i + 1]; e < end; e++) {
      final c = _volcIdx[e];
      final kappa = _volcK[e];
      final eff = _effectivePrecision[c];
      final vope = _volatilityPredictionError[c];
      final ke = kappa * eff;
      wpe +=
          (0.5 * ke * ke + ke * ke * vope - 0.5 * kappa * kappa * eff * vope) *
          _observed[c];
    }
    return wpe;
  }

  /// Volatility-coupling mean increment from the volatility children of [i].
  @pragma('vm:unsafe:no-bounds-checks')
  double _volatilityMeanWpe(int i, double nodePrecision) {
    var wpe = 0.0;
    for (var e = _volcStart[i], end = _volcStart[i + 1]; e < end; e++) {
      final c = _volcIdx[e];
      wpe +=
          (_volcK[e] * _effectivePrecision[c] * _volatilityPredictionError[c]) /
          (2.0 * nodePrecision) *
          _observed[c];
    }
    return wpe;
  }

  /// Posterior precision: the predicted precision plus [wpe] if any child
  /// was observed, otherwise the predicted precision; clipped to
  /// [Network.maxPosteriorPrecision].
  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  @pragma('vm:unsafe:no-bounds-checks')
  double _observedOrPredicted(int i, bool hasObs, double wpe) {
    final ep = _expectedPrecision[i];
    final p = hasObs ? _maxNan(ep + wpe, minPosteriorPrecision) : ep;
    return _minNan(p, maxPosteriorPrecision);
  }

  /// ω̃ = ω_c + Σ_{p≠i} κ_p μ̂_p: the effective tonic volatility of volatility
  /// child [c] seen from its volatility parent [i].
  @pragma('vm:unsafe:no-bounds-checks')
  double _effectiveTonicVolatility(int i, int c) {
    var tv = _tonicVolatility[c];
    for (var e = _volpStart[c], end = _volpStart[c + 1]; e < end; e++) {
      final p = _volpIdx[e];
      if (p != i) tv += _volpK[e] * _expectedMean[p];
    }
    return tv;
  }

  /// Σ_p κ_p² / (2 π̂_p) over the volatility parents of [c]: the MGF terms of
  /// the relaxed prediction's volatility (see [Network.ehgfMgfCorrection]).
  /// Read during the posterior sweep, when every `π̂_p` is still the
  /// prediction.
  @pragma('vm:unsafe:no-bounds-checks')
  double _mgfTerms(int c) {
    var sum = 0.0;
    for (var e = _volpStart[c], end = _volpStart[c + 1]; e < end; e++) {
      final kappa = _volpK[e];
      sum += kappa * kappa / (2.0 * _expectedPrecision[_volpIdx[e]]);
    }
    return sum;
  }

  /// T = 1/π̂ - 1/π̄ (floored at 0): the variance a node's value parents add
  /// to its prediction.
  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  @pragma('vm:unsafe:no-bounds-checks')
  double _valueCouplingVariance(int c) {
    final v =
        1.0 / _expectedPrecision[c] - 1.0 / _conditionalExpectedPrecision[c];
    return _maxNan(v, 0.0);
  }

  // -------------------------------------------------------------------------
  // Standard update
  // -------------------------------------------------------------------------

  @pragma('vm:unsafe:no-bounds-checks')
  void _posteriorContinuous(int i, {required bool meanField}) {
    final hasObs = _hasObservations(i);
    final wpe = hasObs
        ? _volatilityPrecisionWpe(i, _valueCouplingPrecisionWpe(i, meanField))
        : 0.0;
    final precision = _observedOrPredicted(i, hasObs, wpe);
    final meanWpe =
        _valueCouplingMeanWpe(i, precision, meanField) +
        _volatilityMeanWpe(i, precision);
    _precision[i] = precision;
    _mean[i] = _expectedMean[i] + meanWpe;
  }

  // -------------------------------------------------------------------------
  // eHGF update: mean first (at the predicted precision), then a precision
  // update whose volatility increment is re-predicted from the new mean and
  // floored at zero.
  // -------------------------------------------------------------------------

  @pragma('vm:unsafe:no-bounds-checks')
  void _posteriorContinuousEhgf(int i, double t, {required bool meanField}) {
    final ep = _expectedPrecision[i];
    final mean =
        _expectedMean[i] +
        _valueCouplingMeanWpe(i, ep, meanField) +
        _volatilityMeanWpe(i, ep);
    _mean[i] = mean;

    final hasObs = _hasObservations(i);
    var wpe = 0.0;
    if (hasObs) {
      wpe = _valueCouplingPrecisionWpe(i, meanField);
      for (var e = _volcStart[i], end = _volcStart[i + 1]; e < end; e++) {
        final c = _volcIdx[e];
        var tonic = _effectiveTonicVolatility(i, c);
        if (!meanField && ehgfMgfCorrection) tonic += _mgfTerms(c);
        wpe += _ehgfVolatilityIncrement(c, _volcK[e], mean, tonic, t);
      }
    }
    _precision[i] = _observedOrPredicted(i, hasObs, wpe);
  }

  @pragma('vm:unsafe:no-bounds-checks')
  double _ehgfVolatilityIncrement(
    int c,
    double kappa,
    double mean,
    double tonicVolatility,
    double t,
  ) {
    final previousVariance = _currentVariance[c] + _valueCouplingVariance(c);
    final predictedVolatility = t * fastExp(kappa * mean + tonicVolatility);
    final expectedPrecision = 1.0 / (previousVariance + predictedVolatility);
    final eff = predictedVolatility * expectedPrecision;
    final weight = (predictedVolatility - previousVariance) * expectedPrecision;
    final d = _mean[c] - _expectedMean[c];
    final vope = (1.0 / _precision[c] + d * d) * expectedPrecision - 1.0;
    final inc = 0.5 * kappa * kappa * eff * (eff + weight * vope);
    return _maxNan(inc, 0.0) * _observed[c];
  }

  // -------------------------------------------------------------------------
  // Unbounded update (Mathys et al., 2026, section 3.6)
  // -------------------------------------------------------------------------

  /// Moment-matched mixture of quadratic expansions of the variational
  /// energy: one at the prediction (concave part only) and one at the
  /// Lambert-W mode of each observed volatility child, weighted by a softmax
  /// of the energy at their means.
  ///
  /// pyhgf writes the child terms in log space: with logS = log t + κx + ω̃
  /// and logP = log of the child's predicted variance P,
  /// `sigmoid(logS − logP)`, `exp(−logAddExp(logP, logS))` and
  /// `logAddExp(logP, logS)`. Here they are evaluated in linear space as
  /// S / (S + P), 1 / (S + P) and log(S + P), with S = exp(logS): one exp
  /// instead of four exps and a log per term (W3; equal up to rounding).
  /// Above [_linearLogSMax] S could overflow, and the log-space form is used
  /// instead.
  @pragma('vm:unsafe:no-bounds-checks')
  void _posteriorContinuousUnbounded(int i, double t) {
    final logT = t == 1.0 ? 0.0 : math.log(t);
    final expectedMean = _expectedMean[i];

    // Effective prior: the prediction, combined with the value-coupling terms
    // when the node also has value children.
    double priorPrecision, priorMean;
    if (_vcStart[i] == _vcStart[i + 1]) {
      priorPrecision = _expectedPrecision[i];
      priorMean = expectedMean;
    } else {
      final mf = meanFieldUpdates;
      priorPrecision = _maxNan(
        _expectedPrecision[i] + _valueCouplingPrecisionWpe(i, mf),
        minPosteriorPrecision,
      );
      priorMean = expectedMean + _valueCouplingMeanWpe(i, priorPrecision, mf);
    }

    // Observed volatility children.
    final u = _u;
    final coupling = u.coupling,
        tonic = u.tonic,
        prev = u.prev,
        beAux = u.beAux,
        vope = u.vope;
    var nc = 0;
    for (var e = _volcStart[i], end = _volcStart[i + 1]; e < end; e++) {
      final c = _volcIdx[e];
      if (_observed[c] == 0.0) continue;
      final d = _mean[c] - _expectedMean[c];
      final b = 1.0 / _precision[c] + d * d;
      coupling[nc] = _volcK[e];
      tonic[nc] = _effectiveTonicVolatility(i, c);
      prev[nc] = _maxNan(
        _currentVariance[c] + _valueCouplingVariance(c),
        minPredictedVariance,
      );
      beAux[nc] = b;
      vope[nc] = _expectedPrecision[c] * b - 1.0;
      nc++;
    }

    final posteriorMean = u.mix(
      nc,
      priorPrecision,
      priorMean,
      expectedMean,
      logT,
    );
    final p = _hasObservations(i) ? 1.0 / u.variance : _expectedPrecision[i];
    _precision[i] = _minNan(p, maxPosteriorPrecision);
    _mean[i] = posteriorMean;
  }
}

/// Scratch buffers and the mixture step of [Network]'s unbounded update
/// ([BatchNetwork] runs it stage by stage over a block, in
/// `batch/batch_updates.dart`).
///
/// The caller writes one row per observed volatility child (its coupling
/// strength, effective tonic volatility ω̃, predicted variance P, `b` and
/// volatility prediction error) into the first `nc` entries, then calls
/// [mix].
final class _Unbounded {
  // Every buffer is assigned straight from a typed-data constructor, so
  // dart2wasm keeps their exact type (see `Network._compile`).
  _Unbounded(int maxChildren)
    : coupling = Float64List(maxChildren),
      tonic = Float64List(maxChildren),
      prev = Float64List(maxChildren),
      beAux = Float64List(maxChildren),
      vope = Float64List(maxChildren),
      precisions = Float64List(maxChildren + 1),
      means = Float64List(maxChildren + 1),
      energies = Float64List(maxChildren + 1);

  /// Widest volatility fan-out these buffers hold.
  int get capacity => coupling.length;

  final Float64List coupling, tonic, prev, beAux, vope;
  final Float64List precisions, means, energies;

  /// Posterior variance from the last [mix].
  double variance = 0.0;

  /// The posterior mean over the `nc` children in the buffers, given the
  /// effective prior; the posterior variance is left in [variance].
  @pragma('vm:unsafe:no-bounds-checks')
  double mix(
    int nc,
    double priorPrecision,
    double priorMean,
    double expectedMean,
    double logT,
  ) {
    final coupling = this.coupling,
        tonic = this.tonic,
        prev = this.prev,
        beAux = this.beAux,
        vope = this.vope;
    // Expansion 1: at the prediction, concave part only.
    var precision1 = priorPrecision;
    var meanShift1 = priorPrecision * (priorMean - expectedMean);
    for (var k = 0; k < nc; k++) {
      final ck = coupling[k];
      final logS = logT + ck * expectedMean + tonic[k];
      final w = logS <= _linearLogSMax
          ? _ratio(fastExp(logS), prev[k])
          : sigmoid(logS - math.log(prev[k]));
      precision1 += 0.5 * ck * ck * w * (1.0 - w);
      meanShift1 += 0.5 * ck * w * vope[k];
    }
    final precisions = this.precisions, means = this.means;
    precisions[0] = precision1;
    means[0] = expectedMean + meanShift1 / precision1;
    var nx = 1;

    // Expansions 2..n+1: at each child's Lambert W0 mode.
    for (var k = 0; k < nc; k++) {
      final ck = coupling[k];
      if (ck == 0.0) continue;
      final pihatY = priorPrecision / (ck * ck);
      final gammaC = logT + ck * priorMean + tonic[k];
      // pyhgf: exp(min(log b − log 2π̂ + ½/π̂ − γ, logMaxFinite)). Without
      // the logs unless the product under/overflows (or is NaN).
      var wArg = beAux[k] / (2.0 * pihatY) * fastExp(0.5 / pihatY - gammaC);
      if (!(wArg > 0.0 && wArg <= double.maxFinite)) {
        final logWArg =
            math.log(beAux[k]) - math.log(2.0 * pihatY) + 0.5 / pihatY - gammaC;
        wArg = fastExp(_minNan(logWArg, logMaxFinite));
      }
      final yStar = gammaC + lambertW0(wArg) - 0.5 / pihatY;
      final xStar = (yStar - logT - tonic[k]) / ck;

      var full = priorPrecision;
      var concave = priorPrecision;
      var gradient = -priorPrecision * (xStar - priorMean);
      for (var o = 0; o < nc; o++) {
        final co = coupling[o];
        final logS = logT + co * xStar + tonic[o];
        double w, invDenom;
        if (logS <= _linearLogSMax) {
          final s = fastExp(logS);
          invDenom = 1.0 / (s + prev[o]);
          w = s * invDenom;
        } else {
          final logP = math.log(prev[o]);
          w = sigmoid(logS - logP);
          invDenom = fastExp(-logAddExp(logP, logS));
        }
        final da = beAux[o] * invDenom - 1.0;
        full += 0.5 * co * co * w * (w + (2.0 * w - 1.0) * da);
        concave += 0.5 * co * co * w * (1.0 - w);
        gradient += 0.5 * co * w * da;
      }
      final precision2 = full <= 0.0 ? concave : full;
      final mean2 = xStar + gradient / precision2;
      // Drop non-finite expansions (zero weight in the JAX backend).
      if (precision2.isFinite && mean2.isFinite) {
        precisions[nx] = precision2;
        means[nx] = mean2;
        nx++;
      }
    }

    // Softmax weights over the variational energy, then moment matching.
    final energies = this.energies;
    var maxEnergy = double.negativeInfinity;
    for (var j = 0; j < nx; j++) {
      final en = _energy(means[j], priorPrecision, priorMean, nc, logT);
      energies[j] = en;
      if (en > maxEnergy) maxEnergy = en;
    }
    var total = 0.0;
    for (var j = 0; j < nx; j++) {
      // exp(±0) is exactly 1: skip it for the maximum.
      final d = energies[j] - maxEnergy;
      final u = d == 0.0 ? 1.0 : fastExp(d);
      energies[j] = u;
      total += u;
    }
    var posteriorMean = 0.0;
    for (var j = 0; j < nx; j++) {
      energies[j] = energies[j] / total; // now the weights
      posteriorMean += energies[j] * means[j];
    }
    var posteriorVariance = 0.0;
    for (var j = 0; j < nx; j++) {
      final dm = means[j] - posteriorMean;
      posteriorVariance += energies[j] * (1.0 / precisions[j] + dm * dm);
    }

    variance = posteriorVariance;
    return posteriorMean;
  }

  /// Variational energy I(x) of the unbounded update:
  /// −½π(x − μ)² − ½ Σ_k [log(S_k + P_k) + b_k / (S_k + P_k)].
  @pragma('vm:unsafe:no-bounds-checks')
  double _energy(
    double x,
    double priorPrecision,
    double priorMean,
    int nc,
    double logT,
  ) {
    final dx = x - priorMean;
    var energy = -0.5 * priorPrecision * dx * dx;
    final prev = this.prev, beAux = this.beAux;
    for (var k = 0; k < nc; k++) {
      final logS = logT + coupling[k] * x + tonic[k];
      double logDenom, invDenom;
      if (logS <= _linearLogSMax) {
        final denom = fastExp(logS) + prev[k];
        logDenom = math.log(denom);
        invDenom = 1.0 / denom;
      } else {
        logDenom = logAddExp(math.log(prev[k]), logS);
        invDenom = fastExp(-logDenom);
      }
      energy += -0.5 * logDenom - 0.5 * beAux[k] * invDenom;
    }
    return energy;
  }
}

/// Largest logS for which the unbounded update evaluates S = exp(logS)
/// directly; beyond it (S > ~1e304) it switches to log space.
const double _linearLogSMax = 700.0;

/// S / (S + P): `sigmoid(log S − log P)` without the logs.
@pragma('vm:prefer-inline')
@pragma('wasm:prefer-inline')
double _ratio(double s, double p) => s / (s + p);
