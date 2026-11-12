part of '../network.dart';

// Batched update steps. Each runs the scalar step of `updates/*.dart` for the
// elements `[lo, hi)` of one node, **stage by stage**: every stage (an edge's
// contribution, an exp, a Lambert W iteration) loops over the block's
// elements before the next stage starts, with per-element partial results in
// `_BatchScratch`. The elements are independent, so the CPU overlaps their
// exps and divisions; one element at a time, each update is a chain of
// dependent ones (1.5-1.6x on lambertW0 alone, AOT and Wasm).
//
// Per element, the arithmetic is the scalar code's, in the same order, so
// every element matches a [Network] bit for bit (`test/src/batch_check.dart`
// checks it on every fixture). Change both together.
//
// Indexing: node i of element `lo + j` is at `i * B + lo + j` in a state
// column, edge k at `k * B + lo + j` in a strength column, and slot s of a
// per-child scratch array at `s * stride + j`.

/// Per-block scratch for the batched kernels: `stride` (the block size)
/// entries per element array, `stride` per slot in the slot arrays.
final class _BatchScratch {
  // Every array is assigned straight from a typed-data constructor, so
  // dart2wasm keeps their exact type (see `Network._compile`).
  _BatchScratch(this.stride, int maxChildren)
    : f0 = Float64List(stride),
      f1 = Float64List(stride),
      f2 = Float64List(stride),
      f3 = Float64List(stride),
      f4 = Float64List(stride),
      f5 = Float64List(stride),
      f6 = Float64List(stride),
      f7 = Float64List(stride),
      f8 = Float64List(stride),
      f9 = Float64List(stride),
      f10 = Float64List(stride),
      f11 = Float64List(stride),
      flags = Uint8List(stride),
      done = Uint8List(stride),
      nc = Int32List(stride),
      nx = Int32List(stride),
      act = Int32List(stride),
      coupling = Float64List(maxChildren * stride),
      tonic = Float64List(maxChildren * stride),
      prev = Float64List(maxChildren * stride),
      beAux = Float64List(maxChildren * stride),
      vope = Float64List(maxChildren * stride),
      precisions = Float64List((maxChildren + 1) * stride),
      means = Float64List((maxChildren + 1) * stride),
      energies = Float64List((maxChildren + 1) * stride);

  final int stride;

  /// Per-element temporaries (each kernel names its own uses).
  final Float64List f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11;
  final Uint8List flags, done;
  final Int32List nc, nx, act;

  /// Unbounded update: one slot per observed volatility child...
  final Float64List coupling, tonic, prev, beAux, vope;

  /// ... and one per quadratic expansion.
  final Float64List precisions, means, energies;
}

/// [lambertW0] of `z[0..n)` into [w], bit for bit, with the Halley
/// iterations of all elements interleaved.
@pragma('vm:unsafe:no-bounds-checks')
void _lambertW0Block(Float64List z, Float64List w, Uint8List done, int n) {
  for (var a = 0; a < n; a++) {
    w[a] = math.log(z[a] + 1.0);
    done[a] = 0;
  }
  for (var i = 0; i < 6; i++) {
    var active = 0;
    for (var a = 0; a < n; a++) {
      if (done[a] != 0) continue;
      final wa = w[a];
      final f = wa - z[a] * fastExp(-wa);
      final w1 = wa + 1.0;
      final step = (2.0 * f * w1) / (2.0 * w1 * w1 - f * (wa + 2.0));
      final wn = wa - step;
      w[a] = wn;
      if (step.abs() <= 1e-12 * wn.abs()) {
        done[a] = 1;
      } else {
        active++;
      }
    }
    if (active == 0) break;
  }
}

extension on BatchNetwork {
  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  Coupling _couplingOf(int cf, int node) =>
      cf == _kCustomCoupling ? _couplings[node] : CouplingFn.values[cf];

  // ---------------------------------------------------------------------------
  // Predictions (updates/prediction.dart)
  // ---------------------------------------------------------------------------

  @pragma('vm:unsafe:no-bounds-checks')
  void _bPredictionContinuous(int i, double t, int lo, int hi) {
    final nb = _b, m = hi - lo, base = i * nb + lo;
    final s = _s;
    final drift = s.f0, valueCouplingVariance = s.f1, totalVolatility = s.f2;
    final expectedMean = _expectedMean, expectedPrecision = _expectedPrecision;
    final tonicDrift = _tonicDrift, tonicVolatility = _tonicVolatility;
    for (var j = 0; j < m; j++) {
      drift[j] = tonicDrift[base + j];
      valueCouplingVariance[j] = 0.0;
      totalVolatility[j] = tonicVolatility[base + j];
    }

    final vpK = _vpK;
    for (var k = _vpStart[i], end = _vpStart[i + 1]; k < end; k++) {
      final p = _vpIdx[k];
      final pb = p * nb + lo, kb = k * nb + lo;
      final cf = _couplingFn[p];
      if (cf == 0) {
        for (var j = 0; j < m; j++) {
          final alpha = vpK[kb + j];
          drift[j] += alpha * expectedMean[pb + j];
          final coeff = t * alpha; // · g'(μ̂) = 1, exactly
          valueCouplingVariance[j] += coeff * coeff / expectedPrecision[pb + j];
        }
      } else {
        final fn = _couplingOf(cf, p);
        for (var j = 0; j < m; j++) {
          final pMean = expectedMean[pb + j];
          final alpha = vpK[kb + j];
          drift[j] += alpha * fn.f(pMean);
          final coeff = t * alpha * fn.df(pMean);
          valueCouplingVariance[j] += coeff * coeff / expectedPrecision[pb + j];
        }
      }
    }

    // An input's volatility parents are noise parents: Σ κ μ̂ without the
    // MGF terms is the log variance of its observation noise.
    final isInput = _isLeaf[i] == 1;
    final volpK = _volpK;
    final volStart = _volpStart[i], volEnd = _volpStart[i + 1];
    for (var k = volStart; k < volEnd; k++) {
      final pb = _volpIdx[k] * nb + lo, kb = k * nb + lo;
      for (var j = 0; j < m; j++) {
        final kappa = volpK[kb + j];
        totalVolatility[j] += kappa * expectedMean[pb + j];
        if (!isInput) {
          totalVolatility[j] +=
              (kappa * kappa) / (2.0 * expectedPrecision[pb + j]);
        }
      }
    }

    final mean = _mean,
        precision = _precision,
        autoconnection = _autoconnectionStrength,
        currentVariance = _currentVariance,
        effectivePrecision = _effectivePrecision,
        conditionalExpectedPrecision = _conditionalExpectedPrecision;
    if (isInput) {
      _bStoreInputPrediction(i, t, lo, hi, totalVolatility);
      return;
    }
    for (var j = 0; j < m; j++) {
      final ie = base + j;
      final lambda = autoconnection[ie];
      final prec = precision[ie];
      final expMean = lambda * mean[ie] + t * drift[j];
      final pvRaw = t * fastExp(totalVolatility[j]);
      final predictedVolatility = pvRaw > 1e-128 ? pvRaw : double.nan;
      final carriedVariance = (lambda * lambda) / prec;
      final conditional =
          1.0 / floorPredictedVariance(carriedVariance + predictedVolatility);
      final expected =
          1.0 /
          floorPredictedVariance(
            carriedVariance + predictedVolatility + valueCouplingVariance[j],
          );
      currentVariance[ie] = carriedVariance;
      expectedMean[ie] = expMean;
      effectivePrecision[ie] = predictedVolatility * expected;
      expectedPrecision[ie] = expected;
      conditionalExpectedPrecision[ie] = conditional;
    }
  }

  /// The prediction of input node [i] (`Network._storeInputPrediction`),
  /// given the drift in `_s.f0` and `logNoiseVariance[j]` = ε_u + Σ κ μ̂_q
  /// over its noise parents.
  @pragma('vm:unsafe:no-bounds-checks')
  void _bStoreInputPrediction(
    int i,
    double t,
    int lo,
    int hi,
    Float64List logNoiseVariance,
  ) {
    final nb = _b, m = hi - lo, base = i * nb + lo;
    final drift = _s.f0;
    final hasNoiseParents = _volpStart[i] != _volpStart[i + 1];
    final mean = _mean,
        precision = _precision,
        autoconnection = _autoconnectionStrength,
        expectedMean = _expectedMean,
        expectedPrecision = _expectedPrecision,
        currentVariance = _currentVariance,
        effectivePrecision = _effectivePrecision,
        conditionalExpectedPrecision = _conditionalExpectedPrecision;
    for (var j = 0; j < m; j++) {
      final ie = base + j;
      final lambda = autoconnection[ie];
      final prec = precision[ie];
      currentVariance[ie] = (lambda * lambda) / prec;
      expectedMean[ie] = lambda * mean[ie] + t * drift[j];
      if (hasNoiseParents) {
        final noisePrecision = fastExp(-logNoiseVariance[j]);
        expectedPrecision[ie] = noisePrecision;
        conditionalExpectedPrecision[ie] = noisePrecision;
        effectivePrecision[ie] = 1.0;
      } else {
        expectedPrecision[ie] = prec;
        conditionalExpectedPrecision[ie] = prec;
        effectivePrecision[ie] = 0.0;
      }
    }
  }

  @pragma('vm:unsafe:no-bounds-checks')
  void _bPredictionContinuousMeanField(int i, double t, int lo, int hi) {
    final nb = _b, m = hi - lo, base = i * nb + lo;
    final s = _s;
    final drift = s.f0, totalVolatility = s.f2;
    final expectedMean = _expectedMean, expectedPrecision = _expectedPrecision;
    final tonicDrift = _tonicDrift, tonicVolatility = _tonicVolatility;
    for (var j = 0; j < m; j++) {
      drift[j] = tonicDrift[base + j];
      totalVolatility[j] = tonicVolatility[base + j];
    }

    final vpK = _vpK;
    for (var k = _vpStart[i], end = _vpStart[i + 1]; k < end; k++) {
      final p = _vpIdx[k];
      final pb = p * nb + lo, kb = k * nb + lo;
      final cf = _couplingFn[p];
      if (cf == 0) {
        for (var j = 0; j < m; j++) {
          drift[j] += vpK[kb + j] * expectedMean[pb + j];
        }
      } else {
        final fn = _couplingOf(cf, p);
        for (var j = 0; j < m; j++) {
          drift[j] += vpK[kb + j] * fn.f(expectedMean[pb + j]);
        }
      }
    }

    final volpK = _volpK;
    final volStart = _volpStart[i], volEnd = _volpStart[i + 1];
    for (var k = volStart; k < volEnd; k++) {
      final pb = _volpIdx[k] * nb + lo, kb = k * nb + lo;
      for (var j = 0; j < m; j++) {
        totalVolatility[j] += volpK[kb + j] * expectedMean[pb + j];
      }
    }

    if (_isLeaf[i] == 1) {
      _bStoreInputPrediction(i, t, lo, hi, totalVolatility);
      return;
    }
    final mean = _mean,
        precision = _precision,
        autoconnection = _autoconnectionStrength,
        currentVariance = _currentVariance,
        effectivePrecision = _effectivePrecision,
        conditionalExpectedPrecision = _conditionalExpectedPrecision;
    for (var j = 0; j < m; j++) {
      final ie = base + j;
      final lambda = autoconnection[ie];
      final prec = precision[ie];
      final expMean = lambda * mean[ie] + t * drift[j];
      final pvRaw = t * fastExp(totalVolatility[j]);
      final predictedVolatility = pvRaw > 1e-128 ? pvRaw : double.nan;
      final carriedVariance = (lambda * lambda) / prec;
      final expected =
          1.0 / floorPredictedVariance(carriedVariance + predictedVolatility);
      currentVariance[ie] = carriedVariance;
      expectedMean[ie] = expMean;
      effectivePrecision[ie] = predictedVolatility * expected;
      expectedPrecision[ie] = expected;
      conditionalExpectedPrecision[ie] = expected;
    }
  }

  @pragma('vm:unsafe:no-bounds-checks')
  void _bPredictionBinary(int i, int lo, int hi) {
    final nb = _b, m = hi - lo, base = i * nb + lo;
    final x = _s.f0, expectedMean = _expectedMean;
    for (var j = 0; j < m; j++) {
      x[j] = 0.0;
    }
    for (var k = _vpStart[i], end = _vpStart[i + 1]; k < end; k++) {
      final pb = _vpIdx[k] * nb + lo;
      for (var j = 0; j < m; j++) {
        x[j] += expectedMean[pb + j];
      }
    }
    final v = precisionClippingValue;
    final expectedPrecision = _expectedPrecision;
    for (var j = 0; j < m; j++) {
      var p = sigmoid(x[j]);
      if (p < v) {
        p = v;
      } else if (p > 1.0 - v) {
        p = 1.0 - v;
      }
      expectedMean[base + j] = p;
      expectedPrecision[base + j] = p * (1.0 - p);
    }
  }

  // ---------------------------------------------------------------------------
  // Prediction errors (updates/prediction_error.dart)
  // ---------------------------------------------------------------------------

  @pragma('vm:unsafe:no-bounds-checks')
  void _bPredictionErrorContinuous(int i, int lo, int hi) {
    final nb = _b, m = hi - lo, base = i * nb + lo;
    final isLeaf = _isLeaf[i] == 1;
    final flags = _s.flags;
    if (!isLeaf) _bObservedChildren(i, lo, m, flags);
    final mean = _mean,
        expectedMean = _expectedMean,
        expectedPrecision = _expectedPrecision,
        precision = _precision,
        vape = _valuePredictionError,
        vope = _volatilityPredictionError,
        observed = _observed;
    // A noise child sends its noise prediction error instead of Δ
    // (`_bNoisePredictionErrorContinuous`).
    final noiseChild = _noiseChild[i] == 1;
    for (var j = 0; j < m; j++) {
      final ie = base + j;
      final ep = expectedPrecision[ie];
      final delta = mean[ie] - expectedMean[ie];
      vape[ie] = delta;
      if (!noiseChild) {
        vope[ie] = (ep / precision[ie]) + ep * delta * delta - 1.0;
      }
      if (!isLeaf) observed[ie] = flags[j] != 0 ? 1.0 : 0.0;
    }
  }

  /// `Network._noisePredictionErrorContinuous`: ε = π̂_u (Σ_b (t α_b
  /// g_b'(μ_b))² / π_b + (u − m_u)²) − 1 into the volatility prediction
  /// error of input [i].
  @pragma('vm:unsafe:no-bounds-checks')
  void _bNoisePredictionErrorContinuous(int i, double t, int lo, int hi) {
    final nb = _b, m = hi - lo, base = i * nb + lo;
    final s = _s;
    final postMean = s.f0, postVar = s.f1;
    final mean = _mean,
        expectedMean = _expectedMean,
        precision = _precision,
        vpK = _vpK;
    for (var j = 0; j < m; j++) {
      postMean[j] = expectedMean[base + j];
      postVar[j] = 0.0;
    }
    final fixConstant = fixes.constantNoiseVariance;
    for (var k = _vpStart[i], end = _vpStart[i + 1]; k < end; k++) {
      final p = _vpIdx[k];
      final pb = p * nb + lo, kb = k * nb + lo;
      final cf = _couplingFn[p];
      final fn = cf == 0 ? null : _couplingOf(cf, p);
      // A bias parent adds no variance (P21).
      final known = fixConstant && _kind[p] == _kindConstant;
      for (var j = 0; j < m; j++) {
        final alpha = vpK[kb + j];
        final pm = mean[pb + j], pem = expectedMean[pb + j];
        double shift, gPrime;
        if (fn == null) {
          shift = pm - pem;
          gPrime = 1.0;
        } else {
          shift = fn.f(pm) - fn.f(pem);
          gPrime = fn.df(pm);
        }
        postMean[j] += t * alpha * shift;
        if (known) continue;
        final coeff = t * alpha * gPrime;
        postVar[j] += coeff * coeff / precision[pb + j];
      }
    }
    final vope = _volatilityPredictionError,
        expectedPrecision = _expectedPrecision;
    for (var j = 0; j < m; j++) {
      final ie = base + j;
      final d = mean[ie] - postMean[j];
      vope[ie] = expectedPrecision[ie] * (postVar[j] + d * d) - 1.0;
    }
  }

  @pragma('vm:unsafe:no-bounds-checks')
  void _bPredictionErrorBinary(int i, int lo, int hi) {
    final nb = _b;
    final expectedPrecision = _expectedPrecision, precision = _precision;
    for (var ie = i * nb + lo, end = i * nb + hi; ie < end; ie++) {
      final ep = expectedPrecision[ie];
      _valuePredictionError[ie] =
          (_mean[ie] - _expectedMean[ie]) * _observed[ie] / ep;
      precision[ie] = ep;
    }
  }

  /// `flags[j]`: whether any value or volatility child of [i] observed, for
  /// element `lo + j` (`Network._hasObservations`).
  @pragma('vm:unsafe:no-bounds-checks')
  void _bObservedChildren(int i, int lo, int m, Uint8List flags) {
    final nb = _b, observed = _observed;
    for (var j = 0; j < m; j++) {
      flags[j] = 0;
    }
    for (var k = _vcStart[i], end = _vcStart[i + 1]; k < end; k++) {
      final cb = _vcIdx[k] * nb + lo;
      for (var j = 0; j < m; j++) {
        if (observed[cb + j] != 0.0) flags[j] = 1;
      }
    }
    for (var k = _volcStart[i], end = _volcStart[i + 1]; k < end; k++) {
      final cb = _volcIdx[k] * nb + lo;
      for (var j = 0; j < m; j++) {
        if (observed[cb + j] != 0.0) flags[j] = 1;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Posterior building blocks (updates/posterior.dart), one stage each.
  // ---------------------------------------------------------------------------

  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  @pragma('vm:unsafe:no-bounds-checks')
  double _bRelaxedGain(int c, int ce, bool meanField) {
    final cep = _expectedPrecision[ce];
    if (meanField || _kind[c] != _kindContinuous) return cep;
    final cond = _conditionalExpectedPrecision[ce];
    if (_isLeaf[c] == 1) return cond;
    final cp = _precision[ce];
    return cond * cp / (cond + (cp - cep));
  }

  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  @pragma('vm:unsafe:no-bounds-checks')
  double _bRelaxedEffective(int c, int ce, bool meanField) {
    final cep = _expectedPrecision[ce];
    if (meanField || _kind[c] != _kindContinuous) return cep;
    final cond = _conditionalExpectedPrecision[ce];
    if (_isLeaf[c] == 1) return cond;
    final piY = _precision[ce] - cep;
    return cond * piY / (cond + piY);
  }

  /// `out[j]` = the precision increment from the value children of [i]
  /// (`Network._valueCouplingPrecisionWpe`), reading the node's current
  /// mean.
  @pragma('vm:unsafe:no-bounds-checks')
  void _bValuePrecisionWpe(
    int i,
    int lo,
    int m,
    bool meanField,
    Float64List out,
  ) {
    for (var j = 0; j < m; j++) {
      out[j] = 0.0;
    }
    final start = _vcStart[i], end = _vcStart[i + 1];
    if (start == end) return;
    final nb = _b, base = i * nb + lo;
    final cf = _couplingFn[i];
    final mean = _mean, vcK = _vcK, observed = _observed;
    for (var k = start; k < end; k++) {
      final c = _vcIdx[k];
      final cb = c * nb + lo, kb = k * nb + lo;
      if (cf == 0) {
        for (var j = 0; j < m; j++) {
          final ce = cb + j;
          final alpha = vcK[kb + j];
          const gPrimeSq = 1.0, secondTerm = 0.0;
          out[j] +=
              (_bRelaxedEffective(c, ce, meanField) * alpha * alpha * gPrimeSq -
                  _bRelaxedGain(c, ce, meanField) * secondTerm) *
              observed[ce];
        }
      } else {
        final fn = _couplingOf(cf, i);
        for (var j = 0; j < m; j++) {
          final ce = cb + j;
          final alpha = vcK[kb + j];
          final parentMean = mean[base + j];
          final gp = fn.df(parentMean);
          final gPrimeSq = gp * gp;
          final secondTerm =
              alpha * fn.d2f(parentMean) * _valuePredictionError[ce];
          out[j] +=
              (_bRelaxedEffective(c, ce, meanField) * alpha * alpha * gPrimeSq -
                  _bRelaxedGain(c, ce, meanField) * secondTerm) *
              observed[ce];
        }
      }
    }
  }

  /// `out[j]` = the mean increment from the value children of [i] at the
  /// node precision `nodePrecision[j]` (`Network._valueCouplingMeanWpe`).
  @pragma('vm:unsafe:no-bounds-checks')
  void _bValueMeanWpe(
    int i,
    int lo,
    int m,
    Float64List nodePrecision,
    bool meanField,
    Float64List out,
  ) {
    for (var j = 0; j < m; j++) {
      out[j] = 0.0;
    }
    final start = _vcStart[i], end = _vcStart[i + 1];
    if (start == end) return;
    final nb = _b, base = i * nb + lo;
    final cf = _couplingFn[i];
    final fn = cf == 0 ? null : _couplingOf(cf, i);
    final mean = _mean, vcK = _vcK, observed = _observed;
    final vape = _valuePredictionError;
    for (var k = start; k < end; k++) {
      final c = _vcIdx[k];
      final cb = c * nb + lo, kb = k * nb + lo;
      for (var j = 0; j < m; j++) {
        final ce = cb + j;
        final gPrime = fn == null ? 1.0 : fn.df(mean[base + j]);
        final childVape = vape[ce] * observed[ce];
        out[j] +=
            (vcK[kb + j] *
                gPrime *
                _bRelaxedGain(c, ce, meanField) /
                nodePrecision[j]) *
            childVape;
      }
    }
  }

  /// `acc[j] +=` the standard volatility precision increment from the
  /// volatility children of [i] (`Network._volatilityPrecisionWpe`), or
  /// from its noise children only with [noiseOnly].
  @pragma('vm:unsafe:no-bounds-checks')
  void _bVolatilityPrecisionWpe(
    int i,
    int lo,
    int m,
    Float64List acc, {
    bool noiseOnly = false,
  }) {
    final nb = _b;
    final volcK = _volcK,
        eff = _effectivePrecision,
        vope = _volatilityPredictionError,
        observed = _observed;
    for (var k = _volcStart[i], end = _volcStart[i + 1]; k < end; k++) {
      final c = _volcIdx[k];
      if (noiseOnly && _noiseChild[c] == 0) continue;
      final cb = c * nb + lo, kb = k * nb + lo;
      for (var j = 0; j < m; j++) {
        final ce = cb + j;
        final kappa = volcK[kb + j];
        final e = eff[ce];
        final v = vope[ce];
        final ke = kappa * e;
        acc[j] +=
            (0.5 * ke * ke + ke * ke * v - 0.5 * kappa * kappa * e * v) *
            observed[ce];
      }
    }
  }

  /// `out[j]` = the volatility mean increment from the volatility children
  /// of [i] at `nodePrecision[j]` (`Network._volatilityMeanWpe`), or from
  /// its noise children only with [noiseOnly].
  @pragma('vm:unsafe:no-bounds-checks')
  void _bVolatilityMeanWpe(
    int i,
    int lo,
    int m,
    Float64List nodePrecision,
    Float64List out, {
    bool noiseOnly = false,
  }) {
    for (var j = 0; j < m; j++) {
      out[j] = 0.0;
    }
    final nb = _b;
    final volcK = _volcK,
        eff = _effectivePrecision,
        vope = _volatilityPredictionError,
        observed = _observed;
    for (var k = _volcStart[i], end = _volcStart[i + 1]; k < end; k++) {
      final c = _volcIdx[k];
      if (noiseOnly && _noiseChild[c] == 0) continue;
      final cb = c * nb + lo, kb = k * nb + lo;
      for (var j = 0; j < m; j++) {
        final ce = cb + j;
        out[j] +=
            (volcK[kb + j] * eff[ce] * vope[ce]) /
            (2.0 * nodePrecision[j]) *
            observed[ce];
      }
    }
  }

  /// Whether [i] has noise children (inputs among its volatility children).
  @pragma('vm:unsafe:no-bounds-checks')
  bool _hasNoiseChildren(int i) {
    for (var k = _volcStart[i], end = _volcStart[i + 1]; k < end; k++) {
      if (_noiseChild[_volcIdx[k]] == 1) return true;
    }
    return false;
  }

  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  @pragma('vm:unsafe:no-bounds-checks')
  double _bObservedOrPredicted(int ie, bool hasObs, double wpe) {
    final ep = _expectedPrecision[ie];
    final p = hasObs ? _maxNan(ep + wpe, minPosteriorPrecision) : ep;
    return _minNan(p, maxPosteriorPrecision);
  }

  /// ω̃ of volatility child [c] seen from its parent [i], for element [e]
  /// (`Network._effectiveTonicVolatility`).
  @pragma('vm:unsafe:no-bounds-checks')
  double _bEffectiveTonicVolatility(int i, int c, int e) {
    final nb = _b;
    var tv = _tonicVolatility[c * nb + e];
    for (var k = _volpStart[c], end = _volpStart[c + 1]; k < end; k++) {
      final p = _volpIdx[k];
      if (p != i) tv += _volpK[k * nb + e] * _expectedMean[p * nb + e];
    }
    return tv;
  }

  @pragma('vm:unsafe:no-bounds-checks')
  double _bMgfTerms(int c, int e) {
    final nb = _b;
    var sum = 0.0;
    for (var k = _volpStart[c], end = _volpStart[c + 1]; k < end; k++) {
      final kappa = _volpK[k * nb + e];
      sum += kappa * kappa / (2.0 * _expectedPrecision[_volpIdx[k] * nb + e]);
    }
    return sum;
  }

  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  @pragma('vm:unsafe:no-bounds-checks')
  double _bValueCouplingVariance(int ce) {
    final v =
        1.0 / _expectedPrecision[ce] - 1.0 / _conditionalExpectedPrecision[ce];
    return _maxNan(v, 0.0);
  }

  // ---------------------------------------------------------------------------
  // Standard update
  // ---------------------------------------------------------------------------

  @pragma('vm:unsafe:no-bounds-checks')
  void _bPosteriorContinuous(int i, int lo, int hi, bool meanField) {
    final nb = _b, m = hi - lo, base = i * nb + lo;
    final s = _s;
    final wpe = s.f0, prec = s.f1, valueMean = s.f2, volMean = s.f3;
    final flags = s.flags;
    _bObservedChildren(i, lo, m, flags);
    // Computed for every element; ignored where no child observed.
    _bValuePrecisionWpe(i, lo, m, meanField, wpe);
    _bVolatilityPrecisionWpe(i, lo, m, wpe);
    for (var j = 0; j < m; j++) {
      prec[j] = _bObservedOrPredicted(base + j, flags[j] != 0, wpe[j]);
    }
    _bValueMeanWpe(i, lo, m, prec, meanField, valueMean);
    _bVolatilityMeanWpe(i, lo, m, prec, volMean);
    final mean = _mean, precision = _precision, expectedMean = _expectedMean;
    for (var j = 0; j < m; j++) {
      final ie = base + j;
      precision[ie] = prec[j];
      mean[ie] = expectedMean[ie] + (valueMean[j] + volMean[j]);
    }
  }

  // ---------------------------------------------------------------------------
  // eHGF update
  // ---------------------------------------------------------------------------

  @pragma('vm:unsafe:no-bounds-checks')
  void _bPosteriorContinuousEhgf(
    int i,
    double t,
    int lo,
    int hi,
    bool meanField,
  ) {
    final nb = _b, m = hi - lo, base = i * nb + lo;
    final s = _s;
    final ep = s.f0, valueMean = s.f1, volMean = s.f2, wpe = s.f3;
    final flags = s.flags;
    final mean = _mean, expectedMean = _expectedMean;
    for (var j = 0; j < m; j++) {
      ep[j] = _expectedPrecision[base + j];
    }
    // Mean first, at the predicted precision (reading the old mean).
    _bValueMeanWpe(i, lo, m, ep, meanField, valueMean);
    _bVolatilityMeanWpe(i, lo, m, ep, volMean);
    for (var j = 0; j < m; j++) {
      final ie = base + j;
      mean[ie] = expectedMean[ie] + valueMean[j] + volMean[j];
    }

    // Then the precision, re-predicting each volatility child from the new
    // mean.
    _bObservedChildren(i, lo, m, flags);
    _bValuePrecisionWpe(i, lo, m, meanField, wpe);
    final mgf = !meanField && ehgfMgfCorrection;
    final volcK = _volcK;
    final eff = _effectivePrecision,
        vopeC = _volatilityPredictionError,
        observedC = _observed;
    for (var k = _volcStart[i], end = _volcStart[i + 1]; k < end; k++) {
      final c = _volcIdx[k];
      final cb = c * nb + lo, kb = k * nb + lo;
      if (_noiseChild[c] == 1) {
        // ½ κ² (1 + ε) (Eq. 78 of Weber et al., 2026), as in the standard
        // update (`_bVolatilityPrecisionWpe`).
        for (var j = 0; j < m; j++) {
          final ce = cb + j;
          final kappa = volcK[kb + j];
          final e = eff[ce];
          final v = vopeC[ce];
          final ke = kappa * e;
          wpe[j] +=
              (0.5 * ke * ke + ke * ke * v - 0.5 * kappa * kappa * e * v) *
              observedC[ce];
        }
        continue;
      }
      for (var j = 0; j < m; j++) {
        var tonic = _bEffectiveTonicVolatility(i, c, lo + j);
        if (mgf) tonic += _bMgfTerms(c, lo + j);
        wpe[j] += _bEhgfVolatilityIncrement(
          cb + j,
          volcK[kb + j],
          mean[base + j],
          tonic,
          t,
        );
      }
    }
    final precision = _precision;
    for (var j = 0; j < m; j++) {
      precision[base + j] = _bObservedOrPredicted(
        base + j,
        flags[j] != 0,
        wpe[j],
      );
    }
  }

  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  @pragma('vm:unsafe:no-bounds-checks')
  double _bEhgfVolatilityIncrement(
    int ce,
    double kappa,
    double mean,
    double tonicVolatility,
    double t,
  ) {
    final previousVariance = _currentVariance[ce] + _bValueCouplingVariance(ce);
    final predictedVolatility = t * fastExp(kappa * mean + tonicVolatility);
    final expectedPrecision = 1.0 / (previousVariance + predictedVolatility);
    final eff = predictedVolatility * expectedPrecision;
    final weight = (predictedVolatility - previousVariance) * expectedPrecision;
    final d = _mean[ce] - _expectedMean[ce];
    final vope = (1.0 / _precision[ce] + d * d) * expectedPrecision - 1.0;
    final inc = 0.5 * kappa * kappa * eff * (eff + weight * vope);
    return _maxNan(inc, 0.0) * _observed[ce];
  }

  // ---------------------------------------------------------------------------
  // Unbounded update (`Network._posteriorContinuousUnbounded` and
  // `_Unbounded.mix`), with the children's rows and the expansions in slot
  // arrays: element j's slot s at `s * stride + j`. Elements can have
  // different numbers of observed children (`nc`) and of finite expansions
  // (`nx`).
  // ---------------------------------------------------------------------------

  @pragma('vm:unsafe:no-bounds-checks')
  void _bPosteriorContinuousUnbounded(int i, double t, int lo, int hi) {
    final nb = _b, m = hi - lo, base = i * nb + lo;
    final s = _s, st = s.stride;
    final logT = t == 1.0 ? 0.0 : math.log(t);
    final em = s.f0, pp = s.f1, pm = s.f2, tmp = s.f3;
    final nc = s.nc, nx = s.nx;
    final coupling = s.coupling,
        tonic = s.tonic,
        prev = s.prev,
        beAux = s.beAux,
        vope = s.vope;
    final precisions = s.precisions, means = s.means, energies = s.energies;

    // Effective prior.
    final expectedMean = _expectedMean, expectedPrecision = _expectedPrecision;
    for (var j = 0; j < m; j++) {
      em[j] = expectedMean[base + j];
    }
    // With the noise children's quadratic terms (pyhgf #477), which the
    // expansions below leave out.
    final hasNoise = _hasNoiseChildren(i);
    if (_vcStart[i] == _vcStart[i + 1] && !hasNoise) {
      for (var j = 0; j < m; j++) {
        pp[j] = expectedPrecision[base + j];
        pm[j] = em[j];
      }
    } else {
      final mf = meanFieldUpdates;
      final noise = s.f4;
      _bValuePrecisionWpe(i, lo, m, mf, tmp);
      for (var j = 0; j < m; j++) {
        noise[j] = 0.0;
      }
      if (hasNoise) _bVolatilityPrecisionWpe(i, lo, m, noise, noiseOnly: true);
      for (var j = 0; j < m; j++) {
        pp[j] = _maxNan(
          expectedPrecision[base + j] + tmp[j] + noise[j],
          minPosteriorPrecision,
        );
      }
      _bValueMeanWpe(i, lo, m, pp, mf, tmp);
      _bVolatilityMeanWpe(i, lo, m, pp, noise, noiseOnly: true);
      for (var j = 0; j < m; j++) {
        pm[j] = em[j] + tmp[j] + noise[j];
      }
    }

    // Observed volatility children other than noise children, one slot
    // each.
    for (var j = 0; j < m; j++) {
      nc[j] = 0;
    }
    final mean = _mean, precision = _precision, observed = _observed;
    final volcK = _volcK, currentVariance = _currentVariance;
    for (var k = _volcStart[i], end = _volcStart[i + 1]; k < end; k++) {
      final c = _volcIdx[k];
      if (_noiseChild[c] == 1) continue;
      final cb = c * nb + lo, kb = k * nb + lo;
      for (var j = 0; j < m; j++) {
        final ce = cb + j;
        if (observed[ce] == 0.0) continue;
        final slot = nc[j];
        final o = slot * st + j;
        final d = mean[ce] - expectedMean[ce];
        final b = 1.0 / precision[ce] + d * d;
        coupling[o] = volcK[kb + j];
        tonic[o] = _bEffectiveTonicVolatility(i, c, lo + j);
        prev[o] = _maxNan(
          currentVariance[ce] + _bValueCouplingVariance(ce),
          minPredictedVariance,
        );
        beAux[o] = b;
        vope[o] = expectedPrecision[ce] * b - 1.0;
        nc[j] = slot + 1;
      }
    }
    var maxNc = 0;
    for (var j = 0; j < m; j++) {
      if (nc[j] > maxNc) maxNc = nc[j];
    }

    // Expansion 1: at the prediction, concave part only.
    final precision1 = s.f4, meanShift1 = s.f5;
    for (var j = 0; j < m; j++) {
      precision1[j] = pp[j];
      meanShift1[j] = pp[j] * (pm[j] - em[j]);
    }
    for (var k = 0; k < maxNc; k++) {
      final ko = k * st;
      for (var j = 0; j < m; j++) {
        if (k >= nc[j]) continue;
        final o = ko + j;
        final ck = coupling[o];
        final logS = logT + ck * em[j] + tonic[o];
        final w = logS <= _linearLogSMax
            ? _ratio(fastExp(logS), prev[o])
            : sigmoid(logS - math.log(prev[o]));
        precision1[j] += 0.5 * ck * ck * w * (1.0 - w);
        meanShift1[j] += 0.5 * ck * w * vope[o];
      }
    }
    for (var j = 0; j < m; j++) {
      precisions[j] = precision1[j];
      means[j] = em[j] + meanShift1[j] / precision1[j];
      nx[j] = 1;
    }

    // Expansions 2..n+1: at each child's Lambert W0 mode, over the elements
    // ("active", `act[a]`) that have a child with non-zero coupling in slot
    // k.
    final act = s.act;
    final z = s.f4, w0 = s.f5, pihat = s.f6, gamma = s.f7;
    final xs = s.f8, full = s.f9, concave = s.f10, gradient = s.f11;
    for (var k = 0; k < maxNc; k++) {
      final ko = k * st;
      var na = 0;
      for (var j = 0; j < m; j++) {
        if (k >= nc[j]) continue;
        final o = ko + j;
        final ck = coupling[o];
        if (ck == 0.0) continue;
        final pihatY = pp[j] / (ck * ck);
        final gammaC = logT + ck * pm[j] + tonic[o];
        var wArg = beAux[o] / (2.0 * pihatY) * fastExp(0.5 / pihatY - gammaC);
        if (!(wArg > 0.0 && wArg <= double.maxFinite)) {
          final logWArg =
              math.log(beAux[o]) -
              math.log(2.0 * pihatY) +
              0.5 / pihatY -
              gammaC;
          wArg = fastExp(_minNan(logWArg, logMaxFinite));
        }
        act[na] = j;
        z[na] = wArg;
        pihat[na] = pihatY;
        gamma[na] = gammaC;
        na++;
      }
      if (na == 0) continue;
      _lambertW0Block(z, w0, s.done, na);
      for (var a = 0; a < na; a++) {
        final j = act[a];
        final o = ko + j;
        final yStar = gamma[a] + w0[a] - 0.5 / pihat[a];
        final xStar = (yStar - logT - tonic[o]) / coupling[o];
        xs[a] = xStar;
        full[a] = pp[j];
        concave[a] = pp[j];
        gradient[a] = -pp[j] * (xStar - pm[j]);
      }
      for (var q = 0; q < maxNc; q++) {
        final qo = q * st;
        for (var a = 0; a < na; a++) {
          final j = act[a];
          if (q >= nc[j]) continue;
          final o = qo + j;
          final co = coupling[o];
          final logS = logT + co * xs[a] + tonic[o];
          double w, invDenom;
          if (logS <= _linearLogSMax) {
            final sv = fastExp(logS);
            invDenom = 1.0 / (sv + prev[o]);
            w = sv * invDenom;
          } else {
            final logP = math.log(prev[o]);
            w = sigmoid(logS - logP);
            invDenom = fastExp(-logAddExp(logP, logS));
          }
          final da = beAux[o] * invDenom - 1.0;
          full[a] += 0.5 * co * co * w * (w + (2.0 * w - 1.0) * da);
          concave[a] += 0.5 * co * co * w * (1.0 - w);
          gradient[a] += 0.5 * co * w * da;
        }
      }
      for (var a = 0; a < na; a++) {
        final precision2 = full[a] <= 0.0 ? concave[a] : full[a];
        final mean2 = xs[a] + gradient[a] / precision2;
        // Drop non-finite expansions (zero weight in the JAX backend).
        if (precision2.isFinite && mean2.isFinite) {
          final j = act[a];
          final slot = nx[j];
          precisions[slot * st + j] = precision2;
          means[slot * st + j] = mean2;
          nx[j] = slot + 1;
        }
      }
    }
    var maxNx = 0;
    for (var j = 0; j < m; j++) {
      if (nx[j] > maxNx) maxNx = nx[j];
    }

    // Variational energy of each expansion (`_Unbounded._energy`).
    for (var x = 0; x < maxNx; x++) {
      final xo = x * st;
      for (var j = 0; j < m; j++) {
        if (x >= nx[j]) continue;
        final dx = means[xo + j] - pm[j];
        energies[xo + j] = -0.5 * pp[j] * dx * dx;
      }
      for (var k = 0; k < maxNc; k++) {
        final ko = k * st;
        for (var j = 0; j < m; j++) {
          if (x >= nx[j] || k >= nc[j]) continue;
          final o = ko + j;
          final logS = logT + coupling[o] * means[xo + j] + tonic[o];
          double logDenom, invDenom;
          if (logS <= _linearLogSMax) {
            final denom = fastExp(logS) + prev[o];
            logDenom = math.log(denom);
            invDenom = 1.0 / denom;
          } else {
            logDenom = logAddExp(math.log(prev[o]), logS);
            invDenom = fastExp(-logDenom);
          }
          energies[xo + j] += -0.5 * logDenom - 0.5 * beAux[o] * invDenom;
        }
      }
    }

    // Softmax weights over the energies, then moment matching.
    final maxEnergy = s.f4, total = s.f5, postMean = s.f6, postVar = s.f7;
    for (var j = 0; j < m; j++) {
      maxEnergy[j] = double.negativeInfinity;
      total[j] = 0.0;
      postMean[j] = 0.0;
      postVar[j] = 0.0;
    }
    for (var x = 0; x < maxNx; x++) {
      final xo = x * st;
      for (var j = 0; j < m; j++) {
        if (x >= nx[j]) continue;
        final en = energies[xo + j];
        if (en > maxEnergy[j]) maxEnergy[j] = en;
      }
    }
    for (var x = 0; x < maxNx; x++) {
      final xo = x * st;
      for (var j = 0; j < m; j++) {
        if (x >= nx[j]) continue;
        // exp(±0) is exactly 1: skip it for the maximum.
        final d = energies[xo + j] - maxEnergy[j];
        final u = d == 0.0 ? 1.0 : fastExp(d);
        energies[xo + j] = u;
        total[j] += u;
      }
    }
    for (var x = 0; x < maxNx; x++) {
      final xo = x * st;
      for (var j = 0; j < m; j++) {
        if (x >= nx[j]) continue;
        final weight = energies[xo + j] / total[j];
        energies[xo + j] = weight;
        postMean[j] += weight * means[xo + j];
      }
    }
    for (var x = 0; x < maxNx; x++) {
      final xo = x * st;
      for (var j = 0; j < m; j++) {
        if (x >= nx[j]) continue;
        final dm = means[xo + j] - postMean[j];
        postVar[j] += energies[xo + j] * (1.0 / precisions[xo + j] + dm * dm);
      }
    }

    final flags = s.flags;
    _bObservedChildren(i, lo, m, flags);
    for (var j = 0; j < m; j++) {
      final ie = base + j;
      final p = flags[j] != 0 ? 1.0 / postVar[j] : expectedPrecision[ie];
      precision[ie] = _minNan(p, maxPosteriorPrecision);
      mean[ie] = postMean[j];
    }
  }
}
