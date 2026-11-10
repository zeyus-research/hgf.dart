part of '../network.dart';

// Forward-mode tangents through the batched update steps
// (`BatchNetwork.tangents`). Each `_t*` kernel computes what its `_b*`
// counterpart in batch_updates.dart computes, with the same arithmetic in
// the same order (so a batch with tangents stays bit-identical to one
// without; `test/tangent_test.dart` checks it), and alongside every value it
// writes, that value's derivative along each of the k directions.
//
// Layout: tangents are stored `[node][direction][element]`: direction d of
// node i, element e at `(i * k + d) * B + e`; of edge strength k_ at
// `(k_ * k + d) * B + e`; scratch tangents `d * stride + j` (element
// `lo + j` of the block); slot s of the unbounded update's slot arrays at
// `(s * k + d) * stride + j`.
//
// Stage-wise like the primal kernels: a stage computes the primal values
// and, per element, the partial derivatives of its outputs (`_TangentScratch`
// coefficient arrays), then each direction is a tight loop over the block's
// elements combining the inputs' tangents with them. A per-element loop over
// the directions would cost a loop per element per stage, and dominated the
// time in an earlier version.
//
// The derivatives are those of the function as computed: a branch passes the
// derivative of the side taken, an active clamp (floors, precision clipping,
// `_maxNan`/`_minNan`) passes none, and Lambert W contributes
// `W' = W / (z (1 + W))` rather than the derivative of its iterations.
// Where a predicted variance V can reach 1e300, tangents are formed as
// `π̂ (π̂ dV)` or `(dx − (x / V) dV) / V`, never through `1 / V²`, which
// underflows.

/// Tangent scratch for one block.
final class _TangentScratch {
  _TangentScratch(int stride, int maxChildren, int k)
    : g0 = Float64List(stride * k),
      g1 = Float64List(stride * k),
      g2 = Float64List(stride * k),
      g3 = Float64List(stride * k),
      g4 = Float64List(stride * k),
      g5 = Float64List(stride * k),
      g6 = Float64List(stride * k),
      g7 = Float64List(stride * k),
      g8 = Float64List(stride * k),
      g9 = Float64List(stride * k),
      g10 = Float64List(stride * k),
      g11 = Float64List(stride * k),
      c0 = Float64List(stride),
      c1 = Float64List(stride),
      c2 = Float64List(stride),
      c3 = Float64List(stride),
      c4 = Float64List(stride),
      c5 = Float64List(stride),
      c6 = Float64List(stride),
      c7 = Float64List(stride),
      c8 = Float64List(stride),
      c9 = Float64List(stride),
      row0 = Float64List(k),
      dCoupling = Float64List(maxChildren * stride * k),
      dTonic = Float64List(maxChildren * stride * k),
      dPrev = Float64List(maxChildren * stride * k),
      dBeAux = Float64List(maxChildren * stride * k),
      dVope = Float64List(maxChildren * stride * k),
      dPrecisions = Float64List((maxChildren + 1) * stride * k),
      dMeans = Float64List((maxChildren + 1) * stride * k),
      dEnergies = Float64List((maxChildren + 1) * stride * k);

  /// Tangents of `_BatchScratch.f0` ... `f11` (each kernel names its uses),
  /// `d * stride + j`.
  final Float64List g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11;

  /// Per-element partial derivatives of a stage.
  final Float64List c0, c1, c2, c3, c4, c5, c6, c7, c8, c9;

  /// One value's tangents (the unbounded update's per-element helper).
  final Float64List row0;

  /// Tangents of the unbounded update's slot arrays.
  final Float64List dCoupling, dTonic, dPrev, dBeAux, dVope;
  final Float64List dPrecisions, dMeans, dEnergies;
}

extension on BatchNetwork {
  /// `_propagate` with tangents.
  void _tPropagate(
    Float64List data,
    Float64List? observed,
    int offset,
    int observedOffset,
    double timeStep,
    bool perElement,
    int lo,
    int hi, {
    bool predicted = false,
  }) {
    if (!predicted) _tPredictSweep(timeStep, lo, hi);
    _observe(data, observed, offset, observedOffset, perElement, lo, hi);
    // Observations don't depend on the parameters.
    final inputs = _inputs, b = _b, k = _k, dMean = _dMean;
    for (var q = 0; q < inputs.length; q++) {
      final node = inputs[q];
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        final t0 = (node * k + d) * b;
        dMean.fillRange(t0 + lo, t0 + hi, 0.0);
      }
    }
    final updNode = _updNode, updStep = _updStep;
    for (var s = 0; s < updNode.length; s++) {
      _tDispatch(updStep[s], updNode[s], timeStep, lo, hi);
    }
  }

  /// The prediction sweep with tangents, for elements `[lo, hi)`.
  void _tPredictSweep(double timeStep, int lo, int hi) {
    final predNode = _predNode, predStep = _predStep;
    for (var s = 0; s < predNode.length; s++) {
      _tDispatch(predStep[s], predNode[s], timeStep, lo, hi);
    }
  }

  void _tDispatch(int step, int node, double t, int lo, int hi) {
    switch (step) {
      case _kPredictionContinuous:
        _tPredictionContinuous(node, t, lo, hi, false);
      case _kPredictionContinuousMeanField:
        _tPredictionContinuous(node, t, lo, hi, true);
      case _kPredictionBinary:
        _tPredictionBinary(node, lo, hi);
      case _kPosteriorContinuous:
        _tPosteriorContinuous(node, lo, hi, false);
      case _kPosteriorContinuousMeanField:
        _tPosteriorContinuous(node, lo, hi, true);
      case _kPosteriorContinuousEhgf:
        _tPosteriorContinuousEhgf(node, t, lo, hi, false);
      case _kPosteriorContinuousEhgfMeanField:
        _tPosteriorContinuousEhgf(node, t, lo, hi, true);
      case _kPosteriorContinuousUnbounded:
        _tPosteriorContinuousUnbounded(node, t, lo, hi);
      case _kPredictionErrorContinuous:
        _tPredictionErrorContinuous(node, lo, hi);
      case _kPredictionErrorBinary:
        _tPredictionErrorBinary(node, lo, hi);
      default:
        _tDispatchVector(step, node, lo, hi);
    }
  }

  // ---------------------------------------------------------------------------
  // Predictions
  // ---------------------------------------------------------------------------

  /// `_bPredictionContinuous` / `_bPredictionContinuousMeanField`.
  @pragma('vm:unsafe:no-bounds-checks')
  void _tPredictionContinuous(int i, double t, int lo, int hi, bool meanField) {
    final nb = _b, m = hi - lo, base = i * nb + lo, k = _k;
    final s = _s, ts = _ts, st = s.stride;
    final drift = s.f0, vcv = s.f1, tv = s.f2;
    final dDrift = ts.g0, dVcv = ts.g1, dTv = ts.g2;
    final c0 = ts.c0, c1 = ts.c1, c2 = ts.c2, c3 = ts.c3, c4 = ts.c4;
    final c5 = ts.c5, c6 = ts.c6;
    final expectedMean = _expectedMean, expectedPrecision = _expectedPrecision;
    final dExpectedMean = _dExpectedMean,
        dExpectedPrecision = _dExpectedPrecision;
    final tonicDrift = _tonicDrift, tonicVolatility = _tonicVolatility;
    for (var j = 0; j < m; j++) {
      final ie = base + j;
      drift[j] = tonicDrift[ie];
      vcv[j] = 0.0;
      tv[j] = tonicVolatility[ie];
    }
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      final ti = (i * k + d) * nb + lo, so = d * st;
      dDrift.setRange(so, so + m, _dTonicDrift, ti);
      dTv.setRange(so, so + m, _dTonicVolatility, ti);
      if (!meanField) dVcv.fillRange(so, so + m, 0.0);
    }

    // Value parents: drift += α g(μ̂ₚ); vcv += (t α g'(μ̂ₚ))² / π̂ₚ.
    final vpK = _vpK, dVpK = _dVpK;
    for (var e = _vpStart[i], end = _vpStart[i + 1]; e < end; e++) {
      final p = _vpIdx[e];
      final pb = p * nb + lo, kb = e * nb + lo;
      final cf = _couplingFn[p];
      final fn = cf == 0 ? null : _couplingOf(cf, p);
      for (var j = 0; j < m; j++) {
        final alpha = vpK[kb + j];
        final pMean = expectedMean[pb + j];
        double g, g1, g2;
        if (fn == null) {
          g = pMean;
          g1 = 1.0;
          g2 = 0.0;
        } else {
          g = fn.f(pMean);
          g1 = fn.df(pMean);
          g2 = fn.d2f(pMean);
        }
        drift[j] += alpha * g;
        c0[j] = g; // ∂drift/∂α
        c1[j] = alpha * g1; // ∂drift/∂μ̂ₚ
        if (meanField) continue;
        final pPrec = expectedPrecision[pb + j];
        final coeff = fn == null ? t * alpha : t * alpha * g1;
        vcv[j] += coeff * coeff / pPrec;
        // d(c² / π̂) = (2c dc − (c² / π̂) dπ̂) / π̂, dc = t (dα g' + α g'' dμ̂).
        final invP = 1.0 / pPrec;
        final cDc = 2.0 * coeff * t * invP;
        c2[j] = cDc * g1; // ∂vcv/∂α
        c3[j] = cDc * alpha * g2; // ∂vcv/∂μ̂ₚ
        c4[j] = coeff * coeff * invP * invP; // −∂vcv/∂π̂ₚ
      }
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        final pt = (p * k + d) * nb + lo, et = (e * k + d) * nb + lo;
        final so = d * st;
        if (_zeroVpK) {
          for (var j = 0; j < m; j++) {
            final dm = dExpectedMean[pt + j];
            dDrift[so + j] += c1[j] * dm;
            if (!meanField) {
              dVcv[so + j] += c3[j] * dm - c4[j] * dExpectedPrecision[pt + j];
            }
          }
        } else if (meanField) {
          for (var j = 0; j < m; j++) {
            dDrift[so + j] +=
                c0[j] * dVpK[et + j] + c1[j] * dExpectedMean[pt + j];
          }
        } else {
          for (var j = 0; j < m; j++) {
            final da = dVpK[et + j], dm = dExpectedMean[pt + j];
            dDrift[so + j] += c0[j] * da + c1[j] * dm;
            dVcv[so + j] +=
                c2[j] * da + c3[j] * dm - c4[j] * dExpectedPrecision[pt + j];
          }
        }
      }
    }

    // Volatility parents: ω̃ += κ μ̂ₚ (+ κ² / 2π̂ₚ).
    final volpK = _volpK, dVolpK = _dVolpK;
    final volStart = _volpStart[i], volEnd = _volpStart[i + 1];
    for (var e = volStart; e < volEnd; e++) {
      final p = _volpIdx[e];
      final pb = p * nb + lo, kb = e * nb + lo;
      for (var j = 0; j < m; j++) {
        final kappa = volpK[kb + j];
        final pMean = expectedMean[pb + j];
        tv[j] += kappa * pMean;
        c0[j] = pMean; // ∂ω̃/∂κ
        c1[j] = 0.0; // −∂ω̃/∂π̂ₚ
        if (!meanField) {
          final pPrec = expectedPrecision[pb + j];
          tv[j] += (kappa * kappa) / (2.0 * pPrec);
          final invP = 1.0 / pPrec;
          c0[j] += kappa * invP;
          c1[j] = 0.5 * kappa * kappa * invP * invP;
        }
      }
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        final pt = (p * k + d) * nb + lo, et = (e * k + d) * nb + lo;
        final so = d * st;
        for (var j = 0; j < m; j++) {
          dTv[so + j] +=
              volpK[kb + j] * dExpectedMean[pt + j] -
              c1[j] * dExpectedPrecision[pt + j];
        }
        if (!_zeroVolpK) {
          for (var j = 0; j < m; j++) {
            dTv[so + j] += c0[j] * dVolpK[et + j];
          }
        }
      }
    }

    final isInput = _isLeaf[i] == 1;
    final fixedLeaf = isInput && volStart == volEnd;
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
      final pvRaw = t * fastExp(tv[j]);
      final pvOk = pvRaw > 1e-128;
      final predictedVolatility = pvOk ? pvRaw : double.nan;
      final carriedVariance = isInput ? 1.0 / prec : (lambda * lambda) / prec;
      final vc = carriedVariance + predictedVolatility;
      double conditional, expected;
      var ve = vc;
      if (meanField) {
        expected = 1.0 / floorPredictedVariance(vc);
        conditional = expected;
      } else {
        conditional = 1.0 / floorPredictedVariance(vc);
        ve = carriedVariance + predictedVolatility + vcv[j];
        expected = 1.0 / floorPredictedVariance(ve);
      }
      currentVariance[ie] = carriedVariance;
      expectedMean[ie] = expMean;
      effectivePrecision[ie] = predictedVolatility * expected;
      if (!fixedLeaf) {
        expectedPrecision[ie] = expected;
        conditionalExpectedPrecision[ie] = conditional;
      } else {
        expectedPrecision[ie] = prec;
        conditionalExpectedPrecision[ie] = prec;
      }
      // dCarried = c0 dπ + c1 dλ; dpv = c2 dω̃; the floors (c3, c4) pass
      // the derivative unless active.
      final invPrec = 1.0 / prec;
      c0[j] = isInput
          ? -invPrec * invPrec
          : -(lambda * invPrec) * (lambda * invPrec);
      c1[j] = isInput ? 0.0 : 2.0 * lambda * invPrec;
      c2[j] = pvOk ? pvRaw : double.nan;
      c3[j] = vc <= minPredictedVariance ? 0.0 : 1.0;
      c4[j] = ve <= minPredictedVariance ? 0.0 : 1.0;
      c5[j] = expected;
      c6[j] = conditional;
    }
    final dMean = _dMean,
        dPrecision = _dPrecision,
        dAutoconnection = _dAutoconnectionStrength,
        dCurrentVariance = _dCurrentVariance,
        dEffectivePrecision = _dEffectivePrecision,
        dConditional = _dConditionalExpectedPrecision;
    final zeroAuto = _zeroAuto;
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      final ti = (i * k + d) * nb + lo, so = d * st;
      for (var j = 0; j < m; j++) {
        final o = ti + j, ie = base + j;
        final dp = dPrecision[o], dl = zeroAuto ? 0.0 : dAutoconnection[o];
        final dCarried = zeroAuto ? c0[j] * dp : c0[j] * dp + c1[j] * dl;
        final dPv = c2[j] * dTv[so + j];
        final dVc = c3[j] * (dCarried + dPv);
        final dVe = meanField ? dVc : c4[j] * (dCarried + dPv + dVcv[so + j]);
        final expected = c5[j];
        dCurrentVariance[o] = dCarried;
        final dEm = autoconnection[ie] * dMean[o] + t * dDrift[so + j];
        dExpectedMean[o] = zeroAuto ? dEm : dl * mean[ie] + dEm;
        dEffectivePrecision[o] =
            expected * (dPv - effectivePrecision[ie] * dVe);
        if (!fixedLeaf) {
          dExpectedPrecision[o] = -expected * (expected * dVe);
          final conditional = c6[j];
          dConditional[o] = -conditional * (conditional * dVc);
        } else {
          dExpectedPrecision[o] = dp;
          dConditional[o] = dp;
        }
      }
    }
  }

  @pragma('vm:unsafe:no-bounds-checks')
  void _tPredictionBinary(int i, int lo, int hi) {
    final nb = _b, m = hi - lo, base = i * nb + lo, k = _k;
    final st = _s.stride, ts = _ts;
    final x = _s.f0, dx = ts.g0, c0 = ts.c0, c1 = ts.c1;
    final expectedMean = _expectedMean, dExpectedMean = _dExpectedMean;
    for (var j = 0; j < m; j++) {
      x[j] = 0.0;
    }
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      dx.fillRange(d * st, d * st + m, 0.0);
    }
    for (var e = _vpStart[i], end = _vpStart[i + 1]; e < end; e++) {
      final p = _vpIdx[e];
      final pb = p * nb + lo;
      for (var j = 0; j < m; j++) {
        x[j] += expectedMean[pb + j];
      }
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        final pt = (p * k + d) * nb + lo, so = d * st;
        for (var j = 0; j < m; j++) {
          dx[so + j] += dExpectedMean[pt + j];
        }
      }
    }
    final v = precisionClippingValue;
    final expectedPrecision = _expectedPrecision;
    final dExpectedPrecision = _dExpectedPrecision;
    for (var j = 0; j < m; j++) {
      final raw = sigmoid(x[j]);
      var p = raw;
      var clipped = false;
      if (p < v) {
        p = v;
        clipped = true;
      } else if (p > 1.0 - v) {
        p = 1.0 - v;
        clipped = true;
      }
      expectedMean[base + j] = p;
      expectedPrecision[base + j] = p * (1.0 - p);
      final slope = clipped ? 0.0 : raw * (1.0 - raw);
      c0[j] = slope;
      c1[j] = (1.0 - 2.0 * p) * slope;
    }
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      final ti = (i * k + d) * nb + lo, so = d * st;
      for (var j = 0; j < m; j++) {
        final dxj = dx[so + j];
        dExpectedMean[ti + j] = c0[j] * dxj;
        dExpectedPrecision[ti + j] = c1[j] * dxj;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Prediction errors
  // ---------------------------------------------------------------------------

  @pragma('vm:unsafe:no-bounds-checks')
  void _tPredictionErrorContinuous(int i, int lo, int hi) {
    final nb = _b, m = hi - lo, base = i * nb + lo, k = _k;
    final isLeaf = _isLeaf[i] == 1;
    final ts = _ts, c0 = ts.c0, c1 = ts.c1, c2 = ts.c2;
    final flags = _s.flags;
    if (!isLeaf) _bObservedChildren(i, lo, m, flags);
    final mean = _mean,
        expectedMean = _expectedMean,
        expectedPrecision = _expectedPrecision,
        precision = _precision,
        vape = _valuePredictionError,
        vope = _volatilityPredictionError,
        observed = _observed;
    for (var j = 0; j < m; j++) {
      final ie = base + j;
      final ep = expectedPrecision[ie];
      final prec = precision[ie];
      final delta = mean[ie] - expectedMean[ie];
      vape[ie] = delta;
      vope[ie] = (ep / prec) + ep * delta * delta - 1.0;
      if (!isLeaf) observed[ie] = flags[j] != 0 ? 1.0 : 0.0;
      // dν = (1/π + δ²) dπ̂ − (π̂/π²) dπ + 2 π̂ δ dδ.
      final invPrec = 1.0 / prec;
      c0[j] = invPrec + delta * delta;
      c1[j] = ep * invPrec * invPrec;
      c2[j] = 2.0 * ep * delta;
    }
    final dMean = _dMean,
        dExpectedMean = _dExpectedMean,
        dExpectedPrecision = _dExpectedPrecision,
        dPrecision = _dPrecision,
        dVape = _dValuePredictionError,
        dVope = _dVolatilityPredictionError;
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      final ti = (i * k + d) * nb + lo;
      for (var j = 0; j < m; j++) {
        final o = ti + j;
        final dDelta = dMean[o] - dExpectedMean[o];
        dVape[o] = dDelta;
        dVope[o] =
            c0[j] * dExpectedPrecision[o] -
            c1[j] * dPrecision[o] +
            c2[j] * dDelta;
      }
    }
  }

  @pragma('vm:unsafe:no-bounds-checks')
  void _tPredictionErrorBinary(int i, int lo, int hi) {
    final nb = _b, m = hi - lo, base = i * nb + lo, k = _k;
    final c0 = _ts.c0, c1 = _ts.c1;
    final expectedPrecision = _expectedPrecision, precision = _precision;
    for (var j = 0; j < m; j++) {
      final ie = base + j;
      final ep = expectedPrecision[ie];
      final obs = _observed[ie];
      final vape = (_mean[ie] - _expectedMean[ie]) * obs / ep;
      _valuePredictionError[ie] = vape;
      precision[ie] = ep;
      final invEp = 1.0 / ep;
      c0[j] = obs * invEp;
      c1[j] = vape * invEp;
    }
    final dMean = _dMean,
        dExpectedMean = _dExpectedMean,
        dExpectedPrecision = _dExpectedPrecision,
        dPrecision = _dPrecision,
        dVape = _dValuePredictionError;
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      final ti = (i * k + d) * nb + lo;
      for (var j = 0; j < m; j++) {
        final o = ti + j;
        final dEp = dExpectedPrecision[o];
        dVape[o] = c0[j] * (dMean[o] - dExpectedMean[o]) - c1[j] * dEp;
        dPrecision[o] = dEp;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Posterior building blocks
  // ---------------------------------------------------------------------------

  /// Which precision `_bRelaxedGain` / `_bRelaxedEffective` read for child
  /// [c]: 0 its expected precision, 1 its conditional expected precision,
  /// 2 the relaxed combination of both with its precision.
  @pragma('vm:prefer-inline')
  int _relaxedMode(int c, bool meanField) =>
      meanField || _kind[c] != _kindContinuous ? 0 : (_isLeaf[c] == 1 ? 1 : 2);

  /// `dOut[d * stride + j] += qE[j] dπ̂ + qC[j] dπ̂c + qP[j] dπ` of child [c]
  /// (expected, conditional expected and posterior precision) for each
  /// direction, reading only the precisions [mode] involves (mode 0: π̂
  /// with qE, mode 1: π̂c with qC, mode 2: all three).
  @pragma('vm:unsafe:no-bounds-checks')
  void _tAddChildPrecisions(
    int c,
    int lo,
    int m,
    int mode,
    Float64List qE,
    Float64List qC,
    Float64List qP,
    Float64List dOut,
  ) {
    final nb = _b, k = _k, st = _s.stride;
    final dEp = _dExpectedPrecision,
        dCond = _dConditionalExpectedPrecision,
        dPrec = _dPrecision;
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      final ct = (c * k + d) * nb + lo, so = d * st;
      if (mode == 0) {
        for (var j = 0; j < m; j++) {
          dOut[so + j] += qE[j] * dEp[ct + j];
        }
      } else if (mode == 1) {
        for (var j = 0; j < m; j++) {
          dOut[so + j] += qC[j] * dCond[ct + j];
        }
      } else {
        for (var j = 0; j < m; j++) {
          dOut[so + j] +=
              qE[j] * dEp[ct + j] +
              qC[j] * dCond[ct + j] +
              qP[j] * dPrec[ct + j];
        }
      }
    }
  }

  /// The relaxed gain G of child element [ce] (`_bRelaxedGain`) and, in
  /// mode 2, its partials `(∂G/∂π̂, ∂G/∂π̂c, ∂G/∂π)` scaled by [scale] into
  /// `qE[j]`, `qC[j]`, `qP[j]` (in modes 0 and 1, `scale` goes to the one
  /// precision G is).
  @pragma('vm:prefer-inline')
  @pragma('wasm:prefer-inline')
  @pragma('vm:unsafe:no-bounds-checks')
  void _tGainPartials(
    int mode,
    int ce,
    double gain,
    double scale,
    int j,
    Float64List qE,
    Float64List qC,
    Float64List qP,
  ) {
    if (mode == 0) {
      qE[j] = scale;
    } else if (mode == 1) {
      qC[j] = scale;
    } else {
      final cep = _expectedPrecision[ce];
      final cc = _conditionalExpectedPrecision[ce], cp = _precision[ce];
      final s = scale / (cc + (cp - cep));
      qC[j] = (cp - gain) * s;
      qP[j] = (cc - gain) * s;
      qE[j] = gain * s;
    }
  }

  /// `_bValuePrecisionWpe`, with tangents in [dOut] (`d * stride + j`).
  @pragma('vm:unsafe:no-bounds-checks')
  void _tValuePrecisionWpe(
    int i,
    int lo,
    int m,
    bool meanField,
    Float64List out,
    Float64List dOut,
  ) {
    final k = _k, st = _s.stride, ts = _ts;
    for (var j = 0; j < m; j++) {
      out[j] = 0.0;
    }
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      dOut.fillRange(d * st, d * st + m, 0.0);
    }
    final start = _vcStart[i], end = _vcStart[i + 1];
    if (start == end) return;
    final nb = _b, base = i * nb + lo;
    final cf = _couplingFn[i];
    final fn = cf == 0 ? null : _couplingOf(cf, i);
    final mean = _mean, vcK = _vcK, observed = _observed;
    final ep = _expectedPrecision,
        cond = _conditionalExpectedPrecision,
        prec = _precision;
    final dMean = _dMean, dVcK = _dVcK, dVape = _dValuePredictionError;
    // Partials of the increment: on the child's three precisions (via E and
    // G), on α, on the parent's mean and on the child's δ.
    final qE = ts.c0, qC = ts.c1, qP = ts.c2;
    final cA = ts.c3, cM = ts.c4, cV = ts.c5;
    final gE = ts.c6, gC = ts.c7, gP = ts.c8;
    for (var e = start; e < end; e++) {
      final c = _vcIdx[e];
      final cb = c * nb + lo, kb = e * nb + lo;
      final mode = _relaxedMode(c, meanField);
      for (var j = 0; j < m; j++) {
        final ce = cb + j;
        final alpha = vcK[kb + j];
        final obs = observed[ce];
        // The relaxed effective precision E and its partials, as
        // `_bRelaxedEffective`.
        final double eff;
        var eE = 0.0, eC = 0.0, eP = 0.0;
        if (mode == 0) {
          eff = ep[ce];
          eE = 1.0;
        } else if (mode == 1) {
          eff = cond[ce];
          eC = 1.0;
        } else {
          final cep = ep[ce], cc = cond[ce];
          final piY = prec[ce] - cep;
          final den = cc + piY;
          eff = cc * piY / den;
          final invDen = 1.0 / den;
          eC = (piY - eff) * invDen;
          eP = (cc - eff) * invDen;
          eE = -eP;
        }
        if (fn == null) {
          const gPrimeSq = 1.0, secondTerm = 0.0;
          out[j] +=
              (eff * alpha * alpha * gPrimeSq -
                  _bRelaxedGain(c, ce, meanField) * secondTerm) *
              obs;
          final a2 = alpha * alpha * obs;
          qE[j] = a2 * eE;
          qC[j] = a2 * eC;
          qP[j] = a2 * eP;
          cA[j] = 2.0 * eff * alpha * obs;
          continue;
        }
        // Nonlinear coupling: (E α² g'² − G α g'' δ) obs.
        final gain = _bRelaxedGain(c, ce, meanField);
        final parentMean = mean[base + j];
        final gp = fn.df(parentMean);
        final g2 = fn.d2f(parentMean);
        final g3 = fn.d3f(parentMean);
        final gPrimeSq = gp * gp;
        final vape = _valuePredictionError[ce];
        final secondTerm = alpha * g2 * vape;
        out[j] += (eff * alpha * alpha * gPrimeSq - gain * secondTerm) * obs;
        gE[j] = 0.0;
        gC[j] = 0.0;
        gP[j] = 0.0;
        _tGainPartials(mode, ce, gain, -secondTerm * obs, j, gE, gC, gP);
        final cE = alpha * alpha * gPrimeSq * obs;
        qE[j] = cE * eE + gE[j];
        qC[j] = cE * eC + gC[j];
        qP[j] = cE * eP + gP[j];
        cA[j] = (2.0 * eff * alpha * gPrimeSq - gain * g2 * vape) * obs;
        cM[j] =
            (eff * alpha * alpha * 2.0 * gp * g2 - gain * alpha * g3 * vape) *
            obs;
        cV[j] = -gain * alpha * g2 * obs;
      }
      _tAddChildPrecisions(c, lo, m, mode, qE, qC, qP, dOut);
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        final et = (e * k + d) * nb + lo, so = d * st;
        if (fn == null) {
          if (_zeroVcK) continue;
          for (var j = 0; j < m; j++) {
            dOut[so + j] += cA[j] * dVcK[et + j];
          }
        } else {
          final it = (i * k + d) * nb + lo, ct = (c * k + d) * nb + lo;
          for (var j = 0; j < m; j++) {
            dOut[so + j] +=
                cA[j] * dVcK[et + j] +
                cM[j] * dMean[it + j] +
                cV[j] * dVape[ct + j];
          }
        }
      }
    }
  }

  /// `_bValueMeanWpe` at the node precision [nodePrecision] (tangents
  /// [dNodePrecision], `d * stride + j`), with tangents in [dOut].
  @pragma('vm:unsafe:no-bounds-checks')
  void _tValueMeanWpe(
    int i,
    int lo,
    int m,
    Float64List nodePrecision,
    Float64List dNodePrecision,
    bool meanField,
    Float64List out,
    Float64List dOut,
  ) {
    final k = _k, st = _s.stride, ts = _ts;
    for (var j = 0; j < m; j++) {
      out[j] = 0.0;
    }
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      dOut.fillRange(d * st, d * st + m, 0.0);
    }
    final start = _vcStart[i], end = _vcStart[i + 1];
    if (start == end) return;
    final nb = _b, base = i * nb + lo;
    final cf = _couplingFn[i];
    final fn = cf == 0 ? null : _couplingOf(cf, i);
    final mean = _mean, vcK = _vcK, observed = _observed;
    final vape = _valuePredictionError;
    final dMean = _dMean, dVcK = _dVcK, dVape = _dValuePredictionError;
    final qE = ts.c0, qC = ts.c1, qP = ts.c2;
    final cA = ts.c3, cM = ts.c4, cP = ts.c5, cV = ts.c6;
    for (var e = start; e < end; e++) {
      final c = _vcIdx[e];
      final cb = c * nb + lo, kb = e * nb + lo;
      final mode = _relaxedMode(c, meanField);
      for (var j = 0; j < m; j++) {
        final ce = cb + j;
        final parentMean = fn == null ? 0.0 : mean[base + j];
        final gPrime = fn == null ? 1.0 : fn.df(parentMean);
        final obs = observed[ce];
        final childVape = vape[ce] * obs;
        final alpha = vcK[kb + j];
        final gain = _bRelaxedGain(c, ce, meanField);
        final p = nodePrecision[j];
        final a = alpha * gPrime * gain / p;
        out[j] += a * childVape;
        // d(α g' G / P · δ obs): partials on α, μ, G, P and δ.
        final invP = 1.0 / p;
        cA[j] = gPrime * gain * invP * childVape;
        cM[j] = fn == null
            ? 0.0
            : alpha * fn.d2f(parentMean) * gain * invP * childVape;
        cP[j] = a * invP * childVape;
        cV[j] = a * obs;
        _tGainPartials(
          mode,
          ce,
          gain,
          alpha * gPrime * invP * childVape,
          j,
          qE,
          qC,
          qP,
        );
      }
      _tAddChildPrecisions(c, lo, m, mode, qE, qC, qP, dOut);
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        final et = (e * k + d) * nb + lo, ct = (c * k + d) * nb + lo;
        final so = d * st;
        for (var j = 0; j < m; j++) {
          dOut[so + j] +=
              cV[j] * dVape[ct + j] - cP[j] * dNodePrecision[so + j];
        }
        if (!_zeroVcK) {
          for (var j = 0; j < m; j++) {
            dOut[so + j] += cA[j] * dVcK[et + j];
          }
        }
        if (fn != null) {
          final it = (i * k + d) * nb + lo;
          for (var j = 0; j < m; j++) {
            dOut[so + j] += cM[j] * dMean[it + j];
          }
        }
      }
    }
  }

  /// `_bVolatilityPrecisionWpe`, accumulating tangents into [dAcc].
  @pragma('vm:unsafe:no-bounds-checks')
  void _tVolatilityPrecisionWpe(
    int i,
    int lo,
    int m,
    Float64List acc,
    Float64List dAcc,
  ) {
    final nb = _b, k = _k, st = _s.stride, ts = _ts;
    final cK = ts.c0, cE = ts.c1, cV = ts.c2;
    final volcK = _volcK,
        eff = _effectivePrecision,
        vope = _volatilityPredictionError,
        observed = _observed;
    final dVolcK = _dVolcK,
        dEff = _dEffectivePrecision,
        dVope = _dVolatilityPredictionError;
    for (var e = _volcStart[i], end = _volcStart[i + 1]; e < end; e++) {
      final c = _volcIdx[e];
      final cb = c * nb + lo, kb = e * nb + lo;
      for (var j = 0; j < m; j++) {
        final ce = cb + j;
        final kappa = volcK[kb + j];
        final ev = eff[ce];
        final v = vope[ce];
        final ke = kappa * ev;
        final obs = observed[ce];
        acc[j] +=
            (0.5 * ke * ke + ke * ke * v - 0.5 * kappa * kappa * ev * v) * obs;
        // A = κ² (e²/2 + e² ν − e ν/2).
        final inner = 0.5 * ev * ev + ev * ev * v - 0.5 * ev * v;
        cK[j] = 2.0 * kappa * inner * obs;
        cE[j] = kappa * kappa * (ev + 2.0 * ev * v - 0.5 * v) * obs;
        cV[j] = kappa * kappa * (ev * ev - 0.5 * ev) * obs;
      }
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        final et = (e * k + d) * nb + lo, ct = (c * k + d) * nb + lo;
        final so = d * st;
        for (var j = 0; j < m; j++) {
          dAcc[so + j] +=
              cK[j] * dVolcK[et + j] +
              cE[j] * dEff[ct + j] +
              cV[j] * dVope[ct + j];
        }
      }
    }
  }

  /// `_bVolatilityMeanWpe` at [nodePrecision] (tangents [dNodePrecision]),
  /// with tangents in [dOut].
  @pragma('vm:unsafe:no-bounds-checks')
  void _tVolatilityMeanWpe(
    int i,
    int lo,
    int m,
    Float64List nodePrecision,
    Float64List dNodePrecision,
    Float64List out,
    Float64List dOut,
  ) {
    final nb = _b, k = _k, st = _s.stride, ts = _ts;
    for (var j = 0; j < m; j++) {
      out[j] = 0.0;
    }
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      dOut.fillRange(d * st, d * st + m, 0.0);
    }
    final start = _volcStart[i], end = _volcStart[i + 1];
    if (start == end) return;
    final cK = ts.c0, cE = ts.c1, cV = ts.c2, cP = ts.c3;
    final volcK = _volcK,
        eff = _effectivePrecision,
        vope = _volatilityPredictionError,
        observed = _observed;
    final dVolcK = _dVolcK,
        dEff = _dEffectivePrecision,
        dVope = _dVolatilityPredictionError;
    for (var e = start; e < end; e++) {
      final c = _volcIdx[e];
      final cb = c * nb + lo, kb = e * nb + lo;
      for (var j = 0; j < m; j++) {
        final ce = cb + j;
        final kappa = volcK[kb + j], ev = eff[ce], v = vope[ce];
        final p = nodePrecision[j];
        final obs = observed[ce];
        final num = kappa * ev * v;
        out[j] += num / (2.0 * p) * obs;
        final s = 0.5 / p * obs;
        cK[j] = ev * v * s;
        cE[j] = kappa * v * s;
        cV[j] = kappa * ev * s;
        cP[j] = num * s / p;
      }
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        final et = (e * k + d) * nb + lo, ct = (c * k + d) * nb + lo;
        final so = d * st;
        for (var j = 0; j < m; j++) {
          dOut[so + j] +=
              cK[j] * dVolcK[et + j] +
              cE[j] * dEff[ct + j] +
              cV[j] * dVope[ct + j] -
              cP[j] * dNodePrecision[so + j];
        }
      }
    }
  }

  /// `_bObservedOrPredicted` for the block: the posterior precision of node
  /// [i]'s elements from the increments [wpe] (tangents [dWpe]) where
  /// `flags[j]` says a child observed, into [out] and its tangents into
  /// [dOut]: with [toState], `out` is the node's state column and `dOut` its
  /// tangent column; otherwise both are block scratch.
  @pragma('vm:unsafe:no-bounds-checks')
  void _tObservedOrPredicted(
    int i,
    int lo,
    int m,
    Uint8List flags,
    Float64List wpe,
    Float64List dWpe,
    Float64List out,
    Float64List dOut,
    bool toState,
  ) {
    final nb = _b, k = _k, st = _s.stride, base = i * nb + lo;
    final c0 = _ts.c0, c1 = _ts.c1;
    final expectedPrecision = _expectedPrecision, dEp = _dExpectedPrecision;
    final ob = toState ? base : 0;
    for (var j = 0; j < m; j++) {
      final ep = expectedPrecision[base + j];
      final hasObs = flags[j] != 0;
      double p;
      var pass = true;
      if (hasObs) {
        final a = ep + wpe[j];
        p = _maxNan(a, minPosteriorPrecision);
        pass = !(a <= minPosteriorPrecision);
      } else {
        p = ep;
      }
      if (!(p.isNaN || p < maxPosteriorPrecision)) pass = false;
      out[ob + j] = _minNan(p, maxPosteriorPrecision);
      c0[j] = pass ? 1.0 : 0.0;
      c1[j] = pass && hasObs ? 1.0 : 0.0;
    }
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      final ti = (i * k + d) * nb + lo, so = d * st;
      final oo = toState ? ti : so;
      for (var j = 0; j < m; j++) {
        dOut[oo + j] = c0[j] * dEp[ti + j] + c1[j] * dWpe[so + j];
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Standard update
  // ---------------------------------------------------------------------------

  @pragma('vm:unsafe:no-bounds-checks')
  void _tPosteriorContinuous(int i, int lo, int hi, bool meanField) {
    final nb = _b, m = hi - lo, base = i * nb + lo, k = _k;
    final s = _s, ts = _ts, st = s.stride;
    final wpe = s.f0, prec = s.f1, valueMean = s.f2, volMean = s.f3;
    final dWpe = ts.g0, dPrec = ts.g1, dValueMean = ts.g2, dVolMean = ts.g3;
    final flags = s.flags;
    _bObservedChildren(i, lo, m, flags);
    _tValuePrecisionWpe(i, lo, m, meanField, wpe, dWpe);
    _tVolatilityPrecisionWpe(i, lo, m, wpe, dWpe);
    _tObservedOrPredicted(i, lo, m, flags, wpe, dWpe, prec, dPrec, false);
    _tValueMeanWpe(i, lo, m, prec, dPrec, meanField, valueMean, dValueMean);
    _tVolatilityMeanWpe(i, lo, m, prec, dPrec, volMean, dVolMean);
    final mean = _mean, precision = _precision, expectedMean = _expectedMean;
    for (var j = 0; j < m; j++) {
      final ie = base + j;
      precision[ie] = prec[j];
      mean[ie] = expectedMean[ie] + (valueMean[j] + volMean[j]);
    }
    final dMean = _dMean,
        dPrecision = _dPrecision,
        dExpectedMean = _dExpectedMean;
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      final ti = (i * k + d) * nb + lo, so = d * st;
      for (var j = 0; j < m; j++) {
        dPrecision[ti + j] = dPrec[so + j];
        dMean[ti + j] =
            dExpectedMean[ti + j] + dValueMean[so + j] + dVolMean[so + j];
      }
    }
  }

  // ---------------------------------------------------------------------------
  // eHGF update
  // ---------------------------------------------------------------------------

  @pragma('vm:unsafe:no-bounds-checks')
  void _tPosteriorContinuousEhgf(
    int i,
    double t,
    int lo,
    int hi,
    bool meanField,
  ) {
    final nb = _b, m = hi - lo, base = i * nb + lo, k = _k;
    final s = _s, ts = _ts, st = s.stride;
    final ep = s.f0, valueMean = s.f1, volMean = s.f2, wpe = s.f3;
    final tonic = s.f4;
    final dEp = ts.g0, dValueMean = ts.g1, dVolMean = ts.g2, dWpe = ts.g3;
    final dTonic = ts.g4;
    final flags = s.flags;
    final mean = _mean, expectedMean = _expectedMean;
    final dMean = _dMean, dExpectedMean = _dExpectedMean;
    for (var j = 0; j < m; j++) {
      ep[j] = _expectedPrecision[base + j];
    }
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      dEp.setRange(
        d * st,
        d * st + m,
        _dExpectedPrecision,
        (i * k + d) * nb + lo,
      );
    }
    // Mean first, at the predicted precision (reading the old mean).
    _tValueMeanWpe(i, lo, m, ep, dEp, meanField, valueMean, dValueMean);
    _tVolatilityMeanWpe(i, lo, m, ep, dEp, volMean, dVolMean);
    for (var j = 0; j < m; j++) {
      final ie = base + j;
      mean[ie] = expectedMean[ie] + valueMean[j] + volMean[j];
    }
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      final ti = (i * k + d) * nb + lo, so = d * st;
      for (var j = 0; j < m; j++) {
        dMean[ti + j] =
            dExpectedMean[ti + j] + dValueMean[so + j] + dVolMean[so + j];
      }
    }

    // Then the precision, re-predicting each volatility child from the new
    // mean.
    _bObservedChildren(i, lo, m, flags);
    _tValuePrecisionWpe(i, lo, m, meanField, wpe, dWpe);
    final mgf = !meanField && ehgfMgfCorrection;
    final volpK = _volpK, dVolpK = _dVolpK, volcK = _volcK, dVolcK = _dVolcK;
    final expectedPrecision = _expectedPrecision, dEP = _dExpectedPrecision;
    final cond = _conditionalExpectedPrecision;
    final dCond = _dConditionalExpectedPrecision;
    final ts2 = _ts;
    final kK = ts2.c0, kM = ts2.c1, kT = ts2.c2, kCV = ts2.c3, kEP = ts2.c4;
    final kCo = ts2.c5, kPr = ts2.c6, kDd = ts2.c7;
    for (var e = _volcStart[i], end = _volcStart[i + 1]; e < end; e++) {
      final c = _volcIdx[e];
      final cb = c * nb + lo, kb = e * nb + lo;
      // The child's effective tonic volatility seen from i (its other
      // volatility parents), plus the MGF terms, and their tangents.
      for (var j = 0; j < m; j++) {
        tonic[j] = _tonicVolatility[cb + j];
      }
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        dTonic.setRange(
          d * st,
          d * st + m,
          _dTonicVolatility,
          (c * k + d) * nb + lo,
        );
      }
      for (var q = _volpStart[c], qEnd = _volpStart[c + 1]; q < qEnd; q++) {
        final p = _volpIdx[q];
        if (p == i) continue;
        final pb = p * nb + lo, qb = q * nb + lo;
        for (var j = 0; j < m; j++) {
          tonic[j] += volpK[qb + j] * expectedMean[pb + j];
        }
        for (var ad = 0; ad < _nActive; ad++) {
          final d = _activeDirs[ad];
          final pt = (p * k + d) * nb + lo, qt = (q * k + d) * nb + lo;
          final so = d * st;
          for (var j = 0; j < m; j++) {
            dTonic[so + j] +=
                dVolpK[qt + j] * expectedMean[pb + j] +
                volpK[qb + j] * dExpectedMean[pt + j];
          }
        }
      }
      if (mgf) {
        // `_bMgfTerms`, summed per element before it is added.
        final mgfSum = s.f5;
        for (var j = 0; j < m; j++) {
          mgfSum[j] = 0.0;
        }
        for (var q = _volpStart[c], qEnd = _volpStart[c + 1]; q < qEnd; q++) {
          final p = _volpIdx[q];
          final pb = p * nb + lo, qb = q * nb + lo;
          for (var j = 0; j < m; j++) {
            final kappa = volpK[qb + j], pp = expectedPrecision[pb + j];
            mgfSum[j] += kappa * kappa / (2.0 * pp);
          }
          for (var ad = 0; ad < _nActive; ad++) {
            final d = _activeDirs[ad];
            final pt = (p * k + d) * nb + lo, qt = (q * k + d) * nb + lo;
            final so = d * st;
            for (var j = 0; j < m; j++) {
              final kappa = volpK[qb + j], pp = expectedPrecision[pb + j];
              final invP = 1.0 / pp;
              dTonic[so + j] +=
                  kappa * invP * dVolpK[qt + j] -
                  0.5 * kappa * kappa * invP * invP * dEP[pt + j];
            }
          }
        }
        for (var j = 0; j < m; j++) {
          tonic[j] += mgfSum[j];
        }
      }
      // `_bEhgfVolatilityIncrement` and the partials of the increment.
      for (var j = 0; j < m; j++) {
        final ce = cb + j;
        final kappa = volcK[kb + j];
        final nodeMean = mean[base + j];
        final cep = expectedPrecision[ce], cc = cond[ce];
        final vcvRaw = 1.0 / cep - 1.0 / cc;
        final previousVariance = _currentVariance[ce] + _maxNan(vcvRaw, 0.0);
        final pv = t * fastExp(kappa * nodeMean + tonic[j]);
        final ex = 1.0 / (previousVariance + pv);
        final eff = pv * ex;
        final weight = (pv - previousVariance) * ex;
        final dd = _mean[ce] - _expectedMean[ce];
        final prec = _precision[ce];
        final vope = (1.0 / prec + dd * dd) * ex - 1.0;
        final inc = 0.5 * kappa * kappa * eff * (eff + weight * vope);
        final obs = _observed[ce];
        wpe[j] += _maxNan(inc, 0.0) * obs;
        // dInc = κ eff inner dκ + ½κ² ((inner + eff) dEff + eff ν dW
        //        + eff w dν), with dEff = E (dpv − eff dV),
        // dW = E (dpv − dPrev − w dV), dν = E (dB − (ν + 1) dV), dV = dPrev
        // + dpv: collected on dpv, dPrev, dπ and dδ.
        final g = (inc.isNaN || inc > 0.0) ? obs : 0.0;
        final inner = eff + weight * vope;
        final half = 0.5 * kappa * kappa;
        final a = half * (inner + eff), bw = half * eff * vope;
        final cv = half * eff * weight;
        final z = ex * (a * eff + bw * weight + cv * (vope + 1.0));
        final cPv = (ex * (a + bw) - z) * g;
        final cPrev = (-ex * bw - z) * g;
        kK[j] = kappa * eff * inner * g + cPv * pv * nodeMean;
        kM[j] = cPv * pv * kappa;
        kT[j] = cPv * pv;
        kCV[j] = cPrev;
        final passV = !(vcvRaw <= 0.0);
        kEP[j] = passV ? -cPrev / (cep * cep) : 0.0;
        kCo[j] = passV ? cPrev / (cc * cc) : 0.0;
        kPr[j] = -cv * ex * g / (prec * prec);
        kDd[j] = 2.0 * cv * ex * dd * g;
      }
      final dCurrentVariance = _dCurrentVariance, dPrecision = _dPrecision;
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        final ct = (c * k + d) * nb + lo, it = (i * k + d) * nb + lo;
        final et = (e * k + d) * nb + lo, so = d * st;
        for (var j = 0; j < m; j++) {
          dWpe[so + j] +=
              kK[j] * dVolcK[et + j] +
              kM[j] * dMean[it + j] +
              kT[j] * dTonic[so + j] +
              kCV[j] * dCurrentVariance[ct + j] +
              kEP[j] * dEP[ct + j] +
              kCo[j] * dCond[ct + j] +
              kPr[j] * dPrecision[ct + j] +
              kDd[j] * (dMean[ct + j] - dExpectedMean[ct + j]);
        }
      }
    }
    _tObservedOrPredicted(
      i,
      lo,
      m,
      flags,
      wpe,
      dWpe,
      _precision,
      _dPrecision,
      true,
    );
  }

  // ---------------------------------------------------------------------------
  // Unbounded update
  // ---------------------------------------------------------------------------

  /// `_bEffectiveTonicVolatility` for absolute element [e], with its
  /// tangents in [out] (length k).
  @pragma('vm:unsafe:no-bounds-checks')
  double _tEffectiveTonicVolatility(int i, int c, int e, Float64List out) {
    final nb = _b, k = _k;
    var tv = _tonicVolatility[c * nb + e];
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      out[d] = _dTonicVolatility[(c * k + d) * nb + e];
    }
    for (var q = _volpStart[c], end = _volpStart[c + 1]; q < end; q++) {
      final p = _volpIdx[q];
      if (p == i) continue;
      final kappa = _volpK[q * nb + e], pMean = _expectedMean[p * nb + e];
      tv += kappa * pMean;
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        out[d] +=
            _dVolpK[(q * k + d) * nb + e] * pMean +
            kappa * _dExpectedMean[(p * k + d) * nb + e];
      }
    }
    return tv;
  }

  @pragma('vm:unsafe:no-bounds-checks')
  void _tPosteriorContinuousUnbounded(int i, double t, int lo, int hi) {
    final nb = _b, m = hi - lo, base = i * nb + lo, k = _k;
    final s = _s, st = s.stride, ts = _ts;
    final logT = t == 1.0 ? 0.0 : math.log(t);
    final em = s.f0, pp = s.f1, pm = s.f2, tmp = s.f3;
    final dEm = ts.g0, dPp = ts.g1, dPm = ts.g2, dTmp = ts.g3;
    final nc = s.nc, nx = s.nx;
    final coupling = s.coupling,
        tonic = s.tonic,
        prev = s.prev,
        beAux = s.beAux,
        vope = s.vope;
    final dCoupling = ts.dCoupling,
        dTonic = ts.dTonic,
        dPrev = ts.dPrev,
        dBeAux = ts.dBeAux,
        dVope = ts.dVope;
    final precisions = s.precisions, means = s.means, energies = s.energies;
    final dPrecisions = ts.dPrecisions,
        dMeans = ts.dMeans,
        dEnergies = ts.dEnergies;
    // Tangent of node i / child c, direction d, block element j.
    int ti(int node, int d, int j) => (node * k + d) * nb + lo + j;
    // Tangent of slot q, direction d, block element j.
    int si(int q, int d, int j) => (q * k + d) * st + j;

    // Effective prior.
    final expectedMean = _expectedMean, expectedPrecision = _expectedPrecision;
    final dExpectedMean = _dExpectedMean,
        dExpectedPrecision = _dExpectedPrecision;
    for (var j = 0; j < m; j++) {
      em[j] = expectedMean[base + j];
    }
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      dEm.setRange(d * st, d * st + m, dExpectedMean, ti(i, d, 0));
    }
    if (_vcStart[i] == _vcStart[i + 1]) {
      for (var j = 0; j < m; j++) {
        pp[j] = expectedPrecision[base + j];
        pm[j] = em[j];
      }
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        dPp.setRange(d * st, d * st + m, dExpectedPrecision, ti(i, d, 0));
        dPm.setRange(d * st, d * st + m, dEm, d * st);
      }
    } else {
      final mf = meanFieldUpdates;
      _tValuePrecisionWpe(i, lo, m, mf, tmp, dTmp);
      final gate = ts.c9;
      for (var j = 0; j < m; j++) {
        final a = expectedPrecision[base + j] + tmp[j];
        pp[j] = _maxNan(a, minPosteriorPrecision);
        gate[j] = a <= minPosteriorPrecision ? 0.0 : 1.0;
      }
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        for (var j = 0; j < m; j++) {
          dPp[d * st + j] =
              gate[j] * (dExpectedPrecision[ti(i, d, j)] + dTmp[d * st + j]);
        }
      }
      _tValueMeanWpe(i, lo, m, pp, dPp, mf, tmp, dTmp);
      for (var j = 0; j < m; j++) {
        pm[j] = em[j] + tmp[j];
      }
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        for (var j = 0; j < m; j++) {
          dPm[d * st + j] = dEm[d * st + j] + dTmp[d * st + j];
        }
      }
    }

    // Observed volatility children, one slot each.
    for (var j = 0; j < m; j++) {
      nc[j] = 0;
    }
    final mean = _mean, precision = _precision, observed = _observed;
    final dMean = _dMean, dPrecision = _dPrecision;
    final volcK = _volcK, currentVariance = _currentVariance;
    final dVolcK = _dVolcK, dCurrentVariance = _dCurrentVariance;
    final cep = _conditionalExpectedPrecision;
    final dCep = _dConditionalExpectedPrecision;
    final rowTonic = ts.row0;
    for (var e = _volcStart[i], end = _volcStart[i + 1]; e < end; e++) {
      final c = _volcIdx[e];
      final cb = c * nb + lo, kb = e * nb + lo;
      for (var j = 0; j < m; j++) {
        final ce = cb + j;
        if (observed[ce] == 0.0) continue;
        final slot = nc[j];
        final o = slot * st + j;
        final dd = mean[ce] - expectedMean[ce];
        final cPrec = precision[ce];
        final b = 1.0 / cPrec + dd * dd;
        coupling[o] = volcK[kb + j];
        tonic[o] = _tEffectiveTonicVolatility(i, c, lo + j, rowTonic);
        final cExp = expectedPrecision[ce], cCond = cep[ce];
        final vcvRaw = 1.0 / cExp - 1.0 / cCond;
        final prevRaw = currentVariance[ce] + _maxNan(vcvRaw, 0.0);
        prev[o] = _maxNan(prevRaw, minPredictedVariance);
        beAux[o] = b;
        vope[o] = cExp * b - 1.0;
        nc[j] = slot + 1;
        final passV = !(vcvRaw <= 0.0);
        final passP = !(prevRaw <= minPredictedVariance);
        for (var ad = 0; ad < _nActive; ad++) {
          final d = _activeDirs[ad];
          final oo = si(slot, d, j), co = ti(c, d, j);
          dCoupling[oo] = dVolcK[(e * k + d) * nb + lo + j];
          dTonic[oo] = rowTonic[d];
          final dVcv = passV
              ? -dExpectedPrecision[co] / (cExp * cExp) +
                    dCep[co] / (cCond * cCond)
              : 0.0;
          dPrev[oo] = passP ? dCurrentVariance[co] + dVcv : 0.0;
          final db =
              -dPrecision[co] / (cPrec * cPrec) +
              2.0 * dd * (dMean[co] - dExpectedMean[co]);
          dBeAux[oo] = db;
          dVope[oo] = dExpectedPrecision[co] * b + cExp * db;
        }
      }
    }
    var maxNc = 0;
    for (var j = 0; j < m; j++) {
      if (nc[j] > maxNc) maxNc = nc[j];
    }

    // Expansion 1: at the prediction, concave part only.
    final precision1 = s.f4, meanShift1 = s.f5;
    final dPrecision1 = ts.g4, dMeanShift1 = ts.g5;
    for (var j = 0; j < m; j++) {
      precision1[j] = pp[j];
      meanShift1[j] = pp[j] * (pm[j] - em[j]);
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        final jo = d * st + j;
        dPrecision1[jo] = dPp[jo];
        dMeanShift1[jo] =
            dPp[jo] * (pm[j] - em[j]) + pp[j] * (dPm[jo] - dEm[jo]);
      }
    }
    for (var q = 0; q < maxNc; q++) {
      final qo = q * st;
      for (var j = 0; j < m; j++) {
        if (q >= nc[j]) continue;
        final o = qo + j;
        final ck = coupling[o];
        final logS = logT + ck * em[j] + tonic[o];
        final w = logS <= _linearLogSMax
            ? _ratio(fastExp(logS), prev[o])
            : sigmoid(logS - math.log(prev[o]));
        precision1[j] += 0.5 * ck * ck * w * (1.0 - w);
        meanShift1[j] += 0.5 * ck * w * vope[o];
        final ww = w * (1.0 - w), pv = prev[o], vo = vope[o];
        for (var ad = 0; ad < _nActive; ad++) {
          final d = _activeDirs[ad];
          final oo = si(q, d, j), jo = d * st + j;
          final dck = dCoupling[oo];
          final dLogS = dck * em[j] + ck * dEm[jo] + dTonic[oo];
          final dw = ww * (dLogS - dPrev[oo] / pv);
          dPrecision1[jo] +=
              ck * dck * ww + 0.5 * ck * ck * dw * (1.0 - 2.0 * w);
          dMeanShift1[jo] +=
              0.5 * (dck * w * vo + ck * dw * vo + ck * w * dVope[oo]);
        }
      }
    }
    for (var j = 0; j < m; j++) {
      final p1 = precision1[j], ms1 = meanShift1[j];
      precisions[j] = p1;
      means[j] = em[j] + ms1 / p1;
      nx[j] = 1;
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        final jo = d * st + j, xo = si(0, d, j);
        dPrecisions[xo] = dPrecision1[jo];
        dMeans[xo] =
            dEm[jo] + (dMeanShift1[jo] - ms1 * dPrecision1[jo] / p1) / p1;
      }
    }

    // Expansions 2..n+1: at each child's Lambert W0 mode.
    final act = s.act;
    final z = s.f4, w0 = s.f5, pihat = s.f6, gamma = s.f7;
    final xs = s.f8, full = s.f9, concave = s.f10, gradient = s.f11;
    final dLogZ = ts.g4, dPihat = ts.g6, dGamma = ts.g7;
    final dXs = ts.g8, dFull = ts.g9, dConcave = ts.g10, dGradient = ts.g11;
    for (var q = 0; q < maxNc; q++) {
      final qo = q * st;
      var na = 0;
      for (var j = 0; j < m; j++) {
        if (q >= nc[j]) continue;
        final o = qo + j;
        final ck = coupling[o];
        if (ck == 0.0) continue;
        final pihatY = pp[j] / (ck * ck);
        final gammaC = logT + ck * pm[j] + tonic[o];
        var wArg = beAux[o] / (2.0 * pihatY) * fastExp(0.5 / pihatY - gammaC);
        var clamped = false;
        if (!(wArg > 0.0 && wArg <= double.maxFinite)) {
          final logWArg =
              math.log(beAux[o]) -
              math.log(2.0 * pihatY) +
              0.5 / pihatY -
              gammaC;
          clamped = !(logWArg.isNaN || logWArg < logMaxFinite);
          wArg = fastExp(_minNan(logWArg, logMaxFinite));
        }
        act[na] = j;
        z[na] = wArg;
        pihat[na] = pihatY;
        gamma[na] = gammaC;
        final b = beAux[o];
        for (var ad = 0; ad < _nActive; ad++) {
          final d = _activeDirs[ad];
          final oo = si(q, d, j), ao = d * st + na, jo = d * st + j;
          final dck = dCoupling[oo];
          final dPh = dPp[jo] / (ck * ck) - 2.0 * pihatY * dck / ck;
          final dG = dck * pm[j] + ck * dPm[jo] + dTonic[oo];
          dPihat[ao] = dPh;
          dGamma[ao] = dG;
          dLogZ[ao] = clamped
              ? 0.0
              : dBeAux[oo] / b -
                    dPh / pihatY -
                    0.5 * dPh / (pihatY * pihatY) -
                    dG;
        }
        na++;
      }
      if (na == 0) continue;
      _lambertW0Block(z, w0, s.done, na);
      for (var a = 0; a < na; a++) {
        final j = act[a];
        final o = qo + j;
        final yStar = gamma[a] + w0[a] - 0.5 / pihat[a];
        final xStar = (yStar - logT - tonic[o]) / coupling[o];
        xs[a] = xStar;
        full[a] = pp[j];
        concave[a] = pp[j];
        gradient[a] = -pp[j] * (xStar - pm[j]);
        final w = w0[a], wScale = w / (1.0 + w), ck = coupling[o];
        final ph = pihat[a];
        for (var ad = 0; ad < _nActive; ad++) {
          final d = _activeDirs[ad];
          final ao = d * st + a, jo = d * st + j, oo = si(q, d, j);
          final dW = wScale * dLogZ[ao];
          final dY = dGamma[ao] + dW + 0.5 * dPihat[ao] / (ph * ph);
          final dX = (dY - dTonic[oo] - xStar * dCoupling[oo]) / ck;
          dXs[ao] = dX;
          dFull[ao] = dPp[jo];
          dConcave[ao] = dPp[jo];
          dGradient[ao] = -dPp[jo] * (xStar - pm[j]) - pp[j] * (dX - dPm[jo]);
        }
      }
      for (var r = 0; r < maxNc; r++) {
        final ro = r * st;
        for (var a = 0; a < na; a++) {
          final j = act[a];
          if (r >= nc[j]) continue;
          final o = ro + j;
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
          final pv = prev[o], b = beAux[o];
          final f = w * (w + (2.0 * w - 1.0) * da), cw = w * (1.0 - w);
          for (var ad = 0; ad < _nActive; ad++) {
            final d = _activeDirs[ad];
            final oo = si(r, d, j), ao = d * st + a;
            final dco = dCoupling[oo];
            final dLogS = dco * xs[a] + co * dXs[ao] + dTonic[oo];
            final rr = dPrev[oo] / pv;
            final dw = cw * (dLogS - rr);
            final dInv = -invDenom * (w * dLogS + (1.0 - w) * rr);
            final dda = dBeAux[oo] * invDenom + b * dInv;
            final df =
                dw * (w + (2.0 * w - 1.0) * da) +
                w * (dw + 2.0 * dw * da + (2.0 * w - 1.0) * dda);
            dFull[ao] += co * dco * f + 0.5 * co * co * df;
            dConcave[ao] +=
                co * dco * cw + 0.5 * co * co * dw * (1.0 - 2.0 * w);
            dGradient[ao] += 0.5 * (dco * w * da + co * dw * da + co * w * dda);
          }
        }
      }
      for (var a = 0; a < na; a++) {
        final useConcave = full[a] <= 0.0;
        final precision2 = useConcave ? concave[a] : full[a];
        final mean2 = xs[a] + gradient[a] / precision2;
        // Drop non-finite expansions (zero weight in the JAX backend).
        if (precision2.isFinite && mean2.isFinite) {
          final j = act[a];
          final slot = nx[j];
          final so = slot * st + j;
          precisions[so] = precision2;
          means[so] = mean2;
          nx[j] = slot + 1;
          final g = gradient[a];
          for (var ad = 0; ad < _nActive; ad++) {
            final d = _activeDirs[ad];
            final ao = d * st + a, xo = si(slot, d, j);
            final dP2 = useConcave ? dConcave[ao] : dFull[ao];
            dPrecisions[xo] = dP2;
            dMeans[xo] =
                dXs[ao] + (dGradient[ao] - g * dP2 / precision2) / precision2;
          }
        }
      }
    }
    var maxNx = 0;
    for (var j = 0; j < m; j++) {
      if (nx[j] > maxNx) maxNx = nx[j];
    }

    // Variational energy of each expansion.
    for (var x = 0; x < maxNx; x++) {
      final xo = x * st;
      for (var j = 0; j < m; j++) {
        if (x >= nx[j]) continue;
        final dx = means[xo + j] - pm[j];
        energies[xo + j] = -0.5 * pp[j] * dx * dx;
        for (var ad = 0; ad < _nActive; ad++) {
          final d = _activeDirs[ad];
          final xjo = si(x, d, j), jo = d * st + j;
          dEnergies[xjo] =
              -0.5 * dPp[jo] * dx * dx - pp[j] * dx * (dMeans[xjo] - dPm[jo]);
        }
      }
      for (var q = 0; q < maxNc; q++) {
        final qo = q * st;
        for (var j = 0; j < m; j++) {
          if (x >= nx[j] || q >= nc[j]) continue;
          final o = qo + j;
          final mx = means[xo + j];
          final logS = logT + coupling[o] * mx + tonic[o];
          double logDenom, invDenom, w;
          if (logS <= _linearLogSMax) {
            final sv = fastExp(logS);
            final denom = sv + prev[o];
            logDenom = math.log(denom);
            invDenom = 1.0 / denom;
            w = sv * invDenom;
          } else {
            logDenom = logAddExp(math.log(prev[o]), logS);
            invDenom = fastExp(-logDenom);
            w = 1.0 - prev[o] * invDenom;
          }
          energies[xo + j] += -0.5 * logDenom - 0.5 * beAux[o] * invDenom;
          final pv = prev[o], b = beAux[o], co = coupling[o];
          for (var ad = 0; ad < _nActive; ad++) {
            final d = _activeDirs[ad];
            final oo = si(q, d, j), xjo = si(x, d, j);
            final dLogS = dCoupling[oo] * mx + co * dMeans[xjo] + dTonic[oo];
            final dLogDenom = w * dLogS + (1.0 - w) * dPrev[oo] / pv;
            final dInv = -invDenom * dLogDenom;
            dEnergies[xjo] +=
                -0.5 * dLogDenom - 0.5 * (dBeAux[oo] * invDenom + b * dInv);
          }
        }
      }
    }

    // Softmax weights over the energies, then moment matching.
    final maxEnergy = s.f4, total = s.f5, postMean = s.f6, postVar = s.f7;
    final dMeanEnergy = ts.g5, dPostMean = ts.g6, dPostVar = ts.g7;
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
    // Weight tangents: ω'ₓ = ωₓ (E'ₓ − Σ_y ω_y E'_y).
    for (var ad = 0; ad < _nActive; ad++) {
      final d = _activeDirs[ad];
      dMeanEnergy.fillRange(d * st, d * st + m, 0.0);
      dPostMean.fillRange(d * st, d * st + m, 0.0);
      dPostVar.fillRange(d * st, d * st + m, 0.0);
    }
    for (var x = 0; x < maxNx; x++) {
      final xo = x * st;
      for (var j = 0; j < m; j++) {
        if (x >= nx[j]) continue;
        final w = energies[xo + j];
        for (var ad = 0; ad < _nActive; ad++) {
          final d = _activeDirs[ad];
          dMeanEnergy[d * st + j] += w * dEnergies[si(x, d, j)];
        }
      }
    }
    for (var x = 0; x < maxNx; x++) {
      final xo = x * st;
      for (var j = 0; j < m; j++) {
        if (x >= nx[j]) continue;
        final w = energies[xo + j], mx = means[xo + j];
        for (var ad = 0; ad < _nActive; ad++) {
          final d = _activeDirs[ad];
          final xjo = si(x, d, j), jo = d * st + j;
          final dw = w * (dEnergies[xjo] - dMeanEnergy[jo]);
          dEnergies[xjo] = dw;
          dPostMean[jo] += dw * mx + w * dMeans[xjo];
        }
      }
    }
    for (var x = 0; x < maxNx; x++) {
      final xo = x * st;
      for (var j = 0; j < m; j++) {
        if (x >= nx[j]) continue;
        final dm = means[xo + j] - postMean[j];
        final p = precisions[xo + j], w = energies[xo + j];
        postVar[j] += w * (1.0 / p + dm * dm);
        for (var ad = 0; ad < _nActive; ad++) {
          final d = _activeDirs[ad];
          final xjo = si(x, d, j), jo = d * st + j;
          dPostVar[jo] +=
              dEnergies[xjo] * (1.0 / p + dm * dm) +
              w *
                  (-dPrecisions[xjo] / (p * p) +
                      2.0 * dm * (dMeans[xjo] - dPostMean[jo]));
        }
      }
    }

    final flags = s.flags;
    _bObservedChildren(i, lo, m, flags);
    for (var j = 0; j < m; j++) {
      final ie = base + j;
      final hasObs = flags[j] != 0;
      final p = hasObs ? 1.0 / postVar[j] : expectedPrecision[ie];
      precision[ie] = _minNan(p, maxPosteriorPrecision);
      mean[ie] = postMean[j];
      final pass = p.isNaN || p < maxPosteriorPrecision;
      for (var ad = 0; ad < _nActive; ad++) {
        final d = _activeDirs[ad];
        final o = ti(i, d, j), jo = d * st + j;
        final dp = hasObs ? -dPostVar[jo] * p * p : dExpectedPrecision[o];
        dPrecision[o] = pass ? dp : 0.0;
        dMean[o] = dPostMean[jo];
      }
    }
  }
}
