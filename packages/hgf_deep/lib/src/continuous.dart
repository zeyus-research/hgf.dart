/// The sweeps of a network of continuous layers, mirroring pyhgf's
/// `pyhgf/updates/vectorised/continuous/` and the continuous half of
/// `pyhgf/utils/vectorised_belief_propagation.py` (`run_continuous_scan`)
/// term for term.
///
/// A continuous network is a filter over one observation at a time, so
/// every state here has width 1. Layers connect through fixed matrices: a
/// layer is the value parent of at most one layer (its [Layer.weights], the
/// drift `W·g(μ̂)` of the child) and the volatility parent of at most one
/// layer (its [Layer.volatilityWeights], κ in the child's log-volatility).
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'config.dart';
import 'kernels.dart';
import 'matrix.dart';
import 'state.dart';

/// `ln(f64::MAX)`, the cap on the Lambert-W argument's logarithm.
final double _logFloatMax = math.log(double.maxFinite);

/// `max(x, lo)` that keeps a NaN `x` (as `jnp.maximum` does).
double _floor(double x, double lo) => x < lo ? lo : x;

/// The topology of a continuous network, inverted once
/// (`_continuous_parent_maps`, `_continuous_update_order`).
final class ContinuousTopology {
  ContinuousTopology._(
    this.valueParentOf,
    this.volatilityParentOf,
    this.updateOrder,
  );

  /// Invert the child indices of [configs] and order the update sweep.
  /// Throws an [ArgumentError] when the noise-parent dependency closes a
  /// cycle.
  factory ContinuousTopology(List<LayerConfig> configs) {
    final n = configs.length;
    final valueParentOf = List<int?>.filled(n, null);
    final volatilityParentOf = List<int?>.filled(n, null);
    for (var j = 0; j < n; j++) {
      final vc = configs[j].valueChild;
      final vlc = configs[j].volatilityChild;
      if (vc != null) valueParentOf[vc] = j;
      if (vlc != null) volatilityParentOf[vlc] = j;
    }
    // A layer is updated after its value and volatility children; the noise
    // parent of the observation layer also after the observation layer's
    // value parent, whose posterior its prediction error reads. Among the
    // ready layers the lowest index goes first.
    final deps = <int, Set<int>>{for (var i = 1; i < n; i++) i: {}};
    for (var i = 1; i < n; i++) {
      for (final c in [configs[i].valueChild, configs[i].volatilityChild]) {
        if (c != null && c != 0) deps[i]!.add(c);
      }
      if (configs[i].volatilityChild == 0 && valueParentOf[0] != null) {
        deps[i]!.add(valueParentOf[0]!);
      }
    }
    final order = <int>[];
    final done = <int>{};
    while (deps.isNotEmpty) {
      final keys = deps.keys.toList()..sort();
      final ready = keys.where((i) => done.containsAll(deps[i]!));
      if (ready.isEmpty) {
        throw ArgumentError(
          'The layers of the continuous network form a cycle.',
        );
      }
      final next = ready.first;
      order.add(next);
      done.add(next);
      deps.remove(next);
    }
    return ContinuousTopology._(
      List.unmodifiable(valueParentOf),
      List.unmodifiable(volatilityParentOf),
      List.unmodifiable(order),
    );
  }

  /// Per layer, the index of its value parent (or `null`).
  final List<int?> valueParentOf;

  /// Per layer, the index of its volatility parent (or `null`).
  final List<int?> volatilityParentOf;

  /// The layers above the observation layer, in update order.
  final List<int> updateOrder;
}

/// `out[a] = Σ_b M(a, b)·v[b]` (or `Σ_b M(a, b)²·v[b]` with [squared]).
void _mv(Float64List out, Matrix m, Float64List v, {bool squared = false}) {
  final cols = m.cols;
  final d = m.data;
  for (var a = 0; a < m.rows; a++) {
    var sum = 0.0;
    final row = a * cols;
    for (var b = 0; b < cols; b++) {
      final w = d[row + b];
      sum += (squared ? w * w : w) * v[b];
    }
    out[a] = sum;
  }
}

/// `out[b] = Σ_a M(a, b)·v[a]` (or with `M(a, b)²`): the transposed product.
void _mtv(Float64List out, Matrix m, Float64List v, {bool squared = false}) {
  final cols = m.cols;
  final d = m.data;
  out.fillRange(0, cols, 0.0);
  for (var a = 0; a < m.rows; a++) {
    final va = v[a];
    final row = a * cols;
    for (var b = 0; b < cols; b++) {
      final w = d[row + b];
      out[b] += (squared ? w * w : w) * va;
    }
  }
}

Float64List _map(Float64List v, double Function(double) f) {
  final out = Float64List(v.length);
  for (var k = 0; k < v.length; k++) {
    out[k] = f(v[k]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

/// Predict every layer from its parents, top index first
/// (`_continuous_prediction_sweep`).
void continuousPredictionSweep(
  List<Layer> layers,
  List<LayerState> states,
  ContinuousTopology topo,
  double timeStep,
  bool meanField,
) {
  for (var i = layers.length - 1; i >= 0; i--) {
    final vp = topo.valueParentOf[i];
    final vlp = topo.volatilityParentOf[i];
    _predict(
      layers[i],
      states[i],
      vp == null ? null : layers[vp],
      vp == null ? null : states[vp],
      vlp == null ? null : layers[vlp],
      vlp == null ? null : states[vlp],
      timeStep,
      meanField,
    );
  }
}

/// `vectorised_continuous_prediction`.
void _predict(
  Layer layer,
  LayerState c,
  Layer? valueParent,
  LayerState? vps,
  Layer? volParent,
  LayerState? vlps,
  double t,
  bool meanField,
) {
  final n = c.nodes;
  final lambda = layer.autoconnectionStrength!;
  final omega = layer.tonicVolatility!;
  final rho = layer.tonicDrift!;

  // 1. Expected mean: autoregression plus the time-scaled drift. The
  // observation layer is generated by its value parent, not drifted by it
  // (Eq. 64 of Weber et al., 2026; pyhgf #478): its drift takes no Δt.
  final drift = Float64List(n);
  Float64List? vcv;
  if (valueParent != null) {
    final w = valueParent.weights!;
    final g = valueParent.coupling;
    _mv(drift, w, _map(vps!.expectedMean, g.f));
    if (!meanField) {
      // 3. Taylor value-coupling variance: s²·Σ_b W² g'(μ̂_b)² / π̂_b, with
      // s the drift scale.
      final gp = Float64List(vps.nodes);
      for (var b = 0; b < vps.nodes; b++) {
        final d = g.df(vps.expectedMean[b]);
        gp[b] = d * d / vps.expectedPrecision[b];
      }
      vcv = Float64List(n);
      _mv(vcv, w, gp, squared: true);
    }
  }

  // 2. Total volatility, with the MGF term κ²/(2π̂) unless mean-field.
  Float64List? kMu, kVar;
  if (volParent != null) {
    final kappa = volParent.volatilityWeights!;
    kMu = Float64List(n);
    _mv(kMu, kappa, vlps!.expectedMean);
    if (!meanField) {
      kVar = Float64List(n);
      _mv(
        kVar,
        kappa,
        _map(vlps.expectedPrecision, (p) => 1.0 / (2.0 * p)),
        squared: true,
      );
    }
  }

  final driftScale = layer.isInputLayer ? 1.0 : t;
  final s2 = driftScale * driftScale;
  for (var a = 0; a < n; a++) {
    c.expectedMean[a] =
        lambda[a] * c.mean[a] + driftScale * (rho[a] + drift[a]);
    var total = omega[a];
    if (kMu != null) total = total + kMu[a];
    if (kVar != null) total = total + kVar[a];
    var predVol = t * math.exp(total);
    if (!(predVol > 1e-128)) predVol = double.nan;
    final valueVar = vcv == null ? 0.0 : s2 * vcv[a];
    // 4. Conditional and marginal predicted precisions, effective γ.
    final carried = (lambda[a] * lambda[a]) / c.precision[a];
    final cep = 1.0 / _floor(carried + predVol, minPredictedVariance);
    final ep = 1.0 / _floor(carried + predVol + valueVar, minPredictedVariance);
    if (layer.isInputLayer) {
      // 5. The observation layer: both predicted precisions are the
      // precision of the observation noise, set by its noise parent when it
      // has one (γ = 1), its own precision otherwise (γ = 0).
      final noise = kMu == null ? c.precision[a] : math.exp(-omega[a] - kMu[a]);
      c.expectedPrecision[a] = noise;
      c.conditionalExpectedPrecision[a] = noise;
      c.effectivePrecision[a] = kMu == null ? 0.0 : 1.0;
    } else {
      c.expectedPrecision[a] = ep;
      c.conditionalExpectedPrecision[a] = cep;
      c.effectivePrecision[a] = predVol * ep;
    }
  }
}

// ---------------------------------------------------------------------------
// Prediction errors
// ---------------------------------------------------------------------------

/// The value prediction error, and the volatility prediction error when
/// [volatility] (`vectorised_continuous_prediction_error`).
void _predictionErrors(LayerState s, bool volatility) {
  for (var a = 0; a < s.nodes; a++) {
    s.valuePredictionError[a] = s.mean[a] - s.expectedMean[a];
  }
  if (!volatility) return;
  final vpe = s.volatilityPredictionError!;
  for (var a = 0; a < s.nodes; a++) {
    final d = s.mean[a] - s.expectedMean[a];
    vpe[a] =
        (s.expectedPrecision[a] / s.precision[a]) +
        s.expectedPrecision[a] * (d * d) -
        1.0;
  }
}

/// The noise prediction error of the observation layer, from the updated
/// posterior of its value parent (Eq. 72 of Weber et al., 2026;
/// `vectorised_continuous_noise_prediction_error`), into its volatility
/// prediction error. No time step: the observation layer is generated by
/// its value parent, not drifted by it (pyhgf #478).
void _noisePredictionError(
  LayerState leaf,
  Layer? valueParent,
  LayerState? vps,
) {
  final n = leaf.nodes;
  final shift = Float64List(n);
  final variance = Float64List(n);
  if (valueParent != null) {
    final w = valueParent.weights!;
    final g = valueParent.coupling;
    final dg = Float64List(vps!.nodes);
    final gp = Float64List(vps.nodes);
    for (var b = 0; b < vps.nodes; b++) {
      dg[b] = g.f(vps.mean[b]) - g.f(vps.expectedMean[b]);
      final d = g.df(vps.mean[b]);
      gp[b] = d * d / vps.precision[b];
    }
    _mv(shift, w, dg);
    _mv(variance, w, gp, squared: true);
  }
  final out = leaf.volatilityPredictionError!;
  for (var a = 0; a < n; a++) {
    final d = leaf.mean[a] - (leaf.expectedMean[a] + shift[a]);
    out[a] = leaf.expectedPrecision[a] * (variance[a] + d * d) - 1.0;
  }
}

// ---------------------------------------------------------------------------
// Update sweep
// ---------------------------------------------------------------------------

/// Clamp row [row] of [y] on the observation layer, then update every layer
/// above it in [ContinuousTopology.updateOrder]
/// (`_continuous_update_sweep`).
void continuousUpdateSweep(
  List<Layer> layers,
  List<LayerState> states,
  ContinuousTopology topo,
  Matrix y,
  int row,
  double timeStep,
  DeepSettings settings,
) {
  final leaf = states[0];
  for (var a = 0; a < leaf.nodes; a++) {
    leaf.mean[a] = y.data[row * y.cols + a];
  }
  _predictionErrors(leaf, false);

  for (final i in topo.updateOrder) {
    final layer = layers[i];
    final s = states[i];
    final vc = layer.config.valueChild;
    final vlc = layer.config.volatilityChild;
    if (vlc == 0) {
      final vp = topo.valueParentOf[0];
      _noisePredictionError(
        states[0],
        vp == null ? null : layers[vp],
        vp == null ? null : states[vp],
      );
    }
    if (vc == null && vlc == null) continue;
    final valueChild = vc == null
        ? null
        : _ValueChild(
            states[vc],
            layer.weights!,
            layer.coupling,
            vc == 0,
            vc == 0 ? 1.0 : timeStep,
          );
    final volChild = vlc == null
        ? null
        : _VolatilityChild(states[vlc], layer.volatilityWeights!, layers[vlc]);
    final scheme = settings.volatilityUpdates;
    if (volChild == null ||
        vlc == 0 ||
        scheme == DeepVolatilityUpdate.standard) {
      _posteriorStandard(s, valueChild, volChild, settings);
    } else if (scheme == DeepVolatilityUpdate.ehgf) {
      _posteriorEhgf(s, valueChild, volChild, timeStep, settings);
    } else {
      assert(valueChild == null);
      _posteriorUnbounded(s, volChild, timeStep, settings);
    }
    final hasVolParent = topo.volatilityParentOf[i] != null;
    if (topo.valueParentOf[i] != null || hasVolParent) {
      _predictionErrors(s, hasVolParent);
    }
  }
}

final class _ValueChild {
  _ValueChild(
    this.state,
    this.weights,
    this.coupling,
    this.clamped,
    this.timeScale,
  );
  final LayerState state;
  final Matrix weights;
  final Coupling coupling;

  /// The observation layer, whose precision never moves: the relaxed
  /// factors take their fully observed limit.
  final bool clamped;

  /// The factor of Δt in the child's drift: Δt for a state layer, whose
  /// prediction moves by Δt·W·g' per unit of the parent, 1 for the
  /// observation layer (pyhgf #478). The value messages take it too.
  final double timeScale;
}

final class _VolatilityChild {
  _VolatilityChild(this.state, this.kappa, this.layer);
  final LayerState state;
  final Matrix kappa;
  final Layer layer;
}

/// `(effective precision, gain)` of each child node (`_relaxed_factors`).
(Float64List, Float64List) _relaxedFactors(_ValueChild c, bool meanField) {
  final s = c.state;
  if (meanField) return (s.expectedPrecision, s.expectedPrecision);
  final cond = s.conditionalExpectedPrecision;
  if (c.clamped) return (cond, cond);
  final n = s.nodes;
  final eff = Float64List(n), gain = Float64List(n);
  for (var a = 0; a < n; a++) {
    final piY = s.precision[a] - s.expectedPrecision[a];
    final den = cond[a] + piY;
    eff[a] = cond[a] * piY / den;
    gain[a] = cond[a] * s.precision[a] / den;
  }
  return (eff, gain);
}

/// `Σ_a s·W(a, m)·gain_a·δ_a` per parent node, s the child's time scale.
Float64List _gainMessage(_ValueChild c, Float64List gain, int nParent) {
  final s = c.state;
  final ge = Float64List(s.nodes);
  for (var a = 0; a < s.nodes; a++) {
    ge[a] = gain[a] * s.valuePredictionError[a];
  }
  final out = Float64List(nParent);
  _mtv(out, c.weights, ge);
  final ts = c.timeScale;
  if (ts != 1.0) {
    for (var k = 0; k < nParent; k++) {
      out[k] = ts * out[k];
    }
  }
  return out;
}

/// `_value_precision_pwpe`, added to [pwpe].
void _addValuePrecision(
  Float64List pwpe,
  Float64List evalMean,
  _ValueChild c,
  bool meanField,
) {
  final (eff, gain) = _relaxedFactors(c, meanField);
  final m = pwpe.length;
  final info = Float64List(m);
  _mtv(info, c.weights, eff, squared: true);
  // (s·W)² = s²·W²: the time scale of a state-layer child (pyhgf #478).
  final ts2 = c.timeScale * c.timeScale;
  final msg = _gainMessage(c, gain, m);
  final g = c.coupling;
  for (var k = 0; k < m; k++) {
    final g1 = g.df(evalMean[k]);
    pwpe[k] =
        pwpe[k] + (ts2 * info[k] * (g1 * g1) + -g.d2f(evalMean[k]) * msg[k]);
  }
}

/// `_value_mean_pwpe`, added to [out].
void _addValueMean(
  Float64List out,
  Float64List evalMean,
  _ValueChild c,
  Float64List nodePrecision,
  bool meanField,
) {
  final (_, gain) = _relaxedFactors(c, meanField);
  final msg = _gainMessage(c, gain, out.length);
  for (var k = 0; k < out.length; k++) {
    out[k] = out[k] + msg[k] * c.coupling.df(evalMean[k]) / nodePrecision[k];
  }
}

/// `_volatility_mean_pwpe`, added to [out].
void _addVolatilityMean(
  Float64List out,
  _VolatilityChild c,
  Float64List nodePrecision,
) {
  final s = c.state;
  final ge = Float64List(s.nodes);
  for (var a = 0; a < s.nodes; a++) {
    ge[a] = s.effectivePrecision[a] * s.volatilityPredictionError![a];
  }
  final msg = Float64List(out.length);
  _mtv(msg, c.kappa, ge);
  for (var k = 0; k < out.length; k++) {
    out[k] = out[k] + msg[k] / (2.0 * nodePrecision[k]);
  }
}

/// `_volatility_precision_pwpe_standard`, added to [pwpe].
void _addVolatilityPrecisionStandard(Float64List pwpe, _VolatilityChild c) {
  final s = c.state;
  final n = s.nodes;
  final g2 = Float64List(n), g2d = Float64List(n), gd = Float64List(n);
  for (var a = 0; a < n; a++) {
    final gamma = s.effectivePrecision[a];
    final delta = s.volatilityPredictionError![a];
    g2[a] = gamma * gamma;
    g2d[a] = gamma * gamma * delta;
    gd[a] = gamma * delta;
  }
  final m = pwpe.length;
  final a1 = Float64List(m), a2 = Float64List(m), a3 = Float64List(m);
  _mtv(a1, c.kappa, g2, squared: true);
  _mtv(a2, c.kappa, g2d, squared: true);
  _mtv(a3, c.kappa, gd, squared: true);
  for (var k = 0; k < m; k++) {
    pwpe[k] = pwpe[k] + (0.5 * a1[k] + a2[k] - 0.5 * a3[k]);
  }
}

/// `_finalize_precision`.
double _finalize(double expected, double pwpe, double max) {
  var p = expected + pwpe;
  if (!(p > 1e-128)) p = double.nan;
  return math.min(p, max);
}

/// The known variance of each volatility child's prediction: the carried
/// variance reconstructed from π̄ (`_child_previous_variance`) plus the
/// variance its value parents add (`_value_coupling_variance`).
Float64List _knownVariance(
  LayerState parent,
  _VolatilityChild c,
  double t,
  bool meanField,
) {
  final s = c.state;
  final n = s.nodes;
  final omega = c.layer.tonicVolatility!;
  final kMu = Float64List(n);
  _mv(kMu, c.kappa, parent.expectedMean);
  Float64List? kVar;
  if (!meanField) {
    kVar = Float64List(n);
    _mv(
      kVar,
      c.kappa,
      _map(parent.expectedPrecision, (p) => 1.0 / (2.0 * p)),
      squared: true,
    );
  }
  final out = Float64List(n);
  for (var a = 0; a < n; a++) {
    var total = omega[a] + kMu[a];
    if (kVar != null) total = total + kVar[a];
    final previous = _floor(
      1.0 / s.conditionalExpectedPrecision[a] - t * math.exp(total),
      1e-128,
    );
    final valueVar = _floor(
      1.0 / s.expectedPrecision[a] - 1.0 / s.conditionalExpectedPrecision[a],
      0.0,
    );
    out[a] = previous + valueVar;
  }
  return out;
}

/// `vectorised_continuous_posterior_update_standard`.
void _posteriorStandard(
  LayerState s,
  _ValueChild? vc,
  _VolatilityChild? vlc,
  DeepSettings settings,
) {
  final n = s.nodes;
  final mf = settings.meanFieldUpdates;
  final evalMean = Float64List.fromList(s.mean);
  final pwpe = Float64List(n);
  if (vc != null) _addValuePrecision(pwpe, evalMean, vc, mf);
  if (vlc != null) _addVolatilityPrecisionStandard(pwpe, vlc);
  final precision = Float64List(n);
  for (var k = 0; k < n; k++) {
    precision[k] = _finalize(
      s.expectedPrecision[k],
      pwpe[k],
      settings.maxPosteriorPrecision,
    );
  }
  final meanPwpe = Float64List(n);
  if (vc != null) _addValueMean(meanPwpe, evalMean, vc, precision, mf);
  if (vlc != null) _addVolatilityMean(meanPwpe, vlc, precision);
  for (var k = 0; k < n; k++) {
    s.precision[k] = precision[k];
    s.mean[k] = s.expectedMean[k] + meanPwpe[k];
  }
}

/// `vectorised_continuous_posterior_update_ehgf`: the mean first (with the
/// expected precision), then the safe precision at the new mean.
void _posteriorEhgf(
  LayerState s,
  _ValueChild? vc,
  _VolatilityChild vlc,
  double t,
  DeepSettings settings,
) {
  final n = s.nodes;
  final mf = settings.meanFieldUpdates;
  final evalMean = Float64List.fromList(s.mean);
  final meanPwpe = Float64List(n);
  if (vc != null) {
    _addValueMean(meanPwpe, evalMean, vc, s.expectedPrecision, mf);
  }
  _addVolatilityMean(meanPwpe, vlc, s.expectedPrecision);
  final posteriorMean = Float64List(n);
  for (var k = 0; k < n; k++) {
    posteriorMean[k] = s.expectedMean[k] + meanPwpe[k];
  }

  final pwpe = Float64List(n);
  if (vc != null) _addValuePrecision(pwpe, posteriorMean, vc, mf);
  _addVolatilityPrecisionEhgf(pwpe, posteriorMean, s, vlc, t, mf);
  for (var k = 0; k < n; k++) {
    s.precision[k] = _finalize(
      s.expectedPrecision[k],
      pwpe[k],
      settings.maxPosteriorPrecision,
    );
    s.mean[k] = posteriorMean[k];
  }
}

/// `_volatility_precision_pwpe_ehgf`: each κ edge re-predicts the child's
/// volatility with its parent at the new posterior mean (the others at
/// their predictions, no MGF term); each edge's increment is floored at 0.
void _addVolatilityPrecisionEhgf(
  Float64List pwpe,
  Float64List posteriorMean,
  LayerState parent,
  _VolatilityChild c,
  double t,
  bool meanField,
) {
  final s = c.state;
  final previous = _knownVariance(parent, c, t, meanField);
  final omega = c.layer.tonicVolatility!;
  final kMu = Float64List(s.nodes);
  _mv(kMu, c.kappa, parent.expectedMean);
  final cols = c.kappa.cols;
  final kd = c.kappa.data;
  for (var a = 0; a < s.nodes; a++) {
    final d = s.mean[a] - s.expectedMean[a];
    final aux = 1.0 / s.precision[a] + d * d;
    final prev = previous[a];
    for (var m = 0; m < cols; m++) {
      final kappa = kd[a * cols + m];
      final predVol =
          t *
          math.exp(
            omega[a] +
                kMu[a] +
                kappa * (posteriorMean[m] - parent.expectedMean[m]),
          );
      final expected = 1.0 / (prev + predVol);
      final eff = predVol * expected;
      final weight = (predVol - prev) * expected;
      final vpe = aux * expected - 1.0;
      final inc = 0.5 * (kappa * kappa) * eff * (eff + weight * vpe);
      // jnp.maximum(0, NaN) is NaN.
      pwpe[m] = pwpe[m] + (inc.isNaN ? inc : math.max(0.0, inc));
    }
  }
}

/// `vectorised_continuous_posterior_update_unbounded`, for a one-to-one
/// (square, diagonal) κ and no value child.
void _posteriorUnbounded(
  LayerState s,
  _VolatilityChild c,
  double t,
  DeepSettings settings,
) {
  final child = c.state;
  final omega = c.layer.tonicVolatility!;
  final previous = _knownVariance(s, c, t, settings.meanFieldUpdates);
  final logT = math.log(t);
  final maxPp = settings.maxPosteriorPrecision;
  final cols = c.kappa.cols;
  for (var k = 0; k < s.nodes; k++) {
    final kappa = c.kappa.data[k * cols + k];
    final w = omega[k];
    final em = s.expectedMean[k];
    final e = s.expectedPrecision[k];
    final d = child.mean[k] - child.expectedMean[k];
    final beAux = 1.0 / child.precision[k] + d * d;
    final logPrev = math.log(previous[k]);

    final gammaC = logT + kappa * em + w;
    final wJm1 = sigmoid(gammaC - logPrev);
    final daJm1 = child.expectedPrecision[k] * beAux - 1.0;

    // Expansion 1: quadratic at the prediction.
    final pi1 = e + 0.5 * (kappa * kappa) * wJm1 * (1.0 - wJm1);
    final mu1 = em + (kappa * wJm1 / (2.0 * pi1)) * daJm1;

    // Expansion 2: quadratic at the Lambert-W approximate mode (dropped for
    // a zero coupling, with κ = 1 standing in so that no 0/0 is formed).
    final hasCoupling = kappa != 0.0;
    final safeKappa = hasCoupling ? kappa : 1.0;
    final pihatY = e / (safeKappa * safeKappa);
    final logWArg =
        math.log(beAux) - math.log(2.0 * pihatY) + 0.5 / pihatY - gammaC;
    final vW = lambertW0Log(math.min(logWArg, _logFloatMax));
    final yStar = gammaC + vW - 0.5 / pihatY;
    final xStar = (yStar - logT - w) / safeKappa;
    final logS2 = logT + kappa * xStar + w;
    final logDenomS = logAddExp(logPrev, logS2);
    final w2 = sigmoid(logS2 - logPrev);
    final da2 = beAux * math.exp(-logDenomS) - 1.0;
    final k2 = kappa * kappa;
    final pi2Full = e + 0.5 * k2 * w2 * (w2 + (2.0 * w2 - 1.0) * da2);
    final pi2Safe = pi2Full <= 0.0 ? e + 0.5 * k2 * w2 * (1.0 - w2) : pi2Full;
    final mu2Safe =
        xStar + (0.5 * kappa * w2 * da2 - e * (xStar - em)) / pi2Safe;
    final valid = hasCoupling && pi2Safe.isFinite && mu2Safe.isFinite;
    final pi2 = valid ? pi2Safe : pi1;
    final mu2 = valid ? mu2Safe : mu1;

    // Variational energy-based softmax blend.
    final logDenom1 = logAddExp(logPrev, logT + kappa * mu1 + w);
    final i1 =
        -0.5 * logDenom1 -
        0.5 * beAux * math.exp(-logDenom1) -
        0.5 * e * ((mu1 - em) * (mu1 - em));
    final logDenom2 = logAddExp(logPrev, logT + kappa * mu2 + w);
    final i2 =
        -0.5 * logDenom2 -
        0.5 * beAux * math.exp(-logDenom2) -
        0.5 * e * ((mu2 - em) * (mu2 - em));
    final blend = sigmoid(i2 - i1);

    // Gaussian mixture moment matching.
    s.mean[k] = (1.0 - blend) * mu1 + blend * mu2;
    final sig2 =
        (1.0 - blend) / pi1 +
        blend / pi2 +
        blend * (1.0 - blend) * ((mu1 - mu2) * (mu1 - mu2));
    s.precision[k] = math.min(1.0 / sig2, maxPp);
  }
}
