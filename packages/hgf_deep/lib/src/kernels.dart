/// The belief-propagation kernels of the deep network, mirroring pyhgf's
/// `pyhgf/updates/vectorised/` and the sweeps of
/// `pyhgf/utils/vectorised_belief_propagation.py` term for term.
///
/// Every kernel works on [LayerState]s of any width: the network's own
/// state has width 1 (pyhgf's per-sample path), a batched step tiles it to
/// the batch width (pyhgf's `vmap`). Fields are `[node][sample]`, so the
/// weight contractions run with the sample loop innermost.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'config.dart';
import 'matrix.dart';
import 'state.dart';

/// Predicted variances are floored here (pyhgf's `MIN_PREDICTED_VARIANCE`).
const double minPredictedVariance = 1e-128;

/// `time_step · exp(·)` results at or below this are numerically zero and
/// become NaN (pyhgf's `jnp.where(v > 1e-128, v, nan)` guard).
const double minVolatility = 1e-128;

/// `ln(f64::MAX)`, the cap on the Lambert-W argument's logarithm.
final double _logFloatMax = math.log(double.maxFinite);

/// The network-wide settings the kernels read.
final class DeepSettings {
  const DeepSettings({
    required this.volatilityUpdates,
    required this.maxPosteriorPrecision,
    required this.precisionClippingValue,
    required this.predictPrecision,
    required this.feedforwardUncertainty,
    required this.carryPrecision,
    required this.meanFieldUpdates,
  });

  final DeepVolatilityUpdate volatilityUpdates;
  final double maxPosteriorPrecision;
  final double precisionClippingValue;
  final bool predictPrecision;
  final bool feedforwardUncertainty;
  final bool carryPrecision;
  final bool meanFieldUpdates;
}

/// One built layer: its description, per-node parameters and incoming
/// weights.
final class Layer {
  Layer({
    required this.config,
    required this.index,
    required this.tonicVolatilityVol,
    required this.tonicVolatility,
    required this.priorPrecision,
    required this.weights,
  });

  final LayerConfig config;

  /// Position in the network, 0 being the bottom (observation) layer.
  final int index;

  /// Per-node tonic volatility of the volatility level.
  final Float64List tonicVolatilityVol;

  /// Per-node value-level tonic volatility ω, or `null` when not allocated.
  final Float64List? tonicVolatility;

  /// Per-node prior precision π₀ (the initial precision).
  final Float64List priorPrecision;

  /// Weights from the layer below up into this one, `child × (size[+1])`
  /// with a trailing bias column when [LayerConfig.addConstantInput]; `null`
  /// for the bottom layer.
  Matrix? weights;

  int get size => config.size;
  LayerKind get kind => config.kind;
  Coupling get coupling => config.coupling;
  bool get hasConstant => config.addConstantInput;
  bool get isInputLayer => index == 0;
  bool get hasVolatility => config.volatilityParent;
}

double _guard(double v) => v > minVolatility ? v : double.nan;

/// `max(x, lo)` that keeps a NaN `x` (as `jnp.maximum` does).
double _floor(double x, double lo) => x < lo ? lo : x;

/// `jnp.clip(x, lo, hi)`: NaN in any argument gives NaN.
double _clip(double x, double lo, double hi) => math.min(math.max(x, lo), hi);

bool _finite(double x) => x.isFinite;

/// Scratch buffers for one width, reused across calls.
final class Workspace {
  Workspace(this.width);

  final int width;
  final Map<int, Float64List> _buffers = {};

  /// A buffer of exactly [length] values, tagged by [slot].
  Float64List raw(int slot, int length) {
    final buf = _buffers[slot];
    if (buf != null && buf.length == length) return buf;
    return _buffers[slot] = Float64List(length);
  }

  /// A buffer of at least [length] values, tagged by [slot]; it only grows,
  /// for scratch whose size changes from layer to layer.
  Float64List atLeast(int slot, int length) {
    final buf = _buffers[slot];
    if (buf != null && buf.length >= length) return buf;
    return _buffers[slot] = Float64List(length);
  }

  /// A buffer of `rows * width` values, tagged by [slot] so that
  /// buffers in use at the same time never alias.
  Float64List take(int slot, int rows) {
    final need = rows * width;
    final buf = _buffers[slot];
    if (buf != null && buf.length == need) return buf;
    return _buffers[slot] = Float64List(need);
  }
}

// Workspace slots.
const _sCoupled = 0;
const _sPerParent = 1;
const _sPredVol = 2;
const _sVcv = 3;
const _sEff = 4;
const _sGain = 5;
const _sPc1 = 6;
const _sMsg = 7;
const _sSquared = 8;
const _sHt = 9;
const _sOuter = 10;

/// `g(μ)` of [parent] rows, with a trailing bias row of ones when
/// [hasConstant]: `(nParent[+1]) × width`.
Float64List _coupled(
  Workspace ws,
  Float64List values,
  int nParent,
  Coupling g,
  bool hasConstant, {
  bool sanitize = false,
}) {
  final b = ws.width;
  final out = ws.take(_sCoupled, nParent + (hasConstant ? 1 : 0));
  final n = nParent * b;
  if (identical(g, CouplingFn.linear)) {
    for (var k = 0; k < n; k++) {
      final v = values[k];
      out[k] = sanitize && !_finite(v) ? 0.0 : v;
    }
  } else {
    for (var k = 0; k < n; k++) {
      final v = g.f(values[k]);
      out[k] = sanitize && !_finite(v) ? 0.0 : v;
    }
  }
  if (hasConstant) out.fillRange(n, n + b, 1.0);
  return out;
}

/// `out[r][s] = Σ_k W(r, k) · a[k][s]` for `r < rows`, `k < depth`,
/// `s < b`, with `W(r, k) = w[r * rs + k * cs]`: `(rs, cs) = (stride, 1)`
/// reads [w] as stored, `(1, stride)` reads its transpose. At width > 1
/// each output is summed over `k` in ascending order from 0; width 1 keeps
/// four partial sums.
///
/// Width 1 is a matrix–vector product (one register accumulator per row);
/// wider batches go through a 2-row × 4-sample register tile, or on native
/// targets ([useSimd]) a 4-row × 4-sample tile of `Float64x2` lanes, which
/// sums in the same order and so gives the same bits, ~3× faster.
@pragma('vm:unsafe:no-bounds-checks')
void matMul(
  Float64List out,
  Float64List w,
  int rs,
  int cs,
  int rows,
  int depth,
  Float64List a,
  int b,
) {
  assert(out.length >= rows * b && a.length >= depth * b);
  assert(
    rows == 0 || depth == 0 || w.length > (rows - 1) * rs + (depth - 1) * cs,
  );
  if (b == 1 && rs == 1 && cs != 1) {
    // Transposed matrix–vector product: stream W's rows (contiguous) and
    // accumulate into the outputs; each output still sums k ascending.
    out.fillRange(0, rows, 0.0);
    for (var k = 0; k < depth; k++) {
      final ak = a[k];
      final wk = k * cs;
      for (var r = 0; r < rows; r++) {
        out[r] += w[wk + r] * ak;
      }
    }
    return;
  }
  if (b == 1) {
    // Four partial sums (k mod 4) break the add-latency chain.
    final d4 = depth - depth % 4;
    for (var r = 0; r < rows; r++) {
      var s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
      var wi = r * rs;
      var k = 0;
      for (; k < d4; k += 4) {
        s0 += w[wi] * a[k];
        s1 += w[wi + cs] * a[k + 1];
        s2 += w[wi + 2 * cs] * a[k + 2];
        s3 += w[wi + 3 * cs] * a[k + 3];
        wi += 4 * cs;
      }
      for (; k < depth; k++) {
        s0 += w[wi] * a[k];
        wi += cs;
      }
      out[r] = (s0 + s1) + (s2 + s3);
    }
    return;
  }
  if (useSimd && b.isEven && (out.offsetInBytes | a.offsetInBytes) & 15 == 0) {
    _matMulSimd(out, w, rs, cs, rows, depth, a, b);
    return;
  }
  final s4 = b - b % 4;
  var r = 0;
  for (; r + 1 < rows; r += 2) {
    final w0 = r * rs;
    final w1 = w0 + rs;
    final o0 = r * b;
    final o1 = o0 + b;
    for (var s = 0; s < s4; s += 4) {
      var c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
      var c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
      var ak = s;
      var wk = 0;
      for (var k = 0; k < depth; k++) {
        final x0 = w[w0 + wk];
        final x1 = w[w1 + wk];
        final a0 = a[ak], a1 = a[ak + 1], a2 = a[ak + 2], a3 = a[ak + 3];
        c00 += x0 * a0;
        c01 += x0 * a1;
        c02 += x0 * a2;
        c03 += x0 * a3;
        c10 += x1 * a0;
        c11 += x1 * a1;
        c12 += x1 * a2;
        c13 += x1 * a3;
        ak += b;
        wk += cs;
      }
      out[o0 + s] = c00;
      out[o0 + s + 1] = c01;
      out[o0 + s + 2] = c02;
      out[o0 + s + 3] = c03;
      out[o1 + s] = c10;
      out[o1 + s + 1] = c11;
      out[o1 + s + 2] = c12;
      out[o1 + s + 3] = c13;
    }
    for (var s = s4; s < b; s++) {
      var c0 = 0.0, c1 = 0.0;
      for (var k = 0; k < depth; k++) {
        final ak = a[k * b + s];
        c0 += w[w0 + k * cs] * ak;
        c1 += w[w1 + k * cs] * ak;
      }
      out[o0 + s] = c0;
      out[o1 + s] = c1;
    }
  }
  for (; r < rows; r++) {
    final w0 = r * rs;
    final o0 = r * b;
    for (var s = 0; s < s4; s += 4) {
      var c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;
      var ak = s;
      for (var k = 0; k < depth; k++) {
        final x = w[w0 + k * cs];
        c0 += x * a[ak];
        c1 += x * a[ak + 1];
        c2 += x * a[ak + 2];
        c3 += x * a[ak + 3];
        ak += b;
      }
      out[o0 + s] = c0;
      out[o0 + s + 1] = c1;
      out[o0 + s + 2] = c2;
      out[o0 + s + 3] = c3;
    }
    for (var s = s4; s < b; s++) {
      var c = 0.0;
      for (var k = 0; k < depth; k++) {
        c += w[w0 + k * cs] * a[k * b + s];
      }
      out[o0 + s] = c;
    }
  }
}

/// Whether [matMul] uses `Float64x2` lanes: only where they are real SIMD.
/// dart2wasm (Dart 3.13) and dart2js emulate them about 10× slower than
/// scalar code (`hgf/benchmark/bin/runtime_probe.dart`); `identical(1, 1.0)`
/// holds only on dart2js.
const bool useSimd =
    !bool.fromEnvironment('dart.tool.dart2wasm') && !identical(1, 1.0);

/// [matMul] for even `b` with 16-byte-aligned [out] and [a], on `Float64x2`
/// lanes over samples. Lanes are independent and each still adds its terms
/// in ascending `k` from zero, so the result equals the scalar kernel's bit
/// for bit. Tile: 4 rows × 2 vectors (8 accumulators), then single rows,
/// then a single trailing vector when `b / 2` is odd.
@pragma('vm:unsafe:no-bounds-checks')
void _matMulSimd(
  Float64List out,
  Float64List w,
  int rs,
  int cs,
  int rows,
  int depth,
  Float64List a,
  int b,
) {
  final av = Float64x2List.view(a.buffer, a.offsetInBytes, depth * b >> 1);
  final ov = Float64x2List.view(out.buffer, out.offsetInBytes, rows * b >> 1);
  final hb = b >> 1;
  final h2 = hb - hb % 2;
  var r = 0;
  for (; r + 3 < rows; r += 4) {
    final w0 = r * rs, w1 = w0 + rs, w2 = w1 + rs, w3 = w2 + rs;
    final o0 = r * hb, o1 = o0 + hb, o2 = o1 + hb, o3 = o2 + hb;
    for (var s = 0; s < hb; s += 2) {
      final pair = s < h2;
      var c00 = Float64x2.zero(), c01 = Float64x2.zero();
      var c10 = Float64x2.zero(), c11 = Float64x2.zero();
      var c20 = Float64x2.zero(), c21 = Float64x2.zero();
      var c30 = Float64x2.zero(), c31 = Float64x2.zero();
      var ak = s;
      var wk = 0;
      if (pair) {
        for (var k = 0; k < depth; k++) {
          final a0 = av[ak], a1 = av[ak + 1];
          final x0 = Float64x2.splat(w[w0 + wk]);
          c00 += x0 * a0;
          c01 += x0 * a1;
          final x1 = Float64x2.splat(w[w1 + wk]);
          c10 += x1 * a0;
          c11 += x1 * a1;
          final x2 = Float64x2.splat(w[w2 + wk]);
          c20 += x2 * a0;
          c21 += x2 * a1;
          final x3 = Float64x2.splat(w[w3 + wk]);
          c30 += x3 * a0;
          c31 += x3 * a1;
          ak += hb;
          wk += cs;
        }
        ov[o0 + s + 1] = c01;
        ov[o1 + s + 1] = c11;
        ov[o2 + s + 1] = c21;
        ov[o3 + s + 1] = c31;
      } else {
        for (var k = 0; k < depth; k++) {
          final a0 = av[ak];
          c00 += Float64x2.splat(w[w0 + wk]) * a0;
          c10 += Float64x2.splat(w[w1 + wk]) * a0;
          c20 += Float64x2.splat(w[w2 + wk]) * a0;
          c30 += Float64x2.splat(w[w3 + wk]) * a0;
          ak += hb;
          wk += cs;
        }
      }
      ov[o0 + s] = c00;
      ov[o1 + s] = c10;
      ov[o2 + s] = c20;
      ov[o3 + s] = c30;
    }
  }
  for (; r < rows; r++) {
    final w0 = r * rs;
    final o0 = r * hb;
    for (var s = 0; s < hb; s += 2) {
      final pair = s < h2;
      var c0 = Float64x2.zero(), c1 = Float64x2.zero();
      var ak = s;
      var wk = w0;
      for (var k = 0; k < depth; k++) {
        final x = Float64x2.splat(w[wk]);
        c0 += x * av[ak];
        if (pair) c1 += x * av[ak + 1];
        ak += hb;
        wk += cs;
      }
      ov[o0 + s] = c0;
      if (pair) ov[o0 + s + 1] = c1;
    }
  }
}

/// `out[i][s] = Σ_k w[i][k] · a[k][s]` for `i < rows`, `k < cols`, with
/// [w] row-major of stride [stride].
void _gemm(
  Float64List out,
  Float64List w,
  int rows,
  int cols,
  int stride,
  Float64List a,
  int b,
) => matMul(out, w, stride, 1, rows, cols, a, b);

/// `out[j][s] = Σ_i w[i][j] · a[i][s]` for `i < rows`, `j < cols` (the
/// transposed product), [w] row-major of stride [stride].
void _gemmT(
  Float64List out,
  Float64List w,
  int rows,
  int cols,
  int stride,
  Float64List a,
  int b,
) => matMul(out, w, 1, stride, cols, rows, a, b);

// ---------------------------------------------------------------------------
// Top-down prediction
// ---------------------------------------------------------------------------

/// Predict [child] from [parent] (`_predict_layer_from_parent`).
void predictLayer(
  LayerState child,
  Layer childLayer,
  LayerState parent,
  Layer parentLayer,
  Workspace ws,
  double timeStep,
  DeepSettings settings,
) {
  switch (childLayer.kind) {
    case LayerKind.volatile:
      _predictVolatile(
        child,
        childLayer,
        parent,
        parentLayer,
        ws,
        timeStep,
        settings,
      );
    case LayerKind.binary:
      _predictBinary(child, parent, parentLayer, ws, settings);
    case LayerKind.categorical:
      _predictCategorical(child, parent, parentLayer, ws);
  }
}

/// `vectorised_layer_prediction` with `_predict_volatility_level`.
void _predictVolatile(
  LayerState c,
  Layer cl,
  LayerState p,
  Layer pl,
  Workspace ws,
  double t,
  DeepSettings settings,
) {
  final n = c.nodes;
  final b = c.width;
  final np = p.nodes;
  final w = pl.weights!;
  final cols = w.cols;
  final predict = settings.predictPrecision;
  final meanField = settings.meanFieldUpdates;
  final tonic = cl.tonicVolatility;

  // 1. Volatility level, and the diffusion term Ω of the value level.
  final predVol = ws.take(_sPredVol, n);
  if (predict) {
    if (cl.hasVolatility) {
      final mv = c.meanVol!;
      final pv = c.precisionVol!;
      final emv = c.expectedMeanVol!;
      final epv = c.expectedPrecisionVol!;
      final effv = c.effectivePrecisionVol!;
      for (var i = 0; i < n; i++) {
        final pvv = _guard(t * math.exp(cl.tonicVolatilityVol[i]));
        final ti = tonic?[i];
        for (var s = 0; s < b; s++) {
          final k = i * b + s;
          final m = mv[k];
          final e = 1.0 / (1.0 / pv[k] + pvv);
          emv[k] = m;
          epv[k] = e;
          effv[k] = pvv * e;
          var total = m;
          if (!meanField) total = total + 1.0 / (2.0 * e);
          if (ti != null) total = total + ti;
          predVol[k] = _guard(t * math.exp(total));
        }
      }
    } else {
      for (var i = 0; i < n; i++) {
        final v = tonic == null ? 0.0 : _guard(t * math.exp(tonic[i]));
        predVol.fillRange(i * b, i * b + b, v);
      }
    }
  } else if (cl.hasVolatility) {
    // The frozen volatility level passes through.
    c.expectedMeanVol!.setAll(0, c.meanVol!);
    c.expectedPrecisionVol!.setAll(0, c.precisionVol!);
  }

  // 2. Value level: the mean, W @ g(μ̂_parent).
  final coupled = _coupled(ws, p.expectedMean, np, pl.coupling, pl.hasConstant);
  _gemm(c.expectedMean, w.data, n, cols, cols, coupled, b);

  final pi = c.precision;
  final cep = c.conditionalExpectedPrecision;
  final ep = c.expectedPrecision;
  final eff = c.effectivePrecision;
  if (cl.isInputLayer || !predict) {
    cep.setAll(0, pi);
    ep.setAll(0, pi);
    eff.fillRange(0, n * b, 0.0);
    return;
  }

  // Value-coupling variance Σ_k W²·g'(μ̂)²/π̂ (the bias parent adds 0).
  Float64List? vcv;
  if (settings.feedforwardUncertainty) {
    final perParent = ws.take(_sPerParent, np);
    final g = pl.coupling;
    for (var k = 0; k < np * b; k++) {
      final d = g.df(p.expectedMean[k]);
      perParent[k] = d * d / p.expectedPrecision[k];
    }
    vcv = ws.take(_sVcv, n);
    _gemm(vcv, _squared(ws, w), n, np, cols, perParent, b);
  }
  for (var k = 0; k < n * b; k++) {
    final pv = predVol[k];
    final cond = 1.0 / _floor(1.0 / pi[k] + pv, minPredictedVariance);
    final e =
        1.0 /
        _floor(1.0 / cond + (vcv == null ? 0.0 : vcv[k]), minPredictedVariance);
    cep[k] = cond;
    ep[k] = e;
    eff[k] = pv * e;
  }
}

/// The elementwise square of [w] (recomputed every call: the weights
/// change between steps).
Float64List _squared(Workspace ws, Matrix w) {
  final out = ws.raw(_sSquared, w.data.length);
  for (var k = 0; k < out.length; k++) {
    final v = w.data[k];
    out[k] = v * v;
  }
  return out;
}

/// `vectorised_binary_prediction`.
void _predictBinary(
  LayerState c,
  LayerState p,
  Layer pl,
  Workspace ws,
  DeepSettings settings,
) {
  final n = c.nodes;
  final b = c.width;
  final w = pl.weights!;
  final coupled = _coupled(
    ws,
    p.expectedMean,
    p.nodes,
    pl.coupling,
    pl.hasConstant,
  );
  final em = c.expectedMean;
  _gemm(em, w.data, n, w.cols, w.cols, coupled, b);
  final lo = settings.precisionClippingValue;
  final hi = 1 - lo;
  for (var k = 0; k < n * b; k++) {
    final m = _clip(sigmoid(em[k]), lo, hi);
    em[k] = m;
    final e = m * (1 - m);
    c.expectedPrecision[k] = e;
    c.conditionalExpectedPrecision[k] = e;
  }
}

/// `vectorised_categorical_prediction`: softmax over the layer's nodes.
void _predictCategorical(LayerState c, LayerState p, Layer pl, Workspace ws) {
  final n = c.nodes;
  final b = c.width;
  final w = pl.weights!;
  final coupled = _coupled(
    ws,
    p.expectedMean,
    p.nodes,
    pl.coupling,
    pl.hasConstant,
  );
  final em = c.expectedMean;
  _gemm(em, w.data, n, w.cols, w.cols, coupled, b);
  for (var s = 0; s < b; s++) {
    var max = double.negativeInfinity;
    for (var i = 0; i < n; i++) {
      max = math.max(max, em[i * b + s]);
    }
    var sum = 0.0;
    for (var i = 0; i < n; i++) {
      final e = math.exp(em[i * b + s] - max);
      em[i * b + s] = e;
      sum += e;
    }
    for (var i = 0; i < n; i++) {
      em[i * b + s] = em[i * b + s] / sum;
    }
  }
  c.expectedPrecision.fillRange(0, n * b, 1.0);
  c.conditionalExpectedPrecision.fillRange(0, n * b, 1.0);
  c.effectivePrecision.fillRange(0, n * b, 0.0);
}

// ---------------------------------------------------------------------------
// Prediction errors and the volatility-level posteriors
// ---------------------------------------------------------------------------

/// The layer's prediction error, dispatched on its kind (`_leaf_pe`, and
/// the second half of `_posterior_pe_layer`).
void predictionError(
  LayerState state,
  Layer layer,
  double timeStep,
  DeepSettings settings,
) {
  final nb = state.nodes * state.width;
  final m = state.mean;
  final em = state.expectedMean;
  final vpe = state.valuePredictionError;
  switch (layer.kind) {
    case LayerKind.binary:
      for (var k = 0; k < nb; k++) {
        vpe[k] = (m[k] - em[k]) / state.expectedPrecision[k];
      }
      state.precision.setAll(0, state.expectedPrecision);
      return;
    case LayerKind.categorical:
      for (var k = 0; k < nb; k++) {
        vpe[k] = m[k] - em[k];
      }
      state.precision.setAll(0, state.expectedPrecision);
      return;
    case LayerKind.volatile:
      for (var k = 0; k < nb; k++) {
        vpe[k] = m[k] - em[k];
      }
      if (!layer.hasVolatility) return;
      final ep = state.expectedPrecision;
      final pi = state.precision;
      final dpe = state.volatilityPredictionError!;
      for (var k = 0; k < nb; k++) {
        final d = m[k] - em[k];
        dpe[k] = ep[k] / pi[k] + ep[k] * (d * d) - 1.0;
      }
      switch (settings.volatilityUpdates) {
        case DeepVolatilityUpdate.standard:
          _volatilityStandard(state, settings);
        case DeepVolatilityUpdate.ehgf:
          _volatilityEhgf(state, layer, timeStep, settings);
        case DeepVolatilityUpdate.unbounded:
          _volatilityUnbounded(state, layer, timeStep, settings);
      }
  }
}

/// `vectorised_layer_volatility_posterior_standard`.
void _volatilityStandard(LayerState s, DeepSettings settings) {
  final nb = s.nodes * s.width;
  final eff = s.effectivePrecision;
  final dpe = s.volatilityPredictionError!;
  final epv = s.expectedPrecisionVol!;
  final emv = s.expectedMeanVol!;
  final pv = s.precisionVol!;
  final mv = s.meanVol!;
  final maxPp = settings.maxPosteriorPrecision;
  for (var k = 0; k < nb; k++) {
    final g = eff[k];
    final d = dpe[k];
    final contrib = 0.5 * (g * g) + (g * g) * d - 0.5 * g * d;
    final p = math.min(epv[k] + contrib, maxPp);
    pv[k] = p;
    mv[k] = emv[k] + g * d / (2.0 * p);
  }
}

/// `max(1/π̂ − 1/π̄, 0)`: the variance the value parents add.
double _valueCouplingVariance(double ep, double cep) =>
    math.max(1.0 / ep - 1.0 / cep, 0.0);

/// `vectorised_layer_volatility_posterior_ehgf`.
void _volatilityEhgf(
  LayerState s,
  Layer layer,
  double t,
  DeepSettings settings,
) {
  final n = s.nodes;
  final b = s.width;
  final tonic = layer.tonicVolatility;
  final meanField = settings.meanFieldUpdates;
  final ffu = settings.feedforwardUncertainty;
  final maxPp = settings.maxPosteriorPrecision;
  final eff = s.effectivePrecision;
  final dpe = s.volatilityPredictionError!;
  final epv = s.expectedPrecisionVol!;
  final emv = s.expectedMeanVol!;
  final pv = s.precisionVol!;
  final mv = s.meanVol!;
  for (var i = 0; i < n; i++) {
    final omega = tonic == null ? 0.0 : tonic[i];
    for (var j = 0; j < b; j++) {
      final k = i * b + j;
      final e = epv[k];
      final postMean = emv[k] + eff[k] * dpe[k] / (2.0 * e);
      var logVol = omega + emv[k];
      if (!meanField) logVol = logVol + 1.0 / (2.0 * e);
      final predicted = t * math.exp(logVol);
      var previous = _floor(
        1.0 / s.conditionalExpectedPrecision[k] - predicted,
        1e-128,
      );
      if (ffu) {
        previous =
            previous +
            _valueCouplingVariance(
              s.expectedPrecision[k],
              s.conditionalExpectedPrecision[k],
            );
      }
      final repredicted = t * math.exp(omega + postMean);
      final expected = 1.0 / (previous + repredicted);
      final eff2 = repredicted * expected;
      final weight = (repredicted - previous) * expected;
      final d = s.mean[k] - s.expectedMean[k];
      final vpe2 = (1.0 / s.precision[k] + d * d) * expected - 1.0;
      final contrib = math.max(0.0, 0.5 * eff2 * (eff2 + weight * vpe2));
      pv[k] = math.min(e + contrib, maxPp);
      mv[k] = postMean;
    }
  }
}

/// `vectorised_layer_volatility_posterior_unbounded`.
void _volatilityUnbounded(
  LayerState s,
  Layer layer,
  double t,
  DeepSettings settings,
) {
  final n = s.nodes;
  final b = s.width;
  final tonic = layer.tonicVolatility;
  final meanField = settings.meanFieldUpdates;
  final ffu = settings.feedforwardUncertainty;
  final maxPp = settings.maxPosteriorPrecision;
  final epv = s.expectedPrecisionVol!;
  final emv = s.expectedMeanVol!;
  final pv = s.precisionVol!;
  final mv = s.meanVol!;
  final logT = math.log(t);
  for (var i = 0; i < n; i++) {
    final omega = tonic == null ? 0.0 : tonic[i];
    for (var j = 0; j < b; j++) {
      final k = i * b + j;
      final em = emv[k];
      final e = epv[k];
      var logVol = omega + em;
      if (!meanField) logVol = logVol + 1.0 / (2.0 * e);
      final predicted = t * math.exp(logVol);
      var previous = _floor(
        1.0 / s.conditionalExpectedPrecision[k] - predicted,
        1e-128,
      );
      if (ffu) {
        previous =
            previous +
            _valueCouplingVariance(
              s.expectedPrecision[k],
              s.conditionalExpectedPrecision[k],
            );
      }
      final d = s.mean[k] - s.expectedMean[k];
      final beAux = 1.0 / s.precision[k] + d * d;
      final logPrev = math.log(previous);
      final gammaC = logT + em + omega;
      final wJm1 = sigmoid(gammaC - logPrev);
      final daJm1 = s.expectedPrecision[k] * beAux - 1.0;

      // Expansion 1: quadratic at the prediction.
      final pi1 = e + 0.5 * wJm1 * (1.0 - wJm1);
      final mu1 = em + wJm1 / (2.0 * pi1) * daJm1;

      // Expansion 2: quadratic at the Lambert-W approximate mode.
      final logWArg = math.log(beAux) - math.log(2.0 * e) + 0.5 / e - gammaC;
      final wArg = math.exp(math.min(logWArg, _logFloatMax));
      final vW = lambertW0(wArg);
      final yStar = gammaC + vW - 0.5 / e;
      final xStar = yStar - logT - omega;
      final logS2 = logT + xStar + omega;
      final logDenomS = logAddExp(logPrev, logS2);
      final w2 = sigmoid(logS2 - logPrev);
      final da2 = beAux * math.exp(-logDenomS) - 1.0;
      final pi2Full = e + 0.5 * w2 * (w2 + (2.0 * w2 - 1.0) * da2);
      final pi2Safe = pi2Full <= 0.0 ? e + 0.5 * w2 * (1.0 - w2) : pi2Full;
      final mu2Safe = xStar + (0.5 * w2 * da2 - e * (xStar - em)) / pi2Safe;
      final finite2 = pi2Safe.isFinite && mu2Safe.isFinite;
      final pi2 = finite2 ? pi2Safe : pi1;
      final mu2 = finite2 ? mu2Safe : mu1;

      // Variational energy-based softmax blend.
      final logDenom1 = logAddExp(logPrev, logT + mu1 + omega);
      final i1 =
          -0.5 * logDenom1 -
          0.5 * beAux * math.exp(-logDenom1) -
          0.5 * e * ((mu1 - em) * (mu1 - em));
      final logDenom2 = logAddExp(logPrev, logT + mu2 + omega);
      final i2 =
          -0.5 * logDenom2 -
          0.5 * beAux * math.exp(-logDenom2) -
          0.5 * e * ((mu2 - em) * (mu2 - em));
      final blend = sigmoid(i2 - i1);

      // Gaussian mixture moment matching.
      mv[k] = (1.0 - blend) * mu1 + blend * mu2;
      final sig2 =
          (1.0 - blend) / pi1 +
          blend / pi2 +
          blend * (1.0 - blend) * ((mu1 - mu2) * (mu1 - mu2));
      pv[k] = math.min(1.0 / sig2, maxPp);
    }
  }
}

// ---------------------------------------------------------------------------
// Bottom-up posterior
// ---------------------------------------------------------------------------

/// The relaxed gain `π̄·π/(π̄ + π_y)` (or π̂ under mean-field updates) of
/// every child element, times its value prediction error, into [out].
void _gainTimesError(LayerState c, Float64List out, bool meanField) {
  final nb = c.nodes * c.width;
  for (var k = 0; k < nb; k++) {
    final double gain;
    if (meanField) {
      gain = c.expectedPrecision[k];
    } else {
      final piY = c.precision[k] - c.expectedPrecision[k];
      final cep = c.conditionalExpectedPrecision[k];
      gain = cep * c.precision[k] / (cep + piY);
    }
    out[k] = gain * c.valuePredictionError[k];
  }
}

/// Width 1: `pc1[j] = Σ_i w[i][j]²·eff[i]` and `msg[j] = Σ_i w[i][j]·gd[i]`
/// in one pass over the weight rows (each summed over `i` ascending).
@pragma('vm:unsafe:no-bounds-checks')
void _fusedPosteriorMessages(
  Float64List pc1,
  Float64List msg,
  Float64List w,
  int nc,
  int np,
  int stride,
  Float64List eff,
  Float64List gd,
) {
  assert(pc1.length >= np && msg.length >= np && eff.length >= nc);
  assert(gd.length >= nc && (nc == 0 || w.length >= (nc - 1) * stride + np));
  pc1.fillRange(0, np, 0.0);
  msg.fillRange(0, np, 0.0);
  for (var i = 0; i < nc; i++) {
    final e = eff[i];
    final g = gd[i];
    final row = i * stride;
    for (var j = 0; j < np; j++) {
      final wij = w[row + j];
      pc1[j] += wij * wij * e;
      msg[j] += wij * g;
    }
  }
}

/// `vectorised_layer_posterior_update`: the value-level posterior of
/// [parent] from [child]'s prediction errors.
void posteriorUpdate(
  LayerState parent,
  Layer parentLayer,
  LayerState child,
  Layer childLayer,
  Workspace ws,
  DeepSettings settings,
) {
  final w = parentLayer.weights!;
  final stride = w.cols;
  final nc = child.nodes;
  final np = parent.nodes;
  final b = parent.width;
  final meanField = settings.meanFieldUpdates;

  // Child-side factors.
  final effc = ws.take(_sEff, nc);
  final gd = ws.take(_sGain, nc);
  for (var k = 0; k < nc * b; k++) {
    if (meanField) {
      effc[k] = child.expectedPrecision[k];
    } else {
      final piY = child.precision[k] - child.expectedPrecision[k];
      final cep = child.conditionalExpectedPrecision[k];
      effc[k] = childLayer.isInputLayer ? cep : cep * piY / (cep + piY);
    }
  }
  _gainTimesError(child, gd, meanField);

  // Contractions over the children (bias column left out).
  final pc1 = ws.take(_sPc1, np);
  final msg = ws.take(_sMsg, np);
  if (b == 1) {
    _fusedPosteriorMessages(pc1, msg, w.data, nc, np, stride, effc, gd);
  } else {
    _gemmT(pc1, _squared(ws, w), nc, np, stride, effc, b);
    _gemmT(msg, w.data, nc, np, stride, gd, b);
  }

  final g = parentLayer.coupling;
  final maxPp = settings.maxPosteriorPrecision;
  for (var k = 0; k < np * b; k++) {
    final x = parent.expectedMean[k];
    final g1 = g.df(x);
    final g2 = g.d2f(x);
    final ep = parent.expectedPrecision[k];
    final raw = ep + pc1[k] * (g1 * g1) + (-g2 * msg[k]);
    final p = _clip(raw, ep, maxPp);
    parent.precision[k] = p;
    parent.mean[k] = x + msg[k] * g1 / p;
  }
}

// ---------------------------------------------------------------------------
// Sweeps
// ---------------------------------------------------------------------------

/// Top-down prediction sweep: clamp rows `[start, start + width)` of [x] on
/// the top layer, then predict every layer from the one above.
void predictionSweep(
  List<Layer> layers,
  List<LayerState> states,
  Matrix x,
  int start,
  Workspace ws,
  double timeStep,
  DeepSettings settings,
) {
  final top = states.last;
  final b = top.width;
  for (var i = 0; i < top.nodes; i++) {
    for (var s = 0; s < b; s++) {
      final v = x.data[(start + s) * x.cols + i];
      top.expectedMean[i * b + s] = v;
      top.mean[i * b + s] = v;
    }
  }
  for (var i = layers.length - 1; i > 0; i--) {
    predictLayer(
      states[i - 1],
      layers[i - 1],
      states[i],
      layers[i],
      ws,
      timeStep,
      settings,
    );
  }
}

/// Bottom-up sweep: clamp rows `[start, start + width)` of [y] on the
/// bottom layer, then the leaf prediction error and, on every interior
/// layer, the posterior update and its prediction error. The top layer
/// stays clamped.
void updateSweep(
  List<Layer> layers,
  List<LayerState> states,
  Matrix y,
  int start,
  Workspace ws,
  double timeStep,
  DeepSettings settings,
) {
  final bottom = states.first;
  final b = bottom.width;
  for (var i = 0; i < bottom.nodes; i++) {
    for (var s = 0; s < b; s++) {
      bottom.mean[i * b + s] = y.data[(start + s) * y.cols + i];
    }
  }
  predictionError(bottom, layers.first, timeStep, settings);
  for (var i = 1; i < layers.length - 1; i++) {
    posteriorUpdate(
      states[i],
      layers[i],
      states[i - 1],
      layers[i - 1],
      ws,
      settings,
    );
    predictionError(states[i], layers[i], timeStep, settings);
  }
}

/// `_input_prediction_error`: the error message the top layer receives from
/// the layer below, written to rows `[start, start + width)` of [out]
/// (`samples × topSize`).
void inputPredictionError(
  List<Layer> layers,
  List<LayerState> states,
  Workspace ws,
  DeepSettings settings,
  Matrix out,
  int start,
) {
  final topLayer = layers.last;
  final top = states.last;
  final child = states[states.length - 2];
  final w = topLayer.weights!;
  final b = top.width;
  final gd = ws.take(_sGain, child.nodes);
  _gainTimesError(child, gd, settings.meanFieldUpdates);
  final msg = ws.take(_sMsg, top.nodes);
  _gemmT(msg, w.data, child.nodes, top.nodes, w.cols, gd, b);
  final g = topLayer.coupling;
  for (var j = 0; j < top.nodes; j++) {
    for (var s = 0; s < b; s++) {
      final k = j * b + s;
      out.data[(start + s) * out.cols + j] = msg[k] * g.df(top.expectedMean[k]);
    }
  }
}

/// The descent weight gradient of every layer, averaged over the samples
/// and added into [grads] (`grads[i]` matches `layers[i].weights`; the
/// bottom entry is `null`): `Σ_s u[i][s]·h[j][s] / width`.
void accumulateGradients(
  List<Layer> layers,
  List<LayerState> states,
  Workspace ws,
  LearningKind kind,
  List<Float64List?> grads,
) {
  for (var l = 1; l < layers.length; l++) {
    final pl = layers[l];
    final cl = layers[l - 1];
    final p = states[l];
    final c = states[l - 1];
    final b = c.width;
    final nc = c.nodes;
    final cols = pl.weights!.cols;
    final weighted =
        kind != LearningKind.standard && cl.kind != LayerKind.binary;
    final u = ws.take(_sGain, nc);
    for (var k = 0; k < nc * b; k++) {
      var v = c.mean[k] - c.expectedMean[k];
      if (weighted) v = v * c.precision[k];
      u[k] = -(_finite(v) ? v : 0.0);
    }
    final h = _coupled(
      ws,
      p.mean,
      p.nodes,
      pl.coupling,
      pl.hasConstant,
      sanitize: true,
    );
    _addOuter(grads[l]!, u, h, nc, cols, b, ws);
  }
}

/// `g[i][j] += Σ_s u[i][s] · h[j][s] / b` (a batch-mean outer product).
///
/// Wider batches transpose `h` to `[s][j]` (padded to an even width) and
/// run the sum as a [matMul] with the samples as its depth: each sum still
/// runs over `s` ascending from zero, so the bits are those of the plain
/// dot-product loop, but through the register (and SIMD) tiles instead of
/// one latency-bound accumulator.
@pragma('vm:unsafe:no-bounds-checks')
void _addOuter(
  Float64List g,
  Float64List u,
  Float64List h,
  int nc,
  int cols,
  int b,
  Workspace ws,
) {
  assert(g.length >= nc * cols && u.length >= nc * b && h.length >= cols * b);
  if (b == 1) {
    for (var i = 0; i < nc; i++) {
      final ui = u[i];
      final gi = i * cols;
      for (var j = 0; j < cols; j++) {
        g[gi + j] += ui * h[j];
      }
    }
    return;
  }
  final width = cols + (cols & 1);
  final ht = ws.atLeast(_sHt, b * width);
  for (var j = 0; j < cols; j++) {
    final hj = j * b;
    for (var s = 0; s < b; s++) {
      ht[s * width + j] = h[hj + s];
    }
  }
  if (width != cols) {
    for (var s = 0; s < b; s++) {
      ht[s * width + cols] = 0.0;
    }
  }
  final sums = ws.atLeast(_sOuter, nc * width);
  matMul(sums, u, b, 1, nc, b, ht, width);
  for (var i = 0; i < nc; i++) {
    final gi = i * cols;
    final si = i * width;
    for (var j = 0; j < cols; j++) {
      g[gi + j] += sums[si + j] / b;
    }
  }
}

/// `carry_precisions`: replace each volatile layer's posterior precision by
/// the one it carries into the next sample.
void carryPrecisions(
  List<Layer> layers,
  List<LayerState> states,
  DeepSettings settings,
) {
  if (settings.carryPrecision && !settings.feedforwardUncertainty) return;
  for (var l = 0; l < layers.length; l++) {
    final layer = layers[l];
    if (layer.kind != LayerKind.volatile) continue;
    final s = states[l];
    final b = s.width;
    for (var i = 0; i < s.nodes; i++) {
      for (var j = 0; j < b; j++) {
        final k = i * b + j;
        s.precision[k] = settings.carryPrecision
            ? s.precision[k] +
                  math.max(
                    s.conditionalExpectedPrecision[k] - s.expectedPrecision[k],
                    0.0,
                  )
            : layer.priorPrecision[i];
      }
    }
  }
}
