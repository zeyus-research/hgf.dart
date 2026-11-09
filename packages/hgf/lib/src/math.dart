import 'dart:math' as math;
import 'dart:typed_data';

/// Whether this is a dart2wasm build (a compile-time constant).
const bool isWasm = bool.fromEnvironment('dart.tool.dart2wasm');

/// `exp(x)` for the engine's hot path.
///
/// On dart2wasm, `math.exp` is a call into JavaScript's `Math.exp`: about
/// 11 ns per call, 25 ns when each call waits on the previous one (as in
/// Halley iterations), versus 6 and 10 ns on AOT. The table-driven
/// [tableExp] runs in Wasm itself at 8 and 17 ns, so dart2wasm builds use
/// it; on AOT it is slower than the native `exp` and `math.exp` stays.
/// Measured by `benchmark/bin/runtime_probe.dart`.
@pragma('vm:prefer-inline')
@pragma('wasm:prefer-inline')
@pragma('dart2js:prefer-inline')
double fastExp(double x) => isWasm ? tableExp(x) : math.exp(x);

/// `exp(x)` from a 128-entry table of 2^(j/128) and a degree-5 polynomial
/// (the scheme of musl's `exp`, without bit casts): within 1 ulp of
/// `math.exp`. Outside |x| ≤ 708 (overflow, subnormal results, ±∞, NaN) it
/// defers to `math.exp`.
@pragma('vm:prefer-inline')
@pragma('wasm:prefer-inline')
@pragma('vm:unsafe:no-bounds-checks')
double tableExp(double x) {
  if (!(x.abs() <= 708.0)) return math.exp(x);
  const shift = 6755399441055744.0; // 1.5·2^52: rounds to nearest even
  final kd = (x * _invLn2N + shift) - shift;
  final ki = kd.toInt(); // |ki| ≤ 130740
  final r = (x - kd * _ln2HiN) - kd * _ln2LoN; // |r| ≤ ln2/256
  final p = r + r * r * (0.5 + r * (_c3 + r * (_c4 + r * _c5)));
  // 2^(ki/128) = 2^(ki >> 7) · 2^((ki & 127)/128); ki >> 7 is in
  // [-1022, 1021] for |x| ≤ 708, so both indexes are in range.
  final scale = _pow2[(ki >> 7) + 1022] * _exp2Frac[ki & 127];
  return scale + scale * p;
}

// Taylor coefficients 1/3!, 1/4!, 1/5!, as double literals: dart2wasm does
// not fold `1 / 6` on int literals and would convert at run time.
const double _c3 = 0.16666666666666666;
const double _c4 = 0.041666666666666664;
const double _c5 = 0.008333333333333333;

const double _invLn2N = 184.6649652337873; // 128 / ln 2
// Cody-Waite split of ln2/128: the high part has 21 trailing zero bits, so
// kd·_ln2HiN is exact for |kd| < 2^21.
const double _ln2HiN = 0.00541521234663378; // 0.693147180369123816490 / 128
const double _ln2LoN =
    1.4907929134926466e-12; // 1.90821492927058770002e-10 / 128

/// 2^(j/128), j = 0..127. (Top-level finals are lazily initialised: each read checks
/// once; dart2wasm's `wasm:initialize-at-startup` is SDK-internal.)
final Float64List _exp2Frac = () {
  final t = Float64List(128);
  for (var j = 0; j < 128; j++) {
    t[j] = math.pow(2.0, j / 128).toDouble();
  }
  return t;
}();

/// 2^k, k = -1022..1023 (index k + 1022); exact powers of two.
final Float64List _pow2 = () {
  final t = Float64List(2046);
  var v = 1.0;
  for (var k = 0; k <= 1023; k++, v *= 2.0) {
    t[k + 1022] = v;
  }
  v = 1.0;
  for (var k = 0; k >= -1022; k--, v *= 0.5) {
    t[k + 1022] = v;
  }
  return t;
}();

/// Floor on the total predicted variance (carried + volatility + value
/// coupling). Mirrors `MIN_PREDICTED_VARIANCE` in pyhgf.
const double minPredictedVariance = 1e-128;

/// Lower bound applied to posterior precisions before the upper clip.
const double minPosteriorPrecision = 1e-128;

/// Logistic sigmoid.
@pragma('vm:prefer-inline')
@pragma('wasm:prefer-inline')
double sigmoid(double x) => 1.0 / (1.0 + fastExp(-x));

/// `ln(exp(a) + exp(b))`, computed stably (mirrors `jnp.logaddexp`).
@pragma('vm:prefer-inline')
@pragma('wasm:prefer-inline')
double logAddExp(double a, double b) {
  final m = a > b ? a : b;
  // With an infinite m the general formula would produce inf - inf = NaN; the
  // answer is m itself (-inf only when both arguments are -inf).
  if (m.isInfinite) return m;
  return m + math.log(fastExp(a - m) + fastExp(b - m));
}

/// The complementary error function, to about 1e-15 relative.
///
/// Below |x| = 1.5 it is `1 − erf(x)` with erf from its positive-term series
/// `erf(x) = 2/√π · e^(−x²) · Σ 2ⁿ x^(2n+1) / (1·3·…·(2n+1))` (no
/// cancellation); above, the continued fraction
/// `erfc(x) = e^(−x²)/√π · 1/(x + ½/(x + 1/(x + 3/2/(x + …))))`, evaluated
/// with Lentz's method. Negative arguments use `erfc(−x) = 2 − erfc(x)`.
double erfc(double x) {
  if (x.isNaN) return x;
  if (x < 0) return 2.0 - erfc(-x);
  if (x < 1.5) {
    final x2 = x * x;
    var term = x;
    var sum = x;
    for (var n = 1; n < 100; n++) {
      term *= 2.0 * x2 / (2 * n + 1);
      sum += term;
      if (term < sum * 1e-17) break;
    }
    return 1.0 - _twoOverSqrtPi * math.exp(-x2) * sum;
  }
  if (x > 27.3) return 0.0; // e^(−x²) underflows
  // Modified Lentz for b0 + a1/(b1 + a2/(b2 + …)) with b_n = x, a_n = n/2.
  const tiny = 1e-300;
  var f = x;
  var c = x;
  var d = 0.0;
  for (var n = 1; n < 500; n++) {
    final a = n * 0.5;
    d = x + a * d;
    if (d.abs() < tiny) d = tiny;
    c = x + a / c;
    if (c.abs() < tiny) c = tiny;
    d = 1.0 / d;
    final delta = c * d;
    f *= delta;
    if ((delta - 1.0).abs() < 1e-16) break;
  }
  return math.exp(-x * x) / (f * _sqrtPi);
}

const double _sqrtPi = 1.7724538509055159;
const double _twoOverSqrtPi = 1.1283791670955126;

/// Principal branch of the Lambert W function for `z >= 0`.
///
/// Halley iterations from `ln(z + 1)`, the scheme of `pyhgf.math.lambert_w0`
/// and rshgf (at most six), so the backends agree to machine precision.
/// Unlike pyhgf it stops once a step is below `1e-12·|w|`: Halley converges
/// cubically, so the remaining steps would only move `w` by rounding
/// (and each costs an exp).
double lambertW0(double z) {
  var w = math.log(z + 1.0);
  for (var i = 0; i < 6; i++) {
    final ew = fastExp(w);
    final f = w * ew - z;
    final f1 = (w + 1.0) * ew;
    final f2 = (w + 2.0) * ew;
    final step = (2.0 * f * f1) / (2.0 * f1 * f1 - f * f2);
    w -= step;
    if (step.abs() <= 1e-12 * w.abs()) break;
  }
  return w;
}

/// `max(variance, minPredictedVariance)` that propagates NaN (like
/// `jnp.maximum`), so the volatility underflow guard stays visible.
@pragma('vm:prefer-inline')
@pragma('wasm:prefer-inline')
double floorPredictedVariance(double variance) {
  if (variance.isNaN) return double.nan;
  return variance > minPredictedVariance ? variance : minPredictedVariance;
}

/// Largest finite double's natural log (`f64::MAX.ln()`).
const double logMaxFinite = 709.782712893384;

/// `ln(2π)`.
const double _log2Pi = 1.8378770664093453;

/// Surprise (negative log density) of [x] under a Gaussian prediction with
/// [expectedMean] and [expectedPrecision]:
/// `½ (ln 2π − ln π̂ + π̂ (x − μ̂)²)`. Mirrors `pyhgf.math.gaussian_surprise`.
@pragma('vm:prefer-inline')
@pragma('wasm:prefer-inline')
double gaussianSurprise(
  double x,
  double expectedMean,
  double expectedPrecision,
) {
  final d = x - expectedMean;
  return 0.5 *
      (_log2Pi - math.log(expectedPrecision) + expectedPrecision * d * d);
}

/// Surprise at a binary outcome [x] (any non-zero value counts as 1) under
/// the Bernoulli probability [expectedMean]: `−ln μ̂` if `x = 1`, `−ln(1 − μ̂)`
/// otherwise. With [clipping] (the default), μ̂ is first clipped to
/// `[1e-6, 1 − 1e-6]`. Mirrors `pyhgf.math.binary_surprise`.
@pragma('vm:prefer-inline')
@pragma('wasm:prefer-inline')
double binarySurprise(double x, double expectedMean, {bool clipping = true}) {
  var m = expectedMean;
  if (clipping) {
    if (m < 1e-6) {
      m = 1e-6;
    } else if (m > 1.0 - 1e-6) {
      m = 1.0 - 1e-6;
    }
  }
  return x != 0.0 ? -math.log(m) : -math.log(1.0 - m);
}

/// `∂ binarySurprise(x, m) / ∂m`, with its clipping: zero where the clip is
/// active.
@pragma('vm:prefer-inline')
@pragma('wasm:prefer-inline')
double binarySurpriseDerivative(
  double x,
  double expectedMean, {
  bool clipping = true,
}) {
  final m = expectedMean;
  if (clipping && (m < 1e-6 || m > 1.0 - 1e-6)) return 0.0;
  return x != 0.0 ? -1.0 / m : 1.0 / (1.0 - m);
}

/// Gaussian density at [x] as pyhgf defines it (`pyhgf.math.gaussian_density`):
/// `π / √(2π) · exp(−π/2 (x − μ)²)`.
///
/// Note the leading factor is the precision itself, not its square root, as
/// in pyhgf; this is what [binarySurpriseFinitePrecision] uses.
double gaussianDensity(double x, double mean, double precision) {
  final d = x - mean;
  return precision /
      math.sqrt(2.0 * math.pi) *
      math.exp(-precision / 2.0 * d * d);
}

/// Binary surprise when the outcome is observed through Gaussian noise of
/// [expectedPrecision] around the two values [eta0] and [eta1]. Mirrors
/// `pyhgf.math.binary_surprise_finite_precision`.
double binarySurpriseFinitePrecision(
  double value,
  double expectedMean,
  double expectedPrecision, {
  double eta0 = 0.0,
  double eta1 = 1.0,
}) => -math.log(
  expectedMean * gaussianDensity(value, eta1, expectedPrecision) +
      (1.0 - expectedMean) * gaussianDensity(value, eta0, expectedPrecision),
);

/// `x^β / (x^β + (1 − x)^β)`: a belief [x] in `(0, 1)` sharpened (β > 1) or
/// flattened (β < 1) by the inverse temperature β. Mirrors
/// `pyhgf.math.sigmoid_inverse_temperature`.
///
/// Computed as `σ(β · logit x)`, the same function with one `log` and one
/// `exp` instead of two non-integer `pow`s: about 3x faster, which matters
/// when it scores every trial of a fit. Agrees with the `pow` form to a few
/// ulp; at `x = 0` or `1` it returns `0` or `1` as that form does.
double sigmoidInverseTemperature(double x, double inverseTemperature) {
  if (inverseTemperature == 0.0) return x.isNaN ? x : 0.5;
  return 1.0 / (1.0 + math.exp(-inverseTemperature * math.log(x / (1.0 - x))));
}

/// Lanczos coefficients (g = 7, n = 9) for [logGamma].
const List<double> _lanczos = [
  0.99999999999980993,
  676.5203681218851,
  -1259.1392167224028,
  771.32342877765313,
  -176.61502916214059,
  12.507343278686905,
  -0.13857109526572012,
  9.9843695780195716e-6,
  1.5056327351493116e-7,
];

/// `½ ln(2π)`.
const double _halfLog2Pi = 0.9189385332046728;

/// `ln |Γ(x)|`: the Lanczos approximation for `x ≥ ½`, the reflection
/// formula below. `+∞` at the poles (0, −1, −2, ...) and at `±∞`; the sign of
/// Γ is [gammaSign].
double logGamma(double x) {
  if (x.isNaN) return x;
  if (x.isInfinite) return double.infinity;
  if (x < 0.5) {
    final fl = x.floorToDouble();
    if (x == fl) return double.infinity;
    // Γ(x) Γ(1 − x) = π / sin(πx); |sin(πx)| = sin(π (x − ⌊x⌋)).
    return math.log(math.pi / math.sin(math.pi * (x - fl))) - logGamma(1.0 - x);
  }
  final z = x - 1.0;
  var a = _lanczos[0];
  for (var i = 1; i < _lanczos.length; i++) {
    a += _lanczos[i] / (z + i);
  }
  final t = z + 7.5;
  return _halfLog2Pi + (z + 0.5) * math.log(t) - t + math.log(a);
}

/// The sign of Γ(x): `1` for `x > 0`, `(−1)^⌈−x⌉` for negative non-integers,
/// NaN at the poles, at `−∞` and for NaN.
double gammaSign(double x) {
  if (x > 0.0) return 1.0;
  final fl = x.floorToDouble();
  if (x.isNaN || x.isInfinite || x == fl) return double.nan;
  return fl % 2.0 == 0.0 ? 1.0 : -1.0;
}

// Asymptotic-series coefficients of [digamma] (B₂ₖ / 2k).
const double _psi1 = 1.0 / 12.0;
const double _psi2 = 1.0 / 120.0;
const double _psi3 = 1.0 / 252.0;
const double _psi4 = 1.0 / 240.0;
const double _psi5 = 1.0 / 132.0;
const double _psi6 = 691.0 / 32760.0;
const double _psi7 = 1.0 / 12.0;

/// The digamma function ψ(x) = d/dx ln Γ(x): the recurrence
/// `ψ(x) = ψ(x + 1) − 1/x` up to `x ≥ 10`, then the asymptotic series, and
/// the reflection formula for `x ≤ 0`. NaN at the poles (0, −1, ...) and at
/// `−∞`.
double digamma(double x) {
  if (x.isNaN || x == double.negativeInfinity) return double.nan;
  if (x == double.infinity) return x;
  var result = 0.0;
  if (x <= 0.0) {
    final fl = x.floorToDouble();
    if (x == fl) return double.nan;
    // ψ(x) = ψ(1 − x) − π / tan(πx), and tan has period π.
    result = -math.pi / math.tan(math.pi * (x - fl));
    x = 1.0 - x;
  }
  while (x < 10.0) {
    result -= 1.0 / x;
    x += 1.0;
  }
  final inv2 = 1.0 / (x * x);
  return result +
      math.log(x) -
      0.5 / x -
      inv2 *
          (_psi1 -
              inv2 *
                  (_psi2 -
                      inv2 *
                          (_psi3 -
                              inv2 *
                                  (_psi4 -
                                      inv2 *
                                          (_psi5 -
                                              inv2 *
                                                  (_psi6 - inv2 * _psi7))))));
}

/// `ln(Γ(a) / Γ(b))`, NaN when the ratio is negative (as `jnp.log` of it).
double _logGammaRatio(double a, double b) {
  final sa = gammaSign(a), sb = gammaSign(b);
  if (sa != sb) return double.nan; // also when either is NaN
  return logGamma(a) - logGamma(b);
}

/// KL divergence `KL[P ‖ Q]` of the Dirichlet distribution `Q = Dir(alpha2)`
/// from `P = Dir(alpha1)`. Mirrors `pyhgf.math.dirichlet_kullback_leibler`,
/// in log space: pyhgf takes the log of a ratio of Γ values, which is NaN
/// once a Γ overflows (concentrations summing to more than ~171); here those
/// cases stay finite.
double dirichletKullbackLeibler(List<double> alpha1, List<double> alpha2) {
  if (alpha1.length != alpha2.length) {
    throw ArgumentError('alpha1 and alpha2 must have the same length');
  }
  final a1 = Float64List.fromList(alpha1), a2 = Float64List.fromList(alpha2);
  return dirichletKlAt(a1, 0, a2, 0, a1.length);
}

/// [dirichletKullbackLeibler] of `alpha1[o1 ..< o1 + n]` and
/// `alpha2[o2 ..< o2 + n]`, for the engine (not exported). The three terms
/// are summed separately, as in pyhgf.
double dirichletKlAt(
  Float64List alpha1,
  int o1,
  Float64List alpha2,
  int o2,
  int n,
) {
  var sum1 = 0.0, sum2 = 0.0;
  for (var i = 0; i < n; i++) {
    sum1 += alpha1[o1 + i];
    sum2 += alpha2[o2 + i];
  }
  final psiSum1 = digamma(sum1);
  var gammas = 0.0, digammas = 0.0;
  for (var i = 0; i < n; i++) {
    final a = alpha1[o1 + i], b = alpha2[o2 + i];
    gammas += _logGammaRatio(b, a);
    digammas += (a - b) * (digamma(a) - psiSum1);
  }
  return _logGammaRatio(sum1, sum2) + gammas + digammas;
}
