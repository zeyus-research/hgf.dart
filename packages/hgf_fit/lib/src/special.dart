import 'dart:math' as math;

const double _twoOverSqrtPi = 1.1283791670955126;
const double _sqrt2 = 1.4142135623730951;

/// The complementary error function `erfc(x) = 1 − erf(x)`, to about 1e-14
/// relative.
///
/// A Taylor series for `|x| < 2.5` and Lentz's continued fraction beyond, so
/// the far tail keeps full relative precision (`erfc(10) ≈ 2.1e-45`).
double erfc(double x) {
  if (x.isNaN) return double.nan;
  if (x < 0) return 2.0 - erfc(-x);
  if (x < 2.5) return 1.0 - _erfSeries(x);
  if (x > 27.0) return 0.0;
  return math.exp(-x * x) * _erfcContinuedFraction(x);
}

/// The error function.
double erf(double x) {
  if (x.isNaN) return double.nan;
  if (x.abs() < 2.5) return _erfSeries(x);
  return x > 0 ? 1.0 - erfc(x) : erfc(-x) - 1.0;
}

/// `erf(x) = 2/√π Σ (−1)ⁿ x^(2n+1) / (n! (2n+1))`.
double _erfSeries(double x) {
  final x2 = x * x;
  var term = x, sum = x;
  for (var n = 1; n < 200; n++) {
    term *= -x2 / n;
    final add = term / (2 * n + 1);
    sum += add;
    if (add.abs() < 1e-17 * sum.abs()) break;
  }
  return _twoOverSqrtPi * sum;
}

/// `exp(x²) erfc(x)` for `x ≥ 2.5`, by the continued fraction
/// `1/√π · 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + …))))`, evaluated with
/// modified Lentz.
double _erfcContinuedFraction(double x) {
  const tiny = 1e-300;
  var f = x, c = x, d = 0.0;
  for (var n = 1; n < 500; n++) {
    final a = n / 2.0;
    d = x + a * d;
    d = d == 0 ? tiny : 1.0 / d;
    c = x + a / c;
    if (c == 0) c = tiny;
    final delta = c * d;
    f *= delta;
    if ((delta - 1.0).abs() < 1e-16) break;
  }
  return 1.0 / (f * 1.7724538509055159); // √π
}

/// `log Φ(z)`, the log of the standard normal CDF, accurate in the lower
/// tail.
double logNormalCdf(double z) {
  if (z.isNaN) return double.nan;
  if (z == double.negativeInfinity) return double.negativeInfinity;
  if (z > 5.0) return _log1p(-0.5 * erfc(z / _sqrt2));
  if (z > -2.5 * _sqrt2) return math.log(0.5 * erfc(-z / _sqrt2));
  final u = -z / _sqrt2;
  return -u * u + math.log(0.5 * _erfcContinuedFraction(u));
}

/// `log(Φ(b) − Φ(a))` for `a < b`, without cancellation in either tail.
double logNormalCdfDiff(double a, double b) {
  if (!(a < b)) return double.negativeInfinity;
  if (a > 0) return logNormalCdfDiff(-b, -a);
  final lb = logNormalCdf(b), la = logNormalCdf(a);
  if (la == double.negativeInfinity) return lb;
  return lb + _log1p(-math.exp(la - lb));
}

double _log1p(double x) {
  final u = 1.0 + x;
  if (u == 1.0) return x;
  return math.log(u) * x / (u - 1.0);
}

/// `log(1 + exp(x))` without overflow.
double softplus(double x) =>
    x > 0 ? x + _log1p(math.exp(-x)) : _log1p(math.exp(x));

/// The standard normal quantile `Φ⁻¹(p)`, to a few ulp.
///
/// Acklam's rational approximation (1e-9), refined by one Halley step on
/// `erfc`.
double normalQuantile(double p) {
  if (p.isNaN || p < 0 || p > 1) return double.nan;
  if (p == 0) return double.negativeInfinity;
  if (p == 1) return double.infinity;
  // 1 − p is exact for p ≥ 0.5, so the upper half keeps full precision.
  if (p > 0.5) return -normalQuantile(1 - p);
  const a = [
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00,
  ];
  const b = [
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01,
  ];
  const c = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00,
  ];
  const d = [
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00,
  ];
  const pLow = 0.02425;
  double x;
  if (p < pLow) {
    final q = math.sqrt(-2 * math.log(p));
    x =
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else {
    final q = p - 0.5, r = q * q;
    x =
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
        q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  // Halley: e = Φ(x) − p, u = e √(2π) exp(x²/2).
  final e = 0.5 * erfc(-x / _sqrt2) - p;
  final u = e * 2.5066282746310002 * math.exp(x * x / 2);
  return x - u / (1 + x * u / 2);
}

/// A standard normal draw (Box–Muller).
double standardNormal(math.Random rng) {
  double u;
  do {
    u = rng.nextDouble();
  } while (u == 0.0);
  return math.sqrt(-2.0 * math.log(u)) *
      math.cos(2 * math.pi * rng.nextDouble());
}
