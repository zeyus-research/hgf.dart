import 'dart:math' as math;

import 'package:hgf/hgf.dart' show logGamma;

import 'codec.dart' show jsonDouble;
import 'special.dart';
import 'transform.dart';

const double _logSqrt2Pi = 0.9189385332046727;

/// A prior distribution over one parameter, on its natural (constrained)
/// scale.
///
/// [logDensity] includes the normalising constant, so log posteriors from
/// different models are comparable. [lower] and [upper] give the support;
/// the optimiser and the samplers work on an unconstrained scale through
/// [defaultTransform].
abstract class Prior {
  const Prior();

  /// Log probability density at [x] (`-∞` outside the support).
  double logDensity(double x);

  /// `d logDensity / dx` at [x] inside the support (for exact gradients).
  /// The default takes central differences; the built-in priors are exact.
  double dLogDensity(double x) {
    final h = 1e-6 * (x.abs() > 1 ? x.abs() : 1.0);
    return (logDensity(x + h) - logDensity(x - h)) / (2 * h);
  }

  /// `d² logDensity / dx²` at [x] inside the support (for exact Hessians).
  /// The default takes central differences of [dLogDensity]; the built-in
  /// priors are exact.
  double d2LogDensity(double x) {
    final h = 1e-5 * (x.abs() > 1 ? x.abs() : 1.0);
    return (dLogDensity(x + h) - dLogDensity(x - h)) / (2 * h);
  }

  /// Lower end of the support (`-∞` if unbounded).
  double get lower => double.negativeInfinity;

  /// Upper end of the support (`+∞` if unbounded).
  double get upper => double.infinity;

  /// A draw from the prior.
  double sample(math.Random rng);

  /// A central value: the starting point of an optimisation unless given.
  double get typical;

  /// The transform to the unconstrained scale matching the support.
  Transform get defaultTransform => Transform.forSupport(lower, upper);

  /// Whether [x] lies inside the support.
  bool contains(double x) => x >= lower && x <= upper;

  /// This prior as JSON, with a `"type"` key, for `FitCodec` (e.g. to
  /// send a model to a Web Worker). The built-in ones are covered; a custom
  /// one overrides this and gives the `FitCodec` a decoder for its type.
  Map<String, Object?> toJson() => throw UnsupportedError(
    '$runtimeType can\'t be serialised: override toJson() and give '
    'FitCodec a decoder for it',
  );
}

/// Normal prior `N(mean, sd²)`.
final class Normal extends Prior {
  const Normal(this.mean, this.sd) : assert(sd > 0);

  final double mean;
  final double sd;

  @override
  double logDensity(double x) {
    final z = (x - mean) / sd;
    return -0.5 * z * z - math.log(sd) - _logSqrt2Pi;
  }

  @override
  double dLogDensity(double x) => -(x - mean) / (sd * sd);

  @override
  double d2LogDensity(double x) => -1 / (sd * sd);

  @override
  double sample(math.Random rng) => mean + sd * standardNormal(rng);

  @override
  double get typical => mean;

  @override
  String toString() => 'Normal($mean, $sd)';

  @override
  Map<String, Object?> toJson() => {
    'type': 'Normal',
    'mean': jsonDouble(mean),
    'sd': jsonDouble(sd),
  };
}

/// Normal prior `N(mean, sd²)` truncated to `[lower, upper]` (either may be
/// infinite) and renormalised.
///
/// PyMC's `Censored` normal piles the mass outside the bounds onto them; for
/// an interior parameter value the two only differ by a constant.
final class TruncatedNormal extends Prior {
  TruncatedNormal(
    this.mean,
    this.sd, {
    this.lower = double.negativeInfinity,
    this.upper = double.infinity,
  }) : assert(sd > 0),
       assert(lower < upper),
       _logZ =
           math.log(sd) +
           _logSqrt2Pi +
           logNormalCdfDiff((lower - mean) / sd, (upper - mean) / sd);

  final double mean;
  final double sd;
  @override
  final double lower;
  @override
  final double upper;
  final double _logZ;

  @override
  double logDensity(double x) {
    if (!(x >= lower && x <= upper)) return double.negativeInfinity;
    final z = (x - mean) / sd;
    return -0.5 * z * z - _logZ;
  }

  @override
  double dLogDensity(double x) => -(x - mean) / (sd * sd);

  @override
  double d2LogDensity(double x) => -1 / (sd * sd);

  @override
  double sample(math.Random rng) {
    for (var i = 0; i < 10000; i++) {
      final x = mean + sd * standardNormal(rng);
      if (x >= lower && x <= upper) return x;
    }
    // The bulk lies far outside the bounds: fall back to the bound nearest
    // the mean, nudged inside.
    return (mean < lower ? lower : upper) +
        (mean < lower ? 1e-9 : -1e-9) * (1 + mean.abs());
  }

  @override
  double get typical {
    if (mean > lower && mean < upper) return mean;
    // The mode is a bound; start a little inside it.
    final step = math.min(
      sd,
      upper.isFinite && lower.isFinite ? 0.1 * (upper - lower) : sd,
    );
    return mean <= lower ? lower + step : upper - step;
  }

  @override
  String toString() => 'TruncatedNormal($mean, $sd, [$lower, $upper])';

  @override
  Map<String, Object?> toJson() => {
    'type': 'TruncatedNormal',
    'mean': jsonDouble(mean),
    'sd': jsonDouble(sd),
    'lower': jsonDouble(lower),
    'upper': jsonDouble(upper),
  };
}

/// Uniform prior on `[lower, upper]`.
final class Uniform extends Prior {
  const Uniform(this.lower, this.upper) : assert(lower < upper);

  @override
  final double lower;
  @override
  final double upper;

  @override
  double logDensity(double x) => x >= lower && x <= upper
      ? -math.log(upper - lower)
      : double.negativeInfinity;

  @override
  double dLogDensity(double x) => 0.0;

  @override
  double d2LogDensity(double x) => 0.0;

  @override
  double sample(math.Random rng) => lower + (upper - lower) * rng.nextDouble();

  @override
  double get typical => 0.5 * (lower + upper);

  @override
  String toString() => 'Uniform($lower, $upper)';

  @override
  Map<String, Object?> toJson() => {
    'type': 'Uniform',
    'lower': jsonDouble(lower),
    'upper': jsonDouble(upper),
  };
}

/// Half-normal prior on `[0, ∞)` with scale [sd].
final class HalfNormal extends Prior {
  const HalfNormal(this.sd) : assert(sd > 0);

  final double sd;

  @override
  double get lower => 0.0;

  @override
  double logDensity(double x) {
    if (!(x >= 0)) return double.negativeInfinity;
    final z = x / sd;
    return -0.5 * z * z - math.log(sd) - _logSqrt2Pi + math.ln2;
  }

  @override
  double dLogDensity(double x) => -x / (sd * sd);

  @override
  double d2LogDensity(double x) => -1 / (sd * sd);

  @override
  double sample(math.Random rng) => (sd * standardNormal(rng)).abs();

  @override
  double get typical => sd * 0.6744897501960817; // the median

  @override
  String toString() => 'HalfNormal($sd)';

  @override
  Map<String, Object?> toJson() => {'type': 'HalfNormal', 'sd': jsonDouble(sd)};
}

/// Log-normal prior: `log x ~ N(mu, sigma²)`.
final class LogNormal extends Prior {
  const LogNormal(this.mu, this.sigma) : assert(sigma > 0);

  final double mu;
  final double sigma;

  @override
  double get lower => 0.0;

  @override
  double logDensity(double x) {
    if (!(x > 0)) return double.negativeInfinity;
    final lx = math.log(x);
    final z = (lx - mu) / sigma;
    return -0.5 * z * z - math.log(sigma) - _logSqrt2Pi - lx;
  }

  @override
  double dLogDensity(double x) =>
      -((math.log(x) - mu) / (sigma * sigma) + 1) / x;

  @override
  double d2LogDensity(double x) {
    final s2 = sigma * sigma;
    return ((math.log(x) - mu) / s2 + 1 - 1 / s2) / (x * x);
  }

  @override
  double sample(math.Random rng) => math.exp(mu + sigma * standardNormal(rng));

  @override
  double get typical => math.exp(mu);

  @override
  String toString() => 'LogNormal($mu, $sigma)';

  @override
  Map<String, Object?> toJson() => {
    'type': 'LogNormal',
    'mu': jsonDouble(mu),
    'sigma': jsonDouble(sigma),
  };
}

/// Gamma prior with [shape] `α` and [rate] `β` (mean `α/β`).
final class GammaPrior extends Prior {
  const GammaPrior(this.shape, this.rate) : assert(shape > 0 && rate > 0);

  final double shape;
  final double rate;

  @override
  double get lower => 0.0;

  @override
  double logDensity(double x) {
    if (!(x > 0)) return double.negativeInfinity;
    return shape * math.log(rate) -
        logGamma(shape) +
        (shape - 1) * math.log(x) -
        rate * x;
  }

  @override
  double dLogDensity(double x) => (shape - 1) / x - rate;

  @override
  double d2LogDensity(double x) => -(shape - 1) / (x * x);

  /// Marsaglia–Tsang.
  @override
  double sample(math.Random rng) {
    if (shape < 1) {
      final u = rng.nextDouble();
      return GammaPrior(shape + 1, rate).sample(rng) *
          math.pow(u, 1 / shape).toDouble();
    }
    final d = shape - 1 / 3, c = 1 / math.sqrt(9 * d);
    while (true) {
      double x, v;
      do {
        x = standardNormal(rng);
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      final u = rng.nextDouble();
      if (math.log(u) < 0.5 * x * x + d - d * v + d * math.log(v)) {
        return d * v / rate;
      }
    }
  }

  @override
  double get typical => shape / rate;

  @override
  String toString() => 'Gamma($shape, $rate)';

  @override
  Map<String, Object?> toJson() => {
    'type': 'Gamma',
    'shape': jsonDouble(shape),
    'rate': jsonDouble(rate),
  };
}

/// An improper flat prior on `[lower, upper]` (density 1, log density 0):
/// with it, MAP estimation is maximum likelihood.
final class Flat extends Prior {
  const Flat({
    this.lower = double.negativeInfinity,
    this.upper = double.infinity,
    double? typical,
  }) : _typical = typical;

  @override
  final double lower;
  @override
  final double upper;
  final double? _typical;

  @override
  double logDensity(double x) =>
      x >= lower && x <= upper ? 0.0 : double.negativeInfinity;

  @override
  double dLogDensity(double x) => 0.0;

  @override
  double d2LogDensity(double x) => 0.0;

  @override
  double sample(math.Random rng) {
    if (lower.isFinite && upper.isFinite) {
      return lower + (upper - lower) * rng.nextDouble();
    }
    throw UnsupportedError('Cannot sample an unbounded flat prior');
  }

  @override
  double get typical {
    if (_typical case final t?) return t;
    if (lower.isFinite && upper.isFinite) return 0.5 * (lower + upper);
    if (lower.isFinite) return lower + 1.0;
    if (upper.isFinite) return upper - 1.0;
    return 0.0;
  }

  @override
  String toString() => 'Flat([$lower, $upper])';

  @override
  Map<String, Object?> toJson() => {
    'type': 'Flat',
    'lower': jsonDouble(lower),
    'upper': jsonDouble(upper),
    if (_typical case final t?) 'typical': jsonDouble(t),
  };
}
