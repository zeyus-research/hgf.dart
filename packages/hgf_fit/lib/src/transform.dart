import 'dart:math' as math;

import 'codec.dart' show jsonDouble;
import 'special.dart';

/// A bijection between a parameter's constrained scale `x` and an
/// unconstrained scale `y`, on which the optimisers and samplers move.
abstract class Transform {
  const Transform();

  /// The transform for a parameter supported on `[lower, upper]`:
  /// identity, `log(x − lower)`, `log(upper − x)` or a scaled logit.
  factory Transform.forSupport(double lower, double upper) {
    if (lower.isFinite && upper.isFinite) {
      return IntervalTransform(lower, upper);
    }
    if (lower.isFinite) return LowerBoundTransform(lower);
    if (upper.isFinite) return UpperBoundTransform(upper);
    return const IdentityTransform();
  }

  /// `y` for a constrained value `x`.
  double toUnconstrained(double x);

  /// `x` for an unconstrained value `y`.
  double toConstrained(double y);

  /// `log |dx/dy|` at `y`: the Jacobian term a density on `y` needs.
  double logAbsJacobian(double y);

  /// `dx/dy` at `y` (for exact gradients). The default takes central
  /// differences of [toConstrained]; the built-in transforms are exact.
  double dToConstrained(double y) {
    final h = 1e-6 * (y.abs() > 1 ? y.abs() : 1.0);
    return (toConstrained(y + h) - toConstrained(y - h)) / (2 * h);
  }

  /// `d log |dx/dy| / dy` at `y`. The default takes central differences of
  /// [logAbsJacobian]; the built-in transforms are exact.
  double dLogAbsJacobian(double y) {
    final h = 1e-6 * (y.abs() > 1 ? y.abs() : 1.0);
    return (logAbsJacobian(y + h) - logAbsJacobian(y - h)) / (2 * h);
  }

  /// `d²x/dy²` at `y` (for exact Hessians). The default takes central
  /// differences of [dToConstrained]; the built-in transforms are exact.
  double d2ToConstrained(double y) {
    final h = 1e-5 * (y.abs() > 1 ? y.abs() : 1.0);
    return (dToConstrained(y + h) - dToConstrained(y - h)) / (2 * h);
  }

  /// `d² log |dx/dy| / dy²` at `y`. The default takes central differences
  /// of [dLogAbsJacobian]; the built-in transforms are exact.
  double d2LogAbsJacobian(double y) {
    final h = 1e-5 * (y.abs() > 1 ? y.abs() : 1.0);
    return (dLogAbsJacobian(y + h) - dLogAbsJacobian(y - h)) / (2 * h);
  }

  /// This transform as JSON, with a `"type"` key, for `FitCodec` (e.g. to
  /// send a model to a Web Worker). The built-in ones are covered; a custom
  /// one overrides this and gives the `FitCodec` a decoder for its type.
  Map<String, Object?> toJson() => throw UnsupportedError(
    '$runtimeType can\'t be serialised: override toJson() and give '
    'FitCodec a decoder for it',
  );
}

/// `y = x`.
final class IdentityTransform extends Transform {
  const IdentityTransform();

  @override
  double toUnconstrained(double x) => x;

  @override
  double toConstrained(double y) => y;

  @override
  double logAbsJacobian(double y) => 0.0;

  @override
  double dToConstrained(double y) => 1.0;

  @override
  double dLogAbsJacobian(double y) => 0.0;

  @override
  double d2ToConstrained(double y) => 0.0;

  @override
  double d2LogAbsJacobian(double y) => 0.0;

  @override
  Map<String, Object?> toJson() => {'type': 'Identity'};
}

/// `y = log(x − lower)`.
final class LowerBoundTransform extends Transform {
  const LowerBoundTransform(this.lower);

  final double lower;

  @override
  Map<String, Object?> toJson() => {
    'type': 'LowerBound',
    'lower': jsonDouble(lower),
  };

  @override
  double toUnconstrained(double x) => math.log(x - lower);

  @override
  double toConstrained(double y) => lower + math.exp(y);

  @override
  double logAbsJacobian(double y) => y;

  @override
  double dToConstrained(double y) => math.exp(y);

  @override
  double dLogAbsJacobian(double y) => 1.0;

  @override
  double d2ToConstrained(double y) => math.exp(y);

  @override
  double d2LogAbsJacobian(double y) => 0.0;
}

/// `y = log(upper − x)`.
final class UpperBoundTransform extends Transform {
  const UpperBoundTransform(this.upper);

  final double upper;

  @override
  Map<String, Object?> toJson() => {
    'type': 'UpperBound',
    'upper': jsonDouble(upper),
  };

  @override
  double toUnconstrained(double x) => math.log(upper - x);

  @override
  double toConstrained(double y) => upper - math.exp(y);

  @override
  double logAbsJacobian(double y) => y;

  @override
  double dToConstrained(double y) => -math.exp(y);

  @override
  double dLogAbsJacobian(double y) => 1.0;

  @override
  double d2ToConstrained(double y) => -math.exp(y);

  @override
  double d2LogAbsJacobian(double y) => 0.0;
}

/// `y = logit((x − lower) / (upper − lower))`.
final class IntervalTransform extends Transform {
  const IntervalTransform(this.lower, this.upper) : assert(lower < upper);

  final double lower;
  final double upper;

  @override
  Map<String, Object?> toJson() => {
    'type': 'Interval',
    'lower': jsonDouble(lower),
    'upper': jsonDouble(upper),
  };

  @override
  double toUnconstrained(double x) {
    final p = (x - lower) / (upper - lower);
    return math.log(p) - math.log(1 - p);
  }

  @override
  double toConstrained(double y) {
    final p = y >= 0 ? 1 / (1 + math.exp(-y)) : math.exp(y) / (1 + math.exp(y));
    return lower + (upper - lower) * p;
  }

  @override
  double logAbsJacobian(double y) =>
      math.log(upper - lower) - softplus(-y) - softplus(y);

  /// `(upper − lower) σ(y) σ(−y)`.
  @override
  double dToConstrained(double y) {
    final p = _sigmoid(y);
    return (upper - lower) * p * (1 - p);
  }

  /// `σ(−y) − σ(y)`.
  @override
  double dLogAbsJacobian(double y) => 1 - 2 * _sigmoid(y);

  /// `(upper − lower) σ(y) σ(−y) (1 − 2σ(y))`.
  @override
  double d2ToConstrained(double y) {
    final p = _sigmoid(y);
    return (upper - lower) * p * (1 - p) * (1 - 2 * p);
  }

  /// `−2 σ(y) σ(−y)`.
  @override
  double d2LogAbsJacobian(double y) {
    final p = _sigmoid(y);
    return -2 * p * (1 - p);
  }
}

double _sigmoid(double y) =>
    y >= 0 ? 1 / (1 + math.exp(-y)) : math.exp(y) / (1 + math.exp(y));
