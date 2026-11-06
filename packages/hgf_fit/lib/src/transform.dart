import 'dart:math' as math;

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
}

/// `y = log(x − lower)`.
final class LowerBoundTransform extends Transform {
  const LowerBoundTransform(this.lower);

  final double lower;

  @override
  double toUnconstrained(double x) => math.log(x - lower);

  @override
  double toConstrained(double y) => lower + math.exp(y);

  @override
  double logAbsJacobian(double y) => y;
}

/// `y = log(upper − x)`.
final class UpperBoundTransform extends Transform {
  const UpperBoundTransform(this.upper);

  final double upper;

  @override
  double toUnconstrained(double x) => math.log(upper - x);

  @override
  double toConstrained(double y) => upper - math.exp(y);

  @override
  double logAbsJacobian(double y) => y;
}

/// `y = logit((x − lower) / (upper − lower))`.
final class IntervalTransform extends Transform {
  const IntervalTransform(this.lower, this.upper) : assert(lower < upper);

  final double lower;
  final double upper;

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
}
