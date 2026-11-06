import 'dart:math' as math;

import 'math.dart';

/// A value-coupling function `g` between a parent and its value children,
/// with its first three derivatives (the updates need `g`, `g'` and `g''`;
/// exact gradients through the updates (`BatchNetwork.tangents`) also need
/// `g'''`; there is no autodiff).
///
/// The function is declared on the **parent** and applies to all of its value
/// children, as in rshgf. Use one of the named [CouplingFn]s, or your own
/// [CustomCoupling].
abstract interface class Coupling {
  /// g(x)
  double f(double x);

  /// g'(x)
  double df(double x);

  /// g''(x)
  double d2f(double x);

  /// g'''(x)
  double d3f(double x);
}

/// A user-defined value-coupling function, given with its closed-form
/// derivatives:
///
/// ```dart
/// final sine = CustomCoupling(
///   'sin',
///   g: math.sin,
///   dg: math.cos,
///   d2g: (x) => -math.sin(x),
///   d3g: (x) => -math.cos(x), // optional, for exact gradients
/// );
/// hgf.addNodes(valueChildren: [0], coupling: sine);
/// ```
///
/// The derivatives are not checked; [CustomCoupling.check] compares them with
/// finite differences if you want to make sure.
final class CustomCoupling implements Coupling {
  const CustomCoupling(
    this.name, {
    required this.g,
    required this.dg,
    required this.d2g,
    this.d3g,
  });

  /// A name for diagnostics and for `NetworkSpec` (see `NetworkSpec.build`).
  final String name;

  /// g, g' and g''.
  final double Function(double x) g, dg, d2g;

  /// g''', used only for exact gradients (`BatchNetwork.tangents`) through
  /// a nonlinear value coupling. Without it, [d3f] takes central
  /// differences of [d2g] (relative error around 1e-10 for smooth `g`).
  final double Function(double x)? d3g;

  @override
  double f(double x) => g(x);

  @override
  double df(double x) => dg(x);

  @override
  double d2f(double x) => d2g(x);

  @override
  double d3f(double x) {
    final d3 = d3g;
    if (d3 != null) return d3(x);
    final h = 6e-6 * (x.abs() > 1.0 ? x.abs() : 1.0);
    final xp = x + h, xm = x - h;
    return (d2g(xp) - d2g(xm)) / (xp - xm);
  }

  /// Throws [StateError] if [dg] or [d2g] disagree with central finite
  /// differences of [g] (resp. [dg]) at any of [at], beyond [tolerance]
  /// (relative, with an absolute floor of the same size).
  void check({
    Iterable<double> at = const [-3, -1.5, -0.5, -0.1, 0.1, 0.5, 1.5, 3],
    double tolerance = 1e-5,
  }) {
    const h = 1e-5;
    for (final x in at) {
      final fd1 = (g(x + h) - g(x - h)) / (2 * h);
      final fd2 = (dg(x + h) - dg(x - h)) / (2 * h);
      void cmp(String what, double exact, double approx) {
        final err = (exact - approx).abs();
        if (!(err <= tolerance * (1 + exact.abs()))) {
          throw StateError(
            '$name: $what($x) = $exact, finite differences give $approx',
          );
        }
      }

      cmp("g'", dg(x), fd1);
      cmp("g''", d2g(x), fd2);
      if (d3g case final d3?) {
        cmp("g'''", d3(x), (d2g(x + h) - d2g(x - h)) / (2 * h));
      }
    }
  }

  @override
  String toString() => 'CustomCoupling($name)';
}

/// The named coupling functions, with closed-form derivatives. Names match
/// pyhgf/rshgf (`linear`/`identity`, `relu`, `sigmoid`, `tanh`,
/// `leaky_relu`, `gelu`).
enum CouplingFn implements Coupling {
  linear('linear'),
  relu('relu'),
  sigmoid('sigmoid'),
  tanh('tanh'),
  leakyRelu('leaky_relu'),
  gelu('gelu');

  const CouplingFn(this.pyhgfName);

  /// The name used by pyhgf / rshgf.
  final String pyhgfName;

  /// Parse a pyhgf coupling function name.
  static CouplingFn parse(String name) {
    if (name == 'identity') return CouplingFn.linear;
    for (final fn in values) {
      if (fn.pyhgfName == name) return fn;
    }
    throw ArgumentError.value(
      name,
      'name',
      'Unknown coupling function. Choose from '
          '${values.map((v) => v.pyhgfName).join(', ')}, identity',
    );
  }

  @override
  double f(double x) => switch (this) {
    linear => x,
    relu => x > 0.0 ? x : 0.0,
    sigmoid => _sigmoid(x),
    tanh => _tanh(x),
    leakyRelu => x >= 0.0 ? x : 0.01 * x,
    gelu => x * 0.5 * _erfc(-x / math.sqrt2),
  };

  @override
  double df(double x) {
    switch (this) {
      case linear:
        return 1.0;
      case relu:
        return x > 0.0 ? 1.0 : 0.0;
      case sigmoid:
        final s = _sigmoid(x);
        return s * (1.0 - s);
      case tanh:
        final t = _tanh(x);
        return 1.0 - t * t;
      case leakyRelu:
        return x >= 0.0 ? 1.0 : 0.01;
      case gelu:
        final phi = 0.5 * _erfc(-x / math.sqrt2);
        final pdf = math.exp(-(x * x) / 2.0) / math.sqrt(2.0 * math.pi);
        return phi + x * pdf;
    }
  }

  @override
  double d2f(double x) {
    switch (this) {
      case linear:
      case relu:
      case leakyRelu:
        return 0.0;
      case sigmoid:
        final s = _sigmoid(x);
        return s * (1.0 - s) * (1.0 - 2.0 * s);
      case tanh:
        final t = _tanh(x);
        return -2.0 * t * (1.0 - t * t);
      case gelu:
        final pdf = math.exp(-(x * x) / 2.0) / math.sqrt(2.0 * math.pi);
        return pdf * (2.0 - x * x);
    }
  }

  @override
  double d3f(double x) {
    switch (this) {
      case linear:
      case relu:
      case leakyRelu:
        return 0.0;
      case sigmoid:
        final s = _sigmoid(x);
        final q = s * (1.0 - s);
        final r = 1.0 - 2.0 * s;
        return q * (r * r - 2.0 * q);
      case tanh:
        final t = _tanh(x);
        final u = 1.0 - t * t;
        return u * (6.0 * t * t - 2.0);
      case gelu:
        final pdf = math.exp(-(x * x) / 2.0) / math.sqrt(2.0 * math.pi);
        return pdf * x * (x * x - 4.0);
    }
  }
}

double _sigmoid(double x) => sigmoid(x);

/// tanh (dart:math has none), accurate to a few ulps over the whole range:
/// `tanh(x) = -expm1(-2|x|) / (2 + expm1(-2|x|))`, with a cancellation-free
/// expm1.
double _tanh(double x) {
  final ax = x.abs();
  if (ax > 22.0) return x.sign;
  final em = _expm1(-2.0 * ax);
  final t = -em / (2.0 + em);
  return x < 0 ? -t : t;
}

/// exp(x) - 1 without cancellation near 0 (Kahan's trick).
double _expm1(double x) {
  final u = math.exp(x);
  if (u == 1.0) return x;
  final um1 = u - 1.0;
  if (um1 == -1.0) return -1.0;
  return um1 * x / math.log(u);
}

/// Abramowitz-Stegun 7.1.26 erfc, identical to rshgf's (`src/math.rs`).
double _erfc(double x) {
  final t = 1.0 / (1.0 + 0.3275911 * x.abs());
  final poly =
      t *
      (0.254829592 +
          t *
              (-0.284496736 +
                  t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  final approx = poly * math.exp(-(x * x));
  return x >= 0.0 ? approx : 2.0 - approx;
}
