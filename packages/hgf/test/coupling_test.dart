import 'package:hgf/hgf.dart';
import 'package:test/test.dart';

void main() {
  // Correctly rounded tanh from mpmath (200-bit), across both branches
  // (|x| < 0.625 rational, exp formula above) and the saturation cut-off.
  test('tanh coupling is within 2 ulp of mpmath', () {
    const reference = [
      (1e-300, 1e-300),
      (1e-08, 1e-08),
      (-0.0003, -0.0002999999910000003),
      (0.01, 0.00999966667999946),
      (-0.1, -0.09966799462495582),
      (0.3, 0.2913126124515909),
      (0.5, 0.46211715726000974),
      (-0.6249, -0.5545304765944484),
      (0.625, 0.5545997223493823),
      (-0.7, -0.6043677771171635),
      (1.0, 0.7615941559557649),
      (2.5, 0.9866142981514303),
      (-5.0, -0.9999092042625951),
      (10.0, 0.9999999958776927),
      (19.0, 0.9999999999999999),
      (21.9, 1.0),
      (-23.0, -1.0),
    ];
    for (final (x, t) in reference) {
      // 2 ulp of t (ulp ≈ |t| · 2^-52, or the subnormal floor).
      final tol = 2 * (t.abs() * 2.220446049250313e-16 + 5e-324);
      expect(CouplingFn.tanh.f(x), closeTo(t, tol), reason: 'tanh($x)');
    }
    expect(CouplingFn.tanh.f(0.0), 0.0);
    expect(CouplingFn.tanh.f(double.nan).isNaN, isTrue);
  });
}
