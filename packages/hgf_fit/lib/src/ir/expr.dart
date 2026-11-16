import 'package:hgf/hgf.dart'
    show Expr, ExprBinary, ExprCall, ExprNeg, ExprNum, ExprStr, ExprVar;

import 'ir.dart';

/// [e] lowered to the IR, with [variable] giving each variable's [Ex]: the
/// GPU kernels' WGSL, their tangents, and the CPU's reverse mode all follow.
///
/// Integer powers up to 8 become products (`x^-2` is `1 / (x x)`), `x^0.5`
/// a square root, any other power `exp(b log a)` (so a negative base gives
/// NaN, as in the CPU's `pow` for a non-integer exponent). A comparison is
/// `select(0, 1, cond)`, `where(c, a, b)` is `select(b, a, c != 0)` (or the
/// comparison itself).
Ex lowerExpr(Expr e, Ex Function(String name) variable) {
  Ex low(Expr x) => lowerExpr(x, variable);
  switch (e) {
    case ExprNum(:final value):
      return lit(value);
    case ExprStr(:final value):
      throw UnsupportedError('A string ("$value") is only for data columns');
    case ExprVar(:final name):
      return variable(name);
    case ExprNeg(:final a):
      return neg(low(a));
    case ExprBinary(:final op, :final a, :final b) when e.isComparison:
      return sel(zero, one, _cond(op, low(a), low(b)));
    case ExprBinary(:final op, :final a, :final b):
      switch (op) {
        case '+':
          return add(low(a), low(b));
        case '-':
          return sub(low(a), low(b));
        case '*':
          return mul(low(a), low(b));
        case '/':
          return div(low(a), low(b));
        case '^':
          return _pow(low(a), b, low);
      }
      throw StateError(op);
    case ExprCall(:final fn, :final args):
      final xs = [for (final a in args) low(a)];
      return switch (fn.name) {
        'exp' => exp(xs[0]),
        'log' => log(xs[0]),
        'sqrt' => call('sqrt', [xs[0]]),
        'abs' => call('abs', [xs[0]]),
        'sign' => call('sign', [xs[0]]),
        'tanh' => tanh(xs[0]),
        'sigmoid' => sigm(xs[0]),
        'min' => min(xs[0], xs[1]),
        'max' => max(xs[0], xs[1]),
        'clip' => clamp(xs[0], xs[1], xs[2]),
        'where' => switch (args[0]) {
          final ExprBinary c when c.isComparison => sel(
            xs[2],
            xs[1],
            _cond(c.op, low(c.a), low(c.b)),
          ),
          _ => sel(xs[2], xs[1], ne(xs[0], zero)),
        },
        _ => throw UnsupportedError('${fn.name} is for data transforms only'),
      };
  }
}

Cond _cond(String op, Ex a, Ex b) => CondCmp(op, a, b);

Ex _pow(Ex base, Expr exponent, Ex Function(Expr) low) {
  if (exponent case ExprNum(value: final n)) {
    if (n == n.roundToDouble() && n.abs() <= 8) {
      final k = n.abs().toInt();
      if (k == 0) return one;
      var out = base;
      for (var i = 1; i < k; i++) {
        out = mul(out, base);
      }
      return n < 0 ? div(one, out) : out;
    }
    if (n == 0.5) return call('sqrt', [base]);
  }
  return exp(mul(low(exponent), log(base)));
}
