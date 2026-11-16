import 'package:hgf/hgf.dart';

import 'python.dart' show pyFloat;

/// [e] as a JAX (`jnp`) expression, with [variable] naming each variable.
/// Comparisons become `jnp.where(a > b, 1.0, 0.0)`, so that the result is a
/// number, as in hgf.dart.
String pyExpr(Expr e, String Function(String name) variable) =>
    _Py(variable).write(e, 0);

final class _Py {
  _Py(this.variable);

  final String Function(String) variable;

  // 0: any, 1: additive, 2: multiplicative, 3: unary, 4: power, 5: atom.
  String write(Expr e, int min) {
    final (text, p) = _expr(e);
    return p < min ? '($text)' : text;
  }

  (String, int) _expr(Expr e) {
    switch (e) {
      case ExprNum(:final value):
        final s = pyFloat(value);
        return (s, value < 0 ? 3 : 5);
      case ExprStr(:final value):
        throw UnsupportedError('A string ("$value") is only for data columns');
      case ExprVar(:final name):
        return (variable(name), 5);
      case ExprNeg(:final a):
        return ('-${write(a, 4)}', 3);
      case ExprBinary(:final op, :final a, :final b) when e.isComparison:
        return ('jnp.where(${_cond(op, a, b)}, 1.0, 0.0)', 5);
      case ExprBinary(op: '^', :final a, :final b):
        return ('${write(a, 5)} ** ${write(b, 3)}', 4);
      case ExprBinary(:final op, :final a, :final b):
        final p = op == '+' || op == '-' ? 1 : 2;
        return ('${write(a, p)} $op ${write(b, p + 1)}', p);
      case ExprCall(:final fn, :final args):
        final xs = [for (final a in args) write(a, 0)];
        return (
          switch (fn.name) {
            'exp' ||
            'log' ||
            'sqrt' ||
            'abs' ||
            'sign' ||
            'tanh' => 'jnp.${fn.name}(${xs[0]})',
            'sigmoid' => 'jax.nn.sigmoid(${xs[0]})',
            'min' => 'jnp.minimum(${xs[0]}, ${xs[1]})',
            'max' => 'jnp.maximum(${xs[0]}, ${xs[1]})',
            'clip' => 'jnp.clip(${xs[0]}, ${xs[1]}, ${xs[2]})',
            'where' => switch (args[0]) {
              final ExprBinary c when c.isComparison =>
                'jnp.where(${_cond(c.op, c.a, c.b)}, ${xs[1]}, ${xs[2]})',
              _ => 'jnp.where(${write(args[0], 1)} != 0, ${xs[1]}, ${xs[2]})',
            },
            _ => throw UnsupportedError(
              '${fn.name} is for data transforms only',
            ),
          },
          5,
        );
    }
  }

  String _cond(String op, Expr a, Expr b) =>
      '${write(a, 1)} $op ${write(b, 1)}';
}

/// [s] as a Dart string literal.
String dartString(String s) {
  if (!s.contains("'") && !s.contains(r'\') && !s.contains(r'$')) {
    return "'$s'";
  }
  final escaped = s
      .replaceAll(r'\', r'\\')
      .replaceAll("'", r"\'")
      .replaceAll(r'$', r'\$');
  return "'$escaped'";
}
