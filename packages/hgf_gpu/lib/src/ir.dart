/// A small expression IR for the WGSL generator, with forward-mode
/// tangents.
///
/// Scalar `f32` expressions are built as trees of [Ex] (Dart operators
/// build arithmetic). [Tangents.d] differentiates a tree into a tangent
/// expression: a linear combination of the active variables' tangents
/// ([TRef]) with scalar coefficients. A tangent is a vector of up to four
/// directions (one WGSL `vecN<f32>` per chunk of four directions), so the
/// same tangent expression is rendered once per chunk.
library;

/// How to render: the tangent chunk, and its WGSL type.
final class Emit {
  const Emit(this.chunk, this.type);

  /// Rendering a primal expression (no tangent references).
  static const primal = Emit(-1, 'f32');

  final int chunk;

  /// `f32` or `vecN<f32>`.
  final String type;
}

/// A scalar (primal) or tangent (vector) WGSL expression.
sealed class Ex {
  const Ex();

  String emit(Emit c);

  /// Primal rendering.
  String get w => emit(Emit.primal);

  Ex operator +(Object other) => add(this, ex(other));
  Ex operator -(Object other) => sub(this, ex(other));
  Ex operator *(Object other) => mul(this, ex(other));
  Ex operator /(Object other) => div(this, ex(other));
  Ex operator -() => neg(this);

  bool get isZero => false;
  bool get isOne => false;

  @override
  String toString() => w;
}

/// [o] as an expression: an [Ex], or a number as an `f32` literal.
Ex ex(Object o) => switch (o) {
  final Ex e => e,
  final num n => lit(n.toDouble()),
  _ => throw ArgumentError.value(o, 'o', 'Not an expression'),
};

/// A WGSL literal (or any atom: a builtin constant expression).
final class Lit extends Ex {
  const Lit(this.text);

  final String text;

  @override
  String emit(Emit c) => text;

  @override
  bool get isZero => text == '0.0f';

  @override
  bool get isOne => text == '1.0f';
}

const zero = Lit('0.0f');
const one = Lit('1.0f');

/// A finite `f64` as an `f32` WGSL literal.
Lit lit(double x) {
  if (!x.isFinite || x.abs() > 3.4e38) {
    throw ArgumentError.value(x, 'x', 'Not representable as an f32 literal');
  }
  var s = x.toString();
  if (!s.contains('.') && !s.contains('e')) s = '$s.0';
  return Lit('${s}f');
}

/// A scalar variable.
final class Ref extends Ex {
  const Ref(this.name);

  final String name;

  @override
  String emit(Emit c) => name;
}

/// The tangent of variable [name] (in the chunk being rendered).
final class TRef extends Ex {
  const TRef(this.name);

  final String name;

  @override
  String emit(Emit c) {
    assert(c.chunk >= 0, 'Tangent rendered as primal');
    return '${name}_t${c.chunk}';
  }
}

/// A zero tangent.
final class TZero extends Ex {
  const TZero();

  @override
  String emit(Emit c) => '${c.type}()';

  @override
  bool get isZero => true;
}

final class _Neg extends Ex {
  const _Neg(this.a);

  final Ex a;

  @override
  String emit(Emit c) => '(-${a.emit(c)})';
}

final class _Bin extends Ex {
  const _Bin(this.op, this.a, this.b);

  final String op;
  final Ex a, b;

  @override
  String emit(Emit c) => '(${a.emit(c)} $op ${b.emit(c)})';
}

Ex neg(Ex a) => a is _Neg ? a.a : _Neg(a);

Ex add(Ex a, Ex b) {
  if (a.isZero && a is Lit) return b;
  if (b.isZero && b is Lit) return a;
  return _Bin('+', a, b);
}

Ex sub(Ex a, Ex b) {
  if (b.isZero && b is Lit) return a;
  if (a.isZero && a is Lit) return neg(b);
  return _Bin('-', a, b);
}

Ex mul(Ex a, Ex b) {
  if (a.isOne) return b;
  if (b.isOne) return a;
  return _Bin('*', a, b);
}

Ex div(Ex a, Ex b) {
  if (b.isOne) return a;
  return _Bin('/', a, b);
}

/// A boolean WGSL expression (no tangent).
final class Cond {
  const Cond(this.text);

  final String text;

  Cond operator &(Cond o) => Cond('($text && ${o.text})');
  Cond operator |(Cond o) => Cond('($text || ${o.text})');
  Cond operator ~() => Cond('(!$text)');

  @override
  String toString() => text;
}

const yes = Cond('true');
const no = Cond('false');

Cond gt(Object a, Object b) => Cond('(${ex(a).w} > ${ex(b).w})');
Cond ge(Object a, Object b) => Cond('(${ex(a).w} >= ${ex(b).w})');
Cond lt(Object a, Object b) => Cond('(${ex(a).w} < ${ex(b).w})');
Cond le(Object a, Object b) => Cond('(${ex(a).w} <= ${ex(b).w})');
Cond eq(Object a, Object b) => Cond('(${ex(a).w} == ${ex(b).w})');
Cond ne(Object a, Object b) => Cond('(${ex(a).w} != ${ex(b).w})');
Cond isFinite(Ex a) => Cond('isFinite(${a.w})');

/// `select(f, t, c)`: [t] where [c] holds, else [f].
final class Sel extends Ex {
  const Sel(this.f, this.t, this.c);

  final Ex f, t;
  final Cond c;

  @override
  String emit(Emit e) => 'select(${f.emit(e)}, ${t.emit(e)}, ${c.text})';
}

Ex sel(Object f, Object t, Cond c) => Sel(ex(f), ex(t), c);

/// A call of a WGSL builtin or prelude function with a derivative rule.
final class Call extends Ex {
  const Call(this.fn, this.args);

  final String fn;
  final List<Ex> args;

  @override
  String emit(Emit c) => '$fn(${args.map((a) => a.emit(c)).join(', ')})';
}

/// Partial derivatives of a [Call] of a function with respect to each of
/// its arguments (null: none), given the arguments and the call's value.
typedef Partials = List<Ex?> Function(List<Ex> args, Ex self);

Ex _sigm(Ex x) => Call('sigm', [x]);
Ex _dsigm(Ex x) => Call('dsigm', [x]);
Ex _d2sigm(Ex x) => Call('d2sigm', [x]);
Ex _dtanh(Ex x) => Call('dtanh', [x]);
Ex _d2tanh(Ex x) => Call('d2tanh', [x]);
Ex _dgelu(Ex x) => Call('dgelu', [x]);
Ex _d2gelu(Ex x) => Call('d2gelu', [x]);

/// Derivative rules of the functions the generator calls. A clamp that is
/// active passes no derivative; Lambert W passes `W / (z (1 + W))`.
final Map<String, Partials> _rules = {
  'exp': (a, s) => [s],
  'log': (a, s) => [one / a[0]],
  'tanh': (a, s) => [_dtanh(a[0])],
  'sigm': (a, s) => [_dsigm(a[0])],
  'dsigm': (a, s) => [_d2sigm(a[0])],
  'd2sigm': (a, s) => [Call('d3sigm', a)],
  'dtanh': (a, s) => [_d2tanh(a[0])],
  'd2tanh': (a, s) => [Call('d3tanh', a)],
  'gelu': (a, s) => [_dgelu(a[0])],
  'dgelu': (a, s) => [_d2gelu(a[0])],
  'd2gelu': (a, s) => [Call('d3gelu', a)],
  'max': (a, s) => [sel(0, 1, ge(a[0], a[1])), sel(1, 0, ge(a[0], a[1]))],
  'min': (a, s) => [sel(0, 1, le(a[0], a[1])), sel(1, 0, le(a[0], a[1]))],
  'clamp': (a, s) => [sel(0, 1, gt(a[0], a[1]) & lt(a[0], a[2])), null, null],
  'abs': (a, s) => [Call('sign', a)],
  'lambertW0': (a, s) => [s / (a[0] * (one + s))],
  'logAddExp': (a, s) => [_sigm(a[0] - a[1]), _sigm(a[1] - a[0])],
};

Ex call(String fn, List<Object> args) {
  if (!_rules.containsKey(fn)) {
    throw ArgumentError.value(fn, 'fn', 'No derivative rule');
  }
  return Call(fn, [for (final a in args) ex(a)]);
}

Ex exp(Object x) => call('exp', [x]);
Ex log(Object x) => call('log', [x]);
Ex tanh(Object x) => call('tanh', [x]);
Ex sigm(Object x) => call('sigm', [x]);
Ex max(Object a, Object b) => call('max', [a, b]);
Ex min(Object a, Object b) => call('min', [a, b]);
Ex clamp(Object x, Object lo, Object hi) => call('clamp', [x, lo, hi]);
Ex lambertW0(Object z) => call('lambertW0', [z]);
Ex logAddExp(Object a, Object b) => call('logAddExp', [a, b]);

/// Differentiates expressions given which variables carry tangents.
final class Tangents {
  Tangents(this.active);

  /// Whether variable `name` carries a tangent.
  final bool Function(String name) active;

  /// The tangent of [e] (null: zero). [self], if given, is a variable
  /// holding [e]'s value (so a [Call]'s rule needn't call it again).
  Ex? d(Ex e, {Ex? self}) {
    switch (e) {
      case Lit() || TZero():
        return null;
      case Ref(:final name):
        return active(name) ? TRef(name) : null;
      case TRef():
        throw StateError('Tangent of a tangent');
      case _Neg(:final a):
        final da = d(a);
        return da == null ? null : neg(da);
      case _Bin(:final op, :final a, :final b):
        final da = d(a), db = d(b);
        if (da == null && db == null) return null;
        switch (op) {
          case '+':
            return da == null ? db : (db == null ? da : add(da, db));
          case '-':
            return da == null ? neg(db!) : (db == null ? da : sub(da, db));
          case '*':
            final l = da == null ? null : mul(b, da);
            final r = db == null ? null : mul(a, db);
            return l == null ? r : (r == null ? l : add(l, r));
          case '/':
            if (db == null) return div(da!, b);
            // (da − (a/b) db) / b: never through 1/b², which underflows.
            final q = mul(div(a, b), db);
            return div(da == null ? neg(q) : sub(da, q), b);
        }
        throw StateError(op);
      case Sel(:final f, :final t, :final c):
        final df = d(f), dt = d(t);
        if (df == null && dt == null) return null;
        return Sel(df ?? const TZero(), dt ?? const TZero(), c);
      case Call(:final fn, :final args):
        final rule = _rules[fn];
        if (rule == null) throw StateError('No derivative rule for $fn');
        final partials = rule(args, self ?? e);
        Ex? out;
        for (var i = 0; i < args.length; i++) {
          final p = partials[i];
          if (p == null) continue;
          final da = d(args[i]);
          if (da == null) continue;
          final term = mul(p, da);
          out = out == null ? term : add(out, term);
        }
        return out;
    }
  }
}
