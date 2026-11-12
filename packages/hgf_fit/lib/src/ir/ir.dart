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
///
/// [value] is the number it stands for (`null` for a non-numeric atom, such
/// as a data read; the CPU evaluator then parses [text] or resolves it as a
/// source token). [wide] is the value an `f64` evaluation uses where the
/// `f32` constant is a precision-specific stand-in (a floor at the smallest
/// normal `f32`, `ln(f32::MAX)`, a constant rounded to `f32`).
final class Lit extends Ex {
  const Lit(this.text, [this.value, this.wide]);

  final String text;
  final double? value;
  final double? wide;

  /// The value for an `f64` evaluation (`null` for a non-numeric atom).
  double? get f64 => wide ?? value ?? _parse(text);

  static double? _parse(String t) {
    if (!t.endsWith('f')) return null;
    return double.tryParse(t.substring(0, t.length - 1));
  }

  @override
  String emit(Emit c) => text;

  @override
  bool get isZero => text == '0.0f';

  @override
  bool get isOne => text == '1.0f';
}

const zero = Lit('0.0f', 0.0);
const one = Lit('1.0f', 1.0);

/// A finite `f64` as an `f32` WGSL literal.
Lit lit(double x) {
  if (!x.isFinite || x.abs() > 3.4e38) {
    throw ArgumentError.value(x, 'x', 'Not representable as an f32 literal');
  }
  var s = x.toString();
  if (!s.contains('.') && !s.contains('e')) s = '$s.0';
  return Lit('${s}f', x);
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

final class Neg extends Ex {
  const Neg(this.a);

  final Ex a;

  @override
  String emit(Emit c) => '(-${a.emit(c)})';
}

final class Bin extends Ex {
  const Bin(this.op, this.a, this.b);

  final String op;
  final Ex a, b;

  @override
  String emit(Emit c) => '(${a.emit(c)} $op ${b.emit(c)})';
}

Ex neg(Ex a) => a is Neg ? a.a : Neg(a);

Ex add(Ex a, Ex b) {
  if (a.isZero && a is Lit) return b;
  if (b.isZero && b is Lit) return a;
  return Bin('+', a, b);
}

Ex sub(Ex a, Ex b) {
  if (b.isZero && b is Lit) return a;
  if (a.isZero && a is Lit) return neg(b);
  return Bin('-', a, b);
}

Ex mul(Ex a, Ex b) {
  if (a.isOne) return b;
  if (b.isOne) return a;
  return Bin('*', a, b);
}

Ex div(Ex a, Ex b) {
  if (b.isOne) return a;
  return Bin('/', a, b);
}

/// A boolean WGSL expression (no tangent), as a tree: a named atom (a
/// boolean `let`, `true`/`false`, a source read), a comparison of two
/// [Ex]s, `isFinite`, or `&&`, `||`, `!` of conditions.
sealed class Cond {
  /// An atom: a boolean variable, `true`/`false`, or a source expression.
  const factory Cond(String name) = CondAtom;

  const Cond._();

  /// The WGSL rendering.
  String get text;

  Cond operator &(Cond o) => CondLogic('&&', this, o);
  Cond operator |(Cond o) => CondLogic('||', this, o);
  Cond operator ~() => CondNot(this);

  @override
  String toString() => text;
}

final class CondAtom extends Cond {
  const CondAtom(this.name) : super._();

  final String name;

  @override
  String get text => name;
}

/// `a op b` for `op` in `>`, `>=`, `<`, `<=`, `==`, `!=`.
final class CondCmp extends Cond {
  const CondCmp(this.op, this.a, this.b) : super._();

  final String op;
  final Ex a, b;

  @override
  String get text => '(${a.w} $op ${b.w})';
}

/// `a && b` or `a || b`.
final class CondLogic extends Cond {
  const CondLogic(this.op, this.a, this.b) : super._();

  final String op;
  final Cond a, b;

  @override
  String get text => '(${a.text} $op ${b.text})';
}

final class CondNot extends Cond {
  const CondNot(this.a) : super._();

  final Cond a;

  @override
  String get text => '(!${a.text})';
}

final class CondFinite extends Cond {
  const CondFinite(this.a) : super._();

  final Ex a;

  @override
  String get text => 'isFinite(${a.w})';
}

const yes = Cond('true');
const no = Cond('false');

Cond gt(Object a, Object b) => CondCmp('>', ex(a), ex(b));
Cond ge(Object a, Object b) => CondCmp('>=', ex(a), ex(b));
Cond lt(Object a, Object b) => CondCmp('<', ex(a), ex(b));
Cond le(Object a, Object b) => CondCmp('<=', ex(a), ex(b));
Cond eq(Object a, Object b) => CondCmp('==', ex(a), ex(b));
Cond ne(Object a, Object b) => CondCmp('!=', ex(a), ex(b));
Cond isFinite(Ex a) => CondFinite(a);

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
      case Neg(:final a):
        final da = d(a);
        return da == null ? null : neg(da);
      case Bin(:final op, :final a, :final b):
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

/// Reverse mode: emits the adjoint statements of a straight-line program
/// (one SSA trial, [TrialCompiler.compileSsa]).
///
/// Every variable `x` that [hasAdjoint] accepts has an accumulator `x_b`
/// (declared by the caller). [statement] back-propagates `x_b` through the
/// tree of `let x = e` into the accumulators of the variables `e` reads,
/// using the same partial derivatives as [Tangents] (the [Call] rules, the
/// quotient as `(a/b)/b`, an active clamp passing nothing); calling it for
/// the statements in reverse order is the vector-Jacobian product of the
/// whole program.
///
/// Zero adjoints stay zero. A branch of a `select` that is not taken, or a
/// statement whose adjoint is 0, may hold infinite partials (an overflowed
/// branch, an expansion that is off), and `0 · ∞` is NaN. So the conditions
/// on the way down (the `select`s passed, and with [zeroGuard] the
/// statement's adjoint being nonzero) are applied where the adjoint is
/// added: `x_b += select(0, g, conditions)`.
final class Adjoints {
  Adjoints(this.hasAdjoint, {this.zeroGuard = true});

  /// Whether variable `name` has an adjoint accumulator.
  final bool Function(String name) hasAdjoint;

  /// Whether a statement whose adjoint is zero contributes exactly zero.
  final bool zeroGuard;

  /// The adjoint program, in order: `let`s of shared subexpressions and
  /// guarded accumulations.
  final steps = <AdjointStep>[];

  /// [steps] as WGSL statements.
  List<String> get lines => [for (final st in steps) st.wgsl];

  var _tmp = 0;
  final _active = Expando<bool>();

  /// Whether [e] reads any variable with an adjoint.
  bool active(Ex e) {
    final cached = _active[e];
    if (cached != null) return cached;
    final out = switch (e) {
      Lit() || TZero() => false,
      Ref(:final name) => hasAdjoint(name),
      TRef() => throw StateError('Adjoint of a tangent'),
      Neg(:final a) => active(a),
      Bin(:final a, :final b) => active(a) || active(b),
      Sel(:final f, :final t) => active(f) || active(t),
      Call(:final args) => args.any(active),
    };
    _active[e] = out;
    return out;
  }

  /// [e] as an atom: itself if it is one, else a fresh `let`.
  Ex _bind(Ex e, String stem) {
    if (e is Ref || e is Lit) return e;
    final name = '$stem${_tmp++}';
    steps.add(AdjointLet(name, e));
    return Ref(name);
  }

  /// The adjoint statements of `let name = e`, given that `name_b` is
  /// complete (every later reader has been processed).
  void statement(String name, Ex e) => _back(
    e,
    Ref('${name}_b'),
    self: Ref(name),
    guard: zeroGuard && _needsGuard(e)
        ? CondCmp('!=', Ref('${name}_b'), zero)
        : null,
  );

  /// Whether [e] has a partial that can be infinite where its value is
  /// finite or unused (a quotient, a logarithm, a call): only then can a
  /// zero adjoint turn into NaN.
  bool _needsGuard(Ex e) => switch (e) {
    Lit() || TZero() || Ref() || TRef() => false,
    Neg(:final a) => _needsGuard(a),
    Bin(:final op, :final a, :final b) =>
      op == '/' || _needsGuard(a) || _needsGuard(b),
    Sel(:final f, :final t) => _needsGuard(f) || _needsGuard(t),
    Call() => true,
  };

  static Cond? _and(Cond? a, Cond b) => a == null ? b : a & b;

  void _back(Ex e, Ex g, {Ex? self, Cond? guard}) {
    if (!active(e)) return;
    switch (e) {
      case Lit() || TZero():
        return;
      case TRef():
        throw StateError('Adjoint of a tangent');
      case Ref(:final name):
        steps.add(AdjointAdd(name, g, guard));
      case Neg(:final a):
        _back(a, neg(g), guard: guard);
      case Bin(:final op, :final a, :final b):
        final aa = active(a), ab = active(b);
        // The adjoint is used twice below: bind it.
        final gg = aa && ab ? _bind(g, 'adjG') : g;
        switch (op) {
          case '+':
            _back(a, gg, guard: guard);
            _back(b, gg, guard: guard);
          case '-':
            _back(a, gg, guard: guard);
            _back(b, neg(gg), guard: guard);
          case '*':
            if (aa) {
              _back(a, mul(gg, ab ? _bind(b, 'adjP') : b), guard: guard);
            }
            if (ab) {
              _back(b, mul(gg, aa ? _bind(a, 'adjP') : a), guard: guard);
            }
          case '/':
            final bb = _bind(b, 'adjP');
            if (aa) _back(a, div(gg, bb), guard: guard);
            // ∂(a/b)/∂b = −(a/b)/b, never through 1/b² (as in [Tangents]).
            if (ab) _back(b, neg(div(mul(div(a, bb), gg), bb)), guard: guard);
          default:
            throw StateError(op);
        }
      case Sel(:final f, :final t, :final c):
        final both = active(f) && active(t);
        final gg = both ? _bind(g, 'adjG') : g;
        _back(t, gg, guard: _and(guard, c));
        _back(f, gg, guard: _and(guard, ~c));
      case Call(:final fn, :final args):
        final rule = _rules[fn];
        if (rule == null) throw StateError('No derivative rule for $fn');
        final n = args.where(active).length;
        final gg = n > 1 ? _bind(g, 'adjG') : g;
        // The partials re-read the arguments' primal subtrees; the shader
        // compiler merges the repeats.
        final partials = rule(args, self ?? e);
        for (var i = 0; i < args.length; i++) {
          final p = partials[i];
          if (p == null || !active(args[i])) continue;
          _back(args[i], mul(gg, p), guard: guard);
        }
    }
  }
}

/// One statement of an [Adjoints] program.
sealed class AdjointStep {
  const AdjointStep();

  /// The WGSL statement.
  String get wgsl;
}

/// `let name = value;` (a subexpression the adjoints read twice).
final class AdjointLet extends AdjointStep {
  const AdjointLet(this.name, this.value);

  final String name;
  final Ex value;

  @override
  String get wgsl => 'let $name = ${value.w};';
}

/// `target_b += g`, or `+= select(0, g, guard)`.
final class AdjointAdd extends AdjointStep {
  const AdjointAdd(this.target, this.g, this.guard);

  /// The variable whose adjoint accumulates (`${target}_b`).
  final String target;
  final Ex g;
  final Cond? guard;

  @override
  String get wgsl => guard == null
      ? '${target}_b += ${g.w};'
      : '${target}_b += select(0.0f, ${g.w}, ${guard!.text});';
}
