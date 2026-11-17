import 'dart:math' as math;

import 'math.dart';

/// A small expression language for what a model builder lets you type: a
/// coupling function `g(x)`, a choice option's value, a response model's
/// surprise, a data transform.
///
/// ```dart
/// final e = Expr.parse('2 * tanh(x) - 0.5 * x^2');
/// e.eval({'x': 0.3});           // 0.5375...
/// e.derivative('x').toString(); // '2 * (1 - tanh(x)^2) - x'
/// ```
///
/// Syntax: numbers (`1`, `0.5`, `1e-3`), variables (`x`, `ball.mean`, or
/// any text between backticks: `` `press left` ``), strings (`'left'`, only
/// to compare with a data column), `+ - * /`, `^` or `**` (power, right
/// associative, binding tighter than a unary minus: `-x^2` is `-(x^2)`),
/// comparisons `< <= > >= == !=` (1 when true, else 0) and the functions in
/// [Expr.functions]. A variable name may contain dots (`node.field`,
/// `cat.2.mean`).
///
/// An expression prints back in a canonical form ([toString]), with only
/// the parentheses it needs; `Expr.parse(e.toString()) == e`. Equality is
/// structural.
///
/// Derivatives ([derivative]) are symbolic and simplified, so a coupling
/// function gets `g'`, `g''` and `g'''` in closed form, and the fitting IR
/// (`package:hgf_fit/ir.dart`) lowers expressions to the GPU kernels and
/// exact gradients. The data-only functions ([ExprFn.dataOnly]: lags,
/// another participant's column, missing values) are evaluated by the
/// model document's data transforms; [eval] and [compile] reject them.
sealed class Expr {
  const Expr();

  /// Parses [source]; throws a [FormatException] with the offset of the
  /// problem.
  factory Expr.parse(String source) => _Parser(source).parse();

  /// A number.
  const factory Expr.number(double value) = ExprNum;

  /// A variable.
  const factory Expr.variable(String name) = ExprVar;

  /// The functions, by name.
  static const Map<String, ExprFn> functions = {
    'exp': ExprFn('exp', 1, 1, 'e^x'),
    'log': ExprFn('log', 1, 1, 'natural logarithm'),
    'sqrt': ExprFn('sqrt', 1, 1, 'square root'),
    'abs': ExprFn('abs', 1, 1, 'absolute value'),
    'sign': ExprFn('sign', 1, 1, '-1, 0 or 1'),
    'tanh': ExprFn('tanh', 1, 1, 'hyperbolic tangent'),
    'sigmoid': ExprFn('sigmoid', 1, 1, '1 / (1 + e^-x)'),
    'min': ExprFn('min', 2, 2, 'the smaller of two'),
    'max': ExprFn('max', 2, 2, 'the larger of two'),
    'clip': ExprFn('clip', 3, 3, 'clip(x, lo, hi): x limited to [lo, hi]'),
    'where': ExprFn(
      'where',
      3,
      3,
      'where(c, a, b): a where c is nonzero, else b',
    ),
    'lag': ExprFn(
      'lag',
      1,
      2,
      'lag(x, k): x k rows earlier (default 1), per participant',
      dataOnly: true,
    ),
    'diff': ExprFn(
      'diff',
      1,
      1,
      'diff(x): x minus its previous row, per participant',
      dataOnly: true,
    ),
    'mate': ExprFn(
      'mate',
      1,
      2,
      'mate(x, k): x of the k-th other member (default 1) of the '
          'participant\'s group, at the same time',
      dataOnly: true,
    ),
    'mates_sum': ExprFn(
      'mates_sum',
      1,
      1,
      'mates_sum(x): the sum of x over the participant\'s team-mates at '
          'the same time (those with x; empty when none has)',
      dataOnly: true,
    ),
    'mates_mean': ExprFn(
      'mates_mean',
      1,
      1,
      'mates_mean(x): the mean of x over the participant\'s team-mates at '
          'the same time (those with x; empty when none has)',
      dataOnly: true,
    ),
    'mates_count': ExprFn(
      'mates_count',
      1,
      1,
      'mates_count(x): how many of the participant\'s team-mates have x at '
          'the same time (for text, a cell that is not empty)',
      dataOnly: true,
    ),
    'missing': ExprFn(
      'missing',
      1,
      1,
      'missing(x): 1 where x is empty, else 0',
      dataOnly: true,
    ),
    'fill': ExprFn(
      'fill',
      2,
      2,
      'fill(x, v): x, with v where x is empty',
      dataOnly: true,
    ),
  };

  /// The variables this expression reads, in order of first use.
  Set<String> get variables {
    final out = <String>{};
    _visit((e) {
      if (e is ExprVar) out.add(e.name);
    });
    return out;
  }

  /// Whether this expression calls a data-only function or holds a string.
  bool get isDataOnly {
    var found = false;
    _visit((e) {
      if (e is ExprStr) found = true;
      if (e is ExprCall && e.fn.dataOnly) found = true;
    });
    return found;
  }

  void _visit(void Function(Expr) f);

  /// This expression with each variable `v` read as `rename(v)` (e.g. a
  /// column or node that was renamed). Returns this when nothing changes.
  Expr renamed(String Function(String name) rename) {
    final out = switch (this) {
      ExprVar(:final name) => ExprVar(rename(name)),
      ExprNeg(:final a) => ExprNeg(a.renamed(rename)),
      ExprBinary(:final op, :final a, :final b) => ExprBinary(
        op,
        a.renamed(rename),
        b.renamed(rename),
      ),
      ExprCall(:final fn, :final args) => ExprCall(fn, [
        for (final a in args) a.renamed(rename),
      ]),
      ExprNum() || ExprStr() => this,
    };
    return out == this ? this : out;
  }

  /// The derivative with respect to [variable], simplified. Comparisons
  /// and `sign` have derivative 0; `min`, `max`, `clip` and `where` pass the
  /// branch taken; `abs` gives 1 at 0, as JAX does.
  Expr derivative(String variable);

  /// The value with [env]'s variables (throws [ArgumentError] for a
  /// missing one, [UnsupportedError] for a data-only function).
  double eval(Map<String, double> env) =>
      compile(env.keys.toList())(env.values.toList());

  /// A function of the [variables]' values, in that order (variables of
  /// this expression not among them throw [ArgumentError]).
  double Function(List<double> values) compile(List<String> variables) {
    final index = {for (var i = 0; i < variables.length; i++) variables[i]: i};
    return _compile(index);
  }

  /// A function of the one variable [variable].
  double Function(double x) compile1(String variable) {
    final f = _compile({variable: 0});
    final env = [0.0];
    return (x) {
      env[0] = x;
      return f(env);
    };
  }

  double Function(List<double>) _compile(Map<String, int> index);

  /// 0: comparison, 1: additive, 2: multiplicative, 3: unary, 4: power,
  /// 5: atom.
  int get _precedence;

  /// The canonical source.
  @override
  String toString() {
    final b = StringBuffer();
    _write(b);
    return b.toString();
  }

  void _write(StringBuffer b);

  void _writeChild(StringBuffer b, Expr child, int minPrecedence) {
    if (child._precedence < minPrecedence) {
      b.write('(');
      child._write(b);
      b.write(')');
    } else {
      child._write(b);
    }
  }

  @override
  bool operator ==(Object other) =>
      other is Expr && other.toString() == toString();

  @override
  int get hashCode => toString().hashCode;

  // Smart constructors that simplify.

  /// `a + b`.
  static Expr add(Expr a, Expr b) {
    if (a case ExprNum(value: final x)) {
      if (b case ExprNum(value: final y)) return ExprNum(x + y);
      if (x == 0) return b;
    }
    if (b case ExprNum(value: 0)) return a;
    if (b case ExprNeg(a: final inner)) return sub(a, inner);
    if (b case ExprNum(value: final y) when y < 0) return sub(a, ExprNum(-y));
    return ExprBinary('+', a, b);
  }

  /// `a - b`.
  static Expr sub(Expr a, Expr b) {
    if (a case ExprNum(value: final x)) {
      if (b case ExprNum(value: final y)) return ExprNum(x - y);
      if (x == 0) return neg(b);
    }
    if (b case ExprNum(value: 0)) return a;
    if (b case ExprNeg(a: final inner)) return add(a, inner);
    if (a == b) return const ExprNum(0);
    return ExprBinary('-', a, b);
  }

  /// `a * b`.
  static Expr mul(Expr a, Expr b) {
    if (a case ExprNum(value: final x)) {
      if (b case ExprNum(value: final y)) return ExprNum(x * y);
      if (x == 0) return const ExprNum(0);
      if (x == 1) return b;
      if (x == -1) return neg(b);
    }
    if (b case ExprNum(value: final y)) {
      if (y == 0) return const ExprNum(0);
      if (y == 1) return a;
      if (y == -1) return neg(a);
      // Numbers first: 2 * x, not x * 2.
      return mul(b, a);
    }
    if (a case ExprNeg(a: final inner)) return neg(mul(inner, b));
    if (b case ExprNeg(a: final inner)) return neg(mul(a, inner));
    // 2 * (3 * x) → 6 * x.
    if ((a, b) case (
      ExprNum(value: final x),
      ExprBinary(op: '*', a: ExprNum(value: final y), b: final rest),
    )) {
      return mul(ExprNum(x * y), rest);
    }
    return ExprBinary('*', a, b);
  }

  /// `a / b`.
  static Expr div(Expr a, Expr b) {
    if ((a, b) case (
      ExprNum(value: final x),
      ExprNum(value: final y),
    ) when y != 0) {
      return ExprNum(x / y);
    }
    if (a case ExprNum(value: 0)) return const ExprNum(0);
    if (b case ExprNum(value: 1)) return a;
    if (b case ExprNum(value: -1)) return neg(a);
    if (a case ExprNeg(a: final inner)) return neg(div(inner, b));
    return ExprBinary('/', a, b);
  }

  /// `a ^ b`.
  static Expr pow(Expr a, Expr b) {
    if (b case ExprNum(value: final y)) {
      if (y == 0) return const ExprNum(1);
      if (y == 1) return a;
      if (a case ExprNum(value: final x)) {
        final v = math.pow(x, y).toDouble();
        if (v.isFinite) return ExprNum(v);
      }
      // (a^m)^n → a^(m n) for integer n.
      if (a case ExprBinary(
        op: '^',
        a: final base,
        b: ExprNum(value: final m),
      ) when y == y.roundToDouble()) {
        return pow(base, ExprNum(m * y));
      }
    }
    return ExprBinary('^', a, b);
  }

  /// `-a`.
  static Expr neg(Expr a) => switch (a) {
    ExprNum(:final value) => ExprNum(-value),
    ExprNeg(:final a) => a,
    ExprBinary(op: '-', a: final x, b: final y) => sub(y, x),
    ExprBinary(op: '*', a: ExprNum(value: final x), b: final y) => mul(
      ExprNum(-x),
      y,
    ),
    _ => ExprNeg(a),
  };

  /// `fn(args)`.
  static Expr call(String fn, List<Expr> args) {
    final f = functions[fn] ?? (throw ArgumentError.value(fn, 'fn'));
    if (args.length < f.minArgs || args.length > f.maxArgs) {
      throw ArgumentError('$fn takes ${f.arity} argument(s)');
    }
    if (!f.dataOnly && args.every((a) => a is ExprNum)) {
      final v = ExprCall(f, args).eval(const {});
      if (v.isFinite) return ExprNum(v);
    }
    if (fn == 'where') {
      if (args[0] case ExprNum(:final value) when !value.isNaN) {
        return value != 0 ? args[1] : args[2];
      }
      if (args[1] == args[2]) return args[1];
    }
    return ExprCall(f, args);
  }
}

/// A function of [Expr]: its [name], how many arguments it takes, a
/// one-line [help], and whether only data transforms may call it.
final class ExprFn {
  const ExprFn(
    this.name,
    this.minArgs,
    this.maxArgs,
    this.help, {
    this.dataOnly = false,
  });

  final String name;
  final int minArgs, maxArgs;
  final String help;
  final bool dataOnly;

  String get arity => minArgs == maxArgs ? '$minArgs' : '$minArgs or $maxArgs';
}

/// A number.
final class ExprNum extends Expr {
  const ExprNum(this.value);

  final double value;

  @override
  void _visit(void Function(Expr) f) => f(this);

  @override
  Expr derivative(String variable) => const ExprNum(0);

  @override
  double Function(List<double>) _compile(Map<String, int> index) {
    final v = value;
    return (_) => v;
  }

  @override
  int get _precedence => value < 0 || value.isNaN ? 3 : 5;

  @override
  void _write(StringBuffer b) => b.write(formatNumber(value));

  /// [v] as the shortest text that reads back to it (`2`, not `2.0`).
  static String formatNumber(double v) {
    if (v.isNaN) return 'nan';
    if (v.isInfinite) return v > 0 ? 'inf' : '-inf';
    if (v == v.roundToDouble() && v.abs() < 1e15) {
      return v.toInt().toString();
    }
    return v.toString();
  }
}

/// A string: compared with the text of a data column.
final class ExprStr extends Expr {
  const ExprStr(this.value);

  final String value;

  @override
  void _visit(void Function(Expr) f) => f(this);

  @override
  Expr derivative(String variable) => const ExprNum(0);

  @override
  double Function(List<double>) _compile(Map<String, int> index) =>
      throw UnsupportedError('A string ("$value") is only for data columns');

  @override
  int get _precedence => 5;

  @override
  void _write(StringBuffer b) =>
      b.write(value.contains("'") ? '"$value"' : "'$value'");
}

/// A variable.
final class ExprVar extends Expr {
  const ExprVar(this.name);

  final String name;

  static final _plain = RegExp(r'^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)*$');

  /// Whether [name] prints without backticks.
  static bool isPlain(String name) => _plain.hasMatch(name);

  @override
  void _visit(void Function(Expr) f) => f(this);

  @override
  Expr derivative(String variable) => ExprNum(name == variable ? 1 : 0);

  @override
  double Function(List<double>) _compile(Map<String, int> index) {
    final i =
        index[name] ??
        (throw ArgumentError.value(name, 'variable', 'Not bound'));
    return (env) => env[i];
  }

  @override
  int get _precedence => 5;

  @override
  void _write(StringBuffer b) => b.write(isPlain(name) ? name : '`$name`');
}

/// `-a`.
final class ExprNeg extends Expr {
  const ExprNeg(this.a);

  final Expr a;

  @override
  void _visit(void Function(Expr) f) {
    f(this);
    a._visit(f);
  }

  @override
  Expr derivative(String variable) => Expr.neg(a.derivative(variable));

  @override
  double Function(List<double>) _compile(Map<String, int> index) {
    final fa = a._compile(index);
    return (env) => -fa(env);
  }

  @override
  int get _precedence => 3;

  @override
  void _write(StringBuffer b) {
    b.write('-');
    _writeChild(b, a, 4);
  }
}

/// `a op b`: arithmetic (`+ - * / ^`) or a comparison (`< <= > >= == !=`,
/// 1 or 0).
final class ExprBinary extends Expr {
  const ExprBinary(this.op, this.a, this.b);

  final String op;
  final Expr a, b;

  static const comparisons = {'<', '<=', '>', '>=', '==', '!='};

  bool get isComparison => comparisons.contains(op);

  @override
  void _visit(void Function(Expr) f) {
    f(this);
    a._visit(f);
    b._visit(f);
  }

  @override
  Expr derivative(String variable) {
    if (isComparison) return const ExprNum(0);
    final da = a.derivative(variable), db = b.derivative(variable);
    switch (op) {
      case '+':
        return Expr.add(da, db);
      case '-':
        return Expr.sub(da, db);
      case '*':
        return Expr.add(Expr.mul(da, b), Expr.mul(a, db));
      case '/':
        // (a' − (a/b) b') / b.
        return Expr.div(Expr.sub(da, Expr.mul(Expr.div(a, b), db)), b);
      case '^':
        if (b case ExprNum(value: final n)) {
          return Expr.mul(
            Expr.mul(ExprNum(n), Expr.pow(a, ExprNum(n - 1))),
            da,
          );
        }
        // a^b (b' log a + b a'/a).
        return Expr.mul(
          this,
          Expr.add(
            Expr.mul(db, Expr.call('log', [a])),
            Expr.div(Expr.mul(b, da), a),
          ),
        );
    }
    throw StateError(op);
  }

  @override
  double Function(List<double>) _compile(Map<String, int> index) {
    final fa = a._compile(index), fb = b._compile(index);
    return switch (op) {
      '+' => (env) => fa(env) + fb(env),
      '-' => (env) => fa(env) - fb(env),
      '*' => (env) => fa(env) * fb(env),
      '/' => (env) => fa(env) / fb(env),
      '^' => switch (b) {
        ExprNum(value: 2) => (env) {
          final x = fa(env);
          return x * x;
        },
        ExprNum(value: 3) => (env) {
          final x = fa(env);
          return x * x * x;
        },
        _ => (env) => math.pow(fa(env), fb(env)).toDouble(),
      },
      '<' => (env) => fa(env) < fb(env) ? 1.0 : 0.0,
      '<=' => (env) => fa(env) <= fb(env) ? 1.0 : 0.0,
      '>' => (env) => fa(env) > fb(env) ? 1.0 : 0.0,
      '>=' => (env) => fa(env) >= fb(env) ? 1.0 : 0.0,
      '==' => (env) => fa(env) == fb(env) ? 1.0 : 0.0,
      '!=' => (env) => fa(env) != fb(env) ? 1.0 : 0.0,
      _ => throw StateError(op),
    };
  }

  @override
  int get _precedence => switch (op) {
    '+' || '-' => 1,
    '*' || '/' => 2,
    '^' => 4,
    _ => 0,
  };

  @override
  void _write(StringBuffer b) {
    final p = _precedence;
    if (op == '^') {
      // Right associative; the base binds tighter than a unary minus.
      _writeChild(b, a, 5);
      b.write('^');
      _writeChild(b, this.b, 3);
      return;
    }
    // Left associative: the right operand needs one level more (a - (b - c)).
    _writeChild(b, a, p == 0 ? 1 : p);
    b.write(' $op ');
    _writeChild(b, this.b, p + 1);
  }
}

/// A call of one of [Expr.functions].
final class ExprCall extends Expr {
  const ExprCall(this.fn, this.args);

  final ExprFn fn;
  final List<Expr> args;

  @override
  void _visit(void Function(Expr) f) {
    f(this);
    for (final a in args) {
      a._visit(f);
    }
  }

  @override
  Expr derivative(String variable) {
    if (fn.dataOnly) {
      throw UnsupportedError('${fn.name} has no derivative (data only)');
    }
    final x = args[0];
    final dx = x.derivative(variable);
    Expr chain(Expr outer) => Expr.mul(outer, dx);
    switch (fn.name) {
      case 'exp':
        return chain(this);
      case 'log':
        return Expr.div(dx, x);
      case 'sqrt':
        return Expr.div(dx, Expr.mul(const ExprNum(2), this));
      case 'abs':
        // abs'(0) = 1, as in JAX (so exported couplings agree at 0).
        return chain(
          Expr.call('where', [
            ExprBinary('>=', x, const ExprNum(0)),
            const ExprNum(1),
            const ExprNum(-1),
          ]),
        );
      case 'sign':
        return const ExprNum(0);
      case 'tanh':
        return chain(
          Expr.sub(const ExprNum(1), Expr.pow(this, const ExprNum(2))),
        );
      case 'sigmoid':
        return chain(Expr.mul(this, Expr.sub(const ExprNum(1), this)));
      case 'min':
        return Expr.call('where', [
          ExprBinary('<=', args[0], args[1]),
          dx,
          args[1].derivative(variable),
        ]);
      case 'max':
        return Expr.call('where', [
          ExprBinary('>=', args[0], args[1]),
          dx,
          args[1].derivative(variable),
        ]);
      case 'clip':
        // As the GPU's clamp: the bounds pass nothing.
        return Expr.call('where', [
          ExprBinary('<=', x, args[1]),
          const ExprNum(0),
          Expr.call('where', [
            ExprBinary('>=', x, args[2]),
            const ExprNum(0),
            dx,
          ]),
        ]);
      case 'where':
        return Expr.call('where', [
          args[0],
          args[1].derivative(variable),
          args[2].derivative(variable),
        ]);
    }
    throw StateError(fn.name);
  }

  @override
  double Function(List<double>) _compile(Map<String, int> index) {
    if (fn.dataOnly) {
      throw UnsupportedError(
        '${fn.name} is for data transforms only (it reads other rows)',
      );
    }
    final fs = [for (final a in args) a._compile(index)];
    final f0 = fs[0];
    switch (fn.name) {
      case 'exp':
        return (env) => math.exp(f0(env));
      case 'log':
        return (env) => math.log(f0(env));
      case 'sqrt':
        return (env) => math.sqrt(f0(env));
      case 'abs':
        return (env) => f0(env).abs();
      case 'sign':
        return (env) => f0(env).sign;
      case 'tanh':
        return (env) => hgfTanh(f0(env));
      case 'sigmoid':
        return (env) => sigmoid(f0(env));
    }
    final f1 = fs[1];
    switch (fn.name) {
      case 'min':
        return (env) {
          final a = f0(env), b = f1(env);
          return a <= b ? a : b;
        };
      case 'max':
        return (env) {
          final a = f0(env), b = f1(env);
          return a >= b ? a : b;
        };
    }
    final f2 = fs[2];
    switch (fn.name) {
      case 'clip':
        return (env) {
          final x = f0(env), lo = f1(env), hi = f2(env);
          return x < lo ? lo : (x > hi ? hi : x);
        };
      case 'where':
        return (env) {
          final c = f0(env);
          return c.isNaN ? double.nan : (c != 0 ? f1(env) : f2(env));
        };
    }
    throw StateError(fn.name);
  }

  @override
  int get _precedence => 5;

  @override
  void _write(StringBuffer b) {
    b.write('${fn.name}(');
    for (var i = 0; i < args.length; i++) {
      if (i > 0) b.write(', ');
      args[i]._write(b);
    }
    b.write(')');
  }
}

// The parser: precedence climbing over a token list.

enum _T { num, str, id, op, lp, rp, comma, end }

final class _Tok {
  const _Tok(this.type, this.text, this.offset);

  final _T type;
  final String text;
  final int offset;
}

final class _Parser {
  _Parser(this.source) : tokens = _lex(source);

  final String source;
  final List<_Tok> tokens;
  var _i = 0;

  _Tok get _peek => tokens[_i];

  Never _fail(String message, int offset) =>
      throw FormatException(message, source, offset);

  static List<_Tok> _lex(String s) {
    final out = <_Tok>[];
    var i = 0;
    bool isDigit(int c) => c >= 48 && c <= 57;
    bool isIdStart(int c) =>
        (c >= 65 && c <= 90) || (c >= 97 && c <= 122) || c == 95;
    bool isIdPart(int c) => isIdStart(c) || isDigit(c);
    while (i < s.length) {
      final c = s.codeUnitAt(i);
      if (c == 32 || c == 9 || c == 10 || c == 13) {
        i++;
        continue;
      }
      final start = i;
      if (isDigit(c) ||
          (c == 46 && i + 1 < s.length && isDigit(s.codeUnitAt(i + 1)))) {
        while (i < s.length && (isDigit(s.codeUnitAt(i)) || s[i] == '.')) {
          i++;
        }
        if (i < s.length && (s[i] == 'e' || s[i] == 'E')) {
          var j = i + 1;
          if (j < s.length && (s[j] == '+' || s[j] == '-')) j++;
          if (j < s.length && isDigit(s.codeUnitAt(j))) {
            i = j;
            while (i < s.length && isDigit(s.codeUnitAt(i))) {
              i++;
            }
          }
        }
        out.add(_Tok(_T.num, s.substring(start, i), start));
        continue;
      }
      if (isIdStart(c)) {
        i++;
        while (i < s.length) {
          final d = s.codeUnitAt(i);
          if (isIdPart(d)) {
            i++;
          } else if (d == 46 &&
              i + 1 < s.length &&
              isIdPart(s.codeUnitAt(i + 1))) {
            i++;
          } else {
            break;
          }
        }
        out.add(_Tok(_T.id, s.substring(start, i), start));
        continue;
      }
      if (c == 96 || c == 39 || c == 34) {
        // `name`, 'string', "string".
        final end = s.indexOf(s[i], i + 1);
        if (end < 0) {
          throw FormatException('Unclosed ${s[i]}', s, i);
        }
        out.add(_Tok(c == 96 ? _T.id : _T.str, s.substring(i + 1, end), start));
        i = end + 1;
        continue;
      }
      final two = i + 1 < s.length ? s.substring(i, i + 2) : '';
      if (const {'<=', '>=', '==', '!=', '**'}.contains(two)) {
        out.add(_Tok(_T.op, two == '**' ? '^' : two, start));
        i += 2;
        continue;
      }
      final one = s[i];
      i++;
      switch (one) {
        case '+' || '-' || '*' || '/' || '^' || '<' || '>':
          out.add(_Tok(_T.op, one, start));
        case '(':
          out.add(_Tok(_T.lp, one, start));
        case ')':
          out.add(_Tok(_T.rp, one, start));
        case ',':
          out.add(_Tok(_T.comma, one, start));
        default:
          throw FormatException('Unexpected "$one"', s, start);
      }
    }
    out.add(_Tok(_T.end, '', s.length));
    return out;
  }

  Expr parse() {
    if (_peek.type == _T.end) _fail('Empty expression', 0);
    final e = _comparison();
    if (_peek.type != _T.end) {
      _fail('Unexpected "${_peek.text}"', _peek.offset);
    }
    return e;
  }

  bool _isOp(Set<String> ops) =>
      _peek.type == _T.op && ops.contains(_peek.text);

  Expr _comparison() {
    final a = _additive();
    if (_isOp(ExprBinary.comparisons)) {
      final op = tokens[_i++].text;
      final b = _additive();
      if (_isOp(ExprBinary.comparisons)) {
        _fail('Comparisons don\'t chain; use parentheses', _peek.offset);
      }
      return ExprBinary(op, a, b);
    }
    return a;
  }

  Expr _additive() {
    var a = _multiplicative();
    while (_isOp(const {'+', '-'})) {
      final op = tokens[_i++].text;
      a = ExprBinary(op, a, _multiplicative());
    }
    return a;
  }

  Expr _multiplicative() {
    var a = _unary();
    while (_isOp(const {'*', '/'})) {
      final op = tokens[_i++].text;
      a = ExprBinary(op, a, _unary());
    }
    return a;
  }

  Expr _unary() {
    if (_isOp(const {'-'})) {
      _i++;
      final a = _unary();
      return a is ExprNum ? ExprNum(-a.value) : ExprNeg(a);
    }
    if (_isOp(const {'+'})) {
      _i++;
      return _unary();
    }
    return _power();
  }

  Expr _power() {
    final base = _atom();
    if (_isOp(const {'^'})) {
      _i++;
      // Right associative; the exponent may carry a sign (2^-x).
      return ExprBinary('^', base, _unary());
    }
    return base;
  }

  Expr _atom() {
    final t = tokens[_i++];
    switch (t.type) {
      case _T.num:
        final v = double.tryParse(t.text);
        if (v == null) _fail('Not a number: "${t.text}"', t.offset);
        return ExprNum(v);
      case _T.str:
        return ExprStr(t.text);
      case _T.id:
        if (_peek.type != _T.lp) return ExprVar(t.text);
        final fn = Expr.functions[t.text];
        if (fn == null) {
          _fail(
            'Unknown function "${t.text}"; the functions are '
            '${Expr.functions.keys.join(', ')}',
            t.offset,
          );
        }
        _i++;
        final args = <Expr>[];
        if (_peek.type != _T.rp) {
          args.add(_comparison());
          while (_peek.type == _T.comma) {
            _i++;
            args.add(_comparison());
          }
        }
        if (_peek.type != _T.rp) _fail('Expected ")"', _peek.offset);
        _i++;
        if (args.length < fn.minArgs || args.length > fn.maxArgs) {
          _fail('${fn.name} takes ${fn.arity} argument(s)', t.offset);
        }
        return ExprCall(fn, args);
      case _T.lp:
        final e = _comparison();
        if (_peek.type != _T.rp) _fail('Expected ")"', _peek.offset);
        _i++;
        return e;
      case _T.end:
        _fail('Unexpected end', t.offset);
      default:
        _fail('Unexpected "${t.text}"', t.offset);
    }
  }
}
