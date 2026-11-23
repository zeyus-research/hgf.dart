import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'codec.dart' show jsonDouble, readJsonDouble;
import 'response_model.dart';
import 'special.dart' show standardNormal;

/// A belief an [ExprReadout]'s expressions read: [field] of [node].
typedef ExprBelief = ({int node, NodeField field});

/// How an [ExprReadout] turns its expressions into a surprise.
enum ExprReadoutKind {
  /// A choice among options, option `k` worth `options[k]`: probability
  /// `exp(vₖ) / Σⱼ exp(vⱼ)` (put an inverse temperature in the values).
  /// The response is the chosen option's index.
  choice,

  /// A continuous response `~ N(mean, 1 / precision)`.
  gaussian,

  /// Any surprise (negative log-likelihood), an expression that also reads
  /// `response`. It can't simulate responses.
  surprise,
}

/// A response model written as expressions ([Expr]) of the beliefs, its
/// parameters and (for [ExprReadoutKind.surprise]) the `response`: what the
/// model builder writes, and what `ResponseModel.fn`'s closures can't do.
/// It serialises, runs on the GPU kernels, has exact gradients (from the
/// expressions' symbolic derivatives) and exports to Dart and pyhgf.
///
/// ```dart
/// // Two options, a bias towards the second, and a fitted β.
/// final readout = ExprReadout.choice(
///   ['beta * mu1', 'beta * mu2 + bias'],
///   beliefs: {'mu1': (node: 1, field: NodeField.expectedMean),
///             'mu2': (node: 2, field: NodeField.expectedMean)},
///   parameters: {'beta': 1.0, 'bias': 0.0},
/// );
/// ```
///
/// A variable is a belief (in [beliefs]), a parameter (in [parameters],
/// fitted by name with `Parameter.response`), `response`, or `previous`:
/// the previous trial's response, `-1` where there is none (the first
/// trial, or a missed response), so `where(previous == 2, kappa, 0)` adds
/// a stickiness to option 2. A missing response (NaN) scores 0.
final class ExprReadout extends ResponseModel {
  ExprReadout._(
    this.kind,
    this.expressions,
    Map<String, ExprBelief> beliefs,
    Map<String, double> parameters,
  ) : beliefs = Map.unmodifiable(beliefs),
      parameters = Map.unmodifiable(parameters) {
    final names = [...this.beliefs.keys, ...this.parameters.keys];
    final clash = names.toSet().length != names.length;
    if (clash ||
        names.contains(responseVariable) ||
        names.contains(previousVariable)) {
      throw ArgumentError(
        'Beliefs and parameters need distinct names, and none may be '
        '"$responseVariable" or "$previousVariable"',
      );
    }
    _vars = [...names, responseVariable, previousVariable];
    for (final e in expressions) {
      if (e.isDataOnly) {
        throw FormatException(
          '$e: data-only functions and strings are for data transforms',
        );
      }
      for (final v in e.variables) {
        if (!_vars.contains(v)) {
          throw FormatException(
            '"$v" is not a belief, a parameter, '
            '${kind == ExprReadoutKind.surprise ? '"response" ' : ''}'
            'or "previous"',
          );
        }
        if (v == responseVariable && kind != ExprReadoutKind.surprise) {
          throw const FormatException(
            'Only a surprise expression reads "response"',
          );
        }
      }
    }
    final nb = this.beliefs.length, np = this.parameters.length;
    _values = [for (final e in expressions) e.compile(_vars)];
    _partials = [
      for (final e in expressions)
        [
          for (var v = 0; v < nb + np; v++)
            e.variables.contains(_vars[v])
                ? e.derivative(_vars[v]).compile(_vars)
                : null,
        ],
    ];
    _env = Float64List(nb + np + 2);
    readsPreviousResponse = expressions.any(
      (e) => e.variables.contains(previousVariable),
    );
  }

  /// Option `k` worth `options[k]` (see [ExprReadoutKind.choice]).
  factory ExprReadout.choice(
    List<String> options, {
    Map<String, ExprBelief> beliefs = const {},
    Map<String, double> parameters = const {},
  }) {
    if (options.length < 2) {
      throw ArgumentError.value(options, 'options', 'Needs two or more');
    }
    return ExprReadout._(
      ExprReadoutKind.choice,
      [for (final o in options) Expr.parse(o)],
      beliefs,
      parameters,
    );
  }

  /// `response ~ N(mean, 1 / precision)` (see [ExprReadoutKind.gaussian]).
  factory ExprReadout.gaussian({
    required String mean,
    required String precision,
    Map<String, ExprBelief> beliefs = const {},
    Map<String, double> parameters = const {},
  }) => ExprReadout._(
    ExprReadoutKind.gaussian,
    [Expr.parse(mean), Expr.parse(precision)],
    beliefs,
    parameters,
  );

  /// Any [surprise] (see [ExprReadoutKind.surprise]).
  factory ExprReadout.surprise(
    String surprise, {
    Map<String, ExprBelief> beliefs = const {},
    Map<String, double> parameters = const {},
  }) => ExprReadout._(
    ExprReadoutKind.surprise,
    [Expr.parse(surprise)],
    beliefs,
    parameters,
  );

  /// The variable holding the response, in a surprise expression.
  static const responseVariable = 'response';

  /// The variable holding the previous trial's response (`-1`: none).
  static const previousVariable = 'previous';

  @override
  late final bool readsPreviousResponse;

  final ExprReadoutKind kind;

  /// The options' values ([ExprReadoutKind.choice]); the mean and the
  /// precision ([ExprReadoutKind.gaussian]); the surprise.
  final List<Expr> expressions;

  /// The beliefs the expressions read, by variable name.
  final Map<String, ExprBelief> beliefs;

  @override
  final Map<String, double> parameters;

  late final List<String> _vars;
  late final List<double Function(List<double>)> _values;

  /// Per expression, per belief then parameter: the compiled partial
  /// derivative (null where the expression doesn't read it).
  late final List<List<double Function(List<double>)?>> _partials;
  late final Float64List _env;

  @override
  Map<String, Object?> toJson() => {
    'type': 'ExprReadout',
    'kind': kind.name,
    switch (kind) {
      ExprReadoutKind.choice => 'options',
      ExprReadoutKind.gaussian => 'gaussian',
      ExprReadoutKind.surprise => 'surprise',
    }: switch (kind) {
      ExprReadoutKind.choice => [for (final e in expressions) '$e'],
      ExprReadoutKind.gaussian => {
        'mean': '${expressions[0]}',
        'precision': '${expressions[1]}',
      },
      ExprReadoutKind.surprise => '${expressions[0]}',
    },
    if (beliefs.isNotEmpty)
      'beliefs': {
        for (final MapEntry(:key, :value) in beliefs.entries)
          key: {'node': value.node, 'field': value.field.pyhgfName},
      },
    if (parameters.isNotEmpty)
      'parameters': {
        for (final MapEntry(:key, :value) in parameters.entries)
          key: jsonDouble(value),
      },
  };

  /// Reads [toJson]'s output.
  factory ExprReadout.fromJson(Map<String, Object?> json) {
    final beliefs = {
      for (final MapEntry(:key, :value)
          in ((json['beliefs'] as Map<String, Object?>?) ?? const {}).entries)
        key: (
          node: ((value as Map<String, Object?>)['node'] as num).toInt(),
          field: NodeField.parse(value['field'] as String),
        ),
    };
    final parameters = {
      for (final MapEntry(:key, :value)
          in ((json['parameters'] as Map<String, Object?>?) ?? const {})
              .entries)
        key: readJsonDouble(value),
    };
    return switch (json['kind']) {
      'choice' => ExprReadout.choice(
        (json['options'] as List<Object?>).cast<String>(),
        beliefs: beliefs,
        parameters: parameters,
      ),
      'gaussian' => () {
        final g = json['gaussian'] as Map<String, Object?>;
        return ExprReadout.gaussian(
          mean: g['mean'] as String,
          precision: g['precision'] as String,
          beliefs: beliefs,
          parameters: parameters,
        );
      }(),
      'surprise' => ExprReadout.surprise(
        json['surprise'] as String,
        beliefs: beliefs,
        parameters: parameters,
      ),
      final k => throw FormatException('Bad ExprReadout kind: $k'),
    };
  }

  @override
  void prepare(Network network) {
    for (final b in beliefs.values) {
      if (b.node < 0 || b.node >= network.nNodes) {
        throw ArgumentError.value(b.node, 'node', 'No such node');
      }
    }
  }

  @override
  bool get supportsGradients =>
      beliefs.values.every((b) => BatchNetwork.tangentFields.contains(b.field));

  void _load(
    BatchNetwork network,
    int element,
    double response,
    double previous,
    Float64List params,
    int offset,
  ) {
    var i = 0;
    for (final b in beliefs.values) {
      _env[i++] = network.get(element, b.node, b.field);
    }
    for (var j = 0; j < parameters.length; j++) {
      _env[i++] = params[offset + j];
    }
    _env[i] = response;
    _env[i + 1] = previous.isNaN ? -1.0 : previous;
  }

  /// `log Σ exp(vₖ)`, stably.
  static double _logSumExp(Float64List v) {
    var top = double.negativeInfinity;
    for (final x in v) {
      if (x > top) top = x;
    }
    if (!top.isFinite) return top;
    var sum = 0.0;
    for (final x in v) {
      sum += math.exp(x - top);
    }
    return top + math.log(sum);
  }

  int _choice(double response) {
    final r = response.toInt();
    if (r < 0 || r >= expressions.length || r != response) {
      throw ArgumentError.value(
        response,
        'response',
        'Not an option index (0 .. ${expressions.length - 1})',
      );
    }
    return r;
  }

  Float64List _optionValues() =>
      Float64List.fromList([for (final f in _values) f(_env)]);

  @override
  double surprise(
    BatchNetwork network,
    int element,
    int t,
    double response,
    Float64List params,
    int offset,
  ) => surpriseAfter(network, element, t, response, double.nan, params, offset);

  @override
  double surpriseAfter(
    BatchNetwork network,
    int element,
    int t,
    double response,
    double previous,
    Float64List params,
    int offset, {
    Float64List? state,
    int stateAt = 0,
  }) {
    if (response.isNaN) return 0.0;
    _load(network, element, response, previous, params, offset);
    final double s;
    switch (kind) {
      case ExprReadoutKind.choice:
        final v = _optionValues();
        s = _logSumExp(v) - v[_choice(response)];
      case ExprReadoutKind.gaussian:
        s = gaussianSurprise(response, _values[0](_env), _values[1](_env));
      case ExprReadoutKind.surprise:
        s = _values[0](_env);
    }
    return s.isNaN ? double.infinity : s;
  }

  @override
  double surprisePartials(
    BatchNetwork network,
    int element,
    int t,
    double response,
    Float64List params,
    int offset,
    ResponsePartials partials,
  ) => surprisePartialsAfter(
    network,
    element,
    t,
    response,
    double.nan,
    params,
    offset,
    partials,
  );

  @override
  double surprisePartialsAfter(
    BatchNetwork network,
    int element,
    int t,
    double response,
    double previous,
    Float64List params,
    int offset,
    ResponsePartials partials, {
    Float64List? state,
    int stateAt = 0,
  }) {
    if (response.isNaN) return 0.0;
    _load(network, element, response, previous, params, offset);
    // ∂S/∂(each expression's value).
    final Float64List dS;
    final double s;
    switch (kind) {
      case ExprReadoutKind.choice:
        final v = _optionValues();
        final r = _choice(response);
        final z = _logSumExp(v);
        s = z - v[r];
        dS = Float64List(v.length);
        for (var k = 0; k < v.length; k++) {
          dS[k] = math.exp(v[k] - z) - (k == r ? 1.0 : 0.0);
        }
      case ExprReadoutKind.gaussian:
        final m = _values[0](_env), p = _values[1](_env);
        final d = response - m;
        s = gaussianSurprise(response, m, p);
        dS = Float64List.fromList([-p * d, 0.5 * (d * d - 1.0 / p)]);
      case ExprReadoutKind.surprise:
        s = _values[0](_env);
        dS = Float64List.fromList([1.0]);
    }
    final nb = beliefs.length;
    final bs = beliefs.values.toList();
    for (var e = 0; e < expressions.length; e++) {
      final row = _partials[e];
      for (var v = 0; v < row.length; v++) {
        final f = row[v];
        if (f == null) continue;
        final g = dS[e] * f(_env);
        if (v < nb) {
          partials.add(bs[v].node, bs[v].field, g);
        } else {
          partials.addParameter(v - nb, g);
        }
      }
    }
    return s.isNaN ? double.infinity : s;
  }

  @override
  double sample(
    BatchNetwork network,
    int element,
    int t,
    Float64List params,
    int offset,
    math.Random rng,
  ) => sampleAfter(network, element, t, double.nan, params, offset, rng);

  @override
  double sampleAfter(
    BatchNetwork network,
    int element,
    int t,
    double previous,
    Float64List params,
    int offset,
    math.Random rng, {
    Float64List? state,
    int stateAt = 0,
  }) {
    _load(network, element, double.nan, previous, params, offset);
    switch (kind) {
      case ExprReadoutKind.choice:
        final v = _optionValues();
        final z = _logSumExp(v);
        if (!z.isFinite) return double.nan;
        var u = rng.nextDouble();
        for (var k = 0; k < v.length - 1; k++) {
          u -= math.exp(v[k] - z);
          if (u < 0) return k.toDouble();
        }
        return (v.length - 1).toDouble();
      case ExprReadoutKind.gaussian:
        return _values[0](_env) +
            standardNormal(rng) / math.sqrt(_values[1](_env));
      case ExprReadoutKind.surprise:
        return super.sample(network, element, t, params, offset, rng);
    }
  }
}
