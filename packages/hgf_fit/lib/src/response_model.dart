import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'codec.dart' show jsonDouble;
import 'special.dart';

/// How a subject's responses arise from the network's beliefs, with named
/// parameters that may be fitted per subject.
///
/// Unlike `ResponseFunction` in `package:hgf`, which carries fixed
/// parameters, a [ResponseModel] reads its parameters per batch element, so
/// one batched pass can evaluate many parameter sets (and subjects) at once.
///
/// [surprise] and [sample] are called right after trial `t`'s belief update,
/// so the `expected*` fields hold the predictions for that trial (see
/// `ResponseFunction`).
abstract class ResponseModel {
  const ResponseModel();

  /// A model built from closures. [parameters] maps each parameter name to
  /// its default value; [surprise] reads them from `params[offset + i]`, in
  /// the order of [parameters].
  const factory ResponseModel.fn({
    Map<String, double> parameters,
    required double Function(
      BatchNetwork network,
      int element,
      int t,
      double response,
      Float64List params,
      int offset,
    )
    surprise,
    double Function(
      BatchNetwork network,
      int element,
      int t,
      Float64List params,
      int offset,
      math.Random rng,
    )?
    sample,
    bool usesResponses,
    double Function(
      BatchNetwork network,
      int element,
      int t,
      double response,
      Float64List params,
      int offset,
      ResponsePartials partials,
    )?
    surprisePartials,
  }) = _FnResponseModel;

  /// The parameter names with their default values (used when a parameter
  /// isn't fitted), in the order [surprise] reads them.
  Map<String, double> get parameters => const {};

  /// Whether the model scores a subject's responses. If not (e.g.
  /// [ObservationSurprise]), subjects need no responses and [surprise]
  /// receives NaN.
  bool get usesResponses => true;

  /// Called once per batch to check the network fits the model. Throws
  /// [ArgumentError] if not.
  void prepare(Network network) {}

  /// Surprise (negative log-likelihood) of [response] on trial [t] for
  /// [element], whose
  /// parameters are `params[offset + i]` (`i` in the order of
  /// [parameters]). A NaN response marks a missed trial; built-in models
  /// return 0 for it, and `+∞` where the result is NaN.
  double surprise(
    BatchNetwork network,
    int element,
    int t,
    double response,
    Float64List params,
    int offset,
  );

  /// Whether [surprisePartials] is implemented, so the model can be fitted
  /// with exact gradients.
  bool get supportsGradients => false;

  /// [surprise], also reporting its partial derivatives: with respect to
  /// each belief it reads (`partials.add(node, field, ∂s/∂belief)`) and to
  /// each of its parameters (`partials.addParameter(i, ∂s/∂params[offset +
  /// i])`). The fitting code turns them into the gradient.
  double surprisePartials(
    BatchNetwork network,
    int element,
    int t,
    double response,
    Float64List params,
    int offset,
    ResponsePartials partials,
  ) => throw UnsupportedError('$runtimeType has no gradient form');

  /// A response drawn from the model for [element] (for simulation).
  double sample(
    BatchNetwork network,
    int element,
    int t,
    Float64List params,
    int offset,
    math.Random rng,
  ) => throw UnsupportedError('$runtimeType cannot simulate responses');

  /// Whether the model reads the previous trial's response (e.g.
  /// [SoftmaxReadout.stickiness]). The callers always go through
  /// [surpriseAfter], [surprisePartialsAfter] and [sampleAfter], which
  /// such a model overrides; the others ignore `previous`.
  bool get readsPreviousResponse => false;

  /// How many values the model carries from trial to trial for each
  /// element (e.g. [SoftmaxReadout.choiceKernel]'s trace), or 0. The
  /// callers keep them: zeros before the first trial, [state] from
  /// `stateAt` in the `…After` methods, and [advanceState] after each
  /// trial.
  int get stateSize => 0;

  /// Moves [state] (from [stateAt]) past a trial whose response was
  /// [response] (NaN: missed). Called after that trial's `surpriseAfter`,
  /// `surprisePartialsAfter` or `sampleAfter` (with the draw).
  void advanceState(
    Float64List state,
    int stateAt,
    double response,
    Float64List params,
    int offset,
  ) {}

  /// [surprise], given the response on trial `t − 1` ([previous]; NaN on the
  /// first trial and after a missed one) and the carried [state].
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
  }) => surprise(network, element, t, response, params, offset);

  /// [surprisePartials], given the response on trial `t − 1` and the
  /// carried [state].
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
  }) =>
      surprisePartials(network, element, t, response, params, offset, partials);

  /// [sample], given the response on trial `t − 1` (the previous draw when
  /// simulating) and the carried [state].
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
  }) => sample(network, element, t, params, offset, rng);

  /// This response model as JSON, with a `"type"` key, for `FitCodec` (e.g. to
  /// send a model to a Web Worker). The built-in ones are covered; a custom
  /// one overrides this and gives the `FitCodec` a decoder for its type.
  /// `ResponseModel.fn` can't be serialised (closures are code).
  Map<String, Object?> toJson() => throw UnsupportedError(
    '$runtimeType can\'t be serialised: override toJson() and give '
    'FitCodec a decoder for it',
  );
}

final class _FnResponseModel extends ResponseModel {
  const _FnResponseModel({
    this.parameters = const {},
    required double Function(BatchNetwork, int, int, double, Float64List, int)
    surprise,
    double Function(BatchNetwork, int, int, Float64List, int, math.Random)?
    sample,
    this.usesResponses = true,
    double Function(
      BatchNetwork,
      int,
      int,
      double,
      Float64List,
      int,
      ResponsePartials,
    )?
    surprisePartials,
  }) : surpriseFn = surprise,
       sampleFn = sample,
       surprisePartialsFn = surprisePartials;

  @override
  final Map<String, double> parameters;
  @override
  final bool usesResponses;
  final double Function(BatchNetwork, int, int, double, Float64List, int)
  surpriseFn;
  final double Function(BatchNetwork, int, int, Float64List, int, math.Random)?
  sampleFn;
  final double Function(
    BatchNetwork,
    int,
    int,
    double,
    Float64List,
    int,
    ResponsePartials,
  )?
  surprisePartialsFn;

  @override
  bool get supportsGradients => surprisePartialsFn != null;

  @override
  double surprisePartials(
    BatchNetwork network,
    int element,
    int t,
    double response,
    Float64List params,
    int offset,
    ResponsePartials partials,
  ) {
    final f = surprisePartialsFn;
    if (f == null) {
      return super.surprisePartials(
        network,
        element,
        t,
        response,
        params,
        offset,
        partials,
      );
    }
    return f(network, element, t, response, params, offset, partials);
  }

  @override
  double surprise(
    BatchNetwork network,
    int element,
    int t,
    double response,
    Float64List params,
    int offset,
  ) => surpriseFn(network, element, t, response, params, offset);

  @override
  double sample(
    BatchNetwork network,
    int element,
    int t,
    Float64List params,
    int offset,
    math.Random rng,
  ) {
    final s = sampleFn;
    if (s == null) {
      return super.sample(network, element, t, params, offset, rng);
    }
    return s(network, element, t, params, offset, rng);
  }
}

@pragma('vm:prefer-inline')
double _nanToInf(double x) => x.isNaN ? double.infinity : x;

void _checkNode(Network network, int node) {
  if (node < 0 || node >= network.nNodes) {
    throw ArgumentError.value(node, 'node', 'No such node');
  }
}

/// Continuous responses read out from a belief with Gaussian noise:
/// `response ~ N(belief, 1 / precision)`, where the belief is [field] of
/// [node] (by default the prediction `expectedMean` at node 0: what the
/// agent expects to observe).
///
/// Parameter: `precision` (default [precision]), the inverse variance of the
/// response noise. Fixed, this is `GaussianResponse` in `package:hgf`.
final class GaussianReadout extends ResponseModel {
  const GaussianReadout({
    this.node = 0,
    this.field = NodeField.expectedMean,
    this.precision = 1.0,
  });

  final int node;
  final NodeField field;

  /// Default response precision when it isn't fitted.
  final double precision;

  @override
  Map<String, double> get parameters => {'precision': precision};

  @override
  Map<String, Object?> toJson() => {
    'type': 'GaussianReadout',
    'node': node,
    'field': field.pyhgfName,
    'precision': jsonDouble(precision),
  };

  @override
  void prepare(Network network) => _checkNode(network, node);

  @override
  double surprise(
    BatchNetwork network,
    int element,
    int t,
    double response,
    Float64List params,
    int offset,
  ) {
    if (response.isNaN) return 0.0;
    return _nanToInf(
      gaussianSurprise(
        response,
        network.get(element, node, field),
        params[offset],
      ),
    );
  }

  @override
  bool get supportsGradients => BatchNetwork.tangentFields.contains(field);

  @override
  double surprisePartials(
    BatchNetwork network,
    int element,
    int t,
    double response,
    Float64List params,
    int offset,
    ResponsePartials partials,
  ) {
    if (response.isNaN) return 0.0;
    final belief = network.get(element, node, field);
    final precision = params[offset];
    final d = response - belief;
    partials
      ..add(node, field, -precision * d)
      ..addParameter(0, 0.5 * (d * d - 1.0 / precision));
    return _nanToInf(gaussianSurprise(response, belief, precision));
  }

  @override
  double sample(
    BatchNetwork network,
    int element,
    int t,
    Float64List params,
    int offset,
    math.Random rng,
  ) =>
      network.get(element, node, field) +
      standardNormal(rng) / math.sqrt(params[offset]);
}

/// Binary decisions (`1`/`0`) made with the probability predicted at [node],
/// sharpened by an inverse temperature β:
/// `p = μ̂^β / (μ̂^β + (1 − μ̂)^β)`.
///
/// Parameter: `inverseTemperature` (default [inverseTemperature]; at 1 this
/// is pyhgf's `binary_softmax`, otherwise
/// `binary_softmax_inverse_temperature`).
final class BinarySoftmaxReadout extends ResponseModel {
  const BinarySoftmaxReadout({this.node = 0, this.inverseTemperature = 1.0});

  final int node;

  /// Default β when it isn't fitted.
  final double inverseTemperature;

  @override
  Map<String, double> get parameters => {
    'inverseTemperature': inverseTemperature,
  };

  @override
  Map<String, Object?> toJson() => {
    'type': 'BinarySoftmaxReadout',
    'node': node,
    'inverseTemperature': jsonDouble(inverseTemperature),
  };

  @override
  void prepare(Network network) => _checkNode(network, node);

  @override
  double surprise(
    BatchNetwork network,
    int element,
    int t,
    double response,
    Float64List params,
    int offset,
  ) {
    if (response.isNaN) return 0.0;
    final p = sigmoidInverseTemperature(
      network.get(element, node, NodeField.expectedMean),
      params[offset],
    );
    return _nanToInf(binarySurprise(response, p));
  }

  @override
  bool get supportsGradients => true;

  @override
  double surprisePartials(
    BatchNetwork network,
    int element,
    int t,
    double response,
    Float64List params,
    int offset,
    ResponsePartials partials,
  ) {
    if (response.isNaN) return 0.0;
    final m = network.get(element, node, NodeField.expectedMean);
    final beta = params[offset];
    final p = sigmoidInverseTemperature(m, beta);
    // p = σ(β logit m): ∂p/∂m = p (1 − p) β / (m (1 − m)),
    // ∂p/∂β = p (1 − p) logit m.
    final ds = binarySurpriseDerivative(response, p) * p * (1.0 - p);
    partials
      ..add(node, NodeField.expectedMean, ds * beta / (m * (1.0 - m)))
      ..addParameter(0, ds * math.log(m / (1.0 - m)));
    return _nanToInf(binarySurprise(response, p));
  }

  @override
  double sample(
    BatchNetwork network,
    int element,
    int t,
    Float64List params,
    int offset,
    math.Random rng,
  ) {
    final p = sigmoidInverseTemperature(
      network.get(element, node, NodeField.expectedMean),
      params[offset],
    );
    return rng.nextDouble() < p ? 1.0 : 0.0;
  }
}

/// Choices among the categories of a categorical node, made with the
/// probabilities the agent predicts, sharpened by an inverse temperature:
/// category `k` with probability `pₖ = μ̂ₖ^β / Σⱼ μ̂ⱼ^β`, where `μ̂ₖ` is
/// the prediction of category `k`'s binary node. With two categories and
/// `μ̂₂ = 1 − μ̂₁` this is [BinarySoftmaxReadout].
///
/// A response is the chosen category's index (`0`, `1`, …; NaN: no
/// response). Parameter: `inverseTemperature` (default
/// [inverseTemperature]); `β = 0` chooses at random, large `β` the most
/// expected category.
final class CategoricalSoftmaxReadout extends ResponseModel {
  const CategoricalSoftmaxReadout({
    this.node = 0,
    this.inverseTemperature = 1.0,
  });

  /// The categorical node.
  final int node;

  /// Default β when it isn't fitted.
  final double inverseTemperature;

  @override
  Map<String, double> get parameters => {
    'inverseTemperature': inverseTemperature,
  };

  @override
  Map<String, Object?> toJson() => {
    'type': 'CategoricalSoftmaxReadout',
    'node': node,
    'inverseTemperature': jsonDouble(inverseTemperature),
  };

  @override
  void prepare(Network network) {
    _checkNode(network, node);
    if (network.kindOf(node) != NodeKind.categoricalState) {
      throw ArgumentError.value(node, 'node', 'Not a categorical node');
    }
  }

  /// `log Σⱼ μ̂ⱼ^β` (stably) and the category count, with the choice
  /// probabilities in [p] if given.
  static double _logNormaliser(
    BatchNetwork network,
    int element,
    int node,
    double beta, [
    Float64List? p,
  ]) {
    final n = network.valueParentCount(node);
    var top = double.negativeInfinity;
    for (var k = 0; k < n; k++) {
      final m = network.get(
        element,
        network.valueParent(node, k),
        NodeField.expectedMean,
      );
      final z = beta * math.log(m);
      if (z > top) top = z;
    }
    if (!top.isFinite) return top;
    var sum = 0.0;
    for (var k = 0; k < n; k++) {
      final m = network.get(
        element,
        network.valueParent(node, k),
        NodeField.expectedMean,
      );
      final w = math.exp(beta * math.log(m) - top);
      if (p != null) p[k] = w;
      sum += w;
    }
    if (p != null) {
      for (var k = 0; k < n; k++) {
        p[k] /= sum;
      }
    }
    return top + math.log(sum);
  }

  double _expected(BatchNetwork network, int element, int k) => network.get(
    element,
    network.valueParent(node, k),
    NodeField.expectedMean,
  );

  @override
  double surprise(
    BatchNetwork network,
    int element,
    int t,
    double response,
    Float64List params,
    int offset,
  ) {
    if (response.isNaN) return 0.0;
    final beta = params[offset];
    final r = response.toInt();
    final s =
        _logNormaliser(network, element, node, beta) -
        beta * math.log(_expected(network, element, r));
    return _nanToInf(s);
  }

  @override
  bool get supportsGradients => true;

  @override
  double surprisePartials(
    BatchNetwork network,
    int element,
    int t,
    double response,
    Float64List params,
    int offset,
    ResponsePartials partials,
  ) {
    if (response.isNaN) return 0.0;
    final beta = params[offset];
    final r = response.toInt();
    final n = network.valueParentCount(node);
    final p = Float64List(n);
    final s =
        _logNormaliser(network, element, node, beta, p) -
        beta * math.log(_expected(network, element, r));
    // S = log Σ μ̂ⱼ^β − β log μ̂ᵣ: ∂S/∂μ̂ₖ = β (pₖ − [k = r]) / μ̂ₖ,
    // ∂S/∂β = Σ pⱼ log μ̂ⱼ − log μ̂ᵣ.
    var dBeta = 0.0;
    for (var k = 0; k < n; k++) {
      final m = _expected(network, element, k);
      final indicator = k == r ? 1.0 : 0.0;
      partials.add(
        network.valueParent(node, k),
        NodeField.expectedMean,
        beta * (p[k] - indicator) / m,
      );
      dBeta += (p[k] - indicator) * math.log(m);
    }
    partials.addParameter(0, dBeta);
    return _nanToInf(s);
  }

  @override
  double sample(
    BatchNetwork network,
    int element,
    int t,
    Float64List params,
    int offset,
    math.Random rng,
  ) {
    final n = network.valueParentCount(node);
    final p = Float64List(n);
    final z = _logNormaliser(network, element, node, params[offset], p);
    if (!z.isFinite) return double.nan;
    var u = rng.nextDouble();
    for (var k = 0; k < n - 1; k++) {
      u -= p[k];
      if (u < 0) return k.toDouble();
    }
    return (n - 1).toDouble();
  }
}

/// Choices among options whose values the agent reads off its beliefs, made
/// with a softmax: option `k` with probability
/// `pₖ = exp(β vₖ) / Σⱼ exp(β vⱼ)`, where `vₖ = Σ w · belief` sums the
/// [field] of the nodes in `options[k]` (node → weight).
///
/// This is the decision rule of bandit tasks and the Iowa Gambling Task:
/// one option per arm or deck, valued at the predicted outcome of its input
/// node ([SoftmaxReadout.nodes]), or as a combination such as pyhgf's
/// two-armed bandit example, where arm A is worth `W_A − L_A`
/// (`{0: 1, 1: -1}`) from the predicted win and loss probabilities. Pair it
/// with `HgfModel.simulateChoices` to simulate a task where only the chosen
/// option's outcome is seen.
///
/// Optional terms, each a parameter when set (null leaves it out):
///
/// - [bias]: a per-option intercept `bₖ` (parameters `bias_0`, `bias_1`, …),
///   so an option can be worth something of its own; an option with no
///   nodes (`{}`) is worth its bias alone (e.g. "do nothing"). Only
///   differences between biases matter: fit all but one.
/// - [stickiness]: `κ`, added to the option chosen on the previous trial
///   (perseveration for `κ > 0`, alternation for `κ < 0`). Nothing is added
///   on the first trial or after a missed response.
/// - [choiceKernel] and [choiceKernelRate]: a decaying trace of past
///   choices (the choice kernel of Wilson & Collins, 2019), `φ cₖ` added to
///   each option, where `cₖ` starts at 0 and after each response `r` moves
///   `cₖ ← cₖ + α ([k = r] − cₖ)` (a missed response leaves it as it is).
///   `α = 1` is one-back stickiness; smaller rates remember further back.
///   Set both or neither; give the rate a prior on (0, 1).
/// - [lapse]: `ε`, the probability of choosing at random:
///   `pₖ = (1 − ε) softmax(u)ₖ + ε / K`. Give it a prior on (0, 1).
///
/// With all of them, option `k`'s utility is
/// `uₖ = β vₖ + bₖ + κ [k = prev] + φ cₖ` (the bias, stickiness and kernel
/// are on the utility scale, not multiplied by β).
///
/// A response is the chosen option's index (`0`, `1`, …; NaN: no
/// response). Parameters: `inverseTemperature` (default
/// [inverseTemperature]; `β = 0` chooses at random, large `β` the option
/// of highest value), then `stickiness`, `choiceKernel` (φ),
/// `choiceKernelRate` (α), `lapse` and `bias_k` where set.
final class SoftmaxReadout extends ResponseModel {
  const SoftmaxReadout({
    required this.options,
    this.field = NodeField.expectedMean,
    this.inverseTemperature = 1.0,
    this.stickiness,
    this.choiceKernel,
    this.choiceKernelRate,
    this.lapse,
    this.bias,
  });

  /// One option per node, valued at that node's [field].
  SoftmaxReadout.nodes(
    List<int> nodes, {
    NodeField field = NodeField.expectedMean,
    double inverseTemperature = 1.0,
    double? stickiness,
    double? choiceKernel,
    double? choiceKernelRate,
    double? lapse,
    List<double>? bias,
  }) : this(
         options: [
           for (final n in nodes) {n: 1.0},
         ],
         field: field,
         inverseTemperature: inverseTemperature,
         stickiness: stickiness,
         choiceKernel: choiceKernel,
         choiceKernelRate: choiceKernelRate,
         lapse: lapse,
         bias: bias,
       );

  /// Each option's value as weights on the nodes' [field] (empty: worth
  /// its [bias] alone).
  final List<Map<int, double>> options;

  /// The belief the values are read from (default: the prediction).
  final NodeField field;

  /// Default β when it isn't fitted.
  final double inverseTemperature;

  /// Default stickiness `κ`, or null for none.
  final double? stickiness;

  /// Default choice-kernel weight `φ`, or null for no choice kernel.
  final double? choiceKernel;

  /// Default choice-kernel rate `α` (set with [choiceKernel]).
  final double? choiceKernelRate;

  /// Default lapse rate `ε`, or null for none.
  final double? lapse;

  /// Default per-option biases (one per option), or null for none.
  final List<double>? bias;

  bool get _kernel => choiceKernel != null;
  int get _iKernel => stickiness == null ? 1 : 2;
  int get _iLapse => _iKernel + (_kernel ? 2 : 0);
  int get _iBias => _iLapse + (lapse == null ? 0 : 1);

  @override
  Map<String, double> get parameters => {
    'inverseTemperature': inverseTemperature,
    'stickiness': ?stickiness,
    'choiceKernel': ?choiceKernel,
    'choiceKernelRate': ?choiceKernelRate,
    'lapse': ?lapse,
    if (bias case final b?)
      for (var k = 0; k < b.length; k++) 'bias_$k': b[k],
  };

  @override
  bool get readsPreviousResponse => stickiness != null;

  /// The trace `cₖ`, then `∂cₖ/∂α` (for the gradient).
  @override
  int get stateSize => _kernel ? 2 * options.length : 0;

  @override
  void advanceState(
    Float64List state,
    int stateAt,
    double response,
    Float64List params,
    int offset,
  ) {
    if (!_kernel || response.isNaN) return;
    final r = _choice(response);
    final rate = params[offset + _iKernel + 1];
    final n = options.length;
    for (var k = 0; k < n; k++) {
      final c = state[stateAt + k];
      final e = (k == r ? 1.0 : 0.0) - c;
      // ∂c′/∂α = (1 − α) ∂c/∂α + ([k = r] − c).
      state[stateAt + n + k] = (1 - rate) * state[stateAt + n + k] + e;
      state[stateAt + k] = c + rate * e;
    }
  }

  @override
  Map<String, Object?> toJson() => {
    'type': 'SoftmaxReadout',
    'options': [
      for (final o in options)
        [
          for (final MapEntry(:key, :value) in o.entries)
            {'node': key, 'weight': jsonDouble(value)},
        ],
    ],
    'field': field.pyhgfName,
    'inverseTemperature': jsonDouble(inverseTemperature),
    if (stickiness case final k?) 'stickiness': jsonDouble(k),
    if (choiceKernel case final k?) 'choiceKernel': jsonDouble(k),
    if (choiceKernelRate case final k?) 'choiceKernelRate': jsonDouble(k),
    if (lapse case final e?) 'lapse': jsonDouble(e),
    if (bias case final b?) 'bias': [for (final x in b) jsonDouble(x)],
  };

  @override
  void prepare(Network network) {
    if (options.length < 2) {
      throw ArgumentError.value(options, 'options', 'Needs two or more');
    }
    if (bias case final b? when b.length != options.length) {
      throw ArgumentError.value(b, 'bias', 'Needs one per option');
    }
    if ((choiceKernel == null) != (choiceKernelRate == null)) {
      throw ArgumentError(
        'choiceKernel and choiceKernelRate: set both or neither',
      );
    }
    for (final o in options) {
      for (final n in o.keys) {
        _checkNode(network, n);
      }
    }
  }

  double _value(BatchNetwork network, int element, int k) {
    var v = 0.0;
    for (final MapEntry(:key, :value) in options[k].entries) {
      v += value * network.get(element, key, field);
    }
    return v;
  }

  Float64List _state(Float64List? state) {
    if (state == null) {
      throw StateError(
        'SoftmaxReadout with a choice kernel needs its carried state '
        '(stateSize values per element)',
      );
    }
    return state;
  }

  /// `uₖ = β vₖ + bₖ + κ [k = previous] + φ cₖ`.
  double _utility(
    BatchNetwork network,
    int element,
    int k,
    double previous,
    Float64List params,
    int offset,
    Float64List? state,
    int stateAt,
  ) {
    var u = params[offset] * _value(network, element, k);
    if (bias != null) u += params[offset + _iBias + k];
    if (stickiness != null && previous == k) u += params[offset + 1];
    if (_kernel) {
      u += params[offset + _iKernel] * _state(state)[stateAt + k];
    }
    return u;
  }

  /// `log Σⱼ exp(uⱼ)`, computed stably.
  double _logNormaliser(
    BatchNetwork network,
    int element,
    double previous,
    Float64List params,
    int offset,
    Float64List? state,
    int stateAt,
  ) {
    var top = double.negativeInfinity;
    for (var k = 0; k < options.length; k++) {
      final z = _utility(
        network,
        element,
        k,
        previous,
        params,
        offset,
        state,
        stateAt,
      );
      if (z > top) top = z;
    }
    if (!top.isFinite) return top;
    var sum = 0.0;
    for (var k = 0; k < options.length; k++) {
      sum += math.exp(
        _utility(
              network,
              element,
              k,
              previous,
              params,
              offset,
              state,
              stateAt,
            ) -
            top,
      );
    }
    return top + math.log(sum);
  }

  /// `−log((1 − ε) q + ε / K)` from `log q`, or `−log q` without a lapse.
  double _nll(double logq, Float64List params, int offset) {
    if (lapse == null) return -logq;
    final eps = params[offset + _iLapse];
    final a = math.log(1 - eps) + logq;
    final b = math.log(eps / options.length);
    final m = math.max(a, b);
    if (m == double.negativeInfinity) return double.infinity;
    return -(m + math.log(math.exp(a - m) + math.exp(b - m)));
  }

  int _choice(double response) {
    final r = response.toInt();
    if (r < 0 || r >= options.length || r != response) {
      throw ArgumentError.value(
        response,
        'response',
        'Not an option index (0 .. ${options.length - 1})',
      );
    }
    return r;
  }

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
    final r = _choice(response);
    final z = _logNormaliser(
      network,
      element,
      previous,
      params,
      offset,
      state,
      stateAt,
    );
    final ur = _utility(
      network,
      element,
      r,
      previous,
      params,
      offset,
      state,
      stateAt,
    );
    return _nanToInf(_nll(ur - z, params, offset));
  }

  @override
  bool get supportsGradients => BatchNetwork.tangentFields.contains(field);

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
    final beta = params[offset];
    final r = _choice(response);
    final z = _logNormaliser(
      network,
      element,
      previous,
      params,
      offset,
      state,
      stateAt,
    );
    final logq =
        _utility(
          network,
          element,
          r,
          previous,
          params,
          offset,
          state,
          stateAt,
        ) -
        z;
    final s = _nll(logq, params, offset);
    // S = −log p_r, p_r = (1 − ε) q_r + ε / K, q = softmax(u):
    // ∂S/∂uₖ = w (qₖ − [k = r]) with w = (1 − ε) q_r / p_r (1 without a
    // lapse), ∂S/∂ε = (q_r − 1/K) / p_r; uₖ = β vₖ + bₖ + κ [k = prev] +
    // φ cₖ, each vₖ linear in the beliefs, cₖ depending on α alone.
    var w = 1.0;
    if (lapse != null) {
      final eps = params[offset + _iLapse];
      final qr = math.exp(logq), pInv = math.exp(s);
      w = (1 - eps) * qr * pInv;
      partials.addParameter(_iLapse, (qr - 1 / options.length) * pInv);
    }
    final n = options.length;
    final phi = _kernel ? params[offset + _iKernel] : 0.0;
    var dBeta = 0.0, dPhi = 0.0, dRate = 0.0;
    for (var k = 0; k < n; k++) {
      final v = _value(network, element, k);
      final q = math.exp(
        _utility(
              network,
              element,
              k,
              previous,
              params,
              offset,
              state,
              stateAt,
            ) -
            z,
      );
      final du = w * (q - (k == r ? 1.0 : 0.0));
      dBeta += du * v;
      final dv = beta * du;
      for (final MapEntry(:key, :value) in options[k].entries) {
        partials.add(key, field, dv * value);
      }
      if (bias != null) partials.addParameter(_iBias + k, du);
      if (stickiness != null && previous == k) partials.addParameter(1, du);
      if (_kernel) {
        dPhi += du * state![stateAt + k];
        dRate += du * phi * state[stateAt + n + k];
      }
    }
    partials.addParameter(0, dBeta);
    if (_kernel) {
      partials.addParameter(_iKernel, dPhi);
      partials.addParameter(_iKernel + 1, dRate);
    }
    return _nanToInf(s);
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
    final z = _logNormaliser(
      network,
      element,
      previous,
      params,
      offset,
      state,
      stateAt,
    );
    if (!z.isFinite) return double.nan;
    final eps = lapse == null ? 0.0 : params[offset + _iLapse];
    final n = options.length;
    var u = rng.nextDouble();
    for (var k = 0; k < n - 1; k++) {
      final q = math.exp(
        _utility(
              network,
              element,
              k,
              previous,
              params,
              offset,
              state,
              stateAt,
            ) -
            z,
      );
      u -= (1 - eps) * q + eps / n;
      if (u < 0) return k.toDouble();
    }
    return (n - 1).toDouble();
  }
}

/// The surprise of the observations themselves under the network's
/// predictions, from a parameter-free `ResponseFunction` such as
/// `TotalGaussianSurprise`, `FirstLevelBinarySurprise` or
/// `CategoricalSurprise`. Fitting with it
/// tunes the perceptual model to the data; subjects need no responses.
final class ObservationSurprise extends ResponseModel {
  const ObservationSurprise(this.function);

  final ResponseFunction function;

  @override
  bool get usesResponses => false;

  /// The built-in parameter-free functions are written by name; others
  /// need a name in `FitCodec.responseFunctions`.
  @override
  Map<String, Object?> toJson() => {
    'type': 'ObservationSurprise',
    'function': switch (function) {
      FirstLevelGaussianSurprise() => 'FirstLevelGaussianSurprise',
      TotalGaussianSurprise() => 'TotalGaussianSurprise',
      FirstLevelBinarySurprise() => 'FirstLevelBinarySurprise',
      CategoricalSurprise() => 'CategoricalSurprise',
      DpSurprise() => 'DpSurprise',
      _ => throw UnsupportedError(
        'Cannot serialise ${function.runtimeType}: name it in '
        'FitCodec.responseFunctions',
      ),
    },
  };

  @override
  void prepare(Network network) => function.prepare(network, 0);

  @override
  double surprise(
    BatchNetwork network,
    int element,
    int t,
    double response,
    Float64List params,
    int offset,
  ) => function.batchSurprise(network, t, element);

  @override
  bool get supportsGradients => function.supportsGradients;

  @override
  double surprisePartials(
    BatchNetwork network,
    int element,
    int t,
    double response,
    Float64List params,
    int offset,
    ResponsePartials partials,
  ) => function.batchSurprisePartials(network, t, element, partials._inner!);
}

/// Where a [ResponseModel] reports the partial derivatives of one trial's
/// surprise ([ResponseModel.surprisePartials]).
final class ResponsePartials {
  ResponsePartials._(this._paramTangents);

  final Float64List _paramTangents;
  SurprisePartials? _inner;
  int _offset = 0, _k = 0;

  /// The surprise changes by [derivative] per unit of [field] on [node].
  void add(int node, NodeField field, double derivative) =>
      _inner!.add(node, field, derivative);

  /// The surprise changes by [derivative] per unit of the model's parameter
  /// [i] (`params[offset + i]`).
  void addParameter(int i, double derivative) =>
      _inner!.addTangents(derivative, _paramTangents, (_offset + i) * _k);
}

/// Binds [partials] to one element's [SurprisePartials] (for `HgfModel`).
ResponsePartials bindResponsePartials(
  ResponsePartials? partials,
  Float64List paramTangents,
  SurprisePartials inner,
  int offset,
  int k,
) {
  final p = partials ?? ResponsePartials._(paramTangents);
  p
    .._inner = inner
    .._offset = offset
    .._k = k;
  return p;
}
