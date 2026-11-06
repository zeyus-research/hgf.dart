import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

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

  /// A response drawn from the model for [element] (for simulation).
  double sample(
    BatchNetwork network,
    int element,
    int t,
    Float64List params,
    int offset,
    math.Random rng,
  ) => throw UnsupportedError('$runtimeType cannot simulate responses');
}

final class _FnResponseModel extends ResponseModel {
  const _FnResponseModel({
    this.parameters = const {},
    required double Function(BatchNetwork, int, int, double, Float64List, int)
    surprise,
    double Function(BatchNetwork, int, int, Float64List, int, math.Random)?
    sample,
    this.usesResponses = true,
  }) : surpriseFn = surprise,
       sampleFn = sample;

  @override
  final Map<String, double> parameters;
  @override
  final bool usesResponses;
  final double Function(BatchNetwork, int, int, double, Float64List, int)
  surpriseFn;
  final double Function(BatchNetwork, int, int, Float64List, int, math.Random)?
  sampleFn;

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

/// The surprise of the observations themselves under the network's
/// predictions, from a parameter-free `ResponseFunction` such as
/// `TotalGaussianSurprise` or `FirstLevelBinarySurprise`. Fitting with it
/// tunes the perceptual model to the data; subjects need no responses.
final class ObservationSurprise extends ResponseModel {
  const ObservationSurprise(this.function);

  final ResponseFunction function;

  @override
  bool get usesResponses => false;

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
}
