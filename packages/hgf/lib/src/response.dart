import 'dart:typed_data';

import 'math.dart';
import 'network.dart';
import 'types.dart';

/// A response model: the surprise (negative log-likelihood) of trial `t`
/// given the network's beliefs. Summed over trials, this is the objective
/// for fitting (`Network.surprise`).
///
/// [surprise] is called right after trial `t`'s belief update, so on every
/// node the `expected*` fields still hold the predictions for that trial and
/// `mean`/`precision` hold the posteriors (on input nodes, `mean` is the
/// observation). This is the same state pyhgf's response functions read from
/// `node_trajectories` at index `t`.
///
/// The built-in models port `pyhgf.response` ([FirstLevelGaussianSurprise],
/// [TotalGaussianSurprise], [FirstLevelBinarySurprise], [BinarySoftmax],
/// [BinarySoftmaxInverseTemperature]) and add a Gaussian response model
/// ([GaussianResponse]). Write your own by extending this class, or from a
/// closure with [ResponseFunction.fn]:
///
/// ```dart
/// // Decisions follow the belief at node 0 through a fixed lapse rate.
/// final lapse = ResponseFunction.fn((net, t) {
///   final p = 0.9 * net.get(0, NodeField.expectedMean) + 0.05;
///   return binarySurprise(responses[t], p);
/// });
/// ```
///
/// Like pyhgf, the built-ins return `+∞` where the surprise is NaN (the model
/// could not fit), so a diverged trajectory makes the total infinite.
abstract class ResponseFunction {
  const ResponseFunction();

  /// A response function computed by [surprise]. Pass [batch] as well to
  /// use it with a [BatchNetwork] (see [batchSurprise]).
  const factory ResponseFunction.fn(
    double Function(Network network, int t) surprise, {
    double Function(BatchNetwork network, int t, int element)? batch,
  }) = _FnResponse;

  /// Called once before a run of [nTime] trials, to check that the network
  /// and the response data fit this model. Throws [ArgumentError] if not.
  void prepare(Network network, int nTime) {}

  /// Surprise of trial [t], read from [network]'s current state.
  double surprise(Network network, int t);

  /// Surprise of trial [t] for one [element] of a [BatchNetwork]: the same
  /// as [surprise], reading the element's state with
  /// `network.get(element, ...)`. [prepare] is called with the network the
  /// batch was made from.
  ///
  /// The built-in models implement it. A custom model that doesn't throws
  /// [UnsupportedError] when used with a batch.
  double batchSurprise(BatchNetwork network, int t, int element) =>
      throw UnsupportedError('$runtimeType has no batched form');

  /// Whether [batchSurprisePartials] is implemented, so the response can
  /// be differentiated (`BatchNetwork.surprise` with `gradient:`).
  bool get supportsGradients => false;

  /// [batchSurprise], also reporting the surprise's partial derivatives
  /// with respect to the beliefs it reads: for each, call
  /// `partials.add(node, field, ∂surprise/∂belief)`. The batch combines
  /// them with the beliefs' tangents into the gradient.
  ///
  /// ```dart
  /// // surprise = gaussianSurprise(r, μ̂₀, π): ∂/∂μ̂₀ = −π (r − μ̂₀)
  /// partials.add(0, NodeField.expectedMean, -precision * (r - m));
  /// ```
  ///
  /// The built-in models implement it (where the surprise is `+∞` the
  /// partials are meaningless). Others throw [UnsupportedError].
  double batchSurprisePartials(
    BatchNetwork network,
    int t,
    int element,
    SurprisePartials partials,
  ) => throw UnsupportedError('$runtimeType has no gradient form');
}

final class _FnResponse extends ResponseFunction {
  const _FnResponse(this._fn, {this.batch});

  final double Function(Network network, int t) _fn;
  final double Function(BatchNetwork network, int t, int element)? batch;

  @override
  double surprise(Network network, int t) => _fn(network, t);

  @override
  double batchSurprise(BatchNetwork network, int t, int element) {
    final b = batch;
    if (b == null) {
      throw UnsupportedError(
        'This ResponseFunction.fn has no batch form; pass `batch:`',
      );
    }
    return b(network, t, element);
  }
}

@pragma('vm:prefer-inline')
@pragma('wasm:prefer-inline')
double _nanToInf(double x) => x.isNaN ? double.infinity : x;

void _checkNode(Network network, int node, String role) {
  if (node < 0 || node >= network.nNodes) {
    throw ArgumentError.value(node, role, 'No such node');
  }
}

void _checkResponses(Float64List responses, int nTime) {
  if (responses.length < nTime) {
    throw ArgumentError(
      '${responses.length} response(s) for $nTime trial(s)',
      'responses',
    );
  }
}

/// Gaussian surprise of the observation at node 0 under the prediction of
/// node 1 (its value parent in the usual continuous HGF):
/// `gaussianSurprise(μ₀, μ̂₁, π̂₁)`.
///
/// Port of `pyhgf.response.first_level_gaussian_surprise`, including its
/// choice of node 1's expected precision (the observation noise of node 0 is
/// not added).
final class FirstLevelGaussianSurprise extends ResponseFunction {
  const FirstLevelGaussianSurprise();

  @override
  void prepare(Network network, int nTime) {
    if (network.nNodes < 2 ||
        network.kindOf(0) != NodeKind.continuousState ||
        network.kindOf(1) != NodeKind.continuousState) {
      throw ArgumentError(
        'FirstLevelGaussianSurprise needs continuous nodes 0 and 1',
      );
    }
  }

  @override
  double surprise(Network network, int t) => _nanToInf(
    gaussianSurprise(
      network.get(0, NodeField.mean),
      network.get(1, NodeField.expectedMean),
      network.get(1, NodeField.expectedPrecision),
    ),
  );

  @override
  double batchSurprise(BatchNetwork network, int t, int element) => _nanToInf(
    gaussianSurprise(
      network.get(element, 0, NodeField.mean),
      network.get(element, 1, NodeField.expectedMean),
      network.get(element, 1, NodeField.expectedPrecision),
    ),
  );

  @override
  bool get supportsGradients => true;

  @override
  double batchSurprisePartials(
    BatchNetwork network,
    int t,
    int element,
    SurprisePartials partials,
  ) {
    final x = network.get(element, 0, NodeField.mean);
    final m = network.get(element, 1, NodeField.expectedMean);
    final p = network.get(element, 1, NodeField.expectedPrecision);
    final d = x - m;
    partials
      ..add(0, NodeField.mean, p * d)
      ..add(1, NodeField.expectedMean, -p * d)
      ..add(1, NodeField.expectedPrecision, 0.5 * (d * d - 1.0 / p));
    return _nanToInf(gaussianSurprise(x, m, p));
  }
}

/// Sum of the Gaussian surprise `gaussianSurprise(μᵢ, μ̂ᵢ, π̂ᵢ)` over every
/// continuous node.
///
/// Port of `pyhgf.response.total_gaussian_surprise`. pyhgf returns `+∞` for
/// every trial when node 1's mean is NaN anywhere in the series; this
/// trial-wise version returns `+∞` from the trial where it (or the sum)
/// becomes NaN. Totals agree.
final class TotalGaussianSurprise extends ResponseFunction {
  const TotalGaussianSurprise();

  @override
  void prepare(Network network, int nTime) {
    if (network.nNodes < 2) {
      throw ArgumentError('TotalGaussianSurprise needs at least two nodes');
    }
  }

  @override
  double surprise(Network network, int t) {
    if (network.get(1, NodeField.mean).isNaN) return double.infinity;
    var s = 0.0;
    for (var i = 0, n = network.nNodes; i < n; i++) {
      if (network.kindOf(i) != NodeKind.continuousState) continue;
      s += gaussianSurprise(
        network.get(i, NodeField.mean),
        network.get(i, NodeField.expectedMean),
        network.get(i, NodeField.expectedPrecision),
      );
    }
    return _nanToInf(s);
  }

  @override
  double batchSurprise(BatchNetwork network, int t, int element) {
    if (network.get(element, 1, NodeField.mean).isNaN) return double.infinity;
    var s = 0.0;
    for (var i = 0, n = network.nNodes; i < n; i++) {
      if (network.kindOf(i) != NodeKind.continuousState) continue;
      s += gaussianSurprise(
        network.get(element, i, NodeField.mean),
        network.get(element, i, NodeField.expectedMean),
        network.get(element, i, NodeField.expectedPrecision),
      );
    }
    return _nanToInf(s);
  }

  @override
  bool get supportsGradients => true;

  @override
  double batchSurprisePartials(
    BatchNetwork network,
    int t,
    int element,
    SurprisePartials partials,
  ) {
    if (network.get(element, 1, NodeField.mean).isNaN) return double.infinity;
    var s = 0.0;
    for (var i = 0, n = network.nNodes; i < n; i++) {
      if (network.kindOf(i) != NodeKind.continuousState) continue;
      final x = network.get(element, i, NodeField.mean);
      final m = network.get(element, i, NodeField.expectedMean);
      final p = network.get(element, i, NodeField.expectedPrecision);
      s += gaussianSurprise(x, m, p);
      final d = x - m;
      partials
        ..add(i, NodeField.mean, p * d)
        ..add(i, NodeField.expectedMean, -p * d)
        ..add(i, NodeField.expectedPrecision, 0.5 * (d * d - 1.0 / p));
    }
    return _nanToInf(s);
  }
}

/// Sum of the binary surprise of each input node's observation under its
/// predicted probability: `Σ binarySurprise(uᵢ, μ̂ᵢ)` over the input nodes.
///
/// Port of `pyhgf.response.first_level_binary_surprise`.
final class FirstLevelBinarySurprise extends ResponseFunction {
  const FirstLevelBinarySurprise();

  @override
  void prepare(Network network, int nTime) {
    for (final i in network.inputIdxs) {
      if (network.kindOf(i) != NodeKind.binaryState) {
        throw ArgumentError(
          'FirstLevelBinarySurprise needs binary input nodes; node $i is '
          '${network.kindOf(i).pyhgfName}',
        );
      }
    }
  }

  @override
  double surprise(Network network, int t) {
    var s = 0.0;
    for (final i in network.inputIdxs) {
      s += binarySurprise(
        network.get(i, NodeField.mean),
        network.get(i, NodeField.expectedMean),
      );
    }
    return _nanToInf(s);
  }

  @override
  double batchSurprise(BatchNetwork network, int t, int element) {
    var s = 0.0;
    for (final i in network.inputIdxs) {
      s += binarySurprise(
        network.get(element, i, NodeField.mean),
        network.get(element, i, NodeField.expectedMean),
      );
    }
    return _nanToInf(s);
  }

  @override
  bool get supportsGradients => true;

  @override
  double batchSurprisePartials(
    BatchNetwork network,
    int t,
    int element,
    SurprisePartials partials,
  ) {
    var s = 0.0;
    for (final i in network.inputIdxs) {
      final x = network.get(element, i, NodeField.mean);
      final m = network.get(element, i, NodeField.expectedMean);
      s += binarySurprise(x, m);
      partials.add(i, NodeField.expectedMean, binarySurpriseDerivative(x, m));
    }
    return _nanToInf(s);
  }
}

/// The surprise of the observations at the categorical input nodes: for
/// each, `Σ binarySurprise(xₖ, μ̂ₖ)` over its categories, the indicator
/// `xₖ` under the probability its binary node predicted (the node's
/// [NodeField.surprise], as pyhgf computes it in the categorical posterior).
/// Other input nodes are ignored.
final class CategoricalSurprise extends ResponseFunction {
  const CategoricalSurprise();

  @override
  void prepare(Network network, int nTime) {
    if (!network.inputIdxs.any(
      (i) => network.kindOf(i) == NodeKind.categoricalState,
    )) {
      throw ArgumentError('CategoricalSurprise needs a categorical input');
    }
  }

  @override
  double surprise(Network network, int t) {
    var s = 0.0;
    for (final i in network.inputIdxs) {
      if (network.kindOf(i) != NodeKind.categoricalState) continue;
      s += network.get(i, NodeField.surprise);
    }
    return _nanToInf(s);
  }

  @override
  double batchSurprise(BatchNetwork network, int t, int element) {
    var s = 0.0;
    for (final i in network.inputIdxs) {
      if (network.kindOf(i) != NodeKind.categoricalState) continue;
      s += network.get(element, i, NodeField.surprise);
    }
    return _nanToInf(s);
  }

  @override
  bool get supportsGradients => true;

  @override
  double batchSurprisePartials(
    BatchNetwork network,
    int t,
    int element,
    SurprisePartials partials,
  ) {
    var s = 0.0;
    for (final i in network.inputIdxs) {
      if (network.kindOf(i) != NodeKind.categoricalState) continue;
      // The node's surprise, summed as its posterior step sums it.
      var node = 0.0;
      for (var k = 0, n = network.valueParentCount(i); k < n; k++) {
        final b = network.valueParent(i, k);
        final x = network.get(element, b, NodeField.mean);
        final m = network.get(element, b, NodeField.expectedMean);
        node += binarySurprise(x, m);
        partials.add(b, NodeField.expectedMean, binarySurpriseDerivative(x, m));
      }
      s += node;
    }
    return _nanToInf(s);
  }
}

/// Binary decisions (`1` or `0`) made with the probability the agent
/// predicts at [node]: `binarySurprise(responses[t], μ̂_node)`.
///
/// Port of `pyhgf.response.binary_softmax` (node 0). A NaN response marks a
/// missed trial and contributes no surprise.
final class BinarySoftmax extends ResponseFunction {
  BinarySoftmax(this.responses, {this.node = 0});

  /// One decision per trial.
  final Float64List responses;

  /// The node whose `expectedMean` is the predicted probability.
  final int node;

  @override
  void prepare(Network network, int nTime) {
    _checkNode(network, node, 'node');
    _checkResponses(responses, nTime);
  }

  @override
  double surprise(Network network, int t) {
    final r = responses[t];
    if (r.isNaN) return 0.0;
    return _nanToInf(
      binarySurprise(r, network.get(node, NodeField.expectedMean)),
    );
  }

  @override
  double batchSurprise(BatchNetwork network, int t, int element) {
    final r = responses[t];
    if (r.isNaN) return 0.0;
    return _nanToInf(
      binarySurprise(r, network.get(element, node, NodeField.expectedMean)),
    );
  }

  @override
  bool get supportsGradients => true;

  @override
  double batchSurprisePartials(
    BatchNetwork network,
    int t,
    int element,
    SurprisePartials partials,
  ) {
    final r = responses[t];
    if (r.isNaN) return 0.0;
    final m = network.get(element, node, NodeField.expectedMean);
    partials.add(node, NodeField.expectedMean, binarySurpriseDerivative(r, m));
    return _nanToInf(binarySurprise(r, m));
  }
}

/// Binary decisions made with the predicted probability at [node], sharpened
/// by an [inverseTemperature] β: `p = μ̂^β / (μ̂^β + (1 − μ̂)^β)`, then
/// `binarySurprise(responses[t], p)`.
///
/// Port of `pyhgf.response.binary_softmax_inverse_temperature` (node 0). A
/// NaN response marks a missed trial and contributes no surprise.
final class BinarySoftmaxInverseTemperature extends ResponseFunction {
  BinarySoftmaxInverseTemperature(
    this.responses, {
    required this.inverseTemperature,
    this.node = 0,
  });

  /// One decision per trial.
  final Float64List responses;

  /// β: 1 leaves the belief unchanged, larger values make decisions more
  /// deterministic.
  final double inverseTemperature;

  /// The node whose `expectedMean` is the predicted probability.
  final int node;

  @override
  void prepare(Network network, int nTime) {
    _checkNode(network, node, 'node');
    _checkResponses(responses, nTime);
  }

  @override
  double surprise(Network network, int t) {
    final r = responses[t];
    if (r.isNaN) return 0.0;
    final p = sigmoidInverseTemperature(
      network.get(node, NodeField.expectedMean),
      inverseTemperature,
    );
    return _nanToInf(binarySurprise(r, p));
  }

  @override
  double batchSurprise(BatchNetwork network, int t, int element) {
    final r = responses[t];
    if (r.isNaN) return 0.0;
    final p = sigmoidInverseTemperature(
      network.get(element, node, NodeField.expectedMean),
      inverseTemperature,
    );
    return _nanToInf(binarySurprise(r, p));
  }

  @override
  bool get supportsGradients => true;

  @override
  double batchSurprisePartials(
    BatchNetwork network,
    int t,
    int element,
    SurprisePartials partials,
  ) {
    final r = responses[t];
    if (r.isNaN) return 0.0;
    final m = network.get(element, node, NodeField.expectedMean);
    final beta = inverseTemperature;
    final p = sigmoidInverseTemperature(m, beta);
    // p = σ(β logit m): ∂p/∂m = p (1 − p) β / (m (1 − m)).
    partials.add(
      node,
      NodeField.expectedMean,
      binarySurpriseDerivative(r, p) * p * (1.0 - p) * beta / (m * (1.0 - m)),
    );
    return _nanToInf(binarySurprise(r, p));
  }
}

/// Continuous responses read out from a belief with Gaussian noise:
/// `responses[t] ~ N(belief, 1 / precision)`, where the belief is [field] of
/// [node] (by default the prediction `expectedMean` at node 0, i.e. what the
/// agent expects to observe). The surprise is
/// `gaussianSurprise(responses[t], belief, precision)`.
///
/// pyhgf has no built-in equivalent; it is the usual response model for
/// continuous reports (estimates, predictions, reaction-time-like readouts).
/// A NaN response marks a missed trial and contributes no surprise.
final class GaussianResponse extends ResponseFunction {
  GaussianResponse(
    this.responses, {
    this.precision = 1.0,
    this.node = 0,
    this.field = NodeField.expectedMean,
  }) {
    if (!(precision > 0.0)) {
      throw ArgumentError.value(precision, 'precision', 'Must be positive');
    }
  }

  /// One response per trial.
  final Float64List responses;

  /// Precision (inverse variance) of the response noise.
  final double precision;

  /// The node whose [field] is the belief the responses report.
  final int node;

  /// The reported belief (`expectedMean` for a prediction, `mean` for a
  /// posterior estimate).
  final NodeField field;

  @override
  void prepare(Network network, int nTime) {
    _checkNode(network, node, 'node');
    _checkResponses(responses, nTime);
  }

  @override
  double surprise(Network network, int t) {
    final r = responses[t];
    if (r.isNaN) return 0.0;
    return _nanToInf(gaussianSurprise(r, network.get(node, field), precision));
  }

  @override
  double batchSurprise(BatchNetwork network, int t, int element) {
    final r = responses[t];
    if (r.isNaN) return 0.0;
    return _nanToInf(
      gaussianSurprise(r, network.get(element, node, field), precision),
    );
  }

  @override
  bool get supportsGradients => BatchNetwork.tangentFields.contains(field);

  @override
  double batchSurprisePartials(
    BatchNetwork network,
    int t,
    int element,
    SurprisePartials partials,
  ) {
    final r = responses[t];
    if (r.isNaN) return 0.0;
    final m = network.get(element, node, field);
    partials.add(node, field, -precision * (r - m));
    return _nanToInf(gaussianSurprise(r, m, precision));
  }
}
