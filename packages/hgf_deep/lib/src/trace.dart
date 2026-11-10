import 'matrix.dart';
import 'state.dart';

/// One phase of a single learning step, in the order a [DeepTrace] records
/// them.
enum TracePhase {
  /// The input `x` is clamped on the top layer (its mean and expected
  /// mean).
  clampInput,

  /// Layer `layer` is predicted from the layer above:
  /// `μ̂ = W · g(μ̂_parent) + b`, with its expected precision (and, with a
  /// volatility parent, the volatility level's prediction).
  predict,

  /// The observation `y` is clamped on the bottom layer's mean.
  observe,

  /// The bottom layer's value (and volatility) prediction errors.
  leafError,

  /// Layer `layer`'s posterior (value and volatility levels) from the
  /// precision-weighted errors of the layer below.
  posterior,

  /// Layer `layer`'s own prediction errors, sent on to the layer above.
  layerError,

  /// The weights take one optimiser step from the settled beliefs.
  weightUpdate,

  /// The carried beliefs (precision and volatility level) are set for the
  /// next sample.
  carry,
}

/// The network after one [TracePhase].
final class TraceStep {
  TraceStep._(
    this.phase,
    this.layer,
    this.states,
    this.weights,
    this.gradients,
  );

  /// The phase that just ran.
  final TracePhase phase;

  /// The layer it ran on (0 is the bottom/output layer; −1 when the phase
  /// is network-wide).
  final int layer;

  /// A copy of every layer's beliefs after the phase, bottom to top.
  final List<LayerState> states;

  /// The weights after the phase (`null` entry for the bottom layer). Only
  /// recorded on the first step and on [TracePhase.weightUpdate]; `null`
  /// otherwise, when they are those of the last step that has them.
  final List<Matrix?>? weights;

  /// The descent gradients of the step, shaped like [weights] (only on
  /// [TracePhase.weightUpdate]).
  final List<Matrix?>? gradients;
}

/// Every phase of one learning step of a deep network, from
/// [DeepNetwork.trace].
final class DeepTrace {
  DeepTrace._(this.steps, this.prediction, this.initialWeights);

  /// The phases in the order they ran.
  final List<TraceStep> steps;

  /// The output layer's prediction (made before it saw `y`).
  final List<double> prediction;

  /// The weights before the step (`null` entry for the bottom layer).
  final List<Matrix?> initialWeights;

  /// The weights in force after step [index].
  List<Matrix?> weightsAt(int index) {
    for (var i = index; i >= 0; i--) {
      final w = steps[i].weights;
      if (w != null) return w;
    }
    return initialWeights;
  }
}

/// Builds a [DeepTrace] (internal to the network).
final class TraceRecorder {
  TraceRecorder(List<Matrix?> weights) : _initial = _copyAll(weights);

  final List<Matrix?> _initial;
  final List<TraceStep> _steps = [];

  static List<Matrix?> _copyAll(List<Matrix?> ws) => [
    for (final w in ws) w?.copy(),
  ];

  void record(
    TracePhase phase,
    int layer,
    List<LayerState> states, {
    List<Matrix?>? weights,
    List<Matrix?>? gradients,
  }) {
    _steps.add(
      TraceStep._(
        phase,
        layer,
        [for (final s in states) s.copy()],
        weights == null ? null : _copyAll(weights),
        gradients,
      ),
    );
  }

  DeepTrace finish(List<double> prediction) =>
      DeepTrace._(List.unmodifiable(_steps), prediction, _initial);
}
