import 'package:hgf_deep/datasets.dart' as ds;
import 'package:hgf_deep/hgf_deep.dart';

/// A learning task for the deep network page: its data, and the input and
/// output layers it fixes.
enum DeepTask {
  xor('XOR', 'Four noisy corners; class 1 where the signs differ.'),
  moons('Two moons', 'Two interleaving half circles.'),
  circles('Circles', 'A small ring inside a large one.'),
  spirals('Spirals', 'Three interleaved spiral arms (3 classes).'),
  sine('Sine', 'Regression: y = sin(x) + noise.');

  const DeepTask(this.label, this.description);

  final String label;
  final String description;

  bool get isRegression => this == sine;

  int get inputs => isRegression ? 1 : 2;

  int get classes => switch (this) {
    spirals => 3,
    sine => 0,
    _ => 2,
  };

  /// Output layer size and kind: one binary node for two classes, a
  /// categorical layer for more, one volatile node for regression.
  (int, LayerKind) get output => switch (classes) {
    0 => (1, LayerKind.volatile),
    2 => (1, LayerKind.binary),
    final k => (k, LayerKind.categorical),
  };

  /// The task's data (seeded).
  ds.Dataset data({int seed = 0}) => switch (this) {
    xor => ds.xor(400, noise: 0.25, seed: seed),
    moons => ds.moons(500, noise: 0.12, seed: seed),
    circles => ds.circles(500, noise: 0.08, seed: seed),
    spirals => ds.spirals(600, noise: 0.1, seed: seed),
    sine => ds.sine(400, noise: 0.1, seed: seed),
  };

  /// The input range shown on the decision map (both axes for 2-D tasks).
  (double, double) get inputRange => isRegression ? (-3.3, 3.3) : (-1.8, 1.8);
}

/// The coupling functions offered for hidden layers.
const hiddenCouplings = [
  CouplingFn.tanh,
  CouplingFn.sigmoid,
  CouplingFn.relu,
  CouplingFn.leakyRelu,
  CouplingFn.gelu,
  CouplingFn.linear,
];

/// Layer sizes offered for hidden layers.
const hiddenSizes = [
  1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, //
];

/// One hidden layer: its size, activation and whether its nodes carry a
/// volatility level.
final class HiddenLayerSpec {
  const HiddenLayerSpec({
    this.size = 8,
    this.coupling = CouplingFn.tanh,
    this.volatility = false,
  });

  final int size;
  final CouplingFn coupling;

  /// Whether every node of the layer has an internal volatility level (a
  /// volatility parent) that learns how much the node drifts.
  final bool volatility;

  HiddenLayerSpec copyWith({
    int? size,
    CouplingFn? coupling,
    bool? volatility,
  }) => HiddenLayerSpec(
    size: size ?? this.size,
    coupling: coupling ?? this.coupling,
    volatility: volatility ?? this.volatility,
  );

  @override
  bool operator ==(Object other) =>
      other is HiddenLayerSpec &&
      other.size == size &&
      other.coupling == coupling &&
      other.volatility == volatility;

  @override
  int get hashCode => Object.hash(size, coupling, volatility);
}

/// The optimisers offered.
enum OptimiserKind { adam, sgd }

/// Everything that defines the network, apart from the task.
final class DeepSpec {
  const DeepSpec({
    this.hidden = const [HiddenLayerSpec(), HiddenLayerSpec()],
    this.volatilityUpdates = DeepVolatilityUpdate.standard,
    this.tonicVolatility,
    this.fixes = true,
    this.optimiser = OptimiserKind.adam,
    this.learningRate = 0.01,
    this.seed = 1,
  });

  /// Hidden layers, from the input side to the output side.
  final List<HiddenLayerSpec> hidden;
  final DeepVolatilityUpdate volatilityUpdates;

  /// The value-level tonic volatility ω of every volatile layer, or `null`
  /// for none (pyhgf's default: the drift is e^μ_vol alone).
  final double? tonicVolatility;

  /// Whether hgf_deep's corrections to pyhgf are on.
  final bool fixes;
  final OptimiserKind optimiser;
  final double learningRate;

  /// Seed of the initial weights.
  final int seed;

  DeepSpec copyWith({
    List<HiddenLayerSpec>? hidden,
    DeepVolatilityUpdate? volatilityUpdates,
    double? Function()? tonicVolatility,
    bool? fixes,
    OptimiserKind? optimiser,
    double? learningRate,
    int? seed,
  }) => DeepSpec(
    hidden: hidden ?? this.hidden,
    volatilityUpdates: volatilityUpdates ?? this.volatilityUpdates,
    tonicVolatility: tonicVolatility != null
        ? tonicVolatility()
        : this.tonicVolatility,
    fixes: fixes ?? this.fixes,
    optimiser: optimiser ?? this.optimiser,
    learningRate: learningRate ?? this.learningRate,
    seed: seed ?? this.seed,
  );

  /// Whether [other] builds weights of the same shapes (so learned weights
  /// carry over).
  bool sameShapes(DeepSpec other) =>
      other.hidden.length == hidden.length &&
      [
        for (var i = 0; i < hidden.length; i++)
          other.hidden[i].size == hidden[i].size,
      ].every((s) => s);

  /// Layer sizes from input to output, for [task].
  List<int> sizes(DeepTask task) => [
    task.inputs,
    for (final h in hidden) h.size,
    task.output.$1,
  ];

  int nodes(DeepTask task) => sizes(task).fold(0, (a, b) => a + b);

  /// Number of weights (biases included).
  int parameters(DeepTask task) {
    final s = sizes(task);
    var n = 0;
    for (var i = 0; i + 1 < s.length; i++) {
      n += (s[i] + 1) * s[i + 1];
    }
    return n;
  }

  /// The network for [task], with fresh beliefs and seeded Xavier weights.
  DeepNetwork build(DeepTask task) {
    final (outSize, outKind) = task.output;
    final net = DeepNetwork(
      volatilityUpdates: volatilityUpdates,
      tonicVolatility: tonicVolatility != null,
      fixes: fixes ? DeepFixes.recommended : DeepFixes.none,
    )..addLayer(outSize, kind: outKind);
    for (final h in hidden.reversed) {
      net.addLayer(
        h.size,
        coupling: h.coupling,
        volatilityParent: h.volatility,
        tonicVolatility: tonicVolatility,
      );
    }
    return net
      ..addLayer(task.inputs)
      ..weightInitialisation(const XavierInit(), seed: seed);
  }

  Optimiser makeOptimiser() => switch (optimiser) {
    OptimiserKind.adam => Adam(learningRate),
    OptimiserKind.sgd => Sgd(learningRate),
  };
}
