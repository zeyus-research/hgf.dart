import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';

/// What kind of observations a preset's input node takes.
enum InputKind {
  /// 0 or 1.
  binary,

  /// Any number.
  continuous,

  /// A category index, `0 .. nCategories − 1` (one-hot coded for the
  /// network).
  categorical,
}

/// How a preset's observations are simulated.
enum DataModel {
  /// Binary outcomes whose likely value swaps every block.
  reversal,

  /// A random walk whose step size switches between calm and volatile.
  randomWalk,

  /// Categories, one of which is likely; which one changes every block.
  categoricalReversal,

  /// Draws from a few Gaussian sources, switching source every block.
  mixture,

  /// The Iowa Gambling Task's four decks ([IowaGamblingTask.outcomes]).
  igt,

  /// Two arms, each with a chance of a win and of a loss; which arm is the
  /// good one swaps every block.
  bandit,
}

/// A task where an agent chooses among options and sees only what its
/// choice reveals (bandits, the Iowa Gambling Task). The network learns
/// each option's outcomes; a softmax over the options' predicted values
/// ([SoftmaxReadout]) picks one each trial, and only the inputs in
/// `reveals[choice]` are observed.
class ChoiceTask {
  const ChoiceTask({
    required this.options,
    required this.weights,
    required this.reveals,
    required this.valueNodes,
    required this.revealedNames,
    this.scale = 1,
    this.unit = '',
    this.recordedUnit = '',
    this.beta = 4,
  });

  /// The options' names, e.g. the decks.
  final List<String> options;

  /// Each option's value as weights on input nodes' predictions (see
  /// [SoftmaxReadout.options]); also what a choice pays: the same weights
  /// on the outcomes it reveals.
  final List<Map<int, double>> weights;

  /// The inputs choosing each option reveals (by input position).
  final List<List<int>> reveals;

  /// The nodes whose tonic volatility sets how fast the options' values
  /// drift (the fitted ω, shared or one each).
  final List<int> valueNodes;

  /// What a recorded choice reveals, one name per input in `reveals[k]`
  /// (the CSV columns besides the choice), e.g. `['payoff']`.
  final List<String> revealedNames;

  /// Recorded outcomes are divided by this (e.g. dollars to units of
  /// \$100).
  final double scale;

  /// The unit of a payoff, for labels (e.g. `\$100`).
  final String unit;

  /// The unit of recorded outcomes ([scale] of them make one [unit]).
  final String recordedUnit;

  /// The agent's default inverse temperature.
  final double beta;

  SoftmaxReadout readout(double inverseTemperature) =>
      SoftmaxReadout(options: weights, inverseTemperature: inverseTemperature);

  /// What choosing [k] pays on trial [t] of [table] (`nTime × nInputs`),
  /// with [column] mapping a node to its input column.
  double payoff(
    Float64List table,
    int nInputs,
    int t,
    int k,
    int Function(int) column,
  ) {
    var v = 0.0;
    for (final MapEntry(:key, :value) in weights[k].entries) {
      v += value * table[t * nInputs + column(key)];
    }
    return v;
  }

  /// Inputs and observed flags for recorded data: per trial a [choices]
  /// index (NaN: none) and, in [revealed], the values of the inputs that
  /// choice reveals (one series per [revealedNames], in `reveals[choice]`
  /// order, in recorded units: divided by [scale]).
  ({Float64List inputs, Float64List observed}) data(
    Float64List choices,
    List<Float64List> revealed,
    int nInputs,
  ) {
    if (revealed.length != revealedNames.length) {
      throw ArgumentError('One series per ${revealedNames.join(', ')}');
    }
    final n = choices.length;
    final inputs = Float64List(n * nInputs),
        observed = Float64List(n * nInputs);
    for (var t = 0; t < n; t++) {
      final c = choices[t];
      if (c.isNaN) continue;
      final k = c.toInt();
      if (k != c || k < 0 || k >= options.length) {
        throw FormatException(
          'Choice $c on trial ${t + 1} is not an option index '
          '(0 to ${options.length - 1})',
        );
      }
      for (final (j, i) in reveals[k].indexed) {
        final v = revealed[j][t];
        if (!v.isFinite) continue;
        inputs[t * nInputs + i] = v / scale;
        observed[t * nInputs + i] = 1;
      }
    }
    return (inputs: inputs, observed: observed);
  }
}

/// An editable node description; the network is rebuilt from these when its
/// structure or update scheme changes, and edited in place otherwise.
class NodeSpec {
  NodeSpec(
    this.kind, {
    this.valueChildren = const [],
    this.volatilityChildren = const [],
    List<double>? valueStrengths,
    List<double>? volatilityStrengths,
    Map<NodeField, double>? params,
  }) : valueStrengths =
           valueStrengths ?? List.filled(valueChildren.length, 1.0),
       volatilityStrengths =
           volatilityStrengths ?? List.filled(volatilityChildren.length, 1.0),
       params = {...?params};

  final NodeKind kind;
  final List<int> valueChildren;
  final List<int> volatilityChildren;
  final List<double> valueStrengths;
  final List<double> volatilityStrengths;
  final Map<NodeField, double> params;
}

/// A ready-made network with matching simulated data.
///
/// The network is either a list of [nodes] (continuous and binary), or
/// added by [addTo] (for node kinds that imply others: categorical,
/// ef-state, dp-state).
class Preset {
  const Preset({
    required this.id,
    required this.name,
    required this.summary,
    required this.input,
    required this.data,
    this.nodes,
    this.addTo,
    this.defaultTrials = 400,
    this.maxTrials = 10000,
    this.volatilityUpdates,
    this.task,
  }) : assert((nodes == null) != (addTo == null));

  final String id;
  final String name;

  /// One or two sentences on what the network models.
  final String summary;
  final InputKind input;
  final DataModel data;
  final List<NodeSpec> Function()? nodes;
  final void Function(Network net)? addTo;
  final int defaultTrials;

  /// The most trials the data slider offers (slow node kinds take fewer).
  final int maxTrials;

  /// The volatility update to switch to with this preset (null: keep).
  final VolatilityUpdates? volatilityUpdates;

  /// The choice task an agent plays on this network (null: the network
  /// just observes [data]).
  final ChoiceTask? task;

  Network build({
    VolatilityUpdates volatilityUpdates = VolatilityUpdates.unbounded,
    bool meanField = false,
  }) {
    final net = Network(
      volatilityUpdates: volatilityUpdates,
      meanFieldUpdates: meanField,
    );
    if (addTo case final add?) {
      add(net);
      return net;
    }
    for (final s in nodes!()) {
      net.addNodes(
        kind: s.kind,
        valueChildren: s.valueChildren.isEmpty ? null : s.valueChildren,
        valueChildrenStrengths: s.valueChildren.isEmpty
            ? null
            : s.valueStrengths,
        volatilityChildren: s.volatilityChildren.isEmpty
            ? null
            : s.volatilityChildren,
        volatilityChildrenStrengths: s.volatilityChildren.isEmpty
            ? null
            : s.volatilityStrengths,
        parameters: s.params,
      );
    }
    return net;
  }
}

final presets = <Preset>[
  Preset(
    id: 'binary3',
    name: 'Binary 3-level HGF',
    summary:
        'Learning the probability of a binary outcome (x₁, in logit space) '
        'and how fast it changes (x₂, its volatility), as in reversal '
        'learning tasks.',
    input: InputKind.binary,
    data: DataModel.reversal,
    nodes: () => [
      NodeSpec(NodeKind.binaryState),
      NodeSpec(
        NodeKind.continuousState,
        valueChildren: [0],
        params: {NodeField.tonicVolatility: -4.0},
      ),
      // ω₃ = −6: under the default unbounded update, ω₃ = −4 runs away
      // within a few hundred trials (pyhgf does the same: upstream P17).
      NodeSpec(
        NodeKind.continuousState,
        volatilityChildren: [1],
        params: {NodeField.tonicVolatility: -6.0, NodeField.mean: 1.0},
      ),
    ],
  ),
  Preset(
    id: 'continuous2',
    name: 'Continuous 2-level HGF',
    summary:
        'Tracking a noisy continuous signal: x₁ follows the hidden value, '
        'its tonic volatility ω sets how fast it is expected to move.',
    input: InputKind.continuous,
    data: DataModel.randomWalk,
    nodes: () => [
      NodeSpec(NodeKind.continuousState, params: {NodeField.precision: 100.0}),
      NodeSpec(
        NodeKind.continuousState,
        valueChildren: [0],
        params: {NodeField.tonicVolatility: -4.0},
      ),
    ],
  ),
  Preset(
    id: 'continuous3',
    name: 'Continuous 3-level HGF',
    summary:
        'As the 2-level HGF, plus x₂, which learns how volatile the signal '
        'is, so x₁ speeds up its learning when the signal starts to jump.',
    input: InputKind.continuous,
    data: DataModel.randomWalk,
    nodes: () => [
      NodeSpec(NodeKind.continuousState, params: {NodeField.precision: 100.0}),
      NodeSpec(
        NodeKind.continuousState,
        valueChildren: [0],
        params: {NodeField.tonicVolatility: -4.0},
      ),
      NodeSpec(
        NodeKind.continuousState,
        volatilityChildren: [1],
        params: {NodeField.tonicVolatility: -4.0},
      ),
    ],
  ),
  Preset(
    id: 'categorical3',
    name: 'Categorical HGF (3 outcomes)',
    summary:
        'Learning the probabilities of three outcomes. Each category gets '
        'a binary HGF (x₁–x₃ and their logits x₄–x₆), sharing one '
        'volatility parent (x₇); the categorical node x₀ turns them into a '
        'Dirichlet belief.',
    input: InputKind.categorical,
    data: DataModel.categoricalReversal,
    // ω₇ = −8: with pyhgf's −4 (or −6) the shared volatility parent runs
    // away under the default unbounded update within a few hundred (or a
    // few thousand) trials, as in the binary 3-level HGF (upstream P17).
    addTo: (net) => net.addCategoricalState(
      nCategories: 3,
      volatilityTonicVolatility: -8.0,
    ),
  ),
  Preset(
    id: 'dp',
    name: 'Dirichlet process (clustering)',
    summary:
        'Sorting observations into clusters, opening a new one when no '
        'open cluster explains an observation well (a Chinese restaurant '
        'process). x₁–x₄ are the clusters; α sets how readily new ones '
        'open. Slow: 20 000 proposals per trial, as in pyhgf.',
    input: InputKind.continuous,
    data: DataModel.mixture,
    addTo: (net) => net
      ..addDpState(batchSize: 4, alpha: 0.1)
      ..addEfState(nNodes: 4, valueChildren: [0], nus: 3.0),
    defaultTrials: 150,
    maxTrials: 1000,
  ),
  Preset(
    id: 'ef',
    name: 'Exponential family (mean and variance)',
    summary:
        'Tracking the mean and variance of a signal by updating its '
        'sufficient statistics (x, x²) at a fixed rate 1/(1 + ν): no '
        'hierarchy, so it cannot speed up when the signal turns volatile.',
    input: InputKind.continuous,
    data: DataModel.randomWalk,
    addTo: (net) => net.addEfState(nus: 10.0),
  ),
  Preset(
    id: 'igt',
    name: 'Iowa Gambling Task (choices)',
    summary:
        'An agent picks one of four decks per trial and sees only that '
        'card\'s payoff. A two-level HGF per deck (x₀–x₃, the payoffs; '
        'x₄–x₇, their drifting means) learns the decks; a softmax with '
        'inverse temperature β over the predicted payoffs chooses. A and B '
        'pay more per card but lose in the long run.',
    input: InputKind.continuous,
    data: DataModel.igt,
    // IowaGamblingTask.network(): payoffs observed with precision 1, means
    // drifting with ω = −2 from a prior N(0, 1).
    nodes: () => [
      for (var d = 0; d < 4; d++)
        NodeSpec(NodeKind.continuousState, params: {NodeField.precision: 1.0}),
      for (var d = 0; d < 4; d++)
        NodeSpec(
          NodeKind.continuousState,
          valueChildren: [d],
          params: {NodeField.tonicVolatility: -2.0, NodeField.precision: 1.0},
        ),
    ],
    defaultTrials: 100,
    maxTrials: 2000,
    task: ChoiceTask(
      options: IowaGamblingTask.decks,
      weights: [
        for (var d = 0; d < 4; d++) {d: 1.0},
      ],
      reveals: IowaGamblingTask.reveals,
      valueNodes: IowaGamblingTask.valueNodes,
      revealedNames: ['payoff'],
      scale: 100,
      unit: r'$100',
      recordedUnit: 'dollars',
    ),
  ),
  Preset(
    id: 'bandit',
    name: 'Two-armed bandit (choices)',
    summary:
        'As in pyhgf\'s example: each arm may win and may lose on a trial, '
        'and an agent sees both outcomes of the arm it pulls only. A '
        'binary HGF each learns P(win) and P(loss) per arm (x₀–x₃; '
        'logits x₄–x₇); an arm is worth P(win) − P(loss), and a softmax '
        'with inverse temperature β chooses. The good arm swaps every few '
        'trials.',
    input: InputKind.binary,
    data: DataModel.bandit,
    nodes: () => [
      for (var i = 0; i < 4; i++) NodeSpec(NodeKind.binaryState),
      for (var i = 0; i < 4; i++)
        NodeSpec(
          NodeKind.continuousState,
          valueChildren: [i],
          params: {NodeField.tonicVolatility: -2.5},
        ),
    ],
    defaultTrials: 200,
    maxTrials: 5000,
    task: ChoiceTask(
      options: ['A', 'B'],
      weights: [
        {0: 1.0, 1: -1.0},
        {2: 1.0, 3: -1.0},
      ],
      reveals: [
        [0, 1],
        [2, 3],
      ],
      valueNodes: [4, 5, 6, 7],
      revealedNames: ['win', 'loss'],
    ),
  ),
];

Preset presetById(String id) =>
    presets.firstWhere((p) => p.id == id, orElse: () => presets.first);

/// A node parameter the playground exposes as a slider.
class ParamDef {
  const ParamDef(
    this.field,
    this.symbol,
    this.name,
    this.min,
    this.max, {
    this.log = false,
    required this.help,
  });

  final NodeField field;
  final String symbol;
  final String name;

  /// Slider range (of log₁₀ of the value when [log]).
  final double min, max;
  final bool log;
  final String help;
}

/// The sliders for a node of [kind]; [input] when it has no children.
List<ParamDef> paramsFor(NodeKind kind, {required bool input}) {
  if (kind == NodeKind.dpState) {
    return const [
      ParamDef(
        NodeField.alpha,
        'α',
        'concentration',
        -3,
        2,
        log: true,
        help:
            'How readily a new cluster opens. Higher: more, smaller '
            'clusters; once every cluster is open, further openings are '
            'dropped (as in pyhgf).',
      ),
      ParamDef(
        NodeField.sensoryPrecision,
        'π_s',
        'sensory precision',
        -1,
        2,
        log: true,
        help:
            'How tightly a new cluster is fitted to the observation that '
            'opens it.',
      ),
    ];
  }
  if (kind == NodeKind.efState) {
    return const [
      ParamDef(
        NodeField.nus,
        'ν',
        'pseudo-count',
        0,
        3,
        log: true,
        help:
            'How many past observations the estimate is worth: each new '
            'one moves it by 1/(1 + ν). Higher: slower, smoother.',
      ),
    ];
  }
  if (kind != NodeKind.continuousState) return const [];
  if (input) {
    return const [
      ParamDef(
        NodeField.precision,
        'π_u',
        'sensory precision',
        -1,
        6,
        log: true,
        help:
            'How reliable the observations are taken to be. Higher: the '
            'beliefs follow each observation more closely.',
      ),
    ];
  }
  return const [
    ParamDef(
      NodeField.tonicVolatility,
      'ω',
      'tonic volatility',
      -10,
      2,
      help:
          'The log of how much the node is expected to change per trial. '
          'Higher: faster learning, noisier beliefs.',
    ),
    ParamDef(
      NodeField.mean,
      'μ₀',
      'initial mean',
      -5,
      5,
      help: 'The belief before the first observation.',
    ),
    ParamDef(
      NodeField.precision,
      'π₀',
      'initial precision',
      -2,
      4,
      log: true,
      help:
          'How sure the initial belief is. Low: the first observations '
          'move it a lot.',
    ),
    ParamDef(
      NodeField.tonicDrift,
      'ρ',
      'tonic drift',
      -0.5,
      0.5,
      help: 'A constant trend added to the predicted mean each trial.',
    ),
    ParamDef(
      NodeField.autoconnectionStrength,
      'λ',
      'autoconnection',
      0,
      1,
      help:
          'How much of the previous mean carries over to the prediction '
          '(1: a random walk; below 1: drift back towards 0).',
    ),
  ];
}
