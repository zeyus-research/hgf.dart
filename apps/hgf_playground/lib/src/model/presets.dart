import 'package:hgf/hgf.dart';

/// What kind of observations a preset's input node takes.
enum InputKind { binary, continuous }

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
class Preset {
  const Preset({
    required this.id,
    required this.name,
    required this.summary,
    required this.input,
    required this.nodes,
    this.defaultTrials = 400,
  });

  final String id;
  final String name;

  /// One or two sentences on what the network models.
  final String summary;
  final InputKind input;
  final List<NodeSpec> Function() nodes;
  final int defaultTrials;

  Network build({
    VolatilityUpdates volatilityUpdates = VolatilityUpdates.unbounded,
    bool meanField = false,
    required List<NodeSpec> specs,
  }) {
    final net = Network(
      volatilityUpdates: volatilityUpdates,
      meanFieldUpdates: meanField,
    );
    for (final s in specs) {
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
