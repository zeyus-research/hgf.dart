import 'dart:math' as math;

import 'package:hgf_deep/hgf_deep.dart';

/// The belief shown in each node of the network view.
enum NodeField {
  mean('μ', 'posterior mean', LayerField.mean),
  expectedMean('μ̂', 'prediction (expected mean)', LayerField.expectedMean),
  precision('π', 'posterior precision', LayerField.precision),
  expectedPrecision('π̂', 'predicted precision', LayerField.expectedPrecision),
  valuePredictionError(
    'δ',
    'value prediction error',
    LayerField.valuePredictionError,
  );

  const NodeField(this.symbol, this.description, this.field);

  final String symbol;
  final String description;
  final LayerField field;

  /// Precisions are positive and shown on a log scale; the rest diverge
  /// around 0.
  bool get isPrecision =>
      this == NodeField.precision || this == NodeField.expectedPrecision;
}

/// How the network view draws a network: every node and edge, or one strip
/// of node values per layer.
enum ViewMode {
  /// Every node, every edge.
  full,

  /// Every node, the strongest [DeepViewModel.maxEdges] edges.
  topEdges,

  /// One strip per layer: too many nodes to draw one by one.
  summary,
}

/// What the network view shows for one traced phase, independent of
/// painting: layers in display order (input first), node values, and the
/// message carried along each edge in this phase.
final class DeepViewModel {
  DeepViewModel({
    required this.network,
    required this.states,
    required this.weights,
    this.step,
  });

  /// Above this many nodes, layers are drawn as strips ([ViewMode.summary]).
  static const maxNodes = 160;

  /// Above this many nodes in one layer, too.
  static const maxLayerNodes = 64;

  /// Above this many edges, only the strongest are drawn.
  static const maxEdges = 3000;

  final DeepNetwork network;

  /// Beliefs of every network layer (index 0 = output) after the phase.
  final List<LayerState> states;

  /// Weights in force after the phase (index l: into layer l from below,
  /// `null` for the output layer).
  final List<Matrix?> weights;

  /// The phase shown (`null`: the network at rest).
  final TraceStep? step;

  int get nLayers => network.nLayers;

  /// The network layer drawn in display column [d] (0 = input, on the left).
  int layerAt(int d) => nLayers - 1 - d;

  /// The display column of network layer [l].
  int columnOf(int l) => nLayers - 1 - l;

  int get totalNodes => network.layerSizes.fold(0, (a, b) => a + b);

  int get totalEdges {
    var n = 0;
    for (var l = 1; l < nLayers; l++) {
      final w = weights[l];
      if (w == null) continue;
      n += network.configs[l].fullyConnected ? w.rows * w.cols : w.rows;
    }
    return n;
  }

  ViewMode get mode {
    if (totalNodes > maxNodes ||
        network.layerSizes.any((s) => s > maxLayerNodes)) {
      return ViewMode.summary;
    }
    return totalEdges > maxEdges ? ViewMode.topEdges : ViewMode.full;
  }

  /// Why [mode] was chosen, for the page.
  String get modeReason => switch (mode) {
    ViewMode.full => '',
    ViewMode.topEdges =>
      '$totalEdges connections: showing the $maxEdges strongest.',
    ViewMode.summary =>
      '$totalNodes nodes: each layer is drawn as one strip of node values '
          '(max $maxNodes nodes, $maxLayerNodes per layer, to draw them one '
          'by one).',
  };

  /// The value of [field] at node [i] of network layer [l].
  double value(NodeField field, int l, int i) =>
      states[l].field(field.field)![i];

  /// μ_vol of node [i] of layer [l] (`null` without a volatility level).
  double? meanVol(int l, int i) => states[l].meanVol?[i];

  double? volatilityError(int l, int i) =>
      states[l].volatilityPredictionError?[i];

  /// The layer the phase acts on (`null`: network-wide or at rest).
  int? get activeLayer {
    final s = step;
    if (s == null || s.layer < 0) return null;
    return s.layer;
  }

  /// The message along the edge from node [j] of layer [l] (the parent) to
  /// node [i] of layer `l − 1` (the child) in this phase, or `null` when the
  /// edge carries nothing in it:
  ///
  /// * a prediction of layer `l − 1`: `w_ij · g(μ̂_j)`, the parent's
  ///   contribution to the child's predicted mean;
  /// * the posterior of layer `l`: `w_ij · gain_i · δ_i`, the child's
  ///   precision-weighted prediction error sent up to the parent;
  /// * the weight step: the descent gradient of `w_ij` (the step moves the
  ///   weight against it).
  ///
  /// [j] = parent size is the bias (a constant parent with value 1).
  double? message(int l, int i, int j) {
    final s = step;
    if (s == null) return null;
    final w = weights[l]!;
    final cfg = network.configs[l];
    final wij = cfg.fullyConnected ? w.at(i, j) : (i == j ? w.at(i, i) : 0.0);
    switch (s.phase) {
      case TracePhase.predict when s.layer == l - 1:
        final parent = states[l];
        final g = j == parent.nodes
            ? 1.0
            : cfg.coupling.f(parent.expectedMean[j]);
        return wij * g;
      case TracePhase.posterior when s.layer == l:
        return wij * _gainTimesError(l - 1, i);
      case TracePhase.weightUpdate:
        final g = s.gradients?[l];
        if (g == null) return null;
        return cfg.fullyConnected ? g.at(i, j) : (i == j ? g.at(i, i) : 0.0);
      default:
        return null;
    }
  }

  /// `gain · δ` of node [i] of layer [l] as its parents read it (the
  /// relaxed gain `π̄·π/(π̄ + π − π̂)`, or π̂ under mean-field updates).
  double _gainTimesError(int l, int i) {
    final c = states[l];
    final double gain;
    if (network.settings.meanFieldUpdates) {
      gain = c.expectedPrecision[i];
    } else {
      final piY = c.precision[i] - c.expectedPrecision[i];
      final cep = c.conditionalExpectedPrecision[i];
      gain = cep * c.precision[i] / (cep + piY);
    }
    return gain * c.valuePredictionError[i];
  }

  /// The largest |message| over the active edges (for colour scaling).
  double maxMessage() {
    var m = 0.0;
    for (var l = 1; l < nLayers; l++) {
      final w = weights[l]!;
      for (var i = 0; i < w.rows; i++) {
        for (var j = 0; j < w.cols; j++) {
          final v = message(l, i, j);
          if (v != null && v.isFinite) m = math.max(m, v.abs());
        }
      }
    }
    return m;
  }

  /// The largest |weight| (for edge widths at rest).
  double maxWeight() {
    var m = 0.0;
    for (var l = 1; l < nLayers; l++) {
      for (final v in weights[l]!.data) {
        if (v.isFinite) m = math.max(m, v.abs());
      }
    }
    return m;
  }
}

/// A short name for network layer [l] of [net] (input, hidden k, output).
String layerName(DeepNetwork net, int l) {
  if (l == net.nLayers - 1) return 'input';
  if (l == 0) return 'output';
  return 'hidden ${net.nLayers - 1 - l}';
}

/// What phase [step] does, in a sentence, with [net]'s layer names.
String phaseTitle(DeepNetwork net, TraceStep step) {
  final name = step.layer >= 0 ? layerName(net, step.layer) : '';
  return switch (step.phase) {
    TracePhase.clampInput => 'Clamp the input x on the input layer',
    TracePhase.predict => 'Predict the $name layer from the layer before it',
    TracePhase.observe => 'Observe the target y on the output layer',
    TracePhase.leafError => 'Output prediction errors',
    TracePhase.posterior =>
      'Update the $name layer from the errors of the layer after it',
    TracePhase.layerError => 'Prediction errors of the $name layer',
    TracePhase.weightUpdate => 'Learn: one optimiser step on every weight',
    TracePhase.carry => 'Carry the beliefs to the next sample',
  };
}

/// A paragraph on what happens in [step]'s phase, in general terms.
String phaseExplanation(TraceStep step) => switch (step.phase) {
  TracePhase.clampInput =>
    'Each input node is set to its component of x: these nodes are '
        'observed, not inferred.',
  TracePhase.predict =>
    'Top-down prediction. Every node predicts its value as '
        'μ̂ = Σ w·g(μ̂_parent) + b over the layer before it (g is that '
        "layer's activation). Its predicted precision π̂ is how sure it is: "
        'the precision it carried from the last sample, made vaguer by the '
        'drift Ω between samples: 1/π̂ = 1/π + Ω. With a volatility level, '
        'Ω = e^(μ_vol + ω + 1/(2π̂_vol)), so a node that believes it is '
        'volatile makes a vague prediction and the evidence will move it '
        'more.',
  TracePhase.observe =>
    'The output layer is clamped to the target y: the network now sees '
        'what it should have predicted.',
  TracePhase.leafError =>
    'Value prediction errors δ = y − μ̂ at the output. Binary nodes divide '
        'by their predicted precision, categorical nodes use the plain '
        'difference. These errors are the only signal that flows back.',
  TracePhase.posterior =>
    'Bottom-up update. Each node receives the errors of the layer after it, '
        'weighted by their precision (the gain) and by w·g′: '
        'π = π̂ + Σ w²·γ·g′² − g″·Σ w·gain·δ (the evidence, plus a curvature '
        'term), then μ = μ̂ + g′·Σ w·gain·δ / π. A node with a vague '
        'prediction (small π̂) moves further; one with strong evidence '
        'becomes more precise. (With the fixes on, π never drops below '
        'π̂ + Σ w²·γ·g′².)',
  TracePhase.layerError =>
    "The layer's own errors for the layers before it: the value error "
        'δ = μ − μ̂, and with a volatility level the volatility error '
        'δ_vol = π̂/π + π̂·δ² − 1. δ_vol > 0 means the node moved more than '
        'its precision expected, so its volatility level rises (μ_vol up, '
        'the next prediction vaguer); δ_vol < 0 lowers it.',
  TracePhase.weightUpdate =>
    'Learning. Each weight w_ij moves to make child i better predicted by '
        'parent j: its gradient is −(μ_i − μ̂_i)·π_i·g(μ_j) (precision '
        'weighted; without π for binary outputs), and the optimiser takes '
        'one step against it.',
  TracePhase.carry =>
    'The posterior precisions and volatility levels are carried to the '
        'next sample: what the network believes about each node\'s '
        'reliability persists, its values do not.',
};
