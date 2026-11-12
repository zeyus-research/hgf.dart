import 'dart:math' as math;
import 'dart:typed_data';

import 'package:material_ui/material_ui.dart';
import 'package:hgf/hgf.dart';

import 'graph/layout.dart';
import 'graph/network_graph.dart';
import 'plot/layers.dart';
import 'plot/trajectory_plot.dart';

/// Series colours, from a colour-vision-checked categorical palette, stepped
/// separately for light and dark surfaces.
///
/// The roles (belief, observation, surprise) are its first three slots;
/// [categories] is the whole palette in its fixed order, for one series per
/// category or cluster. Category `k` always gets slot `k`; past the last
/// slot, series share [overflow] (never a generated hue).
@immutable
class HgfColors {
  const HgfColors({
    required this.belief,
    required this.observation,
    required this.surprise,
    required this.categories,
    required this.overflow,
  });

  static const light = HgfColors(
    belief: Color(0xFF2A78D6),
    observation: Color(0xFFEB6834),
    surprise: Color(0xFF1BAF7A),
    categories: [
      Color(0xFF2A78D6), // blue
      Color(0xFFEB6834), // orange
      Color(0xFF1BAF7A), // aqua
      Color(0xFFEDA100), // yellow
      Color(0xFFE87BA4), // magenta
      Color(0xFF008300), // green
      Color(0xFF6250D6), // violet
      Color(0xFFE34948), // red
    ],
    overflow: Color(0xFF8A8980),
  );

  static const dark = HgfColors(
    belief: Color(0xFF3987E5),
    observation: Color(0xFFD95926),
    surprise: Color(0xFF199E70),
    categories: [
      Color(0xFF3987E5),
      Color(0xFFD95926),
      Color(0xFF199E70),
      Color(0xFFC98500),
      Color(0xFFD55181),
      Color(0xFF008300),
      Color(0xFF9085E9),
      Color(0xFFE66767),
    ],
    overflow: Color(0xFF8A8980),
  );

  static HgfColors of(BuildContext context) =>
      Theme.of(context).brightness == Brightness.dark ? dark : light;

  /// Beliefs: the expected mean and its uncertainty band.
  final Color belief;

  /// Observations (input nodes).
  final Color observation;

  /// Surprise.
  final Color surprise;

  /// One colour per category or cluster, in a fixed order.
  final List<Color> categories;

  /// For categories past the end of [categories].
  final Color overflow;

  /// The colour of category (or cluster) [k].
  Color category(int k) => k < categories.length ? categories[k] : overflow;
}

/// What one panel of an [HgfTrajectoryView] shows.
sealed class _Plan {
  const _Plan(this.key);

  /// A node index, unique per panel (-1 is the surprise).
  final int key;
}

/// A continuous or binary node.
final class _NodePlan extends _Plan {
  const _NodePlan(super.key);
}

/// A categorical node with its binary value parents folded in: the
/// predicted probability of each category, and the observed one.
final class _CategoricalPlan extends _Plan {
  const _CategoricalPlan(super.key, this.binaries);
  final List<int> binaries;
}

/// The value parents of a categorical node's binary nodes (its logits), one
/// line each.
final class _LogitsPlan extends _Plan {
  const _LogitsPlan(super.key, this.nodes);
  final List<int> nodes;
}

/// A dp-state node with its clusters folded in.
final class _DpPlan extends _Plan {
  const _DpPlan(super.key);
}

/// An ef-state node that is not a cluster.
final class _EfPlan extends _Plan {
  const _EfPlan(super.key);
}

/// One plot per node, top level first, then the total surprise: the
/// Flutter counterpart of pyhgf's `plot_trajectories`.
///
/// Each node shows its prediction (expected mean `μ̂`, with a band of
/// `±1/√π̂`); input nodes also show their observations, binary inputs the
/// predicted probability. Constant nodes are not drawn. Vector-valued nodes
/// get a panel each, with the nodes they imply folded in:
///
/// - a categorical node shows each category's predicted probability (its
///   binary parents' `μ̂`) and the observed category; its binary parents
///   get no panels of their own, and their value parents (the logits) share
///   one;
/// - a dp-state node shows its observations, coloured by the cluster they
///   were assigned to, and each open cluster's predicted mean ± standard
///   deviation; the clusters get no panels of their own;
/// - an ef-state node shows its observations and its expected sufficient
///   statistics before each one (for one dimension, the mean ± standard
///   deviation implied by `ξ`).
///
/// The vector views read the recorded vector fields, which
/// `Network.inputData` records by default.
class HgfTrajectoryView extends StatefulWidget {
  const HgfTrajectoryView({
    super.key,
    required this.network,
    required this.trajectories,
    this.layout,
    this.nodes,
    this.revision = 0,
    this.limit,
    this.hover,
    this.showSurprise = true,
    this.showPosterior = false,
    this.label = defaultNodeLabel,
    this.panelHeight,
    this.extraLayers = const {},
  });

  final Network network;
  final Trajectories trajectories;

  /// Used for the node order (default: [NetworkLayout.of] the network).
  final NetworkLayout? layout;

  /// The nodes to plot, in order (default: all plottable, top down).
  final List<int>? nodes;

  /// Bump when [trajectories] was overwritten in place.
  final int revision;

  /// Draw only trials before [limit] (playback).
  final int? limit;

  /// A shared hover cursor (one is created if null).
  final ValueNotifier<int?>? hover;

  final bool showSurprise;

  /// Also draw the posterior mean `μ` of each node.
  final bool showPosterior;

  final String Function(int node) label;

  /// Fixed height per panel; by default the panels share the height.
  final double? panelHeight;

  /// More layers per node, drawn behind its own (e.g. the true hidden
  /// state of simulated data). A folded node's layers go to the panel it is
  /// folded into.
  final Map<int, List<PlotLayer>> extraLayers;

  /// Whether a node of [kind] can have a panel (others are folded into
  /// their child's or not drawn).
  static bool plottable(NodeKind kind) => kind != NodeKind.constantState;

  /// Whether [network] has a surprise to plot: per-node surprise is defined
  /// for continuous and binary nodes only (as in pyhgf).
  static bool hasSurprise(Network network) {
    for (var i = 0; i < network.nNodes; i++) {
      final k = network.kindOf(i);
      if (k == NodeKind.continuousState || k == NodeKind.binaryState) {
        return true;
      }
    }
    return false;
  }

  /// The number of panels the view draws for [network] (including the
  /// surprise panel when [showSurprise] and [hasSurprise]).
  static int panelCount(
    Network network, {
    NetworkLayout? layout,
    List<int>? nodes,
    bool showSurprise = true,
  }) =>
      _plan(
        network,
        nodes ?? (layout ?? NetworkLayout.of(network)).topDown,
      ).length +
      (showSurprise && hasSurprise(network) ? 1 : 0);

  static List<_Plan> _plan(Network network, List<int> order) {
    final folded = <int>{};
    final logitsOf = <int, List<int>>{}; // first logit node -> all of them
    final logitNodes = <int>{};
    for (var c = 0; c < network.nNodes; c++) {
      switch (network.kindOf(c)) {
        case NodeKind.categoricalState:
          final bins = network.valueParentsOf(c);
          folded.addAll(bins);
          final logits = [for (final b in bins) ...network.valueParentsOf(b)];
          if (logits.isNotEmpty) {
            final first = order.firstWhere(
              logits.contains,
              orElse: () => logits.first,
            );
            logitsOf[first] = logits;
            logitNodes.addAll(logits);
          }
        case NodeKind.dpState:
          folded.addAll(network.valueParentsOf(c));
        default:
      }
    }
    return [
      for (final i in order)
        if (logitsOf[i] case final logits?)
          _LogitsPlan(i, logits)
        else if (!folded.contains(i) && !logitNodes.contains(i))
          ?switch (network.kindOf(i)) {
            NodeKind.continuousState || NodeKind.binaryState => _NodePlan(i),
            NodeKind.categoricalState => _CategoricalPlan(
              i,
              network.valueParentsOf(i),
            ),
            NodeKind.dpState => _DpPlan(i),
            NodeKind.efState => _EfPlan(i),
            NodeKind.constantState => null,
          },
    ];
  }

  @override
  State<HgfTrajectoryView> createState() => _HgfTrajectoryViewState();
}

class _HgfTrajectoryViewState extends State<HgfTrajectoryView> {
  final ValueNotifier<int?> _ownHover = ValueNotifier(null);

  // Series derived from the trajectories (per-category observations,
  // cluster lines), rebuilt when the trajectories or their revision change.
  Trajectories? _derivedFor;
  int _derivedRevision = -1;
  final Map<int, List<PlotLayer>> _derived = {};
  HgfColors? _derivedColors;

  @override
  void dispose() {
    _ownHover.dispose();
    super.dispose();
  }

  Network get network => widget.network;
  ValueNotifier<int?> get hover => widget.hover ?? _ownHover;

  @override
  Widget build(BuildContext context) {
    final colors = HgfColors.of(context);
    final l = widget.layout ?? NetworkLayout.of(network);
    final traj = widget.trajectories;
    final plans = HgfTrajectoryView._plan(network, widget.nodes ?? l.topDown);
    if (!identical(_derivedFor, traj) ||
        _derivedRevision != widget.revision ||
        !identical(_derivedColors, colors)) {
      _derived.clear();
      _derivedFor = traj;
      _derivedRevision = widget.revision;
      _derivedColors = colors;
    }
    final label = widget.label;
    final showSurprise =
        widget.showSurprise &&
        traj.kinds != null &&
        HgfTrajectoryView.hasSurprise(network);
    final panels = <Widget>[];
    for (final (k, plan) in plans.indexed) {
      final last = k == plans.length - 1 && !showSurprise;
      final i = plan.key;
      final input = l.level[i] == 0;
      final extra = widget.extraLayers[i];
      final (
        List<PlotLayer> layers,
        String title,
        (double, double)? y,
      ) = switch (plan) {
        _NodePlan() => (
          _layersFor(i, input, traj, colors),
          '${label(i)}${input ? ' (input)' : ''}',
          network.kindOf(i) == NodeKind.binaryState ? (-0.08, 1.08) : null,
        ),
        _CategoricalPlan(:final binaries) => (
          _derived[i] ??= _categorical(binaries, traj, colors),
          '${label(i)} (categorical input)',
          (-0.04, 1.04 + 0.06 * binaries.length),
        ),
        _LogitsPlan(:final nodes) => (
          _logits(nodes, traj, colors),
          '${label(nodes.first)}–${label(nodes.last)} '
              '(category logits)',
          null,
        ),
        _DpPlan() => (
          _derived[i] ??= _dp(i, traj, colors),
          '${label(i)} (Dirichlet process input)',
          null,
        ),
        _EfPlan() => (
          _derived[i] ??= _ef(i, traj, colors),
          '${label(i)} (exponential family${input ? ' input' : ''})',
          null,
        ),
      };
      panels.add(_panel(i, [...?extra, ...layers], title, y, last));
    }
    if (showSurprise) {
      panels.add(
        _panel(
          -1,
          [
            LineLayer(
              traj.totalSurprise(),
              color: colors.surprise,
              label: 'surprise',
            ),
          ],
          'surprise',
          null,
          true,
        ),
      );
    }
    final h = widget.panelHeight;
    return Column(
      children: [
        for (final p in panels)
          h == null ? Expanded(child: p) : SizedBox(height: h, child: p),
      ],
    );
  }

  Widget _panel(
    int node,
    List<PlotLayer> layers,
    String title,
    (double, double)? yRange,
    bool xAxis,
  ) => TrajectoryPlot(
    key: ValueKey(node),
    layers: layers,
    title: title,
    yRange: yRange,
    limit: widget.limit,
    revision: widget.revision,
    hover: hover,
    showXAxis: xAxis,
  );

  List<PlotLayer> _layersFor(
    int i,
    bool input,
    Trajectories traj,
    HgfColors colors,
  ) {
    final kind = network.kindOf(i);
    final mu = traj.series(i, NodeField.mean);
    final muHat = traj.series(i, NodeField.expectedMean);
    if (kind == NodeKind.binaryState) {
      return [
        LineLayer(muHat, color: colors.belief, label: 'P(u=1)'),
        if (input) PointLayer(mu, color: colors.observation, label: 'u'),
        if (!input && widget.showPosterior)
          LineLayer(mu, color: colors.surprise, label: 'μ', width: 1),
      ];
    }
    final piHat = traj.series(i, NodeField.expectedPrecision);
    return [
      BandLayer.precision(muHat, piHat, color: colors.belief, label: '±σ̂'),
      LineLayer(muHat, color: colors.belief, label: 'μ̂'),
      if (input)
        PointLayer(mu, color: colors.observation, label: 'u', radius: 1.8),
      if (!input && widget.showPosterior)
        LineLayer(mu, color: colors.surprise, label: 'μ', width: 1),
    ];
  }

  /// Each category's predicted probability, and the observed category as a
  /// row of dots above the plot (row `k` for category `k`).
  List<PlotLayer> _categorical(
    List<int> binaries,
    Trajectories traj,
    HgfColors colors,
  ) {
    final n = traj.nTime;
    return [
      for (final (k, b) in binaries.indexed)
        LineLayer(
          traj.series(b, NodeField.expectedMean),
          color: colors.category(k),
          label: 'P($k)',
        ),
      for (final (k, b) in binaries.indexed)
        PointLayer(
          _where(n, (t) => traj.at(b, NodeField.mean, t) == 1, 1.04 + 0.06 * k),
          color: colors.category(k),
          label: 'u = $k',
          radius: 2.2,
          affectsRange: false,
          readoutGaps: false,
        ),
    ];
  }

  List<PlotLayer> _logits(
    List<int> nodes,
    Trajectories traj,
    HgfColors colors,
  ) => [
    for (final (k, i) in nodes.indexed)
      LineLayer(
        traj.series(i, NodeField.expectedMean),
        color: colors.category(k),
        label: 'μ̂ ($k)',
      ),
  ];

  /// Observations coloured by the cluster they went to (where `n` grew),
  /// and each cluster's predicted mean ± standard deviation from the trial
  /// after it opened.
  List<PlotLayer> _dp(int d, Trajectories traj, HgfColors colors) {
    final n = traj.nTime;
    final u = traj.series(d, NodeField.mean);
    final has = traj.vectorFields.toSet();
    if (!has.contains(VectorField.n)) {
      return [PointLayer(u, color: colors.observation, label: 'u')];
    }
    final k = traj.vectorLength(d, VectorField.n);
    final counts = traj.vectorSeries(d, VectorField.n);
    final assigned = Int32List(n)..fillRange(0, n, -1);
    for (var t = 0; t < n; t++) {
      for (var j = 0; j < k; j++) {
        final before = t == 0 ? 0.0 : counts[(t - 1) * k + j];
        if (counts[t * k + j] > before) {
          assigned[t] = j;
          break;
        }
      }
    }
    final layers = <PlotLayer>[];
    final lines = <PlotLayer>[];
    if (has.contains(VectorField.expectedMeans) &&
        has.contains(VectorField.activated)) {
      final m = traj.vectorSeries(d, VectorField.expectedMeans);
      final s = has.contains(VectorField.expectedSigmas)
          ? traj.vectorSeries(d, VectorField.expectedSigmas)
          : null;
      final active = traj.vectorSeries(d, VectorField.activated);
      for (var j = 0; j < k; j++) {
        // Is cluster j open before trial t?
        bool open(int t) => t > 0 && active[(t - 1) * k + j] == 1;
        // Never opened: nothing to draw. (Not just the last row: during
        // playback the rows past the current trial are NaN.)
        var ever = false;
        for (var t = 0; t < n && !ever; t++) {
          ever = active[t * k + j] == 1;
        }
        if (!ever) continue;
        final mean = _where(n, open, 0, (t) => m[t * k + j]);
        final c = colors.category(j);
        if (s != null) {
          layers.add(
            BandLayer.bounds(
              _where(n, open, 0, (t) => m[t * k + j] - s[t * k + j]),
              _where(n, open, 0, (t) => m[t * k + j] + s[t * k + j]),
              color: c,
            ),
          );
        }
        lines.add(
          LineLayer(mean, color: c, label: 'μ̂ ($j)', readoutGaps: false),
        );
      }
    }
    for (var j = 0; j < k; j++) {
      if (!assigned.contains(j)) continue;
      layers.add(
        PointLayer(
          _where(n, (t) => assigned[t] == j, 0, (t) => u[t]),
          color: colors.category(j),
          label: 'u → $j',
          radius: 2.2,
          readoutGaps: false,
        ),
      );
    }
    return [...layers, ...lines];
  }

  /// An ef-state node's observations and, before each one, the expected
  /// sufficient statistics (`ξ` of the previous trial): for one dimension,
  /// the mean `ξ₁` ± `√(ξ₂ − ξ₁²)`; for more, each dimension's mean.
  List<PlotLayer> _ef(int e, Trajectories traj, HgfColors colors) {
    final n = traj.nTime;
    final has = traj.vectorFields.toSet();
    final dim = has.contains(VectorField.mean)
        ? traj.vectorLength(e, VectorField.mean)
        : 1;
    final obs = dim > 1 ? traj.vectorSeries(e, VectorField.mean) : null;
    final u = traj.series(e, NodeField.mean);
    if (!has.contains(VectorField.xis)) {
      return [PointLayer(u, color: colors.observation, label: 'u')];
    }
    final w = traj.vectorLength(e, VectorField.xis);
    final xi = traj.vectorSeries(e, VectorField.xis);
    bool before(int t) => t > 0;
    if (dim == 1) {
      double mean(int t) => xi[(t - 1) * w];
      double sd(int t) {
        final m = mean(t);
        return math.sqrt(math.max(0, xi[(t - 1) * w + 1] - m * m));
      }

      return [
        BandLayer.bounds(
          _where(n, before, 0, (t) => mean(t) - sd(t)),
          _where(n, before, 0, (t) => mean(t) + sd(t)),
          color: colors.belief,
          label: '±σ̂',
        ),
        LineLayer(
          _where(n, before, 0, mean),
          color: colors.belief,
          label: 'E[x]',
        ),
        PointLayer(u, color: colors.observation, label: 'u', radius: 1.8),
      ];
    }
    return [
      for (var j = 0; j < dim; j++) ...[
        PointLayer(
          _where(n, (_) => true, 0, (t) => obs![t * dim + j]),
          color: colors.category(j),
          label: 'u$j',
          radius: 1.6,
        ),
        LineLayer(
          _where(n, before, 0, (t) => xi[(t - 1) * w + j]),
          color: colors.category(j),
          label: 'E[x$j]',
        ),
      ],
    ];
  }

  /// `value(t)` (or [at]) where [test] holds, NaN elsewhere.
  static Float64List _where(
    int n,
    bool Function(int t) test,
    double at, [
    double Function(int t)? value,
  ]) {
    final out = Float64List(n);
    for (var t = 0; t < n; t++) {
      out[t] = test(t) ? (value == null ? at : value(t)) : double.nan;
    }
    return out;
  }
}
