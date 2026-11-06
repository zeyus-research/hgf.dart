import 'package:flutter/material.dart';
import 'package:hgf/hgf.dart';

import 'graph/layout.dart';
import 'graph/network_graph.dart';
import 'plot/layers.dart';
import 'plot/trajectory_plot.dart';

/// Series colours by role, the first three slots of a colour-vision-checked
/// categorical palette, stepped separately for light and dark surfaces.
@immutable
class HgfColors {
  const HgfColors({
    required this.belief,
    required this.observation,
    required this.surprise,
  });

  static const light = HgfColors(
    belief: Color(0xFF2A78D6),
    observation: Color(0xFFEB6834),
    surprise: Color(0xFF1BAF7A),
  );

  static const dark = HgfColors(
    belief: Color(0xFF3987E5),
    observation: Color(0xFFD95926),
    surprise: Color(0xFF199E70),
  );

  static HgfColors of(BuildContext context) =>
      Theme.of(context).brightness == Brightness.dark ? dark : light;

  /// Beliefs: the expected mean and its uncertainty band.
  final Color belief;

  /// Observations (input nodes).
  final Color observation;

  /// Surprise.
  final Color surprise;
}

/// One plot per node, top level first, then the total surprise: the
/// Flutter counterpart of pyhgf's `plot_trajectories`.
///
/// Each node shows its prediction (expected mean `μ̂`, with a band of
/// `±1/√π̂`); input nodes also show their observations, binary inputs the
/// predicted probability. Constant, ef-state, categorical and dp-state
/// nodes are not drawn (their binary or continuous parents are).
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
  /// state of simulated data).
  final Map<int, List<PlotLayer>> extraLayers;

  /// The nodes [HgfTrajectoryView] can draw.
  static bool plottable(NodeKind kind) =>
      kind == NodeKind.continuousState || kind == NodeKind.binaryState;

  @override
  State<HgfTrajectoryView> createState() => _HgfTrajectoryViewState();
}

class _HgfTrajectoryViewState extends State<HgfTrajectoryView> {
  final ValueNotifier<int?> _ownHover = ValueNotifier(null);

  @override
  void dispose() {
    _ownHover.dispose();
    super.dispose();
  }

  Network get network => widget.network;
  ValueNotifier<int?> get hover => widget.hover ?? _ownHover;

  @override
  Widget build(BuildContext context) {
    final layout = widget.layout;
    final nodes = widget.nodes;
    final trajectories = widget.trajectories;
    final showSurprise = widget.showSurprise;
    final label = widget.label;
    final colors = HgfColors.of(context);
    final l = layout ?? NetworkLayout.of(network);
    final shown = [
      for (final i in nodes ?? l.topDown)
        if (HgfTrajectoryView.plottable(network.kindOf(i))) i,
    ];
    final traj = trajectories;
    final panels = <Widget>[];
    for (final (k, i) in shown.indexed) {
      final last = k == shown.length - 1 && !showSurprise;
      panels.add(
        _panel(
          i,
          [
            ...?widget.extraLayers[i],
            ..._layersFor(i, l.level[i] == 0, traj, colors),
          ],
          '${label(i)}${l.level[i] == 0 ? ' (input)' : ''}',
          network.kindOf(i) == NodeKind.binaryState ? (-0.08, 1.08) : null,
          last,
        ),
      );
    }
    if (showSurprise && traj.kinds != null) {
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
}
