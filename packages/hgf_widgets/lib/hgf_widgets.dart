/// Flutter widgets for [hgf](https://pub.dev/packages/hgf) networks: belief
/// trajectory plots that re-draw thousands of trials per frame, and a
/// network graph view.
library;

export 'src/graph/layout.dart' show EdgeKind, GraphEdge, NetworkLayout;
export 'src/graph/network_graph.dart' show NetworkGraph, defaultNodeLabel;
export 'src/hgf_trajectory_view.dart' show HgfColors, HgfTrajectoryView;
export 'src/plot/layers.dart';
export 'src/plot/reduce.dart' show SeriesReducer, finiteRange, niceTicks;
export 'src/plot/trajectory_plot.dart'
    show PlotGeometry, PlotStyle, TrajectoryPlot, formatTick;
