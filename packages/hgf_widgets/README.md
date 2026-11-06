# hgf_widgets

Flutter widgets for [`hgf`](../hgf) networks.

- `TrajectoryPlot`: time-series layers (lines, points, `μ ± 1/√π` bands,
  reference lines) drawn with a `CustomPainter`, each reduced to at most two
  points per pixel column, so re-drawing 10 000 trials every frame is cheap.
  Linked hover cursor with a value readout.
- `HgfTrajectoryView`: one plot per node, top level first, plus the total
  surprise, like pyhgf's `plot_trajectories`.
- `NetworkGraph`: the network drawn in layers, inputs at the bottom; value
  coupling solid, volatility coupling dashed; tap to select a node.

```dart
final hgf = Network()
  ..addNodes(kind: NodeKind.binaryState)
  ..addNodes(valueChildren: [0], tonicVolatility: -4)
  ..addNodes(volatilityChildren: [1], tonicVolatility: -6, mean: 1);
Trajectories? traj = hgf.inputData(outcomes);

// In build():
HgfTrajectoryView(network: hgf, trajectories: traj!, revision: revision);
NetworkGraph(network: hgf, onNodeTap: (i) => setState(() => selected = i));

// Re-filter in place on a slider move, then bump `revision`:
hgf.setInitial(1, NodeField.tonicVolatility, omega);
traj = hgf.inputData(outcomes, into: traj);
```
