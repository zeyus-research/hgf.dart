## Unreleased

- Zoom and pan on the time axis: `PlotViewport`, a visible x window shared
  by linked `TrajectoryPlot`s (`viewport:`), as `hover` is. Ctrl/⌘ +
  scroll, trackpad pinch or a two-finger touch pinch zooms about the
  pointer (plain scroll too, unless the plot is in a scrollable); a mouse
  drag, a sideways scroll or Shift + scroll pans; double-click resets.
  The y range fits the visible trials. `HgfTrajectoryView` shares one
  between its panels (`viewport:`, its own if null). `PlotViewportBar`
  shows the window over the whole series (drag or tap to pan), zoom
  buttons, a reset and the visible trials. `TrajectoryPlot` is now a
  `StatefulWidget`; x ticks fall on whole trials only.

## 0.1.0-dev.1

- `TrajectoryPlot` with per-pixel min/max reduction, `HgfTrajectoryView`,
  `NetworkGraph` and `NetworkLayout`.
