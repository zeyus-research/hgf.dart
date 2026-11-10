import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:hgf_widgets/hgf_widgets.dart';

import '../model/playground.dart';

/// A choice task's plots: the choice probabilities with the choices made,
/// the options' values with what each choice paid, the cumulative payoff,
/// then the network's hidden nodes (the inputs are the options' outcomes,
/// shown through the values).
class ChoiceTaskView extends StatefulWidget {
  const ChoiceTaskView({
    super.key,
    required this.playground,
    this.hover,
    this.panelHeight,
  });

  final Playground playground;
  final ValueNotifier<int?>? hover;

  /// Fixed height per panel; by default the panels share the height.
  final double? panelHeight;

  /// The panels [ChoiceTaskView] draws for [pg].
  static int panelCount(Playground pg) =>
      3 +
      HgfTrajectoryView.panelCount(
        pg.network,
        layout: pg.layout,
        nodes: _hiddenNodes(pg),
        showSurprise: false,
      );

  static List<int> _hiddenNodes(Playground pg) {
    final inputs = pg.network.inputIdxs.toSet();
    return [
      for (final i in pg.layout.topDown)
        if (!inputs.contains(i)) i,
    ];
  }

  @override
  State<ChoiceTaskView> createState() => _ChoiceTaskViewState();
}

class _ChoiceTaskViewState extends State<ChoiceTaskView> {
  final ValueNotifier<int?> _ownHover = ValueNotifier(null);

  // Per option: where it was chosen (a row above the probabilities) and
  // what it paid; rebuilt when the choices change.
  Float64List? _for;
  List<Float64List> _rows = const [], _paid = const [];

  @override
  void dispose() {
    _ownHover.dispose();
    super.dispose();
  }

  void _derive(Playground pg) {
    final choices = pg.choices!, pay = pg.observations;
    if (identical(_for, pay)) return;
    _for = pay;
    final k = pg.task!.options.length, n = choices.length;
    _rows = [
      for (var j = 0; j < k; j++) Float64List(n)..fillRange(0, n, double.nan),
    ];
    _paid = [
      for (var j = 0; j < k; j++) Float64List(n)..fillRange(0, n, double.nan),
    ];
    for (var t = 0; t < n; t++) {
      final c = choices[t];
      if (c.isNaN) continue;
      final j = c.toInt();
      _rows[j][t] = 1.04 + 0.06 * j;
      _paid[j][t] = pay[t];
    }
  }

  @override
  Widget build(BuildContext context) {
    final pg = widget.playground;
    final task = pg.task!;
    final traj = pg.trajectories;
    final series = pg.choiceSeries();
    if (traj == null || series == null || pg.choices == null) {
      return Center(child: Text(pg.error ?? 'No data'));
    }
    _derive(pg);
    final colors = HgfColors.of(context);
    final hover = widget.hover ?? _ownHover;
    final k = task.options.length;
    final unit = task.unit.isEmpty ? '' : ' (${task.unit})';
    final hidden = pg.hidden;
    Widget plot(
      int key,
      List<PlotLayer> layers,
      String title, {
      (double, double)? y,
    }) => TrajectoryPlot(
      key: ValueKey(key),
      layers: layers,
      title: title,
      yRange: y,
      limit: pg.player?.t,
      revision: pg.revision,
      hover: hover,
      showXAxis: false,
    );
    final panels = <Widget>[
      plot(
        -10,
        [
          for (var j = 0; j < k; j++)
            LineLayer(
              series.probabilities[j],
              color: colors.category(j),
              label: 'P(${task.options[j]})',
            ),
          for (var j = 0; j < k; j++)
            PointLayer(
              _rows[j],
              color: colors.category(j),
              label: 'chose ${task.options[j]}',
              radius: 2.2,
              affectsRange: false,
              readoutGaps: false,
            ),
        ],
        'choice probabilities (β = ${_n(pg.agentBeta)}) and choices',
        y: (-0.04, 1.04 + 0.06 * k),
      ),
      plot(-11, [
        if (hidden != null &&
            hidden.length == k &&
            hidden.first.length == traj.nTime)
          for (var j = 0; j < k; j++)
            LineLayer(
              hidden[j],
              color: colors.category(j).withValues(alpha: 0.45),
              width: 1,
              label: 'true ${task.options[j]}',
            ),
        for (var j = 0; j < k; j++)
          LineLayer(
            series.values[j],
            color: colors.category(j),
            label: 'value ${task.options[j]}',
          ),
        for (var j = 0; j < k; j++)
          PointLayer(
            _paid[j],
            color: colors.category(j),
            label: 'paid ${task.options[j]}',
            radius: 1.8,
            affectsRange: false,
            readoutGaps: false,
          ),
      ], 'option values (predicted payoff) and payoffs$unit'),
      plot(-12, [
        LineLayer(series.cumulative, color: colors.belief, label: 'total'),
      ], 'cumulative payoff$unit'),
    ];
    final h = widget.panelHeight;
    Widget sized(Widget p, {int flex = 1}) => h == null
        ? Expanded(flex: flex, child: p)
        : SizedBox(height: h * flex, child: p);
    final rest = ChoiceTaskView.panelCount(pg) - 3;
    return Column(
      children: [
        for (final p in panels) sized(p),
        sized(
          flex: rest,
          HgfTrajectoryView(
            network: pg.network,
            trajectories: traj,
            layout: pg.layout,
            nodes: ChoiceTaskView._hiddenNodes(pg),
            revision: pg.revision,
            limit: pg.player?.t,
            hover: hover,
            showSurprise: false,
            panelHeight: h,
          ),
        ),
      ],
    );
  }
}

String _n(double v) => v >= 10 ? v.toStringAsFixed(1) : v.toStringAsFixed(2);
