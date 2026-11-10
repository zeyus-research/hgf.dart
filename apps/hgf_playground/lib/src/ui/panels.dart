import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_widgets/hgf_widgets.dart';

import '../model/export.dart';
import '../model/fitting.dart';
import '../model/playground.dart';
import '../model/presets.dart';
import '../settings.dart';
import 'files.dart';
import 'home.dart' show nodeCaption;

/// A titled card in the side panel.
class SectionCard extends StatelessWidget {
  const SectionCard({
    super.key,
    required this.title,
    required this.children,
    this.action,
  });

  final String title;
  final List<Widget> children;
  final Widget? action;

  @override
  Widget build(BuildContext context) => Card(
    child: Padding(
      padding: const EdgeInsets.fromLTRB(14, 8, 10, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  title,
                  style: Theme.of(context).textTheme.titleSmall,
                ),
              ),
              ?action,
            ],
          ),
          const SizedBox(height: 4),
          ...children,
        ],
      ),
    ),
  );
}

/// A labelled slider: `symbol name ........ value`.
class LabeledSlider extends StatelessWidget {
  const LabeledSlider({
    super.key,
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.display,
    required this.onChanged,
    this.onChangeEnd,
    this.help,
    this.divisions,
  });

  final String label;
  final double value, min, max;
  final String display;
  final ValueChanged<double> onChanged;
  final ValueChanged<double>? onChangeEnd;
  final String? help;
  final int? divisions;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final text = Row(
      children: [
        Expanded(child: Text(label, style: theme.textTheme.bodyMedium)),
        Text(
          display,
          style: theme.textTheme.bodyMedium?.copyWith(
            fontFeatures: const [FontFeature.tabularFigures()],
          ),
        ),
      ],
    );
    return Padding(
      padding: const EdgeInsets.only(top: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          help == null ? text : Tooltip(message: help!, child: text),
          SliderTheme(
            data: SliderTheme.of(context).copyWith(
              overlayShape: const RoundSliderOverlayShape(overlayRadius: 14),
            ),
            child: SizedBox(
              height: 28,
              child: Slider(
                value: value.isFinite ? value.clamp(min, max) : min,
                min: min,
                max: max,
                divisions: divisions,
                onChanged: onChanged,
                onChangeEnd: onChangeEnd,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

String fmt(double v) {
  final a = v.abs();
  if (a != 0 && (a >= 1e4 || a < 1e-2)) return v.toStringAsExponential(1);
  return v.toStringAsFixed(a >= 100 ? 0 : 2);
}

// -----------------------------------------------------------------------------
// Network
// -----------------------------------------------------------------------------

class NetworkCard extends StatelessWidget {
  const NetworkCard({super.key, required this.playground, required this.hover});

  final Playground playground;
  final ValueNotifier<int?> hover;

  @override
  Widget build(BuildContext context) {
    final pg = playground;
    final theme = Theme.of(context);
    final hasVolatility = pg.layout.edges.any(
      (e) => e.kind == EdgeKind.volatility,
    );
    return SectionCard(
      title: pg.preset.name,
      children: [
        Text(pg.preset.summary, style: theme.textTheme.bodySmall),
        SizedBox(
          height: 70.0 * pg.layout.levels + 20,
          child: ValueListenableBuilder(
            valueListenable: hover,
            builder: (context, t, _) => NetworkGraph(
              network: pg.network,
              layout: pg.layout,
              selected: pg.selected,
              onNodeTap: pg.select,
              caption: (i) => nodeCaption(pg, i, t),
              revision: Object.hash(pg.revision, t),
            ),
          ),
        ),
        Text(
          'Tap a node to edit it. Solid arrows: value coupling; dashed: '
          'volatility coupling. Captions: the posterior mean at the hovered '
          'trial.',
          style: theme.textTheme.bodySmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
        const SizedBox(height: 10),
        Tooltip(
          message: hasVolatility
              ? 'How a node updates from its volatility children.'
              : 'This network has no volatility coupling.',
          child: SegmentedButton<VolatilityUpdates>(
            showSelectedIcon: false,
            segments: const [
              ButtonSegment(
                value: VolatilityUpdates.standard,
                label: Text('standard'),
              ),
              ButtonSegment(value: VolatilityUpdates.eHGF, label: Text('eHGF')),
              ButtonSegment(
                value: VolatilityUpdates.unbounded,
                label: Text('unbounded'),
              ),
            ],
            selected: {pg.volatilityUpdates},
            onSelectionChanged: hasVolatility
                ? (s) => pg.setVolatilityUpdates(s.single)
                : null,
          ),
        ),
      ],
    );
  }
}

// -----------------------------------------------------------------------------
// Node parameters
// -----------------------------------------------------------------------------

class NodeCard extends StatelessWidget {
  const NodeCard({super.key, required this.playground});

  final Playground playground;

  @override
  Widget build(BuildContext context) {
    final pg = playground;
    final theme = Theme.of(context);
    final i = pg.selected;
    if (i == null) {
      return SectionCard(
        title: 'Node parameters',
        children: [
          Text(
            'Select a node in the graph to change its parameters. The '
            'whole series is filtered again on every move.',
            style: theme.textTheme.bodySmall,
          ),
        ],
      );
    }
    final net = pg.network;
    final kind = net.kindOf(i);
    final input = pg.layout.level[i] == 0;
    final defs = paramsFor(kind, input: input);
    // Couplings between continuous and binary nodes; the edges into
    // categorical and dp-state nodes are fixed by those nodes.
    bool scalar(int j) =>
        net.kindOf(j) == NodeKind.continuousState ||
        net.kindOf(j) == NodeKind.binaryState;
    final out = pg.layout.edges
        .where((e) => e.parent == i && scalar(e.parent) && scalar(e.child))
        .toList();
    final about = _about(net, i);
    return SectionCard(
      title:
          '${defaultNodeLabel(i)} · ${_kindNames[kind]}'
          '${input ? ' input' : ''}',
      action: IconButton(
        tooltip: 'Reset this node',
        icon: const Icon(Icons.restart_alt, size: 20),
        onPressed: () => pg.resetNode(i),
      ),
      children: [
        if (about != null) Text(about, style: theme.textTheme.bodySmall),
        for (final d in defs)
          LabeledSlider(
            label: '${d.symbol}  ${d.name}',
            help: d.help,
            value: d.log
                ? math.log(pg.param(i, d.field)) / math.ln10
                : pg.param(i, d.field),
            min: d.min,
            max: d.max,
            display: fmt(pg.param(i, d.field)),
            onChanged: (v) =>
                pg.setParam(i, d.field, d.log ? math.pow(10, v).toDouble() : v),
          ),
        for (final e in out)
          LabeledSlider(
            label: e.kind == EdgeKind.volatility
                ? 'κ  volatility coupling → ${defaultNodeLabel(e.child)}'
                : 'α  value coupling → ${defaultNodeLabel(e.child)}',
            help: e.kind == EdgeKind.volatility
                ? 'How strongly this node\'s mean scales the volatility of '
                      '${defaultNodeLabel(e.child)}.'
                : 'How strongly this node\'s mean predicts '
                      '${defaultNodeLabel(e.child)}.',
            value: e.strength,
            min: 0,
            max: 4,
            display: fmt(e.strength),
            onChanged: (v) => pg.setCoupling(e, v),
          ),
      ],
    );
  }
}

const _kindNames = {
  NodeKind.continuousState: 'continuous',
  NodeKind.binaryState: 'binary',
  NodeKind.constantState: 'constant',
  NodeKind.efState: 'exponential family',
  NodeKind.categoricalState: 'categorical',
  NodeKind.dpState: 'Dirichlet process',
};

/// A sentence on what node [i] does, for the kinds that need one.
String? _about(Network net, int i) {
  final children = net.valueChildrenOf(i);
  switch (net.kindOf(i)) {
    case NodeKind.binaryState:
      final cat = children.where(
        (c) => net.kindOf(c) == NodeKind.categoricalState,
      );
      if (cat.isNotEmpty) {
        return 'Observes whether its category came up (1) or not (0); its '
            'prediction, the sigmoid of its value parent\'s mean, is that '
            'category\'s probability.';
      }
      return 'Observes binary outcomes u ∈ {0, 1}. Its prediction is the '
          'sigmoid of its value parent\'s mean.';
    case NodeKind.categoricalState:
      return 'Observes one of ${net.valueParentsOf(i).length} categories, '
          'passes each binary parent its indicator, and derives a Dirichlet '
          'distribution over the categories from how much each one learned. '
          'It has no parameters of its own: edit its parents.';
    case NodeKind.dpState:
      return 'Assigns each observation to the cluster that explains it '
          'best, or opens a new one. The panel colours each observation by '
          'its cluster and draws each open cluster\'s predicted mean ± sd.';
    case NodeKind.efState:
      if (children.isNotEmpty) {
        return 'A cluster of ${defaultNodeLabel(children.first)}: it learns '
            'the mean and variance of the observations assigned to it.';
      }
      return 'Learns the expected sufficient statistics (x, x²) of its '
          'observations, so their mean and variance, at a fixed rate.';
    case NodeKind.continuousState || NodeKind.constantState:
      return null;
  }
}

// -----------------------------------------------------------------------------
// Data
// -----------------------------------------------------------------------------

class DataCard extends StatelessWidget {
  const DataCard({
    super.key,
    required this.playground,
    required this.settings,
    this.fitting,
  });

  final Playground playground;
  final Settings settings;

  /// Receives a responses column loaded with the observations (if the
  /// network's responses can be fitted).
  final Fitting? fitting;

  Future<void> _load(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    try {
      final picked = await pickTable(
        'Observations: one row per trial (and responses, optionally)',
      );
      if (picked == null) return;
      final (table, name) = picked;
      final f = fitting;
      final responses = f != null && f.responsesUnavailable == null;
      final choice =
          obviousColumns(table, observations: true, responses: responses) ??
          (context.mounted
              ? await showColumnMapping(
                  context,
                  table,
                  name,
                  observations: true,
                  responses: responses,
                )
              : null);
      if (choice == null) return;
      final values = table.columns[choice.observations!];
      final pg = playground;
      if (!Playground.loadedFits(pg.preset, values, pg.nCategories)) {
        throw FormatException(switch (pg.preset.input) {
          InputKind.categorical =>
            'This network has a categorical input: the values must be '
                'category indices 0 to ${pg.nCategories - 1}',
          _ => 'This network has a binary input: the values must be 0 or 1',
        });
      }
      final label = table.columns.length > 1
          ? '$name: ${table.names[choice.observations!]}'
          : name;
      playground.load(values, label);
      final r = choice.responses;
      if (r != null) {
        f!.loadResponses(table.columns[r], '$name: ${table.names[r]}');
      }
    } catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Could not load: $e')));
    }
  }

  /// The trajectories (and responses) as CSV.
  Future<void> _export(BuildContext context) async {
    final pg = playground;
    final traj = pg.trajectories;
    if (traj == null) return;
    final raw = Float64List.fromList(pg.observations);
    final obs = pg.observed;
    final csv = trajectoriesCsv(
      traj,
      observations: raw,
      observed: obs,
      responses: fitting?.responses,
    );
    await saveText(context, '${pg.preset.id}_trajectories.csv', csv);
  }

  @override
  Widget build(BuildContext context) {
    final pg = playground;
    final d = pg.data;
    final theme = Theme.of(context);
    final model = pg.preset.data;
    return SectionCard(
      title: 'Data',
      action: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          IconButton(
            tooltip: 'Save the trajectories as CSV',
            onPressed: pg.trajectories == null ? null : () => _export(context),
            icon: const Icon(Icons.download_outlined, size: 20),
          ),
          TextButton.icon(
            onPressed: () => _load(context),
            icon: const Icon(Icons.file_open_outlined, size: 18),
            label: const Text('Load CSV'),
          ),
        ],
      ),
      children: [
        if (pg.source == DataSource.loaded)
          Row(
            children: [
              Expanded(
                child: Text(
                  '${pg.loadedName} · ${pg.observations.length} trials'
                  '${pg.observed != null ? ' (with missing ones)' : ''}',
                  style: theme.textTheme.bodySmall,
                ),
              ),
              TextButton(
                onPressed: () => pg.updateData((_) {}),
                child: const Text('Simulate instead'),
              ),
            ],
          )
        else ...[
          Text(switch (model) {
            DataModel.reversal =>
              'Simulated reversal learning: the likely outcome swaps every '
                  'few trials.',
            DataModel.categoricalReversal =>
              'Simulated choices among ${pg.nCategories} outcomes: one is '
                  'likely, and which one changes every few trials.',
            DataModel.mixture =>
              'Draws from a few Gaussian sources (grey: the current '
                  'source\'s mean), switching source every few trials.',
            DataModel.randomWalk =>
              'A simulated random walk whose step size switches between '
                  'calm and volatile phases, observed with noise.',
          }, style: theme.textTheme.bodySmall),
          LabeledSlider(
            label: 'trials',
            // log₁₀ of the number of trials, 50 to the preset's maximum.
            value: math.log(d.nTrials) / math.ln10,
            min: math.log(50) / math.ln10,
            max: math.log(pg.preset.maxTrials) / math.ln10,
            display: '${d.nTrials}',
            onChanged: (v) => pg.updateData(
              (d) => d.nTrials = (math.pow(10, v) / 10).round() * 10,
            ),
            onChangeEnd: (_) => settings.remember(nTrials: d.nTrials),
          ),
          if (model == DataModel.mixture) ...[
            LabeledSlider(
              label: 'sources',
              value: d.nSources.toDouble(),
              min: 1,
              max: 6,
              divisions: 5,
              display: '${d.nSources}',
              onChanged: (v) => pg.updateData((d) => d.nSources = v.round()),
            ),
            LabeledSlider(
              label: 'source sd',
              value: d.sourceSd,
              min: 0.05,
              max: 2,
              display: d.sourceSd.toStringAsFixed(2),
              onChanged: (v) => pg.updateData((d) => d.sourceSd = v),
            ),
            LabeledSlider(
              label: 'switch every',
              value: d.block.toDouble(),
              min: 5,
              max: 200,
              display: '${d.block} trials',
              onChanged: (v) => pg.updateData((d) => d.block = v.round()),
            ),
          ] else if (model == DataModel.reversal ||
              model == DataModel.categoricalReversal) ...[
            LabeledSlider(
              label: 'P(likely outcome)',
              value: d.pHigh,
              min: model == DataModel.reversal ? 0.5 : 1 / pg.nCategories,
              max: 1,
              display: d.pHigh.toStringAsFixed(2),
              onChanged: (v) => pg.updateData((d) => d.pHigh = v),
            ),
            LabeledSlider(
              label: model == DataModel.reversal
                  ? 'reversal every'
                  : 'switch every',
              value: d.block.toDouble(),
              min: 5,
              max: 200,
              display: '${d.block} trials',
              onChanged: (v) => pg.updateData((d) => d.block = v.round()),
            ),
          ] else ...[
            LabeledSlider(
              label: 'calm step size',
              value: d.sdLow,
              min: 0,
              max: 1,
              display: d.sdLow.toStringAsFixed(2),
              onChanged: (v) => pg.updateData((d) => d.sdLow = v),
            ),
            LabeledSlider(
              label: 'volatile step size',
              value: d.sdHigh,
              min: 0,
              max: 1,
              display: d.sdHigh.toStringAsFixed(2),
              onChanged: (v) => pg.updateData((d) => d.sdHigh = v),
            ),
            LabeledSlider(
              label: 'observation noise',
              value: d.noise,
              min: 0.01,
              max: 1,
              display: d.noise.toStringAsFixed(2),
              onChanged: (v) => pg.updateData((d) => d.noise = v),
            ),
            LabeledSlider(
              label: 'switch every',
              value: d.block.toDouble(),
              min: 10,
              max: 1000,
              display: '${d.block} trials',
              onChanged: (v) => pg.updateData((d) => d.block = v.round()),
            ),
          ],
          Row(
            children: [
              Text('seed ${d.seed}', style: theme.textTheme.bodySmall),
              const Spacer(),
              TextButton.icon(
                onPressed: pg.reseed,
                icon: const Icon(Icons.casino_outlined, size: 18),
                label: const Text('New sample'),
              ),
            ],
          ),
        ],
      ],
    );
  }
}

class AboutCard extends StatelessWidget {
  const AboutCard({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return SectionCard(
      title: 'Reading the plots',
      children: [
        Text(
          'Each panel is one node, top level first. The line is the node\'s '
          'prediction μ̂ before each observation, the band ±1 standard '
          'deviation (1/√π̂). Input panels add the observations (dots) and, '
          'for simulated data, the hidden truth (grey). The bottom panel is '
          'the total surprise per trial.\n\n'
          'Categorical inputs show each category\'s predicted probability '
          'with the observed category as a row of dots on top; a Dirichlet '
          'process colours each observation by the cluster it joined, with '
          'each cluster\'s predicted mean ± sd.\n\n'
          'Hover or drag across the plots to read values. Play runs the '
          'network one trial at a time with predict/step, as in a real-time '
          'experiment.\n\n'
          'Filtering runs the hgf.dart engine, a pure-Dart port of pyhgf, '
          'checked against it to 1e-9. Fitting runs hgf_fit (MAP, Laplace, '
          'NUTS) on Web Workers, or isolates on desktop.',
          style: theme.textTheme.bodySmall,
        ),
      ],
    );
  }
}
