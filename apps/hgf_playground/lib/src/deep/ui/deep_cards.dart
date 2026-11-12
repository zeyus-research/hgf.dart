import 'dart:math' as math;

import 'package:material_ui/material_ui.dart';
import 'package:hgf_deep/hgf_deep.dart';

import '../../ui/panels.dart' show LabeledSlider, SectionCard, fmt;
import '../deep_playground.dart';
import '../deep_spec.dart';
import '../deep_view.dart';
import 'deep_graph.dart';

/// Builds the network: task, hidden layers, update scheme and learning.
class ArchitectureCard extends StatelessWidget {
  const ArchitectureCard({super.key, required this.deep});

  final DeepPlayground deep;

  DeepSpec get spec => deep.spec;

  void _layers(List<HiddenLayerSpec> hidden) =>
      deep.setSpec(spec.copyWith(hidden: hidden));

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final task = deep.task;
    final (outSize, outKind) = task.output;
    return SectionCard(
      title: 'Network',
      children: [
        DropdownButtonFormField<DeepTask>(
          initialValue: task,
          isExpanded: true,
          decoration: const InputDecoration(labelText: 'Task', isDense: true),
          items: [
            for (final t in DeepTask.values)
              DropdownMenuItem(value: t, child: Text(t.label)),
          ],
          onChanged: (t) => t == null ? null : deep.setTask(t),
        ),
        Padding(
          padding: const EdgeInsets.only(top: 4, bottom: 8),
          child: Text(task.description, style: theme.textTheme.bodySmall),
        ),
        _fixedRow(context, 'Input', '${task.inputs} (x is clamped here)'),
        for (var i = 0; i < spec.hidden.length; i++)
          _HiddenRow(
            index: i,
            layer: spec.hidden[i],
            onChanged: (l) => _layers([...spec.hidden]..[i] = l),
            onRemove: () => _layers([...spec.hidden]..removeAt(i)),
          ),
        Align(
          alignment: Alignment.centerLeft,
          child: TextButton.icon(
            onPressed: spec.hidden.length >= 6
                ? null
                : () => _layers([
                    ...spec.hidden,
                    spec.hidden.isEmpty
                        ? const HiddenLayerSpec()
                        : spec.hidden.last,
                  ]),
            icon: const Icon(Icons.add, size: 18),
            label: const Text('Hidden layer'),
          ),
        ),
        _fixedRow(
          context,
          'Output',
          '$outSize ${switch (outKind) {
            LayerKind.binary => 'binary (probability of class 1)',
            LayerKind.categorical => 'categorical (one per class)',
            LayerKind.volatile => 'continuous',
          }}',
        ),
        const SizedBox(height: 6),
        Text(
          '${spec.nodes(task)} nodes, ${spec.parameters(task)} weights',
          style: theme.textTheme.bodySmall,
        ),
        const Divider(height: 20),
        Text('Volatility update', style: theme.textTheme.labelMedium),
        const SizedBox(height: 4),
        SegmentedButton<DeepVolatilityUpdate>(
          showSelectedIcon: false,
          segments: const [
            ButtonSegment(
              value: DeepVolatilityUpdate.standard,
              label: Text('standard'),
            ),
            ButtonSegment(
              value: DeepVolatilityUpdate.ehgf,
              label: Text('eHGF'),
            ),
            ButtonSegment(
              value: DeepVolatilityUpdate.unbounded,
              label: Text('unbounded'),
            ),
          ],
          selected: {spec.volatilityUpdates},
          onSelectionChanged: (s) =>
              deep.setSpec(spec.copyWith(volatilityUpdates: s.first)),
        ),
        SwitchListTile(
          dense: true,
          contentPadding: EdgeInsets.zero,
          title: const Text('Tonic volatility ω'),
          subtitle: const Text(
            'A baseline drift e^ω for every volatile node (off: pyhgf\'s '
            'default, the drift is e^μ_vol alone)',
          ),
          value: spec.tonicVolatility != null,
          onChanged: (on) => deep.setSpec(
            spec.copyWith(tonicVolatility: () => on ? -4.0 : null),
          ),
        ),
        if (spec.tonicVolatility != null)
          LabeledSlider(
            label: 'ω',
            value: spec.tonicVolatility!,
            min: -8,
            max: 0,
            divisions: 16,
            display: fmt(spec.tonicVolatility!),
            onChanged: (v) =>
                deep.setSpec(spec.copyWith(tonicVolatility: () => v)),
          ),
        SwitchListTile(
          dense: true,
          contentPadding: EdgeInsets.zero,
          title: const Text('hgf_deep fixes'),
          subtitle: const Text(
            'Corrections to pyhgf that keep runaway volatility finite '
            '(off: pyhgf exactly)',
          ),
          value: spec.fixes,
          onChanged: (on) => deep.setSpec(spec.copyWith(fixes: on)),
        ),
        const Divider(height: 20),
        Wrap(
          spacing: 12,
          runSpacing: 4,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            Text('Optimiser', style: theme.textTheme.labelMedium),
            SegmentedButton<OptimiserKind>(
              showSelectedIcon: false,
              segments: const [
                ButtonSegment(value: OptimiserKind.adam, label: Text('Adam')),
                ButtonSegment(value: OptimiserKind.sgd, label: Text('SGD')),
              ],
              selected: {spec.optimiser},
              onSelectionChanged: (s) =>
                  deep.setSpec(spec.copyWith(optimiser: s.first)),
            ),
          ],
        ),
        LabeledSlider(
          label: 'Learning rate (log₁₀)',
          value: math.log(spec.learningRate) / math.ln10,
          min: -4,
          max: -0.5,
          divisions: 14,
          display: fmt(spec.learningRate),
          onChanged: (v) => deep.setSpec(
            spec.copyWith(learningRate: math.pow(10, v).toDouble()),
          ),
        ),
        Row(
          children: [
            Expanded(
              child: Text(
                'Initial weights: seed ${spec.seed}',
                style: theme.textTheme.bodyMedium,
              ),
            ),
            TextButton(
              onPressed: () => deep.setSpec(
                spec.copyWith(seed: spec.seed + 1),
                keepWeights: false,
              ),
              child: const Text('New weights'),
            ),
          ],
        ),
        Text(
          'Changing a layer\'s activation, its volatility level, the update '
          'or ω keeps the learned weights (only the beliefs start afresh), so '
          'you can compare the same sample both ways.',
          style: theme.textTheme.bodySmall,
        ),
      ],
    );
  }

  Widget _fixedRow(BuildContext context, String name, String what) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 4),
    child: Row(
      children: [
        SizedBox(
          width: 64,
          child: Text(name, style: Theme.of(context).textTheme.labelMedium),
        ),
        Expanded(
          child: Text(what, style: Theme.of(context).textTheme.bodySmall),
        ),
      ],
    ),
  );
}

class _HiddenRow extends StatelessWidget {
  const _HiddenRow({
    required this.index,
    required this.layer,
    required this.onChanged,
    required this.onRemove,
  });

  final int index;
  final HiddenLayerSpec layer;
  final ValueChanged<HiddenLayerSpec> onChanged;
  final VoidCallback onRemove;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          SizedBox(
            width: 64,
            child: Text(
              'Hidden ${index + 1}',
              style: theme.textTheme.labelMedium,
            ),
          ),
          DropdownButton<int>(
            value: layer.size,
            isDense: true,
            underline: const SizedBox.shrink(),
            items: [
              for (final s in hiddenSizes)
                DropdownMenuItem(value: s, child: Text('$s')),
            ],
            onChanged: (s) =>
                s == null ? null : onChanged(layer.copyWith(size: s)),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: DropdownButton<CouplingFn>(
              value: layer.coupling,
              isDense: true,
              isExpanded: true,
              underline: const SizedBox.shrink(),
              items: [
                for (final c in hiddenCouplings)
                  DropdownMenuItem(value: c, child: Text(c.name)),
              ],
              onChanged: (c) =>
                  c == null ? null : onChanged(layer.copyWith(coupling: c)),
            ),
          ),
          Tooltip(
            message:
                'Volatility level: each node learns how much it drifts '
                'between samples (μ_vol, the diamond in the graph)',
            child: FilterChip(
              label: const Text('vol'),
              selected: layer.volatility,
              visualDensity: VisualDensity.compact,
              onSelected: (v) => onChanged(layer.copyWith(volatility: v)),
            ),
          ),
          IconButton(
            tooltip: 'Remove layer',
            visualDensity: VisualDensity.compact,
            icon: const Icon(Icons.close, size: 18),
            onPressed: onRemove,
          ),
        ],
      ),
    );
  }
}

const _speeds = <int?>[1, 2, 5, 10, 25, 50, 100, null];

/// Why the beliefs went NaN, for the scheme in use, and what to try.
String _divergedText(DeepPlayground deep) {
  final layer = layerName(deep.network, deep.divergedLayer!);
  final why = switch (deep.spec.volatilityUpdates) {
    DeepVolatilityUpdate.standard =>
      "the standard update's volatility precision "
          'π_vol = π̂_vol + ½γ² + γ²δ_vol − ½γδ_vol turned negative after a '
          'large volatility error, which this scheme cannot recover from',
    DeepVolatilityUpdate.ehgf || DeepVolatilityUpdate.unbounded =>
      "a node's volatility level grew until its predicted precision π̂ was "
          '0, and with no evidence left its posterior is undefined (0/0)',
  };
  final tries = [
    "turn the $layer layer's volatility level off",
    if (deep.spec.tonicVolatility == null) 'add a tonic volatility ω',
    if (!deep.spec.fixes) 'turn the hgf_deep fixes on',
    'lower the learning rate',
  ];
  return 'The beliefs of the $layer layer went NaN: $why (watch the '
      'diamonds, μ_vol). Training stopped. Try to ${tries.join(', ')}.';
}

int _speedIndex(int? n) {
  final i = _speeds.indexOf(n);
  return i < 0 ? _speeds.length - 1 : i;
}

/// Training controls and the latest scores.
class TrainingCard extends StatelessWidget {
  const TrainingCard({super.key, required this.deep});

  final DeepPlayground deep;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final h = deep.history;
    final cls = deep.task.classes > 0;
    final score = cls ? 'accuracy' : 'R²';
    return SectionCard(
      title: 'Training',
      children: [
        Text(
          'One sample at a time (DeepNetwork.fit), shuffled every epoch: '
          '${deep.train.length} training and ${deep.test.length} test samples.',
          style: theme.textTheme.bodySmall,
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 4,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            FilledButton.tonalIcon(
              onPressed: deep.playing ? deep.pause : deep.play,
              icon: Icon(deep.playing ? Icons.pause : Icons.play_arrow),
              label: Text(deep.playing ? 'Pause' : 'Train'),
            ),
            OutlinedButton(
              onPressed: deep.playing ? null : deep.stepEpoch,
              child: const Text('+1 epoch'),
            ),
            IconButton(
              tooltip: 'New weights, empty history',
              onPressed: deep.reset,
              icon: const Icon(Icons.restart_alt),
            ),
          ],
        ),
        LabeledSlider(
          label: 'Speed',
          value: _speedIndex(deep.samplesPerFrame).toDouble(),
          min: 0,
          max: (_speeds.length - 1).toDouble(),
          divisions: _speeds.length - 1,
          display: deep.samplesPerFrame == null
              ? 'as fast as possible'
              : '${deep.samplesPerFrame} sample'
                    '${deep.samplesPerFrame == 1 ? '' : 's'} per frame',
          help:
              'Slow it down to watch the traced sample\'s beliefs change '
              'while the network learns.',
          onChanged: (v) => deep.samplesPerFrame = _speeds[v.round()],
        ),
        const SizedBox(height: 8),
        Text(
          'Epoch ${deep.epoch}'
          '${deep.row > 0 ? ' (+${deep.row}/${deep.train.length})' : ''}',
          style: theme.textTheme.bodyMedium,
        ),
        if (h.isNotEmpty)
          Text(
            'test $score ${h.last.testScore.toStringAsFixed(3)} · '
            'train $score ${h.last.trainScore.toStringAsFixed(3)}',
            style: theme.textTheme.bodyMedium?.copyWith(
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
        if (deep.divergedLayer != null)
          Padding(
            padding: const EdgeInsets.only(top: 6),
            child: Text(
              _divergedText(deep),
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.error,
              ),
            ),
          ),
      ],
    );
  }
}

/// Steps through one sample's pass through the network.
class TraceCard extends StatelessWidget {
  const TraceCard({
    super.key,
    required this.deep,
    required this.field,
    required this.onField,
    required this.selected,
    required this.onSelect,
  });

  final DeepPlayground deep;
  final NodeField field;
  final ValueChanged<NodeField> onField;
  final NodeRef? selected;
  final ValueChanged<NodeRef?> onSelect;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final trace = deep.trace;
    final net = deep.network;
    if (trace == null) return const SizedBox.shrink();
    final step = trace.steps[deep.cursor];
    String vec(List<double> v) => '(${v.map(fmt).join(', ')})';
    return SectionCard(
      title: 'Follow one sample',
      children: [
        Text(
          'x = ${vec(deep.traceInput)} → target y = ${vec(deep.traceTarget)}'
          '${deep.targetIsNearest ? ' (class of the nearest training point)' : ''}',
          style: theme.textTheme.bodySmall,
        ),
        Row(
          children: [
            Expanded(
              child: Text(
                'Prediction ${vec(trace.prediction)}',
                style: theme.textTheme.bodySmall,
              ),
            ),
            TextButton(
              onPressed: () => deep.traceTestSample(
                (deep.traceSample + 1) % deep.test.length,
              ),
              child: const Text('Next test sample'),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Wrap(
          spacing: 4,
          runSpacing: 4,
          children: [
            for (var i = 0; i < trace.steps.length; i++)
              ChoiceChip(
                label: Text(
                  _chip(net, trace.steps[i]),
                  style: const TextStyle(fontSize: 11),
                ),
                selected: i == deep.cursor,
                visualDensity: VisualDensity.compact,
                onSelected: (_) => deep.setCursor(i),
              ),
          ],
        ),
        const SizedBox(height: 6),
        Wrap(
          crossAxisAlignment: WrapCrossAlignment.center,
          runSpacing: 4,
          children: [
            IconButton(
              tooltip: 'Previous phase',
              onPressed: deep.cursor > 0 ? () => deep.stepCursor(-1) : null,
              icon: const Icon(Icons.chevron_left),
            ),
            Text('${deep.cursor + 1} / ${trace.steps.length}'),
            IconButton(
              tooltip: 'Next phase',
              onPressed: deep.cursor < trace.steps.length - 1
                  ? () => deep.stepCursor(1)
                  : null,
              icon: const Icon(Icons.chevron_right),
            ),
            const SizedBox(width: 8),
            Tooltip(
              message:
                  'The trace is a preview: the network is left as it was. '
                  'Learn from it to take the step for real.',
              child: OutlinedButton(
                onPressed: deep.commitTrace,
                child: const Text('Learn this sample'),
              ),
            ),
          ],
        ),
        Text(phaseTitle(net, step), style: theme.textTheme.titleSmall),
        const SizedBox(height: 4),
        Text(phaseExplanation(step), style: theme.textTheme.bodySmall),
        const Divider(height: 20),
        Text('Node colour', style: theme.textTheme.labelMedium),
        const SizedBox(height: 4),
        Wrap(
          spacing: 4,
          children: [
            for (final f in NodeField.values)
              Tooltip(
                message: f.description,
                child: ChoiceChip(
                  label: Text(f.symbol),
                  selected: f == field,
                  onSelected: (_) => onField(f),
                ),
              ),
          ],
        ),
        const SizedBox(height: 8),
        if (selected == null)
          Text(
            'Tap a node to see its beliefs before and after this phase.',
            style: theme.textTheme.bodySmall,
          )
        else
          _Inspector(
            deep: deep,
            node: selected!,
            onClose: () => onSelect(null),
          ),
      ],
    );
  }

  static String _chip(DeepNetwork net, TraceStep s) {
    final n = s.layer >= 0 ? layerName(net, s.layer) : '';
    return switch (s.phase) {
      TracePhase.clampInput => 'x',
      TracePhase.predict => 'predict $n',
      TracePhase.observe => 'y',
      TracePhase.leafError => 'δ output',
      TracePhase.posterior => 'update $n',
      TracePhase.layerError => 'δ $n',
      TracePhase.weightUpdate => 'learn',
      TracePhase.carry => 'carry',
    };
  }
}

/// The selected node's beliefs before and after the shown phase.
class _Inspector extends StatelessWidget {
  const _Inspector({
    required this.deep,
    required this.node,
    required this.onClose,
  });

  final DeepPlayground deep;
  final NodeRef node;
  final VoidCallback onClose;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final trace = deep.trace!;
    final net = deep.network;
    if (node.layer >= net.nLayers || node.node >= net.layerSizes[node.layer]) {
      return const SizedBox.shrink();
    }
    final after = trace.steps[deep.cursor].states[node.layer];
    final before = deep.statesBefore![node.layer];
    final mono = theme.textTheme.bodySmall?.copyWith(
      fontFeatures: const [FontFeature.tabularFigures()],
    );
    const fields = [
      (LayerField.expectedMean, 'μ̂ prediction'),
      (LayerField.expectedPrecision, 'π̂ predicted precision'),
      (LayerField.mean, 'μ posterior mean'),
      (LayerField.precision, 'π posterior precision'),
      (LayerField.valuePredictionError, 'δ value error'),
      (LayerField.effectivePrecision, 'γ effective precision'),
      (LayerField.meanVol, 'μ_vol volatility level'),
      (LayerField.expectedPrecisionVol, 'π̂_vol its precision'),
      (LayerField.volatilityPredictionError, 'δ_vol volatility error'),
    ];
    final rows = <TableRow>[
      TableRow(
        children: [
          Text('', style: mono),
          Text('before', style: mono, textAlign: TextAlign.right),
          Text('after', style: mono, textAlign: TextAlign.right),
        ],
      ),
    ];
    for (final (f, name) in fields) {
      final a = before.field(f), b = after.field(f);
      if (a == null || b == null) continue;
      final va = a[node.node], vb = b[node.node];
      final changed = !(va == vb || (va.isNaN && vb.isNaN));
      final style = changed
          ? mono?.copyWith(
              fontWeight: FontWeight.w600,
              color: theme.colorScheme.primary,
            )
          : mono;
      rows.add(
        TableRow(
          children: [
            Text(name, style: mono),
            Text(fmt(va), style: mono, textAlign: TextAlign.right),
            Text(fmt(vb), style: style, textAlign: TextAlign.right),
          ],
        ),
      );
    }
    final w = trace.weightsAt(deep.cursor)[node.layer];
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                '${layerName(net, node.layer)} node ${node.node + 1}',
                style: theme.textTheme.labelLarge,
              ),
            ),
            IconButton(
              visualDensity: VisualDensity.compact,
              icon: const Icon(Icons.close, size: 18),
              onPressed: onClose,
            ),
          ],
        ),
        Table(
          columnWidths: const {
            0: FlexColumnWidth(2.2),
            1: FlexColumnWidth(),
            2: FlexColumnWidth(),
          },
          children: rows,
        ),
        if (w != null && node.layer < net.nLayers - 1 && w.cols <= 33) ...[
          const SizedBox(height: 6),
          Text(
            'Weights in from the ${layerName(net, node.layer + 1)} layer'
            '${net.configs[node.layer + 1].addConstantInput ? ' (last: bias)' : ''}:',
            style: mono,
          ),
          Text(
            [
              for (var j = 0; j < w.cols; j++) fmt(w.at(node.node, j)),
            ].join('  '),
            style: mono,
          ),
        ],
      ],
    );
  }
}

/// What the page is and how to read it.
class DeepAboutCard extends StatelessWidget {
  const DeepAboutCard({super.key});

  @override
  Widget build(BuildContext context) {
    final s = Theme.of(context).textTheme.bodySmall;
    return SectionCard(
      title: 'About',
      children: [
        Text(
          'A deep HGF network (hgf_deep, a port of pyhgf\'s DeepNetwork) is a '
          'stack of layers of Gaussian beliefs. Each layer predicts the next '
          'through weights and an activation (value coupling): the input x '
          'is clamped on the left, the prediction flows right, the target y '
          'is clamped on the right, and precision-weighted prediction errors '
          'flow back left. The weights then learn from the settled beliefs. '
          'No gradients are back-propagated: every node updates from its '
          'neighbours\' errors.',
          style: s,
        ),
        const SizedBox(height: 6),
        Text(
          'Volatility coupling: a node can carry a volatility level (its '
          'diamond), a belief about how much the node drifts between samples. '
          'A high μ_vol makes the node\'s prediction vague (low π̂), so '
          'evidence moves it a lot; large moves in turn raise μ_vol. In deep '
          'networks this feedback can run away, which you can watch here: '
          'turn a layer\'s "vol" on, train, and follow μ_vol and π̂.',
          style: s,
        ),
        const SizedBox(height: 6),
        Text(
          'Edges: blue positive, orange negative; at rest their width is the '
          'weight, in a traced phase the message it carries (the prediction '
          'contribution, the weighted error, or the weight change). Node '
          'rings: thicker is more precise.',
          style: s,
        ),
      ],
    );
  }
}
