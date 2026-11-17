import 'package:hgf/hgf.dart' show Expr, parseCoupling;
import 'package:hgf_deep/hgf_deep.dart'
    show DeepFixes, DeepVolatilityUpdate, LayerKind, LearningKind;
import 'package:hgf_fit/codegen.dart';
import 'package:hgf_fit/document.dart';
import 'package:material_ui/material_ui.dart';

import '../../ui/files.dart';
import '../../ui/panels.dart';
import '../builder_model.dart';
import 'deep_canvas.dart' show layerSummary;
import 'inspector.dart' show NumberField;
import 'pipeline_inspector.dart' show CommitField, Pick;

/// Save deep network [d] as a runnable Dart or Python (pyhgf) script.
Future<void> exportDeep(
  BuildContext context,
  BuilderModel m,
  DocDeep d, {
  required bool python,
}) async {
  final messenger = ScaffoldMessenger.of(context);
  if (m.partIssues[DocPart.deep(d.id)] == IssueSeverity.error) {
    messenger.showSnackBar(
      SnackBar(content: Text('Fix the errors of "${d.label}" to export it')),
    );
    return;
  }
  final file = m.csvName;
  final layout = m.document.deepPipeline(d);
  final script = python
      ? DeepPythonExport(d, dataFile: file, layout: layout).script()
      : DeepDartExport(d, dataFile: file, layout: layout).script();
  final base = d.id.replaceAll(RegExp(r'[^A-Za-z0-9_]+'), '_');
  await saveText(
    context,
    python ? '$base.py' : '$base.dart',
    script,
    mimeType: python ? 'text/x-python' : 'text/x-dart',
  );
}

/// The couplings offered for a layer's activation.
const _named = ['linear', 'tanh', 'sigmoid', 'relu', 'leaky_relu', 'gelu'];
const _expression = '__expression__';

/// Starting points for a deep network.
enum _Template {
  classifier('Classifier', 'features → 8 tanh → a binary target'),
  regression('Regression', 'features → 16 tanh → a continuous target'),
  filter('Continuous filter', 'observations ← a volatile level (no training)');

  const _Template(this.label, this.hint);
  final String label, hint;

  DocDeep apply(DocDeep d) => switch (this) {
    classifier => DocDeep.starter(
      d.id,
    ).copyWith(name: d.name, features: d.features, training: d.training),
    regression => d.copyWith(
      layers: [
        const DocLayer(size: 1, name: 'target'),
        const DocLayer(size: 16, coupling: 'tanh', name: 'hidden'),
        DocLayer(size: d.features.isEmpty ? 1 : d.features.length),
      ],
      targets: const [],
      categories: () => null,
    ),
    filter => d.copyWith(
      layers: const [
        DocLayer(size: 1, kind: LayerKind.continuous),
        DocLayer(size: 1, kind: LayerKind.continuous, tonicVolatility: -2),
      ],
      features: const [],
      targets: const [],
      categories: () => null,
    ),
  };
}

/// A deep network: its name, columns, network settings, training and
/// export. Shown for its pipeline card, and on its canvas when no layer
/// is selected.
class DeepPanel extends StatelessWidget {
  const DeepPanel({super.key, required this.model, required this.deep});

  final BuilderModel model;
  final DocDeep deep;

  void _set(DocDeep next) => model.setDeep(deep.id, next);

  @override
  Widget build(BuildContext context) {
    final d = deep, s = d.settings, t = d.training;
    final theme = Theme.of(context);
    final small = theme.textTheme.bodySmall;
    final onCanvas = model.view == BuilderView.deep;
    final nodes = d.layers.fold(0, (a, l) => a + l.size);
    int? weights;
    try {
      weights = DocDeep.weightCount(d.build());
    } on Object {
      weights = null;
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SectionCard(
          title: 'Deep network',
          children: [
            CommitField(
              key: ValueKey('deep name ${d.id} ${d.label}'),
              label: 'Name',
              value: d.label,
              onCommit: (v) => _set(
                DocDeep(
                  id: d.id,
                  name: v.trim().isEmpty || v == d.id ? null : v.trim(),
                  settings: d.settings,
                  layers: d.layers,
                  features: d.features,
                  targets: d.targets,
                  categories: d.categories,
                  training: d.training,
                ),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              '${d.layers.length} layers · $nodes nodes'
              '${weights == null ? '' : ' · $weights weights'}'
              '${d.isContinuous ? ' · continuous (filters, does not learn)' : ''}',
              style: small,
            ),
            const SizedBox(height: 10),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                if (!onCanvas)
                  ActionChip(
                    avatar: const Icon(Icons.layers_outlined, size: 16),
                    label: const Text('Edit the layers'),
                    onPressed: () => model.showDeep(d.id),
                  ),
                ActionChip(
                  avatar: const Icon(Icons.code, size: 16),
                  label: const Text('Export Dart'),
                  onPressed: () => exportDeep(context, model, d, python: false),
                ),
                ActionChip(
                  avatar: const Icon(Icons.code, size: 16),
                  label: const Text('Export pyhgf'),
                  onPressed: () => exportDeep(context, model, d, python: true),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text('Start from', style: theme.textTheme.labelMedium),
            const SizedBox(height: 4),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                for (final tp in _Template.values)
                  Tooltip(
                    message: tp.hint,
                    child: ActionChip(
                      label: Text(tp.label),
                      onPressed: () => _set(tp.apply(d)),
                    ),
                  ),
              ],
            ),
          ],
        ),
        const SizedBox(height: 12),
        _ColumnsCard(model: model, deep: d),
        const SizedBox(height: 12),
        SectionCard(
          title: 'Network settings',
          children: [
            Pick<DeepVolatilityUpdate>(
              label: 'Volatility update',
              value: s.volatilityUpdates,
              items: const {
                DeepVolatilityUpdate.unbounded: 'unbounded',
                DeepVolatilityUpdate.ehgf: 'eHGF',
                DeepVolatilityUpdate.standard: 'standard',
              },
              onChanged: (v) =>
                  _set(d.copyWith(settings: s.copyWith(volatilityUpdates: v))),
            ),
            if (!d.isContinuous)
              _CouplingPick(
                label: 'Default activation',
                value: s.coupling,
                allowDefault: false,
                onChanged: (v) => _set(
                  d.copyWith(settings: s.copyWith(coupling: v ?? 'linear')),
                ),
              ),
            _Switch(
              'Tonic volatility ω on volatile layers',
              s.tonicVolatility,
              (v) => _set(d.copyWith(settings: s.copyWith(tonicVolatility: v))),
            ),
            if (!d.isContinuous) ...[
              _Switch(
                'Carry precision between samples',
                s.carryPrecision,
                (v) =>
                    _set(d.copyWith(settings: s.copyWith(carryPrecision: v))),
              ),
              _Switch(
                'Mean-field updates',
                s.meanFieldUpdates,
                (v) => _set(
                  d.copyWith(
                    settings: s.copyWith(
                      meanFieldUpdates: v,
                      feedforwardUncertainty: v
                          ? false
                          : s.feedforwardUncertainty,
                    ),
                  ),
                ),
              ),
              _Switch(
                'Feedforward uncertainty',
                s.feedforwardUncertainty,
                (v) => _set(
                  d.copyWith(
                    settings: s.copyWith(
                      feedforwardUncertainty: v,
                      meanFieldUpdates: v ? false : s.meanFieldUpdates,
                    ),
                  ),
                ),
              ),
            ],
            _Switch(
              "hgf.dart's corrections to pyhgf",
              s.fixes.curvatureFloor || s.fixes.logSpaceVolatility,
              (v) => _set(
                d.copyWith(
                  settings: s.copyWith(
                    fixes: v ? DeepFixes.recommended : DeepFixes.none,
                  ),
                ),
              ),
              hint:
                  'Off: exactly pyhgf (curvature floor, log-space volatility)',
            ),
          ],
        ),
        const SizedBox(height: 12),
        if (d.isContinuous)
          SectionCard(
            title: 'Filtering',
            children: [
              _TrainingStatus(model: model, deep: d),
              const SizedBox(height: 4),
              Text(
                'Each participant\'s rows are filtered in order by a fresh '
                'network; the prediction of each row is made before it is '
                'seen.',
                style: small,
              ),
            ],
          ),
        if (!d.isContinuous) ...[
          SectionCard(
            title: 'Training',
            children: [
              _TrainingStatus(model: model, deep: d),
              const SizedBox(height: 8),
              _TrainOn(model: model, deep: d, onChanged: _set),
              Pick<String>(
                label: 'Learning',
                value: t.learningKind == LearningKind.synapticUncertainty
                    ? 'synaptic'
                    : t.optimiser,
                items: const {
                  'adam': 'Adam',
                  'sgd': 'SGD',
                  'synaptic': 'synaptic uncertainty (no optimiser)',
                },
                onChanged: (v) => _set(
                  d.copyWith(
                    training: v == 'synaptic'
                        ? t.copyWith(
                            learningKind: LearningKind.synapticUncertainty,
                          )
                        : t.copyWith(
                            optimiser: v,
                            learningKind: LearningKind.precisionWeighted,
                          ),
                  ),
                ),
              ),
              if (t.learningKind == LearningKind.synapticUncertainty)
                NumberField(
                  key: ValueKey('window ${t.synapticWindow}'),
                  label: 'Window',
                  value: t.synapticWindow,
                  onSubmitted: (v) => _set(
                    d.copyWith(training: t.copyWith(synapticWindow: v ?? 20)),
                  ),
                )
              else
                NumberField(
                  key: ValueKey('lr ${t.learningRate}'),
                  label: 'Learning rate',
                  value: t.learningRate,
                  onSubmitted: (v) => _set(
                    d.copyWith(training: t.copyWith(learningRate: v ?? 0.01)),
                  ),
                ),
              NumberField(
                key: ValueKey('epochs ${t.epochs}'),
                label: 'Passes over the data',
                value: t.epochs.toDouble(),
                onSubmitted: (v) => _set(
                  d.copyWith(
                    training: t.copyWith(
                      epochs: (v ?? 10).round().clamp(1, 100000),
                    ),
                  ),
                ),
              ),
              NumberField(
                key: ValueKey('batch ${t.batchSize}'),
                label: 'Minibatch size (empty: one row at a time)',
                value: t.batchSize?.toDouble(),
                onSubmitted: (v) => _set(
                  d.copyWith(
                    training: t.copyWith(
                      batchSize: () => v?.round().clamp(1, 1 << 20),
                    ),
                  ),
                ),
              ),
              Pick<DeepInit>(
                label: 'Initial weights',
                value: t.init,
                items: {
                  for (final i in DeepInit.values)
                    i: i == DeepInit.none ? 'as built (1)' : i.pyhgfName,
                },
                onChanged: (v) =>
                    _set(d.copyWith(training: t.copyWith(init: v))),
              ),
              NumberField(
                key: ValueKey('seed ${t.seed}'),
                label: 'Seed',
                value: t.seed.toDouble(),
                onSubmitted: (v) => _set(
                  d.copyWith(training: t.copyWith(seed: (v ?? 0).round())),
                ),
              ),
              _Switch(
                'Standardise the features',
                t.standardise,
                (v) => _set(d.copyWith(training: t.copyWith(standardise: v))),
              ),
              _Switch(
                'Shuffle the rows each pass',
                t.shuffle,
                (v) => _set(d.copyWith(training: t.copyWith(shuffle: v))),
              ),
            ],
          ),
        ],
      ],
    );
  }
}

/// The columns a deep network reads: features and targets, each removable;
/// new ones are dropped onto its card, or typed (without data).
class _ColumnsCard extends StatelessWidget {
  const _ColumnsCard({required this.model, required this.deep});

  final BuilderModel model;
  final DocDeep deep;

  @override
  Widget build(BuildContext context) {
    final d = deep;
    final theme = Theme.of(context);
    Widget chips(List<String> columns, {required bool target}) => Wrap(
      spacing: 6,
      runSpacing: 4,
      children: [
        for (final c in columns)
          InputChip(
            label: Text(c),
            onDeleted: () => model.unbindDeepColumn(d.id, c, target: target),
          ),
      ],
    );
    Widget add({required bool target}) => _AddColumn(
      key: ValueKey('add ${d.id} $target'),
      model: model,
      label: target
          ? (d.isContinuous ? 'Add an observed column' : 'Add a target column')
          : 'Add a feature column',
      onAdd: (c) => model.bindDeepColumn(d.id, c, target: target),
    );
    return SectionCard(
      title: 'Columns',
      children: [
        Text(
          'Drag columns from the data card onto the network\'s card, or '
          'add them here. The ${d.isContinuous ? 'observation' : 'feature and target'} '
          'layers follow the number of columns.',
          style: theme.textTheme.bodySmall,
        ),
        if (!d.isContinuous) ...[
          const SizedBox(height: 8),
          Text('Features (x)', style: theme.textTheme.labelMedium),
          chips(d.features, target: false),
          add(target: false),
        ],
        const SizedBox(height: 10),
        Text(
          d.isContinuous ? 'Observations' : 'Targets (y)',
          style: theme.textTheme.labelMedium,
        ),
        chips(d.targets, target: true),
        if (d.categories case final cats?)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(
              'Labels, one node each: ${cats.join(', ')}',
              style: theme.textTheme.bodySmall,
            ),
          ),
        add(target: true),
        if (!d.isContinuous && d.targets.length == 1)
          CommitField(
            key: ValueKey('cats ${d.id} ${d.categories}'),
            label: 'Read the target as labels (comma-separated)',
            hint: 'e.g. left, right, none',
            value: d.categories?.join(', ') ?? '',
            validate: (v) {
              final n = _labels(v).length;
              return v.trim().isEmpty || n >= 2 ? null : 'At least two labels';
            },
            onCommit: (v) {
              final labels = _labels(v);
              model.setDeep(
                d.id,
                BuilderModel.syncDeepColumns(
                  d.copyWith(categories: () => labels.isEmpty ? null : labels),
                ),
              );
            },
          ),
      ],
    );
  }

  static List<String> _labels(String v) => [
    for (final s in v.split(','))
      if (s.trim().isNotEmpty) s.trim(),
  ];
}

/// A column added by name: picked from the data's columns when data is
/// loaded, typed otherwise.
class _AddColumn extends StatelessWidget {
  const _AddColumn({
    super.key,
    required this.model,
    required this.label,
    required this.onAdd,
  });

  final BuilderModel model;
  final String label;
  final ValueChanged<String> onAdd;

  @override
  Widget build(BuildContext context) {
    final columns = [
      ...?(model.csv == null ? null : model.dataColumns),
      ...model.document.transformNames,
      ...model.deepColumns,
    ];
    if (columns.isEmpty) {
      return CommitField(
        key: ValueKey(label),
        label: label,
        value: '',
        hint: 'a column name, then Enter',
        onCommit: (v) {
          if (v.trim().isNotEmpty) onAdd(v.trim());
        },
      );
    }
    return Pick<String>(
      label: label,
      value: null,
      items: {for (final c in columns) c: c},
      onChanged: (c) {
        if (c != null) onAdd(c);
      },
    );
  }
}

/// One layer of the open deep network.
class DeepLayerPanel extends StatelessWidget {
  const DeepLayerPanel({
    super.key,
    required this.model,
    required this.deep,
    required this.index,
  });

  final BuilderModel model;
  final DocDeep deep;
  final int index;

  void _set(DocLayer l) => model.setLayer(index, l);

  @override
  Widget build(BuildContext context) {
    final d = deep, l = d.layers[index];
    final theme = Theme.of(context);
    final n = d.layers.length;
    final isTarget = index == 0, isFeatures = index == n - 1;
    final hidden = !isTarget && !isFeatures;
    final cont = d.isContinuous;
    final sizeFollows = cont
        ? isTarget && d.targets.isNotEmpty
        : isFeatures && d.features.isNotEmpty ||
              isTarget && (d.targets.isNotEmpty || d.categories != null);
    final child = index > 0 ? d.layers[index - 1] : null;
    final role = cont
        ? (isTarget
              ? 'The observed layer: each node reads an observation column.'
              : 'A level above the observations: it predicts the layer '
                    'below and learns from its prediction errors.')
        : isTarget
        ? 'The target layer (y): clamped on the targets while training, '
              'predicted afterwards.'
        : isFeatures
        ? 'The feature layer (x): its means are set to the features; it '
              'predicts the layer below through the weights.'
        : 'A hidden layer: predicts the layer below through its activation '
              'and the weights, and learns from the errors coming up.';
    return SectionCard(
      title: d.layerName(index),
      action: hidden
          ? IconButton(
              tooltip: 'Remove this layer (Delete)',
              icon: const Icon(Icons.delete_outline),
              onPressed: () => model.removeLayer(index),
            )
          : null,
      children: [
        Text(role, style: theme.textTheme.bodySmall),
        const SizedBox(height: 4),
        Text(layerSummary(d, index), style: theme.textTheme.labelSmall),
        CommitField(
          key: ValueKey('layer name ${d.id} $index ${l.name}'),
          label: 'Name',
          value: l.name ?? '',
          hint: d.layerName(index),
          onCommit: (v) =>
              _set(l.copyWith(name: () => v.trim().isEmpty ? null : v.trim())),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: Text(
                sizeFollows
                    ? 'Nodes: ${l.size} (one per column)'
                    : 'Nodes: ${l.size}',
                style: theme.textTheme.bodyMedium,
              ),
            ),
            IconButton(
              tooltip: 'Fewer nodes',
              onPressed: sizeFollows || l.size <= 1
                  ? null
                  : () => _set(l.copyWith(size: l.size - 1)),
              icon: const Icon(Icons.remove),
            ),
            IconButton(
              tooltip: 'More nodes',
              onPressed: sizeFollows
                  ? null
                  : () => _set(l.copyWith(size: l.size + 1)),
              icon: const Icon(Icons.add),
            ),
            PopupMenuButton<int>(
              tooltip: 'Pick a size',
              enabled: !sizeFollows,
              icon: const Icon(Icons.more_horiz),
              onSelected: (v) => _set(l.copyWith(size: v)),
              itemBuilder: (_) => [
                for (final v in const [1, 2, 4, 8, 16, 32, 64, 128, 256])
                  PopupMenuItem(value: v, child: Text('$v')),
              ],
            ),
          ],
        ),
        if (isTarget && !cont)
          Pick<LayerKind>(
            label: 'Kind',
            value: l.kind,
            items: const {
              LayerKind.volatile: 'continuous value (regression)',
              LayerKind.binary: 'binary (one 0/1 per node)',
              LayerKind.categorical: 'categorical (one choice among the nodes)',
            },
            onChanged: (k) {
              if (k == null || k == l.kind) return;
              _set(
                l.copyWith(
                  kind: k,
                  size: k == LayerKind.categorical && l.size < 2 ? 3 : null,
                ),
              );
            },
          ),
        if (index > 0 && !cont)
          _CouplingPick(
            label: 'Activation (how it predicts the layer below)',
            value: l.coupling,
            allowDefault: true,
            defaultName: d.settings.coupling,
            onChanged: (v) => _set(l.copyWith(coupling: () => v)),
          ),
        if (l.kind == LayerKind.volatile && !cont) ...[
          _Switch(
            'Volatility level (learns how much it drifts)',
            l.volatilityParent ?? d.settings.carryPrecision,
            (v) => _set(l.copyWith(volatilityParent: () => v)),
          ),
          if (d.settings.tonicVolatility)
            NumberField(
              key: ValueKey('omega ${d.id} $index ${l.tonicVolatility}'),
              label: 'Tonic volatility ω (empty: −4)',
              value: l.tonicVolatility,
              onSubmitted: (v) => _set(l.copyWith(tonicVolatility: () => v)),
            ),
        ],
        if (cont && !isTarget)
          NumberField(
            key: ValueKey('omega ${d.id} $index ${l.tonicVolatility}'),
            label: 'Tonic volatility ω',
            value: l.tonicVolatility,
            onSubmitted: (v) => _set(l.copyWith(tonicVolatility: () => v)),
          ),
        if (index > 0 && !cont) ...[
          _Switch(
            'Bias (a constant input)',
            l.addConstantInput ?? l.fullyConnected,
            l.fullyConnected
                ? (v) => _set(l.copyWith(addConstantInput: () => v))
                : null,
          ),
          if (child != null && child.size == l.size)
            _Switch(
              'One-to-one with the layer below',
              !l.fullyConnected,
              (v) => _set(
                l.copyWith(
                  fullyConnected: !v,
                  addConstantInput: () => v ? false : null,
                ),
              ),
            ),
        ],
      ],
    );
  }
}

/// A coupling picked by name, or written as an expression in `x`.
class _CouplingPick extends StatefulWidget {
  const _CouplingPick({
    required this.label,
    required this.value,
    required this.onChanged,
    required this.allowDefault,
    this.defaultName,
  });

  final String label;
  final String? value;
  final bool allowDefault;
  final String? defaultName;
  final ValueChanged<String?> onChanged;

  @override
  State<_CouplingPick> createState() => _CouplingPickState();
}

class _CouplingPickState extends State<_CouplingPick> {
  late bool _writing = widget.value != null && !_named.contains(widget.value);

  @override
  void didUpdateWidget(_CouplingPick old) {
    super.didUpdateWidget(old);
    if (old.value != widget.value && widget.value != null) {
      _writing = !_named.contains(widget.value);
    }
  }

  @override
  Widget build(BuildContext context) {
    final v = widget.value;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Pick<String?>(
          label: widget.label,
          value: _writing ? _expression : v,
          items: {
            if (widget.allowDefault)
              null: 'the network\'s (${widget.defaultName})',
            for (final n in _named) n: n,
            _expression: 'an expression in x…',
          },
          onChanged: (c) {
            if (c == _expression) {
              setState(() => _writing = true);
              return;
            }
            setState(() => _writing = false);
            widget.onChanged(c);
          },
        ),
        if (_writing)
          CommitField(
            key: ValueKey('expr $v'),
            label: 'g(x)',
            value: v != null && !_named.contains(v) ? v : '2 * tanh(x / 2)',
            monospace: true,
            validate: (t) {
              try {
                parseCoupling(t);
                return null;
              } on ArgumentError catch (e) {
                final m = '${e.message}';
                final colon = m.lastIndexOf(': ');
                return colon < 0 ? m : m.substring(colon + 2);
              }
            },
            onCommit: (t) => widget.onChanged(t.trim()),
          ),
      ],
    );
  }
}

class _Switch extends StatelessWidget {
  const _Switch(this.label, this.value, this.onChanged, {this.hint});

  final String label;
  final String? hint;
  final bool value;
  final ValueChanged<bool>? onChanged;

  @override
  Widget build(BuildContext context) => SwitchListTile(
    dense: true,
    contentPadding: EdgeInsets.zero,
    title: Text(label),
    subtitle: hint == null ? null : Text(hint!),
    value: value,
    onChanged: onChanged,
  );
}

/// How deep network [deep]'s training goes: a progress bar while it runs,
/// then its scores (training and held-out rows).
class _TrainingStatus extends StatelessWidget {
  const _TrainingStatus({required this.model, required this.deep});

  final BuilderModel model;
  final DocDeep deep;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final run = model.deepRun(deep.id);
    final error = model.deepError(deep.id);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (run != null) ...[
          LinearProgressIndicator(value: run.progress),
          const SizedBox(height: 4),
        ],
        Text(
          model.deepStatus(deep.id),
          style: theme.textTheme.bodySmall?.copyWith(
            color: error != null ? theme.colorScheme.error : null,
          ),
        ),
      ],
    );
  }
}

/// Which rows deep network [deep] trains on: an expression (empty: every
/// row), with the split's training part offered when there is a split.
class _TrainOn extends StatelessWidget {
  const _TrainOn({
    required this.model,
    required this.deep,
    required this.onChanged,
  });

  final BuilderModel model;
  final DocDeep deep;
  final ValueChanged<DocDeep> onChanged;

  void _set(Expr? e) => onChanged(
    deep.copyWith(training: deep.training.copyWith(trainOn: () => e)),
  );

  @override
  Widget build(BuildContext context) {
    final t = deep.training;
    final split = [
      for (final s in model.document.data.steps)
        if (s is SplitStep) s,
    ].lastOrNull;
    final suggestion = split == null
        ? null
        : Expr.parse(
            "${split.column} == '${split.parts.first.label.replaceAll("'", '')}'",
          );
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        CommitField(
          key: ValueKey('train on ${deep.id} ${t.trainOn}'),
          label: 'Train on the rows where',
          value: t.trainOn?.toString() ?? '',
          hint: 'every row',
          monospace: true,
          validate: (v) {
            if (v.trim().isEmpty) return null;
            try {
              Expr.parse(v);
              return null;
            } on FormatException catch (e) {
              return e.message;
            }
          },
          onCommit: (v) => _set(v.trim().isEmpty ? null : Expr.parse(v)),
        ),
        if (suggestion != null && '${t.trainOn}' != '$suggestion')
          Align(
            alignment: Alignment.centerLeft,
            child: Padding(
              padding: const EdgeInsets.only(top: 4),
              child: ActionChip(
                avatar: const Icon(Icons.call_split, size: 16),
                label: Text('Train on $suggestion'),
                onPressed: () => _set(suggestion),
              ),
            ),
          ),
        const SizedBox(height: 4),
        Text(
          'The other rows are held out: the trained network predicts them '
          'too, and they are scored apart.',
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    );
  }
}
