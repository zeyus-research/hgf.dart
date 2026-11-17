import 'dart:math' as math;

import 'package:hgf_fit/document.dart';
import 'package:hgf_fit/hgf_fit.dart'
    show
        FitCodec,
        GammaPrior,
        HalfNormal,
        LogNormal,
        Normal,
        Prior,
        TruncatedNormal,
        Uniform;
import 'package:material_ui/material_ui.dart';

import '../../model/fit_config.dart';
import '../../model/fit_jobs.dart';
import '../../model/presets.dart' show paramsFor;
import '../../ui/files.dart';
import '../../ui/panels.dart' show SectionCard;
import '../builder_fit.dart';
import '../builder_model.dart';
import '../doc_layout.dart' show ownerIdOf;
import 'inspector.dart' show NumberField;
import 'pipeline_inspector.dart' show CommitField, Pick;

const _codec = FitCodec();

/// The fit's inspector: the free parameters (priors, and how each is
/// fitted across participants), the method, the Fit button and the
/// results.
class FitPanel extends StatelessWidget {
  const FitPanel({super.key, required this.model});

  final BuilderModel model;

  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      _ParametersCard(model: model),
      _MethodCard(model: model),
      if (model.fitting?.result != null) _ResultsCard(model: model),
    ],
  );
}

// -----------------------------------------------------------------------------
// Parameters
// -----------------------------------------------------------------------------

/// A "free" chip beside a slider: estimate [target] in the fit (with a
/// prior around [current]), or fix it again.
class FreeToggle extends StatelessWidget {
  const FreeToggle({
    super.key,
    required this.model,
    required this.target,
    required this.current,
    this.label = 'free',
  });

  final BuilderModel model;
  final DocTarget target;
  final double current;
  final String label;

  @override
  Widget build(BuildContext context) {
    final p = model.document.parameterFor(target);
    return Tooltip(
      message: p == null
          ? 'Estimate it in the fit (Fit card)'
          : 'Estimated as "${p['name']}"; tap to fix it again',
      child: FilterChip(
        label: Text(label),
        selected: p != null,
        showCheckmark: false,
        visualDensity: VisualDensity.compact,
        materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
        labelPadding: const EdgeInsets.symmetric(horizontal: 2),
        onSelected: (on) => model.setFree(target, on, current: current),
      ),
    );
  }
}

/// A parameter the model could free: its target, a label and its value.
typedef _Candidate = ({DocTarget target, String label, double current});

/// What [m]'s document can free and doesn't: the node parameters the
/// network's sliders show, the edges' strengths and the response model's
/// parameters.
List<_Candidate> _candidates(BuilderModel m) {
  final doc = m.doc;
  final out = <_Candidate>[];
  for (final n in doc.nodes) {
    if (n is! ScalarNode && n is! DpStateNode) continue;
    final kind = docKind(n);
    final input = !doc.edges.any((e) => ownerIdOf(e.parent) == n.id);
    for (final d in paramsFor(kind, input: input)) {
      out.add((
        target: nodeTarget(n.id, d.field),
        label: '${d.symbol} ${prettyId(n.id)} · ${d.name}',
        current: m.initial(n.id, d.field),
      ));
    }
  }
  for (final e in doc.edges) {
    out.add((
      target: couplingTarget(e.type, e.parent, e.child),
      label:
          '${e.type == EdgeType.value ? 'value' : 'volatility'} coupling '
          '${prettyId(e.parent)} → ${prettyId(e.child)}',
      current: e.strength,
    ));
  }
  for (final MapEntry(:key, :value) in m.responseParameters.entries) {
    out.add((
      target: responseTarget(key),
      label: 'response · $key',
      current: value,
    ));
  }
  return [
    for (final c in out)
      if (!m.isFree(c.target)) c,
  ];
}

/// [t] in words.
String describeTarget(DocTarget t) => switch (t['type']) {
  'node' => '${t['field']} of ${prettyId('${t['node']}')}',
  'value_coupling' =>
    'value coupling ${prettyId('${t['parent']}')} → ${prettyId('${t['child']}')}',
  'volatility_coupling' =>
    'volatility coupling ${prettyId('${t['parent']}')} → ${prettyId('${t['child']}')}',
  _ => 'the response\'s ${t['name']}',
};

class _ParametersCard extends StatelessWidget {
  const _ParametersCard({required this.model});

  final BuilderModel model;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final doc = model.document;
    final candidates = _candidates(model);
    return SectionCard(
      title: 'Free parameters',
      action: PopupMenuButton<_Candidate>(
        tooltip: 'Free a parameter',
        icon: const Icon(Icons.add, size: 20),
        enabled: candidates.isNotEmpty,
        onSelected: (c) => model.setFree(c.target, true, current: c.current),
        itemBuilder: (context) => [
          for (final c in candidates)
            PopupMenuItem(value: c, child: Text(c.label)),
        ],
      ),
      children: [
        if (doc.parameters.isEmpty)
          Text(
            'None yet. Add one with +, or tick "free" beside a slider in the '
            'network\'s inspector. Free parameters are estimated per '
            'participant; the rest stay as set.',
            style: theme.textTheme.bodySmall,
          ),
        for (final p in doc.parameters)
          _ParameterTile(
            key: ValueKey('param ${p['name']}'),
            model: model,
            parameter: p,
          ),
        if (doc.parameters.length > 1) ...[
          const SizedBox(height: 8),
          Text(
            'Individual: each participant alone, with the prior. Pooled: '
            'participants vary around a group mean, estimated with them '
            '(shrinkage). Shared: one value for everyone (NUTS only).',
            style: theme.textTheme.bodySmall,
          ),
        ],
      ],
    );
  }
}

class _ParameterTile extends StatelessWidget {
  const _ParameterTile({
    super.key,
    required this.model,
    required this.parameter,
  });

  final BuilderModel model;
  final Map<String, Object?> parameter;

  String get name => parameter['name']! as String;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final doc = model.document;
    final role = doc.group.roleOf(name);
    Prior? prior;
    try {
      prior = _codec.decodePrior(parameter['prior']! as Map<String, Object?>);
    } on Object {
      prior = null;
    }
    final targets = targetsOf(parameter);
    return Padding(
      padding: const EdgeInsets.only(top: 10),
      child: DecoratedBox(
        decoration: BoxDecoration(
          border: Border(
            left: BorderSide(
              color: role == ParameterRole.individual
                  ? theme.colorScheme.outlineVariant
                  : const Color(0xFF8E6AC8),
              width: 3,
            ),
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.only(left: 8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Expanded(
                    child: CommitField(
                      key: ValueKey('name $name'),
                      label: 'Name',
                      value: name,
                      validate: (v) => v.trim().isEmpty
                          ? 'Give it a name'
                          : v != name && doc.parameter(v.trim()) != null
                          ? 'Taken'
                          : null,
                      onCommit: (v) => model.update(
                        doc.replaceParameter(name, {
                          ...parameter,
                          'name': v.trim(),
                        }),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  SizedBox(
                    width: 120,
                    child: Pick<ParameterRole>(
                      label: 'Across participants',
                      value: role,
                      items: const {
                        ParameterRole.individual: 'individual',
                        ParameterRole.pooled: 'pooled',
                        ParameterRole.shared: 'shared',
                      },
                      onChanged: (r) {
                        if (r != null && r != role) {
                          model.update(doc.withRole(name, r));
                        }
                      },
                    ),
                  ),
                  IconButton(
                    tooltip: 'Fix it again (not estimated)',
                    icon: const Icon(Icons.close, size: 18),
                    onPressed: () => model.update(doc.withoutParameter(name)),
                  ),
                ],
              ),
              const SizedBox(height: 4),
              Text(
                'Sets ${targets.map(describeTarget).join(', ')}',
                style: theme.textTheme.bodySmall,
              ),
              if (prior == null)
                Text(
                  'A custom prior: edit it in the saved JSON.',
                  style: theme.textTheme.bodySmall,
                )
              else
                PriorEditor(
                  prior: prior,
                  onChanged: (p) => model.update(doc.withPrior(name, p)),
                ),
              if (role == ParameterRole.pooled)
                _PopulationEditor(model: model, name: name),
            ],
          ),
        ),
      ),
    );
  }
}

/// A prior's type and numbers.
class PriorEditor extends StatelessWidget {
  const PriorEditor({super.key, required this.prior, required this.onChanged});

  final Prior prior;
  final ValueChanged<Prior> onChanged;

  static const _types = {
    'Normal': 'normal',
    'TruncatedNormal': 'normal, truncated',
    'LogNormal': 'log-normal (positive)',
    'HalfNormal': 'half-normal (positive)',
    'Gamma': 'gamma (positive)',
    'Uniform': 'uniform',
  };

  /// A prior of [type] about where [prior] is.
  static Prior convert(Prior prior, String type) {
    final t = prior.typical;
    final s = math.max(1.0, t.abs() / 2);
    final pos = t > 0 ? t : 1.0;
    return switch (type) {
      'Normal' => Normal(t, s),
      'TruncatedNormal' => TruncatedNormal(
        t,
        s,
        lower: t - 4 * s,
        upper: t + 4 * s,
      ),
      'LogNormal' => LogNormal(math.log(pos), 1),
      'HalfNormal' => HalfNormal(pos / 0.6744897501960817),
      'Gamma' => GammaPrior(2, 2 / pos),
      _ => Uniform(t - 2 * s, t + 2 * s),
    };
  }

  @override
  Widget build(BuildContext context) {
    final json = prior.toJson();
    final type = json['type'] as String;
    double v(String k) => _num(json[k]);
    void set(String key, double? x) {
      if (x == null) return;
      try {
        onChanged(_codec.decodePrior({...json, key: x}));
      } on Object {
        // an invalid prior (e.g. sd ≤ 0): keep the old one
      }
    }

    Widget field(String key, String label) => Expanded(
      child: NumberField(
        key: ValueKey('$key ${json[key]}'),
        label: label,
        value: v(key),
        onSubmitted: (x) => set(key, x),
      ),
    );
    final fields = switch (type) {
      'Normal' => [field('mean', 'mean'), field('sd', 'sd')],
      'TruncatedNormal' => [
        field('mean', 'mean'),
        field('sd', 'sd'),
        field('lower', 'from'),
        field('upper', 'to'),
      ],
      'LogNormal' => [
        Expanded(
          child: NumberField(
            key: ValueKey('median ${json['mu']}'),
            label: 'median',
            value: math.exp(v('mu')),
            onSubmitted: (x) {
              if (x != null && x > 0) set('mu', math.log(x));
            },
          ),
        ),
        field('sigma', 'σ (log scale)'),
      ],
      'HalfNormal' => [field('sd', 'sd')],
      'Gamma' => [field('shape', 'shape'), field('rate', 'rate')],
      'Uniform' => [field('lower', 'from'), field('upper', 'to')],
      _ => <Widget>[],
    };
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Pick<String>(
          label: 'Prior',
          value: _types.containsKey(type) ? type : null,
          hint: '$prior',
          items: _types,
          onChanged: (t) {
            if (t != null && t != type) onChanged(convert(prior, t));
          },
        ),
        if (fields.isNotEmpty)
          Row(
            children: [
              for (final (i, f) in fields.indexed) ...[
                if (i > 0) const SizedBox(width: 6),
                f,
              ],
            ],
          ),
        if (_inWords(prior) case final w?)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(w, style: Theme.of(context).textTheme.bodySmall),
          ),
      ],
    );
  }

  static String? _inWords(Prior p) => switch (p) {
    LogNormal(:final mu, :final sigma) =>
      'median ${fitNumber(math.exp(mu))}, 95 % within '
          '${fitNumber(math.exp(mu - 1.96 * sigma))}–'
          '${fitNumber(math.exp(mu + 1.96 * sigma))}',
    Normal(:final mean, :final sd) =>
      '95 % within ${fitNumber(mean - 1.96 * sd)} to '
          '${fitNumber(mean + 1.96 * sd)}',
    _ => null,
  };
}

/// A pooled parameter's population: the prior on its mean (normal) and on
/// its sd (half-normal), both on the parameter's unconstrained scale.
class _PopulationEditor extends StatelessWidget {
  const _PopulationEditor({required this.model, required this.name});

  final BuilderModel model;
  final String name;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final doc = model.document;
    final pop = doc.group.pooled[name]!;
    final p = doc.parameter(name)!;
    final scale = switch ((p['transform'] as Map<String, Object?>?)?['type']) {
      'LowerBound' => 'log',
      'UpperBound' => 'log (mirrored)',
      'Interval' => 'logit',
      _ => 'natural',
    };
    void set({Map<String, Object?>? mean, Map<String, Object?>? sd}) =>
        model.update(
          doc.withGroup(
            DocGroup(
              pooled: {
                ...doc.group.pooled,
                name: DocPopulation(mean: mean ?? pop.mean, sd: sd ?? pop.sd),
              },
              shared: doc.group.shared,
              centered: doc.group.centered,
            ),
          ),
        );
    final normalMean = pop.mean['type'] == 'Normal';
    final halfSd = pop.sd['type'] == 'HalfNormal';
    return Padding(
      padding: const EdgeInsets.only(top: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            'Population, on the $scale scale: participants ~ N(μ, σ²)',
            style: theme.textTheme.labelSmall,
          ),
          if (normalMean && halfSd)
            Row(
              children: [
                Expanded(
                  child: NumberField(
                    key: ValueKey('mu ${pop.mean}'),
                    label: 'μ about',
                    value: _num(pop.mean['mean']),
                    onSubmitted: (x) {
                      if (x != null) set(mean: {...pop.mean, 'mean': x});
                    },
                  ),
                ),
                const SizedBox(width: 6),
                Expanded(
                  child: NumberField(
                    key: ValueKey('mu sd ${pop.mean}'),
                    label: '± (sd)',
                    value: _num(pop.mean['sd']),
                    onSubmitted: (x) {
                      if (x != null && x > 0) {
                        set(mean: {...pop.mean, 'sd': x});
                      }
                    },
                  ),
                ),
                const SizedBox(width: 6),
                Expanded(
                  child: NumberField(
                    key: ValueKey('sigma ${pop.sd}'),
                    label: 'σ scale',
                    value: _num(pop.sd['sd']),
                    onSubmitted: (x) {
                      if (x != null && x > 0) set(sd: {...pop.sd, 'sd': x});
                    },
                  ),
                ),
              ],
            )
          else
            Text(
              'μ ~ ${pop.mean['type']}, σ ~ ${pop.sd['type']}: edit them in '
              'the saved JSON.',
              style: theme.textTheme.bodySmall,
            ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// Method
// -----------------------------------------------------------------------------

class _MethodCard extends StatelessWidget {
  const _MethodCard({required this.model});

  final BuilderModel model;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final doc = model.document;
    final f = doc.fit;
    final fitting = model.fitting;
    final grouped = !doc.group.isEmpty;
    final nuts = f.method == 'nuts';
    void set(DocFit next) => model.update(doc.copyWith(fit: next));
    DocFit copy({
      String? method,
      int? restarts,
      int? chains,
      int? warmup,
      int? draws,
      double? targetAccept,
    }) => DocFit(
      method: method ?? f.method,
      restarts: restarts ?? f.restarts,
      seed: f.seed,
      chains: chains ?? f.chains,
      warmup: warmup ?? f.warmup,
      draws: draws ?? f.draws,
      targetAccept: targetAccept ?? f.targetAccept,
      maxTreeDepth: f.maxTreeDepth,
      blockSize: f.blockSize,
      gradientMode: f.gradientMode,
      reverseAbove: f.reverseAbove,
    );
    final n = model.csv == null
        ? model.simParticipants
        : math.max(1, model.participants.length);
    final what = switch ((nuts, grouped)) {
      (false, false) =>
        'The mode of each participant\'s posterior (L-BFGS from '
            '${f.restarts + 1} starts), with the Laplace approximation for '
            'intervals.',
      (false, true) =>
        'Empirical Bayes: the populations\' means and sds that make the '
            'data most likely, with each participant\'s mode under them '
            '(shrunk towards the group). Point estimates of the group.',
      (true, false) =>
        'Draws from each participant\'s posterior (NUTS, ${f.chains} '
            'chains), started at the mode.',
      (true, true) =>
        'Draws from the joint posterior of the group and every '
            'participant (NUTS, ${f.chains} chains): uncertainty at both '
            'levels.',
    };
    return SectionCard(
      title: 'Fit',
      children: [
        SegmentedButton<String>(
          showSelectedIcon: false,
          segments: const [
            ButtonSegment(value: 'map', label: Text('MAP')),
            ButtonSegment(value: 'nuts', label: Text('NUTS')),
          ],
          selected: {f.method == 'nuts' ? 'nuts' : 'map'},
          onSelectionChanged: (s) => set(copy(method: s.single)),
        ),
        const SizedBox(height: 6),
        Text(what, style: theme.textTheme.bodySmall),
        Row(
          children: [
            Expanded(
              child: Pick<int>(
                label: 'Extra starts',
                value: f.restarts,
                items: {for (final r in FitConfig.restartChoices) r: '$r'},
                onChanged: (r) {
                  if (r != null) set(copy(restarts: r));
                },
              ),
            ),
            if (nuts) ...[
              const SizedBox(width: 6),
              Expanded(
                child: Pick<int>(
                  label: 'Chains',
                  value: f.chains,
                  items: {for (final c in FitConfig.chainChoices) c: '$c'},
                  onChanged: (c) {
                    if (c != null) set(copy(chains: c));
                  },
                ),
              ),
            ],
          ],
        ),
        if (nuts)
          Row(
            children: [
              Expanded(
                child: Pick<int>(
                  label: 'Warm-up',
                  value: f.warmup,
                  items: {for (final c in FitConfig.iterationChoices) c: '$c'},
                  onChanged: (c) {
                    if (c != null) set(copy(warmup: c));
                  },
                ),
              ),
              const SizedBox(width: 6),
              Expanded(
                child: Pick<int>(
                  label: 'Draws',
                  value: f.draws,
                  items: {for (final c in FitConfig.iterationChoices) c: '$c'},
                  onChanged: (c) {
                    if (c != null) set(copy(draws: c));
                  },
                ),
              ),
              const SizedBox(width: 6),
              Expanded(
                child: Pick<double>(
                  label: 'Target accept',
                  value: f.targetAccept,
                  items: {
                    for (final a in FitConfig.targetAcceptChoices) a: '$a',
                  },
                  onChanged: (a) {
                    if (a != null) set(copy(targetAccept: a));
                  },
                ),
              ),
            ],
          ),
        if (nuts && grouped)
          SwitchListTile(
            dense: true,
            contentPadding: EdgeInsets.zero,
            title: const Text('Centred parametrisation'),
            subtitle: const Text(
              'Better when every participant is well identified; the '
              'default (non-centred) avoids the funnel when they aren\'t.',
            ),
            value: doc.group.centered,
            onChanged: (on) => model.update(
              doc.withGroup(
                DocGroup(
                  pooled: doc.group.pooled,
                  shared: doc.group.shared,
                  centered: on,
                ),
              ),
            ),
          ),
        if (model.csv == null) ...[
          const SizedBox(height: 4),
          Pick<int>(
            label: 'Simulated participants',
            value: model.simParticipants,
            items: {
              for (final k in const [1, 2, 4, 6, 10, 20, 50]) k: '$k',
              if (!const [1, 2, 4, 6, 10, 20, 50].contains(n)) n: '$n',
            },
            onChanged: (k) {
              if (k != null) model.setSimParticipants(k);
            },
          ),
          if (model.simParticipants > 1)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                'Each with inputs of its own and the free parameters drawn '
                'around their set values: the results show the true values '
                'too (a recovery check).',
                style: theme.textTheme.bodySmall,
              ),
            ),
        ],
        const SizedBox(height: 10),
        if (fitting == null)
          Text(
            'Fitting is not available here.',
            style: theme.textTheme.bodySmall,
          )
        else ...[
          if (fitting.running) ...[
            LinearProgressIndicator(value: fitting.progress),
            const SizedBox(height: 4),
            Text(fitting.stage ?? '', style: theme.textTheme.bodySmall),
          ] else if (fitting.unavailable case final why?)
            Text(
              why,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          if (fitting.error case final e?)
            Text(
              e,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.error,
              ),
            ),
          const SizedBox(height: 6),
          Align(
            alignment: Alignment.centerLeft,
            child: fitting.running
                ? OutlinedButton.icon(
                    onPressed: fitting.cancel,
                    icon: const Icon(Icons.stop, size: 18),
                    label: const Text('Cancel'),
                  )
                : FilledButton.icon(
                    onPressed: fitting.unavailable == null ? fitting.fit : null,
                    icon: const Icon(Icons.play_arrow, size: 18),
                    label: Text(
                      n > 1 ? 'Fit $n participants' : 'Fit the participant',
                    ),
                  ),
          ),
        ],
      ],
    );
  }
}

// -----------------------------------------------------------------------------
// Results
// -----------------------------------------------------------------------------

class _ResultsCard extends StatelessWidget {
  const _ResultsCard({required this.model});

  final BuilderModel model;

  Future<void> _save(BuildContext context) async {
    final text = model.fitting?.estimatesCsv();
    if (text == null) return;
    final base = model.name.replaceAll(RegExp(r'[^A-Za-z0-9_-]+'), '_');
    await saveText(context, '${base}_estimates.csv', text);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final f = model.fitting!;
    final r = f.result!;
    final truth = model.simulatedTruth;
    final seconds = (r.millis / 1000).toStringAsFixed(1);
    return SectionCard(
      title: 'Results',
      action: IconButton(
        tooltip: 'Clear the results',
        icon: const Icon(Icons.clear_all, size: 20),
        onPressed: f.clear,
      ),
      children: [
        Text(
          '${r.method[0].toUpperCase()}${r.method.substring(1)}: '
          '${r.participants.length} '
          '${r.participants.length == 1 ? 'participant' : 'participants'} in '
          '$seconds s on ${r.where}'
          '${r.logEvidence == null ? '' : '; log evidence ${r.logEvidence!.toStringAsFixed(1)}'}'
          '${r.converged ? '' : '; not converged'}.',
          style: theme.textTheme.bodySmall,
        ),
        for (final note in [?r.note, ?r.deviceNote])
          Text(
            note,
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.error,
            ),
          ),
        if (f.stale)
          Text(
            'The model or the data changed since: fit again.',
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.error,
            ),
          ),
        SwitchListTile(
          dense: true,
          contentPadding: EdgeInsets.zero,
          title: const Text('Preview at the estimates'),
          value: f.previewAtEstimates,
          onChanged: f.setPreviewAtEstimates,
        ),
        for (final (i, name) in r.names.indexed) ...[
          const SizedBox(height: 8),
          Text(_groupLine(r, name), style: theme.textTheme.labelMedium),
          ForestPlot(
            result: r,
            parameter: i,
            truth: [for (final id in r.participants) truth[id]?[i]],
            selected: model.participants.isEmpty ? 0 : model.participant,
            onTap: (k) {
              if (model.participants.length > 1) model.showParticipant(k);
            },
          ),
        ],
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          children: [
            OutlinedButton.icon(
              onPressed: () => _save(context),
              icon: const Icon(Icons.table_chart_outlined, size: 18),
              label: const Text('Save estimates (CSV)'),
            ),
          ],
        ),
      ],
    );
  }

  /// The parameter's name and its group-level estimates.
  static String _groupLine(DocumentFitOutcome r, String name) {
    String iv(GroupEstimate g) => g.lower.isFinite
        ? ' [${fitNumber(g.lower)}, ${fitNumber(g.upper)}]'
        : '';
    final parts = <String>[];
    for (final g in r.group) {
      if (g.name != name) continue;
      parts.add(switch (g.kind) {
        GroupRow.typical => 'typical ${fitNumber(g.value)}${iv(g)}',
        GroupRow.spread => 'spread σ ${fitNumber(g.value)}${iv(g)}',
        GroupRow.shared => 'shared ${fitNumber(g.value)}${iv(g)}',
      });
    }
    return parts.isEmpty ? name : '$name · ${parts.join(', ')}';
  }
}

/// One parameter across participants: each participant's estimate (dot)
/// and 95 % interval (line), a simulated participant's true value (ring),
/// and the group's typical value (dashed). Tap a row to preview that
/// participant.
class ForestPlot extends StatelessWidget {
  const ForestPlot({
    super.key,
    required this.result,
    required this.parameter,
    required this.truth,
    required this.selected,
    required this.onTap,
  });

  final DocumentFitOutcome result;
  final int parameter;
  final List<double?> truth;
  final int selected;
  final ValueChanged<int> onTap;

  static const rowHeight = 12.0, maxRows = 60;

  @override
  Widget build(BuildContext context) {
    final n = math.min(result.participants.length, maxRows);
    final height = 18 + n * rowHeight;
    final scheme = Theme.of(context).colorScheme;
    final text = Theme.of(context).textTheme.labelSmall!;
    return GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTapUp: (d) {
        final k = ((d.localPosition.dy - 4) / rowHeight).floor();
        if (k >= 0 && k < n) onTap(k);
      },
      child: SizedBox(
        height: height,
        child: CustomPaint(
          painter: _ForestPainter(
            result: result,
            i: parameter,
            truth: truth,
            selected: selected,
            n: n,
            scheme: scheme,
            text: text,
          ),
        ),
      ),
    );
  }
}

class _ForestPainter extends CustomPainter {
  _ForestPainter({
    required this.result,
    required this.i,
    required this.truth,
    required this.selected,
    required this.n,
    required this.scheme,
    required this.text,
  });

  final DocumentFitOutcome result;
  final int i, selected, n;
  final List<double?> truth;
  final ColorScheme scheme;
  final TextStyle text;

  static const _label = 34.0;

  @override
  void paint(Canvas canvas, Size size) {
    final r = result;
    final values = <double>[
      for (var k = 0; k < n; k++) ...[
        r.estimate[k][i],
        r.lower[k][i],
        r.upper[k][i],
        ?truth[k],
      ],
      for (final g in r.group)
        if (g.name == r.names[i] && g.kind != GroupRow.spread) g.value,
    ].where((v) => v.isFinite).toList();
    if (values.isEmpty) return;
    var lo = values.reduce(math.min), hi = values.reduce(math.max);
    final log = lo > 0 && hi / lo > 20;
    double f(double v) => log ? math.log(v) : v;
    var a = f(lo), b = f(hi);
    if (b - a < 1e-9) {
      a -= 0.5;
      b += 0.5;
    }
    final pad = (b - a) * 0.05;
    a -= pad;
    b += pad;
    final w = size.width - _label - 4;
    double x(double v) => _label + (f(v) - a) / (b - a) * w;
    final muted = scheme.onSurfaceVariant;
    final line = Paint()
      ..color = scheme.primary
      ..strokeWidth = 1.4;
    final dot = Paint()..color = scheme.primary;
    final ring = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2
      ..color = scheme.tertiary;

    for (final g in r.group) {
      if (g.name != r.names[i] || g.kind == GroupRow.spread) continue;
      if (!g.value.isFinite) continue;
      final gx = x(g.value);
      for (var y = 2.0; y < 4 + n * ForestPlot.rowHeight; y += 6) {
        canvas.drawLine(
          Offset(gx, y),
          Offset(gx, y + 3),
          Paint()
            ..color = const Color(0xFF8E6AC8)
            ..strokeWidth = 1.2,
        );
      }
    }
    for (var k = 0; k < n; k++) {
      final y = 4 + (k + 0.5) * ForestPlot.rowHeight;
      if (k == selected) {
        canvas.drawRect(
          Rect.fromLTWH(
            0,
            y - ForestPlot.rowHeight / 2,
            size.width,
            ForestPlot.rowHeight,
          ),
          Paint()..color = scheme.primary.withValues(alpha: 0.08),
        );
      }
      _text(canvas, Offset(0, y - 6), r.participants[k], muted);
      final e = r.estimate[k][i], l = r.lower[k][i], u = r.upper[k][i];
      if (l.isFinite && u.isFinite) {
        canvas.drawLine(Offset(x(l), y), Offset(x(u), y), line);
      }
      if (e.isFinite) canvas.drawCircle(Offset(x(e), y), 2.6, dot);
      if (truth[k] case final t? when t.isFinite) {
        canvas.drawCircle(Offset(x(t), y), 3.6, ring);
      }
    }
    final axisY = 4 + n * ForestPlot.rowHeight + 1;
    String lbl(double v) => fitNumber(log ? math.exp(v) : v);
    _text(canvas, Offset(_label, axisY), lbl(a), muted);
    final hiText = lbl(b);
    _text(
      canvas,
      Offset(size.width - 6.0 * hiText.length - 2, axisY),
      hiText,
      muted,
    );
    if (log) {
      _text(canvas, Offset(_label + w / 2 - 12, axisY), 'log', muted);
    }
  }

  void _text(Canvas canvas, Offset at, String s, Color color) {
    final tp = TextPainter(
      text: TextSpan(
        text: s,
        style: text.copyWith(color: color, fontSize: 10),
      ),
      textDirection: TextDirection.ltr,
      maxLines: 1,
      ellipsis: '…',
    )..layout(maxWidth: _label - 2 > 0 && at.dx == 0 ? _label - 2 : 80);
    tp.paint(canvas, at);
  }

  @override
  bool shouldRepaint(_ForestPainter old) =>
      old.result != result ||
      old.i != i ||
      old.selected != selected ||
      old.scheme != scheme;
}

/// A number from prior JSON (`FitCodec` writes non-finite ones as text).
double _num(Object? v) => switch (v) {
  num() => v.toDouble(),
  'Infinity' => double.infinity,
  '-Infinity' => double.negativeInfinity,
  _ => double.nan,
};
