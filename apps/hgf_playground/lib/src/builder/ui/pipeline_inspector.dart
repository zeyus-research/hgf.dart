import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:material_ui/material_ui.dart';

import '../../ui/files.dart';
import '../../ui/panels.dart' show SectionCard;
import '../bids_source.dart';
import '../builder_model.dart';
import 'bids_dialog.dart';
import '../doc_layout.dart' show ownerIdOf;
import 'fit_panel.dart' show FitPanel, FreeToggle;
import 'inspector.dart' show NumberField;
import 'pipeline_canvas.dart' show readoutName;

/// The inspector of the pipeline canvas: the part selected (data,
/// transform, input, network, response or output), or an overview.
class PipelinePanel extends StatelessWidget {
  const PipelinePanel({super.key, required this.model});

  final BuilderModel model;

  @override
  Widget build(BuildContext context) {
    final part = model.selectedPart;
    final doc = model.document;
    if (part == null) return _Overview(model: model);
    if (part == DocPart.data) return _DataPanel(model: model);
    if (part == DocPart.network) return _NetworkSummary(model: model);
    if (part == DocPart.fit) return FitPanel(model: model);
    if (part == DocPart.response && doc.response != null) {
      return _ResponsePanel(model: model);
    }
    final colon = part.indexOf(':');
    final key = colon < 0 ? '' : part.substring(colon + 1);
    if (part.startsWith('transform:') && doc.transform(key) != null) {
      return _TransformPanel(model: model, name: key);
    }
    if (part.startsWith('input:') && doc.input(key) != null) {
      return _InputPanel(model: model, name: key);
    }
    if (part.startsWith('output:')) {
      for (final o in doc.outputs) {
        if (o.name == key) return _OutputPanel(model: model, output: o);
      }
    }
    return _Overview(model: model);
  }
}

class PipelineHelpCard extends StatelessWidget {
  const PipelineHelpCard({super.key});

  @override
  Widget build(BuildContext context) => SectionCard(
    title: 'How to',
    children: [
      Text(
        'Data flows left to right. Drag a column\'s port (right of the data '
        'card) onto one of the network\'s inputs to read it there, onto the '
        'response to record responses from it, or onto the background to '
        'make a transform of it. Drag the network\'s beliefs port onto the '
        'background to add an output (a node\'s field per time step, e.g. a '
        'prediction error for EEG). Double-click the network to edit it; '
        'right-click for more. Without loaded data the preview simulates '
        'each input and, with a response, the responses.',
        style: Theme.of(context).textTheme.bodySmall,
      ),
    ],
  );
}

// -----------------------------------------------------------------------------
// Overview and network
// -----------------------------------------------------------------------------

class _Overview extends StatelessWidget {
  const _Overview({required this.model});

  final BuilderModel model;

  @override
  Widget build(BuildContext context) {
    final doc = model.document;
    final theme = Theme.of(context);
    String count(int n, String what) => '$n $what${n == 1 ? '' : 's'}';
    return SectionCard(
      title: 'Model',
      children: [
        CommitField(
          key: ValueKey('name ${model.name}'),
          label: 'Name',
          value: model.name,
          onCommit: model.rename,
        ),
        const SizedBox(height: 8),
        Text(
          [
            count(doc.inputs.length, 'input'),
            count(doc.transforms.length, 'transform'),
            doc.response == null ? 'no response' : 'a response',
            count(doc.outputs.length, 'output'),
          ].join(' · '),
          style: theme.textTheme.bodySmall,
        ),
        const SizedBox(height: 10),
        Wrap(
          spacing: 6,
          runSpacing: 6,
          children: [
            ActionChip(
              avatar: const Icon(Icons.functions, size: 16),
              label: const Text('Transform'),
              onPressed: () => model.addTransform(),
            ),
            ActionChip(
              avatar: const Icon(Icons.show_chart, size: 16),
              label: const Text('Output'),
              onPressed: () => model.addOutput(),
            ),
            if (doc.response == null)
              ActionChip(
                avatar: const Icon(Icons.touch_app_outlined, size: 16),
                label: const Text('Response'),
                onPressed: () => model.addResponse(),
              ),
            ActionChip(
              avatar: const Icon(Icons.account_tree_outlined, size: 16),
              label: const Text('Edit the network'),
              onPressed: () => model.showView(BuilderView.network),
            ),
          ],
        ),
        const SizedBox(height: 8),
        _DataButtons(model: model),
      ],
    );
  }
}

class _NetworkSummary extends StatelessWidget {
  const _NetworkSummary({required this.model});

  final BuilderModel model;

  @override
  Widget build(BuildContext context) {
    final doc = model.doc;
    final theme = Theme.of(context);
    return SectionCard(
      title: 'HGF network',
      children: [
        Text(
          '${doc.nodes.length} nodes, ${doc.edges.length} edges; '
          '${doc.volatilityUpdates.pyhgfName} volatility updates.',
          style: theme.textTheme.bodySmall,
        ),
        const SizedBox(height: 8),
        for (final i in model.document.inputs)
          ListTile(
            dense: true,
            contentPadding: EdgeInsets.zero,
            visualDensity: VisualDensity.compact,
            leading: const Icon(Icons.input, size: 18),
            title: Text(i.name),
            subtitle: Text(
              '${prettyId(i.node)} reads '
              '${(i.columns ?? [i.name]).join(', ')}',
            ),
            onTap: () => model.select(PartSelection(DocPart.input(i.name))),
          ),
        const SizedBox(height: 8),
        FilledButton.tonalIcon(
          onPressed: () => model.showView(BuilderView.network),
          icon: const Icon(Icons.open_in_full, size: 18),
          label: const Text('Edit the network'),
        ),
      ],
    );
  }
}

// -----------------------------------------------------------------------------
// Data
// -----------------------------------------------------------------------------

class _DataButtons extends StatelessWidget {
  const _DataButtons({required this.model});

  final BuilderModel model;

  Future<void> _load(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    try {
      final picked = await pickText(
        'Load data (CSV, one row per time step)',
        extensions: const ['csv', 'tsv', 'txt'],
      );
      if (picked == null) return;
      final (text, name) = picked;
      model.loadData(CsvData.parse(text), name: name);
    } on Object catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Could not load: $e')));
    }
  }

  Future<void> _loadBids(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    try {
      final source = await pickBids();
      if (source == null || !context.mounted) return;
      if (source.events.isEmpty) {
        messenger.showSnackBar(
          const SnackBar(
            content: Text('No event tables (sub-*/…/*_events.tsv) found'),
          ),
        );
        return;
      }
      final participants = await source.participants();
      if (!context.mounted) return;
      final choice = await showBidsDialog(
        context,
        name: source.name,
        events: source.events,
        participants: participants,
      );
      if (choice == null) return;
      messenger.showSnackBar(
        SnackBar(
          content: Text('Reading ${choice.participants.length} participants…'),
        ),
      );
      final merged = await source.load(
        participants: participants,
        include: (sub, _) => choice.participants.contains(sub),
      );
      messenger.hideCurrentSnackBar();
      final g = choice.groupColumn;
      final groups = g == null || participants == null
          ? const <String>[]
          : {
              for (final s in merged.participants)
                for (final row in participants.rows)
                  if (row[participants.header.indexOf('participant_id')]
                          .trim() ==
                      s)
                    row[participants.header.indexOf(g)].trim(),
            }.toList();
      model.loadBids(
        merged,
        name:
            '${source.name}'
            '${groups.isEmpty || groups.length > 3 ? '' : ' (${groups.join(', ')})'}',
        sessionHint: g,
      );
    } on Object catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Could not load: $e')));
    }
  }

  Future<void> _save(BuildContext context, String? text, String what) async {
    if (text == null) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Fix the errors first')));
      return;
    }
    final base = model.name.replaceAll(RegExp(r'[^A-Za-z0-9_-]+'), '_');
    await saveText(context, '${base}_$what.csv', text);
  }

  @override
  Widget build(BuildContext context) => Wrap(
    spacing: 6,
    runSpacing: 6,
    children: [
      OutlinedButton.icon(
        onPressed: () => _load(context),
        icon: const Icon(Icons.file_open_outlined, size: 18),
        label: const Text('Load CSV…'),
      ),
      OutlinedButton.icon(
        onPressed: () => _loadBids(context),
        icon: const Icon(Icons.folder_open_outlined, size: 18),
        label: const Text('Load BIDS…'),
      ),
      if (model.csv != null)
        OutlinedButton.icon(
          onPressed: () => model.loadData(null),
          icon: const Icon(Icons.casino_outlined, size: 18),
          label: const Text('Simulate instead'),
        ),
      if (model.csv == null)
        OutlinedButton.icon(
          onPressed: model.compiled == null
              ? null
              : () => _save(context, model.dataCsv(), 'data'),
          icon: const Icon(Icons.download_outlined, size: 18),
          label: const Text('Save simulated data'),
        ),
      OutlinedButton.icon(
        onPressed: model.compiled == null || model.document.outputs.isEmpty
            ? null
            : () => _save(context, model.outputsCsv(), 'outputs'),
        icon: const Icon(Icons.table_view_outlined, size: 18),
        label: const Text('Save with outputs'),
      ),
    ],
  );
}

class _DataPanel extends StatelessWidget {
  const _DataPanel({required this.model});

  final BuilderModel model;

  @override
  Widget build(BuildContext context) {
    final doc = model.document;
    final d = doc.data;
    final theme = Theme.of(context);
    void set({
      String? Function()? participant,
      String? Function()? time,
      String? Function()? timeStep,
      String? Function()? group,
    }) => model.update(
      // copyWith keeps the clock.
      doc.copyWith(
        data: d.copyWith(
          participantColumn: participant,
          timeColumn: time,
          timeStepColumn: timeStep,
          groupColumn: group,
        ),
      ),
    );

    return SectionCard(
      title: 'Data',
      children: [
        Text(
          model.csv == null
              ? 'Simulated: one series per input (reversals for binary '
                    'inputs, random walks for continuous ones), and responses '
                    'drawn from the response model at its parameters\' '
                    'typical values.'
              : '${model.csvName ?? 'Loaded'}: ${model.csv!.rows.length} '
                    'rows, ${model.csv!.header.length} columns'
                    '${model.participants.length > 1 ? ', ${model.participants.length} participants' : ''}.',
          style: theme.textTheme.bodySmall,
        ),
        if (model.dataError case final e?)
          Padding(
            padding: const EdgeInsets.only(top: 6),
            child: Text(
              e,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.error,
              ),
            ),
          ),
        const SizedBox(height: 8),
        _DataButtons(model: model),
        if (model.participants.length > 1) ...[
          const SizedBox(height: 10),
          Pick<int>(
            label: 'Participant in the preview',
            value: model.participant,
            items: {for (final (k, p) in model.participants.indexed) k: p},
            onChanged: (k) => model.showParticipant(k!),
          ),
        ],
        if (model.csv == null) ...[
          const SizedBox(height: 10),
          Pick<int>(
            label: 'Simulated trials',
            value: model.data.nTrials,
            items: {
              for (final n in {100, 300, 1000, 3000, model.data.nTrials})
                n: '$n',
            },
            onChanged: (n) => model.setTrials(n!),
          ),
          if (model.simParticipants > 1) ...[
            Pick<int>(
              label: 'Team size',
              value: model.simTeamSize,
              items: {
                for (final n in {1, 2, 3, 4, 6, model.simTeamSize})
                  n: n == 1 ? 'alone' : '$n',
              },
              onChanged: (n) => model.setSimTeams(size: n),
            ),
            if (model.simTeamSize > 1)
              Pick<int>(
                label: 'Reshuffle teams every',
                value: model.simRegroupEvery,
                items: {
                  for (final n in {
                    0,
                    50,
                    100,
                    300,
                    1000,
                    model.simRegroupEvery,
                  })
                    n: n == 0 ? 'never' : '$n steps',
                },
                onChanged: (n) => model.setSimTeams(regroupEvery: n),
              ),
          ],
          if (model.simulatesTogether)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(
                'Simulated together: the inputs read responses, so on every '
                'step each participant responds, then the transforms see '
                'everyone\'s responses (team-mates\' presses) before the '
                'inputs are read.',
                style: theme.textTheme.bodySmall,
              ),
            ),
        ],
        const SizedBox(height: 12),
        Text('Layout', style: theme.textTheme.labelMedium),
        ColumnPick(
          model: model,
          label: 'Participant column',
          value: d.participantColumn,
          onChanged: (c) => set(participant: () => c),
        ),
        ColumnPick(
          model: model,
          label: 'Time column (time steps from its differences)',
          value: d.timeColumn,
          onChanged: (c) => set(time: () => c, timeStep: () => null),
        ),
        ColumnPick(
          model: model,
          label: 'Time-step column',
          value: d.timeStepColumn,
          onChanged: (c) => set(timeStep: () => c, time: () => null),
        ),
        ColumnPick(
          model: model,
          label: 'Group column (team-mates: mate, mates_sum, …)',
          value: d.groupColumn,
          onChanged: (c) => set(group: () => c),
        ),
        if (model.csv != null || d.clock != null) _ClockSection(model: model),
      ],
    );
  }
}

/// Events onto a shared clock ([DocClock]): the event time column, the
/// step, the session column, the streams (which events, what they hold)
/// and which grid rows to keep.
class _ClockSection extends StatelessWidget {
  const _ClockSection({required this.model});

  final BuilderModel model;

  @override
  Widget build(BuildContext context) {
    final doc = model.document;
    final d = doc.data;
    final k = d.clock;
    final theme = Theme.of(context);
    final raw = model.csv?.header ?? const <String>[];
    void setClock(DocClock? c, {String? time}) => model.update(
      doc.copyWith(
        data: d.copyWith(
          clock: () => c,
          timeColumn: time == null ? null : () => time,
          timeStepColumn: time == null ? null : () => null,
        ),
      ),
    );
    DocClock edit({
      String? onset,
      double? step,
      String? Function()? session,
      List<ClockStream>? streams,
      Expr? Function()? keep,
    }) => DocClock(
      onsetColumn: onset ?? k!.onsetColumn,
      step: step ?? k!.step,
      sessionColumn: session == null ? k!.sessionColumn : session(),
      streams: streams ?? k!.streams,
      keep: keep == null ? k!.keep : keep(),
    );
    String? exprError(String v) {
      if (v.trim().isEmpty) return null;
      try {
        Expr.parse(v);
        return null;
      } on FormatException catch (e) {
        return e.message;
      }
    }

    final children = <Widget>[
      const SizedBox(height: 12),
      SwitchListTile(
        contentPadding: EdgeInsets.zero,
        dense: true,
        title: const Text('Events onto a shared clock'),
        subtitle: const Text(
          'A row per event (presses, …) becomes a row per time step, each '
          'state held until its next event.',
        ),
        value: k != null,
        onChanged: (on) {
          if (!on) {
            setClock(null);
            return;
          }
          final onset =
              const [
                't_lsl_coord',
                'onset',
                'time',
                't',
              ].where(raw.contains).firstOrNull ??
              (raw.isEmpty ? 'onset' : raw.first);
          final session = model.sessionHint;
          setClock(
            DocClock(
              onsetColumn: onset,
              step: 0.1,
              sessionColumn: session != null && raw.contains(session)
                  ? session
                  : null,
              streams: [
                if (raw.contains('trial_type') && raw.contains('value'))
                  ClockStream.parse(
                    when: "trial_type == 'press'",
                    hold: {'press': 'value'},
                  )
                else
                  ClockStream(hold: const {}),
              ],
            ),
            time: d.timeColumn ?? doc.freshName('t', raw),
          );
        },
      ),
    ];
    if (k != null) {
      children.addAll([
        if (raw.isEmpty)
          CommitField(
            key: ValueKey('onset ${k.onsetColumn}'),
            label: 'Event time column',
            value: k.onsetColumn,
            onCommit: (v) => setClock(edit(onset: v.trim())),
          )
        else
          Pick<String>(
            label: 'Event time column (a clock everyone shares)',
            value: k.onsetColumn,
            items: {
              for (final c in {...raw, k.onsetColumn}) c: c,
            },
            onChanged: (c) => setClock(edit(onset: c)),
          ),
        CommitField(
          key: ValueKey('step ${k.step}'),
          label: 'Step (in the event time\'s units)',
          value: '${k.step}',
          validate: (v) {
            final x = double.tryParse(v.trim());
            return x == null || !(x > 0) ? 'A positive number' : null;
          },
          onCommit: (v) => setClock(edit(step: double.parse(v.trim()))),
        ),
        Pick<String?>(
          label: 'Session column (each session its own grid)',
          value: k.sessionColumn,
          items: {
            null: '—',
            for (final c in {...raw, ?k.sessionColumn}) c: c,
          },
          onChanged: (c) => setClock(edit(session: () => c)),
        ),
        for (final (i, st) in k.streams.indexed) ...[
          const SizedBox(height: 8),
          Row(
            children: [
              Text('Stream ${i + 1}', style: theme.textTheme.labelMedium),
              const Spacer(),
              IconButton(
                tooltip: 'Remove the stream',
                icon: const Icon(Icons.close, size: 18),
                onPressed: () =>
                    setClock(edit(streams: [...k.streams]..removeAt(i))),
              ),
            ],
          ),
          CommitField(
            key: ValueKey('when $i ${st.when}'),
            label: 'Events (empty: every row)',
            value: st.when == null ? '' : '${st.when}',
            hint: "trial_type == 'press'",
            monospace: true,
            validate: exprError,
            onCommit: (v) => setClock(
              edit(
                streams: [...k.streams]
                  ..[i] = ClockStream(
                    when: v.trim().isEmpty ? null : Expr.parse(v),
                    hold: st.hold,
                  ),
              ),
            ),
          ),
          CommitField(
            key: ValueKey('hold $i ${st.toJson()}'),
            label: 'Holds (name = expression; …)',
            value: [
              for (final MapEntry(:key, :value) in st.hold.entries)
                '$key = $value',
            ].join('; '),
            hint: 'press = value; team = team_id',
            monospace: true,
            validate: (v) {
              try {
                _parseHolds(v);
                return null;
              } on FormatException catch (e) {
                return e.message;
              }
            },
            onCommit: (v) => setClock(
              edit(
                streams: [...k.streams]
                  ..[i] = ClockStream(when: st.when, hold: _parseHolds(v)),
              ),
            ),
          ),
        ],
        Align(
          alignment: Alignment.centerLeft,
          child: TextButton.icon(
            onPressed: () => setClock(
              edit(
                streams: [
                  ...k.streams,
                  ClockStream(hold: const {}),
                ],
              ),
            ),
            icon: const Icon(Icons.add, size: 18),
            label: const Text('Add a stream'),
          ),
        ),
        CommitField(
          key: ValueKey('keep ${k.keep}'),
          label: 'Keep rows where (empty: all)',
          value: k.keep == null ? '' : '${k.keep}',
          hint: 'missing(press) == 0',
          monospace: true,
          validate: exprError,
          onCommit: (v) => setClock(
            edit(keep: () => v.trim().isEmpty ? null : Expr.parse(v)),
          ),
        ),
        const SizedBox(height: 6),
        Text(
          'The grid\'s times are whole multiples of the step on the event '
          'clock (e.g. LSL time), so outputs line up with other recordings. '
          'Group (team) cells come from a stream too; an empty one means '
          'no team-mates (e.g. rounds played alone). Saved or simulated data '
          'with the time column and no event time column is on the clock '
          'already.',
          style: theme.textTheme.bodySmall,
        ),
      ]);
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: children,
    );
  }
}

/// `name = expression; …` as a stream's held columns.
Map<String, Expr> _parseHolds(String text) {
  final out = <String, Expr>{};
  for (final part in text.split(';')) {
    if (part.trim().isEmpty) continue;
    final eq = part.indexOf('=');
    if (eq <= 0 || part.length > eq + 1 && part[eq + 1] == '=') {
      throw FormatException('Write name = expression, not "${part.trim()}"');
    }
    final name = part.substring(0, eq).trim();
    if (out.containsKey(name)) {
      throw FormatException('"$name" twice');
    }
    out[name] = Expr.parse(part.substring(eq + 1));
  }
  return out;
}

// -----------------------------------------------------------------------------
// Transform
// -----------------------------------------------------------------------------

/// Expressions binding a transform to team-mates' responses, by label:
/// how many pressed each option, the share, the first one's press.
Map<String, String> _mateTemplates(ModelDocument doc) {
  final r = doc.response;
  if (r == null || !r.readsData || r.dataColumns.length != 1) return const {};
  final col = Expr.parse(_quoted(r.dataColumns.single)).toString();
  final opts = r.options;
  if (opts == null) {
    return {
      'Team-mates\' mean $col': 'mates_mean($col)',
      'Team-mates\' sum of $col': 'mates_sum($col)',
      'Team-mate 1\'s $col': 'mate($col, 1)',
    };
  }
  return {
    for (final o in opts) ...{
      'How many team-mates pressed $o': "mates_sum($col == '$o')",
      'Share of team-mates pressing $o': "mates_mean($col == '$o')",
    },
    'How many team-mates responded': 'mates_count($col)',
  };
}

String _quoted(String name) =>
    RegExp(r'^[A-Za-z_][A-Za-z0-9_]*$').hasMatch(name) ? name : '`$name`';

class _TransformPanel extends StatelessWidget {
  const _TransformPanel({required this.model, required this.name});

  final BuilderModel model;
  final String name;

  @override
  Widget build(BuildContext context) {
    final doc = model.document;
    final t = doc.transform(name)!;
    final theme = Theme.of(context);
    final readers = [
      for (final i in doc.inputs)
        if ((i.columns ?? [i.name]).contains(name)) 'input ${i.name}',
      for (final o in doc.transforms)
        if (o.expression.variables.contains(name)) 'transform ${o.name}',
      if (doc.response?.dataColumns.contains(name) ?? false) 'the response',
    ];
    return SectionCard(
      title: 'Transform',
      action: IconButton(
        tooltip: 'Delete',
        icon: const Icon(Icons.delete_outline, size: 20),
        onPressed: () => model.removePart(DocPart.transform(name)),
      ),
      children: [
        CommitField(
          label: 'Column name',
          value: name,
          validate: (v) {
            if (v.trim().isEmpty) return 'Give it a name';
            if (v != name &&
                (doc.transform(v) != null ||
                    (model.csv != null && model.dataColumns.contains(v)))) {
              return 'That name is taken';
            }
            return null;
          },
          onCommit: (v) {
            model
              ..update(
                doc.replaceTransform(name, DocTransform.of(v, t.expression)),
              )
              ..select(PartSelection(DocPart.transform(v)));
          },
        ),
        CommitField(
          label: 'Expression',
          value: '${t.expression}',
          monospace: true,
          validate: (v) {
            try {
              Expr.parse(v);
              return null;
            } on FormatException catch (e) {
              return e.message;
            }
          },
          onCommit: (v) =>
              model.update(doc.replaceTransform(name, DocTransform(name, v))),
        ),
        const SizedBox(height: 6),
        Text(
          'Computed once per row before filtering, per participant. '
          'Columns by name (`with spaces` in backticks), arithmetic, '
          'comparisons (press == \'left\'), exp, log, sqrt, abs, tanh, '
          'sigmoid, min, max, clip, where; lag(x, k), diff(x), '
          'missing(x), fill(x, v); team-mates (the group column): mate(x, k) '
          '(the k-th other member), mates_sum(x), mates_mean(x), '
          'mates_count(x).',
          style: theme.textTheme.bodySmall,
        ),
        if (_mateTemplates(doc) case final templates when templates.isNotEmpty)
          Align(
            alignment: Alignment.centerLeft,
            child: PopupMenuButton<String>(
              tooltip: 'Bind to team-mates',
              itemBuilder: (_) => [
                for (final MapEntry(key: label, value: e) in templates.entries)
                  PopupMenuItem(value: e, child: Text(label)),
              ],
              onSelected: (e) => model.update(
                doc.replaceTransform(name, DocTransform(name, e)),
              ),
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.groups_outlined,
                      size: 18,
                      color: theme.colorScheme.primary,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      'From team-mates…',
                      style: theme.textTheme.labelLarge?.copyWith(
                        color: theme.colorScheme.primary,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        const SizedBox(height: 6),
        Text(
          readers.isEmpty
              ? 'Nothing reads it yet.'
              : 'Read by ${readers.join(', ')}.',
          style: theme.textTheme.bodySmall,
        ),
      ],
    );
  }
}

// -----------------------------------------------------------------------------
// Input
// -----------------------------------------------------------------------------

class _InputPanel extends StatelessWidget {
  const _InputPanel({required this.model, required this.name});

  final BuilderModel model;
  final String name;

  @override
  Widget build(BuildContext context) {
    final doc = model.document;
    final i = doc.input(name)!;
    final theme = Theme.of(context);
    final node = doc.network.ownerOf(i.node);
    var width = 1;
    if (model.network case final net?) {
      try {
        width = inputWidth(
          net,
          doc.network,
          doc.network.addresses.indexOf(i.node),
        );
      } on Object {
        width = 1;
      }
    }
    final cats = i.categories;
    final cols = i.columnsFor(width);
    void set({
      String? newName,
      List<String>? Function()? columns,
      List<String>? Function()? categories,
      String? Function()? observed,
    }) {
      final next = DocInput(
        newName ?? i.name,
        node: i.node,
        columns: columns == null ? i.columns : columns(),
        categories: categories == null ? i.categories : categories(),
        observedColumn: observed == null ? i.observedColumn : observed(),
      );
      model.update(doc.replaceInput(name, next));
      if (newName != null) model.select(PartSelection(DocPart.input(newName)));
    }

    return SectionCard(
      title: 'Input',
      children: [
        CommitField(
          label: 'Name',
          value: i.name,
          validate: (v) => v.trim().isEmpty
              ? 'Give it a name'
              : v != name && doc.input(v) != null
              ? 'Another input has that name'
              : null,
          onCommit: (v) => set(newName: v),
        ),
        const SizedBox(height: 6),
        Text(
          'Feeds ${prettyId(i.node)}'
          '${node == null ? '' : ' (${kindName(docKind(node))})'}'
          '${width > 1 ? ', $width columns wide' : ''}.',
          style: theme.textTheme.bodySmall,
        ),
        if (cats != null || width == 1)
          ColumnPick(
            model: model,
            label: 'Column',
            value: cols.single,
            allowNone: false,
            onChanged: (c) => set(columns: () => c == null ? null : [c]),
          )
        else
          for (var j = 0; j < width; j++)
            ColumnPick(
              model: model,
              label: 'Column ${j + 1} of $width',
              value: cols.length > j ? cols[j] : null,
              allowNone: false,
              onChanged: (c) {
                final next = [...cols];
                while (next.length < width) {
                  next.add('${i.name}_${next.length}');
                }
                next[j] = c!;
                set(columns: () => next);
              },
            ),
        if (node is CategoricalNode || width == 1)
          CommitField(
            label: node is CategoricalNode
                ? 'Category labels (comma-separated; empty: one-hot columns)'
                : 'Labels for 0, 1, … (comma-separated; empty: numbers)',
            value: cats?.join(', ') ?? '',
            validate: (v) {
              final labels = _labels(v);
              if (node is CategoricalNode &&
                  labels != null &&
                  labels.length != width) {
                return 'The node has $width categories';
              }
              return null;
            },
            onCommit: (v) => set(categories: () => _labels(v)),
          ),
        ColumnPick(
          model: model,
          label: 'Observed flags (0/1; empty cells otherwise)',
          value: i.observedColumn,
          onChanged: (c) => set(observed: () => c),
        ),
        const SizedBox(height: 4),
        Align(
          alignment: Alignment.centerLeft,
          child: TextButton(
            onPressed: i.columns == null
                ? null
                : () => model.removePart(DocPart.input(name)),
            child: Text('Read the column "${i.name}" (default)'),
          ),
        ),
      ],
    );
  }
}

List<String>? _labels(String v) {
  final parts = [
    for (final s in v.split(','))
      if (s.trim().isNotEmpty) s.trim(),
  ];
  return parts.isEmpty ? null : parts;
}

// -----------------------------------------------------------------------------
// Response
// -----------------------------------------------------------------------------

/// The readout types the builder offers.
const _readoutTypes = {
  'GaussianReadout': 'Gaussian (a continuous response)',
  'BinarySoftmaxReadout': 'Binary softmax (a binary input\'s prediction)',
  'CategoricalSoftmaxReadout': 'Categorical softmax',
  'SoftmaxReadout': 'Softmax over named options',
  'ExprReadout': 'Expression per option',
  'ObservationSurprise': 'None: fit to the observations',
};

/// The observation surprises a fit to the observations can score.
const _surpriseFunctions = {
  'TotalGaussianSurprise': 'Gaussian (every continuous node)',
  'FirstLevelGaussianSurprise': 'Gaussian (input nodes only)',
  'FirstLevelBinarySurprise': 'binary',
  'CategoricalSurprise': 'categorical',
  'DpSurprise': 'Dirichlet process',
};

class _ResponsePanel extends StatelessWidget {
  const _ResponsePanel({required this.model});

  final BuilderModel model;

  DocResponse get r => model.document.response!;

  void _set({
    String? name,
    Map<String, Object?>? readout,
    List<String>? Function()? columns,
    List<String>? Function()? options,
    Map<String, List<String>>? Function()? reveals,
  }) => model.update(
    model.document.withResponse(
      DocResponse(
        name ?? r.name,
        readout: readout ?? r.readout,
        columns: columns == null ? r.columns : columns(),
        options: options == null ? r.options : options(),
        reveals: reveals == null ? r.reveals : reveals(),
      ),
    ),
  );

  /// A fresh readout of [type], reading what the current one reads where
  /// it can.
  Map<String, Object?> _fresh(String type) {
    final ro = r.readout;
    final node =
        (ro['node'] as String?) ??
        (model.document.inputs.isEmpty
            ? model.doc.nodes.first.id
            : model.document.inputs.first.node);
    final beta = (ro['inverseTemperature'] as num?)?.toDouble() ?? 1.0;
    final k = r.options?.length ?? 2;
    return switch (type) {
      'ObservationSurprise' => {
        'type': type,
        'function': switch (model.doc.node(ownerIdOf(node))) {
          ScalarNode(kind: NodeKind.binaryState) => 'FirstLevelBinarySurprise',
          CategoricalNode() => 'CategoricalSurprise',
          DpStateNode() => 'DpSurprise',
          _ => 'TotalGaussianSurprise',
        },
      },
      'GaussianReadout' => {
        'type': type,
        'node': node,
        'field': 'expected_mean',
        'precision': 1.0,
      },
      'BinarySoftmaxReadout' || 'CategoricalSoftmaxReadout' => {
        'type': type,
        'node': node,
        'inverseTemperature': beta,
      },
      'SoftmaxReadout' => {
        'type': type,
        'options': [
          for (var j = 0; j < k; j++)
            [
              {'node': node, 'weight': j == 0 ? 1.0 : -1.0},
            ],
        ],
        'field': 'expected_mean',
        'inverseTemperature': beta,
      },
      _ => {
        'type': 'ExprReadout',
        'kind': 'choice',
        'options': [
          for (var j = 0; j < k; j++)
            j == 0 ? 'beta * $node.expected_mean' : '0',
        ],
        'parameters': {'beta': beta},
      },
    };
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final ro = r.readout;
    final type = ro['type'] as String;
    final addresses = _addresses(model.doc);
    final choice =
        type == 'SoftmaxReadout' ||
        (type == 'ExprReadout' && ro['kind'] == 'choice');
    final options = r.options;
    return SectionCard(
      title: 'Response',
      action: IconButton(
        tooltip: 'Delete',
        icon: const Icon(Icons.delete_outline, size: 20),
        onPressed: () => model.removePart(DocPart.response),
      ),
      children: [
        CommitField(
          label: 'Name',
          value: r.name,
          validate: (v) => v.trim().isEmpty ? 'Give it a name' : null,
          onCommit: (v) => _set(name: v),
        ),
        Pick<String>(
          label: 'Response model',
          value: _readoutTypes.containsKey(type) ? type : null,
          hint: readoutName(ro),
          items: _readoutTypes,
          onChanged: (t) {
            if (t != null && t != type) _set(readout: _fresh(t));
          },
        ),
        if (r.readsData)
          ColumnPick(
            model: model,
            label: 'Recorded responses',
            value: r.dataColumns.first,
            allowNone: false,
            onChanged: (c) => _set(columns: () => c == null ? null : [c]),
          ),
        ..._readoutFields(context, type, ro, addresses),
        if (model.responseParameters.isNotEmpty) ...[
          const SizedBox(height: 10),
          Text('Estimate in the fit', style: theme.textTheme.labelMedium),
          Wrap(
            spacing: 4,
            runSpacing: 4,
            children: [
              for (final MapEntry(:key, :value)
                  in model.responseParameters.entries)
                FreeToggle(
                  model: model,
                  target: responseTarget(key),
                  current: value,
                  label: key,
                ),
            ],
          ),
        ],
        if (choice || type == 'CategoricalSoftmaxReadout') ...[
          const SizedBox(height: 12),
          Text('Options', style: theme.textTheme.labelMedium),
          CommitField(
            label: 'Labels in the data (comma-separated; empty: 0, 1, …)',
            value: options?.join(', ') ?? '',
            onCommit: (v) => _set(options: () => _labels(v)),
          ),
          if (options != null && model.document.inputs.length > 1) ...[
            const SizedBox(height: 8),
            Text(
              'What a choice reveals (a task where only the chosen '
              'option\'s outcome is seen; none: every input is seen)',
              style: theme.textTheme.bodySmall,
            ),
            for (final o in options)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Wrap(
                  spacing: 4,
                  runSpacing: 4,
                  crossAxisAlignment: WrapCrossAlignment.center,
                  children: [
                    SizedBox(width: 70, child: Text(o)),
                    for (final i in model.document.inputs)
                      FilterChip(
                        label: Text(i.name),
                        visualDensity: VisualDensity.compact,
                        selected: r.reveals?[o]?.contains(i.name) ?? false,
                        onSelected: (on) {
                          final rv = {
                            for (final p in options) p: [...?r.reveals?[p]],
                          };
                          on ? rv[o]!.add(i.name) : rv[o]!.remove(i.name);
                          final any = rv.values.any((l) => l.isNotEmpty);
                          _set(reveals: () => any ? rv : null);
                        },
                      ),
                  ],
                ),
              ),
          ],
        ],
      ],
    );
  }

  List<Widget> _readoutFields(
    BuildContext context,
    String type,
    Map<String, Object?> ro,
    List<String> addresses,
  ) {
    double num_(String key, double fallback) =>
        (ro[key] as num?)?.toDouble() ?? fallback;
    Widget nodePick(String key, {String label = 'Node'}) => Pick<String>(
      label: label,
      value: addresses.contains(ro[key]) ? ro[key] as String? : null,
      hint: '${ro[key]}',
      items: {for (final a in addresses) a: prettyId(a)},
      onChanged: (a) => _set(readout: {...ro, key: a}),
    );
    Widget number(String key, String label, {bool optional = false}) =>
        NumberField(
          key: ValueKey('$key ${ro[key]}'),
          label: label,
          value: ro[key] == null ? null : num_(key, 0),
          onSubmitted: (v) {
            if (v == null && !optional) return;
            _set(readout: v == null ? ({...ro}..remove(key)) : {...ro, key: v});
          },
        );
    switch (type) {
      case 'ObservationSurprise':
        return [
          Pick<String>(
            label: 'Surprise of',
            value: ro['function'] as String?,
            hint: '${ro['function']}',
            items: _surpriseFunctions,
            onChanged: (f) => _set(readout: {...ro, 'function': f}),
          ),
          const SizedBox(height: 4),
          Text(
            'No responses: the fit finds the parameters under which the '
            'observations are least surprising (the perceptual model on its '
            'own).',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ];
      case 'GaussianReadout':
        return [
          nodePick('node'),
          Pick<String>(
            label: 'Belief',
            value: ro['field'] as String?,
            items: {for (final f in NodeField.values) f.pyhgfName: f.pyhgfName},
            onChanged: (f) => _set(readout: {...ro, 'field': f}),
          ),
          number('precision', 'Response precision'),
        ];
      case 'BinarySoftmaxReadout' || 'CategoricalSoftmaxReadout':
        return [
          nodePick('node'),
          number('inverseTemperature', 'β (inverse temperature)'),
        ];
      case 'SoftmaxReadout':
        final opts = [
          for (final o in ro['options']! as List<Object?>)
            [for (final t in o! as List<Object?>) t! as Map<String, Object?>],
        ];
        final labels = r.options;
        final bias = (ro['bias'] as List<Object?>?)?.cast<num>();
        void setOptions(List<List<Map<String, Object?>>> next, {int? removed}) {
          final b = bias == null
              ? null
              : [
                  for (final (q, x) in bias.indexed)
                    if (q != removed) x.toDouble(),
                ];
          _set(
            readout: {...ro, 'options': next, 'bias': ?_resize(b, next.length)},
            options: labels == null || labels.length == next.length
                ? null
                : () => [
                    for (var j = 0; j < labels.length; j++)
                      if (j != removed) labels[j],
                    for (var j = labels.length; j < next.length; j++)
                      'option_$j',
                  ],
          );
        }

        return [
          number('inverseTemperature', 'β (inverse temperature)'),
          number('stickiness', 'Stickiness κ (empty: none)', optional: true),
          number('lapse', 'Lapse ε (empty: none)', optional: true),
          const SizedBox(height: 8),
          for (final (j, terms) in opts.indexed)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  SizedBox(
                    width: 64,
                    child: Text(
                      labels != null && j < labels.length
                          ? labels[j]
                          : 'option $j',
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  Expanded(
                    flex: 3,
                    child: Pick<String?>(
                      label: 'Value of',
                      value: terms.isEmpty
                          ? null
                          : terms.first['node'] as String?,
                      items: {
                        null: '(nothing: bias only)',
                        for (final a in addresses) a: prettyId(a),
                      },
                      onChanged: (a) {
                        final next = [...opts];
                        next[j] = a == null
                            ? const []
                            : [
                                {
                                  'node': a,
                                  'weight': terms.isEmpty
                                      ? 1.0
                                      : terms.first['weight'],
                                },
                              ];
                        setOptions(next);
                      },
                    ),
                  ),
                  const SizedBox(width: 6),
                  Expanded(
                    flex: 2,
                    child: NumberField(
                      key: ValueKey(
                        'w$j ${terms.isEmpty ? '' : terms.first['weight']}',
                      ),
                      label: '× weight',
                      value: terms.isEmpty
                          ? null
                          : (terms.first['weight']! as num).toDouble(),
                      onSubmitted: (w) {
                        if (terms.isEmpty || w == null) return;
                        final next = [...opts];
                        next[j] = [
                          {...terms.first, 'weight': w},
                          ...terms.skip(1),
                        ];
                        setOptions(next);
                      },
                    ),
                  ),
                  const SizedBox(width: 6),
                  Expanded(
                    flex: 2,
                    child: NumberField(
                      key: ValueKey('b$j ${bias?[j]}'),
                      label: '+ bias',
                      value: bias == null || j >= bias.length
                          ? null
                          : bias[j].toDouble(),
                      onSubmitted: (b) {
                        final next = [
                          for (var q = 0; q < opts.length; q++)
                            bias != null && q < bias.length
                                ? bias[q].toDouble()
                                : 0.0,
                        ];
                        next[j] = b ?? 0;
                        final none = next.every((x) => x == 0);
                        _set(
                          readout: none
                              ? ({...ro}..remove('bias'))
                              : {...ro, 'bias': next},
                        );
                      },
                    ),
                  ),
                  IconButton(
                    tooltip: 'Remove this option',
                    icon: const Icon(Icons.close, size: 18),
                    onPressed: opts.length <= 2
                        ? null
                        : () => setOptions([...opts]..removeAt(j), removed: j),
                  ),
                ],
              ),
            ),
          Align(
            alignment: Alignment.centerLeft,
            child: TextButton.icon(
              onPressed: () => setOptions([...opts, const []]),
              icon: const Icon(Icons.add, size: 18),
              label: const Text('Add an option'),
            ),
          ),
        ];
      case 'ExprReadout':
        final kind = ro['kind'] as String? ?? 'choice';
        final params = (ro['parameters'] as Map<String, Object?>?) ?? const {};
        String? check(String v) {
          try {
            Expr.parse(v);
            return null;
          } on FormatException catch (e) {
            return e.message;
          }
        }

        return [
          if (kind == 'choice')
            for (final (j, e) in (ro['options']! as List<Object?>).indexed)
              CommitField(
                label:
                    'Value of ${r.options != null && j < r.options!.length ? r.options![j] : 'option $j'}',
                value: '$e',
                monospace: true,
                validate: check,
                onCommit: (v) {
                  final next = [...ro['options']! as List<Object?>];
                  next[j] = v;
                  _set(readout: {...ro, 'options': next});
                },
              )
          else
            Text(
              'A $kind expression readout: edit it in the saved JSON.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          if (kind == 'choice')
            Align(
              alignment: Alignment.centerLeft,
              child: TextButton.icon(
                onPressed: () => _set(
                  readout: {
                    ...ro,
                    'options': [...ro['options']! as List<Object?>, '0'],
                  },
                ),
                icon: const Icon(Icons.add, size: 18),
                label: const Text('Add an option'),
              ),
            ),
          CommitField(
            label: 'Parameters (name = value, comma-separated)',
            value: [
              for (final MapEntry(:key, :value) in params.entries)
                '$key = $value',
            ].join(', '),
            validate: (v) =>
                _params(v) == null ? 'Write name = value, …' : null,
            onCommit: (v) => _set(readout: {...ro, 'parameters': _params(v)}),
          ),
          const SizedBox(height: 4),
          Text(
            'Beliefs are read as node.field (x1.expected_mean); previous is '
            'the last choice (−1 on the first trial).',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ];
    }
    return [
      Text(
        'Edit this response model in the saved JSON.',
        style: Theme.of(context).textTheme.bodySmall,
      ),
    ];
  }
}

/// [b] cut or padded with zeros to [n] values.
List<double>? _resize(List<double>? b, int n) =>
    b == null ? null : [for (var j = 0; j < n; j++) j < b.length ? b[j] : 0.0];

Map<String, double>? _params(String v) {
  final out = <String, double>{};
  for (final part in v.split(',')) {
    if (part.trim().isEmpty) continue;
    final eq = part.indexOf('=');
    if (eq < 0) return null;
    final name = part.substring(0, eq).trim();
    final value = double.tryParse(part.substring(eq + 1).trim());
    if (name.isEmpty || value == null) return null;
    out[name] = value;
  }
  return out;
}

/// Every address of [doc]: its nodes and their implied nodes.
List<String> _addresses(NetworkDocument doc) => [
  for (final n in doc.nodes) ...[
    n.id,
    for (var k = 1; k < n.size; k++) '${n.id}.$k',
  ],
];

// -----------------------------------------------------------------------------
// Output
// -----------------------------------------------------------------------------

class _OutputPanel extends StatelessWidget {
  const _OutputPanel({required this.model, required this.output});

  final BuilderModel model;
  final DocOutput output;

  @override
  Widget build(BuildContext context) {
    final doc = model.document;
    final o = output;
    final addresses = _addresses(doc.network);
    void set({String? name, String? node, String? field}) {
      final next = DocOutput(
        name ?? o.name,
        node: node ?? o.node,
        field: field ?? o.field,
      );
      model.update(doc.replaceOutput(o.name, next));
      if (name != null) model.select(PartSelection(DocPart.output(name)));
    }

    return SectionCard(
      title: 'Output',
      action: IconButton(
        tooltip: 'Delete',
        icon: const Icon(Icons.delete_outline, size: 20),
        onPressed: () => model.removePart(DocPart.output(o.name)),
      ),
      children: [
        CommitField(
          label: 'Column name',
          value: o.name,
          validate: (v) => v.trim().isEmpty
              ? 'Give it a name'
              : v != o.name && doc.outputs.any((x) => x.name == v)
              ? 'Another output has that name'
              : null,
          onCommit: (v) => set(name: v),
        ),
        Pick<String>(
          label: 'Node',
          value: addresses.contains(o.node) ? o.node : null,
          hint: o.node,
          items: {for (final a in addresses) a: prettyId(a)},
          onChanged: (a) => set(node: a),
        ),
        Pick<String>(
          label: 'Field',
          value: o.field,
          items: {
            for (final f in NodeField.values) f.pyhgfName: f.pyhgfName,
            'surprise': 'surprise',
          },
          onChanged: (f) => set(field: f),
        ),
        const SizedBox(height: 6),
        Text(
          'Written per time step next to the data\'s own columns by "Save '
          'with outputs" (Data), e.g. as regressors for EEG.',
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    );
  }
}

// -----------------------------------------------------------------------------
// Fields
// -----------------------------------------------------------------------------

/// A dropdown with a label.
class Pick<T> extends StatelessWidget {
  const Pick({
    super.key,
    required this.label,
    required this.value,
    required this.items,
    required this.onChanged,
    this.hint,
  });

  final String label;
  final T? value;
  final Map<T, String> items;
  final ValueChanged<T?> onChanged;
  final String? hint;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.only(top: 8),
    child: InputDecorator(
      decoration: InputDecoration(
        labelText: label,
        isDense: true,
        contentPadding: const EdgeInsets.fromLTRB(12, 8, 4, 8),
        border: const OutlineInputBorder(),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<T>(
          value: items.containsKey(value) ? value : null,
          hint: hint == null ? null : Text(hint!),
          isDense: true,
          isExpanded: true,
          items: [
            for (final MapEntry(:key, value: text) in items.entries)
              DropdownMenuItem(
                value: key,
                child: Text(text, overflow: TextOverflow.ellipsis),
              ),
          ],
          onChanged: onChanged,
        ),
      ),
    ),
  );
}

/// A column of the data (or a transform's), picked from the loaded data's
/// columns, or typed when the data is simulated.
class ColumnPick extends StatelessWidget {
  const ColumnPick({
    super.key,
    required this.model,
    required this.label,
    required this.value,
    required this.onChanged,
    this.allowNone = true,
  });

  final BuilderModel model;
  final String label;
  final String? value;
  final ValueChanged<String?> onChanged;
  final bool allowNone;

  @override
  Widget build(BuildContext context) {
    final columns = [
      ...?(model.csv == null ? null : model.dataColumns),
      ...model.document.transformNames,
    ];
    if (model.csv == null) {
      return CommitField(
        key: ValueKey('$label $value'),
        label: label,
        value: value ?? '',
        hint: allowNone ? 'none' : null,
        validate: (v) =>
            !allowNone && v.trim().isEmpty ? 'Name a column' : null,
        onCommit: (v) => onChanged(v.trim().isEmpty ? null : v.trim()),
      );
    }
    return Pick<String?>(
      label: label,
      value: value,
      hint: value == null ? null : '$value (not in the data)',
      items: {if (allowNone) null: '—', for (final c in columns) c: c},
      onChanged: (c) {
        if (c != value) onChanged(c);
      },
    );
  }
}

/// A text field committed on Enter or when it loses focus, when
/// [validate] finds nothing wrong (checked as you type).
class CommitField extends StatefulWidget {
  const CommitField({
    super.key,
    required this.label,
    required this.value,
    required this.onCommit,
    this.validate,
    this.hint,
    this.monospace = false,
  });

  final String label, value;
  final String? hint;
  final ValueChanged<String> onCommit;
  final String? Function(String)? validate;
  final bool monospace;

  @override
  State<CommitField> createState() => _CommitFieldState();
}

class _CommitFieldState extends State<CommitField> {
  late final _text = TextEditingController(text: widget.value);
  String? _error;

  @override
  void didUpdateWidget(CommitField old) {
    super.didUpdateWidget(old);
    if (old.value != widget.value && _text.text != widget.value) {
      _text.text = widget.value;
      _error = null;
    }
  }

  @override
  void dispose() {
    _text.dispose();
    super.dispose();
  }

  void _commit() {
    final v = _text.text;
    final e = widget.validate?.call(v);
    setState(() => _error = e);
    if (e == null && v != widget.value) widget.onCommit(v);
  }

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.only(top: 8),
    child: Focus(
      onFocusChange: (has) {
        if (!has) _commit();
      },
      child: TextField(
        controller: _text,
        style: widget.monospace
            ? const TextStyle(fontFamily: 'monospace')
            : null,
        decoration: InputDecoration(
          labelText: widget.label,
          hintText: widget.hint,
          errorText: _error,
          errorMaxLines: 3,
          isDense: true,
        ),
        onChanged: (v) {
          final e = widget.validate?.call(v);
          if (e != _error) setState(() => _error = e);
        },
        onSubmitted: (_) => _commit(),
      ),
    ),
  );
}
