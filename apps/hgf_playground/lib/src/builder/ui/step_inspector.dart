import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:material_ui/material_ui.dart';

import '../../ui/panels.dart' show SectionCard;
import '../builder_model.dart';
import 'pipeline_inspector.dart' show CommitField, Pick;

/// The inspector of data step [index] ([DocStep]): its settings, moving
/// it earlier or later, deleting it.
class StepPanel extends StatelessWidget {
  const StepPanel({super.key, required this.model, required this.index});

  final BuilderModel model;
  final int index;

  @override
  Widget build(BuildContext context) {
    final steps = model.document.data.steps;
    final st = steps[index];
    final theme = Theme.of(context);
    final columns = model.columnsBeforeStep(index);
    final prepared = model.prepared;
    final rowsIn = prepared == null
        ? null
        : index == 0
        ? null
        : index - 1 < prepared.stepRows.length
        ? prepared.stepRows[index - 1]
        : null;
    final rowsOut = prepared != null && index < prepared.stepRows.length
        ? prepared.stepRows[index]
        : null;
    void set(DocStep s) => model.replaceStep(index, s);

    return SectionCard(
      title: BuilderModel.stepKinds[st.kind] ?? st.kind,
      action: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          IconButton(
            tooltip: 'Move earlier',
            icon: const Icon(Icons.arrow_back, size: 20),
            onPressed: index == 0 ? null : () => model.moveStep(index, -1),
          ),
          IconButton(
            tooltip: 'Move later',
            icon: const Icon(Icons.arrow_forward, size: 20),
            onPressed: index + 1 >= steps.length
                ? null
                : () => model.moveStep(index, 1),
          ),
          IconButton(
            tooltip: 'Delete',
            icon: const Icon(Icons.delete_outline, size: 20),
            onPressed: () => model.removePart(DocPart.step(index)),
          ),
        ],
      ),
      children: [
        Text(
          'Data step ${index + 1} of ${steps.length}'
          '${rowsOut == null ? '' : ' · ${rowsIn == null ? '' : '$rowsIn → '}$rowsOut rows'}',
          style: theme.textTheme.bodySmall,
        ),
        const SizedBox(height: 8),
        ...switch (st) {
          DecimateStep() => _decimate(context, st, columns, set),
          SegmentStep() => _segment(st, columns, set),
          FilterStep() => [
            CommitField(
              key: ValueKey('keep ${st.keep}'),
              label: 'Keep the rows where',
              value: '${st.keep}',
              hint: 'round > 0',
              monospace: true,
              validate: _exprError,
              onCommit: (v) => set(FilterStep(Expr.parse(v))),
            ),
          ],
          StandardiseStep() => _standardise(st, columns, set),
          SplitStep() => _split(st, set),
        },
        const SizedBox(height: 6),
        Text(_help(st), style: theme.textTheme.bodySmall),
      ],
    );
  }

  List<Widget> _decimate(
    BuildContext context,
    DecimateStep st,
    List<String> columns,
    void Function(DocStep) set,
  ) {
    DecimateStep edit({
      int? factor,
      StepAggregate? how,
      Map<String, StepAggregate>? cols,
      List<String>? within,
    }) => DecimateStep(
      factor: factor ?? st.factor,
      how: how ?? st.how,
      columns: cols ?? st.columns,
      within: within ?? st.within,
    );
    final clock = model.document.data.clock;
    final aggs = {for (final a in StepAggregate.values) a: _aggName(a)};
    return [
      CommitField(
        key: ValueKey('factor ${st.factor}'),
        label: 'Keep every k-th row',
        value: '${st.factor}',
        validate: (v) {
          final k = int.tryParse(v.trim());
          return k == null || k < 1 ? 'A whole number, at least 1' : null;
        },
        onCommit: (v) => set(edit(factor: int.parse(v.trim()))),
      ),
      if (clock != null && index == 0)
        Padding(
          padding: const EdgeInsets.only(bottom: 4),
          child: Text(
            'The clock\'s step ${ExprNum.formatNumber(clock.step)} becomes '
            '${ExprNum.formatNumber(clock.step * st.factor)}.',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ),
      Pick<StepAggregate>(
        label: 'Number columns',
        value: st.how,
        items: aggs,
        onChanged: (a) => set(edit(how: a)),
      ),
      for (final MapEntry(key: c, value: a) in st.columns.entries)
        Row(
          children: [
            Expanded(
              child: Pick<StepAggregate>(
                label: c,
                value: a,
                items: aggs,
                onChanged: (x) => set(edit(cols: {...st.columns, c: x ?? a})),
              ),
            ),
            IconButton(
              tooltip: 'Use the default',
              icon: const Icon(Icons.close, size: 18),
              onPressed: () => set(edit(cols: {...st.columns}..remove(c))),
            ),
          ],
        ),
      if (columns.where((c) => !st.columns.containsKey(c)).isNotEmpty)
        Pick<String?>(
          key: ValueKey('add ${st.columns.keys.join(',')}'),
          label: 'Aggregate a column otherwise…',
          value: null,
          items: {
            null: '—',
            for (final c in columns)
              if (!st.columns.containsKey(c)) c: c,
          },
          onChanged: (c) {
            if (c != null) {
              set(edit(cols: {...st.columns, c: StepAggregate.any}));
            }
          },
        ),
      _ColumnChips(
        label: 'Blocks end where these change',
        columns: columns,
        selected: st.within,
        onChanged: (w) => set(edit(within: w)),
      ),
    ];
  }

  List<Widget> _segment(
    SegmentStep st,
    List<String> columns,
    void Function(DocStep) set,
  ) {
    SegmentStep edit({
      Expr? at,
      int? before,
      int? Function()? after,
      String? column,
      String? Function()? offset,
    }) => SegmentStep(
      at: at ?? st.at,
      before: before ?? st.before,
      after: after == null ? st.after : after(),
      column: column ?? st.column,
      offsetColumn: offset == null ? st.offsetColumn : offset(),
    );
    String? count(String v, {bool empty = false}) {
      if (empty && v.trim().isEmpty) return null;
      final k = int.tryParse(v.trim());
      return k == null || k < 0 ? 'A whole number, 0 or more' : null;
    }

    return [
      CommitField(
        key: ValueKey('at ${st.at}'),
        label: 'Events: rows where',
        value: '${st.at}',
        hint: "diff(trial) != 0, press == 'left'",
        monospace: true,
        validate: _exprError,
        onCommit: (v) => set(edit(at: Expr.parse(v))),
      ),
      CommitField(
        key: ValueKey('before ${st.before}'),
        label: 'Rows before each event',
        value: '${st.before}',
        validate: count,
        onCommit: (v) => set(edit(before: int.parse(v.trim()))),
      ),
      CommitField(
        key: ValueKey('after ${st.after}'),
        label: 'Rows after each event (empty: up to the next)',
        value: st.after == null ? '' : '${st.after}',
        validate: (v) => count(v, empty: true),
        onCommit: (v) => set(
          edit(after: () => v.trim().isEmpty ? null : int.parse(v.trim())),
        ),
      ),
      CommitField(
        key: ValueKey('column ${st.column}'),
        label: 'Segment number column',
        value: st.column,
        validate: (v) => v.trim().isEmpty ? 'Give it a name' : null,
        onCommit: (v) => set(edit(column: v.trim())),
      ),
      CommitField(
        key: ValueKey('offset ${st.offsetColumn}'),
        label: 'Rows-from-event column (empty: none)',
        value: st.offsetColumn ?? '',
        onCommit: (v) =>
            set(edit(offset: () => v.trim().isEmpty ? null : v.trim())),
      ),
    ];
  }

  List<Widget> _standardise(
    StandardiseStep st,
    List<String> columns,
    void Function(DocStep) set,
  ) {
    StandardiseStep edit({
      List<String>? cols,
      bool? perParticipant,
      Expr? Function()? fitOn,
    }) => StandardiseStep(
      columns: cols ?? st.columns,
      perParticipant: perParticipant ?? st.perParticipant,
      fitOn: fitOn == null ? st.fitOn : fitOn(),
    );
    final d = model.document.data;
    final roles = {d.participantColumn, d.timeColumn, d.groupColumn};
    return [
      _ColumnChips(
        label: 'Columns',
        columns: [
          for (final c in columns)
            if (!roles.contains(c)) c,
        ],
        selected: st.columns,
        onChanged: (cs) => set(edit(cols: cs)),
      ),
      SwitchListTile(
        contentPadding: EdgeInsets.zero,
        dense: true,
        title: const Text('Per participant'),
        value: st.perParticipant,
        onChanged: (v) => set(edit(perParticipant: v)),
      ),
      CommitField(
        key: ValueKey('fit ${st.fitOn}'),
        label: 'Mean and sd from the rows where (empty: all)',
        value: st.fitOn == null ? '' : '${st.fitOn}',
        hint: "split == 'train'",
        monospace: true,
        validate: (v) => v.trim().isEmpty ? null : _exprError(v),
        onCommit: (v) =>
            set(edit(fitOn: () => v.trim().isEmpty ? null : Expr.parse(v))),
      ),
    ];
  }

  List<Widget> _split(SplitStep st, void Function(DocStep) set) {
    SplitStep edit({
      String? column,
      List<SplitPart>? parts,
      bool? byParticipant,
      int? seed,
    }) => SplitStep(
      column: column ?? st.column,
      parts: parts ?? st.parts,
      byParticipant: byParticipant ?? st.byParticipant,
      seed: seed ?? st.seed,
    );
    return [
      CommitField(
        key: ValueKey('column ${st.column}'),
        label: 'Label column',
        value: st.column,
        validate: (v) => v.trim().isEmpty ? 'Give it a name' : null,
        onCommit: (v) => set(edit(column: v.trim())),
      ),
      CommitField(
        key: ValueKey('parts ${st.toJson()['parts']}'),
        label: 'Parts (label share; …)',
        value: [
          for (final p in st.parts)
            '${p.label} ${ExprNum.formatNumber(p.fraction)}',
        ].join('; '),
        hint: 'train 0.8; test 0.2',
        monospace: true,
        validate: (v) {
          try {
            _parseParts(v);
            return null;
          } on FormatException catch (e) {
            return e.message;
          }
        },
        onCommit: (v) => set(edit(parts: _parseParts(v))),
      ),
      SwitchListTile(
        contentPadding: EdgeInsets.zero,
        dense: true,
        title: const Text('Whole participants'),
        subtitle: const Text('Off: each participant\'s rows, in time order'),
        value: st.byParticipant,
        onChanged: (v) => set(edit(byParticipant: v)),
      ),
      if (st.byParticipant)
        CommitField(
          key: ValueKey('seed ${st.seed}'),
          label: 'Seed (which participants go where)',
          value: '${st.seed}',
          validate: (v) =>
              int.tryParse(v.trim()) == null ? 'A whole number' : null,
          onCommit: (v) => set(edit(seed: int.parse(v.trim()))),
        ),
    ];
  }

  static String _help(DocStep st) => switch (st) {
    DecimateStep() =>
      'Each participant\'s rows in blocks of k, one row per block. Labels '
          'and the participant, group and time columns take the block\'s first '
          'row; "any" keeps a press that falls between kept rows.',
    SegmentStep() =>
      'Each event starts a segment (numbered per participant); where '
          'windows overlap, the later event takes the rows. Rows in no segment '
          'are dropped.',
    FilterStep() =>
      'Rows where the expression is 0 or missing are dropped. Columns by '
          'name; lag, diff, missing, mate, … as in transforms.',
    StandardiseStep() =>
      'z-scores in place: (x − mean) / sd. Fit on the training rows so '
          'the test rows don\'t leak into the scaling. Missing cells stay '
          'missing.',
    SplitStep() =>
      'Adds a label column; every row is kept. Read it in a filter, a '
          'transform or a standardise step.',
  };

  static String _aggName(StepAggregate a) => switch (a) {
    StepAggregate.first => 'first row',
    StepAggregate.last => 'last row',
    StepAggregate.mean => 'mean',
    StepAggregate.sum => 'sum',
    StepAggregate.min => 'min',
    StepAggregate.max => 'max',
    StepAggregate.any => 'any (1 if any is nonzero)',
    StepAggregate.mode => 'most common label',
  };
}

String? _exprError(String v) {
  try {
    Expr.parse(v);
    return null;
  } on FormatException catch (e) {
    return e.message;
  }
}

/// `train 0.8; test 0.2` as split parts.
List<SplitPart> _parseParts(String v) {
  final parts = <SplitPart>[];
  for (final item in v.split(';')) {
    final t = item.trim();
    if (t.isEmpty) continue;
    final m = RegExp(r'^(\S+)\s+([0-9.eE+-]+)$').firstMatch(t);
    final f = m == null ? null : double.tryParse(m.group(2)!);
    if (m == null || f == null || f < 0) {
      throw FormatException('"$t": a label and a share, e.g. train 0.8');
    }
    parts.add((label: m.group(1)!, fraction: f));
  }
  if (parts.isEmpty) throw const FormatException('Give at least one part');
  if (parts.fold(0.0, (s, p) => s + p.fraction) <= 0) {
    throw const FormatException('The shares sum to 0');
  }
  return parts;
}

/// Columns to pick several of, as chips.
class _ColumnChips extends StatelessWidget {
  const _ColumnChips({
    required this.label,
    required this.columns,
    required this.selected,
    required this.onChanged,
  });

  final String label;
  final List<String> columns, selected;
  final ValueChanged<List<String>> onChanged;

  @override
  Widget build(BuildContext context) {
    final all = {...columns, ...selected};
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: Theme.of(context).textTheme.labelMedium),
          const SizedBox(height: 4),
          if (all.isEmpty)
            Text(
              'Load data to pick columns.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          Wrap(
            spacing: 6,
            runSpacing: 4,
            children: [
              for (final c in all)
                FilterChip(
                  label: Text(c),
                  selected: selected.contains(c),
                  onSelected: (on) => onChanged(
                    on
                        ? [...selected, c]
                        : [
                            for (final s in selected)
                              if (s != c) s,
                          ],
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }
}
