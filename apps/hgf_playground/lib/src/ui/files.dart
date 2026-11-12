import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:material_ui/material_ui.dart';

import '../model/data.dart';

/// Saves [text] as [fileName] (a save dialog on desktop, a download on the
/// web), and says how that went.
Future<void> saveText(
  BuildContext context,
  String fileName,
  String text, {
  String mimeType = 'text/csv',
}) async {
  final messenger = ScaffoldMessenger.of(context);
  try {
    final uri = await FilePicker.saveFile(
      fileName: fileName,
      bytes: Uint8List.fromList(utf8.encode(text)),
      mimeType: mimeType,
      dialogTitle: 'Save $fileName',
    );
    if (uri == null) return;
    messenger.showSnackBar(SnackBar(content: Text('Saved $fileName')));
  } catch (e) {
    messenger.showSnackBar(SnackBar(content: Text('Could not save: $e')));
  }
}

/// Lets the user pick a CSV/TSV/text file and parses it ([parseTable]);
/// null if cancelled. The file's name comes back with the table.
Future<(CsvTable, String)?> pickTable(String dialogTitle) async {
  final file = await FilePicker.pickFile(
    dialogTitle: dialogTitle,
    type: FileType.custom,
    allowedExtensions: const ['csv', 'tsv', 'txt'],
  );
  if (file == null) return null;
  final text = utf8.decode(await file.readAsBytes(), allowMalformed: true);
  final table = parseTable(text);
  if (table.numericColumns.isEmpty) {
    throw const FormatException('No column of numbers found');
  }
  return (table, file.name);
}

/// Which columns of a table hold the observations and the responses.
typedef ColumnChoice = ({int? observations, int? responses});

/// Column names that suggest observations.
const observationHints = [
  'observation',
  'obs',
  'u',
  'input',
  'outcome',
  'stimulus',
  'x',
  'value',
];

/// Column names that suggest responses.
const responseHints = [
  'response',
  'resp',
  'choice',
  'decision',
  'action',
  'answer',
  'y',
  'rating',
];

/// The columns a table maps to without asking: the only numeric column as
/// the observations (or the responses, when only those are wanted); null
/// when there is a choice to make.
ColumnChoice? obviousColumns(
  CsvTable t, {
  required bool observations,
  required bool responses,
}) {
  final numeric = t.numericColumns;
  if (numeric.length != 1) return null;
  if (observations) return (observations: numeric.single, responses: null);
  return (observations: null, responses: numeric.single);
}

/// The columns [showColumnMapping] proposes: by name (see
/// [observationHints], [responseHints]), else the first numeric columns.
ColumnChoice guessColumns(
  CsvTable t, {
  required bool observations,
  required bool responses,
}) {
  final numeric = t.numericColumns;
  final obs = observations
      ? t.guess(observationHints, numeric) ?? numeric.first
      : null;
  int? resp;
  if (responses) {
    final rest = [
      for (final c in numeric)
        if (c != obs) c,
    ];
    resp = t.guess(responseHints, rest);
    // Only responses wanted: some column is needed.
    if (resp == null && !observations && rest.isNotEmpty) resp = rest.first;
  }
  return (observations: obs, responses: resp);
}

/// Asks which numeric column of [table] ([name]) holds the observations
/// (if [observations]) and which, if any, the responses (if [responses]),
/// with a preview of the first rows. Null if cancelled.
Future<ColumnChoice?> showColumnMapping(
  BuildContext context,
  CsvTable table,
  String name, {
  required bool observations,
  required bool responses,
}) => showDialog<ColumnChoice>(
  context: context,
  builder: (_) => ColumnMappingDialog(
    table: table,
    name: name,
    observations: observations,
    responses: responses,
  ),
);

class ColumnMappingDialog extends StatefulWidget {
  const ColumnMappingDialog({
    super.key,
    required this.table,
    required this.name,
    required this.observations,
    required this.responses,
  });

  final CsvTable table;
  final String name;
  final bool observations, responses;

  @override
  State<ColumnMappingDialog> createState() => _ColumnMappingDialogState();
}

class _ColumnMappingDialogState extends State<ColumnMappingDialog> {
  late ColumnChoice choice = guessColumns(
    widget.table,
    observations: widget.observations,
    responses: widget.responses,
  );

  @override
  Widget build(BuildContext context) {
    final t = widget.table;
    final theme = Theme.of(context);
    final numeric = t.numericColumns;
    final shown = [?choice.observations, ?choice.responses];
    final text = [
      for (var c = 0; c < t.columns.length; c++)
        if (t.textCells[c] > 0) t.names[c],
    ];
    String count(int c) {
      final missing = t.columns[c].where((v) => !v.isFinite).length;
      return missing == 0 ? '' : ' ($missing missing)';
    }

    return AlertDialog(
      title: Text('Columns of ${widget.name}'),
      content: SizedBox(
        width: 420,
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                '${t.rows} rows, ${numeric.length} numeric '
                '${numeric.length == 1 ? 'column' : 'columns'}. Each row is '
                'a trial; empty cells and NA are missing.',
                style: theme.textTheme.bodySmall,
              ),
              if (text.isNotEmpty)
                Text(
                  'Text columns, not offered: ${text.join(', ')}',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
              const SizedBox(height: 12),
              if (widget.observations)
                DropdownButtonFormField<int>(
                  key: const Key('observations column'),
                  initialValue: choice.observations,
                  decoration: const InputDecoration(labelText: 'Observations'),
                  items: [
                    for (final c in numeric)
                      DropdownMenuItem(
                        value: c,
                        child: Text('${t.names[c]}${count(c)}'),
                      ),
                  ],
                  onChanged: (c) => setState(
                    () => choice = (
                      observations: c,
                      responses: choice.responses == c
                          ? null
                          : choice.responses,
                    ),
                  ),
                ),
              if (widget.responses) ...[
                const SizedBox(height: 8),
                DropdownButtonFormField<int?>(
                  key: const Key('responses column'),
                  initialValue: choice.responses,
                  decoration: InputDecoration(
                    labelText: widget.observations
                        ? 'Responses (optional, for fitting)'
                        : 'Responses',
                  ),
                  items: [
                    if (widget.observations)
                      const DropdownMenuItem(value: null, child: Text('none')),
                    for (final c in numeric)
                      if (c != choice.observations)
                        DropdownMenuItem(
                          value: c,
                          child: Text('${t.names[c]}${count(c)}'),
                        ),
                  ],
                  onChanged: (c) => setState(
                    () => choice = (
                      observations: choice.observations,
                      responses: c,
                    ),
                  ),
                ),
              ],
              if (shown.isNotEmpty) ...[
                const SizedBox(height: 12),
                _preview(theme, shown),
              ],
            ],
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed:
              (widget.observations && choice.observations == null) ||
                  (!widget.observations && choice.responses == null)
              ? null
              : () => Navigator.pop(context, choice),
          child: const Text('Load'),
        ),
      ],
    );
  }

  /// The first rows of the chosen columns.
  Widget _preview(ThemeData theme, List<int> cols) {
    final t = widget.table;
    final rows = math.min(5, t.rows);
    final style = theme.textTheme.bodySmall?.copyWith(
      fontFeatures: const [FontFeature.tabularFigures()],
    );
    String cell(double v) => v.isFinite ? '$v' : '—';
    return Table(
      defaultColumnWidth: const IntrinsicColumnWidth(),
      children: [
        TableRow(
          children: [
            for (final c in cols)
              Padding(
                padding: const EdgeInsets.only(right: 16, bottom: 2),
                child: Text(t.names[c], style: theme.textTheme.labelMedium),
              ),
          ],
        ),
        for (var r = 0; r < rows; r++)
          TableRow(
            children: [
              for (final c in cols)
                Padding(
                  padding: const EdgeInsets.only(right: 16),
                  child: Text(cell(t.columns[c][r]), style: style),
                ),
            ],
          ),
        if (t.rows > rows)
          TableRow(children: [for (final _ in cols) Text('…', style: style)]),
      ],
    );
  }
}

/// A choice task's columns: the choice, then what it revealed (one per
/// [revealed] name), by name if they match ([responseHints] for the
/// choice), else the numeric columns in order. Null if there are too few.
List<int>? guessChoiceColumns(CsvTable t, List<String> revealed) {
  final numeric = t.numericColumns;
  if (numeric.length < 1 + revealed.length) return null;
  final out = <int>[];
  int pick(List<String> hints) {
    final rest = [
      for (final c in numeric)
        if (!out.contains(c)) c,
    ];
    return t.guess(hints, rest) ?? rest.first;
  }

  out.add(pick(responseHints));
  for (final r in revealed) {
    out.add(pick([r.toLowerCase()]));
  }
  return out;
}

/// Asks which numeric columns of [table] hold a choice task's choices and
/// what each choice revealed ([revealed]: e.g. `payoff`), with a preview.
/// Null if cancelled.
Future<List<int>?> showChoiceMapping(
  BuildContext context,
  CsvTable table,
  String name, {
  required List<String> options,
  required List<String> revealed,
  String unit = '',
}) => showDialog<List<int>>(
  context: context,
  builder: (_) => _ChoiceMappingDialog(
    table: table,
    name: name,
    options: options,
    revealed: revealed,
    unit: unit,
  ),
);

class _ChoiceMappingDialog extends StatefulWidget {
  const _ChoiceMappingDialog({
    required this.table,
    required this.name,
    required this.options,
    required this.revealed,
    required this.unit,
  });

  final CsvTable table;
  final String name;
  final List<String> options, revealed;
  final String unit;

  @override
  State<_ChoiceMappingDialog> createState() => _ChoiceMappingDialogState();
}

class _ChoiceMappingDialogState extends State<_ChoiceMappingDialog> {
  late final List<int> cols = guessChoiceColumns(
    widget.table,
    widget.revealed,
  )!;

  @override
  Widget build(BuildContext context) {
    final t = widget.table;
    final theme = Theme.of(context);
    final numeric = t.numericColumns;
    final opts = widget.options;
    final labels = [
      'Choice (0 = ${opts.first} … ${opts.length - 1} = ${opts.last})',
      for (final r in widget.revealed)
        '${r[0].toUpperCase()}${r.substring(1)} of the chosen option'
            '${widget.unit.isEmpty ? '' : ' (${widget.unit})'}',
    ];
    return AlertDialog(
      title: Text('Columns of ${widget.name}'),
      content: SizedBox(
        width: 420,
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                '${t.rows} rows. Each row is a trial: the option chosen and '
                'what it revealed (the other options\' outcomes are '
                'unknown). Empty cells and NA are missing.',
                style: theme.textTheme.bodySmall,
              ),
              const SizedBox(height: 12),
              for (final (j, label) in labels.indexed) ...[
                DropdownButtonFormField<int>(
                  key: Key('choice column $j'),
                  initialValue: cols[j],
                  decoration: InputDecoration(labelText: label),
                  items: [
                    for (final c in numeric)
                      DropdownMenuItem(value: c, child: Text(t.names[c])),
                  ],
                  onChanged: (c) => setState(() => cols[j] = c!),
                ),
                const SizedBox(height: 8),
              ],
              _ColumnPreview(table: t, cols: cols),
            ],
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed: cols.toSet().length == cols.length
              ? () => Navigator.pop(context, cols)
              : null,
          child: const Text('Load'),
        ),
      ],
    );
  }
}

class _ColumnPreview extends StatelessWidget {
  const _ColumnPreview({required this.table, required this.cols});

  final CsvTable table;
  final List<int> cols;

  @override
  Widget build(BuildContext context) {
    final t = table;
    final theme = Theme.of(context);
    final rows = math.min(5, t.rows);
    final style = theme.textTheme.bodySmall?.copyWith(
      fontFeatures: const [FontFeature.tabularFigures()],
    );
    String cell(double v) => v.isFinite ? '$v' : '—';
    return Table(
      defaultColumnWidth: const IntrinsicColumnWidth(),
      children: [
        TableRow(
          children: [
            for (final c in cols)
              Padding(
                padding: const EdgeInsets.only(right: 16, bottom: 2),
                child: Text(t.names[c], style: theme.textTheme.labelMedium),
              ),
          ],
        ),
        for (var r = 0; r < rows; r++)
          TableRow(
            children: [
              for (final c in cols)
                Padding(
                  padding: const EdgeInsets.only(right: 16),
                  child: Text(cell(t.columns[c][r]), style: style),
                ),
            ],
          ),
      ],
    );
  }
}
