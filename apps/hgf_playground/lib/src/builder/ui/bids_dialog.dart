import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:material_ui/material_ui.dart';

/// Which participants of a BIDS dataset to load, and the column of
/// `participants.tsv` that groups them (e.g. `group`: who was recorded
/// together, which the clock takes as its session column).
typedef BidsChoice = ({Set<String> participants, String? groupColumn});

/// Asks which participants of a dataset to load: grouped by a column of
/// [participants] (whole groups at once, e.g. a session's six), or one by
/// one. [events] are the event tables' paths (one or more per
/// participant). Null if cancelled.
Future<BidsChoice?> showBidsDialog(
  BuildContext context, {
  required String name,
  required List<String> events,
  CsvData? participants,
}) => showDialog<BidsChoice>(
  context: context,
  builder: (_) =>
      _BidsDialog(name: name, events: events, participants: participants),
);

/// The participants.tsv columns that usually group participants recorded
/// together, in order of preference.
const _groupHints = ['group', 'session', 'session_id', 'rt_session_id'];

class _BidsDialog extends StatefulWidget {
  const _BidsDialog({
    required this.name,
    required this.events,
    required this.participants,
  });

  final String name;
  final List<String> events;
  final CsvData? participants;

  @override
  State<_BidsDialog> createState() => _BidsDialogState();
}

class _BidsDialogState extends State<_BidsDialog> {
  /// Participants with event tables, in order, and how many tables each.
  late final Map<String, int> subs = () {
    final out = <String, int>{};
    for (final p in widget.events) {
      final s = bidsEntity(p, 'sub')!;
      out[s] = (out[s] ?? 0) + 1;
    }
    return Map.fromEntries(
      out.entries.toList()..sort((a, b) => a.key.compareTo(b.key)),
    );
  }();

  /// participants.tsv's cells per participant.
  late final Map<String, Map<String, String>> info = () {
    final t = widget.participants;
    if (t == null) return const <String, Map<String, String>>{};
    final id = t.header.indexOf('participant_id');
    if (id < 0) return const <String, Map<String, String>>{};
    return {
      for (final row in t.rows)
        _norm(row[id]): {
          for (final (c, h) in t.header.indexed) h: row[c].trim(),
        },
    };
  }();

  static String _norm(String id) {
    final s = id.trim();
    return s.startsWith('sub-') ? s : 'sub-$s';
  }

  /// The columns that group participants: fewer distinct values than
  /// participants, more than one.
  late final List<String> groupable = () {
    final t = widget.participants;
    if (t == null) return const <String>[];
    return [
      for (final h in t.header)
        if (h != 'participant_id')
          if ({for (final s in subs.keys) info[s]?[h] ?? ''}.length case final n
              when n > 1 && n < subs.length)
            h,
    ];
  }();

  late String? group =
      _groupHints.where(groupable.contains).firstOrNull ??
      groupable.firstOrNull;

  /// The first group (a session's participants) is chosen to start with:
  /// a whole dataset may not fit.
  late final Set<String> chosen = {
    if (group case final g?)
      for (final s in subs.keys)
        if ((info[s]?[g] ?? '') == _groups(g).keys.first) s,
    if (group == null) ...subs.keys,
  };

  Map<String, List<String>> _groups(String column) {
    final out = <String, List<String>>{};
    for (final s in subs.keys) {
      (out[info[s]?[column] ?? ''] ??= []).add(s);
    }
    return out;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final small = theme.textTheme.bodySmall;
    final g = group;
    final groups = g == null ? null : _groups(g);
    final noInfo = [
      for (final s in subs.keys)
        if (!info.containsKey(s)) s,
    ];
    Widget subChip(String s) => FilterChip(
      label: Text(s),
      selected: chosen.contains(s),
      onSelected: (on) => setState(() => on ? chosen.add(s) : chosen.remove(s)),
    );

    return AlertDialog(
      title: Text('Load ${widget.name}'),
      content: SizedBox(
        width: 520,
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                '${subs.length} participants with event tables '
                '(${widget.events.length} files). They become one table: '
                'participant_id first, then the events\' columns, then '
                '${widget.participants == null ? 'nothing else (no participants.tsv)' : 'participants.tsv\'s columns'}.',
                style: small,
              ),
              if (noInfo.isNotEmpty && widget.participants != null)
                Text(
                  'Not in participants.tsv: ${noInfo.join(', ')}',
                  style: small?.copyWith(color: theme.colorScheme.error),
                ),
              if (groupable.isNotEmpty) ...[
                const SizedBox(height: 12),
                DropdownButtonFormField<String?>(
                  initialValue: g,
                  decoration: const InputDecoration(
                    labelText: 'Grouped by (who was recorded together)',
                  ),
                  items: [
                    const DropdownMenuItem(value: null, child: Text('—')),
                    for (final h in groupable)
                      DropdownMenuItem(value: h, child: Text(h)),
                  ],
                  onChanged: (v) => setState(() => group = v),
                ),
              ],
              const SizedBox(height: 12),
              if (groups != null)
                for (final MapEntry(key: value, value: members)
                    in groups.entries) ...[
                  CheckboxListTile(
                    dense: true,
                    contentPadding: EdgeInsets.zero,
                    controlAffinity: ListTileControlAffinity.leading,
                    tristate: true,
                    value: members.every(chosen.contains)
                        ? true
                        : members.any(chosen.contains)
                        ? null
                        : false,
                    title: Text(
                      '${value.isEmpty ? '(empty)' : value} · '
                      '${members.length} participants',
                    ),
                    onChanged: (_) => setState(() {
                      if (members.every(chosen.contains)) {
                        chosen.removeAll(members);
                      } else {
                        chosen.addAll(members);
                      }
                    }),
                  ),
                  Padding(
                    padding: const EdgeInsets.only(left: 40, bottom: 4),
                    child: Wrap(
                      spacing: 4,
                      runSpacing: 4,
                      children: [for (final s in members) subChip(s)],
                    ),
                  ),
                ]
              else
                Wrap(
                  spacing: 4,
                  runSpacing: 4,
                  children: [for (final s in subs.keys) subChip(s)],
                ),
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
          onPressed: chosen.isEmpty
              ? null
              : () => Navigator.pop(context, (
                  participants: {...chosen},
                  groupColumn: g,
                )),
          child: Text('Load ${chosen.length}'),
        ),
      ],
    );
  }
}
