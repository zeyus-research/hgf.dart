import 'package:hgf/hgf.dart';

/// One table of a BIDS dataset: its file name (or path) and its rows.
typedef BidsTable = ({String name, CsvData table});

/// The BIDS entity [key] (`sub`, `ses`, `task`, `run`, …) in a file name
/// or path, e.g. `sub-001` for `sub` in
/// `sub-001/beh/sub-001_task-x_events.tsv`; null when it has none. The
/// file name wins over the folders.
String? bidsEntity(String path, String key) {
  final parts = path.split(RegExp(r'[/\\]'));
  final re = RegExp('(?:^|_)$key-([A-Za-z0-9]+)(?=_|\\.|\$)');
  for (final part in parts.reversed) {
    if (re.firstMatch(part) case final m?) return '$key-${m.group(1)}';
  }
  return null;
}

/// Whether [path] is a per-participant events table (`*_events.tsv` with a
/// `sub-` entity) of the [datatype] folder (e.g. `beh`; null: any).
bool isBidsEvents(String path, {String? datatype}) {
  final p = path.replaceAll('\\', '/');
  if (!p.endsWith('_events.tsv') || bidsEntity(p, 'sub') == null) {
    return false;
  }
  if (datatype == null) return true;
  final parts = p.split('/');
  return parts.length >= 2 && parts[parts.length - 2] == datatype;
}

/// What [mergeBidsEvents] found: the merged table, and the participants
/// and sessions it holds.
typedef BidsMerge = ({
  CsvData data,
  List<String> participants,
  List<String> sessions,
});

/// A BIDS dataset's per-participant event tables (`sub-*/…/*_events.tsv`)
/// as one long table, one row per event: what the model builder loads and
/// puts on a shared clock (see `DocClock`).
///
/// The first column is [participantColumn] (`participant_id`, from each
/// file's `sub-` entity), then `session` when the file names have a `ses-`
/// entity, then the events' columns (every file's, in order of first
/// appearance; a file without one has empty cells there), then the
/// [participants] table's columns (BIDS `participants.tsv`, joined on its
/// `participant_id`; a column the events already have keeps the events'
/// cells). Tables come in participant order (then session, then name), each
/// in file order.
///
/// [include] picks participants by their row of [participants] (column →
/// cell; empty without one), e.g. one session's six; [keepRow] picks
/// events (column → index in the file's header, and the cells), e.g.
/// presses only.
BidsMerge mergeBidsEvents(
  List<BidsTable> events, {
  CsvData? participants,
  String participantColumn = 'participant_id',
  bool Function(String participant, Map<String, String> info)? include,
  bool Function(Map<String, int> columns, List<String> row)? keepRow,
}) {
  final info = <String, Map<String, String>>{};
  final infoColumns = <String>[];
  if (participants case final p?) {
    final idCol = p.header.indexOf('participant_id');
    if (idCol < 0) {
      throw const FormatException(
        'participants.tsv has no participant_id column',
      );
    }
    for (final (c, h) in p.header.indexed) {
      if (c != idCol) infoColumns.add(h);
    }
    for (final row in p.rows) {
      info[_normalise(row[idCol])] = {
        for (final (c, h) in p.header.indexed)
          if (c != idCol) h: row[c].trim(),
      };
    }
  }
  final files = <({String sub, String? ses, String name, CsvData table})>[];
  for (final (:name, :table) in events) {
    final sub = bidsEntity(name, 'sub');
    if (sub == null) {
      throw FormatException('$name: no participant (sub-…) in its name');
    }
    if (include != null && !include(sub, info[sub] ?? const {})) continue;
    files.add((
      sub: sub,
      ses: bidsEntity(name, 'ses'),
      name: name,
      table: table,
    ));
  }
  files.sort((a, b) {
    final s = a.sub.compareTo(b.sub);
    if (s != 0) return s;
    final e = (a.ses ?? '').compareTo(b.ses ?? '');
    return e != 0 ? e : a.name.compareTo(b.name);
  });
  final hasSession = files.any((f) => f.ses != null);
  final header = <String>[participantColumn, if (hasSession) 'session'];
  final fixed = header.length;
  final at = <String, int>{};
  for (final f in files) {
    for (final h in f.table.header) {
      if (h == participantColumn || (hasSession && h == 'session')) continue;
      at.putIfAbsent(h, () {
        header.add(h);
        return header.length - 1;
      });
    }
  }
  final nEvents = header.length;
  final joined = [
    for (final h in infoColumns)
      if (!at.containsKey(h) && h != participantColumn) h,
  ];
  header.addAll(joined);
  final rows = <List<String>>[];
  final subs = <String>{}, sessions = <String>{};
  for (final f in files) {
    final t = f.table;
    final cols = {for (final (c, h) in t.header.indexed) h: c};
    final into = [for (final h in t.header) at[h]];
    final extra = info[f.sub];
    final tail = [for (final h in joined) extra?[h] ?? ''];
    var kept = false;
    for (final row in t.rows) {
      if (keepRow != null && !keepRow(cols, row)) continue;
      final out = List<String>.filled(header.length, '');
      out[0] = f.sub;
      if (hasSession) out[1] = f.ses ?? '';
      for (var c = 0; c < row.length; c++) {
        if (into[c] case final i?) out[i] = row[c];
      }
      out.setRange(nEvents, header.length, tail);
      rows.add(out);
      kept = true;
    }
    if (kept) {
      subs.add(f.sub);
      if (f.ses != null) sessions.add(f.ses!);
    }
  }
  assert(fixed <= nEvents);
  return (
    data: CsvData(header, rows),
    participants: subs.toList(),
    sessions: sessions.toList()..sort(),
  );
}

/// `sub-001` for `sub-001` or `001` (participants.tsv may leave out the
/// prefix).
String _normalise(String id) {
  final s = id.trim();
  return s.startsWith('sub-') ? s : 'sub-$s';
}
