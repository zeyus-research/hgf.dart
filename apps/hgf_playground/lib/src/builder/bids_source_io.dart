import 'dart:convert';
import 'dart:io';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:saf_stream/saf_stream.dart';
import 'package:saf_util/saf_util.dart';

import 'bids_source.dart';

/// The BIDS dataset in the folder at [uri] (see [pickBids]): a `file:`
/// folder read with dart:io, or on Android the `content:` tree URI of the
/// Storage Access Framework.
///
/// [BidsSource.events] are paths relative to the folder, with `/`
/// separators on every platform.
Future<BidsSource> bidsFolder(Uri uri) async => switch (uri.scheme) {
  'file' => _BidsFolder(_dirOf(uri)),
  'content' when Platform.isAndroid => await _SafBidsFolder.open(uri),
  _ => throw UnsupportedError(
    'Cannot read a "${uri.scheme}:" folder; pick the files instead',
  ),
};

/// The folder [uri] names, without the trailing separator
/// `Uri.directory` leaves on it (so its last segment is its name).
Directory _dirOf(Uri uri) {
  var p = uri.toFilePath();
  while (p.length > 1 && p.endsWith(Platform.pathSeparator)) {
    p = p.substring(0, p.length - 1);
  }
  return Directory(p);
}

/// The event tables among [all] (relative `/` paths): those of `beh`
/// folders, or every one when there are none.
List<String> _pickEvents(Iterable<String> all) {
  final events = [
    for (final p in all)
      if (isBidsEvents(p)) p,
  ];
  final beh = [
    for (final p in events)
      if (isBidsEvents(p, datatype: 'beh')) p,
  ];
  return (beh.isNotEmpty ? beh : events)..sort();
}

// Not derivatives, sourcedata, code: their tables aren't the raw dataset's.
bool _isSubject(String rel) => rel.startsWith('sub-');

final class _BidsFolder extends BidsSource {
  _BidsFolder(this.dir) {
    final prefix = dir.path.length + 1;
    events = _pickEvents([
      for (final e in dir.listSync(recursive: true, followLinks: false))
        if (e is File)
          if (e.path.substring(prefix).replaceAll(r'\', '/') case final rel
              when _isSubject(rel))
            rel,
    ]);
  }

  final Directory dir;

  File _file(String rel) =>
      File([dir.path, ...rel.split('/')].join(Platform.pathSeparator));

  @override
  String get name => dir.path.split(Platform.pathSeparator).last;

  @override
  late final List<String> events;

  @override
  Future<CsvData?> participants() async {
    final f = _file('participants.tsv');
    return await f.exists() ? CsvData.parse(await f.readAsString()) : null;
  }

  @override
  Future<CsvData> read(String path) async =>
      CsvData.parse(await _file(path).readAsString());
}

/// A folder picked through the Storage Access Framework (Android): no
/// paths, so the tree is walked once up front for the documents' URIs.
final class _SafBidsFolder extends BidsSource {
  _SafBidsFolder._(this.name, this._uris, this._participants)
    : events = _pickEvents(_uris.keys);

  static Future<_SafBidsFolder> open(Uri tree) async {
    final saf = SafUtil();
    final root = tree.toString();
    final dir = await saf.stat(root, true, throws: true);
    final uris = <String, String>{};
    String? participants;

    Future<void> walk(String uri, String prefix) async {
      final children = await saf.list(uri);
      await Future.wait([
        for (final c in children)
          if (c.isDir)
            walk(c.uri, '$prefix${c.name}/')
          else if (c.name.endsWith('_events.tsv'))
            Future.sync(() => uris['$prefix${c.name}'] = c.uri),
      ]);
    }

    final top = await saf.list(root);
    await Future.wait([
      for (final c in top)
        if (c.isDir && _isSubject(c.name))
          walk(c.uri, '${c.name}/')
        else if (!c.isDir && c.name == 'participants.tsv')
          Future.sync(() => participants = c.uri),
    ]);
    return _SafBidsFolder._(dir?.name ?? 'dataset', uris, participants);
  }

  @override
  final String name;

  /// Relative path → document URI, for every `*_events.tsv` under `sub-*`.
  final Map<String, String> _uris;
  final String? _participants;

  @override
  final List<String> events;

  static Future<CsvData> _parse(String uri) async => CsvData.parse(
    utf8.decode(await SafStream().readFileBytes(uri), allowMalformed: true),
  );

  @override
  Future<CsvData?> participants() async =>
      _participants == null ? null : _parse(_participants);

  @override
  Future<CsvData> read(String path) => _parse(_uris[path]!);
}
