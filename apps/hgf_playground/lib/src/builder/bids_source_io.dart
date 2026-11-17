import 'dart:io';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';

import 'bids_source.dart';

/// The BIDS dataset in the folder [path] (see [pickBids]).
BidsSource bidsFolder(String path) => _BidsFolder(Directory(path));

final class _BidsFolder extends BidsSource {
  _BidsFolder(this.dir) {
    final all = <String>[];
    for (final e in dir.listSync(recursive: true, followLinks: false)) {
      if (e is! File) continue;
      final rel = e.path.substring(dir.path.length);
      // Not derivatives, sourcedata, code: their tables aren't the raw
      // dataset's.
      if (!rel.replaceAll('\\', '/').startsWith(RegExp('/?sub-'))) continue;
      if (isBidsEvents(rel)) all.add(rel);
    }
    final beh = [
      for (final p in all)
        if (isBidsEvents(p, datatype: 'beh')) p,
    ];
    events = (beh.isNotEmpty ? beh : all)..sort();
  }

  final Directory dir;

  @override
  String get name => dir.path.split(Platform.pathSeparator).last;

  @override
  late final List<String> events;

  @override
  Future<CsvData?> participants() async {
    final f = File('${dir.path}/participants.tsv');
    return await f.exists() ? CsvData.parse(await f.readAsString()) : null;
  }

  @override
  Future<CsvData> read(String path) async =>
      CsvData.parse(await File('${dir.path}$path').readAsString());
}
