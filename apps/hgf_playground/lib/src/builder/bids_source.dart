import 'dart:convert';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';

import 'bids_source_stub.dart' if (dart.library.io) 'bids_source_io.dart' as io;

/// The files of a BIDS dataset the builder loads: the per-participant
/// event tables and `participants.tsv`, read only when asked (a dataset
/// may hold dozens of large tables; only the chosen participants' are
/// read).
abstract class BidsSource {
  /// Where it came from (the folder, or how many files were picked).
  String get name;

  /// The event tables' paths (or names), each with a `sub-` entity.
  List<String> get events;

  /// `participants.tsv`, if there is one.
  Future<CsvData?> participants();

  /// The table at [path] (one of [events]).
  Future<CsvData> read(String path);

  /// Merges the tables of the participants [include] picks (see
  /// [mergeBidsEvents]).
  Future<BidsMerge> load({
    CsvData? participants,
    bool Function(String participant, Map<String, String> info)? include,
  }) async {
    final info = <String, Map<String, String>>{};
    if (participants != null) {
      final id = participants.header.indexOf('participant_id');
      for (final row in participants.rows) {
        final raw = row[id].trim();
        info[raw.startsWith('sub-') ? raw : 'sub-$raw'] = {
          for (final (c, h) in participants.header.indexed) h: row[c].trim(),
        };
      }
    }
    final tables = <BidsTable>[];
    for (final path in events) {
      final sub = bidsEntity(path, 'sub')!;
      if (include != null && !include(sub, info[sub] ?? const {})) continue;
      tables.add((name: path, table: await read(path)));
    }
    return mergeBidsEvents(tables, participants: participants);
  }
}

/// Lets the user pick a BIDS dataset: its folder on desktop (every
/// `sub-*/beh/*_events.tsv`, or every `*_events.tsv` when there is no
/// `beh` folder), the event tables and `participants.tsv` themselves on the
/// web. Null if cancelled.
Future<BidsSource?> pickBids() async {
  if (!kIsWeb) {
    final dir = await FilePicker.getDirectoryPath(
      dialogTitle: 'A BIDS dataset (the folder with participants.tsv)',
    );
    if (dir == null) return null;
    return io.bidsFolder(dir);
  }
  final files = await FilePicker.pickFiles(
    dialogTitle:
        'A BIDS dataset: the participants\' *_events.tsv and participants.tsv',
    type: FileType.custom,
    allowedExtensions: const ['tsv'],
  );
  if (files.isEmpty) return null;
  return PickedBids(files);
}

/// Files picked one by one (the web: no folders, so participants come
/// from the file names).
final class PickedBids extends BidsSource {
  PickedBids(List<PlatformFile> files)
    : _files = {for (final f in files) f.name: f};

  final Map<String, PlatformFile> _files;

  @override
  String get name => '${events.length} event tables';

  @override
  late final List<String> events = [
    for (final n in _files.keys)
      if (isBidsEvents(n)) n,
  ]..sort();

  Future<String> _text(PlatformFile f) async =>
      utf8.decode(await f.readAsBytes(), allowMalformed: true);

  @override
  Future<CsvData?> participants() async {
    final f = _files['participants.tsv'];
    return f == null ? null : CsvData.parse(await _text(f));
  }

  @override
  Future<CsvData> read(String path) async =>
      CsvData.parse(await _text(_files[path]!));
}
