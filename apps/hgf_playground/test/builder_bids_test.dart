@TestOn('vm')
library;

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_playground/src/builder/bids_source_io.dart';
import 'package:hgf_playground/src/builder/builder_model.dart';
import 'package:hgf_playground/src/builder/ui/bids_dialog.dart';
import 'package:material_ui/material_ui.dart';

/// A BIDS folder: four participants in two sessions of two, presses in
/// beh/, an eeg events table and derivatives that must not be read.
Directory _dataset() {
  final dir = Directory.systemTemp.createTempSync('bids');
  void write(String path, String text) => File('${dir.path}/$path')
    ..createSync(recursive: true)
    ..writeAsStringSync(text);
  write(
    'participants.tsv',
    'participant_id\tgroup\tage\n'
        'sub-01\ts1\t20\nsub-02\ts1\t21\nsub-03\ts2\t22\nsub-04\ts2\t23\n',
  );
  for (final (k, sub) in ['sub-01', 'sub-02', 'sub-03', 'sub-04'].indexed) {
    final t0 = k < 2 ? 100.0 : 500.0;
    write(
      '$sub/beh/${sub}_task-x_events.tsv',
      'onset\tduration\ttrial_type\tvalue\tteam_id\tt_lsl_coord\n'
          '${t0 - 90}\t0\tpress\tleft\t1\t${t0 + 0.05 * k}\n'
          '${t0 - 89}\t0\tpress\tright\t1\t${t0 + 1.02}\n'
          '${t0 - 88}\t0\tpress\tnone\t1\t${t0 + 2.04}\n',
    );
    write('$sub/eeg/${sub}_task-x_events.tsv', 'onset\tvalue\n1\tS 1\n');
  }
  write('derivatives/x/sub-01/beh/sub-01_events.tsv', 'onset\n1\n');
  return dir;
}

ModelDocument _doc() => ModelDocument(
  name: 'bids',
  network: NetworkDocument(
    nodes: [
      ScalarNode('x', kind: NodeKind.continuousState),
      ScalarNode('x_v', kind: NodeKind.continuousState),
    ],
    edges: [const DocEdge(EdgeType.value, 'x_v', 'x')],
  ),
  transforms: [DocTransform('soc', "fill(mates_mean(press == 'left'), 0)")],
  inputs: [
    DocInput('x', node: 'x', columns: ['soc']),
  ],
  data: DocData(
    timeColumn: 't',
    groupColumn: 'team',
    clock: DocClock(
      onsetColumn: 't_lsl_coord',
      step: 0.5,
      sessionColumn: 'group',
      streams: [
        ClockStream.parse(
          when: "trial_type == 'press'",
          hold: {'press': 'value', 'team': 'team_id'},
        ),
      ],
    ),
  ),
);

void main() {
  test('a folder: beh event tables only, a session at a time', () async {
    final dir = _dataset();
    addTearDown(() => dir.deleteSync(recursive: true));
    final src = await bidsFolder(dir.uri);
    expect(src.events, hasLength(4));
    expect(src.events.every((p) => p.contains('/beh/')), isTrue);
    expect(src.events.first, 'sub-01/beh/sub-01_task-x_events.tsv');
    expect(src.name, dir.path.split(Platform.pathSeparator).last);
    final participants = await src.participants();
    final merged = await src.load(
      participants: participants,
      include: (_, info) => info['group'] == 's2',
    );
    expect(merged.participants, ['sub-03', 'sub-04']);
    expect(merged.data.header.first, 'participant_id');
    expect(merged.data.header, containsAll(['group', 'age', 't_lsl_coord']));
    expect(merged.data.rows, hasLength(6));

    // Into the builder: the participant column is named, the clock's grid
    // comes from the session, mates are the other player in the session.
    final m = BuilderModel()..open(_doc());
    m.loadBids(merged, name: 'x (s2)', sessionHint: 'group');
    expect(m.document.data.participantColumn, 'participant_id');
    expect(m.issues.where((i) => i.isError), isEmpty, reason: '${m.issues}');
    expect(m.participants, ['sub-03', 'sub-04']);
    final subs = m.allSubjects();
    // Events from 500.1 to 502.04 on a 0.5 grid: 500, 500.5, …, 502.5.
    expect(subs.first.inputs, hasLength(6));
    // sub-03's mate (sub-04) presses left at 500.15 (seen from 500.5),
    // right at 501.02 (seen from 501.5).
    expect(subs.first.inputs.sublist(0, 4), [0, 1, 1, 0]);
    // Again (another edit that leaves the data alone): the same subjects.
    m.showParticipant(1);
    expect(identical(m.allSubjects().first, subs.first), isTrue);
  });

  testWidgets('the dialog picks a session, or participants', (tester) async {
    final participants = CsvData.parse(
      'participant_id\tgroup\tage\n'
      'sub-01\ts1\t20\nsub-02\ts1\t21\nsub-03\ts2\t22\n',
    );
    BidsChoice? got;
    await tester.pumpWidget(
      MaterialApp(
        home: Builder(
          builder: (context) => TextButton(
            onPressed: () async => got = await showBidsDialog(
              context,
              name: 'ds',
              events: [
                for (final s in ['sub-01', 'sub-02', 'sub-03'])
                  '$s/beh/${s}_events.tsv',
              ],
              participants: participants,
            ),
            child: const Text('open'),
          ),
        ),
      ),
    );
    await tester.tap(find.text('open'));
    await tester.pumpAndSettle();
    // The first session to start with, grouped by `group`.
    expect(find.text('Load 2'), findsOneWidget);
    await tester.tap(find.text('s2 · 1 participants'));
    await tester.pump();
    await tester.tap(find.widgetWithText(FilterChip, 'sub-01'));
    await tester.pump();
    await tester.tap(find.text('Load 2'));
    await tester.pumpAndSettle();
    expect(got!.participants, {'sub-02', 'sub-03'});
    expect(got!.groupColumn, 'group');
  });
}
