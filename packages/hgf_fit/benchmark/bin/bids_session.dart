// The model builder's data path on a real BIDS dataset, stage by stage:
// load and merge the events tables, put them on the shared clock, the
// transforms (team-mates' presses), read subjects, filter one.
//
//   dart run benchmark/bin/bids_session.dart <bids root> [session01]
import 'dart:io';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';

void main(List<String> args) {
  if (args.isEmpty) {
    stderr.writeln('usage: bids_session.dart <bids root> [group]');
    exit(64);
  }
  final root = Directory(args[0]);
  final group = args.length > 1 ? args[1] : 'session01';
  final sw = Stopwatch()..start();
  void lap(String what) {
    print(
      '${what.padRight(34)} ${sw.elapsedMilliseconds.toString().padLeft(6)} ms',
    );
    sw.reset();
  }

  final files = [
    for (final f in root.listSync(recursive: true).whereType<File>())
      if (isBidsEvents(f.path.substring(root.path.length), datatype: 'beh')) f,
  ];
  final participants = CsvData.parse(
    File('${root.path}/participants.tsv').readAsStringSync(),
  );
  final tables = <BidsTable>[];
  for (final f in files) {
    final sub = bidsEntity(f.path, 'sub')!;
    final row = participants.rows.firstWhere((r) => r[0] == sub);
    if (row[participants.header.indexOf('group')] != group) continue;
    tables.add((name: f.path, table: CsvData.parse(f.readAsStringSync())));
  }
  lap('read + parse ${tables.length} events.tsv');
  const kinds = {
    'press',
    'CONTROL_REVERSAL_START',
    'CONTROL_REVERSAL_END',
    'INPUT_LOCK',
    'INPUT_UNLOCK',
  };
  final merged = mergeBidsEvents(
    tables,
    participants: participants,
    keepRow: (c, row) => kinds.contains(row[c['trial_type']!]),
  );
  final data = merged.data;
  lap('merge (${data.rows.length} events)');

  final doc = ModelDocument(
    name: 'bench',
    network: NetworkDocument(
      nodes: [
        ScalarNode('x', kind: NodeKind.continuousState),
        ScalarNode('x_v', kind: NodeKind.continuousState),
      ],
      edges: const [DocEdge(EdgeType.value, 'x_v', 'x')],
    ),
    inputs: [
      DocInput('x', node: 'x', columns: ['soc']),
    ],
    transforms: [
      DocTransform('left_mates', "mates_mean(press == 'left')"),
      DocTransform('n_pressing', 'mates_count(press)'),
      DocTransform('soc', 'fill(left_mates, 0) + reversed - locked'),
    ],
    outputs: const [DocOutput('belief', node: 'x_v', field: 'mean')],
    data: DocData(
      participantColumn: 'participant_id',
      timeColumn: 't',
      groupColumn: 'team',
      clock: DocClock(
        onsetColumn: 't_lsl_coord',
        step: 0.1,
        sessionColumn: 'group',
        streams: [
          ClockStream.parse(
            when: "trial_type == 'press'",
            hold: {
              'press': 'value',
              'team':
                  "where(condition == 'joint', "
                  'rt_session_id * 1000 + round * 10 + team_id, 0 / 0)',
            },
          ),
          ClockStream.parse(
            when:
                "(trial_type == 'CONTROL_REVERSAL_START') + "
                "(trial_type == 'CONTROL_REVERSAL_END')",
            hold: {'reversed': "trial_type == 'CONTROL_REVERSAL_START'"},
          ),
          ClockStream.parse(
            when:
                "(trial_type == 'INPUT_LOCK') + (trial_type == 'INPUT_UNLOCK')",
            hold: {'locked': "trial_type == 'INPUT_LOCK'"},
          ),
        ],
        keep: Expr.parse('missing(press) == 0'),
      ),
    ),
  );
  final c = doc.compile();
  lap('compile');
  final grid = c.layout.clocked(data);
  lap('clock (${grid.rows.length} rows)');
  final t = applyTransforms(
    grid,
    doc.transforms,
    participantColumn: 'participant_id',
    groupColumn: 'team',
    timeColumn: 't',
  );
  lap('transforms (${t.header.length} cols)');
  final subs = c.subjects(grid);
  lap('subjects (transforms + read)');
  final cache = TransformCache();
  c.subjects(grid, cache: cache);
  lap('cache: first subjects');
  c.subjects(grid, cache: cache);
  lap('cache: same again');
  final more = doc.withTransform(DocTransform('soc2', 'soc * 2')).compile();
  more.subjects(grid, cache: cache);
  lap('cache: one transform appended');
  final edited = doc
      .withoutTransform('soc')
      .withTransform(DocTransform('soc', 'fill(left_mates, 0) - locked'))
      .compile();
  edited.subjects(grid, cache: cache);
  lap('cache: last transform edited (${cache.lastComputed} computed)');
  final first = ModelDocument.fromJson(
    (doc.toJson()
      ..['transforms'] = [
        {'name': 'left_mates', 'expr': "mates_mean(press == 'right')"},
        for (final t in doc.transforms.skip(1)) t.toJson(),
      ]),
  ).compile();
  first.subjects(grid, cache: cache);
  lap('cache: first transform edited (${cache.lastComputed} computed)');
  final traj = c.filter(subs.first, c.currentValues());
  lap('filter one (${subs.first.inputs.length} steps)');
  c.outputs(traj);
}
