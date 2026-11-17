import 'dart:convert';
import 'dart:math' as math;

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:test/test.dart';

/// Two players' events as in the RiseTogether events.tsv: presses (held
/// until the next), a control reversal (start/end), touches (ignored).
const events = '''
participant,onset,trial_type,value,round,team_id
a,100.02,touch,none,,
a,100.05,press,right,1,1
a,100.31,press,none,1,1
a,100.25,CONTROL_REVERSAL_START,zone,1,1
a,100.52,press,left,1,1
a,100.71,CONTROL_REVERSAL_END,zone,1,1
b,100.12,press,left,1,1
b,100.33,press,left,1,1
b,100.48,press,none,1,1
b,100.80,press,right,,
''';

DocClock clock({String? keep}) => DocClock(
  onsetColumn: 'onset',
  step: 0.1,
  streams: [
    ClockStream.parse(
      when: "trial_type == 'press'",
      hold: {'press': 'value', 'round': 'round', 'team': 'team_id'},
    ),
    ClockStream.parse(
      when:
          "(trial_type == 'CONTROL_REVERSAL_START') + "
          "(trial_type == 'CONTROL_REVERSAL_END')",
      hold: {'reversed': "trial_type == 'CONTROL_REVERSAL_START'"},
    ),
  ],
  keep: keep == null ? null : Expr.parse(keep),
);

List<String> col(CsvData d, String name) {
  final c = d.header.indexOf(name);
  return [for (final r in d.rows) r[c]];
}

void main() {
  test('events are held on a shared grid', () {
    final g = clock().resample(
      CsvData.parse(events),
      participantColumn: 'participant',
      timeColumn: 't',
    );
    expect(g.header, [
      'participant',
      't',
      'press',
      'round',
      'team',
      'reversed',
    ]);
    // From the first event (100.05, floored) to the last (100.80).
    const times = [
      '100',
      '100.1',
      '100.2',
      '100.3',
      '100.4',
      '100.5',
      '100.6',
      '100.7',
      '100.8',
    ];
    expect(col(g, 't'), [...times, ...times]);
    expect(col(g, 'participant'), [
      for (final p in ['a', 'b']) ...List.filled(9, p),
    ]);
    expect(col(g, 'press').sublist(0, 9), [
      '',
      'right',
      'right',
      'right',
      'none',
      'none',
      'left',
      'left',
      'left',
    ]);
    expect(col(g, 'press').sublist(9), [
      '',
      '',
      'left',
      'left',
      'left',
      'none',
      'none',
      'none',
      'right',
    ]);
    expect(col(g, 'reversed').sublist(0, 9), [
      '',
      '',
      '',
      '1',
      '1',
      '1',
      '1',
      '1',
      '0',
    ]);
    // b never had a reversal event: empty throughout.
    expect(col(g, 'reversed').sublist(9).toSet(), {''});
    expect(col(g, 'round').last, '');
  });

  test('keep drops grid rows', () {
    final g = clock(keep: 'missing(press) == 0').resample(
      CsvData.parse(events),
      participantColumn: 'participant',
      timeColumn: 't',
    );
    expect(col(g, 'participant').where((p) => p == 'a'), hasLength(8));
    expect(col(g, 'participant').where((p) => p == 'b'), hasLength(7));
  });

  test('a document reads events through the clock, and its grid as is', () {
    final doc = ModelDocument(
      network: NetworkDocument(
        nodes: [
          ScalarNode('soc', kind: NodeKind.continuousState),
          ScalarNode('soc_v', kind: NodeKind.continuousState),
        ],
        edges: [const DocEdge(EdgeType.value, 'soc_v', 'soc')],
      ),
      transforms: [DocTransform('soc_in', "mates_mean(press == 'left')")],
      inputs: [
        DocInput('soc', node: 'soc', columns: ['soc_in']),
      ],
      data: DocData(
        participantColumn: 'participant',
        timeColumn: 't',
        groupColumn: 'team',
        clock: clock(keep: 'missing(round) == 0'),
      ),
    );
    final again = ModelDocument.fromJson(
      jsonDecode(jsonEncode(doc.toJson())) as Map<String, Object?>,
    );
    expect(jsonEncode(again.toJson()), jsonEncode(doc.toJson()));
    expect(
      checkModel(
        again,
        columns: CsvData.parse(events).header,
      ).where((i) => i.isError),
      isEmpty,
    );
    expect(doc.data.columnsOf(CsvData.parse(events).header), [
      'participant',
      't',
      'press',
      'round',
      'team',
      'reversed',
    ]);

    final c = again.compile();
    final data = CsvData.parse(events);
    final s = c.subjects(data);
    expect(s.map((x) => x.id), ['a', 'b']);
    // a at 100.2 (b presses left) and 100.5 (b none); team only once known.
    final grid = c.layout.clocked(data);
    final back = c.subjects(grid);
    for (var k = 0; k < 2; k++) {
      expect(back[k].inputs, s[k].inputs);
      expect(back[k].timeSteps, s[k].timeSteps);
    }
    final ta = [
      for (final r in grid.rows)
        if (r[0] == 'a') r[1],
    ];
    expect(s[0].inputs[ta.indexOf('100.2')], 1);
    expect(s[0].inputs[ta.indexOf('100.5')], 0);
    expect(s[0].timeSteps!.every((d) => (d - 0.1).abs() < 1e-9), isTrue);
  });

  test('checks name what the clock reads', () {
    final doc = ModelDocument(
      network: NetworkDocument(
        nodes: [ScalarNode('x', kind: NodeKind.continuousState)],
        edges: const [],
      ),
      inputs: [
        DocInput('x', node: 'x', columns: ['press']),
      ],
      data: DocData(participantColumn: 'participant', clock: clock()),
    );
    final issues = checkModel(doc, columns: const ['participant', 'onset']);
    expect(
      issues.map((i) => i.message),
      contains(contains('needs a time column')),
    );
    final timed = doc.copyWith(data: doc.data.copyWith(timeColumn: () => 't'));
    expect(
      checkModel(
        timed,
        columns: const ['participant', 'onset'],
      ).map((i) => i.message),
      contains(contains('"trial_type"')),
    );
  });

  test('simulating together on the clock', () {
    // Simulated data is on the clock already (time column, no onsets).
    final doc = ModelDocument(
      network: NetworkDocument(
        nodes: [
          ScalarNode('soc', kind: NodeKind.continuousState),
          ScalarNode('soc_v', kind: NodeKind.continuousState),
        ],
        edges: [const DocEdge(EdgeType.value, 'soc_v', 'soc')],
      ),
      transforms: [DocTransform('soc_in', "mates_mean(press == 'left')")],
      inputs: [
        DocInput('soc', node: 'soc', columns: ['soc_in']),
      ],
      response: DocResponse(
        'press',
        readout: {
          'type': 'SoftmaxReadout',
          'options': [
            [
              {'node': 'soc_v', 'weight': 2.0},
            ],
            [
              {'node': 'soc_v', 'weight': -2.0},
            ],
            <Object?>[],
          ],
          'field': 'expected_mean',
          'inverseTemperature': 1.0,
        },
        options: ['left', 'right', 'none'],
      ),
      data: DocData(
        participantColumn: 'participant',
        timeColumn: 't',
        groupColumn: 'team',
        clock: clock(),
      ),
    );
    final c = doc.compile();
    final grid = c.layout.clocked(CsvData.parse(events));
    final sim = c.simulateTogether(grid, math.Random(3));
    expect(sim.data.header, contains('t'));
    final back = c.subjects(sim.data);
    expect(back[1].inputs, sim.subjects[1].inputs);
  });
}
