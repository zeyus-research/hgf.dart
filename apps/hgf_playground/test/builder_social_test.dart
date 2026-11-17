@TestOn('vm')
library;

import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_playground/src/builder/builder_model.dart';

/// Players see the share of their team-mates pressing left.
ModelDocument _social() => ModelDocument(
  name: 'social',
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
          {'node': 'soc_v', 'weight': 3.0},
        ],
        [
          {'node': 'soc_v', 'weight': -3.0},
        ],
        <Object?>[],
      ],
      'field': 'expected_mean',
      'inverseTemperature': 2.0,
      'bias': [0.0, 0.0, -1.0],
    },
    options: ['left', 'right', 'none'],
  ),
  data: const DocData(participantColumn: 'participant'),
);

void main() {
  test('team-mates\' simulated presses are the inputs', () {
    final m = BuilderModel()
      ..open(_social())
      ..setTrials(100)
      ..setSimParticipants(6);
    // Teams need a group column: named for the user.
    m.setSimTeams(size: 3, regroupEvery: 50);
    expect(m.document.data.groupColumn, 'team');
    expect(m.issues.where((i) => i.isError), isEmpty, reason: '${m.issues}');
    expect(m.simulatesTogether, isTrue);
    expect(m.simTeam(0, 0), 'b1_1');
    expect(m.simTeam(3, 0), 'b1_2');

    final subs = m.allSubjects();
    expect(subs, hasLength(6));
    // The saved data is raw (no derived column) and reads back the same.
    final data = m.simulatedCsv();
    expect(data.header, isNot(contains('soc_in')));
    expect(data.header, containsAll(['participant', 'team', 'press']));
    final back = m.compiled!.subjects(data);
    for (var k = 0; k < 6; k++) {
      expect(back[k].inputs, subs[k].inputs);
      expect(back[k].responses, subs[k].responses);
    }
    // Participant 1's input is the share of its mates pressing left.
    final team = [for (var t = 0; t < 100; t++) m.simTeam(0, t)];
    for (var t = 0; t < 100; t++) {
      final mates = [
        for (var k = 1; k < 6; k++)
          if (m.simTeam(k, t) == team[t]) k,
      ];
      final left = mates.where((k) => subs[k].responses![t] == 0).length;
      expect(subs[0].inputs[t], left / mates.length, reason: 'step $t');
    }
    // The preview shows the coupled simulation.
    m.showParticipant(2);
    expect(m.subject!.inputs, subs[2].inputs);
    expect(m.outputsCsv, returnsNormally);
  });

  test('loaded events go onto the clock; team-mates line up', () {
    const events = '''
participant,onset,trial_type,value,condition,team_id
a,100.05,press,right,joint,1
a,100.31,press,left,joint,1
b,100.12,press,left,joint,1
b,100.48,press,none,joint,1
c,100.02,press,left,solo,
c,100.40,press,right,solo,
''';
    final m = BuilderModel()..open(_social());
    m.loadData(CsvData.parse(events), name: 'events.csv');
    // Without the clock the data lacks the press column's rows per step.
    final doc = m.document;
    m.update(
      doc.copyWith(
        data: doc.data.copyWith(
          timeColumn: () => 't',
          groupColumn: () => 'team',
          clock: () => DocClock(
            onsetColumn: 'onset',
            step: 0.1,
            streams: [
              ClockStream.parse(
                when: "trial_type == 'press'",
                hold: {
                  'press': 'value',
                  'team': "where(condition == 'joint', team_id, 0 / 0)",
                },
              ),
            ],
          ),
        ),
      ),
    );
    expect(m.issues.where((i) => i.isError), isEmpty, reason: '${m.issues}');
    expect(m.dataColumns, ['participant', 't', 'press', 'team']);
    expect(m.participants, ['a', 'b', 'c']);
    // a at 100.2: b holds left -> share 1; at 100.5: b none -> 0.
    final a = m.allSubjects().first;
    expect(a.inputs.sublist(0, 6), [0, 0, 1, 1, 1, 0]);
    // c plays alone: no team-mates, its input is never observed.
    final c = m.allSubjects()[2];
    expect(c.observed!.every((o) => o == 0), isTrue);
    expect(m.outputsCsv, returnsNormally);
  });
}
