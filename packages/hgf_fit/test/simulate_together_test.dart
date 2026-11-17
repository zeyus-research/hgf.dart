import 'dart:math' as math;

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:test/test.dart';

/// Six players in teams of three, reshuffled between two blocks; each
/// sees how many of their team-mates pressed left and presses left, right
/// or nothing.
ModelDocument social({double weight = 3}) => ModelDocument(
  name: 'social',
  network: NetworkDocument(
    nodes: [
      ScalarNode('soc', kind: NodeKind.continuousState),
      ScalarNode(
        'soc_v',
        kind: NodeKind.continuousState,
        params: {NodeField.tonicVolatility: -2},
      ),
    ],
    edges: [const DocEdge(EdgeType.value, 'soc_v', 'soc')],
  ),
  transforms: [
    DocTransform('left', "press == 'left'"),
    DocTransform('soc_in', 'mates_mean(left)'),
  ],
  inputs: [
    DocInput('soc', node: 'soc', columns: ['soc_in']),
  ],
  response: DocResponse(
    'press',
    readout: {
      'type': 'SoftmaxReadout',
      'options': [
        [
          {'node': 'soc_v', 'weight': weight},
        ],
        [
          {'node': 'soc_v', 'weight': -weight},
        ],
        <Object?>[],
      ],
      'field': 'expected_mean',
      'inverseTemperature': 2.0,
      'bias': [0.0, 0.0, -1.0],
    },
    options: ['left', 'right', 'none'],
  ),
  data: const DocData(
    participantColumn: 'participant',
    timeColumn: 't',
    groupColumn: 'team',
  ),
);

/// Two blocks of [n] steps: teams abc | def, then abd | cef.
CsvData skeleton(int n) {
  const players = ['a', 'b', 'c', 'd', 'e', 'f'];
  String team(String p, int block) => block == 1
      ? ('abc'.contains(p) ? '1A' : '1B')
      : ('abd'.contains(p) ? '2A' : '2B');
  return CsvData(
    ['participant', 'block', 'team', 't'],
    [
      for (final p in players)
        for (var t = 0; t < 2 * n; t++)
          [p, '${t < n ? 1 : 2}', team(p, t < n ? 1 : 2), '${t * 0.5}'],
    ],
  );
}

void main() {
  test('team-mates\' presses feed each other\'s inputs, step by step', () {
    final c = social().compile();
    expect(c.inputsReadResponses, isTrue);
    final sim = c.simulateTogether(skeleton(60), math.Random(4));
    expect(sim.data.header, ['participant', 'block', 'team', 't', 'press']);
    expect(sim.subjects, hasLength(6));

    // The returned data reads back as the simulated subjects, bit for bit.
    final back = c.subjects(sim.data);
    for (final (k, s) in sim.subjects.indexed) {
      expect(back[k].id, s.id);
      expect(back[k].inputs, s.inputs);
      expect(back[k].responses, s.responses);
      expect(back[k].timeSteps, s.timeSteps);
      final o = back[k].observed;
      expect(o ?? List.filled(s.observed!.length, 1.0), s.observed);
    }

    // a's input at step t is the share of its team-mates pressing left.
    final press = [for (final s in sim.subjects) s.responses!];
    final a = sim.subjects[0];
    for (var t = 0; t < 120; t++) {
      final mates = t < 60 ? [1, 2] : [1, 3]; // b, c, then b, d
      final left = mates.where((m) => press[m][t] == 0).length;
      expect(a.inputs[t], left / 2, reason: 'step $t');
    }
  });

  test('imitation makes teams agree more than independent players', () {
    double agreement(ModelDocument doc, int seed) {
      final c = doc.compile();
      final sim = c.simulateTogether(skeleton(200), math.Random(seed));
      final p = [for (final s in sim.subjects) s.responses!];
      var same = 0, n = 0;
      for (var t = 0; t < 400; t++) {
        final teams = t < 200
            ? [
                [0, 1, 2],
                [3, 4, 5],
              ]
            : [
                [0, 1, 3],
                [2, 4, 5],
              ];
        for (final team in teams) {
          n++;
          if (team.every((m) => p[m][t] == p[team[0]][t])) same++;
        }
      }
      return same / n;
    }

    var social_ = 0.0, alone = 0.0;
    for (var seed = 0; seed < 4; seed++) {
      social_ += agreement(social(), seed);
      alone += agreement(social(weight: 0), seed);
    }
    expect(social_, greaterThan(alone * 1.5));
  });

  test('a model whose inputs read no responses is not coupled', () {
    final doc = social();
    final plain = doc.copyWith(
      transforms: [DocTransform('soc_in', 'mates_mean(cue)')],
    );
    expect(plain.compile().inputsReadResponses, isFalse);
  });

  test('every participant needs as many rows', () {
    final c = social().compile();
    final s = skeleton(10);
    final short = CsvData(s.header, s.rows.sublist(1));
    expect(
      () => c.simulateTogether(short, math.Random(1)),
      throwsA(
        isA<FormatException>().having(
          (e) => e.message,
          'message',
          contains('one clock'),
        ),
      ),
    );
  });
}
