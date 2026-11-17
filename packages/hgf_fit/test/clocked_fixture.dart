import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';

/// Two sessions of events as in the RiseTogether events.tsv (presses held
/// until the next, a control reversal, a lock, touches ignored), teams per
/// session.
const clockedEventsCsv = '''
participant,session,onset,trial_type,value,round,team
a,1,100.02,touch,none,,
a,1,100.05,press,right,1,1
a,1,100.31,press,none,1,1
a,1,100.25,CONTROL_REVERSAL_START,zone,1,1
a,1,100.52,press,left,1,1
a,1,100.71,CONTROL_REVERSAL_END,zone,1,1
b,1,100.12,press,left,1,1
b,1,100.33,press,left,1,1
b,1,100.48,press,none,1,1
b,1,100.48,INPUT_LOCK,,1,1
b,1,100.62,INPUT_UNLOCK,,1,1
b,1,100.80,press,right,,
c,2,7.0,press,left,1,1
c,2,7.26,press,right,1,1
d,2,7.04,press,none,1,1
d,2,7.15,press,left,1,1
d,2,7.15,press,right,1,1
''';

/// A document over [clockedEventsCsv]: the clock, team-mates' presses as input.
ModelDocument clockedDocument() => ModelDocument(
  name: 'clocked',
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
    DocTransform('soc', "mates_mean(press == 'left') + reversed - locked"),
  ],
  outputs: const [DocOutput('belief', node: 'x_v', field: 'mean')],
  data: DocData(
    participantColumn: 'participant',
    timeColumn: 't',
    groupColumn: 'team',
    clock: DocClock(
      onsetColumn: 'onset',
      step: 0.05,
      sessionColumn: 'session',
      streams: [
        ClockStream.parse(
          when: "trial_type == 'press'",
          hold: {'press': 'value', 'team': 'session * 10 + team'},
        ),
        ClockStream.parse(
          when:
              "(trial_type == 'CONTROL_REVERSAL_START') + "
              "(trial_type == 'CONTROL_REVERSAL_END')",
          hold: {'reversed': "trial_type == 'CONTROL_REVERSAL_START'"},
        ),
        ClockStream.parse(
          when: "(trial_type == 'INPUT_LOCK') + (trial_type == 'INPUT_UNLOCK')",
          hold: {'locked': "trial_type == 'INPUT_LOCK'"},
        ),
      ],
      keep: Expr.parse('missing(team) == 0'),
    ),
  ),
);

/// [clockedDocument] with every kind of data-pipeline step after the clock.
ModelDocument steppedDocument() {
  final doc = clockedDocument();
  return doc.copyWith(
    data: doc.data.copyWith(
      steps: [
        SplitStep(
          parts: const [
            (label: 'train', fraction: 0.5),
            (label: 'test', fraction: 0.5),
          ],
          byParticipant: true,
          seed: 3,
        ),
        DecimateStep(
          factor: 2,
          columns: const {
            'press': StepAggregate.mode,
            'reversed': StepAggregate.any,
            'locked': StepAggregate.max,
          },
        ),
        SegmentStep(
          at: Expr.parse("press == 'left'"),
          before: 1,
          offsetColumn: 'seg_t',
        ),
        FilterStep(Expr.parse('seg_t <= 3')),
        StandardiseStep(
          columns: ['seg_t'],
          perParticipant: false,
          fitOn: Expr.parse("split == 'train'"),
        ),
      ],
    ),
  );
}
