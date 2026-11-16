import 'dart:convert';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:test/test.dart';

void main() {
  // Four players; teams of two, reshuffled between blocks (b1: ab | cd,
  // b2: ac | bd). Presses are labels; "none" is a press, not a gap.
  const csv = '''
participant,block,team,t,x,press
a,1,b1_A,0,1,left
a,1,b1_A,1,,none
a,2,b2_A,2,3,right
a,2,b2_A,3,4,left
b,1,b1_A,0,10,right
b,1,b1_A,1,20,left
b,2,b2_B,2,30,none
b,2,b2_B,3,40,right
c,1,b1_B,0,100,none
c,1,b1_B,1,200,none
c,2,b2_A,2,300,left
c,2,b2_A,3,400,left
d,1,b1_B,0,1000,left
d,1,b1_B,1,2000,right
d,2,b2_B,2,3000,right
d,2,b2_B,3,4000,none
''';

  List<String> column(CsvData d, String name) {
    final c = d.header.indexOf(name);
    return [for (final r in d.rows) r[c]];
  }

  CsvData apply(List<DocTransform> ts, {bool time = true}) => applyTransforms(
    CsvData.parse(csv),
    ts,
    participantColumn: 'participant',
    groupColumn: 'team',
    timeColumn: time ? 't' : null,
  );

  test('lag, diff, fill and missing are per participant', () {
    final d = apply([
      DocTransform('prev', 'lag(x)'),
      DocTransform('prev2', 'lag(x, 2)'),
      DocTransform('dx', 'diff(x)'),
      DocTransform('filled', 'fill(x, -1)'),
      DocTransform('gap', 'missing(x)'),
      DocTransform('prev_press', 'lag(press)'),
    ]);
    expect(column(d, 'prev').sublist(0, 6), ['', '1', '', '3', '', '10']);
    expect(column(d, 'prev2').sublist(0, 4), ['', '', '1', '']);
    expect(column(d, 'dx').sublist(4, 8), ['', '10', '10', '10']);
    expect(column(d, 'filled').sublist(0, 3), ['1', '-1', '3']);
    expect(column(d, 'gap').sublist(0, 3), ['0', '1', '0']);
    expect(column(d, 'prev_press').sublist(0, 4), [
      '',
      'left',
      'none',
      'right',
    ]);
  });

  test('labels compare as written, "none" included', () {
    final d = apply([
      DocTransform('left', "press == 'left'"),
      DocTransform('nothing', "press == 'none'"),
      DocTransform('moved', "1 - (press == 'none')"),
      DocTransform('big', 'x > 15'),
    ]);
    expect(column(d, 'left').sublist(0, 4), ['1', '0', '0', '1']);
    expect(column(d, 'nothing').sublist(0, 4), ['0', '1', '0', '0']);
    expect(column(d, 'moved').sublist(0, 4), ['1', '0', '1', '1']);
    // A missing x stays missing through a comparison.
    expect(column(d, 'big').sublist(0, 6), ['0', '', '0', '0', '0', '1']);
  });

  test('mate follows the reshuffled teams', () {
    for (final time in [true, false]) {
      final d = apply([
        DocTransform('mate_x', 'mate(x)'),
        DocTransform('mate_press', 'mate(press)'),
        DocTransform('mate_left', "mate(press) == 'left'"),
      ], time: time);
      // a: b in block 1, c in block 2; b: a, then d; d: c, then b.
      expect(column(d, 'mate_x'), [
        '10', '20', '300', '400', //
        '1', '', '3000', '4000',
        '1000', '2000', '3', '4',
        '100', '200', '30', '40',
      ], reason: 'time column: $time');
      expect(column(d, 'mate_press').sublist(0, 4), [
        'right',
        'left',
        'left',
        'left',
      ]);
      expect(column(d, 'mate_left').sublist(0, 4), ['0', '1', '1', '1']);
    }
    // A second mate doesn't exist in teams of two.
    expect(column(apply([DocTransform('m2', 'mate(x, 2)')]), 'm2').toSet(), {
      '',
    });
  });

  test('errors name the transform and the problem', () {
    expect(
      () => apply([DocTransform('y', 'nope + 1')]),
      throwsA(
        isA<FormatException>().having(
          (e) => e.message,
          'message',
          allOf(contains('"y"'), contains('No column "nope"')),
        ),
      ),
    );
    expect(
      () => apply([DocTransform('x', 'press')]),
      throwsA(isA<FormatException>()),
    );
    expect(
      () => apply([DocTransform('y', 'lag(x, 0.5)')]),
      throwsA(isA<FormatException>()),
    );
    expect(
      () => applyTransforms(CsvData.parse(csv), [
        DocTransform('m', 'mate(x)'),
      ], participantColumn: 'participant'),
      throwsA(isA<FormatException>()),
    );
  });

  test('a document: transformed inputs, an expression choice by node ids', () {
    final doc = ModelDocument(
      name: 'beam',
      network: NetworkDocument(
        nodes: [
          ScalarNode('mate_left', kind: NodeKind.binaryState),
          ScalarNode('belief', kind: NodeKind.continuousState),
          ScalarNode(
            'ball',
            kind: NodeKind.continuousState,
            params: {NodeField.precision: 1},
          ),
          ScalarNode('ball_v', kind: NodeKind.continuousState),
        ],
        edges: const [
          DocEdge(EdgeType.value, 'belief', 'mate_left'),
          DocEdge(EdgeType.value, 'ball_v', 'ball'),
        ],
      ),
      transforms: [
        DocTransform('mate_left', "mate(press) == 'left'"),
        DocTransform('ball', 'x / 1000'),
      ],
      inputs: [
        DocInput('mate_left', node: 'mate_left'),
        DocInput('ball', node: 'ball'),
      ],
      response: DocResponse(
        'press',
        readout: {
          'type': 'ExprReadout',
          'kind': 'choice',
          'options': [
            'beta * belief.expected_mean',
            'beta * ball_v.expected_mean',
            'bias',
          ],
          'parameters': {'beta': 1.0, 'bias': 0.0},
        },
        options: ['left', 'right', 'none'],
      ),
      parameters: [
        {
          'name': 'beta',
          'targets': [
            {'type': 'response', 'name': 'beta'},
          ],
          'prior': const LogNormal(0, 1).toJson(),
          'transform': const LowerBoundTransform(0).toJson(),
        },
        {
          'name': 'bias',
          'targets': [
            {'type': 'response', 'name': 'bias'},
          ],
          'prior': const Normal(0, 1).toJson(),
          'transform': const IdentityTransform().toJson(),
        },
      ],
      data: const DocData(
        participantColumn: 'participant',
        timeColumn: 't',
        groupColumn: 'team',
      ),
    );
    // Round-trips through JSON.
    final json = jsonEncode(doc.toJson());
    final back = ModelDocument.fromJson(
      jsonDecode(json) as Map<String, Object?>,
    );
    expect(jsonEncode(back.toJson()), json);

    final compiled = back.compile();
    final readout = compiled.model!.response as ExprReadout;
    expect(readout.beliefs['belief.expected_mean'], (
      node: 1,
      field: NodeField.expectedMean,
    ));
    expect(readout.beliefs['ball_v.expected_mean']!.node, 3);

    final subjects = compiled.subjects(CsvData.parse(csv));
    expect(subjects.map((s) => s.id), ['a', 'b', 'c', 'd']);
    // a's mate_left: b presses right, left; c presses left, left.
    final a = subjects.first;
    expect([for (var t = 0; t < 4; t++) a.inputs[t * 2]], [0, 1, 1, 1]);
    expect(a.inputs[2 * 2 + 1], closeTo(0.003, 1e-15));
    expect(a.responses, [0, 2, 1, 0]);
    final nll = compiled.model!.negLogPosteriorAt(a, [1.5, 0.2]);
    expect(nll.isFinite, isTrue);

    // The exports name the derived columns.
    final spec = compiled.exportSpec();
    expect(spec.derivedColumns, {
      'mate_left': "mate(press) == 'left'",
      'ball': 'x / 1000',
    });
  });

  test('an unknown node.field in a readout is an error', () {
    final doc = ModelDocument(
      network: NetworkDocument(
        nodes: [ScalarNode('x', kind: NodeKind.continuousState)],
        edges: const [],
      ),
      response: DocResponse(
        'r',
        readout: {
          'type': 'ExprReadout',
          'kind': 'gaussian',
          'gaussian': {'mean': 'y.expected_mean', 'precision': '1'},
        },
      ),
    );
    expect(
      doc.compile,
      throwsA(
        isA<FormatException>().having(
          (e) => e.message,
          'message',
          contains('"y.expected_mean"'),
        ),
      ),
    );
  });
}
