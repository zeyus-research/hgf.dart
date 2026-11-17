/// A model document's Python export against hgf.dart: the generated reader
/// and data transforms give the same participants, the PyMC fit of every
/// participant lands on hgf_fit's MAP and Laplace intervals, the outputs
/// match, and the group level samples.
///
/// Needs `tool/parity/.venv` (`cd tool/parity && uv sync`); run with
/// `melos run test:export-python`.
@TestOn('vm')
@Tags(['pyhgf'])
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/codegen.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:test/test.dart';

final String _python = File('../../tool/parity/.venv/bin/python').absolute.path;
final Directory _dir = Directory('.dart_tool/document_export_python_test');

/// A 3-option choice on a ball's position with a team-mate's press as a
/// categorical input (first: pyhgf crashes on a categorical node after a
/// continuous one, P7), participants and time by column, two outputs.
ModelDocument _choice({DocFit fit = const DocFit(), DocGroup? group}) =>
    ModelDocument(
      name: 'press',
      network: NetworkDocument(
        nodes: [
          CategoricalNode('mate', nCategories: 3),
          ScalarNode('ball', kind: NodeKind.continuousState),
          ScalarNode(
            'ball_v',
            kind: NodeKind.continuousState,
            params: {NodeField.tonicVolatility: -3},
          ),
          ScalarNode('vol', kind: NodeKind.continuousState),
        ],
        edges: const [
          DocEdge(EdgeType.value, 'ball_v', 'ball'),
          DocEdge(EdgeType.volatility, 'vol', 'ball_v'),
        ],
      ),
      inputs: [
        DocInput('mate', node: 'mate', categories: ['left', 'right', 'none']),
        DocInput('ball', node: 'ball'),
      ],
      response: DocResponse(
        'press',
        readout: {
          'type': 'SoftmaxReadout',
          'options': [
            [
              {'node': 'ball_v', 'weight': 1.0},
            ],
            [
              {'node': 'ball_v', 'weight': -1.0},
            ],
            <Map<String, Object?>>[],
          ],
          'field': 'expected_mean',
          'inverseTemperature': 1.0,
          'stickiness': 0.5,
        },
        options: ['left', 'right', 'none'],
      ),
      parameters: [
        {
          'name': 'omega',
          'targets': [
            {'type': 'node', 'node': 'ball_v', 'field': 'tonic_volatility'},
          ],
          'prior': const Normal(-3, 1).toJson(),
          'transform': const IdentityTransform().toJson(),
        },
        {
          'name': 'beta',
          'targets': [
            {'type': 'response', 'name': 'inverseTemperature'},
          ],
          'prior': const LogNormal(0, 1).toJson(),
          'transform': const LowerBoundTransform(0).toJson(),
        },
      ],
      transforms: [DocTransform('ball_before', 'lag(ball)')],
      outputs: const [
        DocOutput('pe', node: 'ball_v', field: 'value_prediction_error'),
        DocOutput('surprise', node: 'ball', field: 'surprise'),
      ],
      data: const DocData(participantColumn: 'participant', timeColumn: 't'),
      fit: fit,
      group: group,
    );

String _choiceData() {
  final c = _choice().compile();
  final rng = math.Random(4);
  const nT = 80;
  final subjects = <Subject>[
    for (var s = 0; s < 3; s++)
      () {
        var x = 0.0;
        final u = Float64List(nT * 4);
        for (var t = 0; t < nT; t++) {
          x += 0.4 * (rng.nextDouble() - 0.5);
          u[t * 4 + rng.nextInt(3)] = 1;
          u[t * 4 + 3] = x;
        }
        return Subject(
          inputs: u,
          timeSteps: Float64List(nT)..fillRange(0, nT, 0.5),
          id: 'p${s + 1}',
        );
      }(),
  ];
  final sim = c.simulate(
    subjects,
    math.Random(5),
    x: [
      [-3.0, 2.0],
      [-2.5, 1.5],
      [-3.5, 3.0],
    ],
  );
  return '${c.toCsv(sim).toText()}\n';
}

/// Teams reshuffled between two blocks, presses as labels, a position with
/// gaps: what the transforms read.
const _teamCsv = '''
participant,block,team,t,press,pos
a,1,red,0,left,0.1
b,1,red,0,none,-0.2
c,1,blue,0,right,0.3
d,1,blue,0,left,
a,1,red,1,right,0.4
b,1,red,1,left,-0.6
c,1,blue,1,none,0.2
d,1,blue,1,,0.9
a,2,blue,2,none,1.5
c,2,blue,2,left,-1
b,2,red,2,right,0
d,2,red,2,none,0.25
a,2,blue,3,left,
c,2,blue,3,right,0.5
b,2,red,3,left,-0.1
d,2,red,3,right,2
''';

/// A network-only document over [_teamCsv], its input read from a
/// transform, with every data function in the transforms.
ModelDocument _team() => ModelDocument(
  name: 'teams',
  network: NetworkDocument(
    nodes: [
      ScalarNode('x', kind: NodeKind.continuousState),
      ScalarNode('x_v', kind: NodeKind.continuousState),
    ],
    edges: const [DocEdge(EdgeType.value, 'x_v', 'x')],
  ),
  inputs: [
    DocInput('x', node: 'x', columns: ['shaped']),
  ],
  transforms: [
    DocTransform('mate1', 'mate(press, 1)'),
    DocTransform('mate_left', "mate(press, 1) == 'left'"),
    DocTransform('mate2_pos', 'mate(pos, 2)'),
    DocTransform('nothing', "press != 'none'"),
    DocTransform('step', 'diff(pos)'),
    DocTransform('back2', 'lag(pos, 2)'),
    DocTransform('gap', 'missing(pos) + missing(press)'),
    DocTransform(
      'shaped',
      'where(pos > 0, 2 * tanh(pos / 2), fill(lag(pos), 0) ^ 2) + '
          'clip(pos, -0.5, 0.5) * sigmoid(pos) - min(pos, 0) * max(1, abs(pos))',
    ),
  ],
  outputs: const [DocOutput('belief', node: 'x_v', field: 'mean')],
  data: const DocData(
    participantColumn: 'participant',
    timeColumn: 't',
    groupColumn: 'team',
  ),
);

/// [script] written and run with [args]; its stdout.
Future<String> _runPython(
  String name,
  String script, [
  List<String>? args,
]) async {
  _dir.createSync(recursive: true);
  File('${_dir.path}/$name.py').writeAsStringSync(script);
  final run = await Process.run(_python, [
    '-I',
    '$name.py',
    ...?args,
  ], workingDirectory: _dir.path);
  expect(run.exitCode, 0, reason: '${run.stdout}\n${run.stderr}');
  return run.stdout as String;
}

/// Numbers and labels of two CSV texts equal up to [relTol].
void _expectCsv(String got, String want, {double relTol = 1e-9}) {
  final a = CsvData.parse(got), b = CsvData.parse(want);
  expect(a.header, b.header);
  expect(a.rows.length, b.rows.length);
  for (var r = 0; r < a.rows.length; r++) {
    for (var c = 0; c < a.header.length; c++) {
      final x = a.rows[r][c], y = b.rows[r][c];
      if (x == y) continue;
      final u = double.tryParse(x), v = double.tryParse(y);
      final where = 'row ${r + 2}, ${a.header[c]}';
      expect(u, isNotNull, reason: '$where: "$x" vs "$y"');
      expect(v, isNotNull, reason: '$where: "$x" vs "$y"');
      expect(
        (u! - v!).abs(),
        lessThanOrEqualTo(relTol * math.max(1e-12, v.abs())),
        reason: '$where: $u vs $v',
      );
    }
  }
}

void main() {
  final venv = File(_python).existsSync();
  final skip = venv ? false : 'needs tool/parity/.venv (uv sync)';

  test(
    'reads participants and computes transforms as hgf.dart does',
    () async {
      final c = _team().compile();
      final script = PythonExport(c.exportSpec(csv: _teamCsv)).script();
      _dir.createSync(recursive: true);
      File('${_dir.path}/teams.py').writeAsStringSync(script);
      const harness = '''
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("teams", "teams.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
table = m.transform(m.parse_csv(m.EMBEDDED_CSV))
ids, subjects = m.participants(table)
print(json.dumps(dict(
    csv=m._csv_text(table["header"], table["rows"]),
    ids=ids,
    subjects=[dict(
        inputs=s["inputs"].ravel().tolist(),
        observed=None if s["observed"] is None else s["observed"].ravel().tolist(),
        time_steps=None if s["time_steps"] is None else s["time_steps"].tolist(),
    ) for s in subjects],
)))
''';
      final out = await _runPython('teams_harness', harness);
      final got =
          jsonDecode(out.trim().split('\n').last) as Map<String, Object?>;
      final data = CsvData.parse(_teamCsv);
      _expectCsv(got['csv']! as String, c.transform(data).toText());
      final want = c.subjects(data);
      expect(got['ids'], [for (final s in want) s.id]);
      final subs = got['subjects']! as List<Object?>;
      for (final (k, s) in want.indexed) {
        final g = subs[k]! as Map<String, Object?>;
        List<double>? doubles(Object? v) =>
            (v as List<Object?>?)?.map((e) => (e! as num).toDouble()).toList();
        final inputs = doubles(g['inputs'])!;
        for (var i = 0; i < inputs.length; i++) {
          expect(inputs[i], closeTo(s.inputs[i], 1e-12), reason: '${s.id} $i');
        }
        expect(doubles(g['observed']), s.observed);
        expect(doubles(g['time_steps']), s.timeSteps);
      }

      // Run as a script: the outputs at the set values.
      await _runPython('teams', script);
      final outputs = File('${_dir.path}/data_outputs.csv').readAsStringSync();
      _expectCsv(outputs, c.withOutputs(data).toText());
    },
    skip: skip,
    timeout: const Timeout(Duration(minutes: 5)),
  );

  test(
    'MAP of every participant: hgf_fit\'s estimates, intervals and outputs',
    () async {
      final csv = _choiceData();
      final data = CsvData.parse(csv);
      final c = _choice().compile();
      final model = c.model!;
      final subjects = c.subjects(data);
      final map = fitMap(model, subjects, restarts: 3, random: math.Random(1));
      final want = ParticipantEstimates.fromLaplace(
        laplace(model, map),
        participants: [for (final s in subjects) '${s.id}'],
      );
      await _runPython('map', PythonExport(c.exportSpec(csv: csv)).script());
      final got = CsvData.parse(
        File('${_dir.path}/data_estimates.csv').readAsStringSync(),
      );
      expect(got.header, CsvData.parse(want.toCsv()).header);
      for (final (k, row) in got.rows.indexed) {
        expect(row.first, want.participants[k]);
        for (var i = 0; i < want.names.length; i++) {
          final name = want.names[i];
          expect(
            double.parse(row[1 + 3 * i]),
            closeTo(want.estimate[k][i], 1e-3),
            reason: '$name of ${row.first}',
          );
          expect(
            double.parse(row[2 + 3 * i]),
            closeTo(want.lower[k][i], 1e-2),
            reason: '$name lower of ${row.first}',
          );
          expect(
            double.parse(row[3 + 3 * i]),
            closeTo(want.upper[k][i], 1e-2),
            reason: '$name upper of ${row.first}',
          );
        }
      }
      // The outputs at the Python estimates, as hgf.dart filters them.
      final x = {
        for (final row in got.rows)
          row.first: [
            for (var i = 0; i < want.names.length; i++)
              double.parse(row[1 + 3 * i]),
          ],
      };
      _expectCsv(
        File('${_dir.path}/data_outputs.csv').readAsStringSync(),
        c.withOutputs(data, x: x).toText(),
        relTol: 1e-8,
      );
    },
    skip: skip,
    timeout: const Timeout(Duration(minutes: 5)),
  );

  test(
    'the group level samples the joint posterior',
    () async {
      final csv = _choiceData();
      final c = _choice(
        fit: const DocFit(method: 'nuts', chains: 2, warmup: 100, draws: 100),
        group: DocGroup(
          pooled: {'omega': DocPopulation.around(const Normal(-3, 1))},
          shared: ['beta'],
        ),
      ).compile();
      final out = await _runPython(
        'hier',
        PythonExport(c.exportSpec(csv: csv)).script(),
      );
      expect(out, contains('divergent transitions'));
      final text = File('${_dir.path}/data_estimates.csv').readAsStringSync();
      final tables = text.trim().split('\n\n');
      expect(tables, hasLength(2));
      final rows = CsvData.parse(tables[0]).rows;
      expect(rows.map((r) => r.first), ['p1', 'p2', 'p3']);
      // beta is shared: one value for everyone.
      expect(rows.map((r) => r[4]).toSet(), hasLength(1));
      final group = CsvData.parse(tables[1]);
      expect(group.rows.map((r) => r.first), [
        'omega_typical',
        'omega_spread',
        'beta',
      ]);
      // Near the truth (-3, -2.5, -3.5 for omega; beta 1.5–3).
      final typical = double.parse(group.rows[0][1]);
      expect(typical, inInclusiveRange(-5, 0));
      expect(double.parse(group.rows[2][1]), inInclusiveRange(1, 4));
    },
    skip: skip,
    timeout: const Timeout(Duration(minutes: 8)),
  );
}
