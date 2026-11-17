@TestOn('vm')
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/codegen.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:test/test.dart';

/// A 3-option choice (left / right / nothing) on a ball's position, a
/// team-mate's press as a categorical input, participants and time by
/// column, a transform and two outputs.
ModelDocument _doc({DocFit fit = const DocFit(), DocGroup? group}) =>
    ModelDocument(
      name: 'press',
      network: NetworkDocument(
        // The categorical node first: pyhgf crashes on one after a
        // continuous node (P7).
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
        DocInput('ball', node: 'ball'),
        DocInput('mate', node: 'mate', categories: ['left', 'right', 'none']),
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

/// Three participants' data, simulated from the document at their own
/// values, as CSV text.
String _data() {
  final c = _doc().compile();
  final rng = math.Random(4);
  const nT = 80;
  final subjects = <Subject>[];
  for (var s = 0; s < 3; s++) {
    var x = 0.0;
    final u = Float64List(nT * 4);
    for (var t = 0; t < nT; t++) {
      x += 0.4 * (rng.nextDouble() - 0.5);
      // The inputs in the network's order: mate (one-hot), ball.
      u[t * 4 + rng.nextInt(3)] = 1;
      u[t * 4 + 3] = x;
    }
    subjects.add(
      Subject(
        inputs: u,
        timeSteps: Float64List(nT)..fillRange(0, nT, 0.5),
        id: 'p${s + 1}',
      ),
    );
  }
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

final Directory _dir = Directory('.dart_tool/document_export_test');

/// [doc]'s Dart script with the data embedded, run; its output and the
/// files it wrote.
Future<({String stdout, String estimates, String outputs})> _run(
  String name,
  ModelDocument doc,
  String csv,
) async {
  _dir.createSync(recursive: true);
  final script = DartExport(doc.compile().exportSpec(csv: csv)).script();
  File('${_dir.path}/$name.dart').writeAsStringSync(script);
  for (final f in ['data_estimates.csv', 'data_outputs.csv']) {
    final file = File('${_dir.path}/$f');
    if (file.existsSync()) file.deleteSync();
  }
  final run = await Process.run(Platform.resolvedExecutable, [
    'run',
    '$name.dart',
  ], workingDirectory: _dir.path);
  expect(run.exitCode, 0, reason: '${run.stdout}\n${run.stderr}');
  String read(String f) {
    final file = File('${_dir.path}/$f');
    return file.existsSync() ? file.readAsStringSync() : '';
  }

  return (
    stdout: run.stdout as String,
    estimates: read('data_estimates.csv'),
    outputs: read('data_outputs.csv'),
  );
}

/// What the script should have written: the estimates' table and the data
/// with the outputs at the estimates.
void _expectMatches(
  ({String stdout, String estimates, String outputs}) got,
  CompiledDocument c,
  CsvData data,
  ParticipantEstimates want,
) {
  expect(got.stdout, contains(want.toString()));
  expect(got.estimates, '${want.toCsv()}\n');
  expect(
    got.outputs,
    '${c.withOutputs(data, x: want.byParticipant).toText()}\n',
  );
}

void main() {
  final csv = _data();
  final data = CsvData.parse(csv);

  test('the layout reads what the document reads', () {
    final c = _doc().compile();
    final a = c.subjects(data), b = c.layout.subjects(data);
    expect(a, hasLength(3));
    for (var k = 0; k < a.length; k++) {
      expect(b[k].id, a[k].id);
      expect(b[k].inputs, a[k].inputs);
      expect(b[k].observed, a[k].observed);
      expect(b[k].timeSteps, a[k].timeSteps);
      expect(
        b[k].responses!.map((v) => v.isNaN ? -1 : v),
        a[k].responses!.map((v) => v.isNaN ? -1 : v),
      );
    }
  });

  group('Dart scripts of every participant', () {
    test('analyse clean', () async {
      _dir.createSync(recursive: true);
      final scripts = {
        'plain': _doc(fit: const DocFit(method: 'none')),
        'eb': _doc(
          group: DocGroup(
            pooled: {'omega': DocPopulation.around(const Normal(-3, 1))},
          ),
        ),
        'hier': _doc(
          fit: const DocFit(method: 'nuts', chains: 2, warmup: 50, draws: 50),
          group: DocGroup(
            pooled: {
              'omega': DocPopulation.around(TruncatedNormal(-3, 1, upper: 0)),
            },
            shared: ['beta'],
            centered: true,
          ),
        ),
      };
      for (final MapEntry(:key, :value) in scripts.entries) {
        File('${_dir.path}/analyse_$key.dart').writeAsStringSync(
          DartExport(value.compile().exportSpec(csv: csv)).script(),
        );
      }
      final r = await Process.run(Platform.resolvedExecutable, [
        'analyze',
        '--fatal-infos',
        for (final k in scripts.keys) '${_dir.path}/analyse_$k.dart',
      ]);
      expect(r.exitCode, 0, reason: '${r.stdout}\n${r.stderr}');
    });

    test('without a fit: the outputs at the set values', () async {
      final doc = _doc(fit: const DocFit(method: 'none'));
      final got = await _run('plain', doc, csv);
      expect(got.stdout, contains('3 participant(s), 240 rows.'));
      expect(got.stdout, contains('p2: 80 rows'));
      expect(got.estimates, isEmpty);
      final c = doc.compile();
      final set = c.currentValues();
      expect(
        got.outputs,
        '${c.withOutputs(data, x: {
          for (final id in ['p1', 'p2', 'p3']) id: set,
        }).toText()}\n',
      );
    });

    test('MAP per participant', () async {
      final doc = _doc();
      final c = doc.compile();
      final model = c.model!;
      final subjects = c.subjects(data);
      final map = fitMap(model, subjects, restarts: 3, random: math.Random(1));
      final want = ParticipantEstimates.fromLaplace(
        laplace(model, map),
        participants: ['p1', 'p2', 'p3'],
      );
      _expectMatches(await _run('map', doc, csv), c, data, want);
    });

    test('NUTS per participant', () async {
      final doc = _doc(
        fit: const DocFit(method: 'nuts', chains: 2, warmup: 60, draws: 60),
      );
      final c = doc.compile();
      final model = c.model!;
      final subjects = c.subjects(data);
      final map = fitMap(model, subjects, restarts: 3, random: math.Random(1));
      final nuts = sampleNuts(
        model,
        subjects,
        sampler: const Nuts(),
        chains: 2,
        warmup: 60,
        draws: 60,
        start: [
          for (final r in map.results)
            r.negLogPosterior.isFinite ? r.x : model.start(),
        ],
        jitter: 0.5,
      );
      final want = ParticipantEstimates.fromNuts(
        nuts,
        participants: ['p1', 'p2', 'p3'],
      );
      _expectMatches(await _run('nuts', doc, csv), c, data, want);
    });

    test('empirical Bayes', () async {
      final doc = _doc(
        group: DocGroup(
          pooled: {'omega': DocPopulation.around(const Normal(-3, 1))},
        ),
      );
      final c = doc.compile();
      final model = c.model!;
      final subjects = c.subjects(data);
      final map = fitMap(model, subjects, restarts: 3, random: math.Random(1));
      final eb = fitEmpiricalBayes(
        c.hierarchical!,
        subjects,
        start: [
          for (final r in map.results)
            r.negLogPosterior.isFinite ? r.x : model.start(),
        ],
      );
      final want = ParticipantEstimates.fromEmpiricalBayes(
        eb,
        participants: ['p1', 'p2', 'p3'],
      );
      expect(want.group.map((g) => g.kind), [
        GroupRow.typical,
        GroupRow.spread,
      ]);
      final got = await _run('eb', doc, csv);
      _expectMatches(got, c, data, want);
      expect(got.estimates, contains('omega_typical'));
    });

    test('hierarchical NUTS, pooled and shared', () async {
      final doc = _doc(
        fit: const DocFit(method: 'nuts', chains: 2, warmup: 60, draws: 60),
        group: DocGroup(
          pooled: {'omega': DocPopulation.around(const Normal(-3, 1))},
          shared: ['beta'],
        ),
      );
      final c = doc.compile();
      final model = c.model!;
      final subjects = c.subjects(data);
      final map = fitMap(model, subjects, restarts: 3, random: math.Random(1));
      final hm = c.hierarchical!;
      final fit = sampleHierarchical(
        hm,
        subjects,
        chains: 2,
        warmup: 60,
        draws: 60,
        start: [
          for (final r in map.results)
            r.negLogPosterior.isFinite ? r.x : model.start(),
        ],
        jitter: 0.5,
      );
      final want = ParticipantEstimates.fromHierarchical(
        hm,
        fit,
        subjects,
        participants: ['p1', 'p2', 'p3'],
      );
      final got = await _run('hier', doc, csv);
      _expectMatches(got, c, data, want);
      expect(got.estimates, contains('\nbeta,'));
    });
  });
}
