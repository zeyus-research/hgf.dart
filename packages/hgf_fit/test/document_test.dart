@TestOn('vm')
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

import 'src/fixture_networks.dart';
import 'src/traj_hash.dart';

void main() {
  group('NetworkDocument', () {
    for (final fx in fixtureNetworks()) {
      test('round-trips ${fx.name}', () {
        final doc = NetworkDocument.fromNetwork(fx.network);
        final back = NetworkDocument.fromJson(
          jsonDecode(jsonEncode(doc.toJson())) as Map<String, Object?>,
        );
        expect(jsonEncode(back.toJson()), jsonEncode(doc.toJson()));
        final net = back.build(couplings: fixtureCouplings);
        expect(_comparable(net.toJson()), _comparable(fx.network.toJson()));
        expect(
          trajectoryHash(
            net,
            fx.data.inputs,
            observed: fx.data.observed,
            timeSteps: fx.data.timeSteps,
          ),
          trajectoryHash(
            fx.network,
            fx.data.inputs,
            observed: fx.data.observed,
            timeSteps: fx.data.timeSteps,
          ),
        );
      });
    }
  });

  group('ModelDocument', () {
    for (final fx in fixtureNetworks().where((f) => !f.perturbed)) {
      test('reads ${fx.name} back from the exported CSV', () {
        final plan = NetworkPlan.fromJson(fx.network.toJson());
        final data = fx.data;
        final cols = ExportData.columns(
          plan,
          observed: data.observed != null,
          timeSteps: data.timeSteps != null,
        );
        final base = ModelDocument.fromNetwork(fx.network, name: fx.name);
        final doc = ModelDocument(
          network: base.network,
          name: base.name,
          inputs: [
            for (var k = 0; k < base.inputs.length; k++)
              DocInput(
                base.inputs[k].name,
                node: base.inputs[k].node,
                observedColumn: data.observed == null ? null : cols.observed[k],
              ),
          ],
          data: DocData(timeStepColumn: cols.timeStep),
        );
        // The names the export writes are the document's defaults.
        expect(
          [for (final i in base.inputs) i.columnsFor(1).first],
          [
            for (final c in cols.inputs)
              c.length == 1
                  ? c.single
                  : c.first.replaceFirst(RegExp(r'_0$'), ''),
          ],
        );
        final json = jsonEncode(doc.toJson());
        final back = ModelDocument.fromJson(
          jsonDecode(json) as Map<String, Object?>,
        );
        expect(jsonEncode(back.toJson()), json);

        final compiled = back.compile(
          codec: FitCodec(couplings: fixtureCouplings),
        );
        final subjects = compiled.subjects(CsvData.parse(data.toCsv(plan)));
        expect(subjects, hasLength(1));
        final s = subjects.single;
        // Empty cells read as 0 (a NaN would reach the beliefs).
        final u = Float64List.fromList([
          for (final v in data.inputs) v.isNaN ? 0.0 : v,
        ]);
        expect(s.inputs, u);
        if (data.observed case final o?) {
          expect(s.observed, o);
        } else {
          expect(s.observed, isNull);
        }
        expect(s.timeSteps, data.timeSteps);
        expect(
          trajectoryHash(
            compiled.network,
            s.inputs,
            observed: s.observed,
            timeSteps: s.timeSteps,
          ),
          trajectoryHash(
            fx.network,
            u,
            observed: data.observed,
            timeSteps: data.timeSteps,
          ),
        );
      });
    }

    test('fromModel round-trips the IGT model', () {
      final model = HgfModel(
        network: IowaGamblingTask.network(),
        response: IowaGamblingTask.readout(),
        parameters: [
          Parameter(
            'omega',
            targets: [
              for (final n in IowaGamblingTask.valueNodes)
                NodeTarget(n, NodeField.tonicVolatility),
            ],
            prior: const Normal(-2, 1.5),
          ),
          Parameter.response(
            'inverseTemperature',
            prior: const LogNormal(0.5, 1),
          ),
        ],
      );
      final doc = ModelDocument.fromModel(model);
      // Targets and readout terms name nodes by address.
      expect(
        jsonEncode(doc.parameters.first['targets']),
        contains('"node":"n4"'),
      );
      final back = ModelDocument.fromJson(
        jsonDecode(jsonEncode(doc.toJson())) as Map<String, Object?>,
      );
      final rebuilt = back.compile().model!;
      const codec = FitCodec();
      final want = codec.encodeModel(model), got = codec.encodeModel(rebuilt);
      expect(jsonEncode(got['response']), jsonEncode(want['response']));
      expect(jsonEncode(got['parameters']), jsonEncode(want['parameters']));
      expect(
        _comparable(got['network'] as Map<String, Object?>),
        _comparable(want['network'] as Map<String, Object?>),
      );
      final rng = math.Random(3);
      final subject = model
          .simulateChoices(
            [Subject(inputs: IowaGamblingTask.outcomes(80, rng))],
            [
              [-2.5, 1.5],
            ],
            rng,
            reveals: IowaGamblingTask.reveals,
          )
          .single;
      expect(
        rebuilt.negLogPosteriorAt(subject, [-2.0, 2.0]),
        model.negLogPosteriorAt(subject, [-2.0, 2.0]),
      );
    });

    group('a three-option choice', () {
      ModelDocument doc({
        List<DocEdge>? extraEdges,
        List<DocInput>? inputs,
      }) => ModelDocument(
        name: 'press',
        network: NetworkDocument(
          nodes: [
            ScalarNode('ball', kind: NodeKind.continuousState),
            ScalarNode(
              'ball_v',
              kind: NodeKind.continuousState,
              params: {NodeField.tonicVolatility: -3},
            ),
            ScalarNode('vol', kind: NodeKind.continuousState),
            CategoricalNode('mate', nCategories: 3),
          ],
          edges: [
            const DocEdge(EdgeType.value, 'ball_v', 'ball'),
            const DocEdge(EdgeType.volatility, 'vol', 'ball_v'),
            ...?extraEdges,
          ],
        ),
        inputs:
            inputs ??
            [
              DocInput('ball', node: 'ball'),
              DocInput(
                'mate',
                node: 'mate',
                categories: ['left', 'right', 'none'],
              ),
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
              [
                {'node': 'ball_v', 'weight': 0.0},
              ],
            ],
            'field': 'expected_mean',
            'inverseTemperature': 1.0,
          },
          options: ['left', 'right', 'none'],
          reveals: {
            'left': ['ball'],
            'right': ['ball', 'mate'],
          },
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
        outputs: const [
          DocOutput('pe', node: 'ball_v', field: 'value_prediction_error'),
          DocOutput('surprise', node: 'ball', field: 'surprise'),
        ],
        data: const DocData(participantColumn: 'participant', timeColumn: 't'),
      );

      const csv = """
participant,t,ball,mate,press
p1,0.0,0.1,left,left
p1,0.5,,right,none
p1,1.0,0.3,,right
p2,0.0,1,none,left
p2,0.25,2,0,
""";

      test('reads subjects by column name', () {
        final c = doc().compile();
        expect(c.inputs.map((i) => i.name), ['ball', 'mate']);
        expect(c.reveals, [
          [0],
          [0, 1],
          <int>[],
        ]);
        final s = c.subjects(CsvData.parse(csv));
        expect(s.map((x) => x.id), ['p1', 'p2']);
        expect(s[0].inputs, [0.1, 1, 0, 0, 0, 0, 1, 0, 0.3, 0, 0, 0]);
        expect(s[0].observed, [1, 1, 0, 1, 1, 0]);
        expect(s[0].timeSteps, [0.5, 0.5, 0.5]);
        expect(s[0].responses, [0, 2, 1]);
        expect(s[1].inputs, [1, 0, 0, 1, 2, 1, 0, 0]);
        expect(s[1].observed, isNull);
        expect(s[1].timeSteps, [0.25, 0.25]);
        expect(s[1].responses![0], 0);
        expect(s[1].responses![1].isNaN, isTrue);

        final model = c.model!;
        expect(model.parameters.map((p) => p.name), ['omega', 'beta']);
        expect(model.negLogPosteriorAt(s[0], [-3, 1]).isFinite, isTrue);

        final traj = c.network.inputData(
          s[0].inputs,
          observed: s[0].observed,
          timeSteps: s[0].timeSteps,
          fields: c.recordedFields,
        )!;
        final out = c.outputs(traj);
        expect(out.keys, ['pe', 'surprise']);
        expect(out['pe'], hasLength(3));
      });

      test('exports a script that reads the named columns', () async {
        final c = doc().compile();
        final model = c.model!;
        final rng = math.Random(4);
        const nT = 150;
        var x = 0.0;
        final u = Float64List(nT * 4);
        for (var t = 0; t < nT; t++) {
          x += 0.3 * (rng.nextDouble() - 0.5);
          u[t * 4] = x;
          u[t * 4 + 1 + rng.nextInt(3)] = 1;
        }
        final responses = model
            .simulate(
              [Subject(inputs: u)],
              [
                [-3.0, 2.0],
              ],
              math.Random(5),
            )
            .single;
        final spec = c.exportSpec(
          data: ExportData(inputs: u, responses: responses),
          dataFile: 'press.csv',
        );
        final csv = spec.data!.toCsv(spec.plan, names: c.columnNames);
        expect(
          csv.split('\n').first,
          'ball,mate_left,mate_right,mate_none,press',
        );
        expect(PythonExport(spec).script(), contains('"mate_none"'));
        final dir = Directory('.dart_tool/document_test')
          ..createSync(recursive: true);
        File(
          '${dir.path}/press.dart',
        ).writeAsStringSync(DartExport(spec).script());
        File('${dir.path}/press.csv').writeAsStringSync(csv);
        final run = await Process.run(Platform.resolvedExecutable, [
          'run',
          'press.dart',
        ], workingDirectory: dir.path);
        expect(run.exitCode, 0, reason: '${run.stdout}\n${run.stderr}');
        final subject = Subject(inputs: u, responses: responses);
        final map = fitMap(
          model,
          [subject],
          restarts: 3,
          random: math.Random(1),
        );
        expect(
          run.stdout as String,
          contains(laplace(model, map).results.single.toString()),
        );
      });

      test('writes subjects back as CSV', () {
        final c = doc().compile();
        final s = c.subjects(CsvData.parse(csv));
        final out = c.toCsv(s);
        expect(out.header, ['participant', 't', 'ball', 'mate', 'press']);
        expect(out.rows.first, ['p1', '0', '0.1', 'left', 'left']);
        expect(out.rows[1], ['p1', '0.5', '', 'right', 'none']);
        final back = c.subjects(out);
        for (var k = 0; k < s.length; k++) {
          expect(back[k].id, s[k].id);
          expect(back[k].inputs, s[k].inputs);
          expect(back[k].observed, s[k].observed);
          expect(back[k].timeSteps, s[k].timeSteps);
          expect(
            back[k].responses!.map((v) => v.isNaN ? -1 : v),
            s[k].responses!.map((v) => v.isNaN ? -1 : v),
          );
        }
      });

      test('simulates choices and filters at parameter values', () {
        final c = doc().compile();
        final s = c.subjects(CsvData.parse(csv));
        final sim = c.simulate(s, math.Random(3));
        for (final x in sim) {
          expect(x.responses!.every((v) => v == 0 || v == 1 || v == 2), true);
        }
        // The choices decide what was seen: "none" reveals nothing.
        for (var t = 0; t < 3; t++) {
          if (sim[0].responses![t] == 2) {
            expect(sim[0].observed!.sublist(2 * t, 2 * t + 2), [0, 0]);
          }
        }
        final back = c.subjects(c.toCsv(sim));
        expect(back[0].responses, sim[0].responses);

        final i = c.document.network.addresses.indexOf('ball_v');
        expect(
          c.networkAt([-2, 1]).getInitial(i, NodeField.tonicVolatility),
          -2,
        );
        expect(
          c.networkAt().getInitial(i, NodeField.tonicVolatility),
          c.model!.start()[0],
        );
      });

      test('appends the outputs to the data, row by row', () {
        final c = doc().compile();
        final data = CsvData.parse(csv);
        final out = c.withOutputs(
          data,
          x: {
            'p2': [-1, 1],
          },
        );
        expect(out.header, [...data.header, 'pe', 'surprise']);
        expect(out.rows, hasLength(data.rows.length));
        final s = c.subjects(data);
        final p1 = c.outputs(c.filter(s[0]));
        final p2 = c.outputs(c.filter(s[1], [-1, 1]));
        expect(double.parse(out.rows[2][5]), p1['pe']![2]);
        expect(double.parse(out.rows[4][6]), p2['surprise']![1]);
      });

      test('checks the pipeline against the data', () {
        final cols = CsvData.parse(csv).header;
        expect(checkModel(doc(), columns: cols), isEmpty);
        final missing = checkModel(doc(), columns: [...cols]..remove('ball'));
        expect(missing.single.parts, [DocPart.input('ball')]);
        expect(missing.single.message, contains('No column "ball"'));

        final bad = doc().copyWith(
          transforms: [DocTransform('fast', 'speed * 2')],
          outputs: const [DocOutput('pe', node: 'nope', field: 'mean')],
        );
        final issues = checkModel(bad, columns: cols);
        expect(issues.map((i) => i.parts.single), [
          DocPart.transform('fast'),
          DocPart.output('pe'),
        ]);

        final stale = doc().withNetwork(doc().network.withoutNode('ball_v'));
        // The parameter and output on ball_v go; the readout is flagged.
        expect(stale.parameters.map((p) => p['name']), ['beta']);
        expect(stale.outputs.map((o) => o.name), ['surprise']);
        // vol lost its child: an input now, without data; and the readout
        // reads a node that is gone.
        expect(
          checkModel(
            stale,
            columns: cols,
          ).where((i) => i.isError).map((i) => i.parts.single),
          [DocPart.network, DocPart.response],
        );
      });

      test('keeps references when nodes and columns are renamed', () {
        final d = doc();
        final net = d.network.replaceNode(
          'ball_v',
          ScalarNode(
            'speed',
            kind: NodeKind.continuousState,
            params: const {NodeField.tonicVolatility: -3},
          ),
        );
        final r = d.withNetwork(net, renamedIds: {'ball_v': 'speed'});
        expect(jsonEncode(r.response!.readout), contains('"speed"'));
        expect(jsonEncode(r.response!.readout), isNot(contains('ball_v')));
        expect(jsonEncode(r.parameters), contains('"speed"'));
        expect(r.outputs.first.node, 'speed');
        expect(checkModel(r), isEmpty);

        // A new childless node is an input, and gets one.
        final more = r.network.withNode(
          ScalarNode('extra', kind: NodeKind.continuousState),
        );
        final withExtra = r.withNetwork(more).withInputsFor(more.build());
        expect(withExtra.inputs.map((i) => i.name), ['ball', 'mate', 'extra']);
        expect(withExtra.withInputsFor(more.build()), same(withExtra));

        // Renaming a transform renames what reads it.
        final t = d
            .copyWith(
              transforms: [DocTransform('ball_x', 'ball * 2')],
              inputs: [
                DocInput('ball', node: 'ball', columns: ['ball_x']),
                d.inputs[1],
              ],
            )
            .withTransform(DocTransform('ball_y', 'ball_x + 1'))
            .replaceTransform('ball_x', DocTransform('scaled', 'ball * 2'));
        expect(t.inputs.first.columns, ['scaled']);
        expect('${t.transforms.last.expression}', 'scaled + 1');

        // An ExprReadout's beliefs follow a rename.
        final readout = renameReadout({
          'type': 'ExprReadout',
          'kind': 'choice',
          'options': ['beta * ball_v.expected_mean', '0'],
          'parameters': {'beta': 1.0},
        }, (a) => a == 'ball_v' ? 'speed' : a);
        expect(readout['options'], ['beta * speed.expected_mean', '0']);
      });

      test('round-trips through JSON', () {
        final d = doc();
        final json = jsonEncode(d.toJson());
        expect(
          jsonEncode(
            ModelDocument.fromJson(
              jsonDecode(json) as Map<String, Object?>,
            ).toJson(),
          ),
          json,
        );
      });

      test('says what is wrong', () {
        final c = doc().compile();
        expect(
          () => c.subjects(
            CsvData.parse('participant,t,ball,press\np,0,1,left\n'),
          ),
          throwsA(
            isA<FormatException>().having(
              (e) => e.message,
              'message',
              contains('No column "mate"'),
            ),
          ),
        );
        expect(
          () => c.subjects(
            CsvData.parse('participant,t,ball,mate,press\np,0,1,up,left\n'),
          ),
          throwsA(
            isA<FormatException>().having(
              (e) => e.message,
              'message',
              contains('"up" is not one of left, right, none'),
            ),
          ),
        );
        // A categorical node declares no edges: one after it can't be its child.
        expect(
          () => doc(
            extraEdges: [const DocEdge(EdgeType.value, 'mate', 'vol')],
          ).compile(),
          throwsA(isA<FormatException>()),
        );
        expect(
          () => doc(
            inputs: [
              DocInput('ball', node: 'ball_v'),
              DocInput('mate', node: 'mate'),
            ],
          ).compile(),
          throwsA(
            isA<FormatException>().having(
              (e) => e.message,
              'message',
              contains('not an input node'),
            ),
          ),
        );
        expect(
          () => doc(inputs: [DocInput('ball', node: 'ball')]).compile(),
          throwsA(
            isA<FormatException>().having(
              (e) => e.message,
              'message',
              contains('"mate" has no named input'),
            ),
          ),
        );
      });
    });
  });
}

/// A snapshot without the declared parameters (the compiled initial values
/// carry them), as `NetworkPlan` compares.
String _comparable(Map<String, Object?> json) => jsonEncode({
  ...json,
  'nodes': [
    for (final n in json['nodes'] as List<Object?>)
      {
        for (final MapEntry(:key, :value)
            in (n as Map<String, Object?>).entries)
          if (key != 'params' && key != 'xis') key: value,
      },
  ],
});
