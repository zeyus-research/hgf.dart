import 'dart:convert';
import 'dart:math' as math;

import 'package:hgf/hgf.dart';
import 'package:hgf_deep/datasets.dart';
import 'package:hgf_deep/hgf_deep.dart';
import 'package:hgf_fit/document.dart';
import 'package:test/test.dart';

/// Two participants, two features, a binary target (x1 + x2 > 0), a
/// label column (left / right / none), a regression target and a split.
CsvData table({int n = 120}) {
  final r = math.Random(3);
  final rows = <List<String>>[];
  for (var p = 1; p <= 2; p++) {
    for (var t = 0; t < n; t++) {
      final x1 = 2 * r.nextDouble() - 1, x2 = 2 * r.nextDouble() - 1;
      rows.add([
        'p$p',
        '$t',
        // A gap: the network skips this row.
        t == 5 ? '' : '$x1',
        '$x2',
        x1 + x2 > 0 ? '1' : '0',
        x1 > 0.3
            ? 'right'
            : x1 < -0.3
            ? 'left'
            : 'none',
        '${math.sin(2 * x1) + 0.1 * x2}',
        t < n * 3 ~/ 4 ? 'train' : 'test',
      ]);
    }
  }
  return CsvData([
    'participant',
    'time',
    'x1',
    'x2',
    'y',
    'press',
    'z',
    'split',
  ], rows);
}

const scope = (
  participantColumn: 'participant',
  groupColumn: null,
  timeColumn: 'time',
);

DocDeep classifier({
  DocDeepTraining training = const DocDeepTraining(epochs: 3),
}) => DocDeep.starter(
  'clf',
).copyWith(features: ['x1', 'x2'], targets: ['y'], training: training);

void main() {
  group('wiring', wiring);
  final data = table();

  test('reads the rows that hold every column, in order', () {
    final d = DeepData.read(classifier(), data, scope: scope);
    expect(d.length, data.rows.length - 2);
    expect(d.rows, isNot(contains(5)));
    expect(d.rows, isNot(contains(125)));
    expect(d.x.cols, 2);
    expect(d.y.cols, 1);
    expect(d.participants.first, 'p1');
    expect(d.participants.last, 'p2');
    expect(d.train.every((t) => t), isTrue);
  });

  test('trainOn picks the training rows; the rest are held out', () {
    final deep = classifier(
      training: DocDeepTraining(
        epochs: 2,
        trainOn: Expr.parse("split == 'train'"),
      ),
    );
    final d = DeepData.read(deep, data, scope: scope);
    expect(d.trainIndices.length, 2 * 90 - 2);
    expect(d.testIndices.length, 2 * 30);
    final r = DeepTraining.run(deep, data, scope: scope);
    expect(r.trainScore!.name, 'accuracy');
    expect(r.testScore, isNotNull);
    expect(r.epochScores, hasLength(2));
    // The standardiser is fitted on the training rows only.
    final train = [for (final i in d.trainIndices) d.x.at(i, 1)];
    final mean = train.reduce((a, b) => a + b) / train.length;
    expect(r.scale!.mean[1], closeTo(mean, 1e-12));
  });

  test('trainOn round-trips', () {
    final t = DocDeepTraining(trainOn: Expr.parse("split == 'train'"));
    final back = DocDeepTraining.fromJson(
      jsonDecode(jsonEncode(t.toJson())) as Map<String, Object?>,
    );
    expect('${back.trainOn}', '${t.trainOn}');
    expect(t.copyWith(trainOn: () => null).trainOn, isNull);
  });

  group('training a piece at a time gives what training at once gives', () {
    final cases = <String, DocDeep>{
      'fit, Adam, shuffled': classifier(),
      'fit, SGD, in order': classifier(
        training: const DocDeepTraining(
          epochs: 2,
          optimiser: 'sgd',
          shuffle: false,
        ),
      ),
      'minibatches': classifier(
        training: const DocDeepTraining(epochs: 3, batchSize: 16),
      ),
      'synaptic uncertainty': classifier(
        training: const DocDeepTraining(
          epochs: 2,
          learningKind: LearningKind.synapticUncertainty,
        ),
      ),
      'categorical target': DocDeep(
        id: 'press',
        layers: const [
          DocLayer(size: 3, kind: LayerKind.categorical),
          DocLayer(size: 6, coupling: 'tanh'),
          DocLayer(size: 2),
        ],
        features: ['x1', 'x2'],
        targets: ['press'],
        categories: ['left', 'right', 'none'],
        training: const DocDeepTraining(epochs: 2),
      ),
      'continuous filter': DocDeep(
        id: 'filter',
        layers: const [
          DocLayer(size: 1, kind: LayerKind.continuous),
          DocLayer(size: 1, kind: LayerKind.continuous),
        ],
        targets: ['z'],
      ),
    };
    for (final MapEntry(key: name, value: deep) in cases.entries) {
      test(name, () {
        final once = DeepTraining.run(deep, data, scope: scope);
        final run = DeepTraining(deep, data, scope: scope);
        var calls = 0;
        while (!run.advance(7)) {
          calls++;
          expect(run.progress, inInclusiveRange(0, 1));
        }
        // Synaptic learning takes a whole pass per call.
        expect(calls, greaterThan(name == 'synaptic uncertainty' ? 0 : 3));
        expect(run.result!.predictions.data, once.predictions.data);
        expect(
          jsonEncode(run.result!.network.toJson()),
          jsonEncode(once.network.toJson()),
        );
      });
    }
  });

  test('matches the steps of an exported script', () {
    final deep = classifier(
      training: const DocDeepTraining(epochs: 3, init: DeepInit.he, seed: 4),
    );
    final r = DeepTraining.run(deep, data, scope: scope);
    // As DeepDartExport's script does it, by hand.
    final d = r.data;
    final net = deep.build()..weightInitialisation(const HeInit(), seed: 4);
    final scale = Standardiser.fit(d.x);
    final x = scale.apply(d.x);
    final train = Dataset('train', x, d.y);
    final adam = Adam(0.01);
    for (var epoch = 1; epoch <= 3; epoch++) {
      final rows = shuffled(train, seed: epoch);
      net.fit(rows.x, rows.y, optimiser: adam);
    }
    expect(r.predictions.data, net.predict(x).data);
  });

  test('a continuous network filters each participant afresh', () {
    final deep = DocDeep(
      id: 'filter',
      layers: const [
        DocLayer(size: 1, kind: LayerKind.continuous),
        DocLayer(size: 1, kind: LayerKind.continuous),
      ],
      targets: ['z'],
    );
    final r = DeepTraining.run(deep, data, scope: scope);
    final second = [
      for (var i = 0; i < r.data.length; i++)
        if (r.data.participants[i] == 'p2') i,
    ];
    final alone = deep.build().inputData(
      Matrix(second.length, 1)
        ..data.setAll(0, [for (final i in second) r.data.y.at(i, 0)]),
    );
    expect([for (final i in second) r.predictions.at(i, 0)], alone.data);
    expect(r.trainScore, isNull);
  });

  test('appends its predictions as columns', () {
    final r = DeepTraining.run(classifier(), data, scope: scope);
    final out = r.appendTo(data);
    expect(out.header.last, 'clf_prediction');
    expect(out.rows[5].last, '');
    expect(double.parse(out.rows[0].last), r.predictions.at(0, 0));
    final cols = r.columnsFor(data.rows.length);
    expect(cols['clf_prediction']![5], isNaN);
  });

  test('the key follows the columns read, not the others', () {
    final deep = classifier();
    final k = DeepTraining.keyOf(deep, data, scope: scope);
    final other = CsvData(data.header, [
      for (final row in data.rows) [...row]..[6] = '0',
    ]);
    expect(DeepTraining.keyOf(deep, other, scope: scope), k);
    final changed = CsvData(data.header, [
      for (final row in data.rows) [...row]..[3] = '0',
    ]);
    expect(DeepTraining.keyOf(deep, changed, scope: scope), isNot(k));
    expect(
      DeepTraining.keyOf(
        deep.copyWith(training: const DocDeepTraining(epochs: 4)),
        data,
        scope: scope,
      ),
      isNot(k),
    );
  });
}

/// An HGF network whose input reads a deep network's prediction.
ModelDocument wired({List<DocTransform> transforms = const []}) =>
    ModelDocument(
      network: NetworkDocument(
        nodes: [
          ScalarNode('u', kind: NodeKind.continuousState),
          ScalarNode('v', kind: NodeKind.continuousState),
        ],
        edges: [const DocEdge(EdgeType.value, 'v', 'u')],
      ),
      inputs: [
        DocInput('pred', node: 'u', columns: ['clf_prediction']),
      ],
      data: DocData(
        participantColumn: 'participant',
        timeColumn: 'time',
        steps: [SplitStep(seed: 2)],
      ),
      transforms: transforms,
      deep: [
        classifier(
          training: DocDeepTraining(
            epochs: 2,
            trainOn: Expr.parse("split == 'train'"),
          ),
        ),
        DocDeep.starter(
          'unused',
        ).copyWith(features: ['x1', 'x2'], targets: ['y']),
      ],
    );

void wiring() {
  final data = table(n: 60);

  test('the HGF network reads a deep network\'s predictions', () {
    final doc = wired();
    expect(
      checkModel(doc, columns: data.header).where((i) => i.isError),
      isEmpty,
    );
    final c = doc.compile();
    expect([for (final d in c.layout.deep) d.id], ['clf']);
    final subs = c.subjects(data);
    expect(subs, hasLength(2));
    // The input is the prediction, row by row (the gap unobserved).
    final t = c.transform(data);
    final col = t.header.indexOf('clf_prediction');
    final p1 = [
      for (final row in t.rows)
        if (row[0] == 'p1') row[col],
    ];
    expect(subs.first.inputs[0], double.parse(p1[0]));
    expect(p1[5], '');
    expect(subs.first.observed![5], 0);
  });

  test('a cache that does not train leaves the columns empty until given', () {
    final c = wired().compile();
    final cache = TransformCache()..trainDeep = false;
    var subs = c.subjects(data, cache: cache);
    expect(cache.pendingDeep.single.deep.id, 'clf');
    expect(subs.first.observed!.every((o) => o == 0), isTrue);
    final p = cache.pendingDeep.single;
    final r = DeepTraining.run(p.deep, p.table, scope: c.layout.scope);
    expect(r.key, p.key);
    cache.putDeep(r);
    subs = c.subjects(data, cache: cache);
    expect(cache.pendingDeep, isEmpty);
    expect(cache.usedDeep.keys, ['clf']);
    expect(subs.first.inputs, c.subjects(data).first.inputs);
  });

  test('checks: transforms run before the deep networks', () {
    final doc = wired(
      transforms: [DocTransform('twice', 'clf_prediction * 2')],
    );
    for (final cols in [null, data.header]) {
      expect(
        checkModel(doc, columns: cols).map((i) => i.message),
        contains(contains('the deep networks run after the transforms')),
      );
    }
  });

  test('checks: an input reading a deep column needs no data column', () {
    final doc = wired();
    expect(
      checkModel(doc, columns: data.header).map((i) => i.message),
      isNot(contains(contains('No column "clf_prediction"'))),
    );
    final clash = CsvData(
      [...data.header, 'clf_prediction'],
      [
        for (final r in data.rows) [...r, '0'],
      ],
    );
    expect(
      checkModel(doc, columns: clash.header).map((i) => i.message),
      contains(contains('writes "clf_prediction", which is already a column')),
    );
  });
}
