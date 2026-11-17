/// The deep-network exports: each Dart script analyses clean and runs,
/// and (with `tool/parity/.venv`, `melos run test:export-python`) the
/// Python script, run by pyhgf, writes the same predictions to 1e-9.
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:hgf/hgf.dart' show CsvData;
import 'package:hgf_deep/hgf_deep.dart';
import 'package:hgf_fit/codegen.dart';
import 'package:hgf_fit/document.dart';
import 'package:test/test.dart';

final Directory _dir = Directory('.dart_tool/deep_export_test');
final String _python = File('../../tool/parity/.venv/bin/python').absolute.path;

/// Same-order training (no shuffle), weights as built (no seeded draw),
/// pyhgf's arithmetic (no fixes): the two scripts must agree.
const _comparable = DocDeepTraining(
  init: DeepInit.none,
  shuffle: false,
  optimiser: 'sgd',
  learningRate: 0.05,
  epochs: 2,
);
const _pyhgf = DocDeepSettings(fixes: DeepFixes.none);

String _csv(
  int n,
  int seed,
  String Function(math.Random r, int i) row,
  String header,
) {
  final r = math.Random(seed);
  return [header, for (var i = 0; i < n; i++) row(r, i)].join('\n');
}

String _f(double v) => v.toStringAsFixed(6);

final cases = <String, (DocDeep, String)>{
  'binary classifier': (
    DocDeep(
      id: 'clf',
      settings: _pyhgf.copyWith(tonicVolatility: true),
      layers: const [
        DocLayer(size: 1, kind: LayerKind.binary),
        DocLayer(size: 4, coupling: 'tanh', tonicVolatility: -3),
        DocLayer(size: 2),
      ],
      features: ['a', 'b'],
      targets: ['label'],
      training: _comparable,
    ),
    _csv(40, 1, (r, i) {
      final a = r.nextDouble() * 2 - 1, b = r.nextDouble() * 2 - 1;
      // A gap the scripts must skip.
      return i == 7
          ? '${_f(a)},,1'
          : '${_f(a * 3 + 5)},${_f(b)},${a * b > 0 ? 1 : 0}';
    }, 'a,b,label'),
  ),
  'three labels, "none" among them': (
    DocDeep(
      id: 'press',
      settings: _pyhgf.copyWith(volatilityUpdates: DeepVolatilityUpdate.ehgf),
      layers: const [
        DocLayer(size: 3, kind: LayerKind.categorical),
        DocLayer(size: 5, coupling: 'sigmoid', volatilityParent: false),
        DocLayer(size: 2),
      ],
      features: ['x', 'y'],
      targets: ['press'],
      categories: ['left', 'right', 'none'],
      training: _comparable.copyWith(optimiser: 'adam', learningRate: 0.01),
    ),
    _csv(45, 2, (r, i) {
      final x = r.nextDouble() * 2 - 1, y = r.nextDouble() * 2 - 1;
      final press = i == 11
          ? 'jump' // not a category: skipped
          : x < -0.3
          ? 'left'
          : x > 0.3
          ? 'right'
          : 'none';
      return '${_f(x)},${_f(y)},$press';
    }, 'x,y,press'),
  ),
  'regression, expression coupling, minibatches': (
    DocDeep(
      id: 'reg',
      // Tonic volatility is tested on the classifier: with it, this
      // regression diverges (SGD on weights of 1), and over several passes
      // the two engines' last-digit differences grow without bound.
      settings: _pyhgf,
      layers: const [
        DocLayer(size: 1),
        DocLayer(size: 6, coupling: '2 * tanh(x / 2)'),
        DocLayer(size: 1),
      ],
      features: ['t'],
      targets: ['v'],
      training: _comparable.copyWith(batchSize: () => 8, epochs: 3),
    ),
    _csv(30, 3, (r, i) {
      final t = i / 30 * 6 - 3;
      return '${_f(t)},${_f(math.sin(t) + 0.1 * (r.nextDouble() - 0.5))}';
    }, 't,v'),
  ),
  'continuous filter': (
    DocDeep(
      id: 'filt',
      settings: _pyhgf,
      layers: const [
        DocLayer(size: 2, kind: LayerKind.continuous),
        DocLayer(size: 1, kind: LayerKind.continuous, tonicVolatility: -2),
      ],
      targets: ['o1', 'o2'],
    ),
    _csv(25, 4, (r, i) {
      final s = math.sin(i / 4);
      return '${_f(s + 0.2 * r.nextDouble())},${_f(-s + 0.2 * r.nextDouble())}';
    }, 'o1,o2'),
  ),
};

List<List<String>> _read(String path) =>
    CsvData.parse(File(path).readAsStringSync()).rows;

void main() {
  setUpAll(() {
    if (_dir.existsSync()) _dir.deleteSync(recursive: true);
    _dir.createSync(recursive: true);
    for (final MapEntry(key: name, value: (d, csv)) in cases.entries) {
      expect(
        [
          for (final i in checkModel(
            ModelDocument(
              network: NetworkDocument(nodes: const []),
              deep: [d],
            ),
            columns: CsvData.parse(csv).header,
          ))
            if (i.isError) i.message,
        ],
        isEmpty,
        reason: name,
      );
      File('${_dir.path}/${d.id}.csv').writeAsStringSync(csv);
      File(
        '${_dir.path}/${d.id}.dart',
      ).writeAsStringSync(DeepDartExport(d).script());
      File(
        '${_dir.path}/${d.id}.py',
      ).writeAsStringSync(DeepPythonExport(d).script());
    }
  });

  test('the Dart scripts analyse clean', () async {
    final r = await Process.run(Platform.resolvedExecutable, [
      'analyze',
      '--fatal-infos',
      _dir.path,
    ]);
    expect(r.exitCode, 0, reason: '${r.stdout}\n${r.stderr}');
  });

  final venv = File(_python).existsSync();
  for (final MapEntry(key: name, value: (d, _)) in cases.entries) {
    group(name, () {
      late List<List<String>> dart;
      test('the Dart script runs', () async {
        final r = await Process.run(Platform.resolvedExecutable, [
          'run',
          '${_dir.path}/${d.id}.dart',
          '${_dir.path}/${d.id}.csv',
        ]);
        expect(r.exitCode, 0, reason: '${r.stdout}\n${r.stderr}');
        dart = _read('${_dir.path}/${d.id}_${d.id}.csv');
        final width = d.targetLayer!.size;
        final filled = dart.where((row) => row.last.isNotEmpty).length;
        expect(filled, greaterThan(15));
        expect(
          dart.first.length,
          CsvData.parse(cases[name]!.$2).header.length + width,
        );
      });

      test(
        'pyhgf writes the same predictions',
        () async {
          final out = '${_dir.path}/${d.id}_${d.id}.csv';
          final dartCopy = '${_dir.path}/${d.id}_dart.csv';
          File(out).copySync(dartCopy);
          final r = await Process.run(_python, [
            '${_dir.path}/${d.id}.py',
            '${_dir.path}/${d.id}.csv',
          ]);
          expect(r.exitCode, 0, reason: '${r.stdout}\n${r.stderr}');
          final py = _read(out), ours = _read(dartCopy);
          expect(py.length, ours.length);
          var compared = 0;
          for (var i = 0; i < py.length; i++) {
            expect(py[i].length, ours[i].length);
            for (var j = 0; j < py[i].length; j++) {
              final a = double.tryParse(ours[i][j]),
                  b = double.tryParse(py[i][j]);
              if (a == null || b == null) {
                expect(py[i][j].trim(), ours[i][j].trim(), reason: 'row $i');
                continue;
              }
              expect(
                (a - b).abs(),
                lessThanOrEqualTo(1e-9 * math.max(1, a.abs())),
                reason: 'row $i column $j: Dart $a, pyhgf $b',
              );
              compared++;
            }
          }
          expect(compared, greaterThan(0));
        },
        tags: ['pyhgf'],
        skip: venv ? false : 'needs tool/parity/.venv (uv sync)',
      );
    });
  }
}
