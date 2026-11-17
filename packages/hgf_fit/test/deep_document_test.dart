@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';

import 'package:hgf/hgf.dart';
import 'package:hgf_deep/hgf_deep.dart';
import 'package:hgf_fit/document.dart';
import 'package:test/test.dart';

import '../../hgf_deep/test/src/deep_check.dart' show buildNetwork;

/// A network's JSON without what [DocDeep] doesn't store (the weights).
Map<String, Object?> structureOf(DeepNetwork net) {
  final j = net.toJson();
  for (final l in j['layers'] as List<Object?>) {
    (l as Map<String, Object?>)
      ..remove('weights')
      ..remove('volatility_weights')
      ..remove('weights_precision_delta');
  }
  return j;
}

void main() {
  final fixtures = [
    for (final f in Directory('../hgf_deep/test/fixtures').listSync())
      if (f is File && f.path.endsWith('.json')) f,
  ]..sort((a, b) => a.path.compareTo(b.path));

  test('fixtures exist', () => expect(fixtures, isNotEmpty));

  group('every hgf_deep fixture round-trips through DocDeep', () {
    for (final f in fixtures) {
      final name = f.uri.pathSegments.last;
      test(name, () {
        final fx = jsonDecode(f.readAsStringSync()) as Map<String, Object?>;
        final net = buildNetwork(fx);
        final doc = DocDeep.fromNetwork(net, id: 'net');
        final back = DocDeep.fromJson(
          jsonDecode(jsonEncode(doc.toJson())) as Map<String, Object?>,
        );
        expect(back.toJson(), doc.toJson());
        expect(
          jsonEncode(structureOf(back.build())),
          jsonEncode(structureOf(net)),
        );
      });
    }
  });

  group('documents with deep networks', () {
    final deep = DocDeep.starter(
      'clf',
    ).copyWith(features: ['x1', 'x2'], targets: ['y']);

    test('a deep-only document round-trips and has no HGF network', () {
      final doc = ModelDocument(
        network: NetworkDocument(nodes: const []),
        deep: [deep],
      );
      final json = jsonDecode(jsonEncode(doc.toJson())) as Map<String, Object?>;
      expect(json.containsKey('network'), isFalse);
      final back = ModelDocument.fromJson(json);
      expect(back.hasNetwork, isFalse);
      expect(back.deep.single.toJson(), deep.toJson());
      expect(() => back.compile(), throwsStateError);
      expect(
        checkModel(back, columns: ['x1', 'x2', 'y']).where((i) => i.isError),
        isEmpty,
      );
    });

    test('documents without deep networks write the same JSON', () {
      final doc = ModelDocument.fromNetwork(
        Network()
          ..addNodes()
          ..addNodes(valueChildren: [0]),
      );
      expect(doc.toJson().containsKey('deep'), isFalse);
      expect(ModelDocument.fromJson(doc.toJson()).toJson(), doc.toJson());
    });

    test('checks name the deep network', () {
      List<DocIssue> errors(DocDeep d, [List<String>? columns]) => [
        for (final i in checkModel(
          ModelDocument(
            network: NetworkDocument(nodes: const []),
            deep: [d],
          ),
          columns: columns,
        ))
          if (i.isError) i,
      ];
      expect(errors(deep), isEmpty);
      final wrongWidth = errors(deep.copyWith(features: ['x1']));
      expect(wrongWidth.single.message, contains('1 feature columns'));
      expect(wrongWidth.single.parts, [DocPart.deep('clf')]);
      expect(
        errors(deep, ['x1', 'y']).single.message,
        contains('"x2", which is not a column'),
      );
      // hgf_deep's own checks: a categorical layer must be the target.
      final bad = deep.copyWith(
        layers: [
          deep.layers[0],
          const DocLayer(size: 3, kind: LayerKind.categorical),
          deep.layers[2],
        ],
      );
      expect(errors(bad).single.message, contains('Layer 1 (hidden 1)'));
      // A categorical target read from one label column.
      final cat = deep.copyWith(
        layers: [
          const DocLayer(size: 3, kind: LayerKind.categorical),
          ...deep.layers.skip(1),
        ],
        targets: ['label'],
        categories: () => ['a', 'b', 'c'],
      );
      expect(errors(cat), isEmpty);
      expect(
        errors(cat.copyWith(categories: () => ['a', 'b'])).single.message,
        contains('2 categories'),
      );
    });

    test('an HGF network and a deep network side by side', () {
      final doc = ModelDocument.fromNetwork(
        Network()
          ..addNodes()
          ..addNodes(valueChildren: [0]),
      ).copyWith(deep: [deep]);
      final back = ModelDocument.fromJson(
        jsonDecode(jsonEncode(doc.toJson())) as Map<String, Object?>,
      );
      expect(back.hasNetwork, isTrue);
      expect(back.deepById('clf'), isNotNull);
      expect(back.compile().network.nNodes, 2);
    });

    test('expression couplings build', () {
      final d = deep.copyWith(
        layers: [
          deep.layers[0],
          deep.layers[1].copyWith(coupling: () => '2 * tanh(x / 2)'),
          deep.layers[2],
        ],
      );
      final net = d.build();
      expect(net.configs[1].coupling, isA<ExprCoupling>());
      expect(DocDeep.weightCount(net), (2 + 1) * 8 + (8 + 1) * 1);
    });
  });
}
