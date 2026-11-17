@TestOn('vm')
library;

import 'dart:convert';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:test/test.dart';

import 'src/fixture_networks.dart';

NetworkDocument _hgf3() => NetworkDocument(
  nodes: [
    ScalarNode('u', kind: NodeKind.binaryState),
    ScalarNode('x1', kind: NodeKind.continuousState),
    ScalarNode('x2', kind: NodeKind.continuousState),
  ],
  edges: const [
    DocEdge(EdgeType.value, 'x1', 'u'),
    DocEdge(EdgeType.volatility, 'x2', 'x1'),
  ],
);

List<String> _errors(NetworkDocument doc) => [
  for (final i in checkNetwork(doc))
    if (i.isError) i.message,
];

void main() {
  group('checkNetwork', () {
    for (final fx in fixtureNetworks()) {
      test('no errors: ${fx.name}', () {
        final doc = NetworkDocument.fromNetwork(fx.network);
        expect(_errors(doc), isEmpty);
      });
    }

    test('a binary 3-level HGF is clean', () {
      expect(checkNetwork(_hgf3()), isEmpty);
    });

    test('a binary node without a continuous value parent: a warning', () {
      final doc = _hgf3().withoutEdge(0);
      final issues = checkNetwork(doc);
      expect(issues.single.severity, IssueSeverity.warning);
      expect(issues.single.nodes, ['u']);
    });

    test('kind rules', () {
      // A binary node's volatility edge.
      final vol = _hgf3().withEdge(
        const DocEdge(EdgeType.volatility, 'x2', 'u'),
      );
      final i = checkNetwork(vol).firstWhere((i) => i.isError);
      expect(i.message, contains('value edges only'));
      expect(i.edges, [2]);

      // A parent of a constant node; a constant node without children.
      final c = _hgf3().withNode(ScalarNode('b', kind: NodeKind.constantState));
      expect(_errors(c).single, contains('without a value child'));
      final cp = c.withEdge(const DocEdge(EdgeType.value, 'x2', 'b'));
      expect(_errors(cp), contains(contains('takes no parents')));

      // Categorical: its own edges; its implied nodes may take parents.
      final cat = _hgf3().withNode(CategoricalNode('cat', nCategories: 3));
      expect(cat.nodes.first.id, 'cat', reason: 'non-scalar kinds go first');
      expect(_errors(cat), isEmpty);
      expect(
        _errors(cat.withEdge(const DocEdge(EdgeType.value, 'x2', 'cat'))),
        [contains('its own')],
      );
    });

    test('dp-state clusters must add up', () {
      final doc = NetworkDocument(
        nodes: [DpStateNode('dp', batchSize: 4)],
      ).withNode(EfStateNode('c', nNodes: 3));
      final e = doc.withEdge(const DocEdge(EdgeType.value, 'c', 'dp'));
      expect(_errors(e), [contains('4 clusters')]);
      final ok = e.replaceNode('c', EfStateNode('c', nNodes: 4));
      expect(_errors(ok), isEmpty);
      ok.build();
    });

    test('cycles, self edges, duplicates, bad addresses', () {
      final cyc = _hgf3().withEdge(const DocEdge(EdgeType.value, 'x1', 'x2'));
      final c = checkNetwork(cyc).firstWhere((i) => i.isError);
      expect(c.message, contains('cycle'));
      expect(c.edges.toSet(), {1, 2});

      expect(
        _errors(_hgf3().withEdge(const DocEdge(EdgeType.value, 'x1', 'x1'))),
        [contains('its own parent')],
      );
      expect(
        _errors(_hgf3().withEdge(const DocEdge(EdgeType.value, 'x1', 'u'))),
        [contains('twice')],
      );
      expect(
        _errors(_hgf3().withEdge(const DocEdge(EdgeType.value, 'q', 'u'))),
        [contains('no node "q"')],
      );
    });

    test('couplings: bad expressions are errors; unused ones warnings', () {
      final bad = _hgf3().replaceNode(
        'x1',
        ScalarNode('x1', kind: NodeKind.continuousState, coupling: '2 * (x'),
      );
      expect(_errors(bad), [contains('x1')]);
      final unused = _hgf3().replaceNode(
        'x2',
        ScalarNode('x2', kind: NodeKind.continuousState, coupling: 'tanh'),
      );
      final w = checkNetwork(unused).single;
      expect(w.severity, IssueSeverity.warning);
      expect(w.message, contains('only applies to value children'));
    });

    test('an empty network', () {
      expect(_errors(NetworkDocument(nodes: const [])), [contains('no nodes')]);
    });
  });

  group('edits', () {
    test('removing a node drops its edges and overrides', () {
      final doc = _hgf3().withParam('u', NodeField.mean, 0.5).withoutNode('x1');
      expect(doc.nodes.map((n) => n.id), ['u', 'x2']);
      expect(doc.edges, isEmpty);
      expect((doc.node('u')! as ScalarNode).params, {NodeField.mean: 0.5});
      final cat = NetworkDocument(
        nodes: [CategoricalNode('c', nCategories: 2)],
      ).withParam('c.5', NodeField.tonicVolatility, -3);
      expect(cat.overrides, hasLength(1));
      expect(cat.withoutNode('c').overrides, isEmpty);
    });

    test('renaming rewrites references; fewer categories drop edges', () {
      final doc = _hgf3().replaceNode(
        'x1',
        ScalarNode('mu', kind: NodeKind.continuousState),
      );
      expect(doc.edges.first.parent, 'mu');
      expect(doc.edges.last.child, 'mu');

      final cat = NetworkDocument(nodes: [CategoricalNode('c', nCategories: 3)])
          .withNode(ScalarNode('v', kind: NodeKind.continuousState))
          .withEdge(const DocEdge(EdgeType.volatility, 'v', 'c.7'))
          .withParam('c.7', NodeField.tonicVolatility, -3);
      expect(_errors(cat), isEmpty);
      final fewer = cat.replaceNode('c', CategoricalNode('c', nCategories: 2));
      expect(fewer.edges, isEmpty, reason: 'c.7 is gone with 2 categories');
      expect(fewer.overrides, isEmpty);
    });

    test('scalar params go to the node; others to overrides', () {
      final doc = _hgf3()
          .withParam('x1', NodeField.tonicVolatility, -3)
          .withParam('x1', NodeField.tonicVolatility, -2);
      expect(
        (doc.node('x1')! as ScalarNode).params[NodeField.tonicVolatility],
        -2,
      );
      expect(doc.overrides, isEmpty);
      final net = doc.build();
      expect(net.getInitial(1, NodeField.tonicVolatility), -2);
    });

    test('a strength edit drops the override that would win', () {
      final doc = NetworkDocument(
        nodes: [
          ScalarNode('a', kind: NodeKind.continuousState),
          ScalarNode('b', kind: NodeKind.continuousState),
        ],
        edges: const [DocEdge(EdgeType.value, 'b', 'a')],
        overrides: const [CouplingOverride(EdgeType.value, 'b', 'a', 0.5)],
      );
      expect(doc.build().valueParentStrengthsOf(0), [0.5]);
      final edited = doc.replaceEdge(
        0,
        const DocEdge(EdgeType.value, 'b', 'a', strength: 2),
      );
      expect(edited.overrides, isEmpty);
      expect(edited.build().valueParentStrengthsOf(0), [2.0]);
    });

    test('ordered keeps a compiling order and repairs others', () {
      final doc = _hgf3();
      expect(identical(doc.ordered(), doc), isTrue);
      // An ef-state cluster before its dp-state node can't compile.
      final bad = NetworkDocument(
        nodes: [
          EfStateNode('c', nNodes: 2),
          ScalarNode('x', kind: NodeKind.continuousState),
          DpStateNode('dp', batchSize: 2),
        ],
        edges: const [DocEdge(EdgeType.value, 'c', 'dp')],
      );
      expect(bad.calls, throwsFormatException);
      final fixed = bad.ordered();
      expect(fixed.nodes.map((n) => n.id), ['dp', 'c', 'x']);
      fixed.build();
    });

    test('freshId and JSON after edits', () {
      final doc = _hgf3();
      expect(doc.freshId(), 'x0');
      final more = doc.withNode(
        ScalarNode(doc.freshId(), kind: NodeKind.continuousState),
      );
      expect(more.freshId(), 'x3');
      final back = NetworkDocument.fromJson(
        jsonDecode(jsonEncode(more.toJson())) as Map<String, Object?>,
      );
      expect(jsonEncode(back.toJson()), jsonEncode(more.toJson()));
    });
  });
}
