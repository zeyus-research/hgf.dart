import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_widgets/hgf_widgets.dart';

void main() {
  test('3-level HGF: input at the bottom, volatility parent on top', () {
    final net = Network()
      ..addNodes(kind: NodeKind.binaryState)
      ..addNodes(valueChildren: [0])
      ..addNodes(volatilityChildren: [1], volatilityChildrenStrengths: [2]);
    final l = NetworkLayout.of(net);
    expect(l.level, [0, 1, 2]);
    expect(l.levels, 3);
    expect(l.topDown, [2, 1, 0]);
    expect(l.position[2].dy, 0);
    expect(l.position[0].dy, 1);
    expect(l.edges, [
      (parent: 1, child: 0, kind: EdgeKind.value, strength: 1.0),
      (parent: 2, child: 1, kind: EdgeKind.volatility, strength: 2.0),
    ]);
  });

  test('categorical network: rows ordered to follow their children', () {
    final net = Network()..addCategoricalState(nCategories: 3);
    final l = NetworkLayout.of(net);
    expect(l.levels, 4);
    expect(l.width, 3);
    // Each value parent sits above its own binary node.
    for (final e in l.edges.where((e) => l.level[e.child] == 1)) {
      expect(l.position[e.parent].dx, closeTo(l.position[e.child].dx, 1e-9));
    }
  });
}
