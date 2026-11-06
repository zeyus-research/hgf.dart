import 'dart:math' as math;
import 'dart:ui' show Offset;

import 'package:hgf/hgf.dart';

/// Whether an edge carries value or volatility coupling.
enum EdgeKind { value, volatility }

/// A directed parent → child edge.
typedef GraphEdge = ({int parent, int child, EdgeKind kind, double strength});

/// A layered drawing of a [Network]: inputs on the bottom row, each other
/// node one row above its highest child, ordered within rows to keep edges
/// short (barycentre sweeps).
final class NetworkLayout {
  NetworkLayout._(this.level, this.position, this.edges, this.levels);

  /// Lay out [network].
  factory NetworkLayout.of(Network network) {
    final n = network.nNodes;
    final children = [
      for (var i = 0; i < n; i++)
        {
          ...network.valueChildrenOf(i),
          ...network.volatilityChildrenOf(i),
        }.toList(),
    ];
    final level = List<int>.filled(n, -1);
    int levelOf(int i, int depth) {
      if (level[i] >= 0) return level[i];
      if (depth > n) throw StateError('The network has a cycle');
      var l = 0;
      for (final c in children[i]) {
        l = math.max(l, levelOf(c, depth + 1) + 1);
      }
      return level[i] = l;
    }

    for (var i = 0; i < n; i++) {
      levelOf(i, 0);
    }
    final levels = n == 0 ? 0 : level.reduce(math.max) + 1;
    final rows = [
      for (var l = 0; l < levels; l++)
        [
          for (var i = 0; i < n; i++)
            if (level[i] == l) i,
        ],
    ];

    final parents = [
      for (var i = 0; i < n; i++)
        {
          ...network.valueParentsOf(i),
          ...network.volatilityParentsOf(i),
        }.toList(),
    ];
    // Order within each row by the mean position of the neighbours in the
    // row below, then in the row above, a few times over.
    final slot = List<double>.filled(n, 0);
    void number() {
      for (final row in rows) {
        for (var k = 0; k < row.length; k++) {
          slot[row[k]] = row.length == 1 ? 0.5 : k / (row.length - 1);
        }
      }
    }

    double bary(int i, List<int> nb) {
      if (nb.isEmpty) return slot[i];
      var s = 0.0;
      for (final j in nb) {
        s += slot[j];
      }
      return s / nb.length;
    }

    number();
    for (var sweep = 0; sweep < 4; sweep++) {
      for (var l = 1; l < levels; l++) {
        final keys = {for (final i in rows[l]) i: bary(i, children[i])};
        rows[l].sort((a, b) => keys[a]!.compareTo(keys[b]!));
        number();
      }
      for (var l = levels - 2; l >= 0; l--) {
        final keys = {for (final i in rows[l]) i: bary(i, parents[i])};
        rows[l].sort((a, b) => keys[a]!.compareTo(keys[b]!));
        number();
      }
    }

    // Unit coordinates: x spread evenly per row (centred), y = 0 at the top.
    final position = List<Offset>.filled(n, Offset.zero);
    final widest = rows.fold<int>(1, (m, r) => math.max(m, r.length));
    for (var l = 0; l < levels; l++) {
      final row = rows[l];
      final y = levels == 1 ? 0.5 : 1 - l / (levels - 1);
      for (var k = 0; k < row.length; k++) {
        final x = (k + 0.5 + (widest - row.length) / 2) / widest;
        position[row[k]] = Offset(x, y);
      }
    }

    final edges = <GraphEdge>[
      for (var c = 0; c < n; c++) ...[
        for (final (k, p) in network.valueParentsOf(c).indexed)
          (
            parent: p,
            child: c,
            kind: EdgeKind.value,
            strength: network.valueParentStrengthsOf(c)[k],
          ),
        for (final (k, p) in network.volatilityParentsOf(c).indexed)
          (
            parent: p,
            child: c,
            kind: EdgeKind.volatility,
            strength: network.volatilityParentStrengthsOf(c)[k],
          ),
      ],
    ];
    return NetworkLayout._(level, position, edges, levels);
  }

  /// Each node's row, counted from the inputs (0).
  final List<int> level;

  /// Each node's position in the unit square (y = 0 at the top).
  final List<Offset> position;

  final List<GraphEdge> edges;

  /// Number of rows.
  final int levels;

  /// The widest row's node count.
  int get width {
    final counts = List<int>.filled(levels, 0);
    for (final l in level) {
      counts[l]++;
    }
    return counts.fold(0, math.max);
  }

  /// Node indexes from the top row down, left to right within rows (the
  /// order pyhgf's `plot_trajectories` uses).
  List<int> get topDown {
    final order = [for (var i = 0; i < level.length; i++) i];
    order.sort((a, b) {
      final byLevel = level[b].compareTo(level[a]);
      return byLevel != 0 ? byLevel : position[a].dx.compareTo(position[b].dx);
    });
    return order;
  }
}
