import 'dart:math' as math;
import 'dart:ui' show Offset, Rect;

import 'package:hgf_fit/document.dart';

/// Horizontal and vertical spacing of [layeredLayout], in canvas units.
const double layoutDx = 130, layoutDy = 120;

/// The id of the document node an address (`id` or `id.k`) belongs to.
String ownerIdOf(String address) {
  final dot = address.indexOf('.');
  return dot < 0 ? address : address.substring(0, dot);
}

/// A layered drawing of [doc], one position per document node (a
/// categorical or ef-state block is one node): inputs on the bottom row
/// (y = 0), each other node one row above its highest child, so parents
/// sit above their children as in the HGF literature. Within rows, nodes
/// are ordered to keep edges short (barycentre sweeps) and centred on
/// x = 0. Cycles are tolerated (the layout just stops climbing).
Map<String, Offset> layeredLayout(NetworkDocument doc) {
  final ids = [for (final n in doc.nodes) n.id];
  final index = {for (final (i, id) in ids.indexed) id: i};
  final n = ids.length;
  final children = List.generate(n, (_) => <int>{});
  final parents = List.generate(n, (_) => <int>{});
  for (final e in doc.edges) {
    final p = index[ownerIdOf(e.parent)], c = index[ownerIdOf(e.child)];
    if (p == null || c == null || p == c) continue;
    children[p].add(c);
    parents[c].add(p);
  }
  final level = List<int>.filled(n, -1);
  final visiting = List<bool>.filled(n, false);
  int levelOf(int i) {
    if (level[i] >= 0) return level[i];
    if (visiting[i]) return 0; // a cycle: stop here
    visiting[i] = true;
    var l = 0;
    for (final c in children[i]) {
      l = math.max(l, levelOf(c) + 1);
    }
    visiting[i] = false;
    return level[i] = l;
  }

  for (var i = 0; i < n; i++) {
    levelOf(i);
  }
  final levels = n == 0 ? 0 : level.reduce(math.max) + 1;
  final rows = [
    for (var l = 0; l < levels; l++)
      [
        for (var i = 0; i < n; i++)
          if (level[i] == l) i,
      ],
  ];
  final slot = List<double>.filled(n, 0);
  void number() {
    for (final row in rows) {
      for (var k = 0; k < row.length; k++) {
        slot[row[k]] = k - (row.length - 1) / 2;
      }
    }
  }

  double bary(int i, Set<int> nb) {
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
      _stableSortBy(rows[l], (i) => keys[i]!);
      number();
    }
    for (var l = levels - 2; l >= 0; l--) {
      final keys = {for (final i in rows[l]) i: bary(i, parents[i])};
      _stableSortBy(rows[l], (i) => keys[i]!);
      number();
    }
  }
  return {
    for (var i = 0; i < n; i++)
      ids[i]: Offset(slot[i] * layoutDx, -level[i] * layoutDy),
  };
}

void _stableSortBy(List<int> list, double Function(int) key) {
  final order = {for (final (k, i) in list.indexed) i: k};
  list.sort((a, b) {
    final c = key(a).compareTo(key(b));
    return c != 0 ? c : order[a]! - order[b]!;
  });
}

/// A free spot near [at] (no node within [layoutDx] / 2).
Offset freeSpot(Iterable<Offset> taken, Offset at) {
  bool free(Offset p) => taken.every((q) => (q - p).distance >= layoutDx * 0.6);
  if (free(at)) return at;
  for (var r = 1; r < 20; r++) {
    for (var k = 0; k < 8; k++) {
      final a = k * math.pi / 4;
      final p = at + Offset(math.cos(a), math.sin(a)) * (r * layoutDx * 0.5);
      if (free(p)) return p;
    }
  }
  return at;
}

/// The bounding box of [points] (null when empty).
Rect? boundsOf(Iterable<Offset> points) {
  Rect? r;
  for (final p in points) {
    final pr = Rect.fromLTWH(p.dx, p.dy, 0, 0);
    r = r == null ? pr : r.expandToInclude(pr);
  }
  return r;
}
