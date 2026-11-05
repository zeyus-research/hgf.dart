import 'dart:convert';
import 'dart:typed_data';

import 'math.dart';
import 'types.dart';

/// Recorded node trajectories, stored column-wise: one contiguous
/// [Float64List] per (field, node), `nTime` long.
///
/// This is the hgf.dart answer to pyhgf's `node_trajectories` / `to_pandas()`:
/// no dataframe library, just typed columns that export to CSV or JSON for
/// analysis elsewhere (pandas, R, ArviZ) or plot directly.
///
/// Vector-valued fields ([VectorField], e.g. an ef-state node's `xis`) are
/// recorded for the nodes that have them: see [vectorSeries].
final class Trajectories {
  /// [vectorStarts] gives, per recorded [VectorField], the offsets of each
  /// node's vector (`nNodes + 1` entries, as `Network` stores them).
  Trajectories(
    this.nTime,
    this.nNodes,
    Iterable<NodeField> fields, {
    List<NodeKind>? kinds,
    this.responseSurprise,
    Map<VectorField, Int32List> vectorStarts = const {},
  }) : fields = List.unmodifiable(fields),
       kinds = kinds == null ? null : List.unmodifiable(kinds),
       vectorFields = List.unmodifiable(vectorStarts.keys),
       _data = {for (final f in fields) f: Float64List(nTime * nNodes)},
       _vStart = {
         for (final e in vectorStarts.entries)
           e.key: Int32List.fromList(e.value),
       },
       _vData = {
         for (final e in vectorStarts.entries)
           e.key: Float64List(nTime * e.value.last),
       } {
    if (kinds != null && kinds.length != nNodes) {
      throw ArgumentError('kinds must have one entry per node');
    }
    for (final start in _vStart.values) {
      if (start.length != nNodes + 1 || start.first != 0) {
        throw ArgumentError('vectorStarts must have nNodes + 1 offsets');
      }
      for (var i = 0; i < nNodes; i++) {
        if (start[i + 1] < start[i]) {
          throw ArgumentError('vectorStarts must be non-decreasing');
        }
      }
    }
    if (responseSurprise != null && responseSurprise!.length != nTime) {
      throw ArgumentError('responseSurprise must have one entry per trial');
    }
  }

  /// Number of recorded time steps.
  final int nTime;

  /// Number of nodes in the network.
  final int nNodes;

  /// The recorded fields.
  final List<NodeField> fields;

  /// The kind of each node, when known (needed by [nodeSurprise]).
  final List<NodeKind>? kinds;

  /// Per-trial surprise of the response model passed to
  /// `Network.inputData`, or `null` if there was none.
  final Float64List? responseSurprise;

  /// The recorded vector fields.
  final List<VectorField> vectorFields;

  final Map<NodeField, Float64List> _data;
  final Map<VectorField, Int32List> _vStart;
  final Map<VectorField, Float64List> _vData;

  /// Offsets of each node's [field] vector (`nNodes + 1` entries).
  Int32List vectorStart(VectorField field) => _vectorParts(field).$1;

  /// Raw storage of [field]: node `i`'s block starts at
  /// `vectorStart(field)[i] * nTime` and holds [nTime] rows of
  /// [vectorLength] values.
  Float64List rawVector(VectorField field) => _vectorParts(field).$2;

  /// Length of [node]'s [field] vector (0 if it has none).
  int vectorLength(int node, VectorField field) {
    RangeError.checkValidIndex(node, null, 'node', nNodes);
    final start = vectorStart(field);
    return start[node + 1] - start[node];
  }

  /// [node]'s [field] over time, row-major `nTime x vectorLength` (a view).
  Float64List vectorSeries(int node, VectorField field) {
    final len = vectorLength(node, field);
    final s = vectorStart(field)[node] * nTime;
    return Float64List.sublistView(rawVector(field), s, s + nTime * len);
  }

  /// [node]'s [field] vector at time step [t] (a view).
  Float64List vectorAt(int node, VectorField field, int t) {
    final len = vectorLength(node, field);
    RangeError.checkValidIndex(t, null, 't', nTime);
    final s = vectorStart(field)[node] * nTime + t * len;
    return Float64List.sublistView(rawVector(field), s, s + len);
  }

  (Int32List, Float64List) _vectorParts(VectorField field) {
    final start = _vStart[field];
    if (start == null) {
      throw ArgumentError.value(field, 'field', 'Not recorded');
    }
    return (start, _vData[field]!);
  }

  /// Raw node-major storage of [field]: node `i`'s series is
  /// `[i * nTime, (i + 1) * nTime)`.
  Float64List raw(NodeField field) => _column(field);

  /// The time series of [field] for [node] (a view, not a copy).
  Float64List series(int node, NodeField field) {
    RangeError.checkValidIndex(node, null, 'node', nNodes);
    return Float64List.sublistView(
      _column(field),
      node * nTime,
      (node + 1) * nTime,
    );
  }

  /// Value of [field] for [node] at time step [t].
  double at(int node, NodeField field, int t) =>
      _column(field)[node * nTime + t];

  Float64List _column(NodeField field) {
    final c = _data[field];
    if (c == null) {
      throw ArgumentError.value(field, 'field', 'Not recorded');
    }
    return c;
  }

  /// Surprise of [node]'s value at each trial under its own prediction, as
  /// pyhgf's `to_pandas()` reports it (`x_<node>_surprise`):
  /// [gaussianSurprise] of `mean` under `expectedMean`/`expectedPrecision`
  /// for a continuous node, [binarySurprise] of `mean` under `expectedMean`
  /// for a binary one.
  ///
  /// Needs `mean`, `expectedMean` (and for continuous nodes
  /// `expectedPrecision`) to have been recorded. Not defined (all NaN) for
  /// constant and ef-state nodes, which pyhgf leaves out.
  Float64List nodeSurprise(int node) {
    RangeError.checkValidIndex(node, null, 'node', nNodes);
    final ks = kinds;
    if (ks == null) throw StateError('Node kinds are unknown');
    final out = Float64List(nTime);
    if (ks[node] == NodeKind.constantState || ks[node] == NodeKind.efState) {
      return out..fillRange(0, nTime, double.nan);
    }
    final x = series(node, NodeField.mean);
    final m = series(node, NodeField.expectedMean);
    switch (ks[node]) {
      case NodeKind.continuousState:
        final p = series(node, NodeField.expectedPrecision);
        for (var t = 0; t < nTime; t++) {
          out[t] = gaussianSurprise(x[t], m[t], p[t]);
        }
      case NodeKind.binaryState:
        for (var t = 0; t < nTime; t++) {
          out[t] = binarySurprise(x[t], m[t]);
        }
      case NodeKind.constantState:
      case NodeKind.efState:
        break; // handled above
    }
    return out;
  }

  /// Sum of [nodeSurprise] over all nodes at each trial (pyhgf's
  /// `total_surprise` column). As in pandas, NaN terms are skipped, and the
  /// sum is NaN only when every term is.
  Float64List totalSurprise() {
    final out = Float64List(nTime)..fillRange(0, nTime, double.nan);
    for (var node = 0; node < nNodes; node++) {
      final s = nodeSurprise(node);
      for (var t = 0; t < nTime; t++) {
        final v = s[t];
        if (v.isNaN) continue;
        out[t] = out[t].isNaN ? v : out[t] + v;
      }
    }
    return out;
  }

  /// Long-format CSV: `time,node,<field>...` (scalar fields only).
  String toCsv() {
    final b = StringBuffer()
      ..write('time,node')
      ..writeAll(fields.map((f) => ',${f.pyhgfName}'))
      ..writeln();
    for (var node = 0; node < nNodes; node++) {
      for (var t = 0; t < nTime; t++) {
        b
          ..write(t)
          ..write(',')
          ..write(node);
        for (final f in fields) {
          b
            ..write(',')
            ..write(_data[f]![node * nTime + t]);
        }
        b.writeln();
      }
    }
    return b.toString();
  }

  /// `{"<node>": {"<field>": [...]}}`, the shape of pyhgf's
  /// `node_trajectories`; a vector field is a list of per-trial lists.
  /// Non-finite values are encoded as strings. For a multivariate ef-state
  /// node, `mean` is the vector one.
  Map<String, Map<String, List<Object>>> toJson() {
    Object enc(double v) => v.isFinite ? v : v.toString();
    return {
      for (var node = 0; node < nNodes; node++)
        '$node': {
          for (final f in fields)
            f.pyhgfName: [for (final v in series(node, f)) enc(v)],
          for (final f in vectorFields)
            if (vectorLength(node, f) > (f == VectorField.mean ? 1 : 0))
              f.pyhgfName: [
                for (var t = 0; t < nTime; t++)
                  [for (final v in vectorAt(node, f, t)) enc(v)],
              ],
        },
    };
  }

  String toJsonString() => jsonEncode(toJson());
}
