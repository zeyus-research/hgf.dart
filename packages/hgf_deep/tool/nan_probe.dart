import 'dart:convert';
import 'dart:io';

import 'package:hgf_deep/hgf_deep.dart';

import '../test/src/deep_check.dart';

/// Step a fixture's first fit one sample at a time and report the first
/// non-finite belief, with the layer's values just before.
void main(List<String> args) {
  final fixture =
      jsonDecode(File('test/fixtures/${args[0]}.json').readAsStringSync())
          as Map<String, Object?>;
  final net = buildNetwork(fixture);
  final opIndex = args.length > 1 ? int.parse(args[1]) : 0;
  final ops = fixture['ops'] as List;
  final first = ops.first as Map<String, Object?>;
  final opt = Adam(((first['optimiser'] as Map)['lr'] as num).toDouble());
  for (var k = 0; k < opIndex; k++) {
    final op = ops[k] as Map<String, Object?>;
    net.fit(matrix(op['x']), matrix(op['y']), optimiser: opt);
  }
  final op = ops[opIndex] as Map<String, Object?>;
  final x = matrix(op['x']);
  final y = matrix(op['y']);
  String dump(int l) => [
    for (final f in LayerField.values)
      if (net.state(l).field(f) != null)
        '    ${f.pyhgfName}: ${net.state(l).field(f)!.map((v) => v.toStringAsPrecision(5)).toList()}',
  ].join('\n');
  for (var t = 0; t < x.rows; t++) {
    final before = [for (var l = 0; l < net.nLayers; l++) dump(l)];
    net.fit(
      Matrix(1, x.cols, x.row(t)),
      Matrix(1, y.cols, y.row(t)),
      optimiser: opt,
    );
    for (var l = 0; l < net.nLayers; l++) {
      for (final f in LayerField.values) {
        final v = net.state(l).field(f);
        if (v == null) continue;
        final i = v.indexWhere((e) => !e.isFinite);
        if (i >= 0) {
          print('sample $t: layer $l ${f.pyhgfName}[$i] = ${v[i]}');
          print('  layer $l before:\n${before[l]}');
          print('  layer $l after:\n${dump(l)}');
          return;
        }
      }
    }
  }
  print('no non-finite values');
}
