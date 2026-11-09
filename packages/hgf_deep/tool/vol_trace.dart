import 'dart:convert';
import 'dart:io';

import 'package:hgf_deep/hgf_deep.dart';

import '../test/src/deep_check.dart';

/// Print layer [layer]'s mean_vol over a fixture's first fit, every 3 steps.
void main(List<String> args) {
  final fixture =
      jsonDecode(File('test/fixtures/${args[0]}.json').readAsStringSync())
          as Map<String, Object?>;
  final layer = int.parse(args[1]);
  final net = buildNetwork(fixture);
  final op = (fixture['ops'] as List).first as Map<String, Object?>;
  final opt = Adam(((op['optimiser'] as Map)['lr'] as num).toDouble());
  net.fit(
    matrix(op['x']),
    matrix(op['y']),
    optimiser: opt,
    weightUpdate: args.length < 3 || args[2] != 'frozen',
    record: {LayerField.meanVol},
  );
  final mv = net.trajectories!.of(LayerField.meanVol, layer)!;
  final n = net.layerSizes[layer];
  for (var t = 0; t < net.trajectories!.steps; t += 3) {
    print(
      't=$t ${[for (var i = 0; i < n; i++) mv[t * n + i].toStringAsFixed(2)]}',
    );
  }
}
