import 'dart:typed_data';

import 'package:hgf/hgf.dart';

/// Filters [data] with every field recorded and hashes the trajectories'
/// bits (FNV-1a over every scalar and vector field): equal hashes mean
/// bit-identical runs. Shared by the code generation test and the
/// harness it compiles.
String trajectoryHash(
  Network network,
  Float64List inputs, {
  Float64List? observed,
  Float64List? timeSteps,
}) {
  final traj = network.inputData(
    inputs,
    observed: observed,
    timeSteps: timeSteps,
    fields: NodeField.values,
  )!;
  var h = 0xcbf29ce484222325;
  void add(Float64List v) {
    for (final x in v.buffer.asUint64List(v.offsetInBytes, v.length)) {
      h = (h ^ x) * 0x100000001b3;
    }
  }

  for (final f in NodeField.values) {
    add(Float64List.fromList(traj.raw(f)));
  }
  for (final f in VectorField.values) {
    try {
      add(Float64List.fromList(traj.rawVector(f)));
    } on ArgumentError {
      // Not recorded: the network has no such vectors.
    }
  }
  return h.toUnsigned(64).toRadixString(16);
}
