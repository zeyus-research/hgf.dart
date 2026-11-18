import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_gpu/hgf_gpu.dart';

double _decode(Object? v) => switch (v) {
  num() => v.toDouble(),
  'NaN' => double.nan,
  'Infinity' => double.infinity,
  '-Infinity' => double.negativeInfinity,
  _ => throw FormatException('Bad fixture number: $v'),
};

Float64List _flatten(Object? rows) => Float64List.fromList([
  for (final row in rows! as List<Object?>)
    for (final x in row! as List<Object?>) _decode(x),
]);

const _elements = 5;

/// f32 tolerance (relative) on the total surprise. The largest error seen on
/// the well-conditioned fixtures is 1.0e-6 (RX 7600 XT and SwiftShader), so
/// 1e-6 fails intermittently. WGSL leaves exp/log accuracy to the
/// implementation (exp: 3 + 2|x| ulp), so other GPUs may need more.
const tolerance = 1e-5;

/// Fixtures where f32 can't track f64, in pyhgf's own float32 mode (JAX)
/// either (`tool/parity/gpu_f32_reference.py --fixtures ...`):
/// * out of f32's range: f64 reaches variances of 1e250 / 1e-307; JAX f32
///   returns an infinite total, the GPU infinite or far off;
/// * sensitive: chaotic (`continuous_2level_eHGF` after ~150 trials); a
///   total that cancels to ~−1 (`noise_parent_time_steps`, element 1:
///   pyhgf f32 is 3.6e-5 off, the GPU 1.1e-5).
///
/// The binary fixtures near a runaway, where JAX f32 is 5e-4 to 9e-3 off,
/// match to ~3e-6: the kernel carries `1 − μ̂` of binary nodes as `π̂ / μ̂`
/// instead of subtracting, which cancels catastrophically near 1.
const outOfRange = {
  'constant_bias_eHGF',
  'continuous_2level_unbounded_overflow',
};
const sensitive = {
  'continuous_2level_eHGF': 0.1,
  'noise_parent_time_steps': 5e-5,
};

/// Element [e]'s value of [field] of continuous node [i] (as the engine's
/// batch check perturbs them).
double _perturbed(Network net, int i, NodeField field, int e) =>
    net.getInitial(i, field) +
    switch (field) {
      NodeField.tonicVolatility => -0.25 * e,
      NodeField.mean => 0.1 * e,
      _ => 0.0,
    };

/// One fixture's result: null if it has no GPU kernel (the reason in
/// [skipped]), else the max relative error (of the totals and of every
/// ef-state node's final ξ), the non-finite mismatches, and the max relative
/// error of the ef-state nodes' final ν (0 without any).
typedef FixtureResult = ({
  double maxRel,
  int nonFiniteMismatch,
  int finite,
  double nusRel,
});

Future<FixtureResult?> checkFixture(
  GpuBackend backend,
  Map<String, Object?> fixture,
  Map<String, String> skipped,
) async {
  final name = fixture['name'] as String;
  final spec = NetworkSpec.fromJson(fixture['spec'] as Map<String, Object?>);
  final Network net;
  final HgfKernel kernel;
  final observed = fixture['observed'] == null
      ? null
      : _flatten(fixture['observed']);
  final steps = fixture['time_steps'] == null
      ? null
      : Float64List.fromList([
          for (final x in fixture['time_steps']! as List<Object?>) _decode(x),
        ]);
  final bool binary;
  try {
    net = spec.build()..build();
    binary = net.inputIdxs.every((i) => net.kindOf(i) == NodeKind.binaryState);
    kernel = HgfKernel(
      net,
      response: binary
          ? GpuResponse.firstLevelBinarySurprise
          : GpuResponse.totalGaussianSurprise,
      observedMask: observed != null,
      timeSteps: steps != null,
    );
  } on UnsupportedError catch (e) {
    skipped[name] = '${e.message}';
    return null;
  } on ArgumentError catch (e) {
    skipped[name] = '${e.message}'; // custom coupling names
    return null;
  }
  final data = _flatten(fixture['inputs']);
  final cpu = BatchNetwork(net, _elements);
  final gpu = await GpuBatch.create(backend, kernel, _elements);
  for (var i = 0; i < net.nNodes; i++) {
    if (net.kindOf(i) == NodeKind.efState) {
      // A learning rate per element (with HGF learning, the first trial's).
      final values = Float64List.fromList([
        for (var e = 0; e < _elements; e++)
          net.getInitial(i, NodeField.nus) + 0.5 * e,
      ]);
      cpu.setInitial(i, NodeField.nus, values);
      gpu.setInitial(i, NodeField.nus, values);
    }
    if (net.kindOf(i) != NodeKind.continuousState) continue;
    for (final f in [NodeField.tonicVolatility, NodeField.mean]) {
      final values = Float64List.fromList([
        for (var e = 0; e < _elements; e++) _perturbed(net, i, f, e),
      ]);
      cpu.setInitial(i, f, values);
      gpu.setInitial(i, f, values);
    }
  }
  final ResponseFunction response = binary
      ? const FirstLevelBinarySurprise()
      : const TotalGaussianSurprise();
  // A lone ef-state node has no continuous node to score (the CPU's
  // TotalGaussianSurprise refuses it; the kernel's sums nothing): its ξ are
  // compared below.
  final Float64List ref;
  if (net.nNodes < 2) {
    cpu.inputData(data, observed: observed, timeSteps: steps);
    ref = Float64List(_elements);
  } else {
    ref = cpu.surprise(data, response, observed: observed, timeSteps: steps);
  }
  final got = await gpu.surprise(data, observed: observed, timeSteps: steps);
  double rel(double a, double b) => (a - b).abs() / math.max(a.abs(), 1.0);
  var maxRel = 0.0, mismatch = 0, finite = 0, nusRel = 0.0;
  for (var e = 0; e < _elements; e++) {
    if (ref[e].isFinite != got[e].isFinite) {
      mismatch++;
    } else if (ref[e].isFinite) {
      finite++;
      maxRel = math.max(maxRel, rel(ref[e], got[e]));
    }
  }
  for (var i = 0; i < net.nNodes; i++) {
    if (net.kindOf(i) != NodeKind.efState) continue;
    for (var k = 0; k < kernel.xisLength(i); k++) {
      final xi = await gpu.readXis(i, k);
      for (var e = 0; e < _elements; e++) {
        maxRel = math.max(
          maxRel,
          rel(cpu.getVector(e, i, VectorField.xis)[k], xi[e]),
        );
      }
    }
    // The CPU leaves ν NaN after an unobserved trial (with HGF learning),
    // the kernel the last one: compared where the CPU's is finite.
    final nus = await gpu.read(i, NodeField.nus);
    for (var e = 0; e < _elements; e++) {
      final want = cpu.get(e, i, NodeField.nus);
      if (want.isFinite) nusRel = math.max(nusRel, rel(want, nus[e]));
    }
  }
  gpu.dispose();
  return (
    maxRel: maxRel,
    nonFiniteMismatch: mismatch,
    finite: finite,
    nusRel: nusRel,
  );
}

/// Checks every fixture on [backend], printing one line each; returns the
/// failures (fixtures with a kernel that miss their tolerance).
Future<List<String>> checkAll(
  GpuBackend backend,
  List<Map<String, Object?>> fixtures,
) async {
  print(backend.description);
  final skipped = <String, String>{};
  final failures = <String>[];
  var checked = 0;
  for (final f in fixtures) {
    final r = await checkFixture(backend, f, skipped);
    if (r == null) continue;
    checked++;
    final name = f['name'];
    print(
      '$name: max rel ${r.maxRel.toStringAsExponential(2)}, '
      '${r.finite} finite, ${r.nonFiniteMismatch} non-finite mismatches'
      '${r.nusRel > 0 ? ', ν ${r.nusRel.toStringAsExponential(2)}' : ''}',
    );
    if (outOfRange.contains(name)) continue;
    final tol = sensitive[name] ?? tolerance;
    if (r.nonFiniteMismatch != 0 || r.maxRel > tol || r.nusRel > tolerance) {
      failures.add('$name: $r');
    }
  }
  print('skipped (no kernel): ${skipped.keys.join(', ')}');
  if (checked < 30) failures.add('only $checked fixtures checked');
  return failures;
}
