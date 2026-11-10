import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'fixtures.dart';

/// Number of elements and block size of the batches checked: 5 elements in
/// blocks of 2 also covers a partial last block.
const int batchElements = 5;
const int batchBlock = 2;

/// Element [e]'s parameters: every continuous node's tonic volatility moved
/// by `-0.25 e` and its prior mean by `0.1 e`, and the volatility edge
/// [kappaEdge] (if any) set to `1 + 0.2 e`.
void _perturb(
  int e,
  Network template,
  (int, int)? kappaEdge, {
  required void Function(int node, NodeField f, double v) setInitial,
  required void Function(int parent, int child, double k) setKappa,
}) {
  for (var i = 0; i < template.nNodes; i++) {
    if (template.kindOf(i) != NodeKind.continuousState) continue;
    for (final (f, d) in [
      (NodeField.tonicVolatility, -0.25 * e),
      (NodeField.mean, 0.1 * e),
    ]) {
      setInitial(i, f, template.getInitial(i, f) + d);
    }
  }
  if (kappaEdge != null) setKappa(kappaEdge.$1, kappaEdge.$2, 1.0 + 0.2 * e);
}

/// The first (parent, child) volatility edge of [probe], found by trying to
/// set it (which changes [probe]: pass a throwaway network).
(int, int)? _firstVolatilityEdge(Network probe) {
  for (var p = 0; p < probe.nNodes; p++) {
    for (var c = 0; c < probe.nNodes; c++) {
      try {
        probe.setVolatilityCoupling(p, c, 1.0);
        return (p, c);
      } on ArgumentError {
        continue;
      }
    }
  }
  return null;
}

bool _same(double a, double b) => a == b || (a.isNaN && b.isNaN);

/// Run [fixture]'s network as a [BatchNetwork] of [batchElements] perturbed
/// elements, and each element as its own [Network], and list every value
/// where they differ (they must agree bit for bit):
///
/// * trial by trial with [BatchNetwork.step], every field of every node;
/// * over the whole series with [BatchNetwork.surprise] in blocks of
///   [batchBlock], the final state and the per-trial surprise;
/// * the same with a different series and missing-observation pattern per
///   element (`perElement`).
List<String> batchMismatches(Map<String, Object?> fixture) {
  final name = fixture['name'] as String;
  final spec = NetworkSpec.fromJson(fixture['spec'] as Map<String, Object?>);
  Network build() => spec.build(couplings: testCouplings)..build();

  final data = flatten(fixture['inputs']);
  final observed = fixture['observed'] == null
      ? null
      : flatten(fixture['observed']);
  final timeSteps = fixture['time_steps'] == null
      ? null
      : Float64List.fromList(series(fixture['time_steps']));

  final template = build();
  final unbatched = {NodeKind.dpState}.intersection({
    for (var i = 0; i < template.nNodes; i++) template.kindOf(i),
  });
  if (unbatched.isNotEmpty) {
    // No batched kernels for dp-state nodes yet: the batch must refuse the
    // network.
    try {
      BatchNetwork(template, batchElements);
      return ['$name: BatchNetwork accepted a network with $unbatched'];
    } on UnsupportedError {
      return const [];
    }
  }
  final kappas = Float64List(batchElements);
  final batch = BatchNetwork(template, batchElements, blockSize: batchBlock);
  final edge = _firstVolatilityEdge(build());
  final nets = <Network>[];
  for (var e = 0; e < batchElements; e++) {
    final net = build();
    _perturb(
      e,
      template,
      edge,
      setInitial: (i, f, v) {
        net.setInitial(i, f, v);
        batch.setInitialAt(e, i, f, v);
      },
      setKappa: (p, c, k) {
        net.setVolatilityCoupling(p, c, k);
        kappas[e] = k;
      },
    );
    nets.add(net..reset());
  }
  if (edge != null) batch.setVolatilityCoupling(edge.$1, edge.$2, kappas);
  batch.reset();

  final failures = <String>[];
  // Data rows have a column per input, `dimension` for vector inputs;
  // observed rows one flag per input.
  final nIn = template.inputIdxs.length;
  bool isVector(int i) =>
      template.kindOf(i) == NodeKind.efState ||
      template.kindOf(i) == NodeKind.categoricalState;
  final dims = [
    for (final i in template.inputIdxs)
      isVector(i) ? template.vectorLength(i, VectorField.mean) : 1,
  ];
  final nCols = dims.fold(0, (a, d) => a + d);
  final nTime = data.length ~/ nCols;
  final n = template.nNodes;
  const vectorFields = [
    VectorField.mean,
    VectorField.xis,
    VectorField.observationSs,
    VectorField.alpha,
  ];

  /// The first field of node [i] where element [e] differs, or null.
  String? differs(Network net, int e, int i) {
    for (final f in NodeField.values) {
      final want = net.get(i, f), got = batch.get(e, i, f);
      if (!_same(want, got)) {
        return '${f.name}: Network $want, BatchNetwork $got';
      }
    }
    for (final f in vectorFields) {
      final want = net.getVector(i, f), got = batch.getVector(e, i, f);
      if (want.length != got.length) return '${f.name} length';
      for (var j = 0; j < want.length; j++) {
        if (!_same(want[j], got[j])) {
          return '${f.name}[$j]: Network ${want[j]}, BatchNetwork ${got[j]}';
        }
      }
    }
    return null;
  }

  // Trial by trial.
  outer:
  for (var t = 0; t < nTime; t++) {
    final obs = Float64List.sublistView(data, t * nCols, (t + 1) * nCols);
    final flags = observed == null
        ? null
        : Float64List.sublistView(observed, t * nIn, (t + 1) * nIn);
    final dt = timeSteps == null ? 1.0 : timeSteps[t];
    batch.step(obs, observed: flags, timeStep: dt);
    for (var e = 0; e < batchElements; e++) {
      nets[e].step(obs, observed: flags, timeStep: dt);
      for (var i = 0; i < n; i++) {
        if (differs(nets[e], e, i) case final d?) {
          failures.add('$name step: t=$t element $e node $i $d');
          break outer;
        }
      }
    }
  }

  // Whole series in blocks, with a response model.
  final firstInput = template.kindOf(template.inputIdxs.first);
  final response = switch (firstInput) {
    NodeKind.binaryState => const FirstLevelBinarySurprise(),
    NodeKind.categoricalState => const CategoricalSurprise(),
    _ when n < 2 =>
      // A lone node (a fixed-rate ef-state): no surprise is defined, so
      // sum its state instead.
      ResponseFunction.fn(
        (net, t) => net.get(0, NodeField.mean) + net.get(0, NodeField.nus),
        batch: (net, t, e) =>
            net.get(e, 0, NodeField.mean) + net.get(e, 0, NodeField.nus),
      ),
    _ => const TotalGaussianSurprise() as ResponseFunction,
  };
  final perTrial = Float64List(nTime * batchElements);
  final totals = batch.surprise(
    data,
    response,
    observed: observed,
    timeSteps: timeSteps,
    perTrial: perTrial,
  );
  void compareRuns(
    String what,
    Float64List Function(int e) dataOf,
    Float64List? Function(int e) observedOf,
  ) {
    final scalarPerTrial = Float64List(nTime);
    for (var e = 0; e < batchElements; e++) {
      final total = nets[e].surprise(
        dataOf(e),
        response,
        observed: observedOf(e),
        timeSteps: timeSteps,
        perTrial: scalarPerTrial,
      );
      if (!_same(total, totals[e])) {
        failures.add(
          '$name $what: surprise element $e: $total vs ${totals[e]}',
        );
      }
      for (var t = 0; t < nTime; t++) {
        if (!_same(scalarPerTrial[t], perTrial[t * batchElements + e])) {
          failures.add('$name $what: per-trial surprise element $e t=$t');
          break;
        }
      }
      for (var i = 0; i < n; i++) {
        if (differs(nets[e], e, i) case final d?) {
          failures.add('$name $what: final state element $e node $i $d');
        }
      }
    }
  }

  compareRuns('shared data', (_) => data, (_) => observed);

  // Each element its own series: continuous (and ef-state) inputs shifted
  // by 0.01 e, and input k of element e also missing where
  // (t + 2e + k) % 7 == 0, so the elements differ in which children they
  // observe.
  final shift = [
    for (final (k, i) in template.inputIdxs.indexed)
      for (var j = 0; j < dims[k]; j++)
        template.kindOf(i) == NodeKind.continuousState ||
            template.kindOf(i) == NodeKind.efState,
  ];
  final own = [
    for (var e = 0; e < batchElements; e++)
      Float64List.fromList([
        for (var t = 0; t < nTime; t++)
          for (var c = 0; c < nCols; c++)
            data[t * nCols + c] + (shift[c] ? 0.01 * e : 0.0),
      ]),
  ];
  final ownObserved = [
    for (var e = 0; e < batchElements; e++)
      Float64List.fromList([
        for (var t = 0; t < nTime; t++)
          for (var k = 0; k < nIn; k++)
            (t + 2 * e + k) % 7 == 0
                ? 0.0
                : (observed == null ? 1.0 : observed[t * nIn + k]),
      ]),
  ];
  Float64List interleave(List<Float64List> perElement) {
    final width = perElement.first.length;
    final out = Float64List(width * batchElements);
    for (var r = 0; r < width; r++) {
      for (var e = 0; e < batchElements; e++) {
        out[r * batchElements + e] = perElement[e][r];
      }
    }
    return out;
  }

  batch.surprise(
    interleave(own),
    response,
    observed: interleave(ownObserved),
    timeSteps: timeSteps,
    perElement: true,
    into: totals,
    perTrial: perTrial,
  );
  compareRuns('per-element data', (e) => own[e], (e) => ownObserved[e]);
  return failures;
}
