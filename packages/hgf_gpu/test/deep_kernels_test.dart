/// The deep-network WGSL generator, without a GPU: what compiles, the
/// bindings each kernel declares, split-K choice, and (when `naga` is on
/// the PATH) that every generated shader validates.
@TestOn('vm')
library;

import 'dart:io';

import 'package:hgf_deep/hgf_deep.dart';
import 'package:hgf_gpu/hgf_gpu.dart';
import 'package:test/test.dart';

DeepNetwork _net({
  DeepVolatilityUpdate update = DeepVolatilityUpdate.unbounded,
  bool ffu = false,
  bool meanField = false,
  LayerKind bottom = LayerKind.volatile,
}) =>
    DeepNetwork(
        coupling: CouplingFn.tanh,
        volatilityUpdates: update,
        feedforwardUncertainty: ffu,
        meanFieldUpdates: meanField,
        tonicVolatility: true,
      )
      ..addLayer(3, kind: bottom)
      ..addLayer(5, kind: LayerKind.binary, coupling: CouplingFn.sigmoid)
      ..addLayer(4, coupling: CouplingFn.gelu)
      ..addLayer(2);

List<DeepKernelSource> _all(DeepKernels k) => [
  for (final opt in DeepOptimiserKind.values)
    ...k.batchUpdate(errors: true, optimiser: opt, updatePrecisions: true),
  ...k.predict(),
];

void main() {
  test('networks the GPU cannot run throw', () {
    expect(
      () => DeepKernels(DeepNetwork()..addLayer(2), capacity: 8),
      throwsArgumentError,
    );
    final custom = DeepNetwork()
      ..addLayer(2)
      ..addLayer(
        2,
        coupling: CustomCoupling(
          'id',
          g: (x) => x,
          dg: (x) => 1,
          d2g: (x) => 0,
        ),
      );
    expect(
      () => DeepKernels(custom, capacity: 8),
      throwsA(isA<UnsupportedError>()),
    );
  });

  test('kernels declare only the buffers they use', () {
    final k = DeepKernels(_net(), capacity: 64);
    final byName = {for (final s in _all(k)) s.name: s};
    expect(byName['top']!.bindings, contains(DeepBinding.x));
    expect(byName['top']!.bindings, isNot(contains(DeepBinding.y)));
    expect(byName['update layer 0']!.bindings, contains(DeepBinding.y));
    expect(byName['input errors']!.bindings, contains(DeepBinding.out));
    expect(byName['carry']!.bindings, isNot(contains(DeepBinding.x)));
    for (final s in byName.values) {
      expect(s.bindings, contains(DeepBinding.uniform));
      for (final b in DeepBinding.values) {
        if (b == DeepBinding.uniform) continue;
        expect(
          s.source.contains('var<storage, ${b.access}> ${b.name}:'),
          s.bindings.contains(b),
          reason: '${s.name}: ${b.name}',
        );
      }
    }
  });

  test('split-K only where chunks get at least 256 samples', () {
    String names(int capacity) => [
      for (final s in DeepKernels(_net(), capacity: capacity).batchUpdate(
        errors: false,
        optimiser: DeepOptimiserKind.adam,
        updatePrecisions: false,
      ))
        if (s.name.contains('apply')) s.name,
    ].join(', ');
    expect(names(256), isEmpty);
    expect(names(4096), contains('gradient layer 1 apply'));
    final k = DeepKernels(_net(), capacity: 4096);
    final partial = k
        .batchUpdate(
          errors: false,
          optimiser: DeepOptimiserKind.sgd,
          updatePrecisions: false,
        )
        .firstWhere((s) => s.name == 'gradient layer 1 partial');
    // The x grid covers every split.
    expect(partial.grid(4096).$1, greaterThan(1));
  });

  test('sequential fit: one workgroup per member, bias corrections '
      'only for Adam', () {
    expect(
      () => DeepKernels(_net(), capacity: 8).sequentialFit(),
      throwsStateError,
    );
    final k = DeepKernels.sequential(_net());
    expect(k.capacity, 1);
    final sgd = k.sequentialFit(optimiser: DeepOptimiserKind.sgd);
    final adam = k.sequentialFit(optimiser: DeepOptimiserKind.adam);
    expect(sgd.bindings, isNot(contains(DeepBinding.biasCorrection)));
    expect(adam.bindings, contains(DeepBinding.biasCorrection));
    expect(k.sequentialFit().bindings, contains(DeepBinding.out));
    expect(adam.grid(3), (3, 1));
    expect(adam.grid(70000), (deepFitGridWidth, 2));
    expect(adam.source, contains('@workgroup_size(64)'));
    expect(
      k.sequentialFit(workgroupSize: 128).source,
      contains('@workgroup_size(128)'),
    );
    // In storage, every phase ends in a storage barrier; with the state in
    // workgroup memory, only each row does; with the parameters there too,
    // none does, and storage is touched only to load and store them.
    final storage = adam.source;
    final ws = DeepKernels.sequential(
      _net(),
      workgroupState: true,
    ).sequentialFit(optimiser: DeepOptimiserKind.adam).source;
    final wp = DeepKernels.sequential(
      _net(),
      workgroupState: true,
      workgroupParams: true,
    ).sequentialFit(optimiser: DeepOptimiserKind.adam).source;
    int count(String src, String what) => what.allMatches(src).length;
    expect(count(storage, 'storageBarrier'), greaterThan(5));
    expect(count(storage, 'var<workgroup>'), 0);
    expect(count(ws, 'storageBarrier'), 1);
    expect(count(ws, 'var<workgroup> WS'), 1);
    expect(count(wp, 'storageBarrier'), 0);
    expect(count(wp, RegExp(r'\bP\[').pattern), 0);
    expect(RegExp(r'\bP\[').allMatches(wp).length, 2);
    expect(
      () => DeepKernels.sequential(_net(), workgroupParams: true),
      throwsArgumentError,
    );
  });

  final naga = Process.runSync('which', ['naga']).exitCode == 0;
  test('every variant validates with naga', () {
    final dir = Directory.systemTemp.createTempSync('deep_wgsl');
    try {
      for (final update in DeepVolatilityUpdate.values) {
        for (final (ffu, meanField) in [
          (false, false),
          (true, false),
          (false, true),
        ]) {
          for (final bottom in LayerKind.values) {
            if (bottom == LayerKind.binary && ffu) continue;
            final k = DeepKernels(
              _net(
                update: update,
                ffu: ffu,
                meanField: meanField,
                bottom: bottom,
              ),
              capacity: 1024,
            );
            final seqs = [
              for (final (ws, wp) in [
                (false, false),
                (true, false),
                (true, true),
              ])
                DeepKernels.sequential(
                  _net(
                    update: update,
                    ffu: ffu,
                    meanField: meanField,
                    bottom: bottom,
                  ),
                  workgroupState: ws,
                  workgroupParams: wp,
                ),
            ];
            for (final s in [
              ..._all(k),
              for (final seq in seqs)
                for (final opt in [null, ...DeepOptimiserKind.values])
                  seq.sequentialFit(optimiser: opt),
            ]) {
              final f = File('${dir.path}/k.wgsl')..writeAsStringSync(s.source);
              final r = Process.runSync('naga', [f.path]);
              expect(
                r.exitCode,
                0,
                reason:
                    '$update ffu=$ffu meanField=$meanField $bottom '
                    '${s.name}:\n${r.stderr}${r.stdout}',
              );
            }
          }
        }
      }
    } finally {
      dir.deleteSync(recursive: true);
    }
  }, skip: naga ? false : 'naga not installed');
}
