/// The WGSL code generator, without a GPU: which networks compile, the slot
/// layout, and (when `naga` is on the PATH, `cargo install naga-cli`) that
/// the generated shaders validate.
@TestOn('vm')
library;

import 'dart:io';

import 'package:hgf/hgf.dart';
import 'package:hgf_gpu/hgf_gpu.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:test/test.dart';

import 'src/fit_check.dart';

Network _hgf(VolatilityUpdates mode, {bool meanField = false}) =>
    Network(volatilityUpdates: mode, meanFieldUpdates: meanField)
      ..addNodes(precision: 1e4)
      ..addNodes(valueChildren: [0], mean: 1.04, tonicVolatility: -8.0)
      ..addNodes(volatilityChildren: [1], tonicVolatility: -2.0);

void main() {
  test('slots: node fields, then edges, then the response', () {
    final k = HgfKernel(_hgf(VolatilityUpdates.standard));
    final n = 3, f = gpuNodeFields.length;
    expect(k.slot(1, NodeField.tonicVolatility), 10 * n + 1);
    expect(k.valueCouplingSlot(1, 0), f * n);
    expect(k.volatilityCouplingSlot(2, 1), f * n + 1);
    expect(k.totalSlot, f * n + 2);
    expect(k.resultSlot, k.nSlots - 1);
    expect(k.initial[k.slot(1, NodeField.mean)], 1.04);
    expect(k.initial[k.slot(0, NodeField.precision)], 1e4);
    expect(k.initial[k.valueCouplingSlot(1, 0)], 1.0);
    final state = k.initialState(4);
    expect(state.length, k.nSlots * 4);
    expect(state[k.slot(1, NodeField.tonicVolatility) * 4 + 3], -8.0);
  });

  test('unsupported node kinds and couplings throw', () {
    // A categorical node whose binary node's parent has another child.
    final shared = Network()
      ..addCategoricalState(nCategories: 2)
      ..addNodes(valueParents: [3]);
    expect(() => HgfKernel(shared), throwsUnsupportedError);
    expect(
      () => HgfKernel(
        Network()..addNodes(),
        response: GpuResponse.categoricalSurprise,
      ),
      throwsArgumentError,
    );
    // ef-state learning whose rate-scaling node has another child.
    expect(
      HgfKernel(Network()..addEfState(learning: EfLearning.hgf1)).nColumns,
      1,
    );
    final extra = Network()
      ..addEfState(learning: EfLearning.hgf1)
      ..addNodes(valueParents: [2]);
    expect(() => HgfKernel(extra), throwsUnsupportedError);
    final custom = Network()
      ..addNodes()
      ..addNodes(
        valueChildren: [0],
        coupling: CustomCoupling(
          'sq',
          g: (x) => x * x,
          dg: (x) => 2 * x,
          d2g: (x) => 2,
        ),
      );
    expect(() => HgfKernel(custom), throwsUnsupportedError);
  });

  final naga = Process.runSync('which', ['naga']).exitCode == 0;
  test('every update mode and option generates valid WGSL', () {
    final dir = Directory.systemTemp.createTempSync('hgf_wgsl');
    addTearDown(() => dir.deleteSync(recursive: true));
    var k = 0;
    for (final mode in VolatilityUpdates.values) {
      for (final mf in [false, true]) {
        for (final options in [(false, false, false), (true, true, true)]) {
          final kernel = HgfKernel(
            _hgf(mode, meanField: mf),
            perElementData: options.$1,
            observedMask: options.$2,
            timeSteps: options.$3,
          );
          final f = File('${dir.path}/k${k++}.wgsl')
            ..writeAsStringSync(kernel.source);
          final r = Process.runSync('naga', [f.path]);
          expect(r.exitCode, 0, reason: '$mode mf=$mf $options\n${r.stderr}');
        }
      }
    }
  }, skip: naga ? false : 'naga not installed');

  test('ef-state networks: columns, slots and valid WGSL', () {
    final dir = Directory.systemTemp.createTempSync('hgf_ef_wgsl');
    addTearDown(() => dir.deleteSync(recursive: true));
    var k = 0;
    for (final mode in VolatilityUpdates.values) {
      final nets = [
        Network(volatilityUpdates: mode)..addEfState(dimension: 3, nus: 2),
        Network(volatilityUpdates: mode)
          ..addEfState(dimension: 2, learning: EfLearning.hgf2),
        Network(volatilityUpdates: mode)
          ..addNodes()
          ..addNodes(valueChildren: [0])
          ..addEfState(nNodes: 2, learning: EfLearning.hgf3),
      ];
      for (final net in nets) {
        for (final options in [(false, false, false), (true, true, true)]) {
          final kernel = HgfKernel(
            net,
            perElementData: options.$1,
            observedMask: options.$2,
            timeSteps: options.$3,
          );
          expect(kernel.nColumns, net.nInputColumns);
          expect(kernel.nInputs, net.inputIdxs.length);
          for (final i in [
            for (var i = 0; i < net.nNodes; i++)
              if (net.kindOf(i) == NodeKind.efState) i,
          ]) {
            final xis = net.getInitialVector(i, VectorField.xis);
            expect(kernel.xisLength(i), xis.length);
            for (var q = 0; q < xis.length; q++) {
              expect(kernel.initial[kernel.xisSlot(i, q)], xis[q]);
            }
            expect(
              kernel.initial[kernel.slot(i, NodeField.nus)],
              net.getInitial(i, NodeField.nus),
            );
          }
          if (!naga) continue;
          final f = File('${dir.path}/e${k++}.wgsl')
            ..writeAsStringSync(kernel.source);
          final r = Process.runSync('naga', [f.path]);
          expect(r.exitCode, 0, reason: '$mode $options\n${r.stderr}');
        }
      }
    }
  });

  test('categorical networks: columns, slots and valid WGSL', () {
    final dir = Directory.systemTemp.createTempSync('hgf_cat_wgsl');
    addTearDown(() => dir.deleteSync(recursive: true));
    var k = 0;
    for (final mode in VolatilityUpdates.values) {
      for (final meanField in [false, true]) {
        final nets = [
          Network(volatilityUpdates: mode, meanFieldUpdates: meanField)
            ..addCategoricalState(nCategories: 3),
          Network(
              volatilityUpdates: mode,
              meanFieldUpdates: meanField,
              fixes: NetworkFixes.none,
            )
            ..addNodes()
            ..addNodes(valueChildren: [0])
            ..addCategoricalState(nCategories: 4)
            ..addCategoricalState(nCategories: 2),
        ];
        for (final net in nets) {
          for (final response in [
            GpuResponse.categoricalSurprise,
            GpuCategoricalSoftmax(
              node: net.inputIdxs.lastWhere(
                (i) => net.kindOf(i) == NodeKind.categoricalState,
              ),
              inverseTemperature: 2,
            ),
          ]) {
            final kernel = HgfKernel(
              net,
              response: response,
              observedMask: true,
              timeSteps: true,
            );
            expect(kernel.nColumns, net.nInputColumns);
            for (var i = 0; i < net.nNodes; i++) {
              if (net.kindOf(i) != NodeKind.categoricalState) {
                expect(kernel.vectorLength(i, VectorField.alpha), 0);
                continue;
              }
              for (final f in [VectorField.alpha, VectorField.mean]) {
                final v = net.getInitialVector(i, f);
                expect(kernel.vectorLength(i, f), v.length);
                for (var q = 0; q < v.length; q++) {
                  expect(kernel.initial[kernel.vectorSlot(i, f, q)], v[q]);
                }
              }
              expect(
                kernel.slot(i, NodeField.klDivergence),
                kernel.slot(i, NodeField.surprise) - 1,
              );
            }
            if (!naga) continue;
            final f = File('${dir.path}/c${k++}.wgsl')
              ..writeAsStringSync(kernel.source);
            final r = Process.runSync('naga', [f.path]);
            expect(r.exitCode, 0, reason: '$mode $response\n${r.stderr}');
          }
        }
      }
    }
  });

  test('dp-state networks: cooperative kernels, slots and valid WGSL', () {
    final dir = Directory.systemTemp.createTempSync('hgf_dp_wgsl');
    addTearDown(() => dir.deleteSync(recursive: true));
    // Two dp nodes (3 and 5 clusters) after a continuous HGF.
    Network build() => Network()
      ..addNodes()
      ..addNodes(valueChildren: [0])
      ..addDpState(batchSize: 3, alpha: 0.5)
      ..addEfState(nNodes: 3, valueChildren: [2], nus: 4)
      ..addDpState(batchSize: 5, sensoryPrecision: 2)
      ..addEfState(nNodes: 5, valueChildren: [6]);
    final net = build()..build();
    var k = 0;
    for (final options in [(false, false, false), (true, true, true)]) {
      final kernel = HgfKernel(
        net,
        response: GpuResponse.dpSurprise,
        perElementData: options.$1,
        observedMask: options.$2,
        timeSteps: options.$3,
      );
      expect(kernel.cooperative, isTrue);
      expect(kernel.workgroupSize, 256);
      expect(kernel.workgroups(70000), (35000, 2));
      expect(kernel.nColumns, 3);
      for (final (dp, clusters) in [(2, 3), (6, 5)]) {
        expect(
          kernel.initial[kernel.slot(dp, NodeField.alpha)],
          net.getInitial(dp, NodeField.alpha),
        );
        expect(
          kernel.initial[kernel.slot(dp, NodeField.sensoryPrecision)],
          net.getInitial(dp, NodeField.sensoryPrecision),
        );
        for (final f in [
          VectorField.n,
          VectorField.expectedMeans,
          VectorField.expectedSigmas,
          VectorField.activated,
        ]) {
          expect(kernel.vectorLength(dp, f), clusters);
          final v = net.getInitialVector(dp, f);
          for (var q = 0; q < clusters; q++) {
            expect(kernel.initial[kernel.vectorSlot(dp, f, q)], v[q]);
          }
        }
        // A cluster carries its mean only (and its variance).
        final cluster = dp + 1;
        expect(kernel.xisLength(cluster), 1);
        expect(
          kernel.initial[kernel.xisSlot(cluster, 0)],
          net.getInitialVector(cluster, VectorField.xis)[0],
        );
      }
      if (!naga) continue;
      final f = File('${dir.path}/d${k++}.wgsl')
        ..writeAsStringSync(kernel.source);
      final r = Process.runSync('naga', [f.path]);
      expect(r.exitCode, 0, reason: '$options\n${r.stderr}');
    }
    // A likelihood that reads only the HGF skips the dp steps.
    FitKernel fit(ResponseModel response, {bool gradients = false}) =>
        FitKernel(
          HgfModel(
            network: build(),
            response: response,
            parameters: [
              Parameter.node(
                'alpha',
                2,
                NodeField.alpha,
                prior: const LogNormal(-2, 2),
              ),
              Parameter.node(
                'omega',
                1,
                NodeField.tonicVolatility,
                prior: const Normal(-4, 2),
              ),
            ],
          ),
          gradients: gradients,
        );
    expect(
      fit(const ObservationSurprise(FirstLevelGaussianSurprise())).cooperative,
      isFalse,
    );
    final dpFit = fit(const ObservationSurprise(DpSurprise()));
    expect(dpFit.cooperative, isTrue);
    expect(dpFit.workgroups(3), (3, 1));
    expect(
      () => fit(const ObservationSurprise(DpSurprise()), gradients: true),
      throwsUnsupportedError,
    );
    if (!naga) return;
    final f = File('${dir.path}/fit.wgsl')..writeAsStringSync(dpFit.source);
    final r = Process.runSync('naga', [f.path]);
    expect(r.exitCode, 0, reason: '${r.stderr}');
  });

  test('fit kernels: every update mode, response and tangent width', () {
    final dir = Directory.systemTemp.createTempSync('hgf_fit_wgsl');
    addTearDown(() => dir.deleteSync(recursive: true));
    var k = 0;
    for (final c in fitCases()) {
      for (final gradients in [false, true]) {
        final kernel = FitKernel(c.model, gradients: gradients);
        expect(kernel.directions, gradients ? c.model.dimension : 0);
        expect(kernel.nSlots, kernel.resultSlot + 1 + kernel.directions);
        if (!naga) continue;
        final f = File('${dir.path}/f${k++}.wgsl')
          ..writeAsStringSync(kernel.source);
        final r = Process.runSync('naga', [f.path]);
        expect(r.exitCode, 0, reason: '${c.name} $gradients\n${r.stderr}');
      }
    }
  });

  test('fit kernels reject what they can\'t compile', () {
    final model = HgfModel(
      network: Network()
        ..addNodes(kind: NodeKind.binaryState)
        ..addNodes(valueChildren: [0]),
      response: ResponseModel.fn(surprise: (_, _, _, _, _, _) => 0),
      parameters: [
        Parameter.node(
          'omega2',
          1,
          NodeField.tonicVolatility,
          prior: const Normal(-3, 2),
        ),
      ],
    );
    expect(() => FitKernel(model), throwsUnsupportedError);
  });
}
