/// The WGSL code generator, without a GPU: which networks compile, the slot
/// layout, and (when `naga` is on the PATH, `cargo install naga-cli`) that
/// the generated shaders validate.
@TestOn('vm')
library;

import 'dart:io';

import 'package:hgf/hgf.dart';
import 'package:hgf_gpu/hgf_gpu.dart';
import 'package:test/test.dart';

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
    final ef = Network()..addEfState();
    expect(() => HgfKernel(ef), throwsUnsupportedError);
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
}
