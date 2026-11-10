import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/codegen.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_playground/src/model/fitting.dart';
import 'package:hgf_playground/src/model/playground.dart';
import 'package:hgf_playground/src/model/presets.dart';

Future<FitWorkers> _inline([int? _]) => FitWorkers.spawn(count: 0);

/// Every recorded field's bits.
List<int> _bits(Trajectories t) => [
  for (final f in t.fields)
    ...Float64List.fromList(t.raw(f)).buffer.asUint64List(),
];

void main() {
  test('every preset exports; the code filters like the playground', () {
    for (final p in presets) {
      final pg = Playground(presetId: p.id, nTrials: 60);
      final f = Fitting(pg, spawnWorkers: _inline);
      // A slider moved after building, which the code must carry.
      final last = pg.network.nNodes - 1;
      if (pg.network.kindOf(last) == NodeKind.continuousState) {
        pg.network.setInitial(last, NodeField.tonicVolatility, -3.25);
      }
      for (final responses in [false, true]) {
        if (responses) {
          if (f.responsesUnavailable != null || f.unsupported != null) break;
          f.setTarget(FitTarget.responses);
          if (p.task == null) f.simulateAgent(seed: 1);
        }
        final spec = f.exportSpec();
        final script = DartExport(spec).script();
        expect(script, contains('Network buildNetwork()'), reason: p.id);
        final python = PythonExport(spec).script();
        expect(python, contains('def build_network():'), reason: p.id);
        expect(python.contains('def fit(data):'), spec.fit != null);
        final withModel = f.canFit && f.free.isNotEmpty;
        expect(spec.model != null, withModel, reason: p.id);
        expect(script.contains('fitMap('), withModel, reason: p.id);
        final data = spec.data!;
        final want = pg.network.inputData(
          data.inputs,
          observed: data.observed,
          fields: NodeField.values,
        )!;
        final got = spec.plan.build().inputData(
          data.inputs,
          observed: data.observed,
          fields: NodeField.values,
        )!;
        expect(_bits(got), _bits(want), reason: p.id);
        if (spec.model case final m?) {
          // The exported model is the one the fit runs.
          final model = const FitCodec().decodeModel(m);
          expect(model.names, [
            for (final c in f.offered)
              if (f.free.contains(c.id)) c.id,
          ], reason: p.id);
          expect(spec.fit, isA<ExportMap>(), reason: p.id);
        }
      }
    }
  });
}
