/// The Python export against pyhgf: every parity fixture (plain and with
/// values changed after building) is exported, run by pyhgf's JAX backend
/// in float64, and compared with hgf.dart's trajectories at the parity
/// suite's tolerance. Then a generated PyMC fit runs and lands near
/// hgf_fit's MAP estimate.
///
/// Needs `tool/parity/.venv` (`cd tool/parity && uv sync`); run with
/// `melos run test:export-python`.
@TestOn('vm')
@Tags(['pyhgf'])
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/codegen.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:test/test.dart';

import 'src/fixture_networks.dart';

final String _python = File('../../tool/parity/.venv/bin/python').absolute.path;
final Directory _dir = Directory('.dart_tool/codegen_python_test');

/// The fixtures' custom couplings in JAX.
const _jaxCouplings = {'sin': 'jnp.sin', 'softplus': 'jax.nn.softplus'};

const _fields = ['mean', 'expected_mean', 'precision', 'expected_precision'];

/// [script] with the custom couplings' stubs replaced by JAX functions.
String _fillStubs(String script) {
  var s = script;
  for (final MapEntry(:key, :value) in _jaxCouplings.entries) {
    s = s.replaceFirst(
      RegExp(
        'def ${key}_coupling\\(x\\):.*?raise NotImplementedError\\([^)]*\\)\\n',
        dotAll: true,
      ),
      'def ${key}_coupling(x):\n    return $value(x)\n',
    );
  }
  return s;
}

/// Mismatches beyond the parity suite's tolerance (rel 1e-9, abs 1e-12;
/// NaN matches NaN).
String? _mismatch(List<double> want, List<double> got, double relTol) {
  if (want.length != got.length) {
    return 'length ${got.length} != ${want.length}';
  }
  for (var t = 0; t < want.length; t++) {
    final e = want[t], a = got[t];
    if (e.isNaN && a.isNaN) continue;
    if (e == a) continue;
    final diff = (e - a).abs();
    if (diff <= 1e-12) continue;
    if (diff / math.max(e.abs(), a.abs()) > relTol || diff.isNaN) {
      return 't=$t: hgf.dart $e, pyhgf $a';
    }
  }
  return null;
}

void main() {
  final venv = File(_python).existsSync();

  test(
    'pyhgf runs the exported networks like hgf.dart',
    () async {
      if (_dir.existsSync()) _dir.deleteSync(recursive: true);
      _dir.createSync(recursive: true);
      // Fixtures that check proposed pyhgf patches, not pyhgf as released.
      final cases = [
        // pyhgf resets inputs' tonic volatility (see the next test).
        for (final fx in fixtureNetworks(perturbInputs: false))
          if ((fx.spec['network'] as Map?)?['ehgf_mgf_correction'] != true &&
              (fx.spec['network'] as Map?)?['dp_skip_missing'] != true)
            fx,
      ];
      final harness = StringBuffer()
        ..writeln('import importlib.util, json, sys')
        ..writeln('import numpy as np')
        ..writeln('out = {}')
        ..writeln(
          'enc = lambda x: x if np.isfinite(x) else '
          '("NaN" if np.isnan(x) else ("Infinity" if x > 0 else "-Infinity"))',
        )
        ..writeln('for i in range(${cases.length}):')
        ..writeln(
          '    spec = importlib.util.spec_from_file_location(f"case_{i}", '
          'f"${_dir.absolute.path}/case_{i}.py")',
        )
        ..writeln('    m = importlib.util.module_from_spec(spec)')
        ..writeln('    spec.loader.exec_module(m)')
        ..writeln('    d = m.embedded_data()')
        ..writeln('    net = m.build_network()')
        ..writeln(
          '    net.input_data(input_data=d["inputs"], observed=d["observed"], '
          'time_steps=d["time_steps"])',
        )
        ..writeln('    out[i] = {')
        ..writeln(
          '        str(n): {f: [enc(float(x)) for x in np.asarray(t[f])]',
        )
        ..writeln(
          '                 for f in ${jsonEncode(_fields)} '
          'if f in t and np.ndim(t[f]) == 1}',
        )
        ..writeln('        for n, t in net.node_trajectories.items() if n >= 0')
        ..writeln('    }')
        ..writeln('json.dump(out, open(sys.argv[1], "w"))');
      for (var i = 0; i < cases.length; i++) {
        final script = PythonExport(
          ExportSpec(network: cases[i].network.toJson(), data: cases[i].data),
        ).script();
        File('${_dir.path}/case_$i.py').writeAsStringSync(_fillStubs(script));
      }
      File('${_dir.path}/harness.py').writeAsStringSync(harness.toString());
      final outFile = '${_dir.absolute.path}/out.json';
      final run = await Process.run(_python, [
        '${_dir.absolute.path}/harness.py',
        outFile,
      ]);
      expect(run.exitCode, 0, reason: '${run.stdout}\n${run.stderr}');
      final out =
          jsonDecode(File(outFile).readAsStringSync()) as Map<String, Object?>;
      final failures = <String>[];
      for (var i = 0; i < cases.length; i++) {
        final c = cases[i];
        // The parity suite's tolerance for this fixture.
        final rs = c.rshgfMaxRelDiff;
        final tol = rs > 1e-9 ? 2 * rs : 1e-9;
        final traj = c.network.inputData(
          c.data.inputs,
          observed: c.data.observed,
          timeSteps: c.data.timeSteps,
          fields: [for (final f in _fields) NodeField.parse(f)],
        )!;
        final nodes = out['$i'] as Map<String, Object?>;
        for (var n = 0; n < c.network.nNodes; n++) {
          // Vector kinds record other things in pyhgf.
          if (c.network.kindOf(n)
              case NodeKind.efState ||
                  NodeKind.categoricalState ||
                  NodeKind.dpState) {
            continue;
          }
          final py = nodes['$n'] as Map<String, Object?>;
          for (final f in _fields) {
            final got = [
              for (final x in (py[f] as List<Object?>?) ?? const [])
                switch (x) {
                  num() => x.toDouble(),
                  'NaN' => double.nan,
                  'Infinity' => double.infinity,
                  _ => double.negativeInfinity,
                },
            ];
            if (got.isEmpty) continue;
            final m = _mismatch(traj.series(n, NodeField.parse(f)), got, tol);
            if (m != null) failures.add('${c.name}: node $n $f: $m');
          }
        }
      }
      File("${_dir.path}/failures.txt").writeAsStringSync(failures.join("\n"));
      expect(failures, isEmpty, reason: failures.take(20).join("\n"));
    },
    skip: venv ? false : 'needs tool/parity/.venv (uv sync)',
    timeout: const Timeout(Duration(minutes: 15)),
  );

  test('flags what pyhgf would reset', () {
    final net = Network()
      ..addNodes()
      ..addNodes(valueChildren: [0])
      ..addNodes(volatilityChildren: [0]);
    net.setInitial(0, NodeField.tonicVolatility, -2);
    final script = PythonExport(ExportSpec(network: net.toJson())).script();
    expect(
      script,
      contains('pyhgf resets the tonic_volatility of input node 0 to 0'),
    );
  });

  test(
    'the exported PyMC fit finds hgf_fit\'s MAP estimate',
    () async {
      _dir.createSync(recursive: true);
      final model = HgfModel(
        network: Network()
          ..addNodes(kind: NodeKind.binaryState)
          ..addNodes(valueChildren: [0])
          ..addNodes(volatilityChildren: [1], tonicVolatility: -6),
        response: const BinarySoftmaxReadout(),
        parameters: [
          Parameter.node(
            'omega2',
            1,
            NodeField.tonicVolatility,
            prior: TruncatedNormal(-3, 2, lower: -8, upper: 2),
          ),
          Parameter.response(
            'beta',
            responseName: 'inverseTemperature',
            prior: const LogNormal(1, 1),
          ),
        ],
      );
      final rng = math.Random(3);
      final u = Float64List.fromList([
        for (var t = 0; t < 150; t++)
          (rng.nextDouble() < (t < 75 ? 0.8 : 0.2)) ? 1.0 : 0.0,
      ]);
      final responses = model
          .simulate(
            [Subject(inputs: u)],
            [
              [-3.0, 4.0],
            ],
            math.Random(5),
          )
          .single;
      final script = PythonExport(
        ExportSpec(
          model: const FitCodec().encodeModel(model),
          data: ExportData(inputs: u, responses: responses),
          fit: const ExportMap(),
        ),
      ).script();
      File('${_dir.path}/fit.py').writeAsStringSync(script);
      final run = await Process.run(_python, ['${_dir.absolute.path}/fit.py']);
      expect(run.exitCode, 0, reason: '${run.stdout}\n${run.stderr}');
      final map = fitMap(
        model,
        [Subject(inputs: u, responses: responses)],
        restarts: 3,
        random: math.Random(1),
      ).results.single;
      final printed = {
        for (final m in RegExp(
          r'^(\w+) = (\S+)$',
          multiLine: true,
        ).allMatches(run.stdout as String))
          m[1]!: double.parse(m[2]!),
      };
      expect(printed['omega2'], closeTo(map.x[0], 1e-3));
      expect(printed['beta'], closeTo(map.x[1], 1e-3));
    },
    skip: venv ? false : 'needs tool/parity/.venv (uv sync)',
    timeout: const Timeout(Duration(minutes: 5)),
  );
}
