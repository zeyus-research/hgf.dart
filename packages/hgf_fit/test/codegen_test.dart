@TestOn('vm')
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
import 'src/traj_hash.dart';

/// The custom couplings of hgf's parity fixtures as Dart source: they
/// replace the generated stubs.
const _couplingSource = {
  'sin': '''
final sinCoupling = CustomCoupling(
  'sin',
  g: math.sin,
  dg: math.cos,
  d2g: (x) => -math.sin(x),
  d3g: (x) => -math.cos(x),
);
''',
  'softplus': '''
double _sigmoid(double x) => 1 / (1 + math.exp(-x));
final softplusCoupling = CustomCoupling(
  'softplus',
  g: (x) => math.max(x, 0.0) + math.log(1.0 + math.exp(-x.abs())),
  dg: _sigmoid,
  d2g: (x) => _sigmoid(x) * (1.0 - _sigmoid(x)),
);
''',
};

/// A generated script, with the custom couplings' stubs filled in.
String _fillStubs(String script) {
  var s = script;
  for (final MapEntry(:key, :value) in _couplingSource.entries) {
    final id = '${key}Coupling';
    s = s.replaceFirst(
      RegExp('final $id = CustomCoupling\\(.*?\\n\\);\\n', dotAll: true),
      value,
    );
  }
  if (s != script && !s.contains("import 'dart:math' as math;")) {
    s = s.replaceFirst(
      "import 'dart:io';",
      "import 'dart:io';\nimport 'dart:math' as math;",
    );
  }
  return s;
}

/// A scratch directory inside the package, so the generated files resolve
/// `package:hgf` and `package:hgf_fit`.
final Directory _dir = Directory('.dart_tool/codegen_test');

void main() {
  test('dartDouble round-trips', () {
    for (final v in [
      0.0,
      -0.0,
      1.0,
      -3.5,
      1e-6,
      1e-7,
      1e4,
      1e10,
      1e21,
      123456789.0,
      0.1 + 0.2,
      math.pi,
      double.minPositive,
      double.maxFinite,
    ]) {
      final s = dartDouble(v);
      expect(double.parse(s), v, reason: s);
      expect(double.parse(s).isNegative, v.isNegative, reason: s);
    }
    expect(dartDouble(1e10), '1e10');
    expect(dartDouble(-4), '-4.0');
    expect(dartDouble(double.nan), 'double.nan');
    expect(dartDouble(double.negativeInfinity), 'double.negativeInfinity');
  });

  group('NetworkPlan', () {
    test('folds implied nodes back into their calls', () {
      final net = Network()
        ..addEfState(nNodes: 2, learning: EfLearning.hgf2)
        ..addCategoricalState(nCategories: 3, valuePrecision: 2)
        ..addDpState(batchSize: 4, alpha: 0.5)
        ..addEfState(nNodes: 4, valueChildren: [22], nus: 10)
        ..addNodes(kind: NodeKind.binaryState, nNodes: 2)
        ..addNodes(valueChildren: [27, 28], tonicVolatility: -2);
      final plan = NetworkPlan.fromJson(net.toJson());
      expect(plan.calls.map((c) => c.runtimeType.toString()), [
        'AddEfState',
        'AddCategoricalState',
        'AddDpState',
        'AddEfState',
        'AddNodes',
        'AddNodes',
      ]);
      final ef = plan.calls.first as AddEfState;
      expect(
        (ef.nNodes, ef.learning, ef.impliedNodes),
        (2, EfLearning.hgf2, 12),
      );
      final cat = plan.calls[1] as AddCategoricalState;
      expect(cat.arguments, {'valuePrecision': 2.0});
      final binary = plan.calls[4] as AddNodes;
      expect((binary.kind, binary.nNodes), (NodeKind.binaryState, 2));
    });

    test('ef-state runs: fixed nodes before a learning call', () {
      final net = Network()
        ..addEfState(nus: 5)
        ..addEfState(learning: EfLearning.hgf1)
        ..addEfState(learning: EfLearning.hgf3, nNodes: 2);
      final plan = NetworkPlan.fromJson(net.toJson());
      expect(
        plan.calls.map(
          (c) => switch (c) {
            AddEfState(:final nNodes, :final learning) => '$nNodes $learning',
            _ => '$c',
          },
        ),
        ['1 EfLearning.fixed', '1 EfLearning.hgf1', '2 EfLearning.hgf3'],
      );
    });

    test('values changed after building become setInitial', () {
      final net = Network()
        ..addNodes(precision: 1e4)
        ..addNodes(valueChildren: [0]);
      net
        ..setInitial(0, NodeField.tonicVolatility, -1) // forced 0 on inputs
        ..setInitial(1, NodeField.tonicVolatility, -3);
      final plan = NetworkPlan.fromJson(net.toJson());
      final code = DartExport(ExportSpec(network: net.toJson())).network();
      expect(plan.calls.whereType<SetInitial>().single.node, 0);
      expect(code, contains('tonicVolatility: -3.0'));
      expect(code, contains('setInitial(0, NodeField.tonicVolatility, -1.0)'));
    });
  });

  group('generated Dart', () {
    late List<_Case> cases;
    late Map<String, String> results;

    setUpAll(() async {
      if (_dir.existsSync()) _dir.deleteSync(recursive: true);
      _dir.createSync(recursive: true);
      cases = _fixtureCases();
      final harness = StringBuffer()
        ..writeln("import 'dart:convert';")
        ..writeln("import 'package:hgf_fit/hgf_fit.dart';")
        ..writeln("import '../../test/src/traj_hash.dart';");
      for (var i = 0; i < cases.length; i++) {
        final c = cases[i];
        final script = _fillStubs(DartExport(c.spec).script());
        File('${_dir.path}/case_$i.dart').writeAsStringSync(script);
        File(
          '${_dir.path}/case_$i.csv',
        ).writeAsStringSync(c.spec.data!.toCsv(c.spec.plan));
        harness.writeln("import 'case_$i.dart' as c$i;");
      }
      harness.writeln('void main() {');
      for (var i = 0; i < cases.length; i++) {
        harness
          ..writeln('  {')
          ..writeln('    final e = c$i.embeddedData();')
          ..writeln("    final f = c$i.readCsv('${_dir.path}/case_$i.csv');")
          ..writeln(
            '    print(jsonEncode([$i, '
            'trajectoryHash(c$i.buildNetwork(), e.inputs, '
            'observed: e.observed, timeSteps: e.timeSteps), '
            'trajectoryHash(c$i.buildNetwork(), f.inputs, '
            'observed: f.observed, timeSteps: f.timeSteps), '
            '${cases[i].spec.model == null ? 'null' : 'jsonEncode(const FitCodec().encodeModel(c$i.buildModel()))'}]));',
          )
          ..writeln('  }');
      }
      harness.writeln('}');
      File('${_dir.path}/harness.dart').writeAsStringSync(harness.toString());
      final run = await Process.run(Platform.resolvedExecutable, [
        'run',
        '${_dir.path}/harness.dart',
      ]);
      if (run.exitCode != 0) {
        fail('The harness failed:\n${run.stdout}\n${run.stderr}');
      }
      results = {};
      for (final line in LineSplitter.split(run.stdout as String)) {
        final r = jsonDecode(line) as List<Object?>;
        results['${r[0]}'] = line;
      }
    });

    test('analyses clean', () async {
      final r = await Process.run(Platform.resolvedExecutable, [
        'analyze',
        '--fatal-infos',
        _dir.path,
      ]);
      expect(r.exitCode, 0, reason: '${r.stdout}\n${r.stderr}');
    });

    test('rebuilds every fixture network bit for bit', () {
      var checked = 0;
      for (var i = 0; i < cases.length; i++) {
        final c = cases[i];
        final r = jsonDecode(results['$i']!) as List<Object?>;
        final data = c.spec.data!;
        final want = trajectoryHash(
          c.network,
          data.inputs,
          observed: data.observed,
          timeSteps: data.timeSteps,
        );
        expect(r[1], want, reason: '${c.name}: embedded data');
        expect(r[2], want, reason: '${c.name}: CSV data');
        if (c.spec.model case final model?) {
          final got = jsonDecode(r[3] as String) as Map<String, Object?>;
          expect(
            _comparableModel(got),
            _comparableModel(model),
            reason: '${c.name}: model',
          );
        }
        checked++;
      }
      expect(checked, greaterThan(120));
    });
  });

  test('a generated fit script runs and matches fitting in process', () async {
    _dir.createSync(recursive: true);
    final model = _binaryModel();
    final rng = math.Random(3);
    final u = Float64List.fromList([
      for (var t = 0; t < 120; t++)
        (rng.nextDouble() < (t < 60 ? 0.8 : 0.2)) ? 1.0 : 0.0,
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
    final spec = ExportSpec(
      model: const FitCodec().encodeModel(model),
      data: ExportData(inputs: u, responses: responses),
      dataFile: 'fit_data.csv',
      fit: const ExportMap(restarts: 2),
      title: 'Binary HGF, 2 levels',
    );
    final script = DartExport(spec).script();
    File('${_dir.path}/fit_script.dart').writeAsStringSync(script);
    File(
      '${_dir.path}/fit_data.csv',
    ).writeAsStringSync(spec.data!.toCsv(spec.plan));
    final run = await Process.run(Platform.resolvedExecutable, [
      'run',
      'fit_script.dart',
    ], workingDirectory: _dir.path);
    expect(run.exitCode, 0, reason: '${run.stdout}\n${run.stderr}');
    final subject = Subject(inputs: u, responses: responses);
    final map = fitMap(model, [subject], restarts: 2, random: math.Random(1));
    expect(
      run.stdout as String,
      contains(laplace(model, map).results.single.toString()),
    );
  });
}

final class _Case {
  _Case(this.name, this.network, this.spec);

  final String name;
  final Network network;
  final ExportSpec spec;
}

HgfModel _binaryModel() => HgfModel(
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

/// A model per response model kind, on the fixture's network where it fits.
HgfModel? _modelFor(String name, Network network) {
  final inputs = network.inputIdxs;
  final kind = network.kindOf(inputs.first);
  if (inputs.length != 1) return null;
  final parentOfInput = network.valueParentsOf(inputs.single);
  switch (kind) {
    case NodeKind.binaryState when parentOfInput.isNotEmpty:
      final p = parentOfInput.first;
      return HgfModel(
        network: network,
        response: const BinarySoftmaxReadout(inverseTemperature: 2),
        parameters: [
          Parameter.node(
            'omega',
            p,
            NodeField.tonicVolatility,
            prior: const Normal(-4, 2),
            transform: const IdentityTransform(),
          ),
          Parameter.response('inverseTemperature', prior: const HalfNormal(5)),
        ],
      );
    case NodeKind.continuousState when parentOfInput.isNotEmpty:
      final p = parentOfInput.first;
      final vol = network.volatilityParentsOf(p);
      return HgfModel(
        network: network,
        response: const GaussianReadout(precision: 4),
        parameters: [
          Parameter(
            'omega',
            targets: [
              NodeTarget(p, NodeField.tonicVolatility),
              for (final v in vol) NodeTarget(v, NodeField.tonicVolatility),
            ],
            prior: const Uniform(-8, 0),
            initial: -3,
          ),
          Parameter.valueCoupling(
            'kappa',
            p,
            inputs.single,
            prior: const Flat(lower: 0, upper: 2),
          ),
          Parameter.response('precision', prior: const GammaPrior(2, 1)),
        ],
        blockSize: 64,
      );
    case NodeKind.categoricalState:
      return HgfModel(
        network: network,
        response: CategoricalSoftmaxReadout(node: inputs.single),
        parameters: [
          Parameter.node(
            'omega',
            inputs.single +
                network.vectorLength(inputs.single, VectorField.mean) * 2 +
                1,
            NodeField.tonicVolatility,
            prior: const Normal(-4, 2),
          ),
        ],
      );
    case NodeKind.dpState:
      return HgfModel(
        network: network,
        response: const ObservationSurprise(DpSurprise()),
        parameters: [
          Parameter.node(
            'alpha',
            inputs.single,
            NodeField.alpha,
            prior: const LogNormal(0, 1),
          ),
        ],
      );
    default:
      return null;
  }
}

List<_Case> _fixtureCases() {
  final cases = <_Case>[
    for (final fx in fixtureNetworks())
      if (fx.perturbed ? null : _modelFor(fx.name, fx.network) case final model)
        _Case(
          fx.name,
          fx.network,
          ExportSpec(
            network: model == null ? fx.network.toJson() : null,
            model: model == null ? null : const FitCodec().encodeModel(model),
            data: fx.data,
            title: fx.name,
          ),
        ),
  ];
  // A choice task's readout and parameters.
  final igtModel = HgfModel(
    network: IowaGamblingTask.network(),
    response: IowaGamblingTask.readout(),
    parameters: [
      Parameter(
        'omega',
        targets: [
          for (final n in IowaGamblingTask.valueNodes)
            NodeTarget(n, NodeField.tonicVolatility),
        ],
        prior: const Normal(-4, 2),
      ),
      Parameter.response('inverseTemperature', prior: const LogNormal(0, 1)),
    ],
  );
  final igtNet = igtModel.network;
  final rng = math.Random(2);
  final nIn = igtNet.inputIdxs.length;
  final payoffs = Float64List.fromList([
    for (var i = 0; i < 50 * nIn; i++) rng.nextDouble() * 2 - 1,
  ]);
  final observed = Float64List(50 * nIn);
  for (var t = 0; t < 50; t++) {
    observed[t * nIn + rng.nextInt(nIn)] = 1;
  }
  cases.add(
    _Case(
      'Iowa Gambling Task',
      igtNet,
      ExportSpec(
        model: const FitCodec().encodeModel(igtModel),
        data: ExportData(
          inputs: payoffs,
          observed: observed,
          responses: Float64List.fromList([
            for (var t = 0; t < 50; t++) rng.nextInt(nIn).toDouble(),
          ]),
        ),
      ),
    ),
  );
  return cases;
}

/// A model's JSON with its network made comparable (builder-call-only keys
/// dropped, see `NetworkPlan`).
String _comparableModel(Map<String, Object?> json) {
  final network = json['network'] as Map<String, Object?>;
  return jsonEncode({
    ...json,
    'network': {
      ...network,
      'nodes': [
        for (final n in network['nodes'] as List<Object?>)
          {
            for (final MapEntry(:key, :value)
                in (n as Map<String, Object?>).entries)
              if (key != 'params' && key != 'xis') key: value,
          },
      ],
    },
  });
}
