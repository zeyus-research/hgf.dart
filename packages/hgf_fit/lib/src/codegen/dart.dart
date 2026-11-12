import 'package:hgf/hgf.dart';

import '../codec.dart';
import '../parameter.dart';
import 'export_spec.dart';
import 'plan.dart';

/// Generates Dart code that uses `package:hgf` (and `package:hgf_fit`).
///
/// [script] writes a runnable script: `buildNetwork()`, `buildModel()` when
/// the spec has a model, the data (embedded, and a CSV reader), and a
/// `main` that filters the data and runs the fit. [network] and [model]
/// give just those expressions.
///
/// What can't be generated (custom couplings, custom priors, transforms
/// and response models, `ResponseModel.fn`) becomes a stub that throws
/// `UnimplementedError`, with a `TODO` comment saying what to write.
final class DartExport {
  DartExport(this.spec);

  final ExportSpec spec;

  final List<String> _stubs = [];
  final Set<String> _stubNames = {};

  /// The network as a `Network(...)..addNodes(...)` cascade.
  /// [terminator] goes after the last call, before its comment.
  String network({String indent = '', String terminator = ''}) {
    final plan = spec.plan;
    final b = StringBuffer();
    final settings = [
      if (plan.volatilityUpdates != VolatilityUpdates.unbounded)
        'volatilityUpdates: VolatilityUpdates.${plan.volatilityUpdates.name}',
      if (plan.meanFieldUpdates) 'meanFieldUpdates: true',
      if (plan.ehgfMgfCorrection) 'ehgfMgfCorrection: true',
      if (plan.fixes != NetworkFixes.recommended)
        plan.fixes == NetworkFixes.none
            ? 'fixes: NetworkFixes.none'
            : 'fixes: NetworkFixes(constantNoiseVariance: '
                  '${plan.fixes.constantNoiseVariance})',
      if (plan.maxPosteriorPrecision != 1e10)
        'maxPosteriorPrecision: ${dartDouble(plan.maxPosteriorPrecision)}',
      if (plan.precisionClippingValue != 1e-6)
        'precisionClippingValue: ${dartDouble(plan.precisionClippingValue)}',
    ];
    b.write(_call('Network', settings, indent));
    final inner = '$indent  ';
    if (plan.calls.isEmpty) return '$b$terminator';
    for (final (i, c) in plan.calls.indexed) {
      final (code, comment) = _buildCall(c, inner);
      b
        ..writeln()
        ..write(inner)
        ..write('..')
        ..write(code);
      if (i == plan.calls.length - 1) b.write(terminator);
      if (comment != null) b.write(' // $comment');
    }
    return b.toString();
  }

  (String, String?) _buildCall(BuildCall c, String indent) {
    switch (c) {
      case AddNodes():
        String ints(List<PlanEdge> e) => '[${e.map((x) => x.node).join(', ')}]';
        String strengths(List<PlanEdge> e) =>
            '[${e.map((x) => dartDouble(x.strength)).join(', ')}]';
        final args = <String>[
          if (c.kind != NodeKind.continuousState)
            'kind: NodeKind.${c.kind.name}',
          if (c.nNodes != 1) 'nNodes: ${c.nNodes}',
          for (final (name, e) in [
            ('valueParents', c.valueParents),
            ('valueChildren', c.valueChildren),
            ('volatilityParents', c.volatilityParents),
            ('volatilityChildren', c.volatilityChildren),
          ])
            if (e.isNotEmpty) ...[
              '$name: ${ints(e)}',
              if (e.any((x) => x.strength != 1.0))
                '${name}Strengths: ${strengths(e)}',
            ],
          if (c.customCoupling case final name?)
            'coupling: ${_couplingStub(name)}'
          else if (c.coupling != CouplingFn.linear)
            'coupling: CouplingFn.${c.coupling.name}',
          for (final MapEntry(:key, :value) in c.parameters.entries)
            if (_namedParameters.contains(key))
              '${key.name}: ${dartDouble(value)}',
          if (c.parameters.keys.any((f) => !_namedParameters.contains(f)))
            'parameters: {${[for (final MapEntry(:key, :value) in c.parameters.entries)
              if (!_namedParameters.contains(key)) 'NodeField.${key.name}: ${dartDouble(value)}'].join(', ')}}',
        ];
        final label = c.nNodes == 1
            ? '${c.first}'
            : '${c.first}–${c.first + c.nNodes - 1}';
        return (_call('addNodes', args, indent), label);
      case AddEfState():
        final args = [
          if (c.nNodes != 1) 'nNodes: ${c.nNodes}',
          if (c.dimension != 1) 'dimension: ${c.dimension}',
          if (c.learning != EfLearning.fixed)
            'learning: EfLearning.${c.learning.name}',
          if (c.nus != 3.0) 'nus: ${dartDouble(c.nus)}',
          if (c.xis case final xis?) 'xis: ${_doubles(xis)}',
          if (c.valueChildren case final vc?)
            'valueChildren: [${vc.join(', ')}]',
        ];
        final last = c.first + c.nNodes - 1;
        final label = c.nNodes == 1 ? '${c.first}' : '${c.first}–$last';
        final implied = c.impliedNodes == 0
            ? ''
            : ' (${last + 1}–${last + c.impliedNodes}: learning chains)';
        return (_call('addEfState', args, indent), '$label$implied');
      case AddCategoricalState():
        final args = [
          'nCategories: ${c.nCategories}',
          for (final MapEntry(:key, :value) in c.arguments.entries)
            '$key: ${dartDouble(value)}',
        ];
        return (
          _call('addCategoricalState', args, indent),
          '${c.node} (${c.node + 1}–${c.node + c.impliedNodes}: its binary '
              'HGFs)',
        );
      case AddDpState():
        final args = [
          if (c.batchSize != 10) 'batchSize: ${c.batchSize}',
          if (c.alpha != 1.0) 'alpha: ${dartDouble(c.alpha)}',
          if (c.sensoryPrecision != 1.0)
            'sensoryPrecision: ${dartDouble(c.sensoryPrecision)}',
        ];
        return (_call('addDpState', args, indent), '${c.node}');
      case SetInitial():
        return (
          'setInitial(${c.node}, NodeField.${c.field.name}, '
              '${dartDouble(c.value)})',
          null,
        );
      case SetCoupling():
        return (
          '${c.volatility ? 'setVolatilityCoupling' : 'setValueCoupling'}'
              '(${c.parent}, ${c.child}, ${dartDouble(c.strength)})',
          null,
        );
      case SetInitialVector():
        return (
          _call('setInitialVector', [
            '${c.node}',
            'VectorField.${c.field.name}',
            _doubles(c.values),
          ], indent),
          null,
        );
    }
  }

  static const _namedParameters = {
    NodeField.mean,
    NodeField.expectedMean,
    NodeField.precision,
    NodeField.expectedPrecision,
    NodeField.tonicVolatility,
    NodeField.tonicDrift,
    NodeField.autoconnectionStrength,
  };

  String _couplingStub(String name) {
    final id = '${dartIdentifier(name)}Coupling';
    if (_stubNames.add(id)) {
      _stubs.add(
        '// TODO: the custom coupling "$name" is code, which can\'t be '
        'exported:\n'
        '// write g, its derivatives g\' and g\'\' (and optionally '
        'g\'\'\').\n'
        'final $id = CustomCoupling(\n'
        "  '${_escape(name)}',\n"
        "  g: (x) => throw UnimplementedError('$name: g'),\n"
        "  dg: (x) => throw UnimplementedError('$name: g\\''),\n"
        "  d2g: (x) => throw UnimplementedError('$name: g\\'\\''),\n"
        ');\n',
      );
    }
    return id;
  }

  /// The model as an `HgfModel(...)` expression; [networkExpression] is
  /// its network (default: `buildNetwork()`).
  String model({
    String indent = '',
    String networkExpression = 'buildNetwork()',
  }) {
    final json = spec.model ?? (throw StateError('The spec has no model'));
    final inner = '$indent  ';
    final params = json['parameters'] as List<Object?>;
    final b = StringBuffer()
      ..writeln('HgfModel(')
      ..writeln('${inner}network: $networkExpression,')
      ..writeln(
        '${inner}response: '
        '${_response(json['response'] as Map<String, Object?>, inner)},',
      )
      ..writeln('${inner}parameters: [');
    for (final p in params) {
      b.writeln(
        '$inner  ${_parameter(p as Map<String, Object?>, '$inner  ')},',
      );
    }
    b.writeln('$inner],');
    final block = (json['block_size'] as num?)?.toInt() ?? 256;
    if (block != 256) b.writeln('${inner}blockSize: $block,');
    b.write('$indent)');
    return b.toString();
  }

  String _parameter(Map<String, Object?> p, String indent) {
    final name = "'${_escape(p['name'] as String)}'";
    final targets = [
      for (final t in p['targets'] as List<Object?>)
        FitCodec.decodeTarget(t as Map<String, Object?>),
    ];
    final priorJson = p['prior'] as Map<String, Object?>;
    final prior = _prior(priorJson);
    // The transform only when it isn't the prior's default.
    String? transform;
    final tJson = p['transform'] as Map<String, Object?>;
    try {
      final def = const FitCodec().decodePrior(priorJson).defaultTransform;
      if (!_sameJson(def.toJson(), tJson)) transform = _transform(tJson);
    } on Object {
      transform = _transform(tJson);
    }
    final rest = [
      'prior: $prior',
      if (transform != null) 'transform: $transform',
      if (p['initial'] case final x?)
        'initial: ${dartDouble(readJsonDouble(x))}',
    ];
    if (targets.length == 1) {
      switch (targets.single) {
        case NodeTarget(:final node, :final field):
          return _call('Parameter.node', [
            name,
            '$node',
            'NodeField.${field.name}',
            ...rest,
          ], indent);
        case ValueCouplingTarget(:final parent, :final child):
          return _call('Parameter.valueCoupling', [
            name,
            '$parent',
            '$child',
            ...rest,
          ], indent);
        case VolatilityCouplingTarget(:final parent, :final child):
          return _call('Parameter.volatilityCoupling', [
            name,
            '$parent',
            '$child',
            ...rest,
          ], indent);
        case ResponseTarget(name: final r):
          return _call('Parameter.response', [
            name,
            if (r != p['name']) "responseName: '${_escape(r)}'",
            ...rest,
          ], indent);
      }
    }
    return _call('Parameter', [
      name,
      'targets: ${_list(targets.map<String>(_target).toList(), '$indent  ')}',
      ...rest,
    ], indent);
  }

  static String _target(ParameterTarget t) => switch (t) {
    NodeTarget(:final node, :final field) =>
      'NodeTarget($node, NodeField.${field.name})',
    ValueCouplingTarget(:final parent, :final child) =>
      'ValueCouplingTarget($parent, $child)',
    VolatilityCouplingTarget(:final parent, :final child) =>
      'VolatilityCouplingTarget($parent, $child)',
    ResponseTarget(:final name) => "ResponseTarget('${_escape(name)}')",
  };

  String _prior(Map<String, Object?> j) {
    String d(String key) => dartDouble(readJsonDouble(j[key]));
    bool finite(String key) => readJsonDouble(j[key]).isFinite;
    return switch (j['type']) {
      'Normal' => 'const Normal(${d('mean')}, ${d('sd')})',
      'TruncatedNormal' =>
        'TruncatedNormal(${[d('mean'), d('sd'), if (finite('lower')) 'lower: ${d('lower')}', if (finite('upper')) 'upper: ${d('upper')}'].join(', ')})',
      'Uniform' => 'const Uniform(${d('lower')}, ${d('upper')})',
      'HalfNormal' => 'const HalfNormal(${d('sd')})',
      'LogNormal' => 'const LogNormal(${d('mu')}, ${d('sigma')})',
      'Gamma' => 'const GammaPrior(${d('shape')}, ${d('rate')})',
      'Flat' =>
        'const Flat(${[if (finite('lower')) 'lower: ${d('lower')}', if (finite('upper')) 'upper: ${d('upper')}', if (j['typical'] != null) 'typical: ${d('typical')}'].join(', ')})',
      final type => _stub('Prior', 'prior', type, j),
    };
  }

  String _transform(Map<String, Object?> j) {
    String d(String key) => dartDouble(readJsonDouble(j[key]));
    return switch (j['type']) {
      'Identity' => 'const IdentityTransform()',
      'LowerBound' => 'const LowerBoundTransform(${d('lower')})',
      'UpperBound' => 'const UpperBoundTransform(${d('upper')})',
      'Interval' => 'const IntervalTransform(${d('lower')}, ${d('upper')})',
      final type => _stub('Transform', 'transform', type, j),
    };
  }

  String _response(Map<String, Object?> j, String indent) {
    String d(String key) => dartDouble(readJsonDouble(j[key]));
    int i(String key) => (j[key] as num).toInt();
    String field() => NodeField.parse(j['field'] as String).name;
    switch (j['type']) {
      case 'GaussianReadout':
        return _call('const GaussianReadout', [
          if (i('node') != 0) 'node: ${i('node')}',
          if (field() != NodeField.expectedMean.name)
            'field: NodeField.${field()}',
          if (readJsonDouble(j['precision']) != 1.0)
            'precision: ${d('precision')}',
        ], indent);
      case 'BinarySoftmaxReadout' || 'CategoricalSoftmaxReadout':
        return _call('const ${j['type']}', [
          if (i('node') != 0) 'node: ${i('node')}',
          if (readJsonDouble(j['inverseTemperature']) != 1.0)
            'inverseTemperature: ${d('inverseTemperature')}',
        ], indent);
      case 'SoftmaxReadout':
        final options = [
          for (final o in j['options'] as List<Object?>)
            [
              for (final t in (o as List<Object?>).cast<Map<String, Object?>>())
                (
                  node: (t['node'] as num).toInt(),
                  weight: readJsonDouble(t['weight']),
                ),
            ],
        ];
        final rest = [
          if (field() != NodeField.expectedMean.name)
            'field: NodeField.${field()}',
          if (readJsonDouble(j['inverseTemperature']) != 1.0)
            'inverseTemperature: ${d('inverseTemperature')}',
        ];
        if (options.every((o) => o.length == 1 && o.single.weight == 1.0)) {
          return _call('SoftmaxReadout.nodes', [
            '[${options.map((o) => o.single.node).join(', ')}]',
            ...rest,
          ], indent);
        }
        return _call('const SoftmaxReadout', [
          'options: [${[for (final o in options) '{${o.map((t) => '${t.node}: ${dartDouble(t.weight)}').join(', ')}}'].join(', ')}]',
          ...rest,
        ], indent);
      case 'ObservationSurprise':
        final fn = j['function'] as String;
        const builtIn = {
          'FirstLevelGaussianSurprise',
          'TotalGaussianSurprise',
          'FirstLevelBinarySurprise',
          'CategoricalSurprise',
          'DpSurprise',
        };
        return builtIn.contains(fn)
            ? 'const ObservationSurprise($fn())'
            : 'ObservationSurprise(${_stub('ResponseFunction', 'response function', fn, j)})';
      case final type:
        return _stub('ResponseModel', 'response model', '$type', j);
    }
  }

  /// A function that throws, standing in for something only code can be.
  String _stub(
    String type,
    String what,
    Object? name,
    Map<String, Object?> json,
  ) {
    final n = dartIdentifier('$name');
    final id = 'custom$type${n[0].toUpperCase()}${n.substring(1)}';
    if (_stubNames.add(id)) {
      _stubs.add(
        '// TODO: write the custom $what "$name" (exported as JSON:\n'
        '// ${_jsonLine(json)}).\n'
        "$type $id() => throw UnimplementedError('$what \"${_escape('$name')}\"');\n",
      );
    }
    return '$id()';
  }

  /// The whole script.
  String script() {
    final plan = spec.plan;
    final net = network(indent: '    ', terminator: ';');
    final model = spec.model == null ? null : this.model();
    final data = spec.data;
    final width = plan.nInputColumns;
    final nTime = data?.nTime(width) ?? 0;
    final r = data?.responsesPerTrial(nTime) ?? _responsesPerTrial();
    final cols = ExportData.columns(
      plan,
      responsesPerTrial: r,
      observed: data?.observed != null,
      timeSteps: data?.timeSteps != null,
    );
    final b = StringBuffer();
    b.writeln(
      '// ${spec.title ?? (model == null ? 'An HGF network' : 'An HGF model')}, '
      'exported from hgf.dart.',
    );
    b
      ..writeln('//')
      ..writeln(
        '// Run it in a Dart package that depends on hgf'
        '${model == null ? '' : ' and hgf_fit'}',
      )
      ..writeln('// (https://github.com/NexusDynamic/hgf.dart):')
      ..writeln('//')
      ..writeln(
        spec.dataFile != null
            ? '//   dart run <this file> [data.csv]   '
                  '(default: ${spec.dataFile})'
            : data != null
            ? '//   dart run <this file> [data.csv]   '
                  '(default: the data below)'
            : '//   dart run <this file> data.csv',
      )
      ..writeln('//')
      ..writeln(
        '// The CSV has a header row and the columns ${_columnList(cols)}.',
      )
      ..writeln(
        '// The observed flags are optional: without them, an input with an '
        'empty',
      )
      ..writeln(
        "// cell wasn't observed on that trial. An empty response is a "
        'missed one.',
      );
    if (plan.customCouplings.isNotEmpty || _stubs.isNotEmpty) {
      b.writeln('//\n// Before running: fill in the stubs marked TODO.');
    }
    b
      ..writeln()
      ..writeln("import 'dart:io';");
    if (model != null && spec.fit != null) {
      b.writeln("import 'dart:math' as math;");
    }
    b
      ..writeln("import 'dart:typed_data';")
      ..writeln()
      ..writeln("import 'package:hgf/hgf.dart';");
    if (model != null) b.writeln("import 'package:hgf_fit/hgf_fit.dart';");
    b
      ..writeln()
      ..writeln(
        '/// The perceptual model: ${plan.nNodes} '
        '${plan.nNodes == 1 ? 'node' : 'nodes'}, '
        '${plan.inputs.length == 1 ? 'input node ${plan.inputs.single}' : 'input nodes ${plan.inputs.join(', ')}'}.',
      )
      ..writeln('Network buildNetwork() =>')
      ..writeln('    $net');
    if (model != null) {
      b
        ..writeln()
        ..writeln(
          '/// The model to fit: the network, how its beliefs explain the data, '
          'and',
        )
        ..writeln('/// the free parameters with their priors.')
        ..writeln('HgfModel buildModel() => $model;');
    }
    for (final s in _stubs) {
      b
        ..writeln()
        ..write(s);
    }
    b
      ..writeln()
      ..writeln(
        'typedef Data = ({\n'
        '  Float64List inputs,\n'
        '  Float64List? observed,\n'
        '  Float64List? timeSteps,\n'
        '  Float64List? responses,\n'
        '});',
      )
      ..writeln()
      ..writeln(
        "/// The CSV columns: each input node's (one per data column), its",
      )
      ..writeln(
        '/// observed flag (optional: without them, an empty cell is a '
        'missing',
      )
      ..writeln('/// observation), the responses.')
      ..writeln(
        'const inputColumns = <List<String>>'
        '${_list([for (final c in cols.inputs) '[${c.map(_str).join(', ')}]'], '', prefix: 32)};',
      )
      ..writeln(
        'const observedColumns = <String>'
        '${_list(cols.observed.map(_str).toList(), '', prefix: 32)};',
      )
      ..writeln(
        'const responseColumns = <String>[${cols.responses.map(_str).join(', ')}];',
      )
      ..writeln()
      ..write(_readCsv(cols.timeStep));
    if (data != null) {
      b
        ..writeln()
        ..writeln('/// The data the network was exported with.')
        ..writeln('Data embeddedData() => (')
        ..writeln('  inputs: ${_float64(data.inputs, '  ')},')
        ..writeln(
          '  observed: ${data.observed == null ? 'null' : _float64(data.observed!, '  ')},',
        )
        ..writeln(
          '  timeSteps: ${data.timeSteps == null ? 'null' : _float64(data.timeSteps!, '  ')},',
        )
        ..writeln(
          '  responses: ${data.responses == null ? 'null' : _float64(data.responses!, '  ')},',
        )
        ..writeln(');');
    }
    final load = spec.dataFile != null
        ? "readCsv(args.isEmpty ? '${_escape(spec.dataFile!)}' : args.first)"
        : data != null
        ? 'args.isEmpty ? embeddedData() : readCsv(args.first)'
        : 'readCsv(args.first)';
    b
      ..writeln()
      ..writeln('void main(List<String> args) {')
      ..writeln('  final data = $load;')
      ..writeln('  final network = buildNetwork();')
      ..writeln('  final traj = network.inputData(')
      ..writeln('    data.inputs,')
      ..writeln('    observed: data.observed,')
      ..writeln('    timeSteps: data.timeSteps,')
      ..writeln('  )!;')
      ..writeln(
        "  print('\${traj.nTime} trials; final beliefs (mean, precision):');",
      )
      ..writeln('  for (var node = 0; node < network.nNodes; node++) {')
      ..writeln('    final t = traj.nTime - 1;')
      ..writeln(
        "    print('  node \$node (\${network.kindOf(node).pyhgfName}): '",
      )
      ..writeln("        '\${traj.series(node, NodeField.mean)[t]}, '")
      ..writeln("        '\${traj.series(node, NodeField.precision)[t]}');")
      ..writeln('  }');
    if (spec.fit case final fit?) b.write(_fitCode(fit));
    b.writeln('}');
    return b.toString();
  }

  int _responsesPerTrial() {
    final type = (spec.model?['response'] as Map<String, Object?>?)?['type'];
    return type == null || type == 'ObservationSurprise' ? 0 : 1;
  }

  static String _str(String s) => "'${_escape(s)}'";

  static String _columnList(ExportColumns cols) {
    final all = cols.header;
    final shown = all.length <= 6 ? all : [...all.take(5), '…', all.last];
    return shown.map((c) => c == '…' ? c : '`$c`').join(', ');
  }

  String _fitCode(ExportFit fit) {
    final restarts = switch (fit) {
      ExportMap(:final restarts) => restarts,
      ExportNuts(:final restarts) => restarts,
    };
    final seed = fit is ExportMap ? fit.seed : 1;
    final b = StringBuffer()
      ..writeln()
      ..writeln('  final model = buildModel();')
      ..writeln('  final subject = Subject(')
      ..writeln('    inputs: data.inputs,')
      ..writeln('    observed: data.observed,')
      ..writeln('    timeSteps: data.timeSteps,')
      ..writeln('    responses: data.responses,')
      ..writeln('  );')
      ..writeln(
        '  // The mode: L-BFGS with exact gradients (Nelder–Mead where the',
      )
      ..writeln(
        '  // likelihood is only piecewise smooth), from the starting values '
        'and',
      )
      ..writeln('  // $restarts more starts drawn from the priors.')
      ..writeln('  final map = fitMap(')
      ..writeln('    model,')
      ..writeln('    [subject],')
      ..writeln(
        '    optimizer: model.hasExactGradients ? const Lbfgs() : '
        'const NelderMead(),',
      )
      ..writeln('    restarts: $restarts,')
      ..writeln('    random: math.Random($seed),')
      ..writeln('  );');
    switch (fit) {
      case ExportMap():
        b
          ..writeln(
            '  // Standard errors, 95 % intervals and the log evidence.',
          )
          ..writeln('  print(laplace(model, map).results.single);');
      case ExportNuts():
        b
          ..writeln('  final best = map.results.single;')
          ..writeln('  print(best);')
          ..writeln('  // The full posterior, started at the mode.')
          ..writeln('  final nuts = sampleNuts(')
          ..writeln('    model,')
          ..writeln('    [subject],')
          ..writeln(
            '    sampler: const Nuts(targetAccept: '
            '${dartDouble(fit.targetAccept)}, maxTreeDepth: ${fit.maxTreeDepth}),',
          )
          ..writeln('    chains: ${fit.chains},')
          ..writeln('    warmup: ${fit.warmup},')
          ..writeln('    draws: ${fit.draws},')
          ..writeln(
            '    start: [if (best.negLogPosterior.isFinite) best.x else '
            'model.start()],',
          )
          ..writeln('    jitter: 0.5,')
          ..writeln('    seed: ${fit.seed},')
          ..writeln('  );')
          ..writeln('  print(nuts.results.single);');
    }
    return b.toString();
  }

  /// [values] as a `Float64List` literal, ten to a line.
  static String _float64(List<double> values, String indent) {
    if (values.isEmpty) return 'Float64List(0)';
    final b = StringBuffer('Float64List.fromList(const [\n');
    for (var i = 0; i < values.length; i += 10) {
      b
        ..write('$indent  ')
        ..write(
          values
              .sublist(i, i + 10 > values.length ? values.length : i + 10)
              .map(dartDouble)
              .join(', '),
        )
        ..writeln(',');
    }
    b.write('$indent])');
    return b.toString();
  }

  /// `[items]` on one line if it fits in 80 columns (after [indent] and
  /// [prefix] more), else one item per line.
  static String _list(List<String> items, String indent, {int prefix = 20}) {
    final one = '[${items.join(', ')}]';
    if (indent.length + prefix + one.length <= 80) return one;
    return '[\n${items.map((i) => '$indent  $i,\n').join()}$indent]';
  }

  static String _doubles(List<double> v) => '[${v.map(dartDouble).join(', ')}]';

  /// `name(args)` on one line if it fits in 80 columns (from [indent]),
  /// else one argument per line.
  static String _call(String name, List<String> args, String indent) {
    final one = '$name(${args.join(', ')})';
    if (indent.length + one.length + 4 <= 80 && !one.contains('\n')) {
      return one;
    }
    final b = StringBuffer('$name(\n');
    for (final a in args) {
      b.writeln('$indent  $a,');
    }
    b.write('$indent)');
    return b.toString();
  }

  static String _readCsv(String? timeStepColumn) =>
      '''
/// Reads a CSV with the columns above (a header row, then one row per
/// trial; `#` lines are comments).
Data readCsv(String path) {
  final lines = [
    for (final l in File(path).readAsLinesSync())
      if (l.trim().isNotEmpty && !l.startsWith('#')) l,
  ];
  final header = [for (final h in lines.first.split(',')) h.trim()];
  int column(String name) {
    final i = header.indexOf(name);
    if (i < 0) throw FormatException('\$path has no column "\$name"');
    return i;
  }

  final rows = [for (final l in lines.skip(1)) l.split(',')];
  double cell(List<String> row, int c) {
    final s = c < row.length ? row[c].trim() : '';
    return switch (s) {
      '' || 'nan' || 'NaN' => double.nan,
      'inf' => double.infinity,
      '-inf' => double.negativeInfinity,
      _ => double.parse(s),
    };
  }

  final n = rows.length, nIn = inputColumns.length;
  final width = inputColumns.fold(0, (w, c) => w + c.length);
  final inCols = [
    for (final c in inputColumns) [for (final name in c) column(name)],
  ];
  final obsCols = [for (final name in observedColumns) header.indexOf(name)];
  final flags = obsCols.every((c) => c >= 0);
  final inputs = Float64List(n * width), observed = Float64List(n * nIn);
  var missing = false;
  for (var t = 0; t < n; t++) {
    var at = t * width;
    for (var k = 0; k < nIn; k++) {
      var seen = true;
      for (final c in inCols[k]) {
        final v = cell(rows[t], c);
        if (v.isNaN) seen = false;
        inputs[at++] = flags || !v.isNaN ? v : 0;
      }
      if (flags) seen = cell(rows[t], obsCols[k]) != 0;
      observed[t * nIn + k] = seen ? 1 : 0;
      missing |= !seen;
    }
  }
  final rCols = [for (final name in responseColumns) column(name)];
${timeStepColumn == null ? '' : "  final ts = column('$timeStepColumn');\n"}  return (
    inputs: inputs,
    observed: flags || missing ? observed : null,
    timeSteps: ${timeStepColumn == null ? 'null' : 'Float64List.fromList([for (final row in rows) cell(row, ts)])'},
    responses: rCols.isEmpty
        ? null
        : Float64List.fromList([
            for (final row in rows)
              for (final c in rCols) cell(row, c),
          ]),
  );
}
''';
}

/// [v] as a Dart expression that evaluates to the same double.
String dartDouble(double v) {
  if (v.isNaN) return 'double.nan';
  if (v == double.infinity) return 'double.infinity';
  if (v == double.negativeInfinity) return 'double.negativeInfinity';
  var s = v.toString().replaceFirst('e+', 'e');
  // 1e10 rather than 10000000000.0.
  if (v == v.roundToDouble() && v.abs() >= 1e6) {
    final e = v.toStringAsExponential().replaceFirst('e+', 'e');
    if (double.parse(e) == v && e.length < s.length) s = e;
  }
  if (!s.contains('.') && !s.contains('e') && !s.contains('N')) s = '$s.0';
  return s;
}

/// [name] as a lower-camel-case Dart identifier.
String dartIdentifier(String name) {
  final parts = name
      .split(RegExp('[^A-Za-z0-9]+'))
      .where((p) => p.isNotEmpty)
      .toList();
  if (parts.isEmpty) return 'unnamed';
  final s = [
    parts.first[0].toLowerCase() + parts.first.substring(1),
    for (final p in parts.skip(1)) p[0].toUpperCase() + p.substring(1),
  ].join();
  return RegExp('^[0-9]').hasMatch(s) ? 'x$s' : s;
}

String _escape(String s) => s
    .replaceAll(r'\', r'\\')
    .replaceAll("'", r"\'")
    .replaceAll(r'$', r'\$')
    .replaceAll('\n', r'\n');

String _jsonLine(Map<String, Object?> json) =>
    json.toString().replaceAll('\n', ' ');

bool _sameJson(Map<String, Object?> a, Map<String, Object?> b) =>
    a.length == b.length &&
    a.entries.every(
      (e) => b.containsKey(e.key) && '${b[e.key]}' == '${e.value}',
    );
