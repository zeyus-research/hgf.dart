part of 'dart.dart';

/// The script of a model document ([ExportSpec.layout]): every participant
/// of the CSV, read through `DataLayout`, fitted together (with the group
/// level), the estimates and outputs written next to the data.
extension _DocumentScript on DartExport {
  /// `DataLayout(...)`: [layout] as a constructor call (deep networks as
  /// `DocDeep.fromJson`).
  static String _layoutCode(DataLayout layout) {
    final b = StringBuffer()
      ..writeln('DataLayout(')
      ..writeln('  inputs: [');
    for (final i in layout.inputs) {
      b.writeln(
        '    ${DartExport._call('LayoutInput', [DartExport._str(i.name), 'columns: [${i.columns.map(DartExport._str).join(', ')}]', if (i.categories case final c?) 'categories: [${c.map(DartExport._str).join(', ')}]', if (i.oneHot) 'oneHot: true', if (i.observedColumn case final o?) 'observedColumn: ${DartExport._str(o)}'], '    ')},',
      );
    }
    b.writeln('  ],');
    if (layout.responseColumns.isNotEmpty) {
      b.writeln(
        '  responseColumns: [${layout.responseColumns.map(DartExport._str).join(', ')}],',
      );
    }
    if (layout.options case final o?) {
      b.writeln('  options: [${o.map(DartExport._str).join(', ')}],');
    }
    for (final (key, v) in [
      ('participantColumn', layout.participantColumn),
      ('timeColumn', layout.timeColumn),
      ('timeStepColumn', layout.timeStepColumn),
      ('groupColumn', layout.groupColumn),
    ]) {
      if (v != null) b.writeln('  $key: ${DartExport._str(v)},');
    }
    if (layout.clock case final k?) {
      String str(String v) => DartExport._str(v);
      b
        ..writeln('  clock: DocClock(')
        ..writeln('    onsetColumn: ${str(k.onsetColumn)},')
        ..writeln('    step: ${dartDouble(k.step)},');
      if (k.sessionColumn case final c?) {
        b.writeln('    sessionColumn: ${str(c)},');
      }
      b.writeln('    streams: [');
      for (final st in k.streams) {
        final hold = [
          for (final MapEntry(:key, :value) in st.hold.entries)
            '${str(key)}: ${str('$value')}',
        ].join(', ');
        b.writeln(
          '      ClockStream.parse(${st.when == null ? '' : 'when: ${str('${st.when}')}, '}hold: {$hold}),',
        );
      }
      b.writeln('    ],');
      if (k.keep case final e?) {
        b.writeln("    keep: Expr.parse(${str('$e')}),");
      }
      b.writeln('  ),');
    }
    if (layout.steps.isNotEmpty) {
      b.writeln('  steps: [');
      for (final st in layout.steps) {
        b.writeln('    ${_stepCode(st)},');
      }
      b.writeln('  ],');
    }
    if (layout.transforms.isNotEmpty) {
      b.writeln('  transforms: [');
      for (final t in layout.transforms) {
        b.writeln(
          '    ${DartExport._call('DocTransform', [DartExport._str(t.name), DartExport._str('${t.expression}')], '    ')},',
        );
      }
      b.writeln('  ],');
    }
    if (layout.deep.isNotEmpty) {
      b.writeln(
        '  // Deep networks trained after the transforms: their predictions',
      );
      b.writeln('  // are columns (hgf_fit\'s DeepTraining).');
      b.writeln('  deep: [');
      for (final d in layout.deep) {
        b.writeln(
          '    DocDeep.fromJson(${dartJsonLiteral(d.toJson(), '    ')}),',
        );
      }
      b.writeln('  ],');
    }
    b.write(')');
    return b.toString();
  }

  /// The names [_layoutCode]'s code takes from `package:hgf_fit/document.dart`.
  static List<String> _layoutNames(DataLayout layout) => [
    if (layout.clock != null) ...['ClockStream', 'DocClock'],
    'DataLayout',
    ...{for (final s in layout.steps) ..._stepNames(s)},
    if (layout.transforms.isNotEmpty) 'DocTransform',
    if (layout.deep.isNotEmpty) 'DocDeep',
    'LayoutInput',
  ];

  String _documentScript() {
    final layout = spec.layout!;
    final plan = spec.plan;
    final net = network(indent: '    ', terminator: ';');
    final modelJson = spec.model;
    final model = modelJson == null ? null : this.model();
    final fit = spec.fit;
    final group = spec.group;
    final grouped = group != null && !group.isEmpty;
    final outputs = spec.outputs;
    final params = [
      for (final p in (modelJson?['parameters'] as List<Object?>?) ?? const [])
        p as Map<String, Object?>,
    ];
    final fitWords = switch (fit) {
      null => null,
      ExportMap() when grouped =>
        'by empirical Bayes (the group level estimated with them)',
      ExportMap() => 'one by one (MAP, then Laplace intervals)',
      ExportNuts() when grouped =>
        'with the group level, sampling the joint posterior (NUTS)',
      ExportNuts() => 'one by one (NUTS, started at the MAP estimates)',
    };

    final b = StringBuffer()
      ..writeln(
        '// ${spec.title ?? (model == null ? 'An HGF network' : 'An HGF model')}, '
        'exported from hgf.dart.',
      )
      ..writeln('//')
      ..writeln(
        '// Run it in a Dart package that depends on hgf'
        '${model == null ? '' : ' and hgf_fit'}',
      )
      ..writeln('// (https://github.com/zeyus-research/hgf.dart):')
      ..writeln('//')
      ..writeln(
        spec.dataFile != null
            ? '//   dart run <this file> [data.csv]   '
                  '(default: ${spec.dataFile})'
            : spec.csv != null
            ? '//   dart run <this file> [data.csv]   '
                  '(default: the data below)'
            : '//   dart run <this file> data.csv',
      )
      ..writeln('//')
      ..writeln(
        '// It reads every participant of the CSV (one row per time step, '
        'the',
      )
      ..writeln("// columns in `layout` below)${fitWords == null ? '' : ','}");
    if (fitWords != null) {
      b
        ..writeln('// fits them $fitWords,')
        ..writeln('// and writes <data>_estimates.csv.');
    }
    if (outputs.isNotEmpty) {
      b.writeln(
        '// Then <data>_outputs.csv: the data with a column per named output, '
        'each',
      );
      b.writeln(
        fit == null
            ? '// participant filtered at the parameters set here.'
            : '// participant filtered at their estimates.',
      );
    }
    if (plan.customCouplings.isNotEmpty || _stubs.isNotEmpty) {
      b.writeln('//\n// Before running: fill in the stubs marked TODO.');
    }
    b
      ..writeln()
      ..writeln("import 'dart:io';");
    if (fit != null) b.writeln("import 'dart:math' as math;");
    b
      ..writeln()
      ..writeln("import 'package:hgf/hgf.dart';")
      ..writeln(
        "import 'package:hgf_fit/document.dart'\n"
        "    show ${_layoutNames(layout).join(', ')};",
      );
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
    if (grouped && fit != null) {
      b
        ..writeln()
        ..writeln('/// The group level: ${_groupWords(group)}.')
        ..writeln(_groupCode(group));
    }
    if (fit != null && outputs.isNotEmpty) {
      b
        ..writeln()
        ..writeln(
          '/// [buildNetwork] with the free parameters set to [x] (in '
          "[buildModel]'s",
        )
        ..writeln('/// order).')
        ..writeln('Network networkAt(List<double> x) => buildNetwork()');
      final sets = <String>[];
      for (final (i, p) in params.indexed) {
        for (final t in p['targets'] as List<Object?>) {
          switch (FitCodec.decodeTarget(t as Map<String, Object?>)) {
            case NodeTarget(:final node, :final field):
              sets.add('..setInitial($node, NodeField.${field.name}, x[$i])');
            case ValueCouplingTarget(:final parent, :final child):
              sets.add('..setValueCoupling($parent, $child, x[$i])');
            case VolatilityCouplingTarget(:final parent, :final child):
              sets.add('..setVolatilityCoupling($parent, $child, x[$i])');
            case ResponseTarget():
          }
        }
      }
      if (sets.isEmpty) {
        b.writeln(';');
      } else {
        for (final (i, s) in sets.indexed) {
          b.writeln('  $s${i == sets.length - 1 ? ';' : ''}');
        }
      }
    }
    for (final s in _stubs) {
      b
        ..writeln()
        ..write(s);
    }

    // The layout.
    b
      ..writeln()
      ..writeln(
        '/// How the CSV is read: per input node (in the network\'s input '
        'order)',
      )
      ..writeln(
        '/// its columns${layout.inputs.any((i) => i.categories != null) ? ' (or a column of category labels)' : ''}'
        '${layout.responseColumns.isEmpty ? '' : ', the responses'}'
        '${layout.participantColumn == null ? '' : ', a participant per subject'}'
        '${layout.timeColumn != null ? ', time steps from the time column' : ''}'
        '${layout.transforms.isEmpty ? '' : ', and columns computed first'}.',
      )
      ..writeln(
        '/// An empty cell is an unobserved input or a missed response.',
      )
      ..writeln('final layout = ${_layoutCode(layout)};');

    if (outputs.isNotEmpty) {
      final extra = {
        for (final o in outputs)
          if (o.field != 'surprise') NodeField.parse(o.field),
      }.where((f) => !defaultRecordedFields.contains(f)).toList();
      b
        ..writeln()
        ..writeln(
          "/// The fields the filter records: the defaults and the outputs'.",
        )
        ..writeln(
          extra.isEmpty
              ? 'const recordedFields = defaultRecordedFields;'
              : 'const recordedFields = [\n'
                    '  ...defaultRecordedFields,\n'
                    '${extra.map((f) => '  NodeField.${f.name},\n').join()}];',
        )
        ..writeln()
        ..writeln("/// The named outputs of one participant's beliefs.")
        ..writeln('Map<String, List<double>> outputs(Trajectories traj) => {');
      for (final o in outputs) {
        b.writeln(
          '  ${DartExport._str(o.name)}: '
          '${o.field == 'surprise' ? 'traj.nodeSurprise(${o.node})' : 'traj.series(${o.node}, NodeField.${NodeField.parse(o.field).name})'},',
        );
      }
      b.writeln('};');
    }

    if (spec.csv case final csv?) {
      b
        ..writeln()
        ..writeln('/// The data the model was exported with.')
        ..writeln("const embeddedCsv = '''")
        ..write(
          csv
              .replaceAll(r'\', r'\\')
              .replaceAll(r'$', r'\$')
              .replaceAll("'''", r"\'\'\'"),
        );
      if (!csv.endsWith('\n')) b.writeln();
      b.writeln("''';");
    }

    // main.
    final read = spec.dataFile != null
        ? "final path = args.isEmpty ? '${_escape(spec.dataFile!)}' : args.first;\n"
              '  final csv = CsvData.parse(File(path).readAsStringSync());'
        : spec.csv != null
        ? "final path = args.isEmpty ? 'data.csv' : args.first;\n"
              '  final csv = CsvData.parse(\n'
              '    args.isEmpty ? embeddedCsv : File(path).readAsStringSync(),\n'
              '  );'
        : 'final path = args.first;\n'
              '  final csv = CsvData.parse(File(path).readAsStringSync());';
    b
      ..writeln()
      ..writeln('void main(List<String> args) {')
      ..writeln('  $read')
      ..writeln(
        "  final base = path.replaceFirst(RegExp(r'\\.(csv|tsv|txt)\$'), '');",
      )
      ..writeln('  final participants = layout.subjects(csv);')
      ..writeln(
        "  final ids = [for (final (s, p) in participants.indexed) '\${p.id ?? s + 1}'];",
      )
      ..writeln(
        "  print('\${participants.length} participant(s), \${csv.rows.length} rows.');",
      );
    if (fit == null) {
      b
        ..writeln('  for (final (s, p) in participants.indexed) {')
        ..writeln('    final network = buildNetwork();')
        ..writeln('    final traj = network.inputData(')
        ..writeln('      p.inputs,')
        ..writeln('      observed: p.observed,')
        ..writeln('      timeSteps: p.timeSteps,')
        ..writeln('    )!;')
        ..writeln('    final t = traj.nTime - 1;')
        ..writeln("    print('\${ids[s]}: \${traj.nTime} rows; final means: '")
        ..writeln(
          "        '\${[for (var n = 0; n < network.nNodes; n++) "
          "traj.series(n, NodeField.mean)[t]].join(', ')}');",
        )
        ..writeln('  }');
    } else {
      b.write(_documentFit(fit, grouped));
    }
    if (outputs.isNotEmpty) {
      final at = fit == null
          ? 'buildNetwork()'
          : 'networkAt(estimates.estimate[s])';
      b
        ..writeln()
        ..writeln(
          '  // The outputs per row, each participant filtered at '
          '${fit == null ? 'the set parameters' : 'their estimates'}.',
        )
        ..writeln('  final withOutputs = layout.withColumns(csv, [')
        ..writeln(
          fit == null
              ? '    for (final p in participants)'
              : '    for (final (s, p) in participants.indexed)',
        )
        ..writeln('      outputs(')
        ..writeln('        $at.inputData(')
        ..writeln('          p.inputs,')
        ..writeln('          observed: p.observed,')
        ..writeln('          timeSteps: p.timeSteps,')
        ..writeln('          fields: recordedFields,')
        ..writeln('        )!,')
        ..writeln('      ),')
        ..writeln('  ]);')
        ..writeln(
          "  File('\${base}_outputs.csv').writeAsStringSync('\${withOutputs.toText()}\\n');",
        )
        ..writeln("  print('Wrote \${base}_outputs.csv');");
    }
    b.writeln('}');
    return b.toString();
  }

  String _documentFit(ExportFit fit, bool grouped) {
    final restarts = switch (fit) {
      ExportMap(:final restarts) => restarts,
      ExportNuts(:final restarts) => restarts,
    };
    final seed = fit is ExportMap ? fit.seed : 1;
    final b = StringBuffer()
      ..writeln()
      ..writeln('  final model = buildModel();')
      ..writeln(
        "  // Each participant's mode: L-BFGS with exact gradients "
        '(Nelder–Mead',
      )
      ..writeln(
        '  // where the likelihood is only piecewise smooth), from the '
        'starting',
      )
      ..writeln(
        '  // values and $restarts more starts drawn from the priors. '
        '${fit is ExportMap && !grouped ? '' : 'Every method starts there.'}',
      )
      ..writeln('  final map = fitMap(')
      ..writeln('    model,')
      ..writeln('    participants,')
      ..writeln(
        '    optimizer: model.hasExactGradients ? const Lbfgs() : '
        'const NelderMead(),',
      )
      ..writeln('    restarts: $restarts,')
      ..writeln('    random: math.Random($seed),')
      ..writeln('  );');
    if (fit is! ExportMap || grouped) {
      b
        ..writeln('  final starts = [')
        ..writeln('    for (final r in map.results)')
        ..writeln('      r.negLogPosterior.isFinite ? r.x : model.start(),')
        ..writeln('  ];');
    }
    switch (fit) {
      case ExportMap() when !grouped:
        b
          ..writeln('  // Standard errors and 95 % intervals (Laplace).')
          ..writeln('  final estimates = ParticipantEstimates.fromLaplace(')
          ..writeln('    laplace(model, map),')
          ..writeln('    participants: ids,')
          ..writeln('  );');
      case ExportMap():
        b
          ..writeln(
            '  // The populations by expectation-maximisation with the Laplace',
          )
          ..writeln(
            "  // approximation; each participant's mode under them "
            '(shrinkage).',
          )
          ..writeln(
            '  final eb = fitEmpiricalBayes(buildGroup(model), participants, '
            'start: starts);',
          )
          ..writeln('  print(eb);')
          ..writeln(
            '  final estimates = ParticipantEstimates.fromEmpiricalBayes(',
          )
          ..writeln('    eb,')
          ..writeln('    participants: ids,')
          ..writeln('  );');
      case ExportNuts() when !grouped:
        b
          ..writeln('  final nuts = sampleNuts(')
          ..writeln('    model,')
          ..writeln('    participants,')
          ..writeln(
            '    sampler: const Nuts(targetAccept: '
            '${dartDouble(fit.targetAccept)}, maxTreeDepth: ${fit.maxTreeDepth}),',
          )
          ..writeln('    chains: ${fit.chains},')
          ..writeln('    warmup: ${fit.warmup},')
          ..writeln('    draws: ${fit.draws},')
          ..writeln('    start: starts,')
          ..writeln('    jitter: 0.5,')
          ..writeln('    seed: ${fit.seed},')
          ..writeln('  );')
          ..writeln(
            "  print('\${nuts.divergences} divergent transitions; converged "
            "(R̂, ESS): '",
          )
          ..writeln("      '\${nuts.results.every((d) => d.converged())}');")
          ..writeln(
            '  final estimates = ParticipantEstimates.fromNuts(nuts, '
            'participants: ids);',
          );
      case ExportNuts():
        b
          ..writeln('  final group = buildGroup(model);')
          ..writeln('  final fit = sampleHierarchical(')
          ..writeln('    group,')
          ..writeln('    participants,')
          ..writeln(
            '    sampler: const Nuts(targetAccept: '
            '${dartDouble(fit.targetAccept)}, maxTreeDepth: ${fit.maxTreeDepth}),',
          )
          ..writeln('    chains: ${fit.chains},')
          ..writeln('    warmup: ${fit.warmup},')
          ..writeln('    draws: ${fit.draws},')
          ..writeln('    start: starts,')
          ..writeln('    jitter: 0.5,')
          ..writeln('    seed: ${fit.seed},')
          ..writeln('  );')
          ..writeln(
            "  print('\${fit.divergences} divergent transitions; converged "
            "(R̂, ESS): '",
          )
          ..writeln("      '\${fit.converged()}');")
          ..writeln(
            '  final estimates = ParticipantEstimates.fromHierarchical(',
          )
          ..writeln('    group,')
          ..writeln('    fit,')
          ..writeln('    participants,')
          ..writeln('    participants: ids,')
          ..writeln('  );');
    }
    b
      ..writeln('  print(estimates);')
      ..writeln(
        "  File('\${base}_estimates.csv').writeAsStringSync('\${estimates.toCsv()}\\n');",
      )
      ..writeln("  print('Wrote \${base}_estimates.csv');");
    return b.toString();
  }

  String _groupWords(DocGroup group) => [
    if (group.pooled.isNotEmpty)
      '${group.pooled.keys.join(', ')} vary by participant around a '
          'population (mean and sd estimated too, on the unconstrained '
          'scale)',
    if (group.shared.isNotEmpty)
      '${group.shared.join(', ')} ${group.shared.length == 1 ? 'is' : 'are'} '
          'shared by everyone',
  ].join('; ');

  String _groupCode(DocGroup group) {
    final b = StringBuffer()
      ..writeln(
        'HierarchicalModel buildGroup(HgfModel model) => HierarchicalModel(',
      )
      ..writeln('  model,');
    if (group.pooled.isNotEmpty) {
      b.writeln('  pooled: {');
      for (final MapEntry(:key, :value) in group.pooled.entries) {
        final mean = _prior(value.mean), sd = _prior(value.sd);
        final allConst = mean.startsWith('const ') && sd.startsWith('const ');
        String bare(String p) => allConst ? p.substring(6) : p;
        b.writeln(
          '    ${DartExport._str(key)}: ${DartExport._call('${allConst ? 'const ' : ''}Population', ['mean: ${bare(mean)}', 'sd: ${bare(sd)}'], '    ')},',
        );
      }
      b.writeln('  },');
    }
    if (group.shared.isNotEmpty) {
      b.writeln('  shared: [${group.shared.map(DartExport._str).join(', ')}],');
    }
    if (group.centered) b.writeln('  centered: true,');
    b.write(');');
    return b.toString();
  }
}

/// The `package:hgf_fit/document.dart` names [_stepCode] uses for [s].
List<String> _stepNames(DocStep s) => switch (s) {
  DecimateStep() => ['DecimateStep', 'StepAggregate'],
  SegmentStep() => ['SegmentStep'],
  FilterStep() => ['FilterStep'],
  StandardiseStep() => ['StandardiseStep'],
  SplitStep() => ['SplitStep'],
};

/// [s] as a Dart constructor call.
String _stepCode(DocStep s) {
  String str(String v) => DartExport._str(v);
  String expr(Expr e) => 'Expr.parse(${str('$e')})';
  String strings(List<String> xs) => '[${xs.map(str).join(', ')}]';
  final args = switch (s) {
    DecimateStep() => [
      'factor: ${s.factor}',
      if (s.how != StepAggregate.first) 'how: StepAggregate.${s.how.name}',
      if (s.columns.isNotEmpty)
        'columns: {${[for (final MapEntry(:key, :value) in s.columns.entries) '${str(key)}: StepAggregate.${value.name}'].join(', ')}}',
      if (s.within.isNotEmpty) 'within: ${strings(s.within)}',
    ],
    SegmentStep() => [
      'at: ${expr(s.at)}',
      if (s.before != 0) 'before: ${s.before}',
      if (s.after case final a?) 'after: $a',
      if (s.column != 'segment') 'column: ${str(s.column)}',
      if (s.offsetColumn case final o?) 'offsetColumn: ${str(o)}',
    ],
    FilterStep() => [expr(s.keep)],
    StandardiseStep() => [
      'columns: ${strings(s.columns)}',
      if (!s.perParticipant) 'perParticipant: false',
      if (s.fitOn case final f?) 'fitOn: ${expr(f)}',
    ],
    SplitStep() => [
      if (s.column != 'split') 'column: ${str(s.column)}',
      'parts: [${[for (final p in s.parts) '(label: ${str(p.label)}, fraction: ${dartDouble(p.fraction)})'].join(', ')}]',
      if (s.byParticipant) 'byParticipant: true',
      if (s.seed != 0) 'seed: ${s.seed}',
    ],
  };
  final name = switch (s) {
    DecimateStep() => 'DecimateStep',
    SegmentStep() => 'SegmentStep',
    FilterStep() => 'FilterStep',
    StandardiseStep() => 'StandardiseStep',
    SplitStep() => 'SplitStep',
  };
  return DartExport._call(name, args, '    ');
}

/// `DataLayout(...)` for [layout]: what a script reads its CSV through.
String dartDataLayout(DataLayout layout) => _DocumentScript._layoutCode(layout);

/// The names [dartDataLayout]'s code imports from
/// `package:hgf_fit/document.dart`.
List<String> dartDataLayoutNames(DataLayout layout) =>
    _DocumentScript._layoutNames(layout);

/// [json] (maps, lists, strings, numbers, booleans, null) as a Dart
/// literal, indented by [indent] after its first line.
String dartJsonLiteral(Object? json, [String indent = '']) {
  String lit(Object? v, String ind) => switch (v) {
    null => 'null',
    bool() || int() => '$v',
    double() => dartDouble(v),
    String() => DartExport._str(v),
    List() when v.isEmpty => '[]',
    List() =>
      '[\n${[for (final x in v) '$ind  ${lit(x, '$ind  ')},'].join('\n')}\n$ind]',
    Map() when v.isEmpty => '<String, Object?>{}',
    Map() =>
      '<String, Object?>{\n${[for (final MapEntry(:key, :value) in v.entries) '$ind  ${DartExport._str('$key')}: ${lit(value, '$ind  ')},'].join('\n')}\n$ind}',
    _ => throw ArgumentError('Not JSON: $v'),
  };
  return lit(json, indent);
}
