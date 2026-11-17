import 'package:hgf/hgf.dart';
import 'package:hgf_deep/hgf_deep.dart' show LayerKind, LearningKind;

import '../document/data_layout.dart';
import '../document/deep_document.dart';
import '../document/deep_training.dart' show deepPredictionColumns;
import 'dart.dart' show dartDataLayout, dartDataLayoutNames, dartDouble;
import 'expr_code.dart' show dartString;
import 'python.dart'
    show pyCoupling, pyDataExpr, pyDataPipeline, pyFloat, pyIdentifier;

/// The data pipeline of [layout] in words, or `null` when there is none.
String? _pipelineWords(DataLayout? layout) {
  if (layout == null) return null;
  final parts = [
    if (layout.clock != null) 'puts the events onto the shared clock',
    if (layout.steps.isNotEmpty)
      'runs the data steps (${[for (final s in layout.steps) s.kind].join(', ')})',
    if (layout.transforms.isNotEmpty)
      'computes the transforms (${[for (final t in layout.transforms) t.name].join(', ')})',
    if (layout.deep.isNotEmpty)
      'trains the deep network${layout.deep.length == 1 ? '' : 's'} it reads '
          '(${[for (final d in layout.deep) d.label].join(', ')})',
  ];
  if (parts.isEmpty) return null;
  return parts.length == 1
      ? parts.single
      : '${parts.sublist(0, parts.length - 1).join(', ')} and ${parts.last}';
}

/// What a deep network's script reads and writes, in words (both targets).
String _summary(DocDeep d, DataLayout? layout, String comment) {
  final t = d.training;
  final lines = <String>[];
  if (_pipelineWords(layout) case final w?) {
    lines.add('It reads the CSV as the model document does: it $w.');
  }
  final per = layout?.participantColumn == null
      ? ''
      : ', each participant (${layout!.participantColumn}) on its own';
  if (d.isContinuous) {
    lines.add(
      'It reads the observation column${d.targets.length == 1 ? '' : 's'} '
      '${d.targets.join(', ')} (rows with a missing value are skipped) and '
      'filters them through the network$per (a continuous network does not '
      'learn: its couplings are parameters).',
    );
  } else {
    final how = t.learningKind == LearningKind.synapticUncertainty
        ? 'synaptic-uncertainty learning, window ${t.synapticWindow}'
        : '${t.optimiser == 'sgd' ? 'SGD' : 'Adam'}, learning rate '
              '${t.learningRate}';
    final rows = t.batchSize == null
        ? 'one row at a time'
        : 'minibatches of ${t.batchSize}';
    lines.add(
      'It reads the feature column${d.features.length == 1 ? '' : 's'} '
      '${d.features.join(', ')} and the target '
      '${d.categories == null ? 'column${d.targets.length == 1 ? '' : 's'} ${d.targets.join(', ')}' : 'labels in ${d.targets.single} (${d.categories!.join(', ')})'} '
      '(rows with a missing value are skipped)'
      '${t.trainOn == null ? '' : ', trains on the rows where ${t.trainOn} (the others are held out)'},'
      '${t.standardise ? ' standardises the features with the training rows\' means and sds,' : ''} '
      'trains for ${t.epochs} pass${t.epochs == 1 ? '' : 'es'} ($how, $rows'
      '${t.shuffle ? ', shuffled' : ', in order'}) and prints the fit.',
    );
  }
  lines.add(
    'It writes <data>_${d.id}.csv: the table it read with a prediction column '
    'per target node'
    '${d.isContinuous ? ' (each made before its row was seen)' : ''}.',
  );
  return _wrap(lines.join(' '), comment);
}

String _wrap(String text, String prefix, {int width = 76}) {
  final out = StringBuffer();
  var line = prefix;
  for (final w in text.split(' ')) {
    if (line.length + w.length + 1 > width && line != prefix) {
      out.writeln(line.trimRight());
      line = prefix;
    }
    line += '$w ';
  }
  out.writeln(line.trimRight());
  return out.toString();
}

/// [layout]'s data pipeline without inputs (what a deep network's script
/// reads through); a plain one when there is none.
DataLayout _pipelineOf(DataLayout? layout) => DataLayout(
  inputs: const [],
  participantColumn: layout?.participantColumn,
  timeColumn: layout?.timeColumn,
  timeStepColumn: layout?.timeStepColumn,
  groupColumn: layout?.groupColumn,
  clock: layout?.clock,
  steps: layout?.steps ?? const [],
  transforms: layout?.transforms ?? const [],
  deep: layout?.deep ?? const [],
);

/// A Dart script for one deep network of a model document: it reads the
/// CSV through the document's data pipeline ([layout]: the clock, the
/// steps, the transforms and the deep networks this one reads, with
/// hgf_fit's `DataLayout`), builds the network, trains it (or filters,
/// for a continuous network) as [DocDeep.training] says, and writes the
/// predictions next to the table. It gives what hgf_fit's `DeepTraining`
/// gives (and the builder shows), to the last digit.
final class DeepDartExport {
  DeepDartExport(this.deep, {this.title, this.dataFile, this.layout});

  final DocDeep deep;
  final String? title;

  /// The CSV read when none is given on the command line.
  final String? dataFile;

  /// The data pipeline (see `ModelDocument.deepPipeline`); `null`: the CSV
  /// as it is.
  final DataLayout? layout;

  String _coupling(String text) => switch (parseCoupling(text)) {
    final CouplingFn f => 'CouplingFn.${f.name}',
    final ExprCoupling e => 'ExprCoupling(${dartString(e.source)})',
    _ => throw UnsupportedError('Coupling "$text"'),
  };

  bool get _usesExpr => [
    deep.settings.coupling,
    for (final l in deep.layers) ?l.coupling,
  ].any((c) => parseCoupling(c) is ExprCoupling);

  /// `buildNetwork()`'s body: the constructor and one `addLayer` per layer.
  String network() {
    final s = deep.settings, d = const DocDeepSettings();
    final args = [
      if (s.coupling != d.coupling) 'coupling: ${_coupling(s.coupling)}',
      if (s.volatilityUpdates != d.volatilityUpdates)
        'volatilityUpdates: DeepVolatilityUpdate.${s.volatilityUpdates.name}',
      if (s.maxPosteriorPrecision != d.maxPosteriorPrecision)
        'maxPosteriorPrecision: ${dartDouble(s.maxPosteriorPrecision)}',
      if (s.precisionClippingValue != d.precisionClippingValue)
        'precisionClippingValue: ${dartDouble(s.precisionClippingValue)}',
      if (s.predictPrecision != d.predictPrecision)
        'predictPrecision: ${s.predictPrecision}',
      if (s.feedforwardUncertainty != d.feedforwardUncertainty)
        'feedforwardUncertainty: ${s.feedforwardUncertainty}',
      if (s.tonicVolatility != d.tonicVolatility)
        'tonicVolatility: ${s.tonicVolatility}',
      if (s.meanFieldUpdates != d.meanFieldUpdates)
        'meanFieldUpdates: ${s.meanFieldUpdates}',
      if (s.carryPrecision != d.carryPrecision)
        'carryPrecision: ${s.carryPrecision}',
      if (s.updateInputLayer != d.updateInputLayer)
        'updateInputLayer: ${s.updateInputLayer}',
      if (!s.fixes.curvatureFloor && !s.fixes.logSpaceVolatility)
        'fixes: DeepFixes.none'
      else if (s.fixes.curvatureFloor != d.fixes.curvatureFloor ||
          s.fixes.logSpaceVolatility != d.fixes.logSpaceVolatility)
        'fixes: const DeepFixes(curvatureFloor: ${s.fixes.curvatureFloor}, '
            'logSpaceVolatility: ${s.fixes.logSpaceVolatility})',
    ];
    final b = StringBuffer(
      args.isEmpty
          ? 'DeepNetwork()'
          : 'DeepNetwork(\n${[for (final a in args) '  $a,'].join('\n')}\n)',
    );
    for (final (i, l) in deep.layers.indexed) {
      final a = [
        '${l.size}',
        if (l.kind != LayerKind.volatile) 'kind: LayerKind.${l.kind.name}',
        if (l.addConstantInput case final v?) 'addConstantInput: $v',
        if (!l.fullyConnected) 'fullyConnected: false',
        if (l.coupling case final c?) 'coupling: ${_coupling(c)}',
        if (l.volatilityParent case final v?) 'volatilityParent: $v',
        if (l.tonicVolatility case final v?)
          'tonicVolatility: ${dartDouble(v)}',
        if (l.tonicVolatilityVol case final v?)
          'tonicVolatilityVol: ${dartDouble(v)}',
        if (l.initial.isNotEmpty)
          'initial: {${[for (final MapEntry(:key, :value) in l.initial.entries) 'LayerField.${key.name}: ${dartDouble(value)}'].join(', ')}}',
        if (l.valueChildren case final v?) 'valueChildren: $v',
        if (l.volatilityChildren case final v?) 'volatilityChildren: $v',
        if (l.valueCoupling case final v?) 'valueCoupling: ${dartDouble(v)}',
        if (l.volatilityCoupling case final v?)
          'volatilityCoupling: ${dartDouble(v)}',
        if (l.volatilityFullyConnected case final v?)
          'volatilityFullyConnected: $v',
        if (l.tonicDrift case final v?) 'tonicDrift: ${dartDouble(v)}',
        if (l.autoconnectionStrength case final v?)
          'autoconnectionStrength: ${dartDouble(v)}',
      ];
      b.write('\n  // ${deep.layerName(i)}\n  ..addLayer(${a.join(', ')})');
    }
    return b.toString();
  }

  String script() {
    final d = deep, t = d.training;
    final cont = d.isContinuous;
    final out = deepPredictionColumns(d);
    final pipeline = _pipelineOf(layout);
    String list(List<String> v) => '[${v.map(dartString).join(', ')}]';
    final names = {
      ...dartDataLayoutNames(pipeline),
      'DeepData',
    }.where((n) => n != 'LayoutInput').toList()..sort();
    final b = StringBuffer()
      ..writeln(
        '// ${title ?? 'Deep network "${d.label}"'}, exported from hgf.dart.',
      )
      ..writeln('//')
      ..writeln(
        '// Run it in a Dart package that depends on hgf, hgf_deep and hgf_fit',
      )
      ..writeln('// (https://github.com/NexusDynamic/hgf.dart):')
      ..writeln('//')
      ..writeln(
        dataFile == null
            ? '//   dart run <this file> data.csv'
            : '//   dart run <this file> [data.csv]   (default: $dataFile)',
      )
      ..writeln('//')
      ..write(_summary(d, layout, '// '))
      ..writeln('//')
      ..writeln(
        '// The network is buildNetwork(); the CSV is read through '
        '`pipeline`.',
      )
      ..writeln()
      ..write(cont ? '' : "import 'dart:convert';\n")
      ..writeln("import 'dart:io';")
      ..writeln()
      ..writeln(
        "import 'package:hgf/hgf.dart' show CsvData, Expr"
        "${_usesExpr ? ', ExprCoupling' : ''};",
      )
      ..write(cont ? '' : "import 'package:hgf_deep/datasets.dart';\n")
      ..writeln("import 'package:hgf_deep/hgf_deep.dart';")
      ..writeln(
        "import 'package:hgf_fit/document.dart'\n"
        "    show ${names.join(', ')};",
      )
      ..writeln()
      ..writeln('const networkId = ${dartString(d.id)};')
      ..writeln('const features = <String>${list(d.features)};')
      ..writeln('const targets = <String>${list(d.targets)};')
      ..writeln(
        d.categories == null
            ? 'const List<String>? categories = null;'
            : 'const categories = <String>${list(d.categories!)};',
      )
      ..writeln('const outputColumns = <String>${list(out)};')
      ..writeln()
      ..writeln(
        '/// The rows trained on (the others are held out and scored '
        'apart); null:',
      )
      ..writeln('/// every row.')
      ..writeln(
        t.trainOn == null || cont
            ? 'const Expr? trainOn = null;'
            : 'final trainOn = Expr.parse(${dartString('${t.trainOn}')});',
      )
      ..writeln()
      ..writeln(
        '/// How the CSV is prepared before the network reads it, as the '
        'model',
      )
      ..writeln('/// document prepares it.')
      ..writeln('final pipeline = ${dartDataLayout(pipeline)};')
      ..writeln()
      ..writeln(
        '/// The network, built from the target layer (layer 0) up to the',
      )
      ..writeln(
        "/// ${cont ? 'top level' : 'feature layer'}, as pyhgf builds it.",
      )
      ..writeln('DeepNetwork buildNetwork() => ${network()};')
      ..writeln()
      ..writeln('void main(List<String> args) {')
      ..writeln(
        dataFile == null
            ? "  if (args.isEmpty) {\n    stderr.writeln('usage: dart run <this file> data.csv');\n    exit(64);\n  }\n  final path = args.first;"
            : '  final path = args.isEmpty ? ${dartString(dataFile!)} : args.first;',
      )
      ..writeln('  final csv = CsvData.parse(File(path).readAsStringSync());')
      ..writeln('  final table = pipeline.transform(csv);')
      ..writeln('  final data = DeepData.columns(')
      ..writeln('    table,')
      ..writeln('    features: features,')
      ..writeln('    targets: targets,')
      ..writeln('    categories: categories,')
      ..writeln('    trainOn: trainOn,')
      ..writeln('    scope: pipeline.scope,')
      ..writeln('  );');
    if (cont) {
      b
        ..writeln(
          '  // Each participant on its own, with a fresh network, in order.',
        )
        ..writeln('  final p = Matrix(data.length, data.y.cols);')
        ..writeln('  final byParticipant = <String, List<int>>{};')
        ..writeln('  for (var i = 0; i < data.length; i++) {')
        ..writeln('    (byParticipant[data.participants[i]] ??= []).add(i);')
        ..writeln('  }')
        ..writeln('  for (final rows in byParticipant.values) {')
        ..writeln('    final q = buildNetwork().inputData(pick(data.y, rows));')
        ..writeln('    for (final (k, i) in rows.indexed) {')
        ..writeln('      for (var j = 0; j < p.cols; j++) {')
        ..writeln('        p.data[i * p.cols + j] = q.at(k, j);')
        ..writeln('      }')
        ..writeln('    }')
        ..writeln('  }');
    } else {
      final opt = t.learningKind == LearningKind.synapticUncertainty
          ? 'null'
          : t.optimiser == 'sgd'
          ? 'Sgd(${dartDouble(t.learningRate)})'
          : 'Adam(${dartDouble(t.learningRate)})';
      final learn = [
        'optimiser: optimiser',
        if (t.learningKind != LearningKind.precisionWeighted)
          'learningKind: LearningKind.${t.learningKind.name}',
        if (t.learningKind == LearningKind.synapticUncertainty)
          'synapticUncertainty: SynapticUncertainty(window: '
              '${dartDouble(t.synapticWindow)})',
      ].join(', ');
      b
        ..writeln(
          '  final trainRows = data.trainIndices, heldOut = data.testIndices;',
        )
        ..writeln('  if (trainRows.isEmpty) {')
        ..writeln("    stderr.writeln('No rows to train on');")
        ..writeln('    exit(65);')
        ..writeln('  }');
      if (t.standardise) {
        b
          ..writeln("  // Standardised with the training rows' means and sds.")
          ..writeln(
            '  final scale = Standardiser.fit(pick(data.x, trainRows));',
          )
          ..writeln('  final x = scale.apply(data.x);');
      } else {
        b.writeln('  final x = data.x;');
      }
      b
        ..writeln('  final net = buildNetwork();')
        ..writeln('  final train = Dataset(')
        ..writeln("    'train',")
        ..writeln('    pick(x, trainRows),')
        ..writeln('    pick(data.y, trainRows),')
        ..writeln('  );');
      if (t.init.strategy != null) {
        b.writeln(
          '  net.weightInitialisation(const ${t.init.strategy.runtimeType}(), '
          'seed: ${t.seed});',
        );
      }
      b
        ..writeln('  final optimiser = $opt;')
        ..writeln('  for (var epoch = 1; epoch <= ${t.epochs}; epoch++) {');
      if (t.batchSize case final n?) {
        b
          ..writeln(
            '    for (final batch in minibatches(train, $n, '
            'shuffle: ${t.shuffle}, seed: epoch)) {',
          )
          ..writeln('      net.batchUpdate(batch.x, batch.y, $learn);')
          ..writeln('    }');
      } else {
        b
          ..writeln(
            t.shuffle
                ? '    final rows = shuffled(train, seed: epoch);'
                : '    final rows = train;',
          )
          ..writeln('    net.fit(rows.x, rows.y, $learn);');
      }
      final regression = d.targetLayer!.kind == LayerKind.volatile;
      b
        ..writeln('  }')
        ..writeln('  final p = net.predict(x);')
        ..writeln(
          "  for (final (name, rows) in [('training', trainRows), "
          "('held-out', heldOut)]) {",
        )
        ..writeln('    if (rows.isEmpty) continue;')
        ..writeln(
          regression
              ? '    final score = r2(pick(p, rows), pick(data.y, rows));'
              : '    final score = accuracy(pick(p, rows), labelsOf(pick(data.y, rows)));',
        )
        ..writeln(
          "    print('${regression ? 'R²' : 'accuracy'} on the \$name rows: \${score.toStringAsFixed(4)}');",
        )
        ..writeln('  }')
        ..writeln(
          "  final saved = File('\${path.replaceFirst(RegExp(r'\\.[^./\\\\]*\$'), '')}_\$networkId.json')",
        )
        ..writeln('    ..writeAsStringSync(')
        ..writeln('      jsonEncode({')
        ..writeln("        'network': net.toJson(),")
        ..writeln(
          t.standardise
              ? "        'scale': scale.toJson(),"
              : "        'scale': null,",
        )
        ..writeln("        'features': features,")
        ..writeln("        'targets': targets,")
        ..writeln("        'categories': categories,")
        ..writeln('      }),')
        ..writeln('    );')
        ..writeln("  print('saved the trained network to \${saved.path}');");
    }
    b
      ..writeln(
        '  final written = writePredictions(table, data.rows, p, path);',
      )
      ..writeln("  print('wrote \${written.path}');")
      ..writeln('}')
      ..writeln()
      ..writeln(_pick)
      ..writeln()
      ..writeln(_writePredictions);
    if (!cont && d.targetLayer!.kind != LayerKind.volatile) {
      b
        ..writeln()
        ..writeln(
          '/// The class of each target row (one 0/1 column, or one-hot).',
        )
        ..writeln('List<int> labelsOf(Matrix y) => [')
        ..writeln('  for (var i = 0; i < y.rows; i++)')
        ..writeln('    if (y.cols == 1)')
        ..writeln('      y.at(i, 0) > 0.5 ? 1 : 0')
        ..writeln('    else')
        ..writeln(
          '      [for (var j = 0; j < y.cols; j++) j].reduce((a, c) => y.at(i, c) > y.at(i, a) ? c : a),',
        )
        ..writeln('];');
    }
    return b.toString();
  }

  static const _pick = r'''
/// The rows [rows] of [m], in that order.
Matrix pick(Matrix m, List<int> rows) {
  final out = Matrix(rows.length, m.cols);
  for (final (i, r) in rows.indexed) {
    out.data.setRange(i * m.cols, (i + 1) * m.cols, m.data, r * m.cols);
  }
  return out;
}''';

  static const _writePredictions = r'''
/// [table] with [outputColumns] appended: [p]'s row for each of [rows],
/// empty for the rows that were skipped.
File writePredictions(CsvData table, List<int> rows, Matrix p, String path) {
  final at = {for (var i = 0; i < rows.length; i++) rows[i]: i};
  String cell(String c) =>
      c.contains(RegExp('[,"\n]')) ? '"${c.replaceAll('"', '""')}"' : c;
  final b = StringBuffer()
    ..writeln([...table.header, ...outputColumns].map(cell).join(','));
  for (var r = 0; r < table.rows.length; r++) {
    final i = at[r];
    b.writeln([
      ...table.rows[r].map(cell),
      for (var j = 0; j < p.cols; j++) i == null ? '' : '${p.at(i, j)}',
    ].join(','));
  }
  final stem = path.replaceFirst(RegExp(r'\.[^./\\]*$'), '');
  return File('${stem}_$networkId.csv')
    ..writeAsStringSync(b.toString());
}''';
}

/// The name of [d]'s training function in Python scripts.
String pyDeepTrainFunction(DocDeep d) => 'train_deep_${pyIdentifier(d.id)}';

/// Python for [deep] networks in a script with the data helpers (`number`,
/// `Transforms`, `LAYOUT`): the shared helpers, then per network a spec
/// dict, `build_deep_<id>()` and `train_deep_<id>(table)`, which returns
/// the table with the prediction columns appended (trained as hgf.dart's
/// `DeepTraining` trains it).
String pyDeepFunctions(List<DocDeep> deep) {
  final b = StringBuffer()..write(DeepPythonExport._helpers);
  for (final d in deep) {
    b
      ..writeln()
      ..writeln()
      ..writeln()
      ..write(DeepPythonExport(d)._functions());
  }
  return b.toString();
}

/// A Python script for one deep network of a model document, with
/// pyhgf's `DeepNetwork` (JAX, float64): the same steps as
/// [DeepDartExport]'s script, the CSV read through the same pipeline.
/// hgf.dart's corrections to pyhgf (`DeepFixes`) have no pyhgf equivalent
/// and are flagged; seeded weight initialisers draw different numbers in
/// JAX, and NumPy shuffles in another order.
final class DeepPythonExport {
  DeepPythonExport(this.deep, {this.title, this.dataFile, this.layout});

  final DocDeep deep;
  final String? title;
  final String? dataFile;

  /// The data pipeline (see [DeepDartExport.layout]).
  final DataLayout? layout;

  String _coupling(String text, {required bool layer}) {
    final c = parseCoupling(text);
    // pyhgf's DeepNetwork reads coupling_fn=None on a layer as the
    // network's coupling, not as the identity.
    if (c == CouplingFn.linear) return 'lambda x: x';
    return pyCoupling(c);
  }

  String _bool(bool v) => v ? 'True' : 'False';

  /// `build_deep_<id>()`'s body.
  String network() {
    final s = deep.settings, d = const DocDeepSettings();
    final args = [
      if (s.coupling != d.coupling)
        'coupling_fn=${_coupling(s.coupling, layer: false)}',
      if (s.volatilityUpdates != d.volatilityUpdates)
        'volatility_updates="${s.volatilityUpdates.pyhgfName}"',
      if (s.maxPosteriorPrecision != d.maxPosteriorPrecision)
        'max_posterior_precision=${pyFloat(s.maxPosteriorPrecision)}',
      if (s.precisionClippingValue != d.precisionClippingValue)
        'precision_clipping_value=${pyFloat(s.precisionClippingValue)}',
      if (s.predictPrecision != d.predictPrecision)
        'predict_precision=${_bool(s.predictPrecision)}',
      if (s.feedforwardUncertainty != d.feedforwardUncertainty)
        'feedforward_uncertainty=${_bool(s.feedforwardUncertainty)}',
      if (s.tonicVolatility != d.tonicVolatility)
        'tonic_volatility=${_bool(s.tonicVolatility)}',
      if (s.meanFieldUpdates != d.meanFieldUpdates)
        'mean_field_updates=${_bool(s.meanFieldUpdates)}',
      if (s.carryPrecision != d.carryPrecision)
        'carry_precision=${_bool(s.carryPrecision)}',
      if (s.updateInputLayer != d.updateInputLayer)
        'update_input_layer=${_bool(s.updateInputLayer)}',
    ];
    final b = StringBuffer();
    if (s.fixes.curvatureFloor || s.fixes.logSpaceVolatility) {
      b.writeln(
        '    # hgf.dart runs this network with its corrections to pyhgf '
        '(DeepFixes:',
      );
      b.writeln(
        '    # ${[if (s.fixes.curvatureFloor) 'curvature floor', if (s.fixes.logSpaceVolatility) 'log-space volatility'].join(', ')}), '
        'which pyhgf does not have: results can differ',
      );
      b.writeln('    # where they apply.');
    }
    b.writeln(
      args.isEmpty
          ? '    net = DeepNetwork()'
          : '    net = DeepNetwork(\n${[for (final a in args) '        $a,'].join('\n')}\n    )',
    );
    for (final (i, l) in deep.layers.indexed) {
      final a = [
        '${l.size}',
        if (l.kind != LayerKind.volatile) 'kind="${l.kind.pyhgfName}"',
        if (l.addConstantInput case final v?) 'add_constant_input=${_bool(v)}',
        if (!l.fullyConnected) 'fully_connected=False',
        if (l.coupling case final c?)
          'coupling_fn=${_coupling(c, layer: true)}',
        if (l.volatilityParent case final v?) 'volatility_parent=${_bool(v)}',
        if (l.tonicVolatility case final v?) 'tonic_volatility=${pyFloat(v)}',
        if (l.tonicVolatilityVol case final v?)
          'tonic_volatility_vol=${pyFloat(v)}',
        for (final MapEntry(:key, :value) in l.initial.entries)
          '${key.pyhgfName}=${pyFloat(value)}',
        if (l.valueChildren case final v?) 'value_children=$v',
        if (l.volatilityChildren case final v?) 'volatility_children=$v',
        if (l.valueCoupling case final v?) 'value_coupling=${pyFloat(v)}',
        if (l.volatilityCoupling case final v?)
          'volatility_coupling=${pyFloat(v)}',
        if (l.volatilityFullyConnected case final v?)
          'volatility_fully_connected=${_bool(v)}',
        if (l.tonicDrift case final v?) 'tonic_drift=${pyFloat(v)}',
        if (l.autoconnectionStrength case final v?)
          'autoconnection_strength=${pyFloat(v)}',
      ];
      b.writeln(
        '    net = net.add_layer(${a.join(', ')})  # ${deep.layerName(i)}',
      );
    }
    b.write('    return net');
    return b.toString();
  }

  String _str(String s) =>
      '"${s.replaceAll(r'\', r'\\').replaceAll('"', r'\"')}"';
  String _list(List<String> v) => '[${v.map(_str).join(', ')}]';

  String get _id => pyIdentifier(deep.id);

  /// `DEEP_<ID>`, `build_deep_<id>()` and `train_deep_<id>(table)`.
  String _functions() {
    final d = deep, t = d.training;
    final cont = d.isContinuous;
    final spec = 'DEEP_${_id.toUpperCase()}';
    final b = StringBuffer()
      ..writeln('# Deep network "${d.label}": the columns it reads and writes.')
      ..writeln('$spec = dict(')
      ..writeln('    id=${_str(d.id)},')
      ..writeln('    features=${_list(d.features)},')
      ..writeln('    targets=${_list(d.targets)},')
      ..writeln(
        '    categories=${d.categories == null ? 'None' : _list(d.categories!)},',
      )
      ..writeln('    outputs=${_list(deepPredictionColumns(d))},')
      ..writeln(
        t.trainOn == null || cont
            ? '    train_on=None,'
            : '    train_on=lambda t: ${pyDataExpr(t.trainOn!)},  # ${t.trainOn}',
      )
      ..writeln(')')
      ..writeln()
      ..writeln()
      ..writeln('def build_deep_$_id():')
      ..writeln(
        '    """The network, built from the target layer (layer 0) up to the',
      )
      ..writeln("    ${cont ? 'top level' : 'feature layer'}.\"\"\"")
      ..writeln(network())
      ..writeln()
      ..writeln()
      ..writeln('def ${pyDeepTrainFunction(d)}(table):')
      ..writeln(
        '    """The table with the network\'s predictions appended '
        '(${cont ? 'filtered' : 'trained'} as',
      )
      ..writeln('    hgf.dart does)."""')
      ..writeln('    x, y, rows, train, who = read_deep(table, $spec)');
    if (cont) {
      b
        ..writeln(
          '    # Each participant on its own, with a fresh network, in order.',
        )
        ..writeln('    p = np.zeros(y.shape)')
        ..writeln('    for pid in dict.fromkeys(who):')
        ..writeln('        idx = [i for i, w in enumerate(who) if w == pid]')
        ..writeln('        net = build_deep_$_id()')
        ..writeln('        net.input_data(y[idx])')
        ..writeln(
          '        p[idx] = np.asarray(net.predictions).reshape(len(idx), -1)',
        );
    } else {
      b
        ..writeln('    tr = np.flatnonzero(train)')
        ..writeln('    if len(tr) == 0:')
        ..writeln(
          "        raise ValueError(f'Deep network \"{$spec[\"id\"]}\" has no rows to train on')",
        );
      if (t.standardise) {
        b
          ..writeln("    # Standardised with the training rows' means and sds.")
          ..writeln('    mean = x[tr].mean(axis=0)')
          ..writeln('    sd = x[tr].std(axis=0)')
          ..writeln('    sd = np.where(sd > 1e-12, sd, 1.0)')
          ..writeln('    x = (x - mean) / sd');
      }
      b.writeln('    net = build_deep_$_id()');
      if (t.init.strategy != null) {
        b.writeln(
          '    net = net.weight_initialisation("${t.init.pyhgfName}", '
          'key=jax.random.key(${t.seed}))',
        );
      }
      final synaptic = t.learningKind == LearningKind.synapticUncertainty;
      b.writeln(
        synaptic
            ? '    optimiser = None'
            : '    optimiser = optax.${t.optimiser}(${pyFloat(t.learningRate)})',
      );
      final learn = [
        'optimiser=optimiser',
        if (t.learningKind != LearningKind.precisionWeighted)
          'learning_kind="${t.learningKind.pyhgfName}"',
        if (synaptic)
          'learning_kwargs={"window": ${pyFloat(t.synapticWindow)}}',
      ].join(', ');
      b.writeln('    for epoch in range(1, ${t.epochs + 1}):');
      b.writeln(
        t.shuffle
            ? '        order = tr[np.random.default_rng(epoch).permutation(len(tr))]'
            : '        order = tr',
      );
      if (t.batchSize case final n?) {
        b
          ..writeln('        for s in range(0, len(order), $n):')
          ..writeln('            batch = order[s : s + $n]')
          ..writeln(
            '            net = net.batch_update(x[batch], y[batch], $learn)',
          );
      } else {
        b.writeln('        net = net.fit(x[order], y[order], $learn)');
      }
      b
        ..writeln('    p = np.asarray(net.predict(x))')
        ..writeln(
          '    print_deep_scores($spec["id"], p, y, train, '
          '${d.targetLayer!.kind == LayerKind.volatile ? 'True' : 'False'})',
        );
    }
    b.write('    return append_columns(table, rows, p, $spec["outputs"])');
    return b.toString();
  }

  String script() {
    final d = deep, t = d.training;
    final cont = d.isContinuous;
    final pipeline = _pipelineOf(layout);
    final b = StringBuffer()
      ..writeln(
        '"""${title ?? 'Deep network "${d.label}"'}, exported from hgf.dart.',
      )
      ..writeln()
      ..writeln(
        'Run it with pyhgf installed (https://github.com/ComputationalPsychiatry/pyhgf):',
      )
      ..writeln()
      ..writeln(
        dataFile == null
            ? '    python <this file> data.csv'
            : '    python <this file> [data.csv]   (default: $dataFile)',
      )
      ..writeln()
      ..write(_summary(d, layout, ''))
      ..writeln()
      ..writeln(
        'The network is build_deep_$_id(); the CSV is read through '
        'transform(table).',
      );
    if (!cont && t.init.strategy != null) {
      b
        ..writeln()
        ..writeln(
          'The initial weights (${t.init.pyhgfName}, seed ${t.seed}) are drawn by JAX, so they',
        )
        ..writeln("differ from hgf.dart's, and so does the trained network.");
    }
    if (!cont && t.shuffle) {
      b.writeln(
        'Rows are shuffled with NumPy, in another order than hgf.dart uses.',
      );
    }
    b
      ..writeln('"""')
      ..writeln()
      ..writeln('import csv')
      ..writeln('import math  # noqa: F401')
      ..writeln('import re  # noqa: F401')
      ..writeln('import sys')
      ..writeln('from functools import partial  # noqa: F401 (couplings)')
      ..writeln()
      ..writeln('import jax')
      ..writeln()
      ..writeln('jax.config.update("jax_enable_x64", True)')
      ..writeln()
      ..writeln('import jax.numpy as jnp  # noqa: E402, F401 (couplings)')
      ..writeln('import numpy as np  # noqa: E402')
      ..writeln('import optax  # noqa: E402, F401')
      ..writeln('from pyhgf.model import DeepNetwork  # noqa: E402')
      ..writeln()
      ..writeln()
      ..write(pyDataPipeline(pipeline))
      ..writeln()
      ..writeln();
    if (pipeline.deep.isEmpty) {
      b
        ..write(_helpers)
        ..writeln()
        ..writeln()
        ..writeln();
    }
    b
      ..writeln(_functions())
      ..writeln()
      ..writeln()
      ..writeln(_writeTable)
      ..writeln()
      ..writeln()
      ..writeln('def main(path):')
      ..writeln('    with open(path, encoding="utf-8") as f:')
      ..writeln('        table = parse_csv(f.read())')
      ..writeln('    table = transform(table)')
      ..writeln('    table = ${pyDeepTrainFunction(d)}(table)')
      ..writeln(
        '    stem = path.rsplit(".", 1)[0] if "." in path.rsplit("/", 1)[-1] '
        'else path',
      )
      ..writeln('    out = f"{stem}_${d.id}.csv"')
      ..writeln('    write_table(table, out)')
      ..writeln('    print("wrote", out)')
      ..writeln()
      ..writeln()
      ..writeln('if __name__ == "__main__":')
      ..writeln(
        dataFile == null
            ? '    if len(sys.argv) < 2:\n        sys.exit("usage: python <this file> data.csv")\n    main(sys.argv[1])'
            : '    main(sys.argv[1] if len(sys.argv) > 1 else ${_str(dataFile!)})',
      );
    return b.toString();
  }

  static const _helpers = r'''
def read_deep(table, spec):
    """What a deep network reads (as hgf.dart's DeepData): the rows that hold
    every column it reads (their indices), the features x and the targets y
    (one-hot over the categories for a label column), whether each row is a
    training row, and each row's participant."""
    header = table["header"]

    def column(name):
        if name not in header:
            raise ValueError(f'Deep network "{spec["id"]}" reads "{name}", which is not a column')
        return header.index(name)

    fx = [column(c) for c in spec["features"]]
    fy = [column(c) for c in spec["targets"]]
    cats = spec["categories"]
    on = None
    if spec["train_on"] is not None:
        on = spec["train_on"](Transforms(table, LAYOUT["participant"], LAYOUT["group"], LAYOUT["time"])).v
    p = LAYOUT["participant"]
    pc = header.index(p) if p is not None and p in header else None
    xs, ys, rows, train, who = [], [], [], [], []
    for r, row in enumerate(table["rows"]):
        x = [number(table, row[c]) for c in fx]
        if any(v is None or not np.isfinite(v) for v in x):
            continue
        if cats is not None:
            label = row[fy[0]].strip()
            if label not in cats:
                continue
            y = [1.0 if c == label else 0.0 for c in cats]
        else:
            y = [number(table, row[c]) for c in fy]
            if any(v is None or not np.isfinite(v) for v in y):
                continue
        xs.append(x)
        ys.append(y)
        rows.append(r)
        train.append(on is None or (not np.isnan(on[r]) and on[r] != 0))
        who.append("" if pc is None else row[pc].strip())
    width = len(cats) if cats is not None else len(fy)
    return (
        np.asarray(xs, dtype=np.float64).reshape(len(rows), len(fx)),
        np.asarray(ys, dtype=np.float64).reshape(len(rows), width),
        rows,
        np.asarray(train, dtype=bool),
        who,
    )


def print_deep_scores(name, p, y, train, regression):
    """R² or accuracy on the training rows and on the held-out ones."""
    for label, mask in (("training", train), ("held-out", ~train)):
        if not mask.any():
            continue
        pm, ym = p[mask], y[mask]
        if regression:
            score = 1 - ((pm - ym) ** 2).sum() / ((ym - ym.mean()) ** 2).sum()
            print(f"{name}: R² on the {label} rows: {score:.4f}")
        else:
            labels = (ym[:, 0] > 0.5) if ym.shape[1] == 1 else ym.argmax(axis=1)
            guess = (pm[:, 0] > 0.5) if pm.shape[1] == 1 else pm.argmax(axis=1)
            print(f"{name}: accuracy on the {label} rows: {(labels == guess).mean():.4f}")


def append_columns(table, rows, p, names):
    """The table with the columns names appended: p's row for each of rows,
    empty for the rows the network skipped."""
    for n in names:
        if n in table["header"]:
            raise ValueError(f'The data already has a column "{n}"')
    at = {r: i for i, r in enumerate(rows)}
    out = [
        row + ([repr(float(v)) for v in p[at[r]]] if r in at else [""] * len(names))
        for r, row in enumerate(table["rows"])
    ]
    return {**table, "header": table["header"] + list(names), "rows": out}''';

  static const _writeTable = r'''
def write_table(table, path):
    """The table as CSV."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(table["header"])
        w.writerows(table["rows"])''';
}
