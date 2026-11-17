import 'package:hgf/hgf.dart';
import 'package:hgf_deep/hgf_deep.dart' show LayerKind, LearningKind;

import '../document/deep_document.dart';
import 'dart.dart' show dartDouble;
import 'expr_code.dart' show dartString;
import 'python.dart' show pyCoupling, pyCsvReader, pyFloat;

/// What a deep network's script reads and writes, in words (both targets).
String _summary(DocDeep d, String comment) {
  final t = d.training;
  final lines = <String>[];
  if (d.isContinuous) {
    lines.add(
      'It reads the observation column${d.targets.length == 1 ? '' : 's'} '
      '${d.targets.join(', ')} from the CSV (rows with a missing value are '
      'skipped), filters them through the network (a continuous network '
      'does not learn: its couplings are parameters), and writes '
      '<data>_${d.id}.csv: the data with the prediction of every '
      'observation made before it was seen.',
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
      'from the CSV (rows with a missing value are skipped),'
      '${t.standardise ? ' standardises the features,' : ''} trains for '
      '${t.epochs} pass${t.epochs == 1 ? '' : 'es'} ($how, $rows'
      '${t.shuffle ? ', shuffled' : ', in order'}), prints the fit to the '
      'training rows and writes <data>_${d.id}.csv: the data with a '
      'prediction column per target node.',
    );
  }
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

/// The prediction columns of [d]'s output CSV.
List<String> deepPredictionColumns(DocDeep d) {
  final n = d.targetLayer!.size;
  if (d.categories case final cats?) {
    return [for (final c in cats) '${d.id}_p_$c'];
  }
  if (n == 1) return ['${d.id}_prediction'];
  return [for (var k = 0; k < n; k++) '${d.id}_prediction_$k'];
}

/// A Dart script for one deep network of a model document: it builds the
/// network, reads its columns from a CSV, trains it (or filters, for a
/// continuous network) as [DocDeep.training] says, and writes the
/// predictions next to the data. Depends on hgf_deep only (and hgf for
/// expression couplings).
final class DeepDartExport {
  DeepDartExport(this.deep, {this.title, this.dataFile});

  final DocDeep deep;
  final String? title;

  /// The CSV read when none is given on the command line.
  final String? dataFile;

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
    String list(List<String> v) => '[${v.map(dartString).join(', ')}]';
    final b = StringBuffer()
      ..writeln(
        '// ${title ?? 'Deep network "${d.label}"'}, exported from hgf.dart.',
      )
      ..writeln('//')
      ..writeln(
        '// Run it in a Dart package that depends on hgf_deep'
        '${_usesExpr ? ' and hgf' : ''}',
      )
      ..writeln('// (https://github.com/NexusDynamic/hgf.dart):')
      ..writeln('//')
      ..writeln(
        dataFile == null
            ? '//   dart run <this file> data.csv'
            : '//   dart run <this file> [data.csv]   (default: $dataFile)',
      )
      ..writeln('//')
      ..write(_summary(d, '// '))
      ..writeln('//')
      ..writeln(
        '// To read your data another way, replace readData(); the network '
        'is',
      )
      ..writeln('// buildNetwork().')
      ..writeln()
      ..write(cont ? '' : "import 'dart:convert';\n")
      ..writeln("import 'dart:io';")
      ..writeln("import 'dart:typed_data';")
      ..writeln();
    if (_usesExpr) {
      b.writeln("import 'package:hgf/hgf.dart' show ExprCoupling;");
    }
    b
      ..writeln("import 'package:hgf_deep/datasets.dart';")
      ..writeln("import 'package:hgf_deep/hgf_deep.dart';")
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
        '/// The network, built from the target layer (layer 0) up to the',
      )
      ..writeln(
        "/// ${cont ? 'top level' : 'feature layer'}, as pyhgf builds it.",
      )
      ..writeln('DeepNetwork buildNetwork() => ${network()};')
      ..writeln()
      ..writeln(_readData)
      ..writeln()
      ..writeln('void main(List<String> args) {')
      ..writeln(
        dataFile == null
            ? "  if (args.isEmpty) {\n    stderr.writeln('usage: dart run <this file> data.csv');\n    exit(64);\n  }\n  final path = args.first;"
            : '  final path = args.isEmpty ? ${dartString(dataFile!)} : args.first;',
      )
      ..writeln('  final csv = CsvData.parse(File(path).readAsStringSync());')
      ..writeln('  final data = readData(csv);')
      ..writeln('  final net = buildNetwork();');
    if (cont) {
      b.writeln('  final p = net.inputData(data.y);');
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
      if (t.standardise) {
        b
          ..writeln('  final scale = Standardiser.fit(data.x);')
          ..writeln('  final x = scale.apply(data.x);');
      } else {
        b.writeln('  final x = data.x;');
      }
      b.writeln("  final train = Dataset('train', x, data.y);");
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
      b
        ..writeln('  }')
        ..writeln('  final p = net.predict(x);')
        ..writeln(
          d.targetLayer!.kind == LayerKind.volatile
              ? "  print('R² on the training rows: \${r2(p, data.y).toStringAsFixed(4)}');"
              : "  print('accuracy on the training rows: \${accuracy(p, labelsOf(data.y)).toStringAsFixed(4)}');",
        )
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
      ..writeln('  final written = writePredictions(csv, data.rows, p, path);')
      ..writeln("  print('wrote \${written.path}');")
      ..writeln('}')
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

  String get _readData {
    final labels = deep.categories != null;
    final target = labels
        ? r'''
    final k = categories.indexOf(csv.rows[r][fy.single].trim());
    if (k < 0) continue;
    final y = [for (var j = 0; j < categories.length; j++) j == k ? 1.0 : 0.0];'''
        : r'''
    final y = [for (final c in fy) csv.number(r, c)];
    if (y.any((v) => v == null || !v.isFinite)) continue;''';
    final width = labels ? 'categories.length' : 'targets.length';
    final readX = deep.features.isEmpty
        ? ''
        : r'''
    final x = [for (final c in fx) csv.number(r, c)];
    if (x.any((v) => v == null || !v.isFinite)) continue;''';
    return '''
/// The rows of [csv] that hold every column the network reads (their
/// indices in `rows`), the features as `x` and the targets as `y`${labels ? ' (one-hot\n/// over `categories`)' : ''}.
({Matrix x, Matrix y, List<int> rows}) readData(CsvData csv) {
  int column(String name) {
    final c = csv.header.indexOf(name);
    if (c < 0) throw FormatException('The CSV has no column "\$name"');
    return c;
  }

${deep.features.isEmpty ? '' : '  final fx = [for (final f in features) column(f)];\n'}  final fy = [for (final t in targets) column(t)];
  final xs = <double>[], ys = <double>[], rows = <int>[];
  for (var r = 0; r < csv.rows.length; r++) {$readX$target
${deep.features.isEmpty ? '' : '    xs.addAll(x.cast<double>());\n'}    ys.addAll(${labels ? 'y' : 'y.cast<double>()'});
    rows.add(r);
  }
  return (
    x: Matrix(rows.length, features.length, Float64List.fromList(xs)),
    y: Matrix(rows.length, $width, Float64List.fromList(ys)),
    rows: rows,
  );
}''';
  }

  static const _writePredictions = r'''
/// The data with [outputColumns] appended: [p]'s row for each of [rows],
/// empty for the rows that were skipped.
File writePredictions(CsvData csv, List<int> rows, Matrix p, String path) {
  final at = {for (var i = 0; i < rows.length; i++) rows[i]: i};
  final b = StringBuffer()
    ..writeln([...csv.header, ...outputColumns].join(','));
  for (var r = 0; r < csv.rows.length; r++) {
    final i = at[r];
    b.writeln([
      ...csv.rows[r],
      for (var j = 0; j < p.cols; j++) i == null ? '' : '${p.at(i, j)}',
    ].join(','));
  }
  final stem = path.replaceFirst(RegExp(r'\.[^./\\]*$'), '');
  return File('${stem}_$networkId.csv')
    ..writeAsStringSync(b.toString());
}''';
}

/// A Python script for one deep network of a model document, with
/// pyhgf's `DeepNetwork` (JAX, float64): the same steps as
/// [DeepDartExport]'s script. hgf.dart's corrections to pyhgf
/// (`DeepFixes`) have no pyhgf equivalent and are flagged; seeded weight
/// initialisers draw different numbers in JAX.
final class DeepPythonExport {
  DeepPythonExport(this.deep, {this.title, this.dataFile});

  final DocDeep deep;
  final String? title;
  final String? dataFile;

  String _coupling(String text, {required bool layer}) {
    final c = parseCoupling(text);
    // pyhgf's DeepNetwork reads coupling_fn=None on a layer as the
    // network's coupling, not as the identity.
    if (c == CouplingFn.linear) return 'lambda x: x';
    return pyCoupling(c);
  }

  String _bool(bool v) => v ? 'True' : 'False';

  /// `build_network()`'s body.
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

  String script() {
    final d = deep, t = d.training;
    final cont = d.isContinuous;
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
      ..write(_summary(d, ''))
      ..writeln()
      ..writeln(
        'To read your data another way, replace read_data(); the network is',
      )
      ..writeln('build_network().');
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
      ..writeln('import sys')
      ..writeln('from functools import partial  # noqa: F401 (couplings)')
      ..writeln()
      ..writeln('import jax')
      ..writeln()
      ..writeln('jax.config.update("jax_enable_x64", True)')
      ..writeln()
      ..writeln('import jax.numpy as jnp  # noqa: E402, F401 (couplings)')
      ..writeln('import numpy as np  # noqa: E402');
    if (!cont) b.writeln('import optax  # noqa: E402');
    b
      ..writeln('from pyhgf.model import DeepNetwork  # noqa: E402')
      ..writeln()
      ..writeln('NETWORK_ID = ${_str(d.id)}')
      ..writeln('FEATURES = ${_list(d.features)}')
      ..writeln('TARGETS = ${_list(d.targets)}')
      ..writeln(
        'CATEGORIES = ${d.categories == null ? 'None' : _list(d.categories!)}',
      )
      ..writeln('OUTPUT_COLUMNS = ${_list(deepPredictionColumns(d))}')
      ..writeln()
      ..writeln()
      ..writeln('def build_network():')
      ..writeln(
        '    """The network, built from the target layer (layer 0) up to the',
      )
      ..writeln("    ${cont ? 'top level' : 'feature layer'}.\"\"\"")
      ..writeln(network())
      ..writeln()
      ..writeln()
      ..write(pyCsvReader)
      ..writeln(_readData)
      ..writeln()
      ..writeln()
      ..writeln(_writePredictions)
      ..writeln()
      ..writeln()
      ..writeln('def main(path):')
      ..writeln('    with open(path, encoding="utf-8") as f:')
      ..writeln('        table = parse_csv(f.read())')
      ..writeln('    x, y, rows = read_data(table)')
      ..writeln('    net = build_network()');
    if (cont) {
      b
        ..writeln('    net.input_data(y)')
        ..writeln('    p = np.asarray(net.predictions).reshape(len(rows), -1)');
    } else {
      if (t.standardise) {
        b
          ..writeln('    mean = x.mean(axis=0)')
          ..writeln('    sd = x.std(axis=0)')
          ..writeln('    sd = np.where(sd > 1e-12, sd, 1.0)')
          ..writeln('    x = (x - mean) / sd');
      }
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
            ? '        order = np.random.default_rng(epoch).permutation(len(x))'
            : '        order = np.arange(len(x))',
      );
      if (t.batchSize case final n?) {
        b
          ..writeln('        for s in range(0, len(x), $n):')
          ..writeln('            batch = order[s : s + $n]')
          ..writeln(
            '            net = net.batch_update(x[batch], y[batch], $learn)',
          );
      } else {
        b.writeln('        net = net.fit(x[order], y[order], $learn)');
      }
      b.writeln('    p = np.asarray(net.predict(x))');
      if (d.targetLayer!.kind == LayerKind.volatile) {
        b
          ..writeln(
            '    r2 = 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()',
          )
          ..writeln('    print(f"R² on the training rows: {r2:.4f}")');
      } else {
        b
          ..writeln(
            '    labels = (y[:, 0] > 0.5) if y.shape[1] == 1 else y.argmax(axis=1)',
          )
          ..writeln(
            '    guess = (p[:, 0] > 0.5) if p.shape[1] == 1 else p.argmax(axis=1)',
          )
          ..writeln(
            '    print(f"accuracy on the training rows: {(labels == guess).mean():.4f}")',
          );
      }
    }
    b
      ..writeln('    print("wrote", write_predictions(table, rows, p, path))')
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

  static const _readData = r'''

def read_data(table):
    """The rows that hold every column the network reads (their indices),
    the features as x and the targets as y (one-hot over CATEGORIES for a
    label column)."""
    header = table["header"]

    def column(name):
        if name not in header:
            raise ValueError(f'The CSV has no column "{name}"')
        return header.index(name)

    fx = [column(c) for c in FEATURES]
    fy = [column(c) for c in TARGETS]
    xs, ys, rows = [], [], []
    for r, row in enumerate(table["rows"]):
        x = [number(table, row[c]) for c in fx]
        if any(v is None or not np.isfinite(v) for v in x):
            continue
        if CATEGORIES is not None:
            label = row[fy[0]].strip()
            if label not in CATEGORIES:
                continue
            y = [1.0 if c == label else 0.0 for c in CATEGORIES]
        else:
            y = [number(table, row[c]) for c in fy]
            if any(v is None or not np.isfinite(v) for v in y):
                continue
        xs.append(x)
        ys.append(y)
        rows.append(r)
    width = len(CATEGORIES) if CATEGORIES is not None else len(TARGETS)
    return (
        np.asarray(xs, dtype=np.float64).reshape(len(rows), len(FEATURES)),
        np.asarray(ys, dtype=np.float64).reshape(len(rows), width),
        rows,
    )''';

  static const _writePredictions = r'''
def write_predictions(table, rows, p, path):
    """The data with OUTPUT_COLUMNS appended: p's row for each of rows,
    empty for the rows that were skipped."""
    at = {r: i for i, r in enumerate(rows)}
    stem = path.rsplit(".", 1)[0] if "." in path.rsplit("/", 1)[-1] else path
    out = f"{stem}_{NETWORK_ID}.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(table["header"] + OUTPUT_COLUMNS)
        for r, row in enumerate(table["rows"]):
            i = at.get(r)
            w.writerow(row + (["" for _ in OUTPUT_COLUMNS] if i is None else [repr(float(v)) for v in p[i]]))
    return out''';
}
