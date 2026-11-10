// Turns a saved network or model into a runnable script.
//
//   dart run hgf_fit:export model.json [options]
//
// See `usage` below.
import 'dart:convert';
import 'dart:io';

import 'package:hgf_fit/codegen.dart';

const usage = '''
Usage: dart run hgf_fit:export <file.json> [options]

<file.json> holds a Network.toJson() snapshot or a FitCodec.encodeModel()
model. Writes a Dart (package:hgf, hgf_fit) or Python (pyhgf, PyMC)
script that builds it, reads data from a CSV, filters it and (for a model)
fits it.

Options:
  --to dart|python     The target language (default: dart).
  --data <file.csv>    The CSV the script reads by default (default: the
                       first command-line argument of the script).
  --fit map|nuts|none  The fit a model's script runs (default: map).
  --restarts <n>       MAP restarts drawn from the priors (default: 3).
  --chains <n>, --warmup <n>, --draws <n>
                       NUTS settings (default: 4, 500, 500).
  --title <text>       A line for the header comment.
  -o <file.dart>       Write there instead of to standard output.
''';

Never _fail(String message) {
  stderr
    ..writeln(message)
    ..writeln()
    ..write(usage);
  exit(64);
}

void main(List<String> args) {
  if (args.isEmpty || args.contains('-h') || args.contains('--help')) {
    stdout.write(usage);
    return;
  }
  String? input, out, dataFile, title;
  var fit = 'map', to = 'dart';
  var restarts = 3, chains = 4, warmup = 500, draws = 500;
  for (var i = 0; i < args.length; i++) {
    final a = args[i];
    String value() =>
        i + 1 < args.length ? args[++i] : _fail('$a needs a value');
    int intValue() => int.tryParse(value()) ?? _fail('$a needs a whole number');
    switch (a) {
      case '--to':
        to = value();
      case '--data':
        dataFile = value();
      case '--fit':
        fit = value();
      case '--restarts':
        restarts = intValue();
      case '--chains':
        chains = intValue();
      case '--warmup':
        warmup = intValue();
      case '--draws':
        draws = intValue();
      case '--title':
        title = value();
      case '-o':
        out = value();
      case final s when s.startsWith('-'):
        _fail('Unknown option $s');
      default:
        if (input != null) _fail('One input file, please');
        input = a;
    }
  }
  if (input == null) _fail('No input file');
  if (to != 'dart' && to != 'python') {
    _fail('Unknown target "$to" (dart or python)');
  }
  final json = jsonDecode(File(input).readAsStringSync());
  if (json is! Map<String, Object?>) _fail('$input is not a JSON object');
  final isModel = switch (json['format']) {
    'hgf.network/1' => false,
    'hgf_fit.model/1' => true,
    final f => _fail(
      '$input is neither a Network.toJson() snapshot nor a '
      'FitCodec.encodeModel() model (format "$f")',
    ),
  };
  final ExportFit? exportFit = !isModel
      ? null
      : switch (fit) {
          'map' => ExportMap(restarts: restarts),
          'nuts' => ExportNuts(
            restarts: restarts,
            chains: chains,
            warmup: warmup,
            draws: draws,
          ),
          'none' => null,
          _ => _fail('--fit takes map, nuts or none'),
        };
  final String script;
  try {
    final spec = ExportSpec(
      network: isModel ? null : json,
      model: isModel ? json : null,
      dataFile: dataFile,
      fit: exportFit,
      title: title,
    );
    script = to == 'python'
        ? PythonExport(spec).script()
        : DartExport(spec).script();
  } on UnsupportedError catch (e) {
    stderr.writeln(e.message);
    exit(1);
  }
  if (out == null) {
    stdout.write(script);
  } else {
    File(out).writeAsStringSync(script);
  }
}
