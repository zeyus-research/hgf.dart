/// Code generation: turns a network or a fitting model (their JSON, from
/// `Network.toJson` and `FitCodec.encodeModel`) into a runnable script, to
/// carry on from the playground in code.
///
/// ```dart
/// final spec = ExportSpec(
///   model: const FitCodec().encodeModel(model),
///   data: ExportData(inputs: u, responses: responses),
///   fit: const ExportMap(),
/// );
/// File('fit.dart').writeAsStringSync(DartExport(spec).script());
/// ```
///
/// [NetworkPlan] recovers the builder calls from a snapshot and checks
/// that they rebuild it exactly; every target writes those calls.
library;

export 'src/codegen/dart.dart' show DartExport, dartDouble, dartIdentifier;
export 'src/codegen/export_spec.dart';
export 'src/codegen/plan.dart';
export 'src/codegen/python.dart' show PythonExport, pyFloat, pyIdentifier;
