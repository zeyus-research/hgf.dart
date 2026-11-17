import 'dart:typed_data';

import '../document/data_layout.dart';
import '../document/model_document.dart' show DocGroup;
import 'plan.dart';

/// What to export: a network or a fitting model, optionally with data and
/// a fit to run on it. The input of every code generator.
final class ExportSpec {
  /// Either [network] (`Network.toJson()`) or [model]
  /// (`FitCodec.encodeModel`, which holds its network).
  ExportSpec({
    Map<String, Object?>? network,
    this.model,
    this.data,
    this.dataFile,
    this.fit,
    this.title,
    this.columnNames,
    this.derivedColumns = const {},
    this.layout,
    this.csv,
    this.group,
    this.outputs = const [],
  }) : network =
           network ??
           (model?['network'] as Map<String, Object?>?) ??
           (throw ArgumentError('Give a network or a model')) {
    if (fit != null && model == null) {
      throw ArgumentError('A fit needs a model');
    }
    if (group != null && !group!.isEmpty && fit == null) {
      throw ArgumentError('A group level needs a fit');
    }
  }

  /// `Network.toJson()`.
  final Map<String, Object?> network;

  /// `FitCodec.encodeModel(...)`, if the script fits.
  final Map<String, Object?>? model;

  /// The data, embedded in the script (and written by [ExportData.toCsv]).
  final ExportData? data;

  /// The CSV file the script reads by default (written by
  /// [ExportData.toCsv]); null: the script uses the embedded [data], or a
  /// file given on the command line.
  final String? dataFile;

  /// The fit to run.
  final ExportFit? fit;

  /// A line for the header comment, e.g. the preset's name.
  final String? title;

  /// The CSV's column names, instead of the defaults (see
  /// [ExportData.columns]): a model document's named inputs and response.
  final ExportColumnNames? columnNames;

  /// Columns a model document computes from the data before filtering
  /// (name → expression, see `DocTransform`): the scripts read them as
  /// columns, so the CSV must hold them (`CompiledDocument.transform`).
  final Map<String, String> derivedColumns;

  /// How a model document reads its CSV: with a layout, the scripts read
  /// every participant by column name (labels, a participant column, a
  /// time column, the transforms) instead of [columnNames]' one series,
  /// fit them all (with the [group] level), and write the estimates and
  /// [outputs] per row. [data] is then unused: [csv] is the embedded data.
  final DataLayout? layout;

  /// The data as CSV text, embedded in a [layout] script.
  final String? csv;

  /// The group level over participants (a [layout] script with a fit).
  final DocGroup? group;

  /// Trajectory outputs a [layout] script writes per row: a node's field
  /// (pyhgf name) or `surprise`.
  final List<ExportOutput> outputs;

  late final NetworkPlan plan = NetworkPlan.fromJson(network);
}

/// A named trajectory output: [node]'s [field] (a `NodeField`'s pyhgf
/// name, or `surprise` for the node's surprise).
typedef ExportOutput = ({String name, int node, String field});

/// How the exported script fits.
sealed class ExportFit {
  const ExportFit();
}

/// MAP (L-BFGS, or Nelder–Mead where the model has no exact gradients)
/// from the parameters' starting values plus [restarts] starts drawn from
/// the priors, then the Laplace approximation: as the playground does.
final class ExportMap extends ExportFit {
  const ExportMap({this.restarts = 3, this.seed = 1});

  final int restarts;

  /// Seeds the restarts' `Random`.
  final int seed;
}

/// NUTS, started at the MAP estimate: as the playground does.
final class ExportNuts extends ExportFit {
  const ExportNuts({
    this.restarts = 3,
    this.chains = 4,
    this.warmup = 500,
    this.draws = 500,
    this.targetAccept = 0.8,
    this.maxTreeDepth = 10,
    this.seed = 0,
  });

  /// The MAP fit's restarts (see [ExportMap]).
  final int restarts;
  final int chains, warmup, draws;
  final double targetAccept;
  final int maxTreeDepth;
  final int seed;
}

/// One series of data, laid out as for `Network.inputData` and
/// `Subject`: [inputs] row-major (`nTime x nInputColumns`), [observed]
/// one flag per trial and input node, [timeSteps] one per trial,
/// [responses] one or more per trial (NaN: no response).
final class ExportData {
  ExportData({
    required this.inputs,
    this.observed,
    this.timeSteps,
    this.responses,
  });

  final Float64List inputs;
  final Float64List? observed, timeSteps, responses;

  /// The number of trials, for a network whose rows are [width] wide.
  int nTime(int width) => inputs.length ~/ width;

  /// The CSV columns the exported script reads: one per input column,
  /// then an `observed` flag per input node (written when the data has an
  /// observed mask; without them, an input node with an empty cell is
  /// unobserved on that trial), the responses and the time steps. A
  /// network with a single scalar input calls it `observation`, as the
  /// playground's export does; otherwise `input_<node>` (`input_<node>_<j>`
  /// for a vector input) and `observed_<node>`.
  ///
  /// [names] replaces these names (the responses' only when there are as
  /// many as [responsesPerTrial]).
  static ExportColumns columns(
    NetworkPlan plan, {
    int responsesPerTrial = 0,
    bool observed = false,
    bool timeSteps = false,
    ExportColumnNames? names,
  }) {
    if (names != null) {
      if (names.inputs.length != plan.inputs.length ||
          names.observed.length != plan.inputs.length) {
        throw ArgumentError('One name list per input node');
      }
      for (var k = 0; k < plan.inputs.length; k++) {
        if (names.inputs[k].length != plan.inputDimensions[k]) {
          throw ArgumentError(
            'Input ${plan.inputs[k]} takes ${plan.inputDimensions[k]} '
            'columns, not ${names.inputs[k].length}',
          );
        }
      }
      final defaults = columns(plan, responsesPerTrial: responsesPerTrial);
      return (
        inputs: names.inputs,
        observed: names.observed,
        writeObserved: observed,
        responses: names.responses.length == responsesPerTrial
            ? names.responses
            : defaults.responses,
        timeStep: timeSteps ? names.timeStep : null,
      );
    }
    final single = plan.inputs.length == 1 && plan.inputDimensions.single == 1;
    return (
      inputs: [
        for (var k = 0; k < plan.inputs.length; k++)
          if (single)
            ['observation']
          else if (plan.inputDimensions[k] == 1)
            ['input_${plan.inputs[k]}']
          else
            [
              for (var j = 0; j < plan.inputDimensions[k]; j++)
                'input_${plan.inputs[k]}_$j',
            ],
      ],
      observed: [
        for (final node in plan.inputs) single ? 'observed' : 'observed_$node',
      ],
      writeObserved: observed,
      responses: responsesPerTrial == 1
          ? ['response']
          : [for (var j = 0; j < responsesPerTrial; j++) 'response_$j'],
      timeStep: timeSteps ? 'time_step' : null,
    );
  }

  /// The responses per trial.
  int responsesPerTrial(int nTime) =>
      responses == null || nTime == 0 ? 0 : responses!.length ~/ nTime;

  /// This data as the CSV the exported scripts read (see [columns]).
  String toCsv(NetworkPlan plan, {ExportColumnNames? names}) {
    final width = plan.nInputColumns, nIn = plan.inputs.length;
    final n = nTime(width), r = responsesPerTrial(n);
    final cols = columns(
      plan,
      responsesPerTrial: r,
      observed: observed != null,
      timeSteps: timeSteps != null,
      names: names,
    );
    final b = StringBuffer()..writeln(cols.header.join(','));
    for (var t = 0; t < n; t++) {
      final row = <String>[
        for (var c = 0; c < width; c++) csvNumber(inputs[t * width + c]),
      ];
      if (observed case final o?) {
        for (var k = 0; k < nIn; k++) {
          row.add(csvNumber(o[t * nIn + k]));
        }
      }
      for (var j = 0; j < r; j++) {
        row.add(csvNumber(responses![t * r + j]));
      }
      if (timeSteps case final ts?) row.add(csvNumber(ts[t]));
      b.writeln(row.join(','));
    }
    return b.toString();
  }
}

/// Column names for [ExportData.columns]: per input node (in the network's
/// input order) its data columns and its observed-flag column, the
/// response columns, and the time-step column.
typedef ExportColumnNames = ({
  List<List<String>> inputs,
  List<String> observed,
  List<String> responses,
  String timeStep,
});

/// The columns of [ExportData.toCsv]'s CSV.
typedef ExportColumns = ({
  List<List<String>> inputs,
  List<String> observed,
  bool writeObserved,
  List<String> responses,
  String? timeStep,
});

extension ExportColumnsHeader on ExportColumns {
  /// The header row, in order.
  List<String> get header => [
    ...inputs.expand((c) => c),
    if (writeObserved) ...observed,
    ...responses,
    ?timeStep,
  ];
}

/// A number for CSV: empty when missing (NaN), `inf`/`-inf` as in pandas,
/// otherwise the shortest text that reads back to the same double.
String csvNumber(double v) => v.isNaN
    ? ''
    : v == double.infinity
    ? 'inf'
    : v == double.negativeInfinity
    ? '-inf'
    : '$v';
