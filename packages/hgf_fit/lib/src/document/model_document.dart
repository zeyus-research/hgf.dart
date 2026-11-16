import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import '../codec.dart';
import '../codegen/export_spec.dart';
import '../codegen/plan.dart';
import '../model.dart';
import 'network_document.dart';

/// A named input: the network's input node [node] (an address), read from
/// the data's [columns].
///
/// An input takes as many columns as its node takes data columns (one, or
/// a categorical node's categories, an ef-state node's dimension); by
/// default [name] for one column, `name_0`, `name_1`, … for more. With
/// [categories], one column holds a category label (or its 0-based code)
/// per row: a categorical input reads it one-hot, a binary or continuous
/// input as the category's index (`['no', 'yes']` → 0, 1).
///
/// A row whose cell is empty (or `NA`, `NaN`, …) is unobserved for this
/// input, unless [observedColumn] names a column of 0/1 flags: then the
/// flags decide (an observed row needs a value; an unobserved one keeps its
/// values, which the filter records in the input node's mean). An empty
/// cell reads as 0, never NaN: a NaN at an unobserved trial still reaches
/// the parents' beliefs.
final class DocInput {
  DocInput(
    this.name, {
    required this.node,
    List<String>? columns,
    List<String>? categories,
    this.observedColumn,
  }) : columns = columns == null ? null : List.unmodifiable(columns),
       categories = categories == null ? null : List.unmodifiable(categories);

  final String name;
  final String node;
  final List<String>? columns, categories;
  final String? observedColumn;

  /// The data columns for an input [width] wide.
  List<String> columnsFor(int width) =>
      columns ??
      (categories != null || width == 1
          ? [name]
          : [for (var j = 0; j < width; j++) '${name}_$j']);

  Map<String, Object?> toJson() => {
    'name': name,
    'node': node,
    'columns': ?columns,
    'categories': ?categories,
    'observed_column': ?observedColumn,
  };

  factory DocInput.fromJson(Map<String, Object?> json) => DocInput(
    json['name'] as String,
    node: json['node'] as String,
    columns: (json['columns'] as List<Object?>?)?.cast<String>(),
    categories: (json['categories'] as List<Object?>?)?.cast<String>(),
    observedColumn: json['observed_column'] as String?,
  );
}

/// The named response: a response model ([readout], `FitCodec` JSON with
/// nodes by address) and where the recorded responses are.
///
/// [columns] default to [name]. With [options], a response cell holds an
/// option's label (or its 0-based code): a choice among them. [reveals]
/// gives, per option label, the names of the inputs a choice of it shows
/// (a task where the choice decides what is seen, as in the Iowa Gambling
/// Task); `HgfModel.simulateChoices` plays it.
final class DocResponse {
  DocResponse(
    this.name, {
    required Map<String, Object?> readout,
    List<String>? columns,
    List<String>? options,
    Map<String, List<String>>? reveals,
  }) : readout = Map.unmodifiable(readout),
       columns = columns == null ? null : List.unmodifiable(columns),
       options = options == null ? null : List.unmodifiable(options),
       reveals = reveals == null
           ? null
           : Map.unmodifiable({
               for (final MapEntry(:key, :value) in reveals.entries)
                 key: List<String>.unmodifiable(value),
             });

  final String name;
  final Map<String, Object?> readout;
  final List<String>? columns, options;
  final Map<String, List<String>>? reveals;

  List<String> get dataColumns => columns ?? [name];

  Map<String, Object?> toJson() => {
    'name': name,
    'readout': readout,
    'columns': ?columns,
    'options': ?options,
    'reveals': ?reveals,
  };

  factory DocResponse.fromJson(Map<String, Object?> json) => DocResponse(
    json['name'] as String,
    readout: json['readout'] as Map<String, Object?>,
    columns: (json['columns'] as List<Object?>?)?.cast<String>(),
    options: (json['options'] as List<Object?>?)?.cast<String>(),
    reveals: switch (json['reveals']) {
      final Map<String, Object?> r => {
        for (final MapEntry(:key, :value) in r.entries)
          key: (value as List<Object?>).cast<String>(),
      },
      _ => null,
    },
  );
}

/// A named trajectory output: a node's [field] over time (a `NodeField`'s
/// pyhgf name, or `surprise` for the node's surprise). For example a
/// prediction error per time stamp, as a regressor for EEG.
final class DocOutput {
  const DocOutput(this.name, {required this.node, required this.field});

  final String name, node, field;

  Map<String, Object?> toJson() => {'name': name, 'node': node, 'field': field};

  factory DocOutput.fromJson(Map<String, Object?> json) => DocOutput(
    json['name'] as String,
    node: json['node'] as String,
    field: json['field'] as String,
  );
}

/// How the data is laid out: long format, one row per time step.
///
/// [participantColumn] splits the rows into one subject per participant
/// (in order of first appearance; rows stay in file order). Time steps
/// come from [timeStepColumn], or from the differences of
/// [timeColumn] (the first row gets the second row's step), or are 1.
final class DocData {
  const DocData({this.participantColumn, this.timeColumn, this.timeStepColumn})
    : assert(
        timeColumn == null || timeStepColumn == null,
        'Give a time column or a time-step column, not both',
      );

  final String? participantColumn, timeColumn, timeStepColumn;

  Map<String, Object?> toJson() => {
    'participant_column': ?participantColumn,
    'time_column': ?timeColumn,
    'time_step_column': ?timeStepColumn,
  };

  factory DocData.fromJson(Map<String, Object?> json) => DocData(
    participantColumn: json['participant_column'] as String?,
    timeColumn: json['time_column'] as String?,
    timeStepColumn: json['time_step_column'] as String?,
  );
}

/// How to fit, as the playground does: MAP (with [restarts] starts drawn
/// from the priors, then the Laplace approximation), or NUTS started at
/// the MAP estimate; and the `HgfModel` settings.
final class DocFit {
  const DocFit({
    this.method = 'map',
    this.restarts = 3,
    this.seed,
    this.chains = 4,
    this.warmup = 500,
    this.draws = 500,
    this.targetAccept = 0.8,
    this.maxTreeDepth = 10,
    this.blockSize = 256,
    this.gradientMode = GradientMode.auto,
    this.reverseAbove = HgfModel.defaultReverseAbove,
  }) : assert(
         method == 'map' || method == 'nuts' || method == 'none',
         'method is map, nuts or none',
       );

  /// `map`, `nuts` or `none`.
  final String method;
  final int restarts;

  /// The seed of the restarts (MAP) or the chains (NUTS); default 1 for
  /// MAP and 0 for NUTS, as the exports use.
  final int? seed;
  final int chains, warmup, draws, maxTreeDepth;
  final double targetAccept;
  final int blockSize;
  final GradientMode gradientMode;
  final int reverseAbove;

  /// The fit for the code generators (null for `none`).
  ExportFit? get exportFit => switch (method) {
    'map' => ExportMap(restarts: restarts, seed: seed ?? 1),
    'nuts' => ExportNuts(
      restarts: restarts,
      chains: chains,
      warmup: warmup,
      draws: draws,
      targetAccept: targetAccept,
      maxTreeDepth: maxTreeDepth,
      seed: seed ?? 0,
    ),
    _ => null,
  };

  Map<String, Object?> toJson() => {
    'method': method,
    'restarts': restarts,
    'seed': ?seed,
    if (method == 'nuts') ...{
      'chains': chains,
      'warmup': warmup,
      'draws': draws,
      'target_accept': targetAccept,
      'max_tree_depth': maxTreeDepth,
    },
    if (blockSize != 256) 'block_size': blockSize,
    if (gradientMode != GradientMode.auto) 'gradient_mode': gradientMode.name,
    if (reverseAbove != HgfModel.defaultReverseAbove)
      'reverse_above': reverseAbove,
  };

  factory DocFit.fromJson(Map<String, Object?> json) => DocFit(
    method: (json['method'] as String?) ?? 'map',
    restarts: (json['restarts'] as num?)?.toInt() ?? 3,
    seed: (json['seed'] as num?)?.toInt(),
    chains: (json['chains'] as num?)?.toInt() ?? 4,
    warmup: (json['warmup'] as num?)?.toInt() ?? 500,
    draws: (json['draws'] as num?)?.toInt() ?? 500,
    targetAccept: (json['target_accept'] as num?)?.toDouble() ?? 0.8,
    maxTreeDepth: (json['max_tree_depth'] as num?)?.toInt() ?? 10,
    blockSize: (json['block_size'] as num?)?.toInt() ?? 256,
    gradientMode: switch (json['gradient_mode']) {
      final String m => GradientMode.values.byName(m),
      _ => GradientMode.auto,
    },
    reverseAbove:
        (json['reverse_above'] as num?)?.toInt() ??
        HgfModel.defaultReverseAbove,
  );
}

/// A whole model as the builder edits it (`hgf.document/1`): the
/// [network], its named [inputs], the [response] model, the free
/// [parameters], trajectory [outputs], the [data] layout and the [fit]
/// settings, plus the canvas [layout] (kept as is, not interpreted).
///
/// Nodes are named by address everywhere (an id, or `id.k` for an implied
/// node; see [NetworkDocument]). [response]'s readout and [parameters] are
/// `FitCodec` JSON with node addresses where `FitCodec` has indices (the
/// `node`, `parent` and `child` keys).
///
/// [compile] builds the `Network` and `HgfModel` and reads subjects from
/// CSV by column name.
final class ModelDocument {
  ModelDocument({
    required this.network,
    this.name,
    this.description,
    List<DocInput> inputs = const [],
    this.response,
    List<Map<String, Object?>> parameters = const [],
    List<DocOutput> outputs = const [],
    this.data = const DocData(),
    this.fit = const DocFit(),
    Map<String, Object?> layout = const {},
  }) : inputs = List.unmodifiable(inputs),
       parameters = List.unmodifiable(parameters),
       outputs = List.unmodifiable(outputs),
       layout = Map.unmodifiable(layout);

  /// The format tag of [toJson].
  static const format = 'hgf.document/1';

  final String? name, description;
  final NetworkDocument network;

  /// The named inputs; empty: every input node gets a default name (as
  /// [ModelDocument.fromNetwork] gives).
  final List<DocInput> inputs;
  final DocResponse? response;
  final List<Map<String, Object?>> parameters;
  final List<DocOutput> outputs;
  final DocData data;
  final DocFit fit;
  final Map<String, Object?> layout;

  /// A document for [network], with ids `n<index>` (or from [id]) and the
  /// input names the code exports' CSV uses: `observation` for a network
  /// with one scalar input, `input_<index>` otherwise.
  factory ModelDocument.fromNetwork(
    Network network, {
    String Function(int index)? id,
    String? name,
  }) {
    final doc = NetworkDocument.fromNetwork(network, id: id);
    return ModelDocument(
      network: doc,
      name: name,
      inputs: defaultInputs(doc, network),
    );
  }

  /// A document for [model]: its network (see [ModelDocument.fromNetwork]),
  /// its response model, named `response`, its parameters and settings.
  /// [codec] encodes custom types.
  factory ModelDocument.fromModel(
    HgfModel model, {
    FitCodec codec = const FitCodec(),
    String Function(int index)? id,
    String? name,
    DocFit? fit,
  }) {
    final doc = NetworkDocument.fromNetwork(model.network, id: id);
    final a = doc.addresses;
    Object? toAddress(Object? v) => v is num ? a.addressOf(v.toInt()) : v;
    return ModelDocument(
      network: doc,
      name: name,
      inputs: defaultInputs(doc, model.network),
      response: DocResponse(
        'response',
        readout:
            _remap(codec.encodeResponse(model.response), toAddress)
                as Map<String, Object?>,
      ),
      parameters: [
        for (final p in codec.encodeParameters(model.parameters))
          _remap(p, toAddress) as Map<String, Object?>,
      ],
      fit:
          fit ??
          DocFit(
            blockSize: model.blockSize,
            gradientMode: model.gradientMode,
            reverseAbove: model.reverseAbove,
          ),
    );
  }

  /// The default named inputs of [network] (built from [doc]).
  static List<DocInput> defaultInputs(NetworkDocument doc, Network network) {
    final a = doc.addresses;
    final idxs = network.inputIdxs;
    final single =
        idxs.length == 1 && _inputWidth(network, doc, idxs.single) == 1;
    return [
      for (final i in idxs)
        DocInput(single ? 'observation' : 'input_$i', node: a.addressOf(i)),
    ];
  }

  factory ModelDocument.fromJson(Map<String, Object?> json) {
    if (json['format'] != format) {
      throw FormatException(
        'Not an hgf model document (format "${json['format']}")',
      );
    }
    return ModelDocument(
      name: json['name'] as String?,
      description: json['description'] as String?,
      network: NetworkDocument.fromJson(
        json['network'] as Map<String, Object?>,
      ),
      inputs: [
        for (final i in (json['inputs'] as List<Object?>?) ?? const [])
          DocInput.fromJson(i as Map<String, Object?>),
      ],
      response: switch (json['response']) {
        final Map<String, Object?> r => DocResponse.fromJson(r),
        _ => null,
      },
      parameters: [
        for (final p in (json['parameters'] as List<Object?>?) ?? const [])
          p as Map<String, Object?>,
      ],
      outputs: [
        for (final o in (json['outputs'] as List<Object?>?) ?? const [])
          DocOutput.fromJson(o as Map<String, Object?>),
      ],
      data: switch (json['data']) {
        final Map<String, Object?> d => DocData.fromJson(d),
        _ => const DocData(),
      },
      fit: switch (json['fit']) {
        final Map<String, Object?> f => DocFit.fromJson(f),
        _ => const DocFit(),
      },
      layout: (json['layout'] as Map<String, Object?>?) ?? const {},
    );
  }

  Map<String, Object?> toJson() => {
    'format': format,
    'name': ?name,
    'description': ?description,
    'network': network.toJson(),
    if (inputs.isNotEmpty) 'inputs': [for (final i in inputs) i.toJson()],
    if (response case final r?) 'response': r.toJson(),
    if (parameters.isNotEmpty) 'parameters': parameters,
    if (outputs.isNotEmpty) 'outputs': [for (final o in outputs) o.toJson()],
    if (data.toJson() case final d when d.isNotEmpty) 'data': d,
    'fit': fit.toJson(),
    if (layout.isNotEmpty) 'layout': layout,
  };

  /// Builds the network and, with a [response], the model. [codec]
  /// decodes custom types and supplies the custom couplings (without them,
  /// the network gets placeholders that build but cannot run).
  CompiledDocument compile({FitCodec codec = const FitCodec()}) {
    final custom = network.customCouplings;
    final couplings = {
      for (final name in custom)
        name:
            codec.couplings[name] ??
            NetworkPlan.placeholderCouplings([name])[name]!,
    };
    final net = network.build(couplings: couplings);
    final a = network.addresses;
    Object? toIndex(Object? v) => v is String ? a.indexOf(v) : v;

    // Named inputs, in the network's input order.
    final named = inputs.isEmpty ? defaultInputs(network, net) : inputs;
    final byIndex = <int, DocInput>{};
    final names = <String>{};
    for (final input in named) {
      if (!names.add(input.name)) {
        throw FormatException('Two inputs are named "${input.name}"');
      }
      final i = a.indexOf(input.node);
      if (!net.inputIdxs.contains(i)) {
        throw FormatException(
          'Input "${input.name}": "${input.node}" is not an input node',
        );
      }
      if (byIndex.containsKey(i)) {
        throw FormatException(
          'Inputs "${byIndex[i]!.name}" and "${input.name}" both name '
          '"${input.node}"',
        );
      }
      byIndex[i] = input;
    }
    final ordered = [
      for (final i in net.inputIdxs)
        byIndex[i] ??
            (throw FormatException(
              'The input node "${a.addressOf(i)}" has no named input',
            )),
    ];

    HgfModel? model;
    List<List<int>>? reveals;
    if (response case final r?) {
      model = HgfModel(
        network: net,
        response: codec.decodeResponse(
          _remap(r.readout, toIndex) as Map<String, Object?>,
        ),
        parameters: codec.decodeParameters([
          for (final p in parameters) _remap(p, toIndex),
        ]),
        blockSize: fit.blockSize,
        gradientMode: fit.gradientMode,
        reverseAbove: fit.reverseAbove,
      );
      if (r.reveals case final rv?) {
        final options =
            r.options ??
            (throw const FormatException('reveals needs named options'));
        final position = {
          for (var k = 0; k < ordered.length; k++) ordered[k].name: k,
        };
        reveals = [
          for (final o in options)
            [
              for (final input in rv[o] ?? const <String>[])
                position[input] ??
                    (throw FormatException(
                      'Option "$o" reveals "$input", which is not an input',
                    )),
            ],
        ];
      }
    }
    for (final o in outputs) {
      a.indexOf(o.node);
      if (o.field != 'surprise') NodeField.parse(o.field);
    }
    return CompiledDocument._(this, net, model, ordered, reveals);
  }
}

/// A [ModelDocument], built: the [network], the [model] (with a
/// response), the named [inputs] in the network's input order, and the
/// option reveals.
final class CompiledDocument {
  CompiledDocument._(
    this.document,
    this.network,
    this.model,
    List<DocInput> inputs,
    this.reveals,
  ) : inputs = List.unmodifiable(inputs);

  final ModelDocument document;
  final Network network;
  final HgfModel? model;
  final List<DocInput> inputs;

  /// Per option, the positions in [inputs] its choice reveals (for
  /// `HgfModel.simulateChoices`), when the response says.
  final List<List<int>>? reveals;

  /// The fields to record for the [outputs] (the defaults and theirs).
  List<NodeField> get recordedFields => {
    ...defaultRecordedFields,
    for (final o in document.outputs)
      if (o.field != 'surprise') NodeField.parse(o.field),
  }.toList();

  /// The named outputs of [traj] (recorded with [recordedFields]).
  Map<String, Float64List> outputs(Trajectories traj) {
    final a = document.network.addresses;
    return {
      for (final o in document.outputs)
        o.name: o.field == 'surprise'
            ? traj.nodeSurprise(a.indexOf(o.node))
            : traj.series(a.indexOf(o.node), NodeField.parse(o.field)),
    };
  }

  /// The subjects in [csv], read by the document's column names (see
  /// [DocInput], [DocResponse] and [DocData]).
  List<Subject> subjects(CsvData csv) {
    final header = csv.header;
    int col(String name, String what) {
      final c = header.indexOf(name);
      if (c < 0) {
        throw FormatException(
          'No column "$name" ($what) in the data; its columns are '
          '${header.join(', ')}',
        );
      }
      return c;
    }

    final d = document.data;
    final groups = <String, List<int>>{};
    if (d.participantColumn case final p?) {
      final c = col(p, 'participants');
      for (var r = 0; r < csv.rows.length; r++) {
        (groups[csv.rows[r][c].trim()] ??= []).add(r);
      }
    } else {
      groups[''] = [for (var r = 0; r < csv.rows.length; r++) r];
    }

    // Per input: its columns, offset and width.
    final widths = [
      for (final i in network.inputIdxs)
        _inputWidth(network, document.network, i),
    ];
    final width = widths.fold(0, (a, b) => a + b);
    final nIn = inputs.length;
    final inputCols = [
      for (var k = 0; k < nIn; k++)
        [
          for (final name in inputs[k].columnsFor(widths[k]))
            col(name, 'input ${inputs[k].name}'),
        ],
    ];
    for (var k = 0; k < nIn; k++) {
      final input = inputs[k];
      final want = input.categories != null ? 1 : widths[k];
      if (inputCols[k].length != want) {
        throw FormatException(
          'Input "${input.name}" takes $want column(s), not '
          '${inputCols[k].length}',
        );
      }
      if (input.categories case final cats?
          when widths[k] > 1 && cats.length != widths[k]) {
        throw FormatException(
          'Input "${input.name}" has ${cats.length} categories; its node '
          'has ${widths[k]}',
        );
      }
    }
    final observedCols = [
      for (final input in inputs)
        input.observedColumn == null
            ? null
            : col(input.observedColumn!, 'observed flags of ${input.name}'),
    ];
    final r = document.response;
    final responseCols = r == null
        ? const <int>[]
        : [for (final name in r.dataColumns) col(name, 'response ${r.name}')];
    final tCol = d.timeColumn == null ? null : col(d.timeColumn!, 'time');
    final dtCol = d.timeStepColumn == null
        ? null
        : col(d.timeStepColumn!, 'time steps');

    double number(int row, int c) =>
        csv.number(row, c) ??
        (throw FormatException(
          'Row ${row + 2}, column "${header[c]}": "${csv.rows[row][c]}" is '
          'not a number',
        ));
    // A label first: "none" is a label here, not a missing value.
    int? code(int row, int c, List<String> labels) {
      final cell = csv.rows[row][c].trim();
      final i = labels.indexOf(cell);
      if (i >= 0) return i;
      if (csv.isMissing(row, c)) return null;
      final v = int.tryParse(cell);
      if (v != null && v >= 0 && v < labels.length) return v;
      throw FormatException(
        'Row ${row + 2}, column "${header[c]}": "$cell" is not one of '
        '${labels.join(', ')}',
      );
    }

    return [
      for (final MapEntry(key: id, value: rows) in groups.entries)
        () {
          final nTime = rows.length;
          final u = Float64List(nTime * width);
          final observed = Float64List(nTime * nIn)
            ..fillRange(0, nTime * nIn, 1);
          var masked = false;
          for (var t = 0; t < nTime; t++) {
            final row = rows[t];
            var off = 0;
            for (var k = 0; k < nIn; k++) {
              final input = inputs[k];
              final w = widths[k];
              final flagCol = observedCols[k];
              var seen = true;
              if (input.categories case final cats?) {
                final i = code(row, inputCols[k].single, cats);
                if (i == null) {
                  seen = false;
                } else {
                  if (w == 1) {
                    u[t * width + off] = i.toDouble();
                  } else {
                    u[t * width + off + i] = 1;
                  }
                }
              } else {
                for (var j = 0; j < w; j++) {
                  final v = number(row, inputCols[k][j]);
                  if (v.isNaN) {
                    seen = false;
                  } else {
                    u[t * width + off + j] = v;
                  }
                }
              }
              if (flagCol != null) {
                final flag = number(row, flagCol) != 0;
                if (flag && !seen) {
                  throw FormatException(
                    'Row ${row + 2}: input "${input.name}" is empty but '
                    'flagged observed',
                  );
                }
                seen = flag;
                masked = true;
              }
              if (!seen) {
                // Empty cells stay 0: a NaN at an unobserved trial would
                // still reach the parents' beliefs.
                observed[t * nIn + k] = 0;
                masked = true;
                if (flagCol == null) {
                  u.fillRange(t * width + off, t * width + off + w, 0);
                }
              }
              off += w;
            }
          }
          Float64List? timeSteps;
          if (dtCol != null) {
            timeSteps = Float64List.fromList([
              for (final row in rows) number(row, dtCol),
            ]);
          } else if (tCol != null) {
            final times = [for (final row in rows) number(row, tCol)];
            timeSteps = Float64List(nTime);
            for (var t = 1; t < nTime; t++) {
              timeSteps[t] = times[t] - times[t - 1];
            }
            if (nTime > 0) timeSteps[0] = nTime > 1 ? timeSteps[1] : 1.0;
          }
          Float64List? responses;
          if (r != null) {
            final nr = responseCols.length;
            responses = Float64List(nTime * nr);
            for (var t = 0; t < nTime; t++) {
              for (var j = 0; j < nr; j++) {
                final c = responseCols[j];
                final row = rows[t];
                final opts = r.options;
                responses[t * nr + j] = opts != null
                    ? code(row, c, opts)?.toDouble() ?? double.nan
                    : number(row, c);
              }
            }
          }
          return Subject(
            inputs: u,
            observed: masked ? observed : null,
            timeSteps: timeSteps,
            responses: responses,
            id: d.participantColumn == null ? null : id,
          );
        }(),
    ];
  }

  /// What the code generators take: the model (or the network alone), the
  /// document's fit, [data] to embed, and the document's column names (see
  /// [columnNames]).
  ExportSpec exportSpec({
    FitCodec codec = const FitCodec(),
    ExportData? data,
    String? dataFile,
  }) => ExportSpec(
    network: network.toJson(),
    model: model == null ? null : codec.encodeModel(model!),
    data: data,
    dataFile: dataFile,
    fit: model == null ? null : document.fit.exportFit,
    title: document.name,
    columnNames: columnNames,
  );

  /// The CSV columns of the exports: the named inputs' columns, their
  /// observed-flag columns (`observed_<name>` by default), the response's
  /// columns and the time-step column (`time_step` by default; a time
  /// column becomes time steps). The exports read numbers: a categorical
  /// input with [DocInput.categories] takes one-hot columns
  /// `<name>_<category>`, and choices are option codes.
  ExportColumnNames get columnNames {
    final single =
        inputs.length == 1 &&
        _inputWidth(network, document.network, network.inputIdxs.single) == 1;
    return (
      inputs: [
        for (var k = 0; k < inputs.length; k++)
          () {
            final input = inputs[k];
            final w = _inputWidth(
              network,
              document.network,
              network.inputIdxs[k],
            );
            final cats = input.categories;
            if (cats != null) {
              return w == 1
                  ? [input.columnsFor(1).single]
                  : [for (final c in cats) '${input.name}_$c'];
            }
            return input.columnsFor(w);
          }(),
      ],
      observed: [
        for (final input in inputs)
          input.observedColumn ??
              (single && input.name == 'observation'
                  ? 'observed'
                  : 'observed_${input.name}'),
      ],
      responses: document.response?.dataColumns ?? const [],
      timeStep: document.data.timeStepColumn ?? 'time_step',
    );
  }
}

/// How many data columns the input node [i] takes.
int _inputWidth(Network network, NetworkDocument doc, int i) =>
    switch (network.kindOf(i)) {
      NodeKind.categoricalState =>
        (doc.addresses.nodeAt(i) as CategoricalNode).nCategories,
      NodeKind.efState => (doc.addresses.nodeAt(i) as EfStateNode).dimension,
      _ => 1,
    };

/// [json] with [f] applied to the values of its `node`, `parent` and
/// `child` keys (node indices ↔ addresses), at any depth.
Object? _remap(Object? json, Object? Function(Object?) f) => switch (json) {
  final Map<String, Object?> m => {
    for (final MapEntry(:key, :value) in m.entries)
      key: key == 'node' || key == 'parent' || key == 'child'
          ? f(value)
          : _remap(value, f),
  },
  final List<Object?> l => [for (final x in l) _remap(x, f)],
  _ => json,
};
