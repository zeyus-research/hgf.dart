import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import '../codec.dart';
import '../codegen/export_spec.dart';
import '../codegen/plan.dart';
import '../expr_readout.dart';
import '../hierarchical.dart';
import '../model.dart';
import '../parameter.dart';
import '../prior.dart';
import '../transform.dart';
import 'data_layout.dart';
import 'network_document.dart';
import 'transforms.dart';

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

  /// The columns holding the recorded responses: none for a readout that
  /// scores the observations themselves (`ObservationSurprise`, a fit to
  /// the inputs), else [columns] or [name].
  List<String> get dataColumns =>
      readsData ? columns ?? [name] : const <String>[];

  /// Whether the readout scores recorded responses (every readout but
  /// `ObservationSurprise`).
  bool get readsData => readout['type'] != 'ObservationSurprise';

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
/// [groupColumn] says which group (e.g. team, per block) a participant's
/// row belongs to: the data transforms' `mate(x, k)` reads the k-th other
/// member of it, at the same time (the same [timeColumn] value, or without
/// one the same position among the participant's rows in that group).
final class DocData {
  const DocData({
    this.participantColumn,
    this.timeColumn,
    this.timeStepColumn,
    this.groupColumn,
  }) : assert(
         timeColumn == null || timeStepColumn == null,
         'Give a time column or a time-step column, not both',
       );

  final String? participantColumn, timeColumn, timeStepColumn, groupColumn;

  Map<String, Object?> toJson() => {
    'participant_column': ?participantColumn,
    'time_column': ?timeColumn,
    'time_step_column': ?timeStepColumn,
    'group_column': ?groupColumn,
  };

  factory DocData.fromJson(Map<String, Object?> json) => DocData(
    participantColumn: json['participant_column'] as String?,
    timeColumn: json['time_column'] as String?,
    timeStepColumn: json['time_step_column'] as String?,
    groupColumn: json['group_column'] as String?,
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

/// A pooled parameter's population: hyperpriors (`FitCodec` prior JSON) on
/// its mean [mean] and standard deviation [sd], both on the parameter's
/// unconstrained scale (see `Population`).
final class DocPopulation {
  DocPopulation({
    required Map<String, Object?> mean,
    required Map<String, Object?> sd,
  }) : mean = Map.unmodifiable(mean),
       sd = Map.unmodifiable(sd);

  /// The population a parameter with [prior] (and [transform], default the
  /// prior's) starts with: the mean `N(y₀, 1.5²)` around the prior's typical
  /// value on the unconstrained scale, the sd `HalfNormal(1)`. On a log
  /// scale, participants then mostly differ by up to a factor of e.
  factory DocPopulation.around(Prior prior, {Transform? transform}) {
    final t = transform ?? prior.defaultTransform;
    final y0 = t.toUnconstrained(prior.typical);
    return DocPopulation(
      mean: Normal(_round(y0), 1.5).toJson(),
      sd: const HalfNormal(1).toJson(),
    );
  }

  final Map<String, Object?> mean, sd;

  Population decode(FitCodec codec) =>
      Population(mean: codec.decodePrior(mean), sd: codec.decodePrior(sd));

  Map<String, Object?> toJson() => {'mean': mean, 'sd': sd};

  factory DocPopulation.fromJson(Map<String, Object?> json) => DocPopulation(
    mean: json['mean']! as Map<String, Object?>,
    sd: json['sd']! as Map<String, Object?>,
  );
}

double _round(double v) => v.isFinite ? (v * 1000).roundToDouble() / 1000 : v;

/// The group level over participants (see `HierarchicalModel`): each
/// [pooled] parameter (by name) varies by participant around a population
/// whose mean and sd are estimated too; each [shared] one takes one value
/// for everyone; the other parameters are fitted per participant alone.
/// [centered] picks the centred parametrisation for NUTS.
///
/// MAP fits a group with pooled parameters by empirical Bayes
/// (`fitEmpiricalBayes`, which can't share parameters), NUTS samples the
/// joint posterior (`sampleHierarchical`).
final class DocGroup {
  DocGroup({
    Map<String, DocPopulation> pooled = const {},
    Iterable<String> shared = const [],
    this.centered = false,
  }) : pooled = Map.unmodifiable(pooled),
       shared = List.unmodifiable(shared);

  static final empty = DocGroup();

  final Map<String, DocPopulation> pooled;
  final List<String> shared;
  final bool centered;

  bool get isEmpty => pooled.isEmpty && shared.isEmpty;

  /// How the parameter [name] is fitted.
  ParameterRole roleOf(String name) => pooled.containsKey(name)
      ? ParameterRole.pooled
      : shared.contains(name)
      ? ParameterRole.shared
      : ParameterRole.individual;

  /// This group with only the parameters in [names] (renamed by [rename]).
  DocGroup only(Set<String> names, [Map<String, String> rename = const {}]) {
    String r(String n) => rename[n] ?? n;
    return DocGroup(
      pooled: {
        for (final MapEntry(:key, :value) in pooled.entries)
          if (names.contains(r(key))) r(key): value,
      },
      shared: [
        for (final n in shared)
          if (names.contains(r(n))) r(n),
      ],
      centered: centered,
    );
  }

  Map<String, Object?> toJson() => {
    if (pooled.isNotEmpty)
      'pooled': {
        for (final MapEntry(:key, :value) in pooled.entries)
          key: value.toJson(),
      },
    if (shared.isNotEmpty) 'shared': shared,
    if (centered) 'centered': true,
  };

  factory DocGroup.fromJson(Map<String, Object?> json) => DocGroup(
    pooled: {
      for (final MapEntry(:key, :value)
          in ((json['pooled'] as Map<String, Object?>?) ?? const {}).entries)
        key: DocPopulation.fromJson(value! as Map<String, Object?>),
    },
    shared: ((json['shared'] as List<Object?>?) ?? const []).cast<String>(),
    centered: (json['centered'] as bool?) ?? false,
  );
}

/// How a free parameter is fitted across participants.
enum ParameterRole {
  /// One value per participant, with the parameter's own prior.
  individual,

  /// Per participant, around a population estimated with them.
  pooled,

  /// One value for all participants.
  shared,
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
    List<DocTransform> transforms = const [],
    this.data = const DocData(),
    this.fit = const DocFit(),
    DocGroup? group,
    Map<String, Object?> layout = const {},
  }) : group = group ?? DocGroup.empty,
       inputs = List.unmodifiable(inputs),
       parameters = List.unmodifiable(parameters),
       outputs = List.unmodifiable(outputs),
       transforms = List.unmodifiable(transforms),
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

  /// Columns computed from the data before filtering (see [DocTransform]),
  /// in order; inputs, the response and the data layout may name them.
  final List<DocTransform> transforms;
  final DocData data;
  final DocFit fit;

  /// The group level over participants: which parameters are pooled or
  /// shared (empty: each participant is fitted alone).
  final DocGroup group;
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
            remapNodeKeys(codec.encodeResponse(model.response), toAddress)
                as Map<String, Object?>,
      ),
      parameters: [
        for (final p in codec.encodeParameters(model.parameters))
          remapNodeKeys(p, toAddress) as Map<String, Object?>,
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
        idxs.length == 1 && inputWidth(network, doc, idxs.single) == 1;
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
      transforms: [
        for (final t in (json['transforms'] as List<Object?>?) ?? const [])
          DocTransform.fromJson(t as Map<String, Object?>),
      ],
      data: switch (json['data']) {
        final Map<String, Object?> d => DocData.fromJson(d),
        _ => const DocData(),
      },
      fit: switch (json['fit']) {
        final Map<String, Object?> f => DocFit.fromJson(f),
        _ => const DocFit(),
      },
      group: switch (json['group']) {
        final Map<String, Object?> g => DocGroup.fromJson(g),
        _ => null,
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
    if (transforms.isNotEmpty)
      'transforms': [for (final t in transforms) t.toJson()],
    if (data.toJson() case final d when d.isNotEmpty) 'data': d,
    'fit': fit.toJson(),
    if (!group.isEmpty) 'group': group.toJson(),
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
          remapNodeKeys(_bindBeliefs(r.readout, a), toIndex)
              as Map<String, Object?>,
        ),
        parameters: codec.decodeParameters([
          for (final p in parameters) remapNodeKeys(p, toIndex),
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
    HierarchicalModel? hierarchical;
    if (model != null && !group.isEmpty) {
      hierarchical = HierarchicalModel(
        model,
        pooled: {
          for (final MapEntry(:key, :value) in group.pooled.entries)
            key: value.decode(codec),
        },
        shared: group.shared,
        centered: group.centered,
      );
    }
    return CompiledDocument._(
      this,
      net,
      model,
      hierarchical,
      ordered,
      reveals,
      couplings,
    );
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
    this.hierarchical,
    List<DocInput> inputs,
    this.reveals,
    this._couplings,
  ) : inputs = List.unmodifiable(inputs);

  final Map<String, Coupling> _couplings;

  final ModelDocument document;
  final Network network;
  final HgfModel? model;

  /// The group-level model over participants, when the document pools or
  /// shares parameters ([ModelDocument.group]).
  final HierarchicalModel? hierarchical;
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

  /// [network] built anew, with the free parameters set to the constrained
  /// values [x] (in `model.parameters` order; default `model.start()`, the
  /// priors' typical values). Without a model, just the network.
  Network networkAt([List<double>? x]) {
    final net = document.network.build(couplings: _couplings);
    final m = model;
    if (m == null) return net;
    final values = x ?? m.start();
    if (values.length != m.dimension) {
      throw ArgumentError('x must have ${m.dimension} values');
    }
    for (final (i, p) in m.parameters.indexed) {
      for (final t in p.targets) {
        switch (t) {
          case NodeTarget(:final node, :final field):
            net.setInitial(node, field, values[i]);
          case ValueCouplingTarget(:final parent, :final child):
            net.setValueCoupling(parent, child, values[i]);
          case VolatilityCouplingTarget(:final parent, :final child):
            net.setVolatilityCoupling(parent, child, values[i]);
          case ResponseTarget():
        }
      }
    }
    return net;
  }

  /// The free parameters' values as the document sets them (in
  /// `model.parameters` order): each parameter's first target's value in
  /// the built network (a node's initial value, an edge's strength) or the
  /// response model. Unlike `model.start()` (the priors' typical values),
  /// these follow the builder's sliders.
  List<double> currentValues() {
    final m = model;
    if (m == null) return const [];
    final net = network;
    double strength(List<int> parents, List<double> strengths, int p) {
      final at = parents.indexOf(p);
      return at < 0 ? double.nan : strengths[at];
    }

    return [
      for (final p in m.parameters)
        switch (p.targets.first) {
          NodeTarget(:final node, :final field) => net.getInitial(node, field),
          ValueCouplingTarget(:final parent, :final child) => strength(
            net.valueParentsOf(child),
            net.valueParentStrengthsOf(child),
            parent,
          ),
          VolatilityCouplingTarget(:final parent, :final child) => strength(
            net.volatilityParentsOf(child),
            net.volatilityParentStrengthsOf(child),
            parent,
          ),
          ResponseTarget(:final name) =>
            m.response.parameters[name] ?? double.nan,
        },
    ];
  }

  /// [subject] filtered by [networkAt] ([x]), recording [recordedFields].
  Trajectories filter(Subject subject, [List<double>? x]) =>
      networkAt(x).inputData(
        subject.inputs,
        observed: subject.observed,
        timeSteps: subject.timeSteps,
        fields: recordedFields,
      )!;

  /// [subjects] with responses simulated by the model at [x] (one vector
  /// per subject; default `model.start()` for each): with [reveals], choices
  /// that decide what is observed (`HgfModel.simulateChoices`, which also
  /// sets the observed flags), else draws of the response model
  /// (`HgfModel.simulate`).
  List<Subject> simulate(
    List<Subject> subjects,
    math.Random rng, {
    List<List<double>>? x,
  }) {
    final m =
        model ?? (throw StateError('The document has no response to simulate'));
    final xs = x ?? [for (final _ in subjects) m.start()];
    if (reveals case final rv?) {
      return m.simulateChoices(subjects, xs, rng, reveals: rv);
    }
    final r = m.simulate(subjects, xs, rng);
    return [
      for (final (k, s) in subjects.indexed)
        Subject(
          inputs: s.inputs,
          observed: s.observed,
          timeSteps: s.timeSteps,
          responses: r[k],
          id: s.id,
        ),
    ];
  }

  /// [subjects] as a CSV this document reads back ([subjects] gives them
  /// again): the participant column (ids from [Subject.id], else 1, 2, …),
  /// the time column (cumulative time steps from 0) or the time-step
  /// column, each input's columns (labels for an input with categories, an
  /// empty cell where unobserved, or the flag in its observed column), and
  /// the response's columns (option labels with options; empty for a
  /// missed response). Columns the document does not read are not written.
  ///
  /// An input that reads a transform's column gets that column here; the
  /// transform must then go before the CSV is read back.
  CsvData toCsv(List<Subject> subjects) {
    final d = document.data;
    if (subjects.length > 1 && d.participantColumn == null) {
      throw ArgumentError(
        'Several subjects need a participant column (DocData.participantColumn)',
      );
    }
    final widths = [
      for (final i in network.inputIdxs)
        inputWidth(network, document.network, i),
    ];
    final width = widths.fold(0, (a, b) => a + b);
    final nIn = inputs.length;
    final header = <String>[];
    int add(String name) {
      final at = header.indexOf(name);
      if (at >= 0) return at;
      header.add(name);
      return header.length - 1;
    }

    final pCol = d.participantColumn == null ? null : add(d.participantColumn!);
    final tCol = d.timeColumn == null ? null : add(d.timeColumn!);
    final dtCol = d.timeStepColumn == null ? null : add(d.timeStepColumn!);
    final inputCols = [
      for (var k = 0; k < nIn; k++)
        [for (final c in inputs[k].columnsFor(widths[k])) add(c)],
    ];
    final flagCols = [
      for (final i in inputs)
        i.observedColumn == null ? null : add(i.observedColumn!),
    ];
    final r = document.response;
    final responseCols = r == null
        ? const <int>[]
        : [for (final c in r.dataColumns) add(c)];

    final rows = <List<String>>[];
    for (final (si, s) in subjects.indexed) {
      final nTime = s.inputs.length ~/ math.max(1, width);
      var time = 0.0;
      for (var t = 0; t < nTime; t++) {
        final row = List.filled(header.length, '');
        if (pCol != null) row[pCol] = '${s.id ?? si + 1}';
        final dt = s.timeSteps?[t] ?? 1.0;
        if (t > 0) time += dt;
        if (tCol != null) row[tCol] = _cell(time);
        if (dtCol != null) row[dtCol] = _cell(dt);
        var off = 0;
        for (var k = 0; k < nIn; k++) {
          final w = widths[k];
          final input = inputs[k];
          final seen = s.observed == null || s.observed![t * nIn + k] != 0;
          final base = t * width + off;
          final flag = flagCols[k];
          if (flag != null) row[flag] = seen ? '1' : '0';
          if (seen || flag != null) {
            if (input.categories case final cats?) {
              var code = 0;
              if (w == 1) {
                code = s.inputs[base].round();
              } else {
                for (var j = 1; j < w; j++) {
                  if (s.inputs[base + j] > s.inputs[base + code]) code = j;
                }
              }
              row[inputCols[k].single] = code >= 0 && code < cats.length
                  ? cats[code]
                  : '$code';
            } else {
              for (var j = 0; j < w; j++) {
                row[inputCols[k][j]] = _cell(s.inputs[base + j]);
              }
            }
          }
          off += w;
        }
        if (r != null && s.responses != null) {
          final nr = responseCols.length;
          for (var j = 0; j < nr; j++) {
            final v = s.responses![t * nr + j];
            if (v.isNaN) continue;
            final opts = r.options;
            row[responseCols[j]] =
                opts != null &&
                    v == v.roundToDouble() &&
                    v >= 0 &&
                    v < opts.length
                ? opts[v.toInt()]
                : _cell(v);
          }
        }
        rows.add(row);
      }
    }
    return CsvData(header, rows);
  }

  /// [data] (after the transforms) with a column per named output
  /// appended, row by row: each participant filtered at [x] (by
  /// participant id, in the order of first appearance; default the
  /// parameters' starting values). The data's own columns (e.g. its time
  /// stamps) stay, so the outputs line up with other recordings.
  CsvData withOutputs(CsvData data, {Map<String, List<double>>? x}) {
    final subs = subjects(data);
    final ids = layout.participantRows(transform(data)).keys.toList();
    for (final o in document.outputs) {
      if (data.header.contains(o.name) ||
          document.transforms.any((t) => t.name == o.name)) {
        throw FormatException(
          'The data already has a column "${o.name}"; rename the output',
        );
      }
    }
    return layout.withColumns(data, [
      for (final (k, id) in ids.indexed) outputs(filter(subs[k], x?[id])),
    ]);
  }

  /// How [subjects] reads a CSV: the named inputs' columns in the
  /// network's input order, the response's columns and options, the data
  /// layout and the transforms. The exported Dart scripts write it out.
  late final DataLayout layout = () {
    final r = document.response;
    final d = document.data;
    return DataLayout(
      inputs: [
        for (final (k, input) in inputs.indexed)
          () {
            final w = inputWidth(
              network,
              document.network,
              network.inputIdxs[k],
            );
            final cats = input.categories;
            return LayoutInput(
              input.name,
              columns: input.columnsFor(cats != null ? 1 : w),
              categories: cats,
              oneHot: cats != null && w > 1,
              observedColumn: input.observedColumn,
            );
          }(),
      ],
      responseColumns: r == null ? const [] : r.dataColumns,
      options: r?.options,
      participantColumn: d.participantColumn,
      timeColumn: d.timeColumn,
      timeStepColumn: d.timeStepColumn,
      groupColumn: d.groupColumn,
      transforms: document.transforms,
    );
  }();

  /// [csv] with the document's transforms' columns appended (see
  /// [DocTransform]); [subjects] reads this.
  CsvData transform(CsvData csv) => layout.transform(csv);

  /// The subjects in [data], read by the document's column names (see
  /// [DocInput], [DocResponse] and [DocData]), after the transforms.
  List<Subject> subjects(CsvData data) {
    for (final (k, input) in inputs.indexed) {
      final w = inputWidth(network, document.network, network.inputIdxs[k]);
      final cats = input.categories;
      final n = input.columnsFor(cats != null ? 1 : w).length;
      if (n != (cats != null ? 1 : w)) {
        throw FormatException(
          'Input "${input.name}" takes ${cats != null ? 1 : w} column(s), '
          'not $n',
        );
      }
      if (cats != null && w > 1 && cats.length != w) {
        throw FormatException(
          'Input "${input.name}" has ${cats.length} categories; its node '
          'has $w',
        );
      }
    }
    return layout.subjects(data);
  }

  /// What the code generators take: the model (or the network alone), the
  /// document's fit, [data] to embed, and the document's column names (see
  /// [columnNames]).
  ExportSpec exportSpec({
    FitCodec codec = const FitCodec(),
    ExportData? data,
    String? dataFile,
    String? csv,
  }) {
    final m = model;
    final fits = m != null && m.dimension > 0;
    final a = document.network.addresses;
    return ExportSpec(
      network: network.toJson(),
      model: m == null ? null : codec.encodeModel(m),
      data: data,
      dataFile: dataFile,
      fit: fits ? document.fit.exportFit : null,
      title: document.name,
      columnNames: columnNames,
      derivedColumns: {
        for (final t in document.transforms) t.name: '${t.expression}',
      },
      layout: layout,
      csv: csv,
      group: fits && document.fit.exportFit != null && !document.group.isEmpty
          ? document.group
          : null,
      outputs: [
        for (final o in document.outputs)
          (name: o.name, node: a.indexOf(o.node), field: o.field),
      ],
    );
  }

  /// The CSV columns of the exports: the named inputs' columns, their
  /// observed-flag columns (`observed_<name>` by default), the response's
  /// columns and the time-step column (`time_step` by default; a time
  /// column becomes time steps). The exports read numbers: a categorical
  /// input with [DocInput.categories] takes one-hot columns
  /// `<name>_<category>`, and choices are option codes.
  ExportColumnNames get columnNames {
    final single =
        inputs.length == 1 &&
        inputWidth(network, document.network, network.inputIdxs.single) == 1;
    return (
      inputs: [
        for (var k = 0; k < inputs.length; k++)
          () {
            final input = inputs[k];
            final w = inputWidth(
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

/// An `ExprReadout`'s JSON with a belief for each variable written
/// `<address>.<field>` (`ball.expected_mean`, `cat.2.mean`) that isn't
/// bound already; other readouts as they are.
Map<String, Object?> _bindBeliefs(
  Map<String, Object?> readout,
  NodeAddresses a,
) {
  if (readout['type'] != 'ExprReadout') return readout;
  final beliefs = {
    ...((readout['beliefs'] as Map<String, Object?>?) ?? const {}),
  };
  final parameters =
      (readout['parameters'] as Map<String, Object?>?) ?? const {};
  final sources = switch (readout['kind']) {
    'choice' => (readout['options'] as List<Object?>).cast<String>(),
    'gaussian' => [
      for (final v in (readout['gaussian'] as Map<String, Object?>).values)
        v as String,
    ],
    _ => [readout['surprise'] as String],
  };
  for (final src in sources) {
    for (final v in Expr.parse(src).variables) {
      if (beliefs.containsKey(v) ||
          parameters.containsKey(v) ||
          v == ExprReadout.responseVariable) {
        continue;
      }
      final dot = v.lastIndexOf('.');
      if (dot < 0) continue;
      final field = v.substring(dot + 1);
      final address = v.substring(0, dot);
      try {
        NodeField.parse(field);
        a.indexOf(address);
      } on Object {
        throw FormatException(
          'The response reads "$v", which is not a belief (node.field, e.g. '
          '"x.expected_mean"), a parameter or "response"',
        );
      }
      beliefs[v] = {'node': address, 'field': field};
    }
  }
  return {...readout, 'beliefs': beliefs};
}

/// [v] for a CSV cell: integers without a decimal point, NaN as empty.
String _cell(double v) => v.isNaN
    ? ''
    : v == v.roundToDouble() && v.abs() < 1e15
    ? '${v.toInt()}'
    : '$v';

/// How many data columns the input node [i] takes.
int inputWidth(Network network, NetworkDocument doc, int i) =>
    switch (network.kindOf(i)) {
      NodeKind.categoricalState =>
        (doc.addresses.nodeAt(i) as CategoricalNode).nCategories,
      NodeKind.efState => (doc.addresses.nodeAt(i) as EfStateNode).dimension,
      _ => 1,
    };

/// [json] with [f] applied to the values of its `node`, `parent` and
/// `child` keys (node indices ↔ addresses), at any depth.
Object? remapNodeKeys(Object? json, Object? Function(Object?) f) =>
    switch (json) {
      final Map<String, Object?> m => {
        for (final MapEntry(:key, :value) in m.entries)
          key: key == 'node' || key == 'parent' || key == 'child'
              ? f(value)
              : remapNodeKeys(value, f),
      },
      final List<Object?> l => [for (final x in l) remapNodeKeys(x, f)],
      _ => json,
    };
