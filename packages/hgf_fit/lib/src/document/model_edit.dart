import 'package:hgf/hgf.dart';

import '../codec.dart';
import 'edit.dart';
import 'model_document.dart';
import 'network_document.dart';
import 'parameters.dart';
import 'transforms.dart';

/// The parts of a [ModelDocument] an issue can name ([DocIssue.parts]): the
/// builder's pipeline nodes.
abstract final class DocPart {
  static const network = 'network', data = 'data', response = 'response';

  /// The free parameters, the group level and the fit.
  static const fit = 'fit';
  static String input(String name) => 'input:$name';
  static String transform(String name) => 'transform:$name';
  static String output(String name) => 'output:$name';
}

/// Identifiers: column, input, transform and output names that expressions
/// read without backticks.
final _identifier = RegExp(r'^[A-Za-z_][A-Za-z0-9_]*$');

/// Whether [name] can be read by an expression as it is.
bool isPlainName(String name) => _identifier.hasMatch(name);

/// Edits of a [ModelDocument], as the model builder makes them. Each
/// returns a new document, with the references between its parts kept
/// consistent (a renamed node or column is renamed where it is read).
extension ModelDocumentEdits on ModelDocument {
  /// This document with the given parts replaced; [clearResponse] removes
  /// the response.
  ModelDocument copyWith({
    String? name,
    String? description,
    NetworkDocument? network,
    List<DocInput>? inputs,
    DocResponse? response,
    bool clearResponse = false,
    List<Map<String, Object?>>? parameters,
    List<DocOutput>? outputs,
    List<DocTransform>? transforms,
    DocData? data,
    DocFit? fit,
    DocGroup? group,
    Map<String, Object?>? layout,
  }) => ModelDocument(
    name: name ?? this.name,
    description: description ?? this.description,
    network: network ?? this.network,
    inputs: inputs ?? this.inputs,
    response: clearResponse ? null : response ?? this.response,
    parameters: parameters ?? this.parameters,
    outputs: outputs ?? this.outputs,
    transforms: transforms ?? this.transforms,
    data: data ?? this.data,
    fit: fit ?? this.fit,
    group: group ?? this.group,
    layout: layout ?? this.layout,
  );

  /// Every name a column can be read by: the transforms' names (the data's
  /// own columns are only known with the data).
  Set<String> get transformNames => {for (final t in transforms) t.name};

  /// A name starting with [base] that no input, transform or output uses
  /// (nor any of [taken]).
  String freshName(String base, [Iterable<String> taken = const []]) {
    final used = {
      for (final i in inputs) i.name,
      for (final t in transforms) t.name,
      for (final o in outputs) o.name,
      ?response?.name,
      ...taken,
    };
    if (!used.contains(base)) return base;
    for (var k = 1; ; k++) {
      if (!used.contains('${base}_$k')) return '${base}_$k';
    }
  }

  /// With [next] as the network. [renamedIds] maps old node ids to new ones
  /// (a rename in the builder); inputs, outputs, the response and the
  /// parameters follow. Outputs and parameters that address a node that is
  /// gone are dropped; inputs are matched to the new network's input nodes
  /// by [withInputsFor].
  ModelDocument withNetwork(
    NetworkDocument next, {
    Map<String, String> renamedIds = const {},
  }) {
    String? map(String address) {
      final dot = address.indexOf('.');
      final owner = dot < 0 ? address : address.substring(0, dot);
      final k = dot < 0 ? 0 : int.tryParse(address.substring(dot + 1));
      final id = renamedIds[owner] ?? owner;
      final node = next.node(id);
      if (node == null || k == null || k < 0 || k >= node.size) return null;
      return k == 0 ? id : '$id.$k';
    }

    final renamed = renamedIds.isNotEmpty;
    final params = <Map<String, Object?>>[];
    for (final p in parameters) {
      var gone = false;
      final q = remapNodeKeys(p, (v) {
        if (v is! String) return v;
        final a = map(v);
        if (a == null) gone = true;
        return a ?? v;
      });
      if (!gone) params.add(q! as Map<String, Object?>);
    }
    return copyWith(
      network: next,
      inputs: !renamed
          ? null
          : [
              for (final i in inputs)
                DocInput(
                  i.name,
                  node: map(i.node) ?? i.node,
                  columns: i.columns,
                  categories: i.categories,
                  observedColumn: i.observedColumn,
                ),
            ],
      outputs: [
        for (final o in outputs)
          if (map(o.node) case final a?)
            DocOutput(o.name, node: a, field: o.field),
      ],
      response: switch (response) {
        final r? when renamed => DocResponse(
          r.name,
          readout: renameReadout(r.readout, (a) => map(a) ?? a),
          columns: r.columns,
          options: r.options,
          reveals: r.reveals,
        ),
        _ => null,
      },
      parameters: params,
      group: group.only({for (final p in params) p['name']! as String}),
    );
  }

  /// The inputs matched to [built]'s input nodes ([built] is [network],
  /// built): an input of a node that is no longer an input is dropped, and
  /// an input node without one gets one named after the node (its label or
  /// id). Returns this when nothing changes.
  ModelDocument withInputsFor(Network built) {
    final a = network.addresses;
    final inputIdx = built.inputIdxs.toSet();
    final kept = <DocInput>[];
    final covered = <int>{};
    for (final i in inputs) {
      final int idx;
      try {
        idx = a.indexOf(i.node);
      } on Object {
        continue;
      }
      if (!inputIdx.contains(idx) || !covered.add(idx)) continue;
      kept.add(i);
    }
    final added = <DocInput>[];
    for (final idx in built.inputIdxs) {
      if (covered.contains(idx)) continue;
      final address = a.addressOf(idx);
      final node = a.nodeAt(idx);
      final base = switch (node.label) {
        final l? when isPlainName(l) => l,
        _ => address.replaceAll('.', '_'),
      };
      added.add(
        DocInput(
          freshName(base, [
            for (final i in [...kept, ...added]) i.name,
          ]),
          node: address,
        ),
      );
    }
    if (added.isEmpty && kept.length == inputs.length) return this;
    return copyWith(inputs: [...kept, ...added]);
  }

  // ---------------------------------------------------------------------------
  // Inputs
  // ---------------------------------------------------------------------------

  /// The input named [name], or null.
  DocInput? input(String name) {
    for (final i in inputs) {
      if (i.name == name) return i;
    }
    return null;
  }

  /// With the input [name] replaced by [input].
  ModelDocument replaceInput(String name, DocInput input) {
    final at = inputs.indexWhere((i) => i.name == name);
    if (at < 0) throw ArgumentError.value(name, 'name', 'No such input');
    final renamed = input.name != name;
    final r = response;
    return copyWith(
      inputs: [...inputs]..[at] = input,
      response: renamed && r?.reveals != null
          ? DocResponse(
              r!.name,
              readout: r.readout,
              columns: r.columns,
              options: r.options,
              reveals: {
                for (final MapEntry(:key, :value) in r.reveals!.entries)
                  key: [for (final v in value) v == name ? input.name : v],
              },
            )
          : null,
    );
  }

  // ---------------------------------------------------------------------------
  // Transforms
  // ---------------------------------------------------------------------------

  /// The transform named [name], or null.
  DocTransform? transform(String name) {
    for (final t in transforms) {
      if (t.name == name) return t;
    }
    return null;
  }

  /// With [t] added after the others.
  ModelDocument withTransform(DocTransform t) =>
      copyWith(transforms: [...transforms, t]);

  /// With the transform [name] replaced by [t]; a new name is read by
  /// whatever read the old one ([renameColumn]).
  ModelDocument replaceTransform(String name, DocTransform t) {
    final at = transforms.indexWhere((x) => x.name == name);
    if (at < 0) throw ArgumentError.value(name, 'name', 'No such transform');
    final next = copyWith(transforms: [...transforms]..[at] = t);
    return t.name == name ? next : next.renameColumn(name, t.name);
  }

  ModelDocument withoutTransform(String name) => copyWith(
    transforms: [
      for (final t in transforms)
        if (t.name != name) t,
    ],
  );

  /// Every reference to the column [from] (inputs, observed flags, the
  /// response's columns, the data layout and the transforms' expressions)
  /// renamed to [to].
  ModelDocument renameColumn(String from, String to) {
    String c(String s) => s == from ? to : s;
    String? co(String? s) => s == null ? null : c(s);
    final r = response;
    final d = data;
    return copyWith(
      inputs: [
        for (final i in inputs)
          DocInput(
            i.name,
            node: i.node,
            columns: i.columns?.map(c).toList(),
            categories: i.categories,
            observedColumn: co(i.observedColumn),
          ),
      ],
      transforms: [
        for (final t in transforms)
          DocTransform.of(c(t.name), t.expression.renamed(c)),
      ],
      response: r == null
          ? null
          : DocResponse(
              r.name,
              readout: r.readout,
              columns: r.columns?.map(c).toList(),
              options: r.options,
              reveals: r.reveals,
            ),
      data: DocData(
        participantColumn: co(d.participantColumn),
        timeColumn: co(d.timeColumn),
        timeStepColumn: co(d.timeStepColumn),
        groupColumn: co(d.groupColumn),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Outputs
  // ---------------------------------------------------------------------------

  ModelDocument withOutput(DocOutput o) => copyWith(outputs: [...outputs, o]);

  ModelDocument replaceOutput(String name, DocOutput o) {
    final at = outputs.indexWhere((x) => x.name == name);
    if (at < 0) throw ArgumentError.value(name, 'name', 'No such output');
    return copyWith(outputs: [...outputs]..[at] = o);
  }

  ModelDocument withoutOutput(String name) => copyWith(
    outputs: [
      for (final o in outputs)
        if (o.name != name) o,
    ],
  );

  /// With [r] as the response (null removes it, and the parameters of the
  /// old response's readout with it).
  ModelDocument withResponse(DocResponse? r) {
    if (r != null) return copyWith(response: r);
    return withParameters([
      for (final p in parameters)
        if (!_targetsResponse(p)) p,
    ]).copyWith(clearResponse: true);
  }
}

bool _targetsResponse(Map<String, Object?> p) => switch (p['targets']) {
  final List<Object?> ts => ts.any(
    (t) => t is Map<String, Object?> && t['type'] == 'response',
  ),
  _ => false,
};

/// [readout] (`FitCodec` JSON with addresses) with every node address
/// mapped by [map]: the `node`, `parent` and `child` keys, and an
/// `ExprReadout`'s `<address>.<field>` variables.
Map<String, Object?> renameReadout(
  Map<String, Object?> readout,
  String Function(String address) map,
) {
  final out =
      remapNodeKeys(readout, (v) => v is String ? map(v) : v)!
          as Map<String, Object?>;
  if (out['type'] != 'ExprReadout') return out;
  String variable(String v) {
    final dot = v.lastIndexOf('.');
    if (dot < 0) return v;
    final field = v.substring(dot + 1);
    try {
      NodeField.parse(field);
    } on Object {
      return v;
    }
    return '${map(v.substring(0, dot))}.$field';
  }

  String expr(Object? s) => '${Expr.parse(s! as String).renamed(variable)}';
  return {
    ...out,
    if (out['options'] case final List<Object?> o)
      'options': [for (final s in o) expr(s)],
    if (out['gaussian'] case final Map<String, Object?> g)
      'gaussian': {
        for (final MapEntry(:key, :value) in g.entries) key: expr(value),
      },
    if (out['surprise'] case final String s) 'surprise': expr(s),
    if (out['beliefs'] case final Map<String, Object?> b)
      'beliefs': {
        for (final MapEntry(:key, :value) in b.entries) variable(key): value,
      },
  };
}

/// What is wrong with [doc] as a whole, errors first: the network's issues
/// ([checkNetwork], each also naming [DocPart.network]), then inputs that
/// name no input node or take the wrong number of columns, names used
/// twice, transforms that read unknown columns, a response that does not
/// decode, outputs that name no node or field, and, if all that passes,
/// whatever compiling throws. With the data's [columns], every column the
/// document reads must be one of them or a transform's.
///
/// [built] is the network built from `doc.network` when the caller has it.
List<DocIssue> checkModel(
  ModelDocument doc, {
  List<String>? columns,
  Network? built,
  FitCodec codec = const FitCodec(),
}) {
  final issues = <DocIssue>[
    for (final i in checkNetwork(doc.network, build: built == null))
      i.inParts(const [DocPart.network]),
  ];
  void error(String m, List<String> parts, {List<String> nodes = const []}) =>
      issues.add(DocIssue(IssueSeverity.error, m, parts: parts, nodes: nodes));
  void warn(String m, List<String> parts) =>
      issues.add(DocIssue(IssueSeverity.warning, m, parts: parts));

  final networkOk = !issues.any((i) => i.isError);
  Network? net = built;
  if (net == null && networkOk) {
    try {
      net = doc.network.build();
    } on Object {
      net = null;
    }
  }
  final a = doc.network.addresses;
  int? indexOf(String address) {
    try {
      return a.indexOf(address);
    } on Object {
      return null;
    }
  }

  // Names: inputs, transforms and outputs each by name.
  final names = <String, String>{};
  void name(String n, String what, String part) {
    if (n.trim().isEmpty) {
      error('A $what has no name.', [part]);
    } else if (names[n] case final other?) {
      error('"$n" names both $other and $what.', [part]);
    } else {
      names[n] = what;
    }
  }

  for (final i in doc.inputs) {
    name(i.name, 'an input', DocPart.input(i.name));
  }
  for (final t in doc.transforms) {
    name(t.name, 'a transform', DocPart.transform(t.name));
  }
  for (final o in doc.outputs) {
    name(o.name, 'an output', DocPart.output(o.name));
  }

  // Columns the document can read.
  final known = columns == null ? null : {...columns};
  final transformNames = <String>{};
  for (final t in doc.transforms) {
    final part = DocPart.transform(t.name);
    if (known != null) {
      if (columns!.contains(t.name)) {
        error(
          'The data already has a column "${t.name}"; name the transform '
          'otherwise.',
          [part],
        );
      }
      for (final v in t.expression.variables) {
        if (!known.contains(v) && !transformNames.contains(v)) {
          error(
            'Transform "${t.name}" reads "$v", which is neither a column nor '
            'an earlier transform.',
            [part],
          );
        }
      }
    }
    transformNames.add(t.name);
  }
  void needColumn(String c, String what, String part) {
    if (known == null) return;
    if (!known.contains(c) && !transformNames.contains(c)) {
      error('No column "$c" ($what) in the data.', [part]);
    }
  }

  // Inputs.
  if (net != null) {
    final byIndex = <int, String>{};
    for (final i in doc.inputs) {
      final part = DocPart.input(i.name);
      final idx = indexOf(i.node);
      if (idx == null || !net.inputIdxs.contains(idx)) {
        error(
          'Input "${i.name}": "${i.node}" is not an input node.',
          [part],
          nodes: [?a.ownerIdOf(i.node)],
        );
        continue;
      }
      if (byIndex[idx] case final other?) {
        error('Inputs "$other" and "${i.name}" both feed "${i.node}".', [
          part,
          DocPart.input(other),
        ]);
      }
      byIndex[idx] = i.name;
      final w = inputWidth(net, doc.network, idx);
      final cols = i.columnsFor(w);
      final want = i.categories != null ? 1 : w;
      if (cols.length != want) {
        error(
          'Input "${i.name}" reads ${cols.length} column(s); its node takes '
          '$want.',
          [part],
        );
      }
      if (i.categories case final cats? when w > 1 && cats.length != w) {
        error(
          'Input "${i.name}" has ${cats.length} categories; its node has $w.',
          [part],
        );
      }
      for (final c in cols) {
        needColumn(c, 'input ${i.name}', part);
      }
      if (i.observedColumn case final c?) {
        needColumn(c, 'observed flags of ${i.name}', part);
      }
    }
    for (final idx in net.inputIdxs) {
      if (!byIndex.containsKey(idx)) {
        final address = a.addressOf(idx);
        error(
          'The input node "$address" reads no data.',
          [DocPart.network],
          nodes: [?a.ownerIdOf(address)],
        );
      }
    }
  }

  // Response.
  Set<String>? responseParameters;
  if (doc.response case final r?) {
    const part = DocPart.response;
    try {
      final readout = renameReadout(r.readout, (s) {
        if (indexOf(s) == null) {
          throw FormatException('"$s" is not a node of the network');
        }
        return s;
      });
      if (net != null) {
        responseParameters = codec
            .decodeResponse(
              remapNodeKeys(readout, (v) => v is String ? indexOf(v) : v)!
                  as Map<String, Object?>,
            )
            .parameters
            .keys
            .toSet();
      }
    } on Object catch (e) {
      error('The response: ${buildErrorMessage(e)}', [part]);
    }
    for (final c in r.dataColumns) {
      needColumn(c, 'response ${r.name}', part);
    }
    if (r.reveals case final rv?) {
      if (r.options == null) {
        error('The response reveals inputs but names no options.', [part]);
      }
      final inputNames = {for (final i in doc.inputs) i.name};
      for (final MapEntry(key: o, value: ins) in rv.entries) {
        if (r.options != null && !r.options!.contains(o)) {
          error('The response reveals inputs for "$o", not an option.', [part]);
        }
        for (final i in ins) {
          if (!inputNames.contains(i)) {
            error('Option "$o" reveals "$i", which is not an input.', [part]);
          }
        }
      }
    }
  }
  issues.addAll(
    checkParameters(doc, responseParameters: responseParameters, codec: codec),
  );

  // Outputs.
  for (final o in doc.outputs) {
    final part = DocPart.output(o.name);
    if (indexOf(o.node) == null) {
      error('Output "${o.name}" reads "${o.node}", which is not a node.', [
        part,
      ]);
    }
    if (o.field != 'surprise') {
      try {
        NodeField.parse(o.field);
      } on Object {
        error('Output "${o.name}": "${o.field}" is not a node field.', [part]);
      }
    }
  }

  // Data layout.
  final d = doc.data;
  for (final (c, what) in [
    (d.participantColumn, 'participants'),
    (d.timeColumn, 'time'),
    (d.timeStepColumn, 'time steps'),
    (d.groupColumn, 'groups'),
  ]) {
    if (c != null) needColumn(c, what, DocPart.data);
  }
  if (d.timeColumn != null && d.timeStepColumn != null) {
    error('Give a time column or a time-step column, not both.', [
      DocPart.data,
    ]);
  }
  if (d.groupColumn != null && d.participantColumn == null) {
    warn('A group column needs a participant column.', [DocPart.data]);
  }

  if (!issues.any((i) => i.isError)) {
    try {
      doc.compile(codec: codec);
    } on Object catch (e) {
      error(buildErrorMessage(e), const []);
    }
  }
  issues.sort((x, y) => x.severity.index - y.severity.index);
  return issues;
}

extension on NodeAddresses {
  /// The id owning [address], if it names a node.
  String? ownerIdOf(String address) {
    try {
      indexOf(address);
    } on Object {
      return null;
    }
    final dot = address.indexOf('.');
    return dot < 0 ? address : address.substring(0, dot);
  }
}
