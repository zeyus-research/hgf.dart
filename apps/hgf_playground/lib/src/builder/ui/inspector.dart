import 'dart:math' as math;

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/codegen.dart' show AddCategoricalState;
import 'package:hgf_fit/document.dart';
import 'package:material_ui/material_ui.dart';

import '../../model/presets.dart' show paramsFor, ParamDef;
import '../../ui/panels.dart' show LabeledSlider, SectionCard, fmt;
import '../builder_model.dart';
import '../doc_layout.dart';

/// Ids are identifiers: expressions and addresses read them.
final _idPattern = RegExp(r'^[A-Za-z_][A-Za-z0-9_]*$');

/// [n] with the id [id].
DocNode withId(DocNode n, String id) => switch (n) {
  ScalarNode() => ScalarNode(
    id,
    kind: n.kind,
    label: n.label,
    coupling: n.coupling,
    customCoupling: n.customCoupling,
    params: n.params,
  ),
  CategoricalNode() => CategoricalNode(
    id,
    nCategories: n.nCategories,
    label: n.label,
    params: n.params,
  ),
  EfStateNode() => EfStateNode(
    id,
    label: n.label,
    nNodes: n.nNodes,
    dimension: n.dimension,
    learning: n.learning,
    nus: n.nus,
    xis: n.xis,
  ),
  DpStateNode() => DpStateNode(
    id,
    label: n.label,
    batchSize: n.batchSize,
    alpha: n.alpha,
    sensoryPrecision: n.sensoryPrecision,
  ),
};

/// The side panel of the builder: what is selected (a node's kind,
/// parameters, coupling and edges; an edge's type and strength), or the
/// network's settings and a palette when nothing is; and the checks'
/// issues.
class BuilderInspector extends StatelessWidget {
  const BuilderInspector({super.key, required this.model, this.shrinkWrap});

  final BuilderModel model;

  /// Lay out as a column (inside another scroll view) instead of a list.
  final bool? shrinkWrap;

  @override
  Widget build(BuildContext context) => ListenableBuilder(
    listenable: model,
    builder: (context, _) {
      final children = [
        switch (model.selection) {
          NodeSelection(:final id) when model.doc.node(id) != null =>
            _NodePanel(
              key: ValueKey('node $id'),
              model: model,
              node: model.doc.node(id)!,
            ),
          EdgeSelection(:final index) when index < model.doc.edges.length =>
            _EdgePanel(
              key: ValueKey('edge $index'),
              model: model,
              index: index,
            ),
          _ => _NetworkPanel(model: model),
        },
        if (model.issues.isNotEmpty) ...[
          const SizedBox(height: 12),
          _IssuesCard(model: model),
        ],
        const SizedBox(height: 12),
        const _HelpCard(),
      ];
      if (shrinkWrap ?? false) {
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: children,
        );
      }
      return ListView(padding: const EdgeInsets.all(12), children: children);
    },
  );
}

// -----------------------------------------------------------------------------
// Network
// -----------------------------------------------------------------------------

class _NetworkPanel extends StatelessWidget {
  const _NetworkPanel({required this.model});

  final BuilderModel model;

  @override
  Widget build(BuildContext context) {
    final doc = model.doc;
    final theme = Theme.of(context);
    final net = model.network;
    return SectionCard(
      title: 'Network',
      children: [
        Text(
          '${doc.nodes.length} node${doc.nodes.length == 1 ? '' : 's'}, '
          '${doc.edges.length} edge${doc.edges.length == 1 ? '' : 's'}'
          '${net == null ? '' : '; ${net.nNodes} in the engine, '
                    '${net.inputIdxs.length} input${net.inputIdxs.length == 1 ? '' : 's'}'}',
          style: theme.textTheme.bodySmall,
        ),
        const SizedBox(height: 10),
        Text('Add a node', style: theme.textTheme.labelMedium),
        const SizedBox(height: 4),
        Wrap(
          spacing: 6,
          runSpacing: 6,
          children: [
            for (final kind in NodeKind.values)
              ActionChip(
                avatar: Icon(_kindIcon(kind), size: 16),
                label: Text(kindName(kind)),
                onPressed: () => model.addNode(kind),
              ),
          ],
        ),
        const SizedBox(height: 12),
        Text('Updates', style: theme.textTheme.labelMedium),
        const SizedBox(height: 6),
        DropdownMenu<VolatilityUpdates>(
          initialSelection: doc.volatilityUpdates,
          expandedInsets: EdgeInsets.zero,
          label: const Text('Volatility update'),
          onSelected: (v) {
            if (v != null && v != doc.volatilityUpdates) {
              model.setSettings(volatilityUpdates: v);
            }
          },
          dropdownMenuEntries: [
            for (final v in VolatilityUpdates.values)
              DropdownMenuEntry(value: v, label: v.pyhgfName),
          ],
        ),
        SwitchListTile(
          contentPadding: EdgeInsets.zero,
          dense: true,
          title: const Text('Mean-field value updates'),
          subtitle: const Text('Off: relaxed (rHGF) updates'),
          value: doc.meanFieldUpdates,
          onChanged: (v) => model.setSettings(meanFieldUpdates: v),
        ),
        SwitchListTile(
          contentPadding: EdgeInsets.zero,
          dense: true,
          title: const Text('eHGF MGF correction'),
          subtitle: const Text('Only changes relaxed eHGF updates'),
          value: doc.ehgfMgfCorrection,
          onChanged: (v) => model.setSettings(ehgfMgfCorrection: v),
        ),
      ],
    );
  }
}

IconData _kindIcon(NodeKind kind) => switch (kind) {
  NodeKind.continuousState => Icons.circle_outlined,
  NodeKind.binaryState => Icons.crop_square,
  NodeKind.constantState => Icons.fiber_manual_record_outlined,
  NodeKind.efState => Icons.change_history,
  NodeKind.categoricalState => Icons.hexagon_outlined,
  NodeKind.dpState => Icons.bubble_chart_outlined,
};

// -----------------------------------------------------------------------------
// Node
// -----------------------------------------------------------------------------

class _NodePanel extends StatelessWidget {
  const _NodePanel({super.key, required this.model, required this.node});

  final BuilderModel model;
  final DocNode node;

  bool get _isInput =>
      !model.doc.edges.any((e) => ownerIdOf(e.parent) == node.id);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final kind = docKind(node);
    return SectionCard(
      title:
          '${prettyId(node.id)} · ${kindName(kind)}${_isInput ? ' input' : ''}',
      action: IconButton(
        tooltip: 'Delete this node (Del)',
        icon: const Icon(Icons.delete_outline, size: 20),
        onPressed: () => model.removeNode(node.id),
      ),
      children: [
        _IdField(model: model, node: node),
        const SizedBox(height: 8),
        ...switch (node) {
          final ScalarNode n => _scalar(context, n),
          final CategoricalNode n => _categorical(context, n),
          final EfStateNode n => _ef(context, n),
          final DpStateNode n => _dp(context, n),
        },
        const SizedBox(height: 8),
        Text('Edges', style: theme.textTheme.labelMedium),
        ..._edges(context),
      ],
    );
  }

  List<Widget> _scalar(BuildContext context, ScalarNode n) {
    final defs = paramsFor(n.kind, input: _isInput);
    return [
      SegmentedButton<NodeKind>(
        showSelectedIcon: false,
        segments: const [
          ButtonSegment(
            value: NodeKind.continuousState,
            label: Text('continuous'),
          ),
          ButtonSegment(value: NodeKind.binaryState, label: Text('binary')),
          ButtonSegment(value: NodeKind.constantState, label: Text('bias')),
        ],
        selected: {n.kind},
        onSelectionChanged: (s) => model.replaceNode(
          n.id,
          ScalarNode(n.id, kind: s.single, label: n.label),
        ),
      ),
      for (final d in defs) _paramSlider(model, n.id, d),
      if (n.kind == NodeKind.continuousState) ...[
        const SizedBox(height: 10),
        _CouplingField(model: model, node: n),
      ],
      if (n.kind == NodeKind.constantState)
        Padding(
          padding: const EdgeInsets.only(top: 8),
          child: Text(
            'A constant 1, added to its value children\'s predictions; '
            'scale it with the edge strength.',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ),
    ];
  }

  List<Widget> _categorical(BuildContext context, CategoricalNode n) => [
    _Stepper(
      label: 'Categories',
      value: n.nCategories,
      min: 2,
      max: 12,
      onChanged: (v) => model.replaceNode(
        n.id,
        CategoricalNode(n.id, nCategories: v, label: n.label, params: n.params),
      ),
    ),
    const SizedBox(height: 6),
    Text(
      'Each category has a binary HGF (${n.id}.1 to ${n.id}.${n.nCategories}, '
      'their values ${n.id}.${n.nCategories + 1} to ${n.id}.${2 * n.nCategories}) '
      'sharing a volatility parent (${n.id}.${2 * n.nCategories + 1}). Empty '
      'fields keep pyhgf\'s defaults.',
      style: Theme.of(context).textTheme.bodySmall,
    ),
    for (final MapEntry(key: dart, value: key)
        in AddCategoricalState.pyhgfKeys.entries)
      _NumberField(
        key: ValueKey('${n.id} $key'),
        label: '$key  ($dart)',
        value: n.params[key],
        onSubmitted: (v) => model.replaceNode(
          n.id,
          CategoricalNode(
            n.id,
            nCategories: n.nCategories,
            label: n.label,
            params: {...n.params, key: ?v}
              ..removeWhere((k, _) => v == null && k == key),
          ),
        ),
      ),
  ];

  List<Widget> _ef(BuildContext context, EfStateNode n) {
    EfStateNode copy({
      int? nNodes,
      int? dimension,
      EfLearning? learning,
      double? nus,
    }) => EfStateNode(
      n.id,
      label: n.label,
      nNodes: nNodes ?? n.nNodes,
      dimension: dimension ?? n.dimension,
      learning: learning ?? n.learning,
      nus: nus ?? n.nus,
      xis: dimension == null ? n.xis : null,
    );
    return [
      _Stepper(
        label: 'Nodes in the group',
        value: n.nNodes,
        min: 1,
        max: 32,
        onChanged: (v) => model.replaceNode(n.id, copy(nNodes: v)),
      ),
      _Stepper(
        label: 'Dimension',
        value: n.dimension,
        min: 1,
        max: 8,
        onChanged: (v) => model.replaceNode(n.id, copy(dimension: v)),
      ),
      const SizedBox(height: 6),
      DropdownMenu<EfLearning>(
        initialSelection: n.learning,
        expandedInsets: EdgeInsets.zero,
        label: const Text('Learning rate'),
        onSelected: (v) {
          if (v != null) model.replaceNode(n.id, copy(learning: v));
        },
        dropdownMenuEntries: [
          for (final l in EfLearning.values)
            DropdownMenuEntry(
              value: l,
              label: l == EfLearning.fixed
                  ? 'fixed (ν)'
                  : 'HGF, ${l.levels} level${l.levels == 1 ? '' : 's'}',
            ),
        ],
      ),
      if (n.learning == EfLearning.fixed)
        for (final d in paramsFor(NodeKind.efState, input: true))
          _blockSlider(d, n.nus, (v, record) {
            model.replaceNode(n.id, copy(nus: v), record: record);
          }),
    ];
  }

  List<Widget> _dp(BuildContext context, DpStateNode n) {
    DpStateNode copy({int? batchSize, double? alpha, double? sp}) =>
        DpStateNode(
          n.id,
          label: n.label,
          batchSize: batchSize ?? n.batchSize,
          alpha: alpha ?? n.alpha,
          sensoryPrecision: sp ?? n.sensoryPrecision,
        );
    final defs = paramsFor(NodeKind.dpState, input: true);
    return [
      _Stepper(
        label: 'Clusters (batch size)',
        value: n.batchSize,
        min: 1,
        max: 32,
        onChanged: (v) => model.replaceNode(n.id, copy(batchSize: v)),
      ),
      Text(
        'Its clusters are an ef-state group above it with as many nodes, '
        'joined by a value edge.',
        style: Theme.of(context).textTheme.bodySmall,
      ),
      for (final d in defs)
        _blockSlider(
          d,
          d.field == NodeField.alpha ? n.alpha : n.sensoryPrecision,
          (v, record) => model.replaceNode(
            n.id,
            d.field == NodeField.alpha ? copy(alpha: v) : copy(sp: v),
            record: record,
          ),
        ),
    ];
  }

  Widget _blockSlider(
    ParamDef d,
    double value,
    void Function(double v, bool record) set,
  ) => LabeledSlider(
    label: '${d.symbol}  ${d.name}',
    help: d.help,
    value: d.log ? math.log(value) / math.ln10 : value,
    min: d.min,
    max: d.max,
    display: fmt(value),
    onChangeStart: (_) => model.checkpoint(),
    onChanged: (v) => set(d.log ? math.pow(10, v).toDouble() : v, false),
  );

  List<Widget> _edges(BuildContext context) {
    final theme = Theme.of(context);
    final mine = [
      for (final (i, e) in model.doc.edges.indexed)
        if (ownerIdOf(e.parent) == node.id || ownerIdOf(e.child) == node.id)
          (i, e),
    ];
    if (mine.isEmpty) {
      return [
        Text(
          'None yet: drag from a port (on top: parents; below: children) '
          'to another node.',
          style: theme.textTheme.bodySmall,
        ),
      ];
    }
    return [
      for (final (i, e) in mine)
        ListTile(
          dense: true,
          contentPadding: EdgeInsets.zero,
          visualDensity: VisualDensity.compact,
          leading: Icon(
            e.type == EdgeType.value ? Icons.arrow_downward : Icons.more_vert,
            size: 18,
            color: model.edgeIssues[i] == IssueSeverity.error
                ? theme.colorScheme.error
                : null,
          ),
          title: Text('${prettyId(e.parent)} → ${prettyId(e.child)}'),
          subtitle: Text(
            '${e.type.name}${e.strength == 1 ? '' : ', ×${fmt(e.strength)}'}',
          ),
          onTap: () => model.select(EdgeSelection(i)),
        ),
    ];
  }
}

Widget _paramSlider(BuilderModel model, String address, ParamDef d) {
  final v = model.initial(address, d.field);
  return LabeledSlider(
    label: '${d.symbol}  ${d.name}',
    help: d.help,
    value: d.log ? math.log(v) / math.ln10 : v,
    min: d.min,
    max: d.max,
    display: v.isFinite ? fmt(v) : '–',
    onChangeStart: (_) => model.checkpoint(),
    onChanged: (x) => model.setParam(
      address,
      d.field,
      d.log ? math.pow(10, x).toDouble() : x,
      record: false,
    ),
  );
}

class _IdField extends StatefulWidget {
  const _IdField({required this.model, required this.node});

  final BuilderModel model;
  final DocNode node;

  @override
  State<_IdField> createState() => _IdFieldState();
}

class _IdFieldState extends State<_IdField> {
  late final _text = TextEditingController(text: widget.node.id);
  String? _error;

  @override
  void dispose() {
    _text.dispose();
    super.dispose();
  }

  String? _check(String id) {
    if (!_idPattern.hasMatch(id)) {
      return 'Letters, digits and _; not starting with a digit';
    }
    if (id != widget.node.id && widget.model.doc.node(id) != null) {
      return 'Another node is called $id';
    }
    return null;
  }

  void _commit() {
    final id = _text.text.trim();
    final error = _check(id);
    setState(() => _error = error);
    if (error != null || id == widget.node.id) return;
    widget.model.replaceNode(widget.node.id, withId(widget.node, id));
  }

  @override
  Widget build(BuildContext context) => Focus(
    onFocusChange: (has) {
      if (!has) _commit();
    },
    child: TextField(
      controller: _text,
      decoration: InputDecoration(
        labelText: 'Name (id)',
        helperText: 'What addresses, expressions and exports call it',
        errorText: _error,
        isDense: true,
      ),
      onChanged: (v) => setState(() => _error = _check(v.trim())),
      onSubmitted: (_) => _commit(),
    ),
  );
}

/// A coupling function: a named one, or an expression in `x` with its
/// parse error shown as you type.
class _CouplingField extends StatefulWidget {
  const _CouplingField({required this.model, required this.node});

  final BuilderModel model;
  final ScalarNode node;

  @override
  State<_CouplingField> createState() => _CouplingFieldState();
}

class _CouplingFieldState extends State<_CouplingField> {
  static const _expression = '__expression__';
  late final _text = TextEditingController(text: _initialExpression);
  String? _error;

  bool get _named =>
      CouplingFn.values.any((f) => f.pyhgfName == widget.node.coupling) ||
      widget.node.coupling == 'identity';

  String get _initialExpression =>
      _named ? '2 * tanh(x / 2)' : widget.node.coupling;

  @override
  void dispose() {
    _text.dispose();
    super.dispose();
  }

  void _set(String coupling) {
    final n = widget.node;
    widget.model.replaceNode(
      n.id,
      ScalarNode(
        n.id,
        kind: n.kind,
        label: n.label,
        coupling: coupling,
        params: n.params,
      ),
    );
  }

  String? _check(String text) {
    try {
      parseCoupling(text);
      return null;
    } on ArgumentError catch (e) {
      final m = '${e.message}';
      final colon = m.lastIndexOf(': ');
      return colon < 0 ? m : m.substring(colon + 2);
    }
  }

  @override
  Widget build(BuildContext context) {
    final n = widget.node;
    if (n.customCoupling) {
      return Text(
        'Custom coupling "${n.coupling}" (code only)',
        style: Theme.of(context).textTheme.bodySmall,
      );
    }
    final named = _named;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        DropdownMenu<String>(
          initialSelection: named
              ? (n.coupling == 'identity' ? 'linear' : n.coupling)
              : _expression,
          expandedInsets: EdgeInsets.zero,
          label: const Text('Coupling to value children'),
          onSelected: (v) {
            if (v == null) return;
            if (v == _expression) {
              if (_check(_text.text) == null) _set(_text.text);
              setState(() {});
            } else if (v != n.coupling) {
              _set(v);
            }
          },
          dropdownMenuEntries: [
            for (final f in CouplingFn.values)
              DropdownMenuEntry(value: f.pyhgfName, label: f.pyhgfName),
            const DropdownMenuEntry(
              value: _expression,
              label: 'expression in x…',
            ),
          ],
        ),
        if (!named) ...[
          const SizedBox(height: 8),
          TextField(
            controller: _text,
            decoration: InputDecoration(
              labelText: 'g(x)',
              hintText: 'e.g. 2 * tanh(x / 2)',
              errorText: _error,
              errorMaxLines: 3,
              isDense: true,
            ),
            style: const TextStyle(fontFamily: 'monospace'),
            onChanged: (v) {
              final e = _check(v);
              setState(() => _error = e);
              if (e == null && v.trim() != n.coupling) _set(v.trim());
            },
          ),
        ],
      ],
    );
  }
}

// -----------------------------------------------------------------------------
// Edge
// -----------------------------------------------------------------------------

class _EdgePanel extends StatelessWidget {
  const _EdgePanel({super.key, required this.model, required this.index});

  final BuilderModel model;
  final int index;

  @override
  Widget build(BuildContext context) {
    final e = model.doc.edges[index];
    final theme = Theme.of(context);
    DocEdge copy({EdgeType? type, double? strength, bool flip = false}) =>
        DocEdge(
          type ?? e.type,
          flip ? e.child : e.parent,
          flip ? e.parent : e.child,
          strength: strength ?? e.strength,
        );
    String describe(String address) {
      final dot = address.indexOf('.');
      if (dot < 0) return prettyId(address);
      final node = model.doc.node(address.substring(0, dot));
      final k = int.tryParse(address.substring(dot + 1));
      if (node == null || k == null) return address;
      return '${prettyId(address)}: ${impliedNodeName(node, k)}';
    }

    return SectionCard(
      title: '${prettyId(e.parent)} → ${prettyId(e.child)}',
      action: IconButton(
        tooltip: 'Delete this edge (Del)',
        icon: const Icon(Icons.delete_outline, size: 20),
        onPressed: () => model.removeEdge(index),
      ),
      children: [
        Text(
          'Parent ${describe(e.parent)}; child ${describe(e.child)}',
          style: theme.textTheme.bodySmall,
        ),
        const SizedBox(height: 8),
        SegmentedButton<EdgeType>(
          showSelectedIcon: false,
          segments: const [
            ButtonSegment(value: EdgeType.value, label: Text('value')),
            ButtonSegment(
              value: EdgeType.volatility,
              label: Text('volatility'),
            ),
          ],
          selected: {e.type},
          onSelectionChanged: (s) =>
              model.replaceEdge(index, copy(type: s.single)),
        ),
        const SizedBox(height: 4),
        Text(
          e.type == EdgeType.value
              ? 'The parent\'s value sets the child\'s expected value '
                    '(through the parent\'s coupling function).'
              : 'The parent\'s value sets how fast the child is expected '
                    'to change (its log volatility).',
          style: theme.textTheme.bodySmall,
        ),
        LabeledSlider(
          label: 'strength (κ / α)',
          value: e.strength,
          min: -2,
          max: 2,
          display: fmt(e.strength),
          onChangeStart: (_) => model.checkpoint(),
          onChanged: (v) => model.replaceEdge(
            index,
            copy(strength: (v * 100).roundToDouble() / 100),
            record: false,
          ),
        ),
        _NumberField(
          key: ValueKey('strength $index ${e.strength}'),
          label: 'exact strength',
          value: e.strength,
          onSubmitted: (v) {
            if (v != null) model.replaceEdge(index, copy(strength: v));
          },
        ),
        const SizedBox(height: 8),
        Align(
          alignment: Alignment.centerLeft,
          child: TextButton.icon(
            icon: const Icon(Icons.swap_vert, size: 18),
            label: const Text('Reverse (swap parent and child)'),
            onPressed: () => model.replaceEdge(index, copy(flip: true)),
          ),
        ),
      ],
    );
  }
}

// -----------------------------------------------------------------------------
// Issues and help
// -----------------------------------------------------------------------------

class _IssuesCard extends StatelessWidget {
  const _IssuesCard({required this.model});

  final BuilderModel model;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final errors = model.issues.where((i) => i.isError).length;
    final warnings = model.issues.length - errors;
    return SectionCard(
      title: [
        if (errors > 0) '$errors error${errors == 1 ? '' : 's'}',
        if (warnings > 0) '$warnings warning${warnings == 1 ? '' : 's'}',
      ].join(', '),
      children: [
        for (final i in model.issues)
          ListTile(
            dense: true,
            contentPadding: EdgeInsets.zero,
            visualDensity: VisualDensity.compact,
            leading: Icon(
              i.isError ? Icons.error_outline : Icons.warning_amber,
              size: 20,
              color: i.isError ? scheme.error : const Color(0xFFD08A00),
            ),
            title: Text(i.message),
            onTap: i.edges.isEmpty && i.nodes.isEmpty
                ? null
                : () => model.select(
                    i.edges.isNotEmpty
                        ? EdgeSelection(i.edges.last)
                        : NodeSelection(i.nodes.first),
                  ),
          ),
      ],
    );
  }
}

class _HelpCard extends StatelessWidget {
  const _HelpCard();

  @override
  Widget build(BuildContext context) => SectionCard(
    title: 'How to',
    children: [
      Text(
        'Parents sit above their children. Drag from a port to a node to '
        'connect them: the ports on top take parents, those below '
        'children; filled ports are value edges (solid), hollow ones '
        'volatility edges (dashed). Drag a node to move it, the background '
        'to pan; scroll or pinch to zoom. Double-click the background to '
        'add a node; right-click (or long-press) a node for its parents '
        'and children. Del deletes, Ctrl+Z / Ctrl+Shift+Z undo and redo, '
        'F fits the view. Filled nodes are inputs (no children).',
        style: Theme.of(context).textTheme.bodySmall,
      ),
    ],
  );
}

// -----------------------------------------------------------------------------
// Small fields
// -----------------------------------------------------------------------------

class _Stepper extends StatelessWidget {
  const _Stepper({
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.onChanged,
  });

  final String label;
  final int value, min, max;
  final ValueChanged<int> onChanged;

  @override
  Widget build(BuildContext context) => Row(
    children: [
      Expanded(child: Text(label)),
      IconButton(
        tooltip: 'Fewer',
        icon: const Icon(Icons.remove, size: 18),
        onPressed: value > min ? () => onChanged(value - 1) : null,
      ),
      SizedBox(
        width: 28,
        child: Text(
          '$value',
          textAlign: TextAlign.center,
          style: const TextStyle(fontFeatures: [FontFeature.tabularFigures()]),
        ),
      ),
      IconButton(
        tooltip: 'More',
        icon: const Icon(Icons.add, size: 18),
        onPressed: value < max ? () => onChanged(value + 1) : null,
      ),
    ],
  );
}

/// A number typed in, committed on Enter or when the field loses focus;
/// empty commits null.
class _NumberField extends StatefulWidget {
  const _NumberField({
    super.key,
    required this.label,
    required this.value,
    required this.onSubmitted,
  });

  final String label;
  final double? value;
  final ValueChanged<double?> onSubmitted;

  @override
  State<_NumberField> createState() => _NumberFieldState();
}

class _NumberFieldState extends State<_NumberField> {
  late final _text = TextEditingController(
    text: widget.value == null ? '' : '${widget.value}',
  );
  String? _error;

  @override
  void dispose() {
    _text.dispose();
    super.dispose();
  }

  void _commit() {
    final t = _text.text.trim();
    if (t.isEmpty) {
      if (widget.value != null) widget.onSubmitted(null);
      return;
    }
    final v = double.tryParse(t);
    setState(() => _error = v == null ? 'Not a number' : null);
    if (v != null && v != widget.value) widget.onSubmitted(v);
  }

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.only(top: 6),
    child: Focus(
      onFocusChange: (has) {
        if (!has) _commit();
      },
      child: TextField(
        controller: _text,
        keyboardType: const TextInputType.numberWithOptions(
          signed: true,
          decimal: true,
        ),
        decoration: InputDecoration(
          labelText: widget.label,
          hintText: 'default',
          errorText: _error,
          isDense: true,
        ),
        onSubmitted: (_) => _commit(),
      ),
    ),
  );
}
