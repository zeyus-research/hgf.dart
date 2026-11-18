import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/codegen.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_widgets/hgf_widgets.dart';
import 'package:material_ui/material_ui.dart';

import '../../model/presets.dart';
import '../../ui/files.dart';
import '../builder_model.dart';
import 'builder_canvas.dart';
import 'deep_canvas.dart';
import 'deep_inspector.dart' show exportDeep;
import 'deep_preview.dart';
import 'inspector.dart';
import 'pipeline_canvas.dart';
import 'search_picker.dart';

/// The model builder: the HGF graph on a canvas, an inspector for what is
/// selected, and a live preview (the network built from the document and
/// filtered on simulated data after every edit).
class BuilderPage extends StatefulWidget {
  const BuilderPage({super.key, required this.model});

  final BuilderModel model;

  @override
  State<BuilderPage> createState() => _BuilderPageState();
}

class _BuilderPageState extends State<BuilderPage> {
  BuilderModel get m => widget.model;

  /// The inspector's sheet on phones ([_narrow]).
  final _sheet = DraggableScrollableController();

  /// The sheet's sizes (fractions of the body): a peek at what is
  /// selected, and open to just below the canvas.
  double _peekSize = 0.12, _openSize = 0.45;
  BuilderSelection? _selection;

  @override
  void initState() {
    super.initState();
    _selection = m.selection;
    m.addListener(_selectionChanged);
  }

  @override
  void didUpdateWidget(BuilderPage old) {
    super.didUpdateWidget(old);
    if (old.model != m) {
      old.model.removeListener(_selectionChanged);
      m.addListener(_selectionChanged);
    }
  }

  @override
  void dispose() {
    m.removeListener(_selectionChanged);
    _sheet.dispose();
    super.dispose();
  }

  /// Something new was selected: open the sheet (if it is showing and
  /// below half way) so that its properties are in view.
  void _selectionChanged() {
    final s = m.selection;
    if (s == _selection) return;
    _selection = s;
    if (s == null || !_sheet.isAttached || _sheet.size >= _openSize - 0.01) {
      return;
    }
    _sheet.animateTo(
      _openSize,
      duration: const Duration(milliseconds: 260),
      curve: Curves.easeOutCubic,
    );
  }

  void _toggleSheet() {
    if (!_sheet.isAttached) return;
    _sheet.animateTo(
      _sheet.size > _peekSize + 0.02 ? _peekSize : _openSize,
      duration: const Duration(milliseconds: 260),
      curve: Curves.easeOutCubic,
    );
  }

  Future<void> _open() async {
    final messenger = ScaffoldMessenger.of(context);
    try {
      final picked = await pickText('Open a model');
      if (picked == null) return;
      final (text, name) = picked;
      m.openJson(text, name: name.replaceAll(RegExp(r'\.json$'), ''));
    } on Object catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Could not open: $e')));
    }
  }

  Future<void> _save() => saveText(
    context,
    '${m.name.replaceAll(RegExp(r'[^A-Za-z0-9_-]+'), '_')}.json',
    m.toJsonText(),
    mimeType: 'application/json',
  );

  /// The document as a runnable script: its network, every participant
  /// of the data (the loaded file's, read from it; simulated ones,
  /// embedded) and, with free parameters, the fit of all of them with the
  /// group level, writing the estimates and the outputs per row.
  Future<void> _exportCode({required bool python}) async {
    final messenger = ScaffoldMessenger.of(context);
    final c = m.compiled;
    if (c == null || m.subject == null) {
      messenger.showSnackBar(
        const SnackBar(content: Text('Fix the errors to export the model')),
      );
      return;
    }
    final String script;
    try {
      final loaded = m.csv != null;
      final spec = c.exportSpec(
        dataFile: loaded ? (m.csvName ?? 'data.csv') : null,
        csv: loaded ? null : m.simulatedCsv().toText(),
      );
      script = python ? PythonExport(spec).script() : DartExport(spec).script();
    } on Object catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Could not export: $e')));
      return;
    }
    if (!mounted) return;
    if (m.csv != null) {
      messenger.showSnackBar(
        SnackBar(
          content: Text(
            'The script reads ${m.csvName ?? 'data.csv'}: keep it next to '
            'the script, or pass the CSV\'s path when you run it.',
          ),
        ),
      );
    }
    final base = m.name.replaceAll(RegExp(r'[^A-Za-z0-9_]+'), '_');
    await saveText(
      context,
      python ? '$base.py' : '$base.dart',
      script,
      mimeType: python ? 'text/x-python' : 'text/x-dart',
    );
  }

  @override
  Widget build(BuildContext context) => ListenableBuilder(
    listenable: m,
    builder: (context, _) => Scaffold(
      appBar: AppBar(
        title: Text(
          MediaQuery.sizeOf(context).width < 600
              ? 'Builder'
              : 'Model builder · ${m.name}',
          overflow: TextOverflow.ellipsis,
        ),
        actions: [
          IconButton(
            tooltip: 'Undo (Ctrl+Z)',
            onPressed: m.canUndo ? m.undo : null,
            icon: const Icon(Icons.undo),
          ),
          IconButton(
            tooltip: 'Redo (Ctrl+Shift+Z)',
            onPressed: m.canRedo ? m.redo : null,
            icon: const Icon(Icons.redo),
          ),
          IconButton(
            tooltip: m.view == BuilderView.network
                ? 'Lay out again (parents above children)'
                : 'Lay out again',
            onPressed: switch (m.view) {
              BuilderView.network => m.autoLayout,
              BuilderView.pipeline => m.autoLayoutPipeline,
              BuilderView.deep => null,
            },
            icon: const Icon(Icons.auto_fix_high_outlined),
          ),
          PopupMenuButton<String>(
            tooltip: 'Open a preset',
            icon: const Icon(Icons.library_books_outlined),
            onSelected: (id) => m.openPreset(presetById(id)),
            itemBuilder: (context) => [
              for (final p in presets)
                PopupMenuItem(value: p.id, child: Text(p.name)),
            ],
          ),
          PopupMenuButton<void Function()>(
            tooltip: 'File',
            icon: const Icon(Icons.more_vert),
            onSelected: (f) => f(),
            itemBuilder: (context) => [
              PopupMenuItem(
                value: () =>
                    m.openNetworkDocument(_emptyDocument(), name: 'Untitled'),
                child: const ListTile(
                  leading: Icon(Icons.note_add_outlined),
                  title: Text('New (empty)'),
                ),
              ),
              PopupMenuItem(
                value: _open,
                child: const ListTile(
                  leading: Icon(Icons.folder_open_outlined),
                  title: Text('Open…'),
                ),
              ),
              PopupMenuItem(
                value: _save,
                child: const ListTile(
                  leading: Icon(Icons.save_outlined),
                  title: Text('Save as JSON'),
                ),
              ),
              const PopupMenuDivider(),
              if (m.document.hasNetwork) ...[
                PopupMenuItem(
                  value: () => _exportCode(python: false),
                  child: const ListTile(
                    leading: Icon(Icons.code),
                    title: Text('Export: Dart script'),
                  ),
                ),
                PopupMenuItem(
                  value: () => _exportCode(python: true),
                  child: const ListTile(
                    leading: Icon(Icons.code),
                    title: Text('Export: Python script (pyhgf)'),
                  ),
                ),
              ],
              for (final d in m.document.deep) ...[
                PopupMenuItem(
                  value: () => exportDeep(context, m, d, python: false),
                  child: ListTile(
                    leading: const Icon(Icons.layers_outlined),
                    title: Text('Export "${d.label}": Dart script'),
                  ),
                ),
                PopupMenuItem(
                  value: () => exportDeep(context, m, d, python: true),
                  child: ListTile(
                    leading: const Icon(Icons.layers_outlined),
                    title: Text('Export "${d.label}": Python (pyhgf)'),
                  ),
                ),
              ],
            ],
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, box) =>
            box.maxWidth >= 900 ? _wide(context) : _narrow(context),
      ),
    ),
  );

  Widget _canvas() => Card(
    clipBehavior: Clip.antiAlias,
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _Breadcrumbs(model: m),
        const Divider(height: 1),
        Expanded(
          child: switch (m.view) {
            BuilderView.pipeline => PipelineCanvas(model: m),
            BuilderView.network => BuilderCanvas(model: m),
            BuilderView.deep => DeepCanvas(model: m),
          },
        ),
      ],
    ),
  );

  Widget _wide(BuildContext context) => Row(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      Expanded(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(12, 12, 8, 8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(flex: 5, child: _canvas()),
              const SizedBox(height: 8),
              _PreviewBar(model: m),
              Expanded(flex: 4, child: BuilderPreview(model: m)),
            ],
          ),
        ),
      ),
      const VerticalDivider(width: 1),
      SizedBox(width: 360, child: BuilderInspector(model: m)),
    ],
  );

  /// Phones: the canvas and the preview scroll; the inspector is a sheet
  /// over them that peeks at what is selected and opens (to just below the
  /// canvas) when something new is.
  Widget _narrow(BuildContext context) {
    const h = 130.0, canvasHeight = 420.0, peekHeight = 76.0;
    final net = m.network;
    final shownDeep = BuilderPreview.shownDeep(m);
    final panels = shownDeep != null
        ? DeepPreview.panelCount(m, shownDeep) + 1
        : (net == null || m.trajectories == null
                  ? 1
                  : HgfTrajectoryView.panelCount(net)) +
              BuilderPreview.extraPanels(m);
    final scheme = Theme.of(context).colorScheme;
    return LayoutBuilder(
      builder: (context, box) {
        final height = math.max(box.maxHeight, 1.0);
        _peekSize = (peekHeight / height).clamp(0.05, 0.3);
        _openSize = ((height - canvasHeight - 20) / height).clamp(
          _peekSize + 0.1,
          0.8,
        );
        return Stack(
          children: [
            ListView(
              padding: const EdgeInsets.fromLTRB(12, 12, 12, peekHeight + 12),
              children: [
                SizedBox(height: canvasHeight, child: _canvas()),
                const SizedBox(height: 8),
                _PreviewBar(model: m),
                SizedBox(
                  height: panels * h,
                  child: BuilderPreview(model: m, panelHeight: h),
                ),
              ],
            ),
            DraggableScrollableSheet(
              controller: _sheet,
              initialChildSize: _peekSize,
              minChildSize: _peekSize,
              maxChildSize: 0.92,
              snap: true,
              snapSizes: [_openSize],
              builder: (context, scroll) => Material(
                elevation: 6,
                color: scheme.surfaceContainerLow,
                surfaceTintColor: scheme.surfaceTint,
                shape: const RoundedRectangleBorder(
                  borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
                ),
                clipBehavior: Clip.antiAlias,
                child: BuilderInspector(
                  model: m,
                  controller: scroll,
                  leading: _SheetHandle(onTap: _toggleSheet),
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  static NetworkDocument _emptyDocument() => NetworkDocument(nodes: const []);
}

/// The inspector sheet's handle: drag it (or the panel) to resize, tap it
/// to open or close.
class _SheetHandle extends StatelessWidget {
  const _SheetHandle({required this.onTap});

  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => Semantics(
    button: true,
    label: 'Properties: open or close',
    child: GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 10),
        child: Center(
          child: Container(
            width: 36,
            height: 4,
            decoration: BoxDecoration(
              color: Theme.of(
                context,
              ).colorScheme.onSurfaceVariant.withValues(alpha: 0.45),
              borderRadius: BorderRadius.circular(2),
            ),
          ),
        ),
      ),
    ),
  );
}

/// Where the canvas is: the pipeline, or the network inside it.
class _Breadcrumbs extends StatelessWidget {
  const _Breadcrumbs({required this.model});

  final BuilderModel model;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final view = model.view;
    final deep = view == BuilderView.deep ? model.deep : null;
    final issuePart = deep != null ? DocPart.deep(deep.id) : DocPart.network;
    final hasIssue = model.partIssues.containsKey(issuePart);
    final showNetwork =
        view == BuilderView.network ||
        (view == BuilderView.pipeline && model.document.hasNetwork);
    // Phones: no room for the hint beside the crumbs.
    final hint = MediaQuery.sizeOf(context).width >= 600;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
      child: Row(
        children: [
          TextButton.icon(
            onPressed: view != BuilderView.pipeline
                ? () => model.showView(BuilderView.pipeline)
                : null,
            icon: const Icon(Icons.linear_scale, size: 18),
            label: const Text('Pipeline'),
          ),
          if (showNetwork || deep != null) ...[
            const Icon(Icons.chevron_right, size: 18),
            Flexible(
              child: TextButton.icon(
                onPressed: view == BuilderView.pipeline
                    ? () => model.showView(BuilderView.network)
                    : null,
                icon: Icon(
                  hasIssue
                      ? Icons.error_outline
                      : deep != null
                      ? Icons.layers_outlined
                      : Icons.account_tree_outlined,
                  size: 18,
                  color: hasIssue ? theme.colorScheme.error : null,
                ),
                label: Text(
                  deep?.label ?? 'HGF network',
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ),
          ],
          const Spacer(),
          if (hint)
            Flexible(
              child: Text(
                switch (view) {
                  BuilderView.network =>
                    'Parents above children; drag from a port to connect',
                  BuilderView.deep =>
                    'Features on top, targets below; + inserts a layer',
                  BuilderView.pipeline =>
                    'Data → transforms → models → response and outputs',
                },
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          const SizedBox(width: 8),
        ],
      ),
    );
  }
}

/// The preview's header: what was simulated, how long building and
/// filtering took, the number of trials and a new seed.
class _PreviewBar extends StatelessWidget {
  const _PreviewBar({required this.model});

  final BuilderModel model;

  static const _trials = [100, 300, 1000, 3000];

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final ms = (model.previewMicros / 1000).toStringAsFixed(1);
    // Phones: the description on a line of its own, under the controls.
    final narrow = MediaQuery.sizeOf(context).width < 600;
    final description = Text(
      model.network == null
          ? ''
          : model.csv != null
          ? '${model.csvName ?? 'Loaded data'}'
                '${model.participants.isEmpty ? '' : ', participant ${model.participants[model.participant.clamp(0, model.participants.length - 1)]}'}'
                '; checked, built and filtered in $ms ms'
          : 'Simulated data per input (reversals for binary, random '
                'walks for continuous)'
                '${model.subject?.responses != null ? ' and responses' : ''}'
                '; checked, built and filtered in '
                '$ms ms${model.trials < model.data.nTrials ? '; ${model.trials} trials (dp-state nodes are slow)' : ''}',
      style: theme.textTheme.bodySmall,
      overflow: TextOverflow.ellipsis,
      maxLines: narrow ? 2 : 1,
    );
    final controls = Row(
      children: [
        if (narrow)
          Expanded(
            child: Text(
              'Preview',
              style: theme.textTheme.titleSmall,
              overflow: TextOverflow.ellipsis,
            ),
          )
        else ...[
          Text('Preview', style: theme.textTheme.titleSmall),
          const SizedBox(width: 12),
          Expanded(child: description),
        ],
        if (model.participants.length > searchPickerAbove)
          TextButton.icon(
            icon: const Icon(Icons.person_search_outlined, size: 18),
            label: Text(
              model.participants[model.participant.clamp(
                0,
                model.participants.length - 1,
              )],
            ),
            onPressed: () async {
              final pick = await showSearchPicker<int>(
                context,
                title: 'Participant',
                items: {for (final (k, p) in model.participants.indexed) k: p},
                selected: model.participant,
              );
              if (pick case (final k,)) model.showParticipant(k);
            },
          )
        else if (model.participants.length > 1)
          DropdownButton<int>(
            value: model.participant.clamp(0, model.participants.length - 1),
            isDense: true,
            underline: const SizedBox.shrink(),
            items: [
              for (final (k, p) in model.participants.indexed)
                DropdownMenuItem(value: k, child: Text(p)),
            ],
            onChanged: (k) {
              if (k != null) model.showParticipant(k);
            },
          ),
        if (model.csv == null)
          DropdownButton<int>(
            value: _trials.contains(model.data.nTrials)
                ? model.data.nTrials
                : null,
            hint: Text('${model.data.nTrials}'),
            isDense: true,
            underline: const SizedBox.shrink(),
            items: [
              for (final n in _trials)
                DropdownMenuItem(value: n, child: Text('$n trials')),
            ],
            onChanged: (n) {
              if (n != null) model.setTrials(n);
            },
          ),
        if (model.csv == null)
          IconButton(
            tooltip: 'New random data',
            icon: const Icon(Icons.casino_outlined, size: 20),
            onPressed: model.reseed,
          ),
      ],
    );
    if (!narrow) return controls;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [controls, description],
    );
  }
}

/// The playground's trajectory plots of the built network, with the
/// simulated hidden states drawn behind the inputs.
class BuilderPreview extends StatefulWidget {
  const BuilderPreview({super.key, required this.model, this.panelHeight});

  final BuilderModel model;
  final double? panelHeight;

  /// At most this many outputs are plotted.
  static const maxOutputs = 4;

  /// The deep network the preview shows: the one on the canvas, the
  /// selected card's, or the first of a model without an HGF network.
  static DocDeep? shownDeep(BuilderModel m) {
    if (m.view == BuilderView.deep) return m.deep;
    if (m.view != BuilderView.pipeline) return null;
    if (m.selection case PartSelection(
      :final part,
    ) when part.startsWith('deep:')) {
      for (final d in m.document.deep) {
        if (DocPart.deep(d.id) == part) return d;
      }
    }
    if (!m.document.hasNetwork && m.document.deep.isNotEmpty) {
      return m.document.deep.first;
    }
    return null;
  }

  /// The panels drawn above the network's: the responses and the named
  /// outputs (on the pipeline only).
  static int extraPanels(BuilderModel m) => m.view != BuilderView.pipeline
      ? 0
      : (m.subject?.responses != null ? 1 : 0) +
            math.min(m.outputs.length, maxOutputs);

  @override
  State<BuilderPreview> createState() => _BuilderPreviewState();
}

class _BuilderPreviewState extends State<BuilderPreview> {
  final _hover = ValueNotifier<int?>(null);

  /// The responses and the named outputs, one plot each.
  List<Widget> _extras(BuildContext context, BuilderModel m) {
    if (m.view != BuilderView.pipeline) return const [];
    final colors = HgfColors.of(context);
    final scheme = Theme.of(context).colorScheme;
    final r = m.subject?.responses;
    final n = m.trajectories?.nTime ?? 0;
    return [
      if (r != null && r.length >= n)
        TrajectoryPlot(
          title:
              '${m.document.response?.name ?? 'response'} '
              '(${m.csv == null ? 'simulated' : 'recorded'})',
          layers: [
            PointLayer(
              Float64List.sublistView(r, 0, n),
              color: scheme.tertiary,
              radius: 1.6,
            ),
          ],
          hover: _hover,
          revision: m.revision,
          showXAxis: false,
        ),
      for (final MapEntry(:key, :value) in m.outputs.entries.take(
        BuilderPreview.maxOutputs,
      ))
        TrajectoryPlot(
          title: '$key (output)',
          layers: [LineLayer(value, color: colors.category(1))],
          hover: _hover,
          revision: m.revision,
          showXAxis: false,
        ),
    ];
  }

  Network? _laidOut;
  NetworkLayout? _layout;

  @override
  void dispose() {
    _hover.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final m = widget.model;
    final net = m.network, traj = m.trajectories;
    final theme = Theme.of(context);
    if (BuilderPreview.shownDeep(m) case final d?) {
      if (m.csv == null) {
        return Center(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Text(
              '"${d.label}" trains on loaded data: load a CSV (Data panel) '
              'and bind its columns on its card. Or export it (Dart or '
              'pyhgf) and train it there.',
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
        );
      }
      return DeepPreview(
        model: m,
        deep: d,
        hover: _hover,
        panelHeight: widget.panelHeight,
      );
    }
    if (net == null || traj == null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Text(
            m.previewError ?? 'No preview',
            textAlign: TextAlign.center,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ),
      );
    }
    if (!identical(net, _laidOut)) {
      _laidOut = net;
      _layout = NetworkLayout.of(net);
    }
    final addresses = m.doc.addresses;
    final muted = theme.colorScheme.onSurfaceVariant;
    final colors = HgfColors.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (m.previewError case final e?)
          Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: Text(
              e,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.error,
              ),
            ),
          ),
        if (m.deepPending case final p when p.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: Text(
              'Waiting for the predictions of ${p.map((l) => '"$l"').join(', ')} '
              '(training): the inputs that read them are unobserved until '
              'then.',
              style: theme.textTheme.bodySmall?.copyWith(color: muted),
            ),
          ),
        for (final w in _extras(context, m))
          SizedBox(height: widget.panelHeight ?? 64, child: w),
        Expanded(
          child: HgfTrajectoryView(
            network: net,
            trajectories: traj,
            layout: _layout,
            revision: m.revision,
            hover: _hover,
            panelHeight: widget.panelHeight,
            label: (i) => prettyId(addresses.addressOf(i)),
            extraLayers: {
              for (final MapEntry(key: i, value: hs) in m.hidden.entries)
                i: [
                  for (final (k, h) in hs.indexed)
                    if (h.length == traj.nTime)
                      LineLayer(
                        h,
                        color: hs.length > 1
                            ? colors.category(k).withValues(alpha: 0.45)
                            : muted.withValues(alpha: 0.7),
                        width: 1,
                        label: 'simulated',
                      ),
                ],
            },
          ),
        ),
      ],
    );
  }
}
