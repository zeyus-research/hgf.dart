import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_widgets/hgf_widgets.dart';
import 'package:material_ui/material_ui.dart';

import '../../model/presets.dart';
import '../../ui/files.dart';
import '../builder_model.dart';
import 'builder_canvas.dart';
import 'inspector.dart';

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
            tooltip: 'Lay out again (parents above children)',
            onPressed: m.autoLayout,
            icon: const Icon(Icons.account_tree_outlined),
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
                value: () => m.open(_emptyDocument(), name: 'Untitled'),
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
    child: BuilderCanvas(model: m),
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

  Widget _narrow(BuildContext context) {
    const h = 130.0;
    final net = m.network;
    final panels = net == null || m.trajectories == null
        ? 1
        : HgfTrajectoryView.panelCount(net);
    return ListView(
      padding: const EdgeInsets.all(12),
      children: [
        SizedBox(height: 420, child: _canvas()),
        const SizedBox(height: 8),
        _PreviewBar(model: m),
        SizedBox(
          height: panels * h,
          child: BuilderPreview(model: m, panelHeight: h),
        ),
        const SizedBox(height: 12),
        BuilderInspector(model: m, shrinkWrap: true),
      ],
    );
  }

  static NetworkDocument _emptyDocument() => NetworkDocument(nodes: const []);
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
    return Row(
      children: [
        Text('Preview', style: theme.textTheme.titleSmall),
        const SizedBox(width: 12),
        Expanded(
          child: Text(
            model.network == null
                ? ''
                : 'Simulated data per input (reversals for binary, random '
                      'walks for continuous); checked, built and filtered in '
                      '$ms ms${model.trials < model.data.nTrials ? '; ${model.trials} trials (dp-state nodes are slow)' : ''}',
            style: theme.textTheme.bodySmall,
            overflow: TextOverflow.ellipsis,
          ),
        ),
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
        IconButton(
          tooltip: 'New random data',
          icon: const Icon(Icons.casino_outlined, size: 20),
          onPressed: model.reseed,
        ),
      ],
    );
  }
}

/// The playground's trajectory plots of the built network, with the
/// simulated hidden states drawn behind the inputs.
class BuilderPreview extends StatefulWidget {
  const BuilderPreview({super.key, required this.model, this.panelHeight});

  final BuilderModel model;
  final double? panelHeight;

  @override
  State<BuilderPreview> createState() => _BuilderPreviewState();
}

class _BuilderPreviewState extends State<BuilderPreview> {
  final _hover = ValueNotifier<int?>(null);
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
