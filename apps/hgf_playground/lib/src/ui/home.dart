import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_widgets/hgf_widgets.dart';

import '../model/fitting.dart';
import '../model/playground.dart';
import '../model/presets.dart';
import '../settings.dart';
import 'choice_plots.dart';
import 'fit_card.dart';
import 'frame_stats.dart';
import 'panels.dart';
import 'playback_bar.dart';
import 'recovery_card.dart';
import 'settings_panel.dart';

class HomePage extends StatefulWidget {
  const HomePage({
    super.key,
    required this.settings,
    required this.playground,
    required this.fitting,
  });

  final Settings settings;
  final Playground playground;
  final Fitting fitting;

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage>
    with SingleTickerProviderStateMixin {
  late final Ticker _ticker = createTicker(_onTick);
  Duration _last = Duration.zero;
  final ValueNotifier<int?> _hover = ValueNotifier(null);
  final FrameStats _frames = FrameStats();
  final _scaffold = GlobalKey<ScaffoldState>();

  void _openSettings() => _scaffold.currentState?.openEndDrawer();

  /// Parameters, couplings, update scheme and data back to the preset's.
  void _reset() {
    pg.resetToPreset();
    widget.settings.remember(nTrials: pg.data.nTrials);
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          content: Text('${pg.preset.name}: back to its defaults'),
          duration: const Duration(seconds: 2),
        ),
      );
  }

  Playground get pg => widget.playground;
  Fitting get fit => widget.fitting;

  /// Notifies when the fitted responses change (not on every fit update).
  late final _responses = _Selected(fit, () => fit.revision);

  (int, Float64List?)? _responseCache;

  /// The fitted responses as plotted: binary choices just outside the
  /// outcomes (1.05 / −0.05), so both stay visible.
  Float64List? _shownResponses() {
    final c = _responseCache;
    if (c != null && c.$1 == fit.revision) return c.$2;
    final r = fit.responses;
    final shown = r == null || pg.preset.input != InputKind.binary
        ? r
        : Float64List.fromList([
            for (final v in r) v.isNaN ? v : v * 1.1 - 0.05,
          ]);
    _responseCache = (fit.revision, shown);
    return shown;
  }

  @override
  void initState() {
    super.initState();
    pg.addListener(_syncTicker);
  }

  @override
  void dispose() {
    pg.removeListener(_syncTicker);
    _ticker.dispose();
    _hover.dispose();
    _frames.dispose();
    _responses.dispose();
    super.dispose();
  }

  void _syncTicker() {
    if (pg.playing && !_ticker.isActive) {
      _last = Duration.zero;
      _ticker.start();
    } else if (!pg.playing && _ticker.isActive) {
      _ticker.stop();
    }
  }

  void _onTick(Duration elapsed) {
    final dt = (elapsed - _last).inMicroseconds / 1e6;
    _last = elapsed;
    pg.tick(dt);
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      // The fit card listens on its own; the plots need the responses.
      listenable: Listenable.merge([pg, _responses]),
      builder: (context, _) => Scaffold(
        key: _scaffold,
        endDrawer: Drawer(
          width: 380,
          child: SafeArea(
            child: SettingsPanel(settings: widget.settings, fitting: fit),
          ),
        ),
        appBar: AppBar(
          title: Text(
            MediaQuery.sizeOf(context).width < 600 ? 'HGF' : 'HGF playground',
          ),
          actions: [
            _PresetMenu(playground: pg, settings: widget.settings),
            IconButton(
              tooltip: 'Reset to the preset\'s defaults',
              onPressed: _reset,
              icon: const Icon(Icons.restart_alt),
            ),
            _ThemeMenu(settings: widget.settings),
            IconButton(
              tooltip: 'Settings',
              onPressed: _openSettings,
              icon: const Icon(Icons.tune),
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
  }

  Widget _plots({double? panelHeight}) {
    final traj = pg.trajectories;
    if (traj == null) {
      return Center(child: Text(pg.error ?? 'No data'));
    }
    if (pg.task != null) {
      return ChoiceTaskView(
        playground: pg,
        hover: _hover,
        panelHeight: panelHeight,
      );
    }
    final input = pg.network.inputIdxs.first;
    final hidden = pg.hidden;
    final muted = Theme.of(context).colorScheme.onSurfaceVariant;
    final colors = HgfColors.of(context);
    final categorical = pg.preset.input == InputKind.categorical;
    final binary = pg.preset.input == InputKind.binary;
    return HgfTrajectoryView(
      network: pg.network,
      trajectories: traj,
      layout: pg.layout,
      revision: pg.revision,
      limit: pg.player?.t,
      hover: _hover,
      panelHeight: panelHeight,
      extraLayers: {
        input: [
          if (hidden != null && hidden.first.length == traj.nTime)
            for (final (k, h) in hidden.indexed)
              LineLayer(
                h,
                // Categories keep their colour; one hidden series is grey.
                color: categorical
                    ? colors.category(k).withValues(alpha: 0.45)
                    : muted.withValues(alpha: 0.7),
                width: 1,
                label: switch (pg.preset.data) {
                  DataModel.reversal => 'true p',
                  DataModel.categoricalReversal => 'true P($k)',
                  DataModel.mixture => 'source',
                  DataModel.randomWalk => 'hidden',
                  DataModel.igt || DataModel.bandit => 'expected payoff',
                },
              ),
          if (_shownResponses() case final r? when r.length == traj.nTime)
            PointLayer(
              r,
              color: colors.category(6),
              radius: 1.6,
              label: binary ? 'choice (offset)' : 'response',
              readoutGaps: false,
            ),
        ],
      },
    );
  }

  Widget _status() => PlaybackBar(playground: pg, frames: _frames);

  Widget _wide(BuildContext context) => Row(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      SizedBox(
        width: 360,
        child: ListView(padding: const EdgeInsets.all(12), children: _cards()),
      ),
      const VerticalDivider(width: 1),
      Expanded(
        child: Column(
          children: [
            _status(),
            if (pg.error != null) _ErrorBanner(pg.error!),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(4, 0, 12, 8),
                child: _plots(),
              ),
            ),
          ],
        ),
      ),
    ],
  );

  Widget _narrow(BuildContext context) {
    final traj = pg.trajectories;
    final panels = traj == null
        ? 1
        : pg.task != null
        ? ChoiceTaskView.panelCount(pg)
        : HgfTrajectoryView.panelCount(pg.network, layout: pg.layout);
    const h = 150.0;
    return ListView(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      children: [
        _status(),
        if (pg.error != null) _ErrorBanner(pg.error!),
        SizedBox(
          height: panels * h,
          child: _plots(panelHeight: h),
        ),
        const SizedBox(height: 12),
        ..._cards(),
      ],
    );
  }

  List<Widget> _cards() => [
    NetworkCard(playground: pg, hover: _hover),
    const SizedBox(height: 12),
    NodeCard(playground: pg),
    const SizedBox(height: 12),
    DataCard(playground: pg, settings: widget.settings, fitting: fit),
    const SizedBox(height: 12),
    FitCard(fitting: fit, onOpenSettings: _openSettings),
    const SizedBox(height: 12),
    RecoveryCard(fitting: fit, onOpenSettings: _openSettings),
    const SizedBox(height: 12),
    const AboutCard(),
  ];
}

class _ErrorBanner extends StatelessWidget {
  const _ErrorBanner(this.message);

  final String message;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      width: double.infinity,
      color: scheme.errorContainer,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      child: Row(
        children: [
          Icon(Icons.warning_amber, size: 18, color: scheme.onErrorContainer),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              message,
              style: TextStyle(color: scheme.onErrorContainer),
            ),
          ),
        ],
      ),
    );
  }
}

class _PresetMenu extends StatelessWidget {
  const _PresetMenu({required this.playground, required this.settings});

  final Playground playground;
  final Settings settings;

  @override
  Widget build(BuildContext context) => PopupMenuButton<String>(
    tooltip: 'Network',
    initialValue: playground.preset.id,
    onSelected: (id) {
      playground.setPreset(id);
      settings.remember(preset: id);
    },
    itemBuilder: (context) => [
      for (final p in presets) PopupMenuItem(value: p.id, child: Text(p.name)),
    ],
    child: Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.account_tree_outlined, size: 20),
          const SizedBox(width: 6),
          ConstrainedBox(
            constraints: BoxConstraints(
              maxWidth: MediaQuery.sizeOf(context).width < 600 ? 150 : 260,
            ),
            child: Text(
              playground.preset.name,
              overflow: TextOverflow.ellipsis,
            ),
          ),
          const Icon(Icons.arrow_drop_down),
        ],
      ),
    ),
  );
}

class _ThemeMenu extends StatelessWidget {
  const _ThemeMenu({required this.settings});

  final Settings settings;

  static const _icons = {
    ThemeMode.system: Icons.brightness_auto,
    ThemeMode.light: Icons.light_mode,
    ThemeMode.dark: Icons.dark_mode,
  };
  static const _names = {
    ThemeMode.system: 'Follow the system',
    ThemeMode.light: 'Light',
    ThemeMode.dark: 'Dark',
  };

  @override
  Widget build(BuildContext context) => PopupMenuButton<ThemeMode>(
    tooltip: 'Theme',
    icon: Icon(_icons[settings.themeMode]),
    initialValue: settings.themeMode,
    onSelected: settings.setThemeMode,
    itemBuilder: (context) => [
      for (final m in ThemeMode.values)
        PopupMenuItem(
          value: m,
          child: ListTile(
            leading: Icon(_icons[m]),
            title: Text(_names[m]!),
            contentPadding: EdgeInsets.zero,
          ),
        ),
    ],
  );
}

/// The current value of a node, for the graph's captions.
String? nodeCaption(Playground pg, int node, int? t) {
  final traj = pg.trajectories;
  if (traj == null) return null;
  final end = pg.player?.t ?? traj.nTime;
  final at = t ?? end - 1;
  if (at < 0 || at >= end) return null;
  final v = traj.at(node, NodeField.mean, at);
  if (!v.isFinite) return null;
  return 'μ ${v.toStringAsFixed(2)}';
}

/// A listenable that fires when [select] of [source] changes.
class _Selected extends ChangeNotifier {
  _Selected(this.source, this.select) : _last = select() {
    source.addListener(_check);
  }

  final Listenable source;
  final Object? Function() select;
  Object? _last;

  void _check() {
    final v = select();
    if (v == _last) return;
    _last = v;
    notifyListeners();
  }

  @override
  void dispose() {
    source.removeListener(_check);
    super.dispose();
  }
}
