import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:hgf_fit/hgf_fit.dart';

import '../model/fit_config.dart';
import '../model/fitting.dart';
import '../settings.dart';

/// The settings drawer: theme, where fits run (CPU workers or the GPU),
/// and the optimiser's and sampler's settings.
class SettingsPanel extends StatefulWidget {
  const SettingsPanel({
    super.key,
    required this.settings,
    required this.fitting,
  });

  final Settings settings;
  final Fitting fitting;

  @override
  State<SettingsPanel> createState() => _SettingsPanelState();
}

class _SettingsPanelState extends State<SettingsPanel> {
  @override
  void initState() {
    super.initState();
    // Find out whether there is a GPU, to say so.
    widget.fitting.probeGpu();
  }

  @override
  Widget build(BuildContext context) => ListenableBuilder(
    listenable: Listenable.merge([widget.settings, widget.fitting]),
    builder: (context, _) => _build(context),
  );

  Widget _build(BuildContext context) {
    final s = widget.settings;
    final c = s.fit;
    final theme = Theme.of(context);
    final small = theme.textTheme.bodySmall?.copyWith(
      color: theme.colorScheme.onSurfaceVariant,
    );
    void set(FitConfig n) {
      s.setFit(n);
      widget.fitting.configure(n);
    }

    final cores = FitWorkers.defaultCount;
    final workerChoices = {
      0,
      1,
      2,
      4,
      cores,
      if (cores < 8) 8,
      if (cores < 16) 16,
    }.toList()..sort();
    final autoChains = FitConfig.autoGpuChains;
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
      children: [
        Row(
          children: [
            Expanded(
              child: Text('Settings', style: theme.textTheme.titleLarge),
            ),
            IconButton(
              tooltip: 'Close',
              onPressed: () => Navigator.of(context).maybePop(),
              icon: const Icon(Icons.close),
            ),
          ],
        ),
        _heading(context, 'Appearance'),
        SegmentedButton<ThemeMode>(
          showSelectedIcon: false,
          segments: const [
            ButtonSegment(
              value: ThemeMode.system,
              icon: Icon(Icons.brightness_auto, size: 18),
              label: Text('System'),
            ),
            ButtonSegment(
              value: ThemeMode.light,
              icon: Icon(Icons.light_mode, size: 18),
              label: Text('Light'),
            ),
            ButtonSegment(
              value: ThemeMode.dark,
              icon: Icon(Icons.dark_mode, size: 18),
              label: Text('Dark'),
            ),
          ],
          selected: {s.themeMode},
          onSelectionChanged: (m) => s.setThemeMode(m.single),
        ),
        _heading(context, 'Compute'),
        SegmentedButton<FitDevice>(
          showSelectedIcon: false,
          segments: const [
            ButtonSegment(value: FitDevice.auto, label: Text('Auto')),
            ButtonSegment(value: FitDevice.cpu, label: Text('CPU (f64)')),
            ButtonSegment(value: FitDevice.gpu, label: Text('GPU (f32)')),
          ],
          selected: {c.device},
          onSelectionChanged: (d) => set(c.copyWith(device: d.single)),
        ),
        const SizedBox(height: 6),
        Text(switch (c.device) {
          FitDevice.auto =>
            'Auto fits on the CPU, except NUTS with $autoChains or more '
                'chains, which runs on the GPU if there is one: one series '
                'is a small batch, and each sampler step costs the GPU a '
                'round trip.',
          FitDevice.cpu =>
            'Every pass on the CPU workers, in double precision.',
          FitDevice.gpu =>
            'The likelihood and its gradient on the GPU, in single '
                'precision (≈1e-6 relative); Laplace\'s Hessian and the '
                'reported values on the CPU. The GPU pays off with many '
                'chains: each sampler step is one round trip.',
        }, style: small),
        const SizedBox(height: 6),
        _GpuStatus(fitting: widget.fitting),
        const SizedBox(height: 8),
        _ChoiceRow<int>(
          label: kIsWeb ? 'Web Workers' : 'Worker isolates',
          help:
              'CPU workers the fits run on. Each NUTS chain runs on one, '
              'so more workers than chains don\'t help NUTS. Applies to '
              'the next fit.',
          value: workerChoices.contains(c.workers) ? c.workers : 0,
          choices: workerChoices,
          format: (n) => n == 0 ? 'Auto ($cores)' : '$n',
          onChanged: (n) => set(c.copyWith(workers: n)),
        ),
        _heading(context, 'MAP'),
        _ChoiceRow<int>(
          label: 'Restarts',
          help:
              'Extra optimiser starts drawn from the priors; the best fit '
              'is kept. Guards against local optima.',
          value: c.restarts,
          choices: FitConfig.restartChoices,
          format: (n) => '$n',
          onChanged: (n) => set(c.copyWith(restarts: n)),
        ),
        _heading(context, 'NUTS'),
        _ChoiceRow<int>(
          label: 'Chains',
          help:
              'Independent chains, run in lock step. More chains give '
              'more draws and a better R̂; on the GPU they are nearly free.',
          value: c.chains,
          choices: FitConfig.chainChoices,
          format: (n) => '$n',
          onChanged: (n) => set(c.copyWith(chains: n)),
        ),
        _ChoiceRow<int>(
          label: 'Warm-up',
          help:
              'Iterations per chain that adapt the step size and metric; '
              'discarded.',
          value: c.warmup,
          choices: FitConfig.iterationChoices,
          format: (n) => '$n',
          onChanged: (n) => set(c.copyWith(warmup: n)),
        ),
        _ChoiceRow<int>(
          label: 'Draws',
          help: 'Iterations per chain kept as posterior draws.',
          value: c.draws,
          choices: FitConfig.iterationChoices,
          format: (n) => '$n',
          onChanged: (n) => set(c.copyWith(draws: n)),
        ),
        _ChoiceRow<double>(
          label: 'Target acceptance',
          help:
              'Higher: smaller steps, fewer divergences, slower. Raise it '
              'if a fit reports divergent transitions.',
          value: c.targetAccept,
          choices: FitConfig.targetAcceptChoices,
          format: (v) => v.toStringAsFixed(2),
          onChanged: (v) => set(c.copyWith(targetAccept: v)),
        ),
        _ChoiceRow<int>(
          label: 'Max tree depth',
          help:
              'At most 2^depth leapfrog steps per draw. Fits that hit it '
              'explore slowly.',
          value: c.maxTreeDepth,
          choices: FitConfig.treeDepthChoices,
          format: (n) => '$n',
          onChanged: (n) => set(c.copyWith(maxTreeDepth: n)),
        ),
        const SizedBox(height: 4),
        Text(
          '${c.chains} × (${c.warmup} + ${c.draws}) iterations; '
          '${c.chains * c.draws} draws in all.',
          style: small,
        ),
        const SizedBox(height: 16),
        Align(
          alignment: Alignment.centerLeft,
          child: OutlinedButton.icon(
            onPressed: c == const FitConfig()
                ? null
                : () => set(const FitConfig()),
            icon: const Icon(Icons.restart_alt, size: 18),
            label: const Text('Reset fitting defaults'),
          ),
        ),
      ],
    );
  }

  Widget _heading(BuildContext context, String text) => Padding(
    padding: const EdgeInsets.only(top: 20, bottom: 8),
    child: Text(text, style: Theme.of(context).textTheme.titleSmall),
  );
}

class _GpuStatus extends StatelessWidget {
  const _GpuStatus({required this.fitting});

  final Fitting fitting;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final g = fitting.gpu;
    final (icon, color, text) = g == null
        ? (
            Icons.hourglass_empty,
            theme.colorScheme.onSurfaceVariant,
            'Looking for a GPU…',
          )
        : g.adapter != null
        ? (Icons.memory, theme.colorScheme.primary, g.adapter!)
        : (Icons.block, theme.colorScheme.error, 'No GPU: ${g.reason}');
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 16, color: color),
        const SizedBox(width: 6),
        Expanded(
          child: Text(
            text,
            style: theme.textTheme.bodySmall?.copyWith(color: color),
          ),
        ),
      ],
    );
  }
}

/// `label ............ [value ▾]`, with the help as a tooltip.
class _ChoiceRow<T> extends StatelessWidget {
  const _ChoiceRow({
    super.key,
    required this.label,
    required this.help,
    required this.value,
    required this.choices,
    required this.format,
    required this.onChanged,
  });

  final String label, help;
  final T value;
  final List<T> choices;
  final String Function(T) format;
  final ValueChanged<T> onChanged;

  @override
  Widget build(BuildContext context) {
    final items = choices.contains(value) ? choices : [value, ...choices];
    return Tooltip(
      message: help,
      waitDuration: const Duration(milliseconds: 400),
      child: Row(
        children: [
          Expanded(child: Text(label)),
          DropdownButton<T>(
            value: value,
            isDense: true,
            underline: const SizedBox.shrink(),
            items: [
              for (final v in items)
                DropdownMenuItem(value: v, child: Text(format(v))),
            ],
            onChanged: (v) {
              if (v != null) onChanged(v);
            },
          ),
        ],
      ),
    );
  }
}
