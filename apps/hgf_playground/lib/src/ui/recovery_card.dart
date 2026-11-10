import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:hgf_widgets/hgf_widgets.dart';

import '../model/fitting.dart';
import 'panels.dart';

/// Parameter recovery: simulate many agents with known parameters, fit
/// them all, and plot the estimates against the truth.
class RecoveryCard extends StatelessWidget {
  const RecoveryCard({super.key, required this.fitting, this.onOpenSettings});

  final Fitting fitting;
  final VoidCallback? onOpenSettings;

  @override
  Widget build(BuildContext context) => ListenableBuilder(
    listenable: fitting,
    builder: (context, _) => _build(context),
  );

  Widget _build(BuildContext context) {
    final f = fitting;
    final theme = Theme.of(context);
    final small = theme.textTheme.bodySmall;
    final muted = small?.copyWith(color: theme.colorScheme.onSurfaceVariant);
    if (f.unsupported != null) return const SizedBox.shrink();
    if (f.target != FitTarget.responses) {
      return SectionCard(
        title: 'Parameter recovery',
        children: [
          Text(
            'Can an experiment like this one tell the parameters apart? '
            'Recovery simulates many agents and fits their responses.',
            style: small,
          ),
          Align(
            alignment: Alignment.centerLeft,
            child: TextButton(
              onPressed: f.running
                  ? null
                  : () => f.setTarget(FitTarget.responses),
              child: const Text('Fit to responses'),
            ),
          ),
        ],
      );
    }
    final c = f.config;
    final runs = f.recoveryAgents * (c.restarts + 1);
    final where = !f.recoveryWantsGpu
        ? 'CPU'
        : f.gpu != null && f.gpu!.adapter == null
        ? 'CPU (no GPU)'
        : 'GPU';
    final r = f.recovery;
    return SectionCard(
      title: 'Parameter recovery',
      children: [
        Text(
          'Simulates agents on these observations, each with its free '
          'parameters drawn from their priors (the others at the sliders\' '
          'values), fits every agent by MAP, and plots the estimates '
          'against the truth: recovered parameters lie on the diagonal.',
          style: small,
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Text('Agents', style: theme.textTheme.labelLarge),
            const SizedBox(width: 12),
            Expanded(
              child: SegmentedButton<int>(
                showSelectedIcon: false,
                segments: [
                  for (final n in Fitting.recoveryAgentChoices)
                    ButtonSegment(value: n, label: Text('$n')),
                ],
                selected: {f.recoveryAgents},
                onSelectionChanged: f.running
                    ? null
                    : (s) => f.setRecoveryAgents(s.single),
              ),
            ),
          ],
        ),
        Row(
          children: [
            Expanded(
              child: Text(
                '$runs L-BFGS ${runs == 1 ? 'run' : 'runs'} '
                '(${c.restarts + 1} per agent) · $where',
                style: muted,
              ),
            ),
            TextButton.icon(
              onPressed: onOpenSettings,
              icon: const Icon(Icons.tune, size: 16),
              label: const Text('Settings'),
            ),
          ],
        ),
        _runRow(context),
        if (f.error != null && f.lastJob == FitJob.recovery)
          Padding(
            padding: const EdgeInsets.only(top: 6),
            child: Text(
              f.error!,
              style: small?.copyWith(color: theme.colorScheme.error),
            ),
          ),
        if (r != null) ..._results(context, r),
      ],
    );
  }

  Widget _runRow(BuildContext context) {
    final f = fitting;
    final theme = Theme.of(context);
    if (f.runningJob != FitJob.recovery) {
      return Row(
        children: [
          FilledButton.icon(
            onPressed: f.canRecover && !f.running ? f.recover : null,
            icon: const Icon(Icons.scatter_plot_outlined, size: 18),
            label: const Text('Run recovery'),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              f.running
                  ? 'A fit is running.'
                  : !f.canRecover
                  ? 'Free at least one parameter.'
                  : '',
              style: theme.textTheme.bodySmall,
            ),
          ),
        ],
      );
    }
    return Row(
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(f.stage ?? '', style: theme.textTheme.bodySmall),
              const SizedBox(height: 4),
              const LinearProgressIndicator(),
            ],
          ),
        ),
        const SizedBox(width: 8),
        TextButton(onPressed: f.cancel, child: const Text('Cancel')),
      ],
    );
  }

  List<Widget> _results(BuildContext context, RecoveryOutcome r) {
    final theme = Theme.of(context);
    final muted = theme.textTheme.bodySmall?.copyWith(
      color: theme.colorScheme.onSurfaceVariant,
    );
    final stats = [
      '${r.agents} agents',
      if (r.failed > 0) '${r.failed} without a finite fit',
      if (r.unconverged > 0) '${r.unconverged} not converged',
      if (r.redrawn > 0) '${r.redrawn} redrawn',
      if (r.dropped > 0) '${r.dropped} left out',
      '${r.millis} ms on ${r.where}',
    ];
    return [
      const Divider(height: 20),
      if (r.deviceNote != null)
        Padding(
          padding: const EdgeInsets.only(bottom: 4),
          child: Text(r.deviceNote!, style: muted),
        ),
      LayoutBuilder(
        builder: (context, box) {
          final cols = box.maxWidth >= 300 ? 2 : 1;
          final w = (box.maxWidth - 12 * (cols - 1)) / cols;
          return Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              for (var i = 0; i < r.params.length; i++)
                SizedBox(width: w, child: _RecoveryPlot(r, i)),
            ],
          );
        },
      ),
      const SizedBox(height: 6),
      Text(stats.join(' · '), style: muted),
      Text(
        'r: correlation of truth and estimate; RMSE on the plotted scale '
        '(log for positive parameters). Redrawn: agents whose beliefs '
        'diverged at their drawn values.',
        style: muted,
      ),
    ];
  }
}

class _RecoveryPlot extends StatelessWidget {
  const _RecoveryPlot(this.r, this.i);

  final RecoveryOutcome r;
  final int i;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final p = r.params[i];
    final s = r.stats[i];
    final tab = theme.textTheme.bodySmall?.copyWith(
      fontFeatures: const [FontFeature.tabularFigures()],
    );
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Tooltip(
          message: '${p.help}\nPrior: ${p.priorText}',
          child: Text(p.symbol, style: theme.textTheme.bodyMedium),
        ),
        Text(
          s.r.isFinite
              ? 'r ${s.r.toStringAsFixed(2)} · RMSE ${fmt(s.rmse)}'
              : 'r —',
          style: tab,
        ),
        const SizedBox(height: 4),
        AspectRatio(
          aspectRatio: 1,
          child: CustomPaint(
            painter: RecoveryPainter(
              truth: r.truth[i],
              estimate: r.estimate[i],
              ok: r.ok,
              log: p.logScale,
              colors: HgfColors.of(context),
              muted: theme.colorScheme.onSurfaceVariant,
              text: theme.textTheme.labelSmall!,
            ),
          ),
        ),
      ],
    );
  }
}

/// Truth (x) against estimate (y) for one parameter, with the identity
/// line; log axes for positive parameters.
class RecoveryPainter extends CustomPainter {
  RecoveryPainter({
    required this.truth,
    required this.estimate,
    required this.ok,
    required this.log,
    required this.colors,
    required this.muted,
    required this.text,
  });

  final Float64List truth, estimate;
  final List<bool> ok;
  final bool log;
  final HgfColors colors;
  final Color muted;
  final TextStyle text;

  static const _pad = 14.0;

  double _s(double v) => log ? math.log(v) : v;

  @override
  void paint(Canvas canvas, Size size) {
    // A common range for both axes, so the diagonal is the identity.
    var lo = double.infinity, hi = double.negativeInfinity;
    for (var j = 0; j < truth.length; j++) {
      if (!ok[j]) continue;
      for (final v in [_s(truth[j]), _s(estimate[j])]) {
        if (!v.isFinite) continue;
        lo = math.min(lo, v);
        hi = math.max(hi, v);
      }
    }
    if (!lo.isFinite) return;
    if (!(hi > lo)) {
      lo -= 0.5;
      hi += 0.5;
    }
    final span = hi - lo;
    lo -= span * 0.04;
    hi += span * 0.04;
    final w = size.width - _pad, h = size.height - _pad;
    double px(double v) => _pad + (v - lo) / (hi - lo) * w;
    double py(double v) => h - (v - lo) / (hi - lo) * h;

    final axis = Paint()
      ..color = muted.withValues(alpha: 0.4)
      ..strokeWidth = 1;
    canvas
      ..drawLine(Offset(_pad, h), Offset(size.width, h), axis)
      ..drawLine(Offset(_pad, 0), Offset(_pad, h), axis)
      ..drawLine(
        Offset(px(lo), py(lo)),
        Offset(px(hi), py(hi)),
        Paint()
          ..color = muted.withValues(alpha: 0.7)
          ..strokeWidth = 1,
      );

    final n = truth.length;
    final radius = n > 1000 ? 1.4 : (n > 200 ? 1.8 : 2.4);
    final alpha = n > 1000 ? 0.35 : (n > 200 ? 0.5 : 0.75);
    final dot = Paint()..color = colors.belief.withValues(alpha: alpha);
    for (var j = 0; j < n; j++) {
      if (!ok[j]) continue;
      final x = _s(truth[j]), y = _s(estimate[j]);
      if (!x.isFinite || !y.isFinite) continue;
      canvas.drawCircle(Offset(px(x), py(y)), radius, dot);
    }

    String label(double v) => fmt(log ? math.exp(v) : v);
    _text(canvas, label(lo), Offset(_pad, h + 1), size.width);
    _text(
      canvas,
      label(hi),
      Offset(size.width, h + 1),
      size.width,
      right: true,
    );
    _text(
      canvas,
      'true →',
      Offset(size.width / 2 + _pad / 2, h + 1),
      size.width,
      centre: true,
    );
    canvas
      ..save()
      ..translate(0, h / 2)
      ..rotate(-math.pi / 2);
    _text(canvas, 'estimate →', Offset.zero, h, centre: true);
    canvas.restore();
  }

  void _text(
    Canvas canvas,
    String s,
    Offset at,
    double width, {
    bool right = false,
    bool centre = false,
  }) {
    final tp = TextPainter(
      text: TextSpan(
        text: s,
        style: text.copyWith(color: muted, fontSize: 10),
      ),
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: width / 2);
    final dx = right
        ? -tp.width
        : centre
        ? -tp.width / 2
        : 0.0;
    tp.paint(canvas, at + Offset(dx, 0));
    tp.dispose();
  }

  @override
  bool shouldRepaint(RecoveryPainter old) =>
      !identical(old.truth, truth) ||
      !identical(old.estimate, estimate) ||
      old.colors != colors ||
      old.muted != muted;
}
