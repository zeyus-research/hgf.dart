import 'dart:convert';
import 'dart:math' as math;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:hgf_widgets/hgf_widgets.dart';

import '../model/data.dart';
import '../model/fitting.dart';
import 'panels.dart';

/// Parameter estimation: what to fit, which parameters are free, the
/// method, and the results (estimates, intervals, prior vs posterior).
class FitCard extends StatelessWidget {
  const FitCard({super.key, required this.fitting});

  final Fitting fitting;

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
    if (f.unsupported != null) {
      return SectionCard(
        title: 'Fit parameters',
        children: [Text(f.unsupported!, style: small)],
      );
    }
    final offered = f.offered;
    return SectionCard(
      title: 'Fit parameters',
      children: [
        SegmentedButton<FitTarget>(
          showSelectedIcon: false,
          segments: const [
            ButtonSegment(
              value: FitTarget.observations,
              label: Text('to the observations'),
            ),
            ButtonSegment(
              value: FitTarget.responses,
              label: Text('to responses'),
            ),
          ],
          selected: {f.target},
          onSelectionChanged: f.running ? null : (s) => f.setTarget(s.single),
        ),
        const SizedBox(height: 6),
        if (f.target == FitTarget.observations)
          Text(
            'Finds the parameters under which the observations are least '
            'surprising (the bottom panel, summed, plus the prior): what '
            'an ideal observer would have assumed.',
            style: small,
          )
        else
          ..._responses(context),
        const SizedBox(height: 8),
        Text('Free parameters', style: theme.textTheme.labelLarge),
        const SizedBox(height: 4),
        Wrap(
          spacing: 6,
          runSpacing: 6,
          children: [
            for (final c in offered)
              Tooltip(
                message: '${c.help}\nPrior: ${c.priorText}',
                child: FilterChip(
                  label: Text(c.symbol),
                  selected: f.free.contains(c.id),
                  onSelected: f.running ? null : (on) => f.toggle(c.id, on),
                  visualDensity: VisualDensity.compact,
                ),
              ),
          ],
        ),
        Text(
          'The others stay at the sliders\' values. Hover a parameter for '
          'its prior.',
          style: muted,
        ),
        const SizedBox(height: 8),
        Tooltip(
          message:
              'MAP: the most probable values and a Gaussian around them '
              '(fast). NUTS: draws from the full posterior with Hamiltonian '
              'Monte Carlo (4 chains × 1000 steps).',
          child: SegmentedButton<FitMethod>(
            showSelectedIcon: false,
            segments: const [
              ButtonSegment(value: FitMethod.map, label: Text('MAP + Laplace')),
              ButtonSegment(value: FitMethod.nuts, label: Text('NUTS')),
            ],
            selected: {f.method},
            onSelectionChanged: f.running ? null : (s) => f.setMethod(s.single),
          ),
        ),
        const SizedBox(height: 8),
        _runRow(context),
        if (f.error != null)
          Padding(
            padding: const EdgeInsets.only(top: 6),
            child: Text(
              f.error!,
              style: small?.copyWith(color: theme.colorScheme.error),
            ),
          ),
        if (f.result != null) ..._results(context, f.result!),
      ],
    );
  }

  List<Widget> _responses(BuildContext context) {
    final f = fitting;
    final theme = Theme.of(context);
    final small = theme.textTheme.bodySmall;
    final r = f.responses;
    return [
      Text(
        f.binary
            ? 'Recover an agent\'s parameters from its choices: each '
                  'trial it predicts outcome 1 with the probability μ̂ of '
                  'x₀, sharpened by β.'
            : 'Recover an agent\'s parameters from its responses: each '
                  'trial it reports its prediction μ̂ of x₀, plus noise.',
        style: small,
      ),
      LabeledSlider(
        label: f.binary ? 'agent β' : 'agent noise sd',
        help: f.binary
            ? 'The simulated agent\'s inverse temperature.'
            : 'The standard deviation of the simulated agent\'s responses '
                  'around its prediction.',
        value: math.log(f.binary ? f.agentBeta : f.agentNoise) / math.ln10,
        min: f.binary ? -1 : -2.5,
        max: f.binary ? 1.5 : 0,
        display: fmt(f.binary ? f.agentBeta : f.agentNoise),
        onChanged: (v) {
          final x = math.pow(10, v).toDouble();
          f.binary ? f.setAgent(beta: x) : f.setAgent(noise: x);
        },
      ),
      Row(
        children: [
          Expanded(
            child: Text(
              r == null
                  ? 'No responses yet.'
                  : '${f.responsesSource} · '
                        '${r.where((v) => v.isFinite).length} responses',
              style: small,
            ),
          ),
          TextButton.icon(
            onPressed: f.running ? null : () => f.simulateAgent(),
            icon: const Icon(Icons.person_outline, size: 18),
            label: const Text('Simulate'),
          ),
          IconButton(
            tooltip: 'Load responses (one per trial, or the first column)',
            onPressed: f.running ? null : () => _load(context),
            icon: const Icon(Icons.file_open_outlined, size: 18),
          ),
        ],
      ),
      Text(
        'Simulate draws responses from an agent with the sliders\' current '
        'values: those become the truth to recover. Then move the sliders '
        'away, or don\'t, and fit.',
        style: small?.copyWith(color: theme.colorScheme.onSurfaceVariant),
      ),
    ];
  }

  Future<void> _load(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    try {
      final file = await FilePicker.pickFile(
        dialogTitle: 'Responses (one per trial, or the first column)',
        type: FileType.custom,
        allowedExtensions: const ['csv', 'tsv', 'txt'],
      );
      if (file == null) return;
      final text = utf8.decode(await file.readAsBytes(), allowMalformed: true);
      fitting.loadResponses(parseColumn(text), file.name);
    } catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Could not load: $e')));
    }
  }

  Widget _runRow(BuildContext context) {
    final f = fitting;
    final theme = Theme.of(context);
    if (!f.running) {
      return Row(
        children: [
          FilledButton.icon(
            onPressed: f.canFit ? f.fit : null,
            icon: const Icon(Icons.play_arrow, size: 18),
            label: const Text('Fit'),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              f.free.isEmpty
                  ? 'Free at least one parameter.'
                  : f.target == FitTarget.responses && f.responses == null
                  ? 'Simulate or load responses first.'
                  : '',
              style: theme.textTheme.bodySmall,
            ),
          ),
        ],
      );
    }
    final p = f.progress;
    return Row(
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(f.stage ?? '', style: theme.textTheme.bodySmall),
              const SizedBox(height: 4),
              LinearProgressIndicator(
                value: p == null || p.total == 0 ? null : p.fraction,
              ),
            ],
          ),
        ),
        const SizedBox(width: 8),
        TextButton(onPressed: f.cancel, child: const Text('Cancel')),
      ],
    );
  }

  List<Widget> _results(BuildContext context, FitOutcome r) {
    final theme = Theme.of(context);
    final small = theme.textTheme.bodySmall;
    final muted = small?.copyWith(color: theme.colorScheme.onSurfaceVariant);
    final hasTruth = r.estimates.any((e) => e.truth != null);
    final stats = [
      if (r.method == FitMethod.map && r.logEvidence != null)
        'log evidence ${r.logEvidence!.isFinite ? r.logEvidence!.toStringAsFixed(1) : '—'}',
      if (r.divergences != null) '${r.divergences} divergences',
      '${r.millis} ms on ${r.where}',
    ];
    return [
      const Divider(height: 20),
      Row(
        children: [
          Expanded(
            child: Text(
              r.method == FitMethod.map
                  ? 'MAP estimates, 95 % Laplace intervals'
                  : 'Posterior means, 95 % intervals (NUTS)',
              style: theme.textTheme.labelLarge,
            ),
          ),
          TextButton(
            onPressed: r.estimates.every((e) => e.param.isResponse)
                ? null
                : fitting.apply,
            child: const Text('Apply'),
          ),
        ],
      ),
      if (r.note != null)
        Padding(
          padding: const EdgeInsets.only(bottom: 4),
          child: Text(
            r.note!,
            style: small?.copyWith(color: theme.colorScheme.error),
          ),
        ),
      for (final e in r.estimates) _EstimateRow(e, method: r.method),
      const SizedBox(height: 4),
      _Legend(truth: hasTruth, nuts: r.method == FitMethod.nuts),
      const SizedBox(height: 4),
      Text(stats.join(' · '), style: muted),
      Text(
        'Apply sets the sliders to the estimates, so the plots show the '
        'fitted model.',
        style: muted,
      ),
    ];
  }
}

class _EstimateRow extends StatelessWidget {
  const _EstimateRow(this.e, {required this.method});

  final ParamEstimate e;
  final FitMethod method;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final tab = theme.textTheme.bodyMedium?.copyWith(
      fontFeatures: const [FontFeature.tabularFigures()],
    );
    final diag = [
      if (e.rhat != null) 'R̂ ${e.rhat!.toStringAsFixed(3)}',
      if (e.ess != null) 'ESS ${e.ess!.toStringAsFixed(0)}',
      if (e.truth != null) 'true ${fmt(e.truth!)}',
    ];
    return Padding(
      padding: const EdgeInsets.only(top: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Expanded(
                child: Tooltip(
                  message: '${e.param.help}\nPrior: ${e.param.priorText}',
                  child: Text(
                    e.param.symbol,
                    style: theme.textTheme.bodyMedium,
                  ),
                ),
              ),
              Text(
                '${fmt(e.estimate)}  [${fmt(e.lower)}, ${fmt(e.upper)}]',
                style: tab,
              ),
            ],
          ),
          SizedBox(
            height: 48,
            child: CustomPaint(
              painter: PosteriorPainter(
                e,
                colors: HgfColors.of(context),
                muted: theme.colorScheme.onSurfaceVariant,
                text: theme.textTheme.labelSmall!,
              ),
            ),
          ),
          if (diag.isNotEmpty)
            Text(
              diag.join(' · '),
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
        ],
      ),
    );
  }
}

class _Legend extends StatelessWidget {
  const _Legend({required this.truth, required this.nuts});

  final bool truth, nuts;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = HgfColors.of(context);
    final muted = theme.colorScheme.onSurfaceVariant;
    final style = theme.textTheme.bodySmall?.copyWith(color: muted);
    Widget key(Color c, String label, {bool line = false}) => Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(width: line ? 2 : 12, height: line ? 12 : 8, color: c),
        const SizedBox(width: 4),
        Text(label, style: style),
      ],
    );
    return Wrap(
      spacing: 12,
      runSpacing: 4,
      children: [
        key(muted.withValues(alpha: 0.5), 'prior'),
        key(colors.belief.withValues(alpha: 0.6), 'posterior'),
        key(colors.belief, nuts ? 'mean' : 'MAP', line: true),
        if (truth) key(colors.observation, 'truth', line: true),
      ],
    );
  }
}

/// One parameter's prior and posterior on its natural scale: the prior
/// density (grey), the posterior (the Laplace density, or a histogram of
/// the NUTS draws) with its 95 % interval, the estimate, and the truth.
class PosteriorPainter extends CustomPainter {
  PosteriorPainter(
    this.e, {
    required this.colors,
    required this.muted,
    required this.text,
  });

  final ParamEstimate e;
  final HgfColors colors;
  final Color muted;
  final TextStyle text;

  static const _labelHeight = 14.0;

  @override
  void paint(Canvas canvas, Size size) {
    final (lo, hi) = _range();
    if (!(hi > lo)) return;
    final h = size.height - _labelHeight;
    final w = size.width;
    double px(double x) => (x - lo) / (hi - lo) * w;
    const n = 120;
    final xs = [for (var i = 0; i <= n; i++) lo + (hi - lo) * i / n];

    // Posterior heights (density or histogram), and the prior on the same
    // scale.
    final post = <double>[];
    var histogram = false;
    final draws = e.draws;
    if (draws != null && draws.isNotEmpty) {
      histogram = true;
      const bins = 40;
      final counts = List<double>.filled(bins, 0);
      for (final d in draws) {
        final b = ((d - lo) / (hi - lo) * bins).floor();
        if (b >= 0 && b < bins) counts[b]++;
      }
      final width = (hi - lo) / bins;
      for (final c in counts) {
        post.add(c / (draws.length * width));
      }
    } else if (e.density != null) {
      for (final x in xs) {
        post.add(e.density!(x));
      }
    }
    final prior = [for (final x in xs) math.exp(e.param.prior.logDensity(x))];
    var top = 0.0;
    for (final v in [...post, ...prior]) {
      if (v.isFinite && v > top) top = v;
    }
    if (top <= 0) top = 1;
    double py(double d) => h - (d.isFinite ? d : 0) / top * (h - 2);

    final base = Paint()
      ..color = muted.withValues(alpha: 0.4)
      ..strokeWidth = 1;
    canvas.drawLine(Offset(0, h), Offset(w, h), base);

    // 95 % interval.
    canvas.drawRect(
      Rect.fromLTRB(px(e.lower), h - 3, px(e.upper), h),
      Paint()..color = colors.belief.withValues(alpha: 0.35),
    );

    // Prior.
    final priorPath = Path()..moveTo(0, py(prior.first));
    for (var i = 1; i <= n; i++) {
      priorPath.lineTo(px(xs[i]), py(prior[i]));
    }
    canvas.drawPath(
      priorPath,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.2
        ..color = muted.withValues(alpha: 0.6),
    );

    // Posterior.
    if (post.isNotEmpty) {
      final fill = Paint()..color = colors.belief.withValues(alpha: 0.35);
      final stroke = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.2
        ..color = colors.belief;
      if (histogram) {
        final bw = w / post.length;
        final path = Path()..moveTo(0, h);
        for (var b = 0; b < post.length; b++) {
          path
            ..lineTo(b * bw, py(post[b]))
            ..lineTo((b + 1) * bw, py(post[b]));
        }
        path.lineTo(w, h);
        canvas
          ..drawPath(path, fill)
          ..drawPath(path, stroke);
      } else {
        final path = Path()..moveTo(0, h);
        for (var i = 0; i <= n; i++) {
          path.lineTo(px(xs[i]), py(post[i]));
        }
        path.lineTo(w, h);
        canvas
          ..drawPath(path, fill)
          ..drawPath(path, stroke);
      }
    }

    // Estimate and truth.
    canvas.drawLine(
      Offset(px(e.estimate), 0),
      Offset(px(e.estimate), h),
      Paint()
        ..color = colors.belief
        ..strokeWidth = 2,
    );
    final t = e.truth;
    if (t != null && t.isFinite) {
      final x = px(t).clamp(1.0, w - 1);
      final paint = Paint()
        ..color = colors.observation
        ..strokeWidth = 2;
      for (var y = 0.0; y < h; y += 6) {
        canvas.drawLine(Offset(x, y), Offset(x, math.min(y + 3.5, h)), paint);
      }
    }

    // Axis ends.
    _label(canvas, fmt(lo), Offset(0, h + 1), left: true, width: w);
    _label(canvas, fmt(hi), Offset(w, h + 1), left: false, width: w);
  }

  void _label(
    Canvas canvas,
    String s,
    Offset at, {
    required bool left,
    required double width,
  }) {
    final tp = TextPainter(
      text: TextSpan(
        text: s,
        style: text.copyWith(color: muted, fontSize: 10),
      ),
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: width / 2);
    tp.paint(canvas, left ? at : at - Offset(tp.width, 0));
    tp.dispose();
  }

  /// The x range: the posterior's interval widened on both sides, enough
  /// to show the estimate and the truth.
  (double, double) _range() {
    var lo = e.lower, hi = e.upper;
    final draws = e.draws;
    if (draws != null && draws.isNotEmpty) {
      // Sorted: the central 99 %, so a few far draws don't squash the
      // histogram.
      final n = draws.length - 1;
      lo = math.min(lo, draws[(n * 0.005).round()]);
      hi = math.max(hi, draws[(n * 0.995).round()]);
    }
    for (final v in [e.estimate, ?e.truth]) {
      if (v.isFinite) {
        lo = math.min(lo, v);
        hi = math.max(hi, v);
      }
    }
    if (!lo.isFinite || !hi.isFinite) {
      final c = e.estimate.isFinite ? e.estimate : 0.0;
      return (c - 1, c + 1);
    }
    var span = hi - lo;
    if (span <= 0) span = math.max(1e-3, lo.abs() * 0.1);
    lo -= span * 0.4;
    hi += span * 0.4;
    // Positive parameters stay positive.
    if (!e.param.prior.contains(-1e-9) && lo < 0) {
      lo = math.max(0, e.lower - (hi - e.lower) * 0.05);
    }
    return (lo, hi);
  }

  @override
  bool shouldRepaint(PosteriorPainter old) =>
      old.e != e || old.colors != colors || old.muted != muted;
}
