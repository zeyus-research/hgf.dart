import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../model/playground.dart';
import 'frame_stats.dart';

/// Trial-by-trial playback controls, the total surprise and timings.
class PlaybackBar extends StatelessWidget {
  const PlaybackBar({
    super.key,
    required this.playground,
    required this.frames,
  });

  final Playground playground;
  final FrameStats frames;

  @override
  Widget build(BuildContext context) {
    final pg = playground;
    final n = pg.observations.length;
    final t = pg.player?.t;
    final theme = Theme.of(context);
    final small = theme.textTheme.bodySmall?.copyWith(
      fontFeatures: const [FontFeature.tabularFigures()],
    );
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      child: Wrap(
        crossAxisAlignment: WrapCrossAlignment.center,
        spacing: 4,
        runSpacing: 4,
        children: [
          IconButton.filledTonal(
            tooltip: pg.playing ? 'Pause' : 'Play trial by trial',
            icon: Icon(pg.playing ? Icons.pause : Icons.play_arrow),
            onPressed: n == 0 ? null : (pg.playing ? pg.pause : pg.play),
          ),
          IconButton(
            tooltip: 'Next trial',
            icon: const Icon(Icons.skip_next),
            onPressed: n == 0 || (t != null && t >= n) ? null : pg.stepOnce,
          ),
          IconButton(
            tooltip: 'Show the whole series',
            icon: const Icon(Icons.stop),
            onPressed: pg.player == null ? null : pg.stop,
          ),
          SizedBox(
            width: 92,
            child: Text(
              t == null ? '$n trials' : 'trial $t / $n',
              style: small,
            ),
          ),
          Tooltip(
            message: 'Playback speed (trials per second)',
            child: SizedBox(
              width: 150,
              child: Row(
                children: [
                  const Icon(Icons.speed, size: 18),
                  Expanded(
                    child: Slider(
                      // log₁₀ of trials per second, 1 to 1000.
                      value: math.log(pg.speed) / math.ln10,
                      min: 0,
                      max: 3,
                      onChanged: (v) => pg.setSpeed(math.pow(10, v).toDouble()),
                    ),
                  ),
                  SizedBox(
                    width: 34,
                    child: Text(pg.speed.round().toString(), style: small),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(width: 12),
          _Stat(
            label: 'Σ surprise',
            value: pg.totalSurprise.isFinite
                ? pg.totalSurprise.toStringAsFixed(1)
                : '—',
            tooltip:
                'Total surprise of the observations so far (the negative '
                'log-likelihood that fitting minimises).',
          ),
          const SizedBox(width: 12),
          ListenableBuilder(
            listenable: frames,
            builder: (context, _) => _Stat(
              label: 'filter',
              value:
                  '${(pg.filterMicros / 1000).toStringAsFixed(2)} ms · '
                  '${kIsWeb ? 'frame ${frames.intervalMs.toStringAsFixed(1)} ms' : 'frame ${frames.buildMs.toStringAsFixed(1)} + ${frames.rasterMs.toStringAsFixed(1)} ms'}',
              tooltip: kIsWeb
                  ? 'Filter: time to re-filter the whole series on the last '
                        'change. Frame: mean time between frames while '
                        'something moves (16.7 ms at 60 Hz).'
                  : 'Filter: time to re-filter the whole series on the last '
                        'change. Frame: mean build (UI thread) + raster time '
                        'of the last ${frames.frames} frames (worst '
                        '${frames.worstMs.toStringAsFixed(1)} ms).',
            ),
          ),
        ],
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat({required this.label, required this.value, this.tooltip});

  final String label;
  final String value;
  final String? tooltip;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final child = Text.rich(
      TextSpan(
        children: [
          TextSpan(
            text: '$label ',
            style: TextStyle(color: theme.colorScheme.onSurfaceVariant),
          ),
          TextSpan(text: value),
        ],
      ),
      style: theme.textTheme.bodySmall?.copyWith(
        fontFeatures: const [FontFeature.tabularFigures()],
      ),
    );
    return tooltip == null ? child : Tooltip(message: tooltip, child: child);
  }
}
