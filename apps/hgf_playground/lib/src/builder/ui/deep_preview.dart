import 'dart:typed_data';

import 'package:hgf_deep/hgf_deep.dart' show LayerKind;
import 'package:hgf_fit/document.dart';
import 'package:hgf_widgets/hgf_widgets.dart';
import 'package:material_ui/material_ui.dart';

import '../builder_model.dart';

/// A deep network trained on the loaded data: per target (or category),
/// the targets of the shown participant's rows as points (training rows
/// and held-out rows in two colours) and the network's prediction as a
/// line; under them the training rows' score after each pass.
class DeepPreview extends StatelessWidget {
  const DeepPreview({
    super.key,
    required this.model,
    required this.deep,
    required this.hover,
    this.panelHeight,
  });

  final BuilderModel model;
  final DocDeep deep;
  final ValueNotifier<int?> hover;
  final double? panelHeight;

  /// At most this many targets (or categories) are plotted.
  static const maxTargets = 4;

  /// How many plots [DeepPreview] draws for [d].
  static int panelCount(BuilderModel m, DocDeep d) {
    final r = m.deepResult(d.id);
    if (r == null) return 1;
    return r.predictions.cols.clamp(1, maxTargets) +
        (r.epochScores.length > 1 ? 1 : 0);
  }

  /// The ids of the participants [r] read, in order of first appearance.
  static List<String> participantsOf(DeepResult r) =>
      r.data.participants.toSet().toList();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final muted = theme.colorScheme.onSurfaceVariant;
    final r = model.deepResult(deep.id);
    if (r == null) {
      final run = model.deepRun(deep.id);
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (run != null)
                SizedBox(
                  width: 240,
                  child: LinearProgressIndicator(value: run.progress),
                ),
              const SizedBox(height: 8),
              Text(
                '"${deep.label}": ${model.deepStatus(deep.id)}',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium?.copyWith(color: muted),
              ),
            ],
          ),
        ),
      );
    }
    final colors = HgfColors.of(context);
    // The shown participant's rows (by id: rows with a gap are skipped,
    // so the network may have read fewer participants than the data has).
    final ids = participantsOf(r);
    final shown = model.participants.isEmpty
        ? (ids.isEmpty ? '' : ids.first)
        : model.participants[model.participant.clamp(
            0,
            model.participants.length - 1,
          )];
    final rows = [
      for (var i = 0; i < r.data.length; i++)
        if (r.data.participants[i] == shown || ids.length == 1) i,
    ];
    final cats = deep.categories;
    final names = r.columns;
    final continuous = deep.isContinuous;
    final regression =
        continuous || deep.targetLayer?.kind == LayerKind.volatile;
    Float64List series(double Function(int i) f) =>
        Float64List.fromList([for (final i in rows) f(i)]);
    final panels = <Widget>[];
    final n = r.predictions.cols.clamp(1, maxTargets);
    for (var j = 0; j < n; j++) {
      final target = cats != null
          ? '${deep.targets.single} = ${cats[j]}'
          : continuous || deep.targets.length > j
          ? deep.targets[j]
          : names[j];
      panels.add(
        TrajectoryPlot(
          title: '$target · ${names[j]}',
          layers: [
            PointLayer(
              series((i) => r.data.train[i] ? r.data.y.at(i, j) : double.nan),
              color: muted.withValues(alpha: 0.7),
              radius: 1.6,
              label: continuous ? 'observed' : 'target (training)',
            ),
            if (!continuous && r.data.testIndices.isNotEmpty)
              PointLayer(
                series((i) => r.data.train[i] ? double.nan : r.data.y.at(i, j)),
                color: colors.category(2),
                radius: 1.8,
                label: 'target (held out)',
              ),
            LineLayer(
              series((i) => r.predictions.at(i, j)),
              color: colors.category(0),
              label: 'prediction',
            ),
          ],
          yRange: regression ? null : (-0.05, 1.05),
          hover: hover,
          revision: model.revision,
          showXAxis: j == n - 1 && r.epochScores.length <= 1,
        ),
      );
    }
    if (r.epochScores.length > 1) {
      panels.add(
        TrajectoryPlot(
          title: '${r.epochScores.first.name} on the training rows, by pass',
          layers: [
            LineLayer(
              Float64List.fromList([for (final s in r.epochScores) s.value]),
              color: colors.category(1),
              label: r.epochScores.first.name,
            ),
          ],
          revision: model.revision,
        ),
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.only(bottom: 4),
          child: Text(
            '"${deep.label}" · ${model.deepStatus(deep.id)}'
            '${ids.length > 1 ? ' · participant $shown, ${rows.length} rows' : ''}',
            style: theme.textTheme.bodySmall,
            overflow: TextOverflow.ellipsis,
          ),
        ),
        for (final p in panels)
          panelHeight == null
              ? Expanded(child: p)
              : SizedBox(height: panelHeight, child: p),
      ],
    );
  }
}
