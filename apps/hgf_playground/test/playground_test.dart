import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_playground/src/app.dart';
import 'package:hgf_playground/src/model/data.dart';
import 'package:hgf_playground/src/model/playground.dart';
import 'package:hgf_playground/src/model/presets.dart';
import 'package:hgf_playground/src/settings.dart';
import 'package:hgf_widgets/hgf_widgets.dart';

void main() {
  group('Playground', () {
    for (final p in presets) {
      test('${p.id}: playback (predict/step) records what inputData does', () {
        final pg = Playground(presetId: p.id, nTrials: 120);
        final whole = pg.trajectories!;
        final expected = {
          for (final f in whole.fields) f: Float64List.fromList(whole.raw(f)),
        };
        final expectedVectors = {
          for (final f in whole.vectorFields)
            f: Float64List.fromList(whole.rawVector(f)),
        };
        pg.stepOnce();
        pg.player!.advance(1000);
        expect(pg.player!.t, 120);
        for (final f in whole.fields) {
          expect(pg.trajectories!.raw(f), expected[f], reason: '$f');
        }
        for (final f in whole.vectorFields) {
          expect(
            pg.trajectories!.rawVector(f),
            expectedVectors[f],
            reason: '$f',
          );
        }
        expect(pg.error, isNull);
      });
    }

    test('categorical preset: one-hot input, per-category hidden truth', () {
      final pg = Playground(presetId: 'categorical3', nTrials: 200);
      expect(pg.nCategories, 3);
      expect(pg.hidden, hasLength(3));
      for (var t = 0; t < 200; t++) {
        final p = pg.hidden!.fold(0.0, (s, h) => s + h[t]);
        expect(p, closeTo(1, 1e-12));
      }
      expect(pg.observations.every((v) => v == 0 || v == 1 || v == 2), isTrue);
      // The binary node of the observed category saw a 1.
      final t = 17, k = pg.observations[t].toInt();
      expect(pg.trajectories!.at(1 + k, NodeField.mean, t), 1);
      // Loading category indices (with a gap) works; other values don't fit.
      pg.load(Float64List.fromList([0, 2, double.nan, 1]), 'c.csv');
      expect(pg.observed, [1, 1, 0, 1]);
      expect(pg.error, isNull);
      expect(
        Playground.loadedFits(pg.preset, Float64List.fromList([0, 3]), 3),
        isFalse,
      );
      expect(
        Playground.loadedFits(pg.preset, Float64List.fromList([0.5]), 3),
        isFalse,
      );
    });

    test('dp preset finds the simulated sources', () {
      final pg = Playground(presetId: 'dp', nTrials: 150);
      final traj = pg.trajectories!;
      final n = traj.vectorAt(0, VectorField.n, 149);
      // Three sources, 50 trials each, three clusters.
      expect(n.where((c) => c > 0), hasLength(3));
      expect(n.reduce((a, b) => a + b), 150);
      // Edits on implied nodes survive a rebuild.
      pg.setParam(0, NodeField.alpha, 5);
      pg.setVolatilityUpdates(VolatilityUpdates.eHGF);
      expect(pg.param(0, NodeField.alpha), 5);
    });

    test('the trial count is capped per preset', () {
      final pg = Playground(presetId: 'binary3', nTrials: 5000);
      pg.setPreset('dp');
      expect(pg.data.nTrials, pg.preset.defaultTrials);
      expect(Playground(presetId: 'dp', nTrials: 5000).data.nTrials, 1000);
    });

    test('parameter edits re-filter into the same buffers', () {
      final pg = Playground(presetId: 'binary3', nTrials: 200);
      final traj = pg.trajectories;
      final before = pg.trajectories!.at(1, NodeField.mean, 199);
      final rev = pg.revision;
      pg.setParam(1, NodeField.tonicVolatility, -1);
      expect(identical(pg.trajectories, traj), isTrue);
      expect(pg.revision, greaterThan(rev));
      expect(pg.trajectories!.at(1, NodeField.mean, 199), isNot(before));
      pg.resetNode(1);
      expect(pg.trajectories!.at(1, NodeField.mean, 199), before);
    });

    test('coupling edits reach the network and the layout', () {
      final pg = Playground(presetId: 'continuous3', nTrials: 100);
      final edge = pg.layout.edges.firstWhere(
        (e) => e.kind == EdgeKind.volatility,
      );
      pg.setCoupling(edge, 2);
      expect(pg.network.volatilityParentStrengthsOf(1), [2.0]);
      expect(
        pg.layout.edges
            .firstWhere((e) => e.kind == EdgeKind.volatility)
            .strength,
        2,
      );
    });

    test('loaded data with gaps is filtered with observed flags', () {
      final pg = Playground(presetId: 'continuous2', nTrials: 100);
      pg.load(Float64List.fromList([0.1, double.nan, 0.3, 0.2]), 'x.csv');
      expect(pg.observed, [1, 0, 1, 1]);
      expect(pg.trajectories!.nTime, 4);
      expect(pg.error, isNull);
      // A binary preset can't take continuous data: back to simulated.
      pg.setPreset('binary3');
      expect(pg.source, DataSource.simulated);
    });

    test('volatility updates rebuild the network', () {
      final pg = Playground(presetId: 'binary3', nTrials: 50);
      pg.setVolatilityUpdates(VolatilityUpdates.eHGF);
      expect(pg.network.volatilityUpdates, VolatilityUpdates.eHGF);
      expect(pg.trajectories!.nTime, 50);
    });
  });

  test('parseColumn skips a header and reads NaN as missing', () {
    expect(parseColumn('u,rt\n1,0.3\n0,0.5\nNaN,0.1\n').length, 3);
    expect(parseColumn('a\n1\n2', column: 0), [1, 2]);
    expect(() => parseColumn('1\nx\n'), throwsFormatException);
  });

  testWidgets('app renders wide and narrow, in both themes', (tester) async {
    final settings = Settings();
    for (final size in const [Size(1400, 900), Size(420, 900)]) {
      tester.view.physicalSize = size;
      tester.view.devicePixelRatio = 1;
      for (final mode in [ThemeMode.light, ThemeMode.dark]) {
        settings.setThemeMode(mode);
        await tester.pumpWidget(
          PlaygroundApp(
            settings: settings,
            playground: Playground(presetId: 'continuous3', nTrials: 300),
          ),
        );
        await tester.pump();
        expect(tester.takeException(), isNull);
        expect(find.byType(TrajectoryPlot), findsNWidgets(4));
      }
    }
    // Every preset renders: the vector nodes get their own panels, and the
    // nodes folded into them none.
    const panels = {
      'binary3': 4,
      'continuous2': 3,
      'continuous3': 4,
      // x₇, the logits, the categorical input, surprise.
      'categorical3': 4,
      // Only the input: no node kind with a surprise.
      'dp': 1,
      'ef': 1,
    };
    tester.view.physicalSize = const Size(1400, 900);
    for (final p in presets) {
      final pg = Playground(presetId: p.id, nTrials: 100);
      await tester.pumpWidget(
        PlaygroundApp(settings: settings, playground: pg),
      );
      await tester.pump();
      expect(tester.takeException(), isNull, reason: p.id);
      expect(
        find.byType(TrajectoryPlot),
        findsNWidgets(panels[p.id]!),
        reason: p.id,
      );
      expect(
        HgfTrajectoryView.panelCount(pg.network),
        panels[p.id],
        reason: p.id,
      );
      // Each node's card renders.
      for (var i = 0; i < pg.network.nNodes; i++) {
        pg.select(i);
        await tester.pump();
        expect(tester.takeException(), isNull, reason: '${p.id} x$i');
        pg.select(i);
      }
    }
    // Selecting a node shows its sliders.
    tester.view.physicalSize = const Size(1400, 900);
    await tester.pumpWidget(
      PlaygroundApp(
        settings: settings,
        playground: Playground(presetId: 'binary3', nTrials: 300)..select(1),
      ),
    );
    expect(find.textContaining('tonic volatility'), findsOneWidget);
    // Play advances the trial counter.
    await tester.tap(find.byTooltip('Play trial by trial'));
    await tester.pump();
    await tester.pump(const Duration(seconds: 1));
    expect(find.textContaining('trial '), findsWidgets);
    addTearDown(tester.view.reset);
  });
}
