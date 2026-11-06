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
        pg.stepOnce();
        pg.player!.advance(1000);
        expect(pg.player!.t, 120);
        for (final f in whole.fields) {
          expect(pg.trajectories!.raw(f), expected[f], reason: '$f');
        }
        expect(pg.error, isNull);
      });
    }

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
