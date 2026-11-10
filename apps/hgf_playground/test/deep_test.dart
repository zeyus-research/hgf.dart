import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hgf_deep/hgf_deep.dart';
import 'package:hgf_playground/src/app.dart';
import 'package:hgf_playground/src/deep/deep_playground.dart';
import 'package:hgf_playground/src/deep/deep_spec.dart';
import 'package:hgf_playground/src/deep/deep_view.dart';
import 'package:hgf_playground/src/deep/ui/decision_map.dart';
import 'package:hgf_playground/src/deep/ui/deep_graph.dart';
import 'package:hgf_playground/src/model/fitting.dart';
import 'package:hgf_playground/src/model/playground.dart';
import 'package:hgf_playground/src/settings.dart';
import 'package:hgf_playground/src/ui/shell.dart';

DeepViewModel _view(DeepPlayground d, {int? cursor}) {
  final t = d.trace!;
  final i = cursor ?? d.cursor;
  return DeepViewModel(
    network: d.network,
    states: t.steps[i].states,
    weights: t.weightsAt(i),
    step: t.steps[i],
  );
}

void main() {
  group('DeepSpec', () {
    test('builds input → hidden → output in addLayer order', () {
      const spec = DeepSpec(
        hidden: [
          HiddenLayerSpec(size: 5, coupling: CouplingFn.relu),
          HiddenLayerSpec(size: 3, volatility: true),
        ],
      );
      final net = spec.build(DeepTask.spirals);
      // Network layers are bottom (output) first.
      expect(net.layerSizes, [3, 3, 5, 2]);
      expect(net.configs[0].kind, LayerKind.categorical);
      expect(net.configs[1].volatilityParent, isTrue);
      expect(net.configs[2].volatilityParent, isFalse);
      expect(net.configs[2].coupling, CouplingFn.relu);
      expect(spec.sizes(DeepTask.spirals), [2, 5, 3, 3]);
      expect(spec.parameters(DeepTask.spirals), 3 * 5 + 6 * 3 + 4 * 3);
      expect(net.settings.fixes, DeepFixes.recommended);
    });

    test('tonic volatility and fixes off', () {
      final net = const DeepSpec(
        tonicVolatility: -3,
        fixes: false,
      ).build(DeepTask.sine);
      expect(net.configs[1].tonicVolatility, -3);
      expect(net.configs[0].kind, LayerKind.volatile);
      expect(net.settings.fixes.isNone, isTrue);
    });
  });

  group('DeepPlayground', () {
    test('training improves the test accuracy', () {
      final d = DeepPlayground(
        task: DeepTask.moons,
        spec: const DeepSpec(
          hidden: [HiddenLayerSpec(size: 12), HiddenLayerSpec(size: 12)],
        ),
      );
      for (var e = 0; e < 8; e++) {
        d.stepEpoch();
      }
      expect(d.epoch, 8);
      expect(d.history.last.testScore, greaterThan(0.9));
      expect(d.divergedLayer, isNull);
    });

    test('tick trains in slices across epochs', () {
      final d = DeepPlayground(frameBudget: const Duration(milliseconds: 2));
      d.play();
      for (var i = 0; i < 40 && d.epoch == 0; i++) {
        d.tick();
      }
      expect(d.epoch + d.row, greaterThan(0));
    });

    test('the trace is a preview; learning from it changes the network', () {
      final d = DeepPlayground();
      final w = d.network.weights[1]!.copy();
      expect(d.trace!.steps.last.phase, TracePhase.carry);
      d.traceTestSample(3);
      expect(d.network.weights[1]!.data, w.data);
      d.commitTrace();
      expect(d.network.weights[1]!.data, isNot(w.data));
    });

    test('phase list: clamp, predictions, observe, errors, learn, carry', () {
      final d = DeepPlayground(
        spec: const DeepSpec(
          hidden: [HiddenLayerSpec(size: 4), HiddenLayerSpec(size: 4)],
        ),
      );
      expect(
        [for (final s in d.trace!.steps) s.phase.name],
        [
          'clampInput',
          'predict',
          'predict',
          'predict',
          'observe',
          'leafError',
          'posterior',
          'layerError',
          'posterior',
          'layerError',
          'weightUpdate',
          'carry',
        ],
      );
    });

    test('a volatility toggle keeps the learned weights', () {
      final d = DeepPlayground();
      d.stepEpoch();
      final w = d.network.weights[2]!.copy();
      final hidden = [...d.spec.hidden]
        ..[0] = d.spec.hidden[0].copyWith(volatility: true);
      d.setSpec(d.spec.copyWith(hidden: hidden));
      expect(d.network.weights[2]!.data, w.data);
      expect(d.epoch, 0);
      // A size change cannot keep them.
      final bigger = [...hidden]..[0] = hidden[0].copyWith(size: 16);
      d.setSpec(d.spec.copyWith(hidden: bigger));
      expect(d.network.layerSizes, [1, 8, 16, 2]);
    });

    test('custom inputs take a target', () {
      final d = DeepPlayground(task: DeepTask.sine);
      d.traceCustomInput([1.0]);
      expect(d.traceTarget.single, closeTo(0.841470984807896, 1e-12));
      expect(d.trace!.prediction, hasLength(1));
    });
  });

  group('DeepViewModel', () {
    test('modes by size', () {
      DeepViewModel at(List<HiddenLayerSpec> hidden) {
        final d = DeepPlayground(spec: DeepSpec(hidden: hidden));
        return _view(d);
      }

      expect(at([const HiddenLayerSpec(size: 8)]).mode, ViewMode.full);
      expect(
        at([
          const HiddenLayerSpec(size: 48),
          const HiddenLayerSpec(size: 48),
        ]).mode,
        ViewMode.full,
      );
      // 2·64 + 65·64 + 65 = 4417 edges: only the strongest are drawn.
      expect(
        at([
          const HiddenLayerSpec(size: 64),
          const HiddenLayerSpec(size: 64),
        ]).mode,
        ViewMode.topEdges,
      );
      expect(at([const HiddenLayerSpec(size: 256)]).mode, ViewMode.summary);
    });

    test('messages follow the phase', () {
      final d = DeepPlayground(
        spec: const DeepSpec(hidden: [HiddenLayerSpec(size: 3)]),
      );
      final steps = d.trace!.steps;
      int find(TracePhase p, int layer) =>
          steps.indexWhere((s) => s.phase == p && s.layer == layer);
      // Predicting the hidden layer (1) from the input (2): w·g(μ̂) on the
      // edges into layer 1.
      final v = _view(d, cursor: find(TracePhase.predict, 1));
      final w = v.weights[2]!;
      final x = d.traceInput;
      expect(v.message(2, 0, 1), closeTo(w.at(0, 1) * x[1], 1e-12));
      expect(v.message(2, 0, 2), w.at(0, 2)); // bias
      expect(v.message(1, 0, 0), isNull);
      // The weight step: the gradients.
      final learn = _view(d, cursor: find(TracePhase.weightUpdate, -1));
      expect(learn.message(1, 0, 0), learn.step!.gradients![1]!.at(0, 0));
    });
  });

  group('page', () {
    Widget app(DeepPlayground deep, {ThemeMode mode = ThemeMode.light}) {
      final settings = Settings()..themeMode = mode;
      final pg = Playground(presetId: 'binary3', nTrials: 100);
      return PlaygroundApp(
        settings: settings,
        playground: pg,
        fitting: Fitting(pg),
        deep: deep,
        initialPage: ShellPage.deep,
      );
    }

    for (final mode in [ThemeMode.light, ThemeMode.dark]) {
      for (final (w, h) in [(1400.0, 1000.0), (420.0, 900.0)]) {
        testWidgets('renders ${w.toInt()} wide, ${mode.name}', (tester) async {
          tester.view.physicalSize = Size(w, h);
          tester.view.devicePixelRatio = 1;
          addTearDown(tester.view.reset);
          final deep = DeepPlayground();
          await tester.pumpWidget(app(deep, mode: mode));
          await tester.pump();
          expect(tester.takeException(), isNull);
          expect(find.byType(DeepGraphView), findsOneWidget);
          expect(find.byType(DecisionMap), findsOneWidget);
          if (w >= 900) {
            expect(find.text('Follow one sample'), findsOneWidget);
          } else {
            // The cards follow the plots in one scrolling list.
            await tester.scrollUntilVisible(
              find.text('Follow one sample'),
              300,
              scrollable: find.byType(Scrollable).first,
            );
            expect(tester.takeException(), isNull);
          }
        });
      }
    }

    testWidgets('stepping the trace and switching pages', (tester) async {
      tester.view.physicalSize = const Size(1400, 1000);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.reset);
      final deep = DeepPlayground();
      await tester.pumpWidget(app(deep));
      await tester.pump();
      expect(deep.cursor, 0);
      await tester.tap(find.byTooltip('Next phase'));
      await tester.pump();
      expect(deep.cursor, 1);
      expect(
        find.text(phaseTitle(deep.network, deep.trace!.steps[1])),
        findsOneWidget,
      );
      await tester.tap(find.text('learn'));
      await tester.pump();
      expect(deep.trace!.steps[deep.cursor].phase, TracePhase.weightUpdate);
      // To the HGF page and back.
      await tester.tap(find.text('HGF'));
      await tester.pump();
      expect(find.text('Follow one sample'), findsNothing);
      await tester.tap(find.text('Deep network'));
      await tester.pump();
      expect(find.text('Follow one sample'), findsOneWidget);
      expect(tester.takeException(), isNull);
    });

    testWidgets('a big network draws layer strips', (tester) async {
      tester.view.physicalSize = const Size(1400, 1000);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.reset);
      final deep = DeepPlayground(
        spec: const DeepSpec(hidden: [HiddenLayerSpec(size: 512)]),
      );
      await tester.pumpWidget(app(deep));
      await tester.pump();
      expect(tester.takeException(), isNull);
      expect(
        find.textContaining('each layer is drawn as one strip'),
        findsOneWidget,
      );
    });
  });
}
