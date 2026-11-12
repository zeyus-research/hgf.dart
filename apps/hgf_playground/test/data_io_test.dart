import 'dart:typed_data';

import 'package:material_ui/material_ui.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_playground/src/model/data.dart';
import 'package:hgf_playground/src/model/export.dart';
import 'package:hgf_playground/src/model/fit_config.dart';
import 'package:hgf_playground/src/model/fitting.dart';
import 'package:hgf_playground/src/model/playground.dart';
import 'package:hgf_playground/src/ui/files.dart';
import 'package:hgf_fit/hgf_fit.dart';

Future<FitWorkers> _inline([int? _]) => FitWorkers.spawn(count: 0);

void main() {
  group('parseTable', () {
    test('comma-separated with a header; empty cells are missing', () {
      final t = parseTable('trial,u,response\n0,1,0\n1,,1\n2,0,NA\n');
      expect(t.names, ['trial', 'u', 'response']);
      expect(t.rows, 3);
      expect(t.columns[1][0], 1);
      expect(t.columns[1][1].isNaN, isTrue);
      expect(t.columns[2][2].isNaN, isTrue);
      expect(t.numericColumns, [0, 1, 2]);
    });

    test('tab, semicolon with decimal commas, and whitespace', () {
      expect(parseTable('a\tb\n1.5\t2\n').columns[0], [1.5]);
      final eu = parseTable('obs;resp\n1,5;0\n-2,25;1\n');
      expect(eu.columns[0], [1.5, -2.25]);
      expect(eu.columns[1], [0, 1]);
      final ws = parseTable('# a comment\n1  2\n\n3 4\n');
      expect(ws.names, ['column 1', 'column 2']);
      expect(ws.columns[1], [2, 4]);
    });

    test('quoted cells, text columns, ragged rows', () {
      final t = parseTable(
        'cond,"value, raw",rt\n"a, b",1,0.3\nb,2\nc,"3",0.5\n',
      );
      expect(t.names, ['cond', 'value, raw', 'rt']);
      expect(t.textCells, [3, 0, 0]);
      expect(t.numericColumns, [1, 2]);
      expect(t.columns[1], [1, 2, 3]);
      expect(t.columns[2][1].isNaN, isTrue);
    });

    test('parseColumn reads the first numeric column', () {
      expect(parseColumn('label,x\na,1\nb,2\n'), [1, 2]);
      expect(parseColumn('1\n2\n'), [1, 2]);
      expect(() => parseColumn('a\nb\n'), throwsFormatException);
      expect(() => parseColumn(''), throwsFormatException);
    });
  });

  group('columns', () {
    final t = parseTable('trial,stimulus,choice,rt\n0,1,1,0.4\n1,0,1,0.5\n');

    test('guessed by name', () {
      expect(guessColumns(t, observations: true, responses: true), (
        observations: 1,
        responses: 2,
      ));
      // Responses only: the named one, else the first numeric column.
      expect(guessColumns(t, observations: false, responses: true), (
        observations: null,
        responses: 2,
      ));
      final plain = parseTable('1,2\n3,4\n');
      expect(guessColumns(plain, observations: true, responses: true), (
        observations: 0,
        responses: null,
      ));
    });

    test('one numeric column needs no question', () {
      final one = parseTable('cond,u\na,1\nb,0\n');
      expect(obviousColumns(one, observations: true, responses: true), (
        observations: 1,
        responses: null,
      ));
      expect(obviousColumns(t, observations: true, responses: true), isNull);
    });

    testWidgets('the mapping dialog returns the chosen columns', (
      tester,
    ) async {
      ColumnChoice? choice;
      await tester.pumpWidget(
        MaterialApp(
          home: Builder(
            builder: (context) => TextButton(
              onPressed: () async => choice = await showColumnMapping(
                context,
                t,
                'data.csv',
                observations: true,
                responses: true,
              ),
              child: const Text('open'),
            ),
          ),
        ),
      );
      await tester.tap(find.text('open'));
      await tester.pumpAndSettle();
      expect(find.text('Columns of data.csv'), findsOneWidget);
      // The preview shows the guessed columns.
      expect(find.text('stimulus'), findsWidgets);
      expect(find.text('choice'), findsWidgets);
      await tester.tap(find.text('Load'));
      await tester.pumpAndSettle();
      expect(choice, (observations: 1, responses: 2));
    });
  });

  group('exports', () {
    test('trajectories: one row per trial, pyhgf column names', () {
      final pg = Playground(presetId: 'binary3', nTrials: 50);
      final traj = pg.trajectories!;
      final responses = Float64List(50)..[3] = double.nan;
      final observed = Float64List(50)..fillRange(0, 50, 1);
      observed[7] = 0;
      final csv = trajectoriesCsv(
        traj,
        observations: pg.observations,
        observed: observed,
        responses: responses,
      );
      final rows = parseTable(csv);
      expect(rows.rows, 50);
      expect(rows.names.take(3), ['trial', 'observation', 'response']);
      expect(rows.names, contains('x_1_expected_mean'));
      expect(rows.names, contains('x_0_surprise'));
      expect(rows.names.last, 'total_surprise');
      expect(rows.columns[1][7].isNaN, isTrue);
      expect(rows.columns[2][3].isNaN, isTrue);
      final c = rows.names.indexOf('x_1_expected_mean');
      expect(rows.columns[c], traj.series(1, NodeField.expectedMean));
    });

    test('fit results and recovery', () async {
      final pg = Playground(presetId: 'continuous2', nTrials: 120);
      final f = Fitting(
        pg,
        spawnWorkers: _inline,
        config: const FitConfig(restarts: 0),
      );
      await f.fit();
      final csv = fitResultsCsv(f.result!);
      expect(csv, startsWith('# method: MAP + Laplace'));
      final t = parseTable(csv);
      expect(t.names, [
        'parameter',
        'symbol',
        'estimate',
        'lower',
        'upper',
        'truth',
      ]);
      expect(t.rows, f.result!.estimates.length);
      expect(t.columns[2][0], f.result!.estimates.first.estimate);
      expect(fitDrawsCsv(f.result!), isNull);

      f
        ..simulateAgent(seed: 3)
        ..setRecoveryAgents(50);
      await f.recover(seed: 1);
      final rec = parseTable(recoveryCsv(f.recovery!));
      expect(rec.rows, f.recovery!.ok.length);
      expect(rec.names.take(2), ['agent', 'fitted']);
      expect(rec.names.length, 2 + 2 * f.recovery!.params.length);
    });
  });
}
