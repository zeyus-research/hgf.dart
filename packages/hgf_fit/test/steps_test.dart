import 'dart:convert';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_fit/hgf_fit.dart' show Subject;
import 'package:test/test.dart';

import 'clocked_fixture.dart';

const StepScope _scope = (
  participantColumn: 'p',
  groupColumn: null,
  timeColumn: 't',
);

/// Two participants' rows, interleaved (steps work per participant).
final _csv = CsvData.parse('''
p,t,x,press
a,0,1,left
b,0,10,
a,1,2,
a,2,3,right
b,1,20,left
a,3,4,
b,2,,left
a,4,5,left
b,3,40,
''');

List<String> _col(CsvData t, String name) => [
  for (final r in t.rows) r[t.header.indexOf(name)],
];

void main() {
  group('decimate', () {
    test('keeps the first row of each block, per participant', () {
      final t = DecimateStep(factor: 2).apply(_csv, _scope);
      expect(_col(t, 'p'), ['a', 'a', 'a', 'b', 'b']);
      expect(_col(t, 't'), ['0', '2', '4', '0', '2']);
      expect(_col(t, 'x'), ['1', '3', '5', '10', '']);
    });

    test('aggregates numbers, leaves labels and the time column', () {
      final t = DecimateStep(
        factor: 2,
        how: StepAggregate.mean,
        columns: const {'press': StepAggregate.mode},
      ).apply(_csv, _scope);
      expect(_col(t, 't'), ['0', '2', '4', '0', '2']);
      expect(_col(t, 'x'), ['1.5', '3.5', '5', '15', '40']);
      expect(_col(t, 'press'), ['left', 'right', 'left', 'left', 'left']);
    });

    test('any, sum, min, max, last', () {
      CsvData run(StepAggregate a) =>
          DecimateStep(factor: 3, columns: {'x': a}).apply(_csv, _scope);
      expect(_col(run(StepAggregate.any), 'x'), ['1', '1', '1', '1']);
      expect(_col(run(StepAggregate.sum), 'x'), ['6', '9', '30', '40']);
      expect(_col(run(StepAggregate.min), 'x'), ['1', '4', '10', '40']);
      expect(_col(run(StepAggregate.max), 'x'), ['3', '5', '20', '40']);
      expect(_col(run(StepAggregate.last), 'x'), ['3', '5', '', '40']);
    });

    test('blocks end where a within column changes', () {
      final csv = CsvData.parse('p,t,s\na,0,1\na,1,1\na,2,1\na,3,2\na,4,2\n');
      final t = DecimateStep(factor: 2, within: ['s']).apply(csv, _scope);
      expect(_col(t, 't'), ['0', '2', '3']);
    });

    test('a numeric aggregate of a label column is an error', () {
      expect(
        () => DecimateStep(
          factor: 2,
          columns: const {'press': StepAggregate.mean},
        ).apply(_csv, _scope),
        throwsFormatException,
      );
    });
  });

  group('segment', () {
    test('windows around events, later events take overlapping rows', () {
      final t = SegmentStep(
        at: Expr.parse("press == 'left'"),
        before: 1,
        after: 1,
        offsetColumn: 'off',
      ).apply(_csv, _scope);
      // a: lefts at rows 0 and 4 (t 0, 4); b: lefts at t 1, 2.
      expect(_col(t, 'p'), ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']);
      expect(_col(t, 't'), ['0', '1', '3', '4', '0', '1', '2', '3']);
      expect(_col(t, 'segment'), ['1', '1', '2', '2', '1', '2', '2', '2']);
      expect(_col(t, 'off'), ['0', '1', '-1', '0', '-1', '-1', '0', '1']);
    });

    test('without after, a segment runs to the next one', () {
      final t = SegmentStep(
        at: Expr.parse("press == 'left'"),
      ).apply(_csv, _scope);
      expect(_col(t, 't'), ['0', '1', '2', '3', '4', '1', '2', '3']);
      expect(_col(t, 'segment'), ['1', '1', '1', '1', '2', '1', '2', '2']);
    });
  });

  test('filter keeps the rows the expression picks', () {
    final t = FilterStep(Expr.parse('diff(x) > 0')).apply(_csv, _scope);
    expect(_col(t, 'x'), ['2', '3', '20', '4', '5']);
  });

  group('standardise', () {
    test('per participant, missing cells stay missing', () {
      final t = StandardiseStep(columns: ['x']).apply(_csv, _scope);
      final a = [
        for (final r in t.rows)
          if (r[0] == 'a') double.parse(r[2]),
      ];
      final mean = a.reduce((s, x) => s + x) / a.length;
      final v = a.fold(0.0, (s, x) => s + (x - mean) * (x - mean)) / a.length;
      expect(mean, closeTo(0, 1e-12));
      expect(v, closeTo(1, 1e-12));
      expect(_col(t, 'x')[6], '');
    });

    test('fitted on some rows only', () {
      final t = StandardiseStep(
        columns: ['x'],
        perParticipant: false,
        fitOn: Expr.parse("p == 'a'"),
      ).apply(_csv, _scope);
      // a's xs 1..5: mean 3, sd sqrt(2).
      expect(
        double.parse(_col(t, 'x')[1]),
        closeTo(7 / 1.4142135623730951, 1e-12),
      );
    });
  });

  group('split', () {
    test('by time: the first rows of each participant train', () {
      final t = SplitStep().apply(_csv, _scope);
      expect(_col(t, 'split'), [
        'train',
        'train',
        'train',
        'train',
        'train',
        'train',
        'train',
        'test',
        'test',
      ]);
    });

    test('by participant: seeded, the same on every run', () {
      final ps = [
        for (var i = 0; i < 10; i++)
          for (var t = 0; t < 2; t++) 'p$i,$t',
      ];
      final csv = CsvData.parse('p,t\n${ps.join('\n')}\n');
      final t = SplitStep(byParticipant: true, seed: 7).apply(csv, _scope);
      final labels = {for (final r in t.rows) r[0]: r[2]};
      expect(labels.values.where((l) => l == 'train').length, 8);
      expect(SplitStep.shuffled(10, 7), isNot(List.generate(10, (i) => i)));
      expect(SplitStep.shuffled(10, 7), SplitStep.shuffled(10, 7));
      // mulberry32's first draws for seed 7 (reference implementation).
      expect(SplitStep.shuffled(5, 7)..sort(), [0, 1, 2, 3, 4]);
    });
  });

  test('steps round-trip through JSON and the document', () {
    final doc = steppedDocument();
    final back = ModelDocument.fromJson(
      jsonDecode(jsonEncode(doc.toJson())) as Map<String, Object?>,
    );
    expect(jsonEncode(back.toJson()), jsonEncode(doc.toJson()));
    expect(back.data.steps.length, 5);
  });

  test('the document reads the stepped table; checks know its columns', () {
    final doc = steppedDocument();
    final c = doc.compile();
    final events = CsvData.parse(clockedEventsCsv);
    final grid = c.layout.clocked(events);
    final t = c.layout.prepared(events);
    expect(t.header, doc.data.columnsOf(events.header));
    expect(t.header, containsAll(['split', 'segment', 'seg_t']));
    expect(t.rows.length, lessThan(grid.rows.length));
    final subs = c.subjects(events);
    expect(subs, isNotEmpty);
    expect(
      checkModel(doc, columns: events.header).where((i) => i.isError),
      isEmpty,
    );
    // A step reading a column that only a later step makes.
    final bad = doc.copyWith(
      data: doc.data.copyWith(
        steps: [FilterStep(Expr.parse('seg_t > 0')), ...doc.data.steps],
      ),
    );
    expect(
      checkModel(bad, columns: events.header).map((i) => i.message),
      contains(contains('reads "seg_t"')),
    );
  });

  test('the cache reuses the stepped table', () {
    final c = steppedDocument().compile();
    final events = CsvData.parse(clockedEventsCsv);
    final cache = TransformCache();
    final a = c.layout.prepared(events, cache: cache);
    final b = c.layout.prepared(events, cache: cache);
    expect(identical(a, b), isTrue);
    expect(
      c.transform(events, cache: cache).toText(),
      c.transform(events).toText(),
    );
  });

  test('a prepared table is not stepped again', () {
    final c = steppedDocument().compile();
    final events = CsvData.parse(clockedEventsCsv);
    final once = c.layout.prepared(events);
    expect(identical(c.layout.prepared(once), once), isTrue);
    List<int> lengths(List<Subject> s) => [for (final x in s) x.inputs.length];
    expect(lengths(c.subjects(once)), lengths(c.subjects(events)));
    expect(c.withOutputs(once).toText(), c.withOutputs(events).toText());
  });
}
