import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:test/test.dart';

import 'clocked_fixture.dart';

void main() {
  // Four players, teams reshuffled between blocks; presses are labels.
  const csv = '''
participant,block,team,t,x,press
a,1,b1_A,0,1,left
a,1,b1_A,1,,none
a,2,b2_A,2,3,right
a,2,b2_A,3,4,left
b,1,b1_A,0,10,right
b,1,b1_A,1,20,left
b,2,b2_B,2,30,none
b,2,b2_B,3,40,right
c,1,b1_B,0,100,none
c,1,b1_B,1,200,none
c,2,b2_A,2,300,left
c,2,b2_A,3,400,left
d,1,b1_B,0,1000,left
d,1,b1_B,1,2000,right
d,2,b2_B,2,3000,right
d,2,b2_B,3,4000,none
''';
  final data = CsvData.parse(csv);

  CsvData plain(List<DocTransform> ts, [CsvData? d]) => applyTransforms(
    d ?? data,
    ts,
    participantColumn: 'participant',
    groupColumn: 'team',
    timeColumn: 't',
  );

  CsvData cached(TransformCache c, List<DocTransform> ts, [CsvData? d]) =>
      c.apply(
        d ?? data,
        ts,
        participantColumn: 'participant',
        groupColumn: 'team',
        timeColumn: 't',
      );

  void same(CsvData a, CsvData b) {
    expect(a.header, b.header);
    expect(a.rows, b.rows);
  }

  final base = [
    DocTransform('left', "press == 'left'"),
    DocTransform('mates_left', 'mates_mean(left)'),
    DocTransform('prev', 'lag(mates_left)'),
    DocTransform('first_mate', 'mate(press)'),
  ];

  test('every edit gives what applyTransforms gives', () {
    final c = TransformCache();
    same(cached(c, base), plain(base));
    expect(c.lastComputed, 4);

    // The same again: the same table, nothing computed.
    final again = cached(c, base);
    expect(c.lastComputed, -1);
    expect(identical(again, cached(c, base)), isTrue);

    // Appended: only the new one.
    final appended = [
      ...base,
      DocTransform('both', "where(first_mate == 'left', prev + x, 0 / 0)"),
    ];
    same(cached(c, appended), plain(appended));
    expect(c.lastComputed, 1);

    // The last edited, then one in the middle, then removed.
    final lastEdited = [...base, DocTransform('both', 'lag(prev, 2) * x')];
    same(cached(c, lastEdited), plain(lastEdited));
    expect(c.lastComputed, 1);
    final midEdited = [
      base[0],
      DocTransform('mates_left', 'mates_count(left)'),
      ...lastEdited.skip(2),
    ];
    same(cached(c, midEdited), plain(midEdited));
    expect(c.lastComputed, 4);
    final shorter = midEdited.take(2).toList();
    same(cached(c, shorter), plain(shorter));
    same(cached(c, base), plain(base));

    // New data (even with equal cells) starts again.
    final copy = CsvData.parse(csv);
    same(cached(c, base, copy), plain(base, copy));
    expect(c.lastComputed, 4);
  });

  test('a transform named like a data column is still an error', () {
    final c = TransformCache();
    cached(c, base);
    expect(
      () => cached(c, [...base, DocTransform('x', '1')]),
      throwsA(isA<FormatException>()),
    );
    // And the cache still works after it.
    same(cached(c, base), plain(base));
  });

  test('the clocked document reads the same subjects through a cache', () {
    final c = clockedDocument().compile();
    final events = CsvData.parse(clockedEventsCsv);
    final cache = TransformCache();
    final want = c.subjects(events);
    final got = c.subjects(events, cache: cache);
    expect(identical(c.subjects(events, cache: cache), got), isTrue);
    expect(got.length, want.length);
    for (final (k, s) in got.indexed) {
      expect(s.id, want[k].id);
      expect(s.inputs, want[k].inputs);
      expect(s.timeSteps, want[k].timeSteps);
      expect(s.observed, want[k].observed);
    }
    // The grid is kept while the data and the clock are.
    final grid = c.layout.clocked(events, cache: cache);
    expect(identical(c.layout.clocked(events, cache: cache), grid), isTrue);
    expect(
      c.withOutputs(events, cache: cache).rows,
      c.withOutputs(events).rows,
    );
  });
}
