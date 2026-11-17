import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:test/test.dart';

void main() {
  test('entities from names and folders', () {
    expect(
      bidsEntity('sub-001/beh/sub-001_task-x_events.tsv', 'sub'),
      'sub-001',
    );
    expect(bidsEntity('sub-01_ses-2_task-x_events.tsv', 'ses'), 'ses-2');
    expect(bidsEntity('sub-01/ses-pre/beh/x_events.tsv', 'ses'), 'ses-pre');
    expect(bidsEntity('subject_events.tsv', 'sub'), isNull);
    expect(isBidsEvents('sub-001/beh/sub-001_task-x_events.tsv'), isTrue);
    expect(
      isBidsEvents('sub-001/eeg/sub-001_task-x_events.tsv', datatype: 'beh'),
      isFalse,
    );
    expect(isBidsEvents('participants.tsv'), isFalse);
  });

  final s2 = CsvData.parse(
    'onset\ttrial_type\tvalue\n'
    '1.0\tpress\tleft\n'
    '1.5\ttouch\tn/a\n',
  );
  final s1 = CsvData.parse(
    'onset\ttrial_type\tvalue\textra\n'
    '0.5\tpress\tright\t7\n',
  );
  final participants = CsvData.parse(
    'participant_id\tgroup\tage\n'
    'sub-001\tsession01\t26\n'
    'sub-002\tsession02\t24\n',
  );

  test('one long table, participants in order, info joined', () {
    final m = mergeBidsEvents([
      (name: 'sub-002/beh/sub-002_task-x_events.tsv', table: s2),
      (name: 'sub-001/beh/sub-001_task-x_events.tsv', table: s1),
    ], participants: participants);
    expect(m.participants, ['sub-001', 'sub-002']);
    expect(m.sessions, isEmpty);
    expect(m.data.header, [
      'participant_id',
      'onset',
      'trial_type',
      'value',
      'extra',
      'group',
      'age',
    ]);
    expect(m.data.rows, [
      ['sub-001', '0.5', 'press', 'right', '7', 'session01', '26'],
      ['sub-002', '1.0', 'press', 'left', '', 'session02', '24'],
      ['sub-002', '1.5', 'touch', 'n/a', '', 'session02', '24'],
    ]);
  });

  test('picks participants and rows; sessions from ses- entities', () {
    final m = mergeBidsEvents(
      [
        (name: 'sub-001_ses-2_events.tsv', table: s1),
        (name: 'sub-001_ses-1_events.tsv', table: s2),
        (name: 'sub-002_ses-1_events.tsv', table: s2),
      ],
      participants: participants,
      include: (_, info) => info['group'] == 'session01',
      keepRow: (c, row) => row[c['trial_type']!] == 'press',
    );
    expect(m.participants, ['sub-001']);
    expect(m.sessions, ['ses-1', 'ses-2']);
    expect(
      [for (final r in m.data.rows) r.take(3).toList()],
      [
        ['sub-001', 'ses-1', '1.0'],
        ['sub-001', 'ses-2', '0.5'],
      ],
    );
  });

  test('a file without a participant is an error', () {
    expect(
      () => mergeBidsEvents([(name: 'events.tsv', table: s1)]),
      throwsA(isA<FormatException>()),
    );
  });
}
