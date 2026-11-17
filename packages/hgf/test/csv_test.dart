import 'package:hgf/hgf.dart';
import 'package:test/test.dart';

void main() {
  test('toText quotes what needs it and parses back', () {
    final data = CsvData(
      ['participant', 'note', 'x'],
      [
        ['p1', 'left, then right', '0.5'],
        ['p2', 'said "none"', ''],
        ['p3', 'plain', '-1e-3'],
      ],
    );
    final text = data.toText();
    expect(text.split('\n')[1], 'p1,"left, then right",0.5');
    expect(text.split('\n')[2], 'p2,"said ""none""",');
    final back = CsvData.parse(text);
    expect(back.header, data.header);
    expect(back.rows, data.rows);
    expect(back.number(2, 2), -1e-3);
    expect(back.number(1, 2)!.isNaN, isTrue);
  });
}
