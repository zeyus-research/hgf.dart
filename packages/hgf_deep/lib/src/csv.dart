import 'dart:convert';

/// Rows of a CSV file as text, with its header: what `Dataset.fromCsv`
/// reads. Pure Dart (read the file yourself: `File(path).readAsStringSync()`
/// on the VM, a `fetch` or a file picker on the web).
final class CsvData {
  CsvData(this.header, this.rows, {this.decimalComma = false});

  /// Parses [text]: the separator is a tab, a semicolon or a comma (the
  /// first of those the header row contains); double-quoted cells may hold
  /// the separator (`""` is a quote); blank lines and lines starting with
  /// `#` are skipped. The first row is the header. With semicolons, a
  /// decimal comma (`1,5`) reads as a number.
  factory CsvData.parse(String text) {
    final lines = [
      for (final raw in const LineSplitter().convert(text))
        if (raw.trim().isNotEmpty && !raw.trimLeft().startsWith('#')) raw,
    ];
    if (lines.isEmpty) throw const FormatException('No rows');
    final first = lines.first;
    final sep = first.contains('\t')
        ? '\t'
        : first.contains(';')
        ? ';'
        : ',';
    final header = [for (final h in _split(first, sep)) h.trim()];
    final rows = [
      for (final l in lines.skip(1))
        [
          ..._split(l, sep),
          for (var i = 0; i < header.length; i++) '',
        ].sublist(0, header.length),
    ];
    return CsvData(header, rows, decimalComma: sep == ';');
  }

  final List<String> header;

  /// One list of cells per row, as many as the header has.
  final List<List<String>> rows;

  /// Whether a decimal comma is accepted.
  final bool decimalComma;

  static const _missing = {'', 'na', 'nan', 'null', 'n/a', 'none', '.'};

  /// Whether row [r]'s cell in column [c] is missing.
  bool isMissing(int r, int c) =>
      _missing.contains(rows[r][c].trim().toLowerCase());

  /// Row [r]'s cell in column [c] as a number (NaN when missing, null when
  /// it is text).
  double? number(int r, int c) {
    if (isMissing(r, c)) return double.nan;
    final s = rows[r][c].trim();
    return double.tryParse(decimalComma ? s.replaceAll(',', '.') : s);
  }

  /// The first row whose cell in column [c] is text (null if none).
  int? firstText(int c) {
    for (var r = 0; r < rows.length; r++) {
      if (number(r, c) == null) return r;
    }
    return null;
  }

  static List<String> _split(String line, String sep) {
    final out = <String>[];
    final cell = StringBuffer();
    var quoted = false;
    for (var i = 0; i < line.length; i++) {
      final ch = line[i];
      if (quoted) {
        if (ch == '"') {
          if (i + 1 < line.length && line[i + 1] == '"') {
            cell.write('"');
            i++;
          } else {
            quoted = false;
          }
        } else {
          cell.write(ch);
        }
      } else if (ch == '"') {
        quoted = true;
      } else if (ch == sep) {
        out.add(cell.toString());
        cell.clear();
      } else {
        cell.write(ch);
      }
    }
    out.add(cell.toString());
    return out;
  }
}
