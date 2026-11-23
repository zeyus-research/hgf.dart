import 'dart:typed_data';

import 'package:test/test.dart';

import 'package:hgf_gpu/src/hook/macho_headroom.dart';

/// A minimal thin Mach-O 64: one LC_SEGMENT_64 with one section at file
/// offset [textOffset], then the given extra load commands (cmd, size,
/// payload word at +12).
Uint8List _macho(List<(int, int, int)> extra, {int textOffset = 512}) {
  final bytes = Uint8List(1024);
  final d = ByteData.sublistView(bytes);
  d.setUint32(0, 0xfeedfacf, Endian.little);
  var p = 32;
  // LC_SEGMENT_64 with one section_64 (72 + 80 bytes).
  d.setUint32(p, 0x19, Endian.little);
  d.setUint32(p + 4, 152, Endian.little);
  d.setUint32(p + 64, 1, Endian.little);
  d.setUint32(p + 72 + 48, textOffset, Endian.little);
  p += 152;
  for (final (cmd, size, word) in extra) {
    d.setUint32(p, cmd, Endian.little);
    d.setUint32(p + 4, size, Endian.little);
    d.setUint32(p + 12, word, Endian.little);
    p += size;
  }
  d.setUint32(16, 1 + extra.length, Endian.little);
  d.setUint32(20, p - 32, Endian.little);
  return bytes;
}

List<int> _cmds(Uint8List bytes) {
  final d = ByteData.sublistView(bytes);
  final out = <int>[];
  var p = 32;
  for (var i = 0; i < d.getUint32(16, Endian.little); i++) {
    out.add(d.getUint32(p, Endian.little));
    p += d.getUint32(p + 4, Endian.little);
  }
  return out;
}

void main() {
  const idDylib = 0x0d, sourceVersion = 0x2a, dataInCode = 0x29;
  const functionStarts = 0x26, codeSignature = 0x1d;

  test('drops LC_SOURCE_VERSION and an empty LC_DATA_IN_CODE', () {
    final bytes = _macho([
      (idDylib, 56, 24),
      (sourceVersion, 16, 0),
      (functionStarts, 16, 4096),
      (dataInCode, 16, 0), // datasize 0
      (codeSignature, 16, 8192),
    ]);
    final before = headerSlack(bytes)!;
    expect(makeHeadroom(bytes), 32);
    expect(headerSlack(bytes), before + 32);
    expect(_cmds(bytes), [0x19, idDylib, functionStarts, codeSignature]);
    // The freed tail is zeroed, and a second pass changes nothing.
    final end = 32 + ByteData.sublistView(bytes).getUint32(20, Endian.little);
    expect(bytes.sublist(end, end + 32), everyElement(0));
    expect(makeHeadroom(bytes), 0);
  });

  test('keeps an LC_DATA_IN_CODE with data', () {
    final bytes = _macho([(dataInCode, 16, 8), (sourceVersion, 16, 0)]);
    expect(makeHeadroom(bytes), 16);
    expect(_cmds(bytes), [0x19, dataInCode]);
  });

  test('leaves other files alone', () {
    expect(makeHeadroom(Uint8List(64)), 0);
    expect(headerSlack(Uint8List(64)), isNull);
    final bytes = _macho([(sourceVersion, 4, 0)]); // cmdsize < 8
    final copy = Uint8List.fromList(bytes);
    expect(makeHeadroom(bytes), 0);
    expect(bytes, copy);
  });
}
