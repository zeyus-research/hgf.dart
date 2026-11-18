// Header room in a prebuilt Mach-O dylib, for the install-name rewrite of
// `dart test`, `dart run` and `flutter test` on macOS.
//
// Those set a bundled dylib's install name (LC_ID_DYLIB) to its absolute
// path, ~100 characters, with install_name_tool. wgpu-native's release is
// linked without -headerpad, so its load commands end 72 bytes before the
// first section: room for about 70 more characters than `@rpath/...`, too
// few for that path ("larger updated load commands do not fit").
//
// [makeHeadroom] removes the load commands that carry nothing a loader,
// debugger or code signature needs: LC_SOURCE_VERSION (wgpu-native's says
// 0.0) and an empty LC_DATA_IN_CODE. That is 32 more bytes, room for an
// install name of up to 135 characters. The caller re-signs the library
// (ad hoc), since the code signature covers the header.

import 'dart:typed_data';

const _mhMagic64 = 0xfeedfacf;
const _lcSourceVersion = 0x2a;
const _lcDataInCode = 0x29;
const _lcSegment64 = 0x19;

/// The number of bytes between the end of the load commands and the first
/// section's file offset of the thin 64-bit Mach-O [bytes], or null if
/// [bytes] is not one.
int? headerSlack(Uint8List bytes) {
  final d = ByteData.sublistView(bytes);
  if (bytes.length < 32 || d.getUint32(0, Endian.little) != _mhMagic64) {
    return null;
  }
  final ncmds = d.getUint32(16, Endian.little);
  final sizeofcmds = d.getUint32(20, Endian.little);
  var first = bytes.length;
  var p = 32;
  for (var i = 0; i < ncmds; i++) {
    final cmd = d.getUint32(p, Endian.little);
    final size = d.getUint32(p + 4, Endian.little);
    if (cmd == _lcSegment64) {
      final nsects = d.getUint32(p + 64, Endian.little);
      for (var s = 0; s < nsects; s++) {
        // section_64: offset at 48 (after sectname, segname, addr, size).
        final offset = d.getUint32(p + 72 + s * 80 + 48, Endian.little);
        if (offset != 0 && offset < first) first = offset;
      }
    }
    p += size;
  }
  return first - 32 - sizeofcmds;
}

/// [bytes] (a thin 64-bit little-endian Mach-O) without LC_SOURCE_VERSION
/// and without an empty LC_DATA_IN_CODE: the commands after them move up
/// and the freed bytes are zeroed. Returns the number of bytes freed (0 if
/// there was nothing to remove or [bytes] is not such a file). Edits in
/// place.
int makeHeadroom(Uint8List bytes) {
  final d = ByteData.sublistView(bytes);
  if (bytes.length < 32 || d.getUint32(0, Endian.little) != _mhMagic64) {
    return 0;
  }
  final ncmds = d.getUint32(16, Endian.little);
  final sizeofcmds = d.getUint32(20, Endian.little);
  final end = 32 + sizeofcmds;
  final kept = BytesBuilder();
  var keptCount = 0;
  var p = 32;
  for (var i = 0; i < ncmds; i++) {
    final cmd = d.getUint32(p, Endian.little);
    final size = d.getUint32(p + 4, Endian.little);
    if (size < 8 || p + size > end) return 0; // Malformed: leave it alone.
    final drop =
        cmd == _lcSourceVersion ||
        (cmd == _lcDataInCode && d.getUint32(p + 12, Endian.little) == 0);
    if (!drop) {
      kept.add(Uint8List.sublistView(bytes, p, p + size));
      keptCount++;
    }
    p += size;
  }
  final out = kept.takeBytes();
  final freed = sizeofcmds - out.length;
  if (freed == 0) return 0;
  bytes.setRange(32, 32 + out.length, out);
  bytes.fillRange(32 + out.length, end, 0);
  d.setUint32(16, keptCount, Endian.little);
  d.setUint32(20, out.length, Endian.little);
  return freed;
}
