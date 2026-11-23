import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

// One symbol of the wgpu-native that hook/build.dart bundles (its code asset
// `package:hgf_gpu/wgpu_native`), only to find which file that is: the
// bindings are ffigen's DynamicLibrary ones, and `dart test` / `dart run`
// keep the bundled library in the hooks' output, which DynamicLibrary.open
// by name doesn't search (Flutter apps put it next to the runner).
@Native<Uint32 Function()>(
  symbol: 'wgpuGetVersion',
  assetId: 'package:hgf_gpu/wgpu_native',
)
external int _bundledGetVersion();

/// The path of the wgpu-native library the build hook bundled, or null if
/// there is none (opted out, offline at build time, no prebuilt for this
/// target).
String? bundledLibraryPath() {
  final Pointer<Void> address;
  try {
    address = Native.addressOf<NativeFunction<Uint32 Function()>>(
      _bundledGetVersion,
    ).cast();
  } on Object {
    return null; // Unresolved: no bundled library.
  }
  try {
    return Platform.isWindows ? _windowsPath(address) : _posixPath(address);
  } on Object {
    return null;
  }
}

final class _DlInfo extends Struct {
  external Pointer<Utf8> fname;
  external Pointer<Void> fbase;
  external Pointer<Utf8> sname;
  external Pointer<Void> saddr;
}

String? _posixPath(Pointer<Void> address) {
  final dladdr = DynamicLibrary.process()
      .lookupFunction<
        Int32 Function(Pointer<Void>, Pointer<_DlInfo>),
        int Function(Pointer<Void>, Pointer<_DlInfo>)
      >('dladdr');
  return using((arena) {
    final info = arena<_DlInfo>();
    if (dladdr(address, info) == 0 || info.ref.fname == nullptr) return null;
    return info.ref.fname.toDartString();
  });
}

String? _windowsPath(Pointer<Void> address) {
  final kernel32 = DynamicLibrary.open('kernel32.dll');
  final getHandle = kernel32
      .lookupFunction<
        Int32 Function(Uint32, Pointer<Void>, Pointer<Pointer<Void>>),
        int Function(int, Pointer<Void>, Pointer<Pointer<Void>>)
      >('GetModuleHandleExW');
  final getName = kernel32
      .lookupFunction<
        Uint32 Function(Pointer<Void>, Pointer<Utf16>, Uint32),
        int Function(Pointer<Void>, Pointer<Utf16>, int)
      >('GetModuleFileNameW');
  const fromAddress = 0x4, unchangedRefcount = 0x2;
  return using((arena) {
    final module = arena<Pointer<Void>>();
    if (getHandle(fromAddress | unchangedRefcount, address, module) == 0) {
      return null;
    }
    const size = 32768;
    final name = arena<Uint16>(size).cast<Utf16>();
    final n = getName(module.value, name, size);
    return n == 0 || n >= size ? null : name.toDartString(length: n);
  });
}
