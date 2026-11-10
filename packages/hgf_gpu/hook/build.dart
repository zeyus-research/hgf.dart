// Bundles wgpu-native into apps that depend on hgf_gpu (Flutter on macOS, iOS,
// Linux, Windows and Android; `dart run`/`dart test` on the host), so
// `WgpuBackend.create` finds it without an install or `$WGPU_LIB`.
//
// Downloads the prebuilt release that lib/src/native/wgpu_bindings.g.dart was
// generated for (see tool/fetch_wgpu.sh), checks its sha256 and emits the
// shared library as a bundled code asset. If the download fails (offline),
// nothing is bundled and the backend falls back to the system library.
//
// Opt out with the user-define `bundle: false` (in the workspace root
// pubspec.yaml):
//
//   hooks:
//     user_defines:
//       hgf_gpu:
//         bundle: false
//
// The backend then loads the system library (`$WGPU_LIB`, Homebrew's
// wgpu-native, ...). `flutter test` on macOS needs this today: it rewrites a
// bundled dylib's install name to an absolute path, which the prebuilt
// release has no header room for (install_name_tool aborts).

import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:crypto/crypto.dart';
import 'package:hooks/hooks.dart';

const _version = 'v29.0.1.1';

/// Release asset and its sha256, per target.
final _releases = <(OS, Architecture), (String, String)>{
  (OS.macOS, Architecture.arm64): (
    'macos-aarch64',
    'a5797a37b1adf720bcd5dcffb291edbbd5b7b14be0a3874c28e6393a655a7a3e',
  ),
  (OS.macOS, Architecture.x64): (
    'macos-x86_64',
    '8e2f7378548ddd0e2cf21e7d864dda46e953f0af724855a33778b85ead206d41',
  ),
  (OS.linux, Architecture.x64): (
    'linux-x86_64',
    '95a4d90c071005a98d03eab348beaa6b07e16eb00d1dcdb9f8348f75eb97ec5a',
  ),
  (OS.linux, Architecture.arm64): (
    'linux-aarch64',
    '015fcdf1dbae82e614a783cc38017e5399ae0927a889fe9b69c9b664bc61b47a',
  ),
  (OS.windows, Architecture.x64): (
    'windows-x86_64-msvc',
    '7e67d7445c42aeb85e30f88930fd8d7d83ee769e3390aeb1ada75ebf3cf78132',
  ),
  (OS.windows, Architecture.arm64): (
    'windows-aarch64-msvc',
    '4a876421a8c1e5fe72f849b3722214280fe485cb1c56f77f8b0c82414be5b29f',
  ),
  (OS.android, Architecture.arm64): (
    'android-aarch64',
    '721741f1b05a20c1738166bedf7a5efb2ba4b382da689526d3fc33de22bdd573',
  ),
  (OS.android, Architecture.arm): (
    'android-armv7',
    'f9d76c77b3fda3f7121476884eb16ec067f7dada83276298a3cc8bf6a8403d60',
  ),
  (OS.android, Architecture.x64): (
    'android-x86_64',
    'ef16fc0644bf0e308a39ac4516742da8e22d8c201d3a542cc5baf533d272c491',
  ),
  (OS.android, Architecture.ia32): (
    'android-i686',
    '593b94875bc4fcc1506ea0b6714dd12b96b7c852921caa63f45eb61517793312',
  ),
  (OS.iOS, Architecture.arm64): (
    'ios-aarch64',
    'e36c9913b9e5095a530fa9121c50b16a4e3dd020e1eebf601f2f47ce24d56941',
  ),
};

/// As [_releases], for the iOS simulator SDK.
final _iosSimulatorReleases = <(OS, Architecture), (String, String)>{
  (OS.iOS, Architecture.arm64): (
    'ios-aarch64-simulator',
    '750e706765bef3744313745194774d095c916fc21d2a0e7d4d7b0bc4d0c92789',
  ),
  (OS.iOS, Architecture.x64): (
    'ios-x86_64-simulator',
    '94f67e1b268e8dd31b8e59b32f211ac469f09ed7950fceee52bd84f0623da3d9',
  ),
};

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) return;
    final bundle = input.userDefines['bundle'];
    if (bundle != null &&
        bundle is! bool &&
        bundle != 'true' &&
        bundle != 'false') {
      throw const FormatException(
        'hooks.user_defines.hgf_gpu.bundle must be a boolean (or omitted).',
      );
    }
    if (bundle == false || bundle == 'false') return;
    final code = input.config.code;
    final simulator =
        code.targetOS == OS.iOS && code.iOS.targetSdk == IOSSdk.iPhoneSimulator;
    final release = (simulator
        ? _iosSimulatorReleases
        : _releases)[(code.targetOS, code.targetArchitecture)];
    if (release == null) return; // No prebuilt for this target.
    final (name, sha256Hex) = release;
    final libName = code.targetOS.dylibFileName('wgpu_native');

    // Shared between configs that only differ in irrelevant fields.
    final dir = Directory.fromUri(
      input.outputDirectoryShared.resolve('wgpu-$_version-$name/'),
    );
    final lib = File.fromUri(dir.uri.resolve(libName));
    if (!lib.existsSync()) {
      try {
        await _fetch(name, sha256Hex, libName, dir);
      } on Object catch (e) {
        stderr.writeln(
          'hgf_gpu: could not bundle wgpu-native $_version ($name): $e. '
          'Falling back to the system library at runtime.',
        );
        return;
      }
    }

    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'wgpu_native',
        linkMode: DynamicLoadingBundled(),
        file: lib.uri,
      ),
    );
  });
}

Future<void> _fetch(
  String name,
  String sha256Hex,
  String libName,
  Directory dir,
) async {
  final url = Uri.parse(
    'https://github.com/gfx-rs/wgpu-native/releases/download/'
    '$_version/wgpu-$name-release.zip',
  );
  final client = HttpClient();
  final List<int> bytes;
  try {
    final request = await client.getUrl(url);
    final response = await request.close();
    if (response.statusCode != HttpStatus.ok) {
      throw HttpException('HTTP ${response.statusCode}', uri: url);
    }
    bytes = [for (final chunk in await response.toList()) ...chunk];
  } finally {
    client.close();
  }
  final digest = sha256.convert(bytes).toString();
  if (digest != sha256Hex) {
    throw StateError('sha256 mismatch for $url: $digest');
  }

  final tmp = await Directory.systemTemp.createTemp('hgf_gpu_wgpu');
  try {
    final zip = File('${tmp.path}/wgpu.zip');
    await zip.writeAsBytes(bytes);
    // macOS/Linux ship unzip; Windows 10+ ships bsdtar, which reads zips.
    final result = Platform.isWindows
        ? await Process.run('tar', [
            '-xf',
            zip.path,
            '-C',
            tmp.path,
            'lib/$libName',
          ])
        : await Process.run('unzip', [
            '-q',
            '-o',
            zip.path,
            'lib/$libName',
            '-d',
            tmp.path,
          ]);
    if (result.exitCode != 0) {
      throw ProcessException(
        Platform.isWindows ? 'tar' : 'unzip',
        const [],
        '${result.stderr}',
        result.exitCode,
      );
    }
    await dir.create(recursive: true);
    // Copy then rename, so a half-written library is never picked up.
    final partial = await File(
      '${tmp.path}/lib/$libName',
    ).copy('${dir.path}/$libName.partial');
    await partial.rename('${dir.path}/$libName');
  } finally {
    await tmp.delete(recursive: true);
  }
}
