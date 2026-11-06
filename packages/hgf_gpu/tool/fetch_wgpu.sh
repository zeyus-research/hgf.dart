#!/usr/bin/env bash
# Downloads the wgpu-native release that lib/src/native/wgpu_bindings.g.dart
# was generated for into .dart_tool/wgpu (Linux x86_64 and aarch64, macOS,
# Windows). Run from packages/hgf_gpu.
set -euo pipefail
VERSION=v29.0.1.1
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) ASSET=wgpu-linux-x86_64-release.zip ;;
  Linux-aarch64) ASSET=wgpu-linux-aarch64-release.zip ;;
  Darwin-arm64) ASSET=wgpu-macos-aarch64-release.zip ;;
  Darwin-x86_64) ASSET=wgpu-macos-x86_64-release.zip ;;
  MINGW*|MSYS*) ASSET=wgpu-windows-x86_64-msvc-release.zip ;;
  *) echo "No wgpu-native build for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
DEST=.dart_tool/wgpu
mkdir -p "$DEST"
curl -fsSL -o "$DEST/wgpu.zip" \
  "https://github.com/gfx-rs/wgpu-native/releases/download/$VERSION/$ASSET"
unzip -q -o "$DEST/wgpu.zip" -d "$DEST"
rm "$DEST/wgpu.zip"
echo "$DEST/lib"
