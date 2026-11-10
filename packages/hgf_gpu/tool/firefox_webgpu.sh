#!/bin/sh
# Firefox with WebGPU enabled, for package:test (which only takes absolute
# executables, hence the variable):
#   FIREFOX_EXECUTABLE=$PWD/tool/firefox_webgpu.sh \
#     dart test -p firefox -c dart2wasm --tags gpu
# package:test starts Firefox with `--profile DIR` and its own prefs.js, so
# this adds the WebGPU prefs to that profile first.
# FIREFOX overrides the binary (default: firefox on the PATH).
prev=
for a in "$@"; do
  if [ "$prev" = "--profile" ]; then
    cat >>"$a/prefs.js" <<'PREFS'
user_pref("dom.webgpu.enabled", true);
user_pref("dom.webgpu.workers.enabled", true);
user_pref("gfx.webgpu.ignore-blocklist", true);
user_pref("gfx.webgpu.force-enabled", true);
PREFS
  fi
  prev=$a
done
exec "${FIREFOX:-firefox}" --headless "$@"
