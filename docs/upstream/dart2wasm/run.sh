#!/usr/bin/env bash
# Build and run the dart2wasm repros. Needs dart (3.13+) and node; wasm-opt
# ships with the Dart SDK (bin/utils/wasm-opt). Pin to a core (taskset) for
# steadier timings.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
sdk=$(dirname "$(readlink -f "$(command -v dart)")")
wasm_opt=$(find "$sdk/.." -path '*bin/utils/wasm-opt' -type f | head -1)

# Print the body of function $2 from the disassembly $1.
body() {
  F="(func \$$2 " awk 'index($0, ENVIRON["F"]) {p=1} p {print} p && /^ \)$/ {exit}' "$1"
}

echo "== typed-data views (dart compile wasm -O4)"
for view in false true; do
  dart compile wasm -O4 --no-strip-wasm -Dview=$view \
    "$here/typed_data_views.dart" -o "$out/views_$view.wasm" >/dev/null
  "$wasm_opt" --all-features --print "$out/views_$view.wasm" >"$out/views_$view.wat"
  echo "-- view in program: $view; kernel reads _offsetInElements" \
    "$(body "$out/views_$view.wat" Engine.kernel | grep -c _offsetInElements) time(s)"
done
for round in 1 2 3; do
  for view in false true; do
    node "$repo/tool/web/run_wasm.mjs" "$out/views_$view.wasm"
  done
done

echo
echo "== code generation (dart compile wasm -O2 --no-strip-wasm)"
dart compile wasm -O2 --no-strip-wasm "$here/codegen.dart" \
  -o "$out/codegen.wasm" >/dev/null
"$wasm_opt" --all-features --print "$out/codegen.wasm" >"$out/codegen.wat"
for fn in intDivision doubleDivision; do
  echo "-- $fn: constants and calls"
  body "$out/codegen.wat" $fn | grep -E 'call|f64.const' | sed 's/^ */   /'
done
echo "-- tableLookup: $(body "$out/codegen.wat" tableLookup | wc -l) lines for" \
  "one table read: a null check on the global, the initialiser inlined:"
body "$out/codegen.wat" tableLookup |
  grep -E 'br_on_non_null|global.get \$table|array.new_fixed' | sed 's/^ */   /'
