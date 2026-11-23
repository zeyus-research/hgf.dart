#!/usr/bin/env bash
# Wait until every workspace package that <package> depends on (dependencies,
# not dev_dependencies) is on pub.dev at the version in its pubspec. Tags of a
# release train are pushed together, so hgf_fit waits here for hgf and
# hgf_deep instead of failing to resolve against pub.dev.
#
#   .github/scripts/wait_workspace_deps.sh hgf_fit [timeout-minutes]
set -euo pipefail

PACKAGE="${1:?usage: $0 <package> [timeout-minutes]}"
TIMEOUT="${2:-60}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

fail() { echo "::error::$*" >&2; exit 1; }

pubspec_of() {
  local p
  for p in "$ROOT"/packages/"$1"/pubspec.yaml "$ROOT"/apps/"$1"/pubspec.yaml; do
    if [ -f "$p" ]; then echo "$p"; return 0; fi
  done
  return 0
}
field() { sed -nE "s/^$2:[[:space:]]*['\"]?([^'\"[:space:]#]+).*/\1/p" "$1" | head -n1; }

PUBSPEC="$(pubspec_of "$PACKAGE")"
[ -n "$PUBSPEC" ] || fail "No pubspec for '$PACKAGE'."

# Names under the top-level `dependencies:` block.
DEPS="$(awk '
  /^[^[:space:]#]/ { in_deps = ($0 ~ /^dependencies:/); next }
  in_deps && /^  [a-z_][a-z0-9_]*:/ { sub(/^  /, ""); sub(/:.*/, ""); print }
' "$PUBSPEC")"

WAIT=()
for dep in $DEPS; do
  dep_pubspec="$(pubspec_of "$dep")"
  [ -n "$dep_pubspec" ] || continue
  [ "$(field "$dep_pubspec" publish_to)" = "none" ] &&
    fail "$PACKAGE depends on $dep, which is publish_to: none."
  WAIT+=("$dep@$(field "$dep_pubspec" version)")
done

[ ${#WAIT[@]} -gt 0 ] || { echo "$PACKAGE has no workspace dependencies."; exit 0; }

DEADLINE=$(( $(date +%s) + TIMEOUT * 60 ))
for item in "${WAIT[@]}"; do
  dep="${item%@*}"; version="${item#*@}"
  until [ "$(curl -s -o /dev/null -w '%{http_code}' "https://pub.dev/api/packages/$dep/versions/$version")" = 200 ]; do
    [ "$(date +%s)" -lt "$DEADLINE" ] ||
      fail "$dep $version is still not on pub.dev after $TIMEOUT minutes (push its tag: $dep-v$version)."
    echo "Waiting for $dep $version on pub.dev…"
    sleep 30
  done
  echo "$dep $version is on pub.dev."
done
