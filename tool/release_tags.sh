#!/usr/bin/env bash
# Tag every publishable package and every app whose current version has no
# tag yet (`<package>-v<version>`, the form .github/workflows/release.yml and
# `melos version` use). Prints the tags; pass --push to push them.
#
#   tool/release_tags.sh          # create the tags locally
#   tool/release_tags.sh --push   # ...and push them (starts the releases)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PUSH=false
[ "${1:-}" = "--push" ] && PUSH=true

[ -z "$(git status --porcelain --untracked-files=no)" ] ||
  { echo "error: uncommitted changes; commit the release first." >&2; exit 1; }

field() { sed -nE "s/^$2:[[:space:]]*['\"]?([^'\"[:space:]#]+).*/\1/p" "$1" | head -n1; }

TAGS=()
for pubspec in packages/*/pubspec.yaml apps/*/pubspec.yaml; do
  # Apps are never on pub.dev but are released (with their builds).
  [[ "$pubspec" == packages/* ]] && [ "$(field "$pubspec" publish_to)" = "none" ] && continue
  name="$(field "$pubspec" name)"
  version="$(field "$pubspec" version)"
  tag="$name-v${version%%+*}"
  if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
    continue
  fi
  git tag -a "$tag" -m "$name ${version%%+*}"
  TAGS+=("$tag")
  echo "tagged $tag"
done

[ ${#TAGS[@]} -gt 0 ] || { echo "Nothing to tag: every publishable package's and app's version is tagged."; exit 0; }

# One push per tag: GitHub starts no workflows for a push of more than three
# tags. Dependents wait for their dependencies on pub.dev, so order is free.
if $PUSH; then
  for tag in "${TAGS[@]}"; do git push origin "refs/tags/$tag"; done
else
  echo "Push with: tool/release_tags.sh --push (or git push origin <tag>, one tag per push)"
fi
