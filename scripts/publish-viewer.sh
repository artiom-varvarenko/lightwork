#!/usr/bin/env bash
# Publish only an intact, audited Meta bundle to an explicitly named public GitHub repository.
set -euo pipefail
usage() {
  printf '%s\n' 'Usage: scripts/publish-viewer.sh --repo OWNER/REPO [--site site] (--dry-run | --publish)'
}
repo='' site='site' mode=''
while (($#)); do
  case "$1" in
    --repo|--site)
      (($# >= 2)) || { usage >&2; exit 2; }
      if [[ "$1" == --repo ]]; then repo=$2; else site=$2; fi
      shift 2 ;;
    --dry-run|--publish)
      [[ -z "$mode" ]] || { usage >&2; exit 2; }
      mode=$1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ "$repo" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ && -n "$mode" ]] || { usage >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ "$site" == /* ]] || site="$PWD/$site"
# Query GitHub even for a dry run: a private/missing target must never pass the release gate.
visibility=$(gh api "repos/$repo" --jq 'if .private == false and .visibility == "public" then "public" else "refused" end')
[[ "$visibility" == public ]] || { printf '%s\n' 'Refusing: target repository is not public.' >&2; exit 1; }
# Validate hashes, path containment, symlinks, 100 MiB cap, provenance and fresh deterministic verification.
uv run --project "$root" python -c 'from pathlib import Path; import sys; from lightwork.runs.export import validate_site; b=validate_site(Path(sys.argv[1])); print("Validated", len(b["runs"]), "Meta run(s), 0 diffs")' "$site"
if [[ "$mode" == --dry-run ]]; then
  printf 'Dry run OK: would replace gh-pages on https://github.com/%s from %s; no push performed.\n' "$repo" "$site"
  exit 0
fi
# A disposable repository keeps release operations away from development history.
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
mkdir -- "$tmp/repo"
cp -a -- "$site/." "$tmp/repo/"
uv run --project "$root" python -c 'from pathlib import Path; import sys; from lightwork.runs.export import validate_site; validate_site(Path(sys.argv[1]))' "$tmp/repo"
# No original repository history is published.
git -C "$tmp/repo" init --quiet
git -C "$tmp/repo" config user.name 'Lightwork viewer release'
git -C "$tmp/repo" config user.email 'viewer-release@users.noreply.github.com'
git -C "$tmp/repo" add --all
git -C "$tmp/repo" commit --quiet -m 'Publish verified Lightwork recorded replay'
git -C "$tmp/repo" push --force "https://github.com/$repo.git" HEAD:gh-pages
printf 'Published verified viewer to %s branch gh-pages. Enable GitHub Pages for that branch.\n' "$repo"
