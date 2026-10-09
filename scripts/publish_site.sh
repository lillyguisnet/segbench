#!/usr/bin/env bash
# Publish the points site (scripts/build_site.py) to GitHub Pages.
#
#     scripts/publish_site.sh
#
# Builds the site, then replaces the gh-pages branch with it: one commit, no
# history (the site is rebuilt from results each time; the code that built it
# is on main). Commits as this repository's git user (Lilly) and pushes
# through this repository's credentials. Then makes sure Pages serves the
# branch. The site is live a minute later at
# https://lillyguisnet.github.io/segbench/
set -euo pipefail
cd "$(dirname "$0")/.."

repo=lillyguisnet/segbench
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT

uv run scripts/build_site.py --out "$out/site"
built_from=$(git rev-parse --short HEAD)
dirty=$(git diff --quiet HEAD -- scripts segbench assets || echo " (with uncommitted changes)")

git -C "$out/site" init -q -b gh-pages
git -C "$out/site" config user.name "$(git config user.name)"
git -C "$out/site" config user.email "$(git config user.email)"
git -C "$out/site" add -A
git -C "$out/site" commit -q -m "Points site, built from main at ${built_from}${dirty}"
git -C "$out/site" -c credential.helper= -c "credential.https://github.com.helper=$(git config --get-all credential.https://github.com.helper | tail -1)" \
  push -q -f "$(git remote get-url origin)" gh-pages:gh-pages

token=$(gh auth token -h github.com -u lillyguisnet)
# GitHub may switch Pages on by itself for a new gh-pages branch: "already enabled" is fine.
if ! GH_TOKEN=$token gh api "repos/$repo/pages" >/dev/null 2>&1; then
  GH_TOKEN=$token gh api -X POST "repos/$repo/pages" -f "source[branch]=gh-pages" -f "source[path]=/" >/dev/null 2>&1 \
    || GH_TOKEN=$token gh api "repos/$repo/pages" >/dev/null
fi
echo "published: https://lillyguisnet.github.io/segbench/ (live in about a minute)"
