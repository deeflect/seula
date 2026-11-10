#!/bin/sh
# Builds the commit that goes to GitHub: HEAD without the video projects (single files over GitHub's 100 MB limit),
# the unrelated labs, working notes and the old invoice-demo share assets. It moves the local branch `public`; push with: git push origin public:main
set -eu
cd "$(git rev-parse --show-toplevel)"
export GIT_INDEX_FILE="$(mktemp -u)"
git ls-tree -r HEAD | grep -Ev "	(video/|labs/|design/|HANDOFF|LEDGER\.md|DECISION\.md|docs/commercial\.md|docs/v1-HANDOFF\.md|docs/v1-DECISION\.md|docs/demo\.md|docs/demo-assets/)" | git update-index --index-info
tree=$(git write-tree)
parent=$(git rev-parse -q --verify refs/heads/public || true)
[ -n "$parent" ] && [ "$(git rev-parse "$parent^{tree}")" = "$tree" ] && { echo "public is up to date"; exit 0; }
git update-ref refs/heads/public "$(git commit-tree "$tree" ${parent:+-p "$parent"} -m "${1:-$(git log -1 --format=%s)}")"
git log -1 --format='public -> %h %an <%ae> %s' public
