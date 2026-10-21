#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/lib-release-cohort.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
git init -q -b main "$work/source"
git init -q --bare "$work/remote"
cd "$work/source"
git config user.name Test
git config user.email test@example.test
printf '{"version":"1.6.0-dev"}\n' > package.json
git add package.json
git commit -qm source
source_sha="$(git rev-parse HEAD)"
git remote add origin "$work/remote"
git push -q -u origin main
printf '{"version":"1.6.0","gitHead":"%s"}\n' "$source_sha" > package.json
git add package.json
git commit -qm release
release_sha="$(git rev-parse HEAD)"
git tag v1.6.0
git push -q origin main
git checkout -q --detach "$source_sha"
resume_release_tag 1.6.0 "$source_sha"
[[ "$(git rev-parse HEAD)" == "$release_sha" ]]
if resume_release_tag 1.6.0 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb; then
  echo 'Mismatched source was accepted' >&2; exit 1
fi
[[ "$(git rev-parse v1.6.0)" == "$release_sha" ]]
git checkout -q --detach "$source_sha"
printf '{"version":"1.6.0","gitHead":"%s"}\n' "$source_sha" > package.json
git add package.json
reuse_pushed_tree main
[[ "$(git rev-parse HEAD)" == "$release_sha" ]]
git checkout -q --detach "$source_sha"
printf '{"version":"1.7.0"}\n' > package.json
git add package.json
if reuse_pushed_tree main; then
  echo 'A different remote tree was accepted' >&2; exit 1
fi
[[ "$(git rev-parse HEAD)" == "$source_sha" ]]
echo 'Release retry/tag and competing-branch tests passed'
