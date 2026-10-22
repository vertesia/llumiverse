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
printf 'lockfileVersion: 9\n' > pnpm-lock.yaml
git add .
git commit -qm source
source_sha="$(git rev-parse HEAD)"
git remote add origin "$work/remote"
git push -q -u origin main
git clone -q "$work/remote" "$work/contributor" --branch main
git -C "$work/contributor" config user.name Other
git -C "$work/contributor" config user.email other@example.test
# Main moves while the pinned source is being built. Do not put these changes in the artifact.
printf 'new feature\n' > "$work/contributor/feature.txt"
git -C "$work/contributor" add .
git -C "$work/contributor" commit -qm feature
git -C "$work/contributor" push -q origin main
validate_release_source main "$source_sha"
RELEASE_IDENTITY="$(release_identity "$source_sha" '')"
export RELEASE_IDENTITY
RELEASE_TYPE=snapshot COORDINATED_RELEASE=true
version=1.6.0-dev.20261021.123456Z
node - "$version" "$source_sha" "$RELEASE_IDENTITY" <<'JS'
const fs=require('node:fs');fs.writeFileSync('package.json',JSON.stringify({version:process.argv[2],gitHead:process.argv[3],releaseIdentity:JSON.parse(process.argv[4])})+'\n');
JS
persist_release_commit "$version"
release_sha="$(git rev-parse "v$version")"
! git cat-file -e "$release_sha:feature.txt" 2>/dev/null
# Inject another ordinary commit at the first push, after propagation has fetched main.
injected=false
git() {
  if [[ "$1" == "push" && "${3:-}" == "HEAD:refs/heads/main" && "$injected" == "false" ]]; then
    injected=true
    printf 'racing feature\n' > "$work/contributor/racing.txt"
    command git -C "$work/contributor" add .
    command git -C "$work/contributor" commit -qm racing
    command git -C "$work/contributor" push -q origin main
    return 1
  fi
  command git "$@"
}
# A restarted runner has no identity from persist_release_commit.
git config --unset user.name
git config --unset user.email
GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null propagate_release main "$version"
unset -f git
[[ "$injected" == "true" ]]
[[ "$(git --git-dir="$work/remote" show main:feature.txt)" == 'new feature' ]]
[[ "$(git --git-dir="$work/remote" show main:racing.txt)" == 'racing feature' ]]
git --git-dir="$work/remote" merge-base --is-ancestor "$release_sha" main
[[ "$(git --git-dir="$work/remote" log -1 --format=%B main)" == *'Coordinated-Release: true'* ]]
# Retry after successful propagation AND after another main commit; tag identity stays unchanged.
git -C "$work/contributor" fetch -q origin main
git -C "$work/contributor" reset -q --hard FETCH_HEAD
printf 'later\n' > "$work/contributor/later.txt"
git -C "$work/contributor" add .
git -C "$work/contributor" commit -qm later
git -C "$work/contributor" push -q origin main
git checkout -q --detach "$source_sha"
resume_release_tag "$version" "$source_sha"
propagate_release main "$version"
[[ "$(git rev-parse "v$version")" == "$release_sha" ]]
[[ "$(git --git-dir="$work/remote" show main:later.txt)" == 'later' ]]
# A different dependency pin cannot reuse a partially published version.
original_identity="$RELEASE_IDENTITY"
RELEASE_IDENTITY="$(release_identity "$source_sha" bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)"
if resume_release_tag "$version" "$source_sha"; then echo 'Accepted a different dependency' >&2; exit 1; fi
RELEASE_IDENTITY="$original_identity"
# A source that has never been on main must fail before publishing.
printf 'feature branch only\n' > unpublished.txt
git add unpublished.txt
git commit -qm unpublished
if validate_release_source main "$(git rev-parse HEAD)"; then echo 'Accepted an unrelated source' >&2; exit 1; fi
# npm preflight refuses mismatched provenance, distinguishes a missing version from an API failure.
npm() { printf '%s\n' "$registry_metadata"; return "$registry_status"; }
registry_status=0
registry_metadata="$(git show "v$version:package.json")"
check_npm_identity @example/package "$version"
registry_metadata='{"gitHead":"different"}'
if check_npm_identity @example/package "$version"; then echo 'Accepted a foreign registry version' >&2; exit 1; else [[ "$?" == "2" ]]; fi
registry_status=1 registry_metadata='{"error":{"code":"E404"}}'
if check_npm_identity @example/package "$version"; then exit 1; else [[ "$?" == "1" ]]; fi
registry_metadata='{"error":{"code":"E403"}}'
if check_npm_identity @example/package "$version"; then exit 1; else [[ "$?" == "2" ]]; fi
echo 'Moving main, rejected push, durable snapshot, retry, source membership and registry identity tests passed'
