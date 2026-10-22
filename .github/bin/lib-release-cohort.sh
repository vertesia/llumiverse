#!/usr/bin/env bash
# Build provenance stays at an immutable version tag. Only its version commit is merged into a moving branch.
validate_release_source() {
  local ref="$1" source_sha="$2"
  [[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] || return 2
  git check-ref-format "refs/heads/$ref" || return 2
  git -c fetch.recurseSubmodules=false fetch --no-tags origin "refs/heads/$ref" || return 2
  if ! git merge-base --is-ancestor "$source_sha" FETCH_HEAD; then
    echo "Release source is not on $ref; refusing registry or Git writes" >&2
    return 2
  fi
}

release_identity() {
  node --input-type=module - "$1" "${2:-}" <<'JS'
import fs from 'node:fs';
import crypto from 'node:crypto';
console.log(JSON.stringify({source_sha:process.argv[2],dependency_sha:process.argv[3],
  lock_sha256:crypto.createHash('sha256').update(fs.readFileSync('pnpm-lock.yaml')).digest('hex')}));
JS
}

resume_release_tag() {
  local version="$1" source_sha="$2" tag="v$1" tag_sha
  if ! git rev-parse --verify "refs/tags/$tag" >/dev/null 2>&1; then return 1; fi
  tag_sha="$(git rev-parse "$tag^{commit}")" || return 2
  git show "$tag:package.json" | node -e '
    const {isDeepStrictEqual}=require("node:util");
    let data="";process.stdin.on("data",c=>data+=c);process.stdin.on("end",()=>{
      const p=JSON.parse(data);
      if(p.version!==process.argv[1] || p.gitHead!==process.argv[2] ||
         !isDeepStrictEqual(p.releaseIdentity,JSON.parse(process.argv[3])))process.exit(1);
    });' "$version" "$source_sha" "$RELEASE_IDENTITY" || {
      echo "Existing release tag belongs to a different source/dependency/lockfile" >&2; return 2;
    }
  if [[ -n "${LLUMIVERSE_SHA:-}" ]]; then
    [[ "$(git ls-tree "$tag_sha" llumiverse | awk '{print $3}')" == "$LLUMIVERSE_SHA" ]] || return 2
  fi
  git checkout --detach "$tag_sha" || return 2
  # A superproject checkout alone does not update a submodule's working tree.
  if [[ -f .gitmodules ]]; then git submodule update --init --recursive || return 2; fi
}

# 0 = matching immutable version; 1 = absent; 2 = conflict or registry failure.
check_npm_identity() {
  local package="$1" version="$2" metadata
  if metadata="$(npm view "$package@$version" --json 2>/dev/null)"; then
    printf '%s' "$metadata" | node -e '
      const {isDeepStrictEqual}=require("node:util");
      let data="";process.stdin.on("data",c=>data+=c);process.stdin.on("end",()=>{
        const p=JSON.parse(data), expected=JSON.parse(process.argv[1]);
        if(p.gitHead!==expected.source_sha || !isDeepStrictEqual(p.releaseIdentity,expected))process.exit(1);
      });' "$RELEASE_IDENTITY" || {
        echo "Published $package@$version has a different release identity" >&2; return 2;
      }
    return 0
  fi
  if printf '%s' "$metadata" | node -e '
    let data="";process.stdin.on("data",c=>data+=c);
    process.stdin.on("end",()=>{if(JSON.parse(data).error?.code!=="E404")process.exit(1);});' 2>/dev/null; then
    return 1
  fi
  echo "Cannot establish registry ownership of $package@$version" >&2
  return 2
}

persist_release_commit() {
  local version="$1" tag="v$1" message="chore: snapshot $1"
  git config user.email "github-actions[bot]@users.noreply.github.com" || return 2
  git config user.name "github-actions[bot]" || return 2
  if ! git rev-parse --verify "refs/tags/$tag" >/dev/null 2>&1; then
    git add -u || return 2
    if [[ "$RELEASE_TYPE" == "release" ]]; then message="chore: release $version"; fi
    if [[ "${COORDINATED_RELEASE:-false}" == "true" ]]; then
      git commit -m "$message" -m "Coordinated-Release: true" || return 2
    else
      git commit -m "$message" || return 2
    fi
    git tag "$tag" || return 2
  fi
  git push origin "refs/tags/$tag" || return 2
}

propagate_release() {
  local ref="$1" version="$2" release_sha remote_sha attempt message
  release_sha="$(git rev-parse "refs/tags/v$version^{commit}")" || return 2
  for attempt in 1 2 3 4 5; do
    git -c fetch.recurseSubmodules=false fetch --no-tags origin "refs/heads/$ref" || return 2
    remote_sha="$(git rev-parse FETCH_HEAD)" || return 2
    if git merge-base --is-ancestor "$release_sha" "$remote_sha"; then
      echo "Version commit is already on $ref (possibly behind newer commits)"
      return 0
    fi
    git diff --quiet && git diff --cached --quiet || return 2
    if git merge-base --is-ancestor "$remote_sha" "$release_sha"; then
      git checkout --detach "$release_sha" || return 2
    else
      git checkout --detach "$remote_sha" || return 2
      message="chore: propagate v$version"
      if [[ "${COORDINATED_RELEASE:-false}" == "true" ]]; then
        message="$(printf '%s\n\n%s' "$message" 'Coordinated-Release: true')"
      fi
      if ! git merge --no-ff -m "$message" "$release_sha"; then
        git merge --abort || return 2
        echo "Version metadata conflicts with $ref. Resolve it using immutable tag v$version; never republish under another source." >&2
        return 2
      fi
    fi
    if [[ -f .gitmodules ]]; then git submodule update --init --recursive || return 2; fi
    if git push origin "HEAD:refs/heads/$ref"; then return 0; fi
    echo "Branch advanced during propagation; retrying the version merge ($attempt/5)" >&2
  done
  echo "Branch remains busy; resume the same immutable version tag" >&2
  return 2
}
