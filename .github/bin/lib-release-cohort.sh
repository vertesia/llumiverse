#!/usr/bin/env bash
# A retry may reuse an existing release tag, but it must never change its source.
resume_release_tag() {
  local version="$1" source_sha="$2" tag="v$1" tag_sha recorded_source
  if ! git rev-parse --verify "refs/tags/$tag" >/dev/null 2>&1; then
    return 1
  fi
  tag_sha="$(git rev-parse "$tag^{commit}")" || return 2
  recorded_source="$(git show "$tag:package.json" | node -e '
    let data=""; process.stdin.on("data", c => data += c); process.stdin.on("end", () => {
      const p=JSON.parse(data); if (p.version !== process.argv[1]) process.exit(1); console.log(p.gitHead);
    });' "$version")" || return 2
  if [[ "$recorded_source" != "$source_sha" && "$tag_sha" != "$source_sha" ]]; then
    echo "Existing release tag belongs to another source commit" >&2
    return 2
  fi
  git checkout --detach "$tag_sha" || return 2
}

# After a successful publish followed by a failed smoke test, the version commit may already
# be on the branch. Reuse that exact tree instead of creating a competing commit on retry.
reuse_pushed_tree() {
  local ref="$1" remote_sha
  git fetch origin "$ref" || return 2
  remote_sha="$(git rev-parse FETCH_HEAD)" || return 2
  if [[ "$(git rev-parse HEAD)" == "$remote_sha" ]]; then
    return 1
  fi
  if [[ "$(git write-tree)" != "$(git rev-parse "$remote_sha^{tree}")" ]]; then
    echo "Publish branch moved to a different tree; refusing to overwrite it" >&2
    return 2
  fi
  git reset --hard "$remote_sha" || return 2
}
