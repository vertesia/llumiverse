#!/bin/bash

# Resolve the npm dist-tag owned by a source track. Feature branches never own a
# shared moving tag; their snapshot tag is derived from the immutable source SHA.
resolve_package_channel() {
  local ref="$1"
  local release_type="$2"
  local source_sha="$3"

  if [[ ! "$release_type" =~ ^(release|snapshot)$ ]]; then
    echo "Error: Release type must be 'release' or 'snapshot'." >&2
    return 1
  fi

  if [ "$release_type" = "release" ]; then
    printf '%s\n' "latest"
    return
  fi

  if [ "$ref" = "main" ]; then
    printf '%s\n' "dev"
    return
  fi

  if [[ "$ref" =~ ^release/([0-9]+\.[0-9]+)$ ]]; then
    printf 'dev-%s\n' "${BASH_REMATCH[1]}"
    return
  fi

  if [[ ! "$source_sha" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
    echo "Error: A 7-40 character git SHA is required to publish an isolated branch or tag snapshot." >&2
    return 1
  fi

  printf 'snapshot-%s\n' "${source_sha:0:7}"
}
