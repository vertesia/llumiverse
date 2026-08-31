#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib-package-channel.sh"

assert_channel() {
  local expected="$1"
  shift
  local actual
  actual="$(resolve_package_channel "$@")"
  if [ "$actual" != "$expected" ]; then
    echo "Expected '$expected', got '$actual' for: $*" >&2
    exit 1
  fi
}

assert_channel latest release/1.5 release 0123456789abcdef0123456789abcdef01234567
assert_channel dev main snapshot 0123456789abcdef0123456789abcdef01234567
assert_channel dev-1.5 release/1.5 snapshot 0123456789abcdef0123456789abcdef01234567
assert_channel snapshot-0123456 1.4 snapshot 0123456789abcdef0123456789abcdef01234567
assert_channel snapshot-0123456 feat/provider-test snapshot 0123456789abcdef0123456789abcdef01234567
assert_channel snapshot-0123456 feat/short-sha snapshot 0123456

if resolve_package_channel feat/provider-test snapshot invalid >/dev/null 2>&1; then
  echo 'Expected an invalid feature-branch SHA to fail' >&2
  exit 1
fi

sha_error="$(resolve_package_channel feat/provider-test snapshot invalid 2>&1 || true)"
if [[ "$sha_error" != *"7-40 character git SHA"* ]]; then
  echo "Expected the SHA validation error to describe the accepted length, got: $sha_error" >&2
  exit 1
fi

if resolve_package_channel main invalid 0123456789abcdef0123456789abcdef01234567 >/dev/null 2>&1; then
  echo 'Expected an invalid release type to fail' >&2
  exit 1
fi

echo 'Package channel tests passed'
