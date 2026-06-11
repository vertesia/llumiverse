#!/usr/bin/env bash
set -euo pipefail

ACTIONLINT_VERSION="${ACTIONLINT_VERSION:-v1.7.12}"
ACTIONLINT_ARGS=("-shellcheck" "" "-pyflakes" "")

if command -v actionlint >/dev/null 2>&1; then
    exec actionlint "${ACTIONLINT_ARGS[@]}" "$@"
fi

if command -v go >/dev/null 2>&1; then
    echo "actionlint not found; running actionlint ${ACTIONLINT_VERSION} via go run." >&2
    exec go run "github.com/rhysd/actionlint/cmd/actionlint@${ACTIONLINT_VERSION}" "${ACTIONLINT_ARGS[@]}" "$@"
fi

cat >&2 <<EOF
Error: actionlint is not installed and Go is unavailable for the fallback runner.

Install actionlint:
  brew install actionlint

Or install the pinned CLI:
  go install github.com/rhysd/actionlint/cmd/actionlint@${ACTIONLINT_VERSION}
EOF
exit 127
