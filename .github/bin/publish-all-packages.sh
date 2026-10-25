#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib-package-channel.sh"
source "${SCRIPT_DIR}/lib-release-cohort.sh"

# Script to publish all llumiverse packages to NPM
# Usage: publish-all-packages.sh --ref <ref> --release-type <type> --bump-type <type> [--dry-run [true|false]]
#   --ref: Git reference (main for dev builds, release/X.Y for releases, e.g. release/1.4)
#   --release-type: Release type (release, snapshot). Release creates stable versions, snapshot creates dev versions.
#   --bump-type: Bump type (minor, patch, keep). How to change the version.
#   --dry-run: Optional flag for dry run mode (value can be true, false, or omitted which means true)

# Packages to publish (in dependency order)
PACKAGES=(
  common
  core
  drivers
)

# =============================================================================
# Functions
# =============================================================================

update_package_versions() {
  echo "=== Updating package versions ==="

  if [[ -n "${VERSION_OVERRIDE:-}" ]]; then
    new_version="$VERSION_OVERRIDE"
  else
  # Get current version and strip any existing -dev* suffix to get base version
  current_version=$(npm pkg get version | tr -d '"')
  base_version=$(echo "$current_version" | sed 's/-dev.*//')

  # Apply bump if needed (for both snapshot and release)
  if [ "$BUMP_TYPE" = "minor" ]; then
    # Bump minor version: X.Y.Z -> X.(Y+1).0
    IFS='.' read -r major minor patch <<< "$base_version"
    base_version="${major}.$((minor + 1)).0"
    echo "Bumped minor version to ${base_version}"
  elif [ "$BUMP_TYPE" = "patch" ]; then
    # Bump patch version: X.Y.Z -> X.Y.(Z+1)
    IFS='.' read -r major minor patch <<< "$base_version"
    base_version="${major}.${minor}.$((patch + 1))"
    echo "Bumped patch version to ${base_version}"
  fi

  if [ "$RELEASE_TYPE" = "snapshot" ]; then
    # Snapshot: create dev version with date/time stamp
    date_part=$(date -u +"%Y%m%d")
    time_part=$(date -u +"%H%M%SZ")
    new_version="${base_version}-dev.${date_part}.${time_part}"
    echo "Updating to snapshot version ${new_version}"
  else
    # Release: use base version as-is
    new_version="${base_version}"
    echo "Updating to release version ${new_version}"
  fi

  fi

  # Update root package.json
  npm version "${new_version}" --no-git-tag-version --allow-same-version --workspaces=false

  # Update all workspace packages
  pnpm -r --filter "./*" exec npm version "${new_version}" --no-git-tag-version --allow-same-version
  npm pkg set gitHead="$source_sha" --workspaces=false
  npm pkg set "releaseIdentity=$RELEASE_IDENTITY" --json --workspaces=false
  for pkg in "${PACKAGES[@]}"; do
    (cd "$pkg" && npm pkg set gitHead="$source_sha" && npm pkg set "releaseIdentity=$RELEASE_IDENTITY" --json)
  done
}

publish_packages() {
  echo "=== Publishing llumiverse packages ==="

  package_index=0
  for pkg in "${PACKAGES[@]}"; do
    tarball="${PACKAGE_TARBALLS[$package_index]}"
    package_index=$((package_index + 1))
    if [ -d "$pkg" ] && [ -f "$pkg/package.json" ]; then
      cd "$pkg"

      pkg_version=$(npm pkg get version | tr -d '"')

      # Fail if npm_tag is not set (safety check to prevent publishing without explicit tag)
      if [ -z "$npm_tag" ]; then
        echo "Error: npm_tag is not set. This indicates an invalid ref/version-type combination."
        exit 1
      fi

      echo "Publishing @llumiverse/${pkg}@${pkg_version} with tag ${npm_tag}"

      if [[ "$DRY_RUN" == "false" ]]; then
        local ownership=0
        check_npm_identity "@llumiverse/${pkg}" "$pkg_version" || ownership=$?
        if [[ "$ownership" == "0" ]]; then cd ..; continue; fi
        [[ "$ownership" == "1" ]] || return 1
      fi

      # Publish
      if [ -n "$DRY_RUN_FLAG" ]; then
        pnpm publish "$tarball" --access public --tag "${npm_tag}" --no-git-checks ${DRY_RUN_FLAG}
      else
        pnpm publish "$tarball" --access public --tag "${npm_tag}" --no-git-checks
      fi

      cd ..
    fi
  done
}

prepare_verified_tarballs() {
  echo "=== Packing and verifying all package tarballs before publication ==="
  local expected_version pkg directory
  local tarballs
  PACKAGE_TARBALLS=()
  expected_version=$(node -p "require('./package.json').version")
  for pkg in "${PACKAGES[@]}"; do
    if [[ ! -f "$pkg/package.json" ]]; then
      echo "Missing package manifest: $pkg/package.json" >&2
      return 1
    fi
    directory="$verification_dir/$pkg"
    mkdir -p "$directory"
    (cd "$pkg" && pnpm pack --pack-destination "$directory")
    tarballs=("$directory"/*.tgz)
    if [[ ${#tarballs[@]} -ne 1 || ! -f "${tarballs[0]}" ]]; then
      echo "Expected exactly one packed tarball for $pkg" >&2
      return 1
    fi
    node "$SCRIPT_DIR/verify-package-tarball.mjs" "${tarballs[0]}" "$expected_version"
    PACKAGE_TARBALLS+=("${tarballs[0]}")
  done
}

write_github_summary() {
  # Skip if not running in GitHub Actions
  if [ -z "$GITHUB_STEP_SUMMARY" ]; then
    echo "Skipping GitHub summary (not running in GitHub Actions)"
    return
  fi

  echo "=== Writing GitHub Summary ==="

  # Get the version from root package.json
  version=$(npm pkg get version | tr -d '"')

  # Determine title based on dry run mode
  if [ "$DRY_RUN" = "true" ]; then
    title="## 🧪 Dry Run Summary"
  else
    title="## 📦 Published Packages"
  fi

  # Write summary table
  cat >> "$GITHUB_STEP_SUMMARY" << EOF
${title}

| Package | Version |
| ------- | ------- |
EOF

  for pkg in "${PACKAGES[@]}"; do
    pkg_name="@llumiverse/${pkg}"
    pkg_url="https://www.npmjs.com/package/@llumiverse/${pkg}?activeTab=versions"
    echo "| \`${pkg_name}\` | [${version}](${pkg_url}) |" >> "$GITHUB_STEP_SUMMARY"
  done

  # Add metadata
  cat >> "$GITHUB_STEP_SUMMARY" << EOF

**NPM Tag:** \`${npm_tag}\`
**Branch:** \`${REF}\`
**Dry Run:** \`${DRY_RUN}\`
EOF
}

# =============================================================================
# Argument parsing and validation
# =============================================================================

# Default values
REF=""
DRY_RUN=false
RELEASE_TYPE=""
BUMP_TYPE=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --ref)
      REF="$2"
      shift 2
      ;;
    --dry-run)
      # Check if next argument is a value (true/false) or another flag/end of args
      if [[ -n "$2" && "$2" != --* ]]; then
        if [[ "$2" = "true" ]]; then
          DRY_RUN=true
        elif [[ "$2" = "false" ]]; then
          DRY_RUN=false
        else
          echo "Error: Invalid value for --dry-run '$2'. Must be 'true' or 'false'."
          exit 1
        fi
        shift 2
      else
        # No value provided, default to true
        DRY_RUN=true
        shift
      fi
      ;;
    --release-type)
      RELEASE_TYPE="$2"
      shift 2
      ;;
    --bump-type)
      BUMP_TYPE="$2"
      shift 2
      ;;
    *)
      echo "Error: Unknown argument '$1'"
      echo "Usage: $0 --ref <ref> --release-type <type> --bump-type <type> [--dry-run [true|false]]"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$REF" ]; then
  echo "Error: Missing required argument: --ref"
  echo "Usage: $0 --ref <ref> --release-type <type> --bump-type <type> [--dry-run [true|false]]"
  exit 1
fi

if [ -z "$RELEASE_TYPE" ]; then
  echo "Error: Missing required argument: --release-type"
  echo "Usage: $0 --ref <ref> --release-type <type> --bump-type <type> [--dry-run [true|false]]"
  exit 1
fi

if [ -z "$BUMP_TYPE" ]; then
  echo "Error: Missing required argument: --bump-type"
  echo "Usage: $0 --ref <ref> --release-type <type> --bump-type <type> [--dry-run [true|false]]"
  exit 1
fi

# Validate release type
if [[ ! "$RELEASE_TYPE" =~ ^(release|snapshot)$ ]]; then
  echo "Error: Invalid release type '$RELEASE_TYPE'. Must be 'release' or 'snapshot'."
  exit 1
fi

# Validate bump type
if [[ ! "$BUMP_TYPE" =~ ^(minor|patch|keep)$ ]]; then
  echo "Error: Invalid bump type '$BUMP_TYPE'. Must be 'minor', 'patch', or 'keep'."
  exit 1
fi

# Validate that releases can only be published from a release branch (release/X.Y, e.g. release/1.4)
if [ "$RELEASE_TYPE" = "release" ] && [[ "$REF" != release/* ]]; then
  echo "Error: Release versions can only be published from a 'release/*' branch (e.g. 'release/1.4')."
  echo "Current branch: $REF"
  exit 1
fi

# Set dry run flag
if [ "$DRY_RUN" = "true" ]; then
  echo "=== DRY RUN MODE ENABLED ==="
  DRY_RUN_FLAG="--dry-run"
else
  DRY_RUN_FLAG=""
fi

source_sha="${RELEASE_SOURCE_SHA:-$(git rev-parse HEAD)}"
validate_release_source "$REF" "$source_sha"
npm_tag=$(resolve_package_channel "$REF" "$RELEASE_TYPE" "$source_sha")

echo "=== Package channel ==="
echo "Source ref: ${REF}"
echo "Source SHA: ${source_sha}"
echo "Target tag: ${npm_tag}"

# =============================================================================
# Main flow
# =============================================================================

RELEASE_IDENTITY="$(release_identity "$source_sha" "")"
export RELEASE_IDENTITY
resume_release=false
if [[ -n "${VERSION_OVERRIDE:-}" ]]; then
  resume_status=0
  resume_release_tag "$VERSION_OVERRIDE" "$source_sha" || resume_status=$?
  if [[ "$resume_status" == "0" ]]; then
    resume_release=true
    new_version="$VERSION_OVERRIDE"
  elif [[ "$resume_status" != "1" ]]; then
    exit "$resume_status"
  fi
fi
if [[ "$resume_release" == "false" ]]; then
  update_package_versions
fi

echo "=== Building all packages ==="
pnpm build

verification_dir=$(mktemp -d)
trap 'rm -rf "$verification_dir"' EXIT
prepare_verified_tarballs

if [[ "$DRY_RUN" == "false" ]]; then
  for pkg in "${PACKAGES[@]}"; do
    ownership=0
    check_npm_identity "@llumiverse/$pkg" "$new_version" || ownership=$?
    [[ "$ownership" != "2" ]] || exit 1
  done
  if [[ "$resume_release" == "false" ]]; then persist_release_commit "$new_version"; fi
fi

publish_packages

if [[ "$DRY_RUN" == "false" ]]; then
  propagate_release "$REF" "$new_version"
fi

write_github_summary

echo "=== Done ==="
