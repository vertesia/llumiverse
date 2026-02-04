#!/bin/bash
set -e

# Script to publish all llumiverse packages to NPM
# Usage: publish-all-packages.sh --ref <ref> --release-type <type> --bump-type <type> [--dry-run [true|false]]
#   --ref: Git reference (main for dev builds, preview for releases)
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

  # Determine npm tag based on release type
  if [ "$RELEASE_TYPE" = "snapshot" ]; then
    npm_tag="dev"
  else
    npm_tag="latest"
  fi

  # Get current version and strip any existing -dev* suffix to get base version
  current_version=$(pnpm pkg get version | tr -d '"')
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

  # Update root package.json
  npm version "${new_version}" --no-git-tag-version --workspaces=false

  # Update all workspace packages
  pnpm -r --filter "./*" exec npm version "${new_version}" --no-git-tag-version
}

publish_packages() {
  echo "=== Publishing llumiverse packages ==="

  for pkg in "${PACKAGES[@]}"; do
    if [ -d "$pkg" ] && [ -f "$pkg/package.json" ]; then
      cd "$pkg"

      pkg_version=$(pnpm pkg get version | tr -d '"')

      # Fail if npm_tag is not set (safety check to prevent publishing without explicit tag)
      if [ -z "$npm_tag" ]; then
        echo "Error: npm_tag is not set. This indicates an invalid ref/version-type combination."
        exit 1
      fi

      echo "Publishing @llumiverse/${pkg}@${pkg_version} with tag ${npm_tag}"

      # Publish
      if [ -n "$DRY_RUN_FLAG" ]; then
        pnpm publish --access public --tag "${npm_tag}" --no-git-checks ${DRY_RUN_FLAG}
      else
        pnpm publish --access public --tag "${npm_tag}" --no-git-checks
      fi

      cd ..
    fi
  done
}

verify_published_packages() {
  echo "=== Verifying package tarballs ==="

  # Array to track failed packages
  failed_packages=()

  # Get the expected version from root package.json
  expected_version=$(pnpm pkg get version | tr -d '"')

  for pkg in "${PACKAGES[@]}"; do
    if [ -d "$pkg" ]; then
      cd "$pkg"
      pkg_name="@llumiverse/${pkg}"

      echo "Packing ${pkg_name}..."
      pnpm pack --pack-destination . > /dev/null 2>&1
      tarball=$(ls -t *.tgz 2>/dev/null | head -1)

      if [ -n "$tarball" ] && [ -f "$tarball" ]; then
        echo "Checking ${pkg_name}:"
        packed_json=$(tar -xzOf "$tarball" package/package.json)

        # Check version
        packed_version=$(echo "$packed_json" | grep '"version":' | head -1 | sed 's/.*: "\(.*\)".*/\1/')
        has_issues=false

        if [ "$packed_version" = "${expected_version}" ]; then
          echo "  ✓ Version: ${packed_version}"
        else
          echo "  ✗ WARNING: Version mismatch (expected: ${expected_version}, got: ${packed_version})"
          has_issues=true
        fi

        # Extract dependencies section
        deps_section=$(echo "$packed_json" | sed -n '/"dependencies":/,/^  [}]/p')

        # Check for @llumiverse dependencies
        llumiverse_deps=$(echo "$deps_section" | grep '"@llumiverse/' || true)
        if [ -n "$llumiverse_deps" ]; then
          echo "$llumiverse_deps"
          if echo "$llumiverse_deps" | grep -q "${expected_version}"; then
            echo "  ✓ llumiverse dependencies: ${expected_version}"
          else
            echo "  ✗ WARNING: llumiverse dependencies version mismatch"
            has_issues=true
          fi
        fi

        # Add to failed packages if there were issues
        if [ "$has_issues" = true ]; then
          failed_packages+=("${pkg_name}")
        fi

        # Clean up tarball
        rm -f "$tarball"
      fi

      cd ..
    fi
  done

  # Print summary
  echo ""
  echo "=== Verification Summary ==="
  if [ ${#failed_packages[@]} -eq 0 ]; then
    echo "✓ All packages passed verification"
  else
    echo "✗ ${#failed_packages[@]} package(s) failed verification:"
    for pkg in "${failed_packages[@]}"; do
      echo "  - ${pkg}"
    done
  fi
}

commit_and_push() {
  echo "=== Committing version changes ==="

  # Get the version from root package.json
  version=$(pnpm pkg get version | tr -d '"')

  git config user.email "github-actions[bot]@users.noreply.github.com"
  git config user.name "github-actions[bot]"
  git add .

  if [ "$RELEASE_TYPE" = "release" ]; then
    git commit -m "chore: release ${version}"
  else
    git commit -m "chore: snapshot ${version}"
  fi

  git push origin "$REF"

  echo "Version changes pushed to ${REF}"
}

write_github_summary() {
  # Skip if not running in GitHub Actions
  if [ -z "$GITHUB_STEP_SUMMARY" ]; then
    echo "Skipping GitHub summary (not running in GitHub Actions)"
    return
  fi

  echo "=== Writing GitHub Summary ==="

  # Get the version from root package.json
  version=$(pnpm pkg get version | tr -d '"')

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

# Validate that releases can only be published from 'preview' branch
if [ "$RELEASE_TYPE" = "release" ] && [ "$REF" != "preview" ]; then
  echo "Error: Release versions can only be published from the 'preview' branch."
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

# =============================================================================
# Main flow
# =============================================================================

update_package_versions

if [ "$DRY_RUN" = "false" ]; then
  commit_and_push
fi

publish_packages

if [ "$DRY_RUN" = "true" ]; then
  verify_published_packages
fi

write_github_summary

echo "=== Done ==="
