#!/bin/bash
set -e

# Script to publish all llumiverse packages to NPM
# Usage: publish-all-packages.sh <ref> [dry-run] [version-type]
#   ref: Git reference (main or preview)
#   dry-run: Optional boolean (true/false) for dry run mode
#   version-type: Version type (dev, patch, minor, major)

REF=$1
DRY_RUN=${2:-false}
VERSION_TYPE=${3:-dev}

if [ -z "$REF" ]; then
  echo "Error: Missing required argument: ref"
  echo "Usage: $0 <ref> [dry-run] [version-type]"
  exit 1
fi

# Validate version type
if [[ ! "$VERSION_TYPE" =~ ^(dev|patch|minor|major)$ ]]; then
  echo "Error: Invalid version type '$VERSION_TYPE'. Must be dev, patch, minor, or major."
  exit 1
fi

# Check if dry run mode is enabled
if [ "$DRY_RUN" = "true" ]; then
  echo "=== DRY RUN MODE ENABLED ==="
  DRY_RUN_FLAG="--dry-run"
else
  DRY_RUN_FLAG=""
fi

# Packages to publish (in dependency order)
PACKAGES=(
  common
  core
  drivers
)

# Step 1: Update all package versions
echo "=== Updating package versions ==="

# Determine npm tag based on branch
if [ "$REF" = "main" ]; then
  npm_tag="dev"
elif [ "$REF" = "preview" ]; then
  npm_tag="latest"
else
  npm_tag="experimental"
fi

if [ "$VERSION_TYPE" = "dev" ]; then
  # Dev: create dev version with date/time stamp
  # Use format without leading zeros to avoid npm stripping them
  base_version=$(pnpm pkg get version | tr -d '"')
  date_part=$(date -u +"%Y%m%d")
  time_part=$(date -u +"%H%M%S" | sed 's/^0*//')  # Remove leading zeros
  dev_version="${base_version}-dev.${date_part}.${time_part}"
  echo "Updating to dev version ${dev_version}"

  # Update root package.json
  npm version ${dev_version} --no-git-tag-version --workspaces=false

  # Update all workspace packages
  pnpm -r --filter "./*" exec npm version ${dev_version} --no-git-tag-version
else
  # Release: bump version (patch, minor, or major)
  echo "Bumping ${VERSION_TYPE} version"

  # Update root package.json
  npm version ${VERSION_TYPE} --no-git-tag-version --workspaces=false

  # Get the new version from root
  new_version=$(pnpm pkg get version | tr -d '"')
  echo "Setting all packages to version ${new_version}"

  # Set all workspace packages to the same version as root
  pnpm -r --filter "./*" exec npm version ${new_version} --no-git-tag-version
fi

# Step 2: Publish packages (in dependency order)
echo "=== Publishing llumiverse packages ==="

for pkg in "${PACKAGES[@]}"; do
  if [ -d "$pkg" ] && [ -f "$pkg/package.json" ]; then
    cd "$pkg"

    pkg_version=$(pnpm pkg get version | tr -d '"')
    echo "Publishing @llumiverse/${pkg}@${pkg_version} with tag ${npm_tag}"

    # Publish
    if [ -n "$DRY_RUN_FLAG" ]; then
      pnpm publish --access public --tag ${npm_tag} --no-git-checks ${DRY_RUN_FLAG}
    else
      pnpm publish --access public --tag ${npm_tag} --no-git-checks
    fi

    cd ..
  fi
done

# Step 3: Verify published packages (only in dry-run mode)
if [ "$DRY_RUN" = "true" ]; then
  echo "=== Verifying package tarballs ==="

  # Array to track failed packages
  failed_packages=()

  # Get the expected version
  if [ "$VERSION_TYPE" = "dev" ]; then
    expected_version="${dev_version}"
  else
    expected_version=$(pnpm pkg get version | tr -d '"')
  fi

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
fi

# Step 4: Commit version changes (only for release versions + not dry-run)
if [ "$VERSION_TYPE" != "dev" ] && [ "$DRY_RUN" = "false" ]; then
  echo "=== Committing version changes ==="

  git config user.email "github-actions[bot]@users.noreply.github.com"
  git config user.name "github-actions[bot]"
  git add .
  git commit -m "chore: bump package versions (${VERSION_TYPE})"
  git push origin ${REF}

  echo "Version changes pushed to ${REF}"
fi

echo "=== Done ==="
