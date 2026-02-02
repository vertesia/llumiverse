# NPM Package Publishing Scripts

This directory contains scripts for publishing llumiverse packages to NPM with different versioning strategies.

## Overview

The `publish-all-packages.sh` script handles publishing the following packages in dependency order:
- `@llumiverse/common`
- `@llumiverse/core`
- `@llumiverse/drivers`

## Usage

```bash
./publish-all-packages.sh --ref <ref> --version-type <type> --dry-run <value>
```

### Parameters

- `--ref` (required): Git reference - `main` for dev builds, other branches for releases
- `--version-type` (required): Version bump type (`minor`, `patch`, `dev`)
  - `minor` increases the minor version in the package version
  - `patch` increases the patch version in the package version
  - `dev` creates a new development version in format `{base-version}-dev.{date}.{time}`, such as `1.0.0-dev.20260128.144200Z`. Note that the time part contains 'Z', which means that the time is in UTC; it also allows NPM to use leading zeros, as it turns the segment into alphanumeric.
- `--dry-run` (optional): Flag to enable dry run mode. The value can be `true`, `false` or no value (which means `true`). If not specified, it means that it is not a dry-run.

### Examples

```bash
# Dry run for main branch
./publish-all-packages.sh --ref main --dry-run --version-type dev
./publish-all-packages.sh --ref main --dry-run true --version-type dev

# Publish release with patch bump
./publish-all-packages.sh --ref preview --dry-run --version-type patch

# Publish release with minor bump
./publish-all-packages.sh --ref preview --version-type minor
```

## Scenarios

### Scenario 1: Publishing from `main` branch

**Purpose**: Publish development versions for testing

**Steps**:
1. Updates all package versions to dev format
   - Version format: `{base-version}-dev.{YYYYMMDD}.{time}` (e.g., `0.23.0-dev.20251218.131500`)
2. Publishes all packages in dependency order
   - NPM tag: `dev`
3. Commit and push changes back to the branch (only if dry-run is false), but do not create Git tag

**Result**:
- All packages published with `dev` tag
- Consumers can install with: `npm install @llumiverse/core@dev`

**Example**:
```bash
# Before (package.json):
# @llumiverse/common: 0.23.0
# @llumiverse/core: 0.23.0
# @llumiverse/drivers: 0.23.0

# After publishing (on npm):
# @llumiverse/common@0.23.0-dev.20251218.131500 (tag: dev)
# @llumiverse/core@0.23.0-dev.20251218.131500 (tag: dev)
# @llumiverse/drivers@0.23.0-dev.20251218.131500 (tag: dev)
```

### Scenario 2: Publishing from other branches (e.g., `preview`)

**Purpose**: Publish official releases

**Steps**:
1. Bumps root `package.json` version using semantic versioning
   - Bump type: specified by `version-type` parameter (patch/minor)
2. Updates all package versions to match
   - Version format: standard semver (e.g., `0.23.0` → `0.23.1` for patch)
3. Publishes all packages in dependency order
   - NPM tag: `latest`
4. **Commits and pushes** version changes back to the branch (only if dry-run is false)

**Result**:
- All packages published with `latest` tag
- Consumers can install with: `npm install @llumiverse/core` (gets latest)
- Git repository updated with new versions

**Example (patch bump)**:
```bash
# Before (package.json):
# @llumiverse/common: 0.23.0

# After publishing (on npm):
# @llumiverse/common@0.23.1 (tag: latest)

# Git commit:
# "chore: bump package versions (patch)"
```

### Scenario 3: Dry Run Mode

**Purpose**: Test the publishing process without actually publishing

**Steps**:
- All version updates happen normally
- `npm publish` commands run with `--dry-run` flag
- Package tarballs are created and verified
- No actual packages are published to NPM
- No git commits are made

**Usage**:

```bash
# Test main branch publishing
./publish-all-packages.sh --ref main --dry-run --version-type dev

# Test release publishing with minor bump
./publish-all-packages.sh --ref preview --dry-run --version-type minor
```

**Result**:
- Shows what would be published
- Validates package versions and dependencies
- Safe to run multiple times
- No side effects

## GitHub Actions Workflow

The script is designed to be run from the `publish-npm.yaml` GitHub Actions workflow:

```yaml
- name: Publish all packages
  run: ./.github/bin/publish-all-packages.sh \
      --ref "${{ inputs.ref }}" \
      --dry-run "${{ inputs.dry_run }}" \
      --version-type "${{ inputs.version_type }}"
```

### Workflow Inputs

- `ref`: Text input for git reference (default: `main`) → maps to `--ref`
- `dry_run`: Checkbox (default: true for safety) → maps to `--dry-run true` or `--dry-run false`
- `version_type`: Dropdown for `patch`, `minor`, or `dev` → maps to `--version-type`

## Key Features

### Dependency Order

Packages are published in dependency order to ensure dependencies are available:
1. `@llumiverse/common` (no internal dependencies)
2. `@llumiverse/core` (depends on common)
3. `@llumiverse/drivers` (depends on common and core)

### Version Resolution

pnpm automatically resolves `workspace:*` dependencies during publish:
- When `@llumiverse/drivers` references `"@llumiverse/core": "workspace:*"`
- pnpm reads the actual version from `core/package.json`
- The published package will contain the exact version

### Verification (Dry Run)

In dry run mode, the script:
- Packs each package into a tarball
- Extracts and verifies the version matches expected
- Checks that internal dependencies point to correct versions
- Reports any mismatches

### Safety

- Dry run enabled by default in GitHub Actions
- All version updates happen before any publishing
- Portable shell syntax (works on macOS and Linux)

### Requirements

- pnpm workspace setup
- npm 11.5.1 or later
