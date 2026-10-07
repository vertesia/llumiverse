# NPM package publication

Run `.github/bin/publish-all-packages.sh` from the Llumiverse repository root. It publishes this
coordinated dependency set in order:

1. `@llumiverse/conversation`
2. `@llumiverse/common`
3. `@llumiverse/core`
4. `@llumiverse/drivers`

The conversation package is currently private and experimental. Publication of this cohort is
blocked until its adoption and release gate is complete. The script checks this before modifying
versions, building, committing, or publishing. It also rejects a missing workspace dependency or an
incorrect publication order. Do not remove the private flag just to make a release command pass.

## Workflow

Use `.github/workflows/publish-npm.yaml` on the intended source branch. Run first with
`dry_run=true`, inspect its results, then use `dry_run=false` for the authorized publication.

```bash
./.github/bin/publish-all-packages.sh \
    --ref main \
    --release-type snapshot \
    --bump-type keep \
    --dry-run
```

| Argument | Values |
| --- | --- |
| `--ref` | Source branch/ref; required. |
| `--release-type` | `snapshot` or `release`; required. Releases require a `release/*` branch. |
| `--bump-type` | `keep`, `patch`, or `minor`; required. |
| `--dry-run` | Optional `true`/`false`; omitting its value means `true`. Omitting the flag means a real publication. The GitHub workflow defaults to `true`. |

Both modes update local package versions, build all packages, and pack and validate all tarballs.
Validation checks every package version, resolved dependency range, and declared JavaScript/type
entry point. Missing artifacts or mismatched dependencies stop the script before a Git push or npm
publication. ESM output uses `lib/`; no obsolete `lib/esm` or `lib/cjs` directory is required.

A dry run then calls `pnpm publish --dry-run` without committing or pushing. **It changes local
package versions and build outputs**, so use a disposable clean checkout. A real run commits and
pushes the version changes before publishing each package. The script never removes `private`.

## Versions and channels

The root version determines the coordinated package version. `patch`/`minor` updates that base;
`snapshot` adds `-dev.YYYYMMDD.HHMMSSZ`. `workspace:*` dependencies are resolved by pnpm to the
corresponding exact package version when packed.

| Source and release type | npm tag |
| --- | --- |
| `main`, snapshot | `dev` |
| `release/X.Y`, snapshot | `dev-X.Y` |
| Other ref, snapshot | `snapshot-<source SHA prefix>` |
| `release/*`, release | `latest` |

For a release-line dry run, for example:

```bash
./.github/bin/publish-all-packages.sh \
    --ref release/1.6 \
    --release-type release \
    --bump-type keep \
    --dry-run
```

The workflow requires Node 24, pnpm from `packageManager`, and npm 11.5.1 or later for trusted
publication. Its real publication step needs the configured npm trust and Git push credentials.

## Local contract checks

These checks use isolated fixtures and do not publish or change package versions:

```bash
node --test .github/bin/verify-package-release.test.mjs
bash .github/bin/lib-package-channel.test.sh
```

The publication contract tests run in the repository lint workflow.
