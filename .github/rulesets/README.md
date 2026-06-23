# RuleSets

These JSON files mirror rulesets defined in this repository's settings. Import them
at <https://github.com/vertesia/llumiverse/settings/rules>. Note that editing these
files does **not** apply to GitHub automatically — it requires a manual import by
someone with an administrator role on the repo (or `gh api
repos/vertesia/llumiverse/rulesets --input <file>`, dropping the export-only
`source`/`source_type` fields). Reach out to the `#dev` Slack channel if needed.

## `release.json`

Protects the `release/X.Y` release lines coordinated by the studio release process
(see `docs/release-process.md` in `vertesia/studio`). It builds on this repo's `main`
ruleset (PR + 1 approval, no deletion / non-fast-forward) but targets
`refs/heads/release/**` instead of the default branch, and **adds a CodeQL
`code_scanning` rule** that `main` does not have: `release/X.Y` is the production
line, so CodeQL must gate it. CodeQL default setup already scans pull requests
targeting protected branches, so protecting `release/**` makes those scans run
automatically — no advanced-setup workflow is needed.

Note: this is distinct from the existing `maintenance` ruleset, which targets bare
`X.Y` branches (the older scheme); the coordinated release lines use the
`release/X.Y` prefix.

When importing, an admin must configure the **bypass actors** (left empty in the
committed JSON, as for `main`) so the release automation can operate on `release/*`:

- the submodule-sync automerge app (so `(sync) Update Git submodule` PRs auto-merge
  into `release/X.Y`), and
- the actor used by studio's `release-cut.yaml` for `bump_via=push` to push the
  version-bump commits past protection (the GitHub App that mints the cross-repo
  token, or the same deploy-key/automation actor used on `main`). The `bump_via=pr`
  fallback opens PRs instead and needs no push bypass — use it until the bypass is
  configured.
