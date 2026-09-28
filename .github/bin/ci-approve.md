# CI-based PR approval

`ci-approve.yaml` uses the existing `vertesia-automerge` App to approve ready,
same-repository PRs targeting `main` or `release/X.Y` after the current head passes
this repository's `automerge-ci-policy.json` against the current base branch and
commit. The CI run's recorded PR base must match; missing or older base metadata
blocks approval. Studio uses its selected-suite
validator; composableai also requires plugin template tests; llumiverse includes
curated live provider smoke tests. A green workflow with skipped required jobs
or steps does not qualify. Copilot and review threads are not prerequisites.

## Review ownership

Only an APPROVED review authored by `vertesia-automerge[bot]` with the exact
`<!-- vertesia-ci-approval:v1 -->` marker at the start of its body belongs to this
gate. Human reviews and reviews from the existing deployment, Renovate, release,
backport and submodule gates are never dismissed by this workflow.

On a new push, the gate removes its approvals for older commits without inspecting
CI. When a CI workflow finishes, it approves the current head only if all required
lint, build and test checks pass; otherwise it withdraws its approval. Draft and
eligibility changes also trigger reassessment. Manual dispatch recovers missed
events or API failures. There is no scheduled reconciliation or rerun-start handler.

During a same-commit rerun, an existing approval can remain until CI finishes.
Runner queues can delay withdrawal after a push. A delayed push event preserves
an approval already granted for the current commit. If the target branch advances
or the PR is retargeted, run CI with the updated base (typically by updating the PR
branch). Re-running an old workflow may retain its original base metadata. Base
updates alone do not trigger this gate; reevaluation happens on its next event or
manual dispatch.

Approval writes explicitly name the tested commit. PR metadata and CI are read
again before publication and after a new review is submitted. Events are serialized
per repository, and delayed events always evaluate the latest PR state.

`human-review-required` opts a PR out of automatic review. Changes to `.github/`,
`.githooks/`, `scripts/`, package manifests and build/test configuration also require
human review. Bot-authored and `deployment` PRs retain their existing review route.
These PRs still receive the CI status when tests pass. Fork PRs are not approved.

## Permissions and trusted code

The workflow checks out `github.workflow_sha`, never PR code, and installs no
packages. Node 24 on `ubuntu-slim` runs the dependency-free scripts directly.
`GITHUB_TOKEN` reads PR/CI metadata and writes the `PR approval gate` commit status.
Only review creation and dismissal use the App token, scoped to the current
repository with `pull-requests: write`. The App must remain a distinct identity
from the PR author.

The script and tests are identical across studio, composableai and llumiverse,
following the existing `automerge-ci.mjs` distribution pattern. CI requirements
remain repository-local; update the shared approval implementation in all three.

## Activation

1. Merge the scripts and workflows into `main` in all three repositories, and
   backport them to each maintained `release/X.Y` base. `pull_request_target` uses
   the base branch's workflow; `workflow_run` uses the default branch.
2. Confirm each `renovate-automerge` environment provides
   `APP_VERTESIA_RENOVATE_AUTOMERGE_PEM`, and the environment/repository/organization
   supplies `APP_VERTESIA_RENOVATE_AUTOMERGE_CLIENT_ID` for the existing App.
3. Exercise a same-repository PR: observe the marked approval after CI, push an
   update, and verify that only this gate's approval disappears until CI passes.
4. Once the status is being emitted, import `.github/rulesets/ci-approval.json` as
   an additional ruleset. It requires `PR approval gate` from GitHub Actions
   (integration ID 15368) on `main` and `release/**`. Keep
   `dismiss_stale_reviews_on_push` false in existing rulesets. This template makes
   no changes to review counts, conversation resolution, or Copilot configuration.

The ruleset JSON is not applied automatically. Do not import it before the workflow
is installed on the covered bases. With it enabled, forks without supported CI are
blocked; use a reviewed same-repository branch for such contributions. Existing
specialized automerge jobs must also wait for this status before merging.

A new head has no successful gate status, so requiring the status closes the normal
new-push/stale-approval gap. During a same-commit rerun, the previous successful status can remain until
a CI workflow finishes and triggers reassessment. If reruns must block merges immediately,
require the native lint/build checks as well. No asynchronous approval workflow can
make GitHub's push/rerun and review APIs atomic.

## Validation

The lint workflow runs `node --test .github/bin/ci-approve.test.mjs` in its existing
CI-policy test step. Tests cover human-review preservation, stale heads, reruns,
publication races, opt-out, file-list completeness, and API failures. Run
`pnpm lint:actions` to validate workflows. The narrowly scoped actionlint exception
for `concurrency.queue` follows GitHub's supported queue syntax until actionlint
understands that key.
