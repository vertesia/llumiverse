#!/usr/bin/env bash
#
# Backport (cherry-pick) a merged PR onto one or more target branches and open a
# PR per target. Driven by `backport/<target>` labels on the merged PR.
#
# Typical flow: a maintenance fix merged into `release/X.Y` is forward-ported to
# `main` (label `backport/main`). Also supports `main -> release/X.Y` and
# `release/X.Y -> release/X.Z`.
#
#   Clean cherry-pick  -> a ready-to-merge PR into <target>.
#   Conflicting pick   -> conflict markers are committed and a DRAFT PR is opened,
#                         with resolution steps posted back on the original PR.
#
# Required env:
#   GH_TOKEN         App token with contents:write + pull-requests:write on REPO
#   REPO             owner/name (github.repository)
#   PR_NUMBER        merged PR number
#   PR_TITLE         merged PR title
#   PR_AUTHOR        merged PR author login
#   PR_URL           merged PR html_url
#   PR_HEAD_REF      merged PR head branch (never a backport target)
#   PR_LABELS_JSON   JSON array of {name} label objects from the event payload
#   MERGE_SHA        merge_commit_sha of the merged PR
#
# Optional env (provided by GitHub Actions): GITHUB_SERVER_URL, GITHUB_RUN_ID
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN required}"
: "${REPO:?REPO required}"
: "${PR_NUMBER:?PR_NUMBER required}"
: "${MERGE_SHA:?MERGE_SHA required}"
PR_TITLE="${PR_TITLE:-}"
PR_AUTHOR="${PR_AUTHOR:-}"
PR_URL="${PR_URL:-}"
PR_HEAD_REF="${PR_HEAD_REF:-}"
PR_LABELS_JSON="${PR_LABELS_JSON:-[]}"
RUN_URL="${GITHUB_SERVER_URL:-https://github.com}/${REPO}/actions/runs/${GITHUB_RUN_ID:-}"
export GH_TOKEN

# --- helpers --------------------------------------------------------------

comment_original() {
    gh pr comment "$PR_NUMBER" --repo "$REPO" --body "$1" >/dev/null 2>&1 \
        || echo "::warning::Could not comment on PR #${PR_NUMBER}."
}

clean_body() {
    local target="$1"
    cat <<EOF
Backport of #${PR_NUMBER} to \`${target}\`.

- Original PR: ${PR_URL}
- Author: @${PR_AUTHOR}
- Source commit: \`${MERGE_SHA}\`

Clean cherry-pick — no conflicts. CI runs as usual; review and merge.
EOF
}

conflict_body() {
    local target="$1" branch="$2"
    cat <<EOF
Backport of #${PR_NUMBER} to \`${target}\`. :warning: **The cherry-pick had conflicts.**

- Original PR: ${PR_URL}
- Author: @${PR_AUTHOR}
- Source commit: \`${MERGE_SHA}\`

Conflicting hunks were committed **with conflict markers** so this branch exists
for you to fix. This PR is a **draft** and stays red until the markers are gone.

### Resolve locally
\`\`\`sh
git fetch origin ${branch}
git switch ${branch}
# Resolve every <<<<<<< / ======= / >>>>>>> marker, then:
git add -A
git commit --amend --no-edit
git push --force-with-lease
\`\`\`

> **Submodule (gitlink) conflicts:** choose the intended pointer commit, then run
> \`git submodule update --init --recursive\` before building.
EOF
}

# --- parse backport/<target> labels ---------------------------------------

mapfile -t targets < <(
    jq -r '.[].name | select(startswith("backport/")) | sub("^backport/"; "")' <<<"$PR_LABELS_JSON" \
        | sort -u
)
if [ "${#targets[@]}" -eq 0 ]; then
    echo "No backport/<target> labels on PR #${PR_NUMBER}; nothing to do."
    exit 0
fi
echo "Backport targets for PR #${PR_NUMBER}: ${targets[*]}"

# --- git identity + authenticated remote ----------------------------------

git config user.email "github-actions[bot]@users.noreply.github.com"
git config user.name "github-actions[bot]"
git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/${REPO}.git"

# Make sure the merged commit object is present locally (GitHub serves it by SHA).
if ! git fetch --no-tags --quiet origin "$MERGE_SHA"; then
    echo "::error::Could not fetch merge commit ${MERGE_SHA}." >&2
    exit 1
fi

# A merge commit (>= 2 parents) needs `--mainline 1`; squash/rebase merges don't.
parent_count="$(git rev-list --parents -n 1 "$MERGE_SHA" | wc -w)"
cp_mainline=()
if [ "$parent_count" -ge 3 ]; then
    cp_mainline=(--mainline 1)
fi

# --- shared PR finalization -----------------------------------------------

# Echoes 1 if the pushed branch contains cherry-pick conflict markers, else 0.
# Defaults to 1 (open as draft) when the branch can't be inspected, so an
# unverified backport is never presented as ready-to-merge.
branch_has_conflict_markers() {
    local branch="$1"
    if ! git fetch --no-tags --quiet origin "$branch"; then
        echo 1
        return 0
    fi
    if git grep -qI -e '^<<<<<<< ' FETCH_HEAD -- . 2>/dev/null; then
        echo 1
    else
        echo 0
    fi
}

# Opens the backport PR for an already-pushed branch, assigns the author, adds the
# marker label, and reports back on the original PR. Draft iff conflicted. A failing
# `gh pr create` aborts (set -e) so a partial run fails loudly; the next run then
# recovers through the existing-branch path.
finalize_pr() {
    local target="$1" branch="$2" conflicted="$3"
    local title body
    local draft=()
    title="[Backport ${target}] ${PR_TITLE}"
    if [ "$conflicted" -eq 1 ]; then
        body="$(conflict_body "$target" "$branch")"
        draft=(--draft)
    else
        body="$(clean_body "$target")"
    fi

    local pr_url
    pr_url="$(gh pr create --repo "$REPO" --base "$target" --head "$branch" \
        --title "$title" --body "$body" "${draft[@]}")"
    echo "Opened ${pr_url}"

    if [ -n "$PR_AUTHOR" ]; then
        gh pr edit "$pr_url" --add-assignee "$PR_AUTHOR" >/dev/null 2>&1 \
            || echo "::warning::Could not assign @${PR_AUTHOR} on ${pr_url}."
    fi
    gh pr edit "$pr_url" --add-label "backport" >/dev/null 2>&1 \
        || echo "::warning::Could not add 'backport' label to ${pr_url} (create it in ${REPO})."

    if [ "$conflicted" -eq 1 ]; then
        comment_original ":warning: Opened a **draft** backport to \`${target}\` with conflicts to resolve: ${pr_url}"
    else
        comment_original ":white_check_mark: Opened a backport to \`${target}\`: ${pr_url}"
    fi
}

# --- backport one target --------------------------------------------------

backport_one() {
    local target="$1"

    # Start clean (a previous target may have left cherry-pick state behind).
    git cherry-pick --quit 2>/dev/null || true
    git reset --hard --quiet HEAD 2>/dev/null || true

    if ! [[ "$target" =~ ^(main|release/[0-9]+\.[0-9]+)$ ]]; then
        echo "::warning::Skipping invalid backport target '${target}'."
        comment_original ":warning: Skipped backport to \`${target}\`: not an allowed target (use \`main\` or \`release/X.Y\`)."
        return 0
    fi
    if [ "$target" = "$PR_HEAD_REF" ]; then
        echo "Skipping '${target}': same as the PR head ref."
        return 0
    fi

    local sanitized branch
    sanitized="${target//\//-}"
    branch="backport-${PR_NUMBER}-to-${sanitized}"

    # An existing backport branch may hold manual conflict fixes — never clobber it.
    # Tell three cases apart so a partial prior run (branch pushed, no PR) recovers
    # instead of being mistaken for "already done".
    if git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
        local prs_json open_url any_count
        prs_json="$(gh pr list --repo "$REPO" --head "$branch" --base "$target" --state all \
            --json url,state)"
        open_url="$(jq -r 'map(select(.state == "OPEN")) | .[0].url // ""' <<<"$prs_json")"
        any_count="$(jq 'length' <<<"$prs_json")"
        if [ -n "$open_url" ]; then
            echo "Open backport PR already exists: ${open_url}; leaving as-is."
            comment_original ":information_source: A backport PR to \`${target}\` already exists: ${open_url}"
            return 0
        fi
        if [ "$any_count" -gt 0 ]; then
            echo "A closed/merged backport PR for ${branch} exists; not reopening."
            comment_original ":information_source: A backport PR to \`${target}\` was already created and closed or merged; not reopening."
            return 0
        fi
        # Branch exists but no PR was ever opened -> a prior run pushed the branch
        # then failed before `gh pr create`. Recover by opening the PR now.
        echo "Backport branch ${branch} exists with no PR; recovering by opening the PR."
        finalize_pr "$target" "$branch" "$(branch_has_conflict_markers "$branch")"
        return 0
    fi

    if ! git fetch --no-tags --quiet origin "$target"; then
        echo "::warning::Target '${target}' not found on origin; skipping."
        comment_original ":warning: Skipped backport to \`${target}\`: branch not found on origin."
        return 0
    fi

    git checkout -B "$branch" "origin/${target}" --quiet

    local conflicted=0
    if ! git cherry-pick -x "${cp_mainline[@]}" "$MERGE_SHA"; then
        if [ -z "$(git status --porcelain)" ]; then
            git cherry-pick --quit 2>/dev/null || true
            echo "Cherry-pick onto ${target} is empty; change already present."
            comment_original ":information_source: Backport to \`${target}\` skipped: the change is already present."
            return 0
        fi
        if git diff --name-only --diff-filter=U | grep -q .; then
            echo "Conflicts cherry-picking onto ${target}; committing markers for a draft PR."
            git add -A
            git -c core.editor=true cherry-pick --continue
            conflicted=1
        else
            git cherry-pick --abort 2>/dev/null || true
            echo "::error::Unexpected cherry-pick failure onto ${target}." >&2
            comment_original ":x: Backport to \`${target}\` failed unexpectedly: ${RUN_URL}"
            return 1
        fi
    fi

    git push --quiet origin "HEAD:${branch}"

    finalize_pr "$target" "$branch" "$conflicted"
    return 0
}

# --- run, isolating each target so one failure can't abort the rest -------

overall_status=0
for target in "${targets[@]}"; do
    echo "::group::Backport to ${target}"
    if ! ( backport_one "$target" ); then
        overall_status=1
        echo "::error::Backport to ${target} failed."
    fi
    echo "::endgroup::"
done

exit "$overall_status"
