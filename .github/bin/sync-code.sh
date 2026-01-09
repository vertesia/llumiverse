#!/bin/bash
#
# This script is used to sync code from one branch to another, mainly from preview to main.
#
# option `-x` is used to enable debugging, which will print each command before executing it.
set -x

# Prerequisites
# -----
source_ref="${SOURCE_REF:?Environment variable SOURCE_REF is not set}"
source_sha="${SOURCE_SHA:?Environment variable SOURCE_SHA is not set}"

target_ref="${TARGET_REF:?Environment variable TARGET_REF is not set}"

merge_message="Auto-merge branch '${source_ref}' (${source_sha::7})

Generated-by: https://github.com/vertesia/llumiverse/actions/runs/${GITHUB_RUN_ID:-0}"

git config --global user.email "github-actions[bot]@users.noreply.github.com"
git config --global user.name "github-actions[bot]"


# Checkout new branch
# -----
temp_branch="${TEMP_BRANCH:?Environment variable TEMP_BRANCH is not set}"
echo "[INFO] Creating a temporary branch \"${temp_branch}\" to sync code from \"${source_ref}\" to \"${target_ref}\"" >&2
git branch "$temp_branch" "$source_sha"
git checkout "$temp_branch"


# Sync code
# -----
# option `--no-ff` is used to ensure that the merge is recorded as a merge commit
git merge "origin/${target_ref}" --no-ff -m "$merge_message"
is_merged=$?

# Handle merge conflicts
# -----
if [ $is_merged -ne 0 ]; then
    if git diff --quiet && ! git ls-files -u | grep .; then
        echo "[INFO] Successfully resolved conflicts and staged changes" >&2
        # note: we don't use `git merge --continue` because it requires an editor which is not
        # available in the GitHub Actions environment.
        git commit -m "$merge_message"
    else
        echo "[ERROR] Failed to continue the merge due to conflicts. Please resolve conflicts manually and commit the changes." >&2
        git diff --name-only --diff-filter=U >&2
        # TODO send a pull request to the target branch
        exit 1
    fi
else
    echo "[INFO] Successfully merged code from \"${source_ref}\" to \"${target_ref}\"" >&2
fi


# Push changes
# -----
echo "[INFO] Pushing changes to remote branch \"${temp_branch}\"" >&2
if ! git push origin "${temp_branch}"; then
    echo "[ERROR] Failed to push changes to remote branch ${temp_branch}" >&2
    exit 1
fi