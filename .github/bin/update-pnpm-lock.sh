#!/bin/bash
set -euo pipefail

# Required environment variables
: "${PR_NUMBER:?PR_NUMBER is required}"
: "${GITHUB_TOKEN:?GITHUB_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

# Extract PR details
pr_json=$(gh pr view "$PR_NUMBER" --json headRefName,baseRefName)
PR_BRANCH=$(jq -r '.headRefName' <<< "$pr_json")
BASE_REF=$(jq -r '.baseRefName' <<< "$pr_json")

git config --global user.email "github-actions[bot]@users.noreply.github.com"
git config --global user.name "github-actions[bot]"

if [[ -n "${GITHUB_TOKEN:-}" && -n "${GITHUB_REPOSITORY:-}" ]]; then
  git remote set-url origin "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
fi

# Fetch branches
git fetch origin "$BASE_REF:refs/remotes/origin/$BASE_REF"
git fetch origin "$PR_BRANCH:refs/remotes/origin/$PR_BRANCH"

# Checkout the PR branch
git checkout -B "$PR_BRANCH" "origin/$PR_BRANCH"

# If the base branch is already contained in the PR branch there is nothing to repair.
# This short-circuits stale workflow_run events fired after a repair push.
if git merge-base --is-ancestor "origin/$BASE_REF" HEAD; then
  echo "Base branch $BASE_REF is already contained in $PR_BRANCH; skipping lock repair."
  exit 0
fi

# Cap the number of automated repairs per branch. Under sustained base-branch churn the
# branch can be re-dirtied faster than it merges; without a cap that is an unbounded
# repair/push/CI cycle. Past the threshold, hand the conflict to a human.
MAX_LOCK_REPAIRS=3
repair_commits=$(git rev-list --count --author='github-actions' "origin/$BASE_REF..HEAD")
if [[ "$repair_commits" -ge "$MAX_LOCK_REPAIRS" ]]; then
  echo "Already pushed $repair_commits github-actions[bot] commit(s) to $PR_BRANCH; skipping further lock repair (needs manual resolution)." >&2
  exit 0
fi

# Merge the base branch into the PR branch. pnpm-lock.yaml conflicts are
# deterministic: accept one side temporarily, then regenerate the lockfile.
if ! GIT_MERGE_AUTOEDIT=no git merge --no-edit "origin/$BASE_REF"; then
  mapfile -t conflicted_files < <(git diff --name-only --diff-filter=U)
  if [[ "${#conflicted_files[@]}" -ne 1 || "${conflicted_files[0]}" != "pnpm-lock.yaml" ]]; then
    echo "Merge conflict detected outside pnpm-lock.yaml; skipping lockfile repair." >&2
    printf 'Unresolved conflicts:\n' >&2
    printf '  %s\n' "${conflicted_files[@]}" >&2
    git merge --abort || true
    exit 2
  fi

  echo "Only pnpm-lock.yaml conflicted; regenerating it from merged manifests."
  git checkout --ours pnpm-lock.yaml
fi

# Use the package manager version declared by the checked-out PR branch.
PACKAGE_MANAGER=$(node -p "require('./package.json').packageManager || ''")
if [[ "$PACKAGE_MANAGER" != pnpm@* ]]; then
  echo "package.json must declare packageManager as pnpm@<version>" >&2
  exit 1
fi
corepack enable
corepack prepare "$PACKAGE_MANAGER" --activate
pnpm --version

# Run pnpm install to update the lockfile with the same pnpm version/config as CI.
pnpm install --lockfile-only

# Commit the merge and/or updated lockfile.
git add pnpm-lock.yaml
if [[ -f .git/MERGE_HEAD ]]; then
  git commit --no-edit
elif ! git diff --cached --quiet; then
  git commit -m "chore: update lockfile"
fi

ahead_count=$(git rev-list --count "origin/$PR_BRANCH"..HEAD)
if [[ "$ahead_count" == "0" ]]; then
  echo "No merge or lockfile changes to push."
  exit 0
fi

# Push the changes.
git push origin "$PR_BRANCH"

echo "Updated lockfile pushed to branch $PR_BRANCH."
