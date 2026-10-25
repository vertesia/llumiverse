import { execFileSync } from 'node:child_process';
import { appendFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { CONTEXT } from './ci-approve.mjs';

const SHA = /^[0-9a-f]{40}$/;

// ci-approve.yaml starts a specialized automerge workflow after publishing its successful status.
// Act only on that exact revision: the PR must still have the same head and base, and the latest
// gate status on the head must be the Actions-published success.
export function dispatchedRevisionReason(pr, statuses, { repo, headBranch, headSha, baseRef, baseSha }) {
    if (pr.state !== 'open') return 'the PR is no longer open';
    if (pr.head.repo?.full_name !== repo) return 'the PR head is not in this repository';
    if (pr.head.ref !== headBranch) return `the head branch is ${pr.head.ref}, not ${headBranch}`;
    if (pr.head.sha !== headSha) return `the head moved from ${headSha} to ${pr.head.sha}`;
    if (pr.base.ref !== baseRef) return `the base branch is ${pr.base.ref}, not ${baseRef}`;
    if (pr.base.sha !== baseSha) return `the base moved from ${baseSha} to ${pr.base.sha}`;
    // GitHub lists statuses newest first.
    const gate = statuses.find((status) => status.context === CONTEXT);
    if (gate?.state !== 'success' || gate.creator?.login !== 'github-actions[bot]') {
        return `the ${CONTEXT} status is not successful for ${headSha}`;
    }
    return undefined;
}

export function ghApi(endpoint, { pages = false } = {}) {
    const args = ['api', endpoint];
    if (pages) args.push('--paginate', '--slurp');
    return JSON.parse(execFileSync('gh', args, { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 }));
}

export function main(env, api = ghApi) {
    const number = Number(env.PR_NUMBER);
    const { REPO: repo, HEAD_BRANCH: headBranch, HEAD_SHA: headSha, BASE_REF: baseRef, BASE_SHA: baseSha } = env;
    if (!Number.isSafeInteger(number) || number <= 0) throw new Error('Invalid PR number');
    if (
        !/^[\w.-]+\/[\w.-]+$/.test(repo ?? '') ||
        !headBranch ||
        !baseRef ||
        !SHA.test(headSha) ||
        !SHA.test(baseSha)
    ) {
        throw new Error('Missing or malformed repository, branch, head or base');
    }
    const pr = api(`repos/${repo}/pulls/${number}`);
    const statuses = api(`repos/${repo}/commits/${headSha}/statuses?per_page=100`, { pages: true }).flat();
    const reason = dispatchedRevisionReason(pr, statuses, { repo, headBranch, headSha, baseRef, baseSha });
    console.log(reason ? `Not acting on PR #${number}: ${reason}.` : `PR #${number} is current at ${headSha}.`);
    if (env.GITHUB_OUTPUT) appendFileSync(env.GITHUB_OUTPUT, `current=${!reason}\n`);
    return !reason;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    main(process.env);
}
