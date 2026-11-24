import { execFileSync } from 'node:child_process';
import { appendFileSync, existsSync, readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { main as verifyCi } from './automerge-ci.mjs';

export const CONTEXT = 'PR approval gate';
export const MARKER = '<!-- vertesia-ci-approval:v1 -->';
export const APP_LOGIN = 'vertesia-automerge[bot]';
export const ENGINEERING_TEAM_SLUG = 'engineering';
export const CI_SETTLE_TIMEOUT_MS = 180_000;
export const CI_SETTLE_POLL_MS = 10_000;

export function supportedBase(ref) {
    return ref === 'main' || /^release\/\d+\.\d+$/.test(ref);
}

export function ownsReview(review) {
    return (
        review.user?.type === 'Bot' &&
        review.user.login === APP_LOGIN &&
        review.body?.startsWith(`${MARKER}\n`) &&
        review.state === 'APPROVED'
    );
}

export function approvalBody(sha) {
    return `${MARKER}\nApproved after lint, build, and selected tests passed for ${sha}.`;
}

// GitHub moves an undismissed review's commit_id forward when the PR head changes, so commit_id can
// name a commit this gate never tested. The body this gate wrote is the only record of the tested
// commit; a review without one is never current.
export function approvedSha(review) {
    return /^Approved after lint, build, and selected tests passed for ([0-9a-f]{40})\.$/m.exec(review.body ?? '')?.[1];
}

export function requiresHuman(pr) {
    if (pr.labels.some(({ name }) => name === 'human-review-required')) return true;
    // Existing specialized gates retain ownership of deployment and automation PRs.
    return pr.labels.some(({ name }) => name === 'deployment') || pr.user?.type !== 'User' || !pr.user.login;
}

export async function evaluate(api, pr, ci) {
    if (pr.state !== 'open' || pr.draft || !supportedBase(pr.base.ref) || pr.head.repo?.full_name !== api.repo) {
        return {
            state: 'pending',
            approve: false,
            reason: 'PR must be ready, same-repository, and target main or release/X.Y.',
        };
    }
    if (!(await ci(pr))) {
        return {
            state: 'pending',
            approve: false,
            reason: 'Waiting for successful lint, build, and selected tests for this commit.',
        };
    }
    const specialized = requiresHuman(pr);
    const engineeringMember = !specialized && (await api.engineeringMember(pr.user.login));
    const human = specialized || !engineeringMember;
    return {
        state: 'success',
        approve: !human,
        reason: specialized
            ? 'CI passed; approval remains with a human or the specialized gate.'
            : engineeringMember
              ? 'Lint, build, and selected tests passed for this commit.'
              : 'CI passed; the PR author is not an active @vertesia/engineering member.',
    };
}

// Specialized automerge workflows merge only after this gate has published success, so the gate
// starts them instead of letting both race on the same CI completion. Head-branch prefixes map to
// repository-local workflows in ci-approve-handoffs.json; repositories without it hand off nothing.
export function handoffFor(pr, handoffs = {}) {
    const workflow = Object.entries(handoffs).find(([prefix]) => pr.head.ref.startsWith(prefix))?.[1];
    if (!workflow) return undefined;
    const eventType = {
        'renovate-automerge.yaml': 'ci-approved-renovate',
        'backport-automerge.yaml': 'ci-approved-backport',
        'release-automerge.yaml': 'ci-approved-release',
        'automerge-composableai.yaml': 'ci-approved-composableai',
        'automerge-llumiverse.yaml': 'ci-approved-llumiverse',
    }[workflow];
    if (!eventType) throw new Error(`Unsupported automerge hand-off workflow: ${workflow}`);
    return {
        workflow,
        eventType,
        inputs: {
            pr_number: String(pr.number),
            head_branch: pr.head.ref,
            head_sha: pr.head.sha,
            base_ref: pr.base.ref,
            base_sha: pr.base.sha,
        },
    };
}

function sameRevision(a, b) {
    return a.head.sha === b.head.sha && a.base.sha === b.base.sha && a.base.ref === b.base.ref;
}

const sleepAsync = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export async function waitForCiToSettle(
    api,
    number,
    expected,
    ci,
    {
        trigger,
        timeoutMs = CI_SETTLE_TIMEOUT_MS,
        pollMs = CI_SETTLE_POLL_MS,
        sleep = sleepAsync,
        now = () => performance.now(),
    } = {},
) {
    const deadline = now() + timeoutMs;
    let current = expected;
    for (;;) {
        // Waiting covers the triggering run's CI data lagging behind its completion event. It cannot
        // help when that run tested an older head, or when CI waits on another run that has not
        // finished: that run's own completion re-triggers the gate. Missing or failed data for the
        // triggering run keeps polling, because no later event is guaranteed to re-check it.
        let otherRunning = false;
        const passed = await ci(current, ({ running }) => {
            otherRunning = running !== undefined && running.id !== trigger?.id;
        });
        current = await api.pr(number);
        if (!sameRevision(expected, current)) return { passed: false, changed: true };
        const remaining = deadline - now();
        const futile = trigger !== undefined && (trigger.head_sha !== current.head.sha || otherRunning);
        if (passed || futile || remaining <= 0) return { passed, changed: false, pr: current };
        await sleep(Math.min(pollMs, remaining));
    }
}

export const CLOSED_REASON = 'The PR is closed; reopening it starts a new evaluation.';

// The commit status belongs to the commit, not the PR. An open PR always owns the status of its
// head; a closed PR owns it only when no open PR shares the commit, or it would overwrite that PR's
// verdict. The PR can close at any read of a run, not only the first one.
export async function ownsCommitStatus(api, pr) {
    if (pr.head.repo?.full_name !== api.repo) return false;
    if (pr.state === 'open') return true;
    return !(await api.open()).some(
        (open) => open.number !== pr.number && open.state === 'open' && open.head.sha === pr.head.sha,
    );
}

// A closed PR keeps no approval from this gate, so a reopening at the same commit starts from
// pending. This path never publishes an error status, which would also land on an open PR at the
// same commit.
export async function settleClosed(api, number, pr) {
    for (const review of (await api.reviews(number)).filter(ownsReview)) {
        await api.dismiss(number, review.id, CLOSED_REASON);
    }
    const owns = await ownsCommitStatus(api, pr);
    if (owns) await api.status(pr.head.sha, 'pending', CLOSED_REASON);
    return {
        state: 'pending',
        approve: false,
        reason: owns ? CLOSED_REASON : `${CLOSED_REASON} An open PR at the same commit owns its status.`,
    };
}

export async function reconcile(api, number, ci, { settle = false, settleOptions, branch, handoffs } = {}) {
    // Without a current PR head, fail with the original API error before attempting writes.
    let pr = await api.pr(number);
    if (branch && pr.head.ref !== branch) {
        return { reason: 'PR head branch changed; waiting for another event with its branch lock.' };
    }
    if (pr.state !== 'open') return settleClosed(api, number, pr);
    // Only the marked reviews from this App are ever candidates for dismissal.
    let standing = [];
    async function withdraw(reason, all = true) {
        for (const review of standing) {
            if (all || approvedSha(review) !== pr.head.sha) await api.dismiss(number, review.id, reason);
        }
        standing = all ? [] : standing.filter((review) => approvedSha(review) === pr.head.sha);
    }
    try {
        standing = (await api.reviews(number)).filter(ownsReview);
        let settled;
        if (settle) {
            settled = await waitForCiToSettle(api, number, pr, ci, settleOptions);
            if (settled.changed) {
                return {
                    state: 'pending',
                    approve: false,
                    reason: 'PR changed while waiting for CI data; waiting for another completion event.',
                };
            }
            pr = settled.pr;
        }
        // Withdraw old-head approvals before spending time inspecting CI.
        await withdraw('The PR head changed; waiting for checks on the new commit.', false);
        let firstEvaluation = true;
        const evaluateCi = (current) => {
            if (firstEvaluation && settled) {
                firstEvaluation = false;
                return settled.passed;
            }
            return ci(current);
        };
        let result = await evaluate(api, pr, evaluateCi);
        if (!result.approve) await withdraw(result.reason);
        if (result.approve) {
            const fresh = await api.pr(number);
            if (!sameRevision(pr, fresh)) {
                await withdraw('The PR changed during evaluation.');
                pr = fresh;
                result = {
                    state: 'pending',
                    approve: false,
                    reason: 'PR changed during evaluation; waiting for another check.',
                };
            } else {
                // Labels, draft state and rerun results can change without a new SHA.
                pr = fresh;
                result = await evaluate(api, pr, ci);
                if (!result.approve) await withdraw(result.reason);
            }
        }
        if (result.approve && standing.length === 0) {
            const review = await submitApproval(api, number, pr.head.sha, approvalBody(pr.head.sha));
            if (!ownsReview(review)) throw new Error('Unexpected approval identity or response');
            standing.push(review);
            if (review.commit_id !== pr.head.sha) throw new Error('Approval commit does not match the tested head');
            const after = await api.pr(number);
            if (!sameRevision(pr, after)) {
                await withdraw('The PR changed while the approval was being submitted.');
                pr = after;
                result = {
                    state: 'pending',
                    approve: false,
                    reason: 'PR changed during approval; waiting for another check.',
                };
            } else {
                // The latest read decides who owns the commit status: the PR may have closed.
                pr = after;
                result = await evaluate(api, pr, ci);
                if (!result.approve) await withdraw(result.reason);
            }
        }
        if (await ownsCommitStatus(api, pr)) await api.status(pr.head.sha, result.state, result.reason);
        // Hand off only once the successful status for this exact revision is published.
        const handoff = result.state === 'success' ? handoffFor(pr, handoffs) : undefined;
        return handoff ? { ...result, handoff } : result;
    } catch (error) {
        // Best-effort cleanup: status/review writes can also fail during an API outage.
        try {
            // A failed lookup counts as not owning it: an error must not land on another PR's verdict.
            if (await ownsCommitStatus(api, pr).catch(() => false)) {
                await api.status(
                    pr.head.sha,
                    'error',
                    'Could not verify approval eligibility; inspect the workflow log.',
                );
            }
        } finally {
            await withdraw('Approval eligibility could not be verified.');
        }
        throw error;
    }
}

// Delays before each retry of a read: exponential, 62 s in all. A CI-completion event is the only
// trigger that re-checks a finished head, so a read that fails during a GitHub 5xx incident would
// otherwise leave its PR pending until a manual dispatch. The old two retries (7 s) ran out in runs
// 42054194248 and 42061447529, where 5xx responses kept coming for 28 s and 12 s after the first.
export const READ_RETRY_DELAYS_MS = [2000, 4000, 8000, 16000, 32000];

export function isTransientApiError(error) {
    return /HTTP 5\d\d|error connecting to|connection reset|i\/o timeout|unexpected EOF/i.test(
        `${error?.stderr ?? ''}\n${error?.message ?? ''}`,
    );
}

export function isAmbiguousApprovalError(error) {
    if (isTransientApiError(error)) return true;
    // GitHub can create the review and still return this internal-error response.
    try {
        const response = JSON.parse(String(error?.stdout ?? ''));
        return (
            Number(response.status) === 422 &&
            Array.isArray(response.errors) &&
            response.errors.includes('An internal error occurred, please try again.')
        );
    } catch {
        return false;
    }
}

export async function submitApproval(api, number, sha, body, { sleep = sleepAsync } = {}) {
    try {
        return await api.approve(number, sha, body);
    } catch (error) {
        if (!isAmbiguousApprovalError(error)) throw error;
        // Never replay the POST: a failed response does not imply a failed write.
        for (let attempt = 0; ; attempt++) {
            let reviews;
            try {
                reviews = await api.reviews(number);
            } catch (readError) {
                throw new AggregateError([error, readError], 'Could not confirm approval after an ambiguous response');
            }
            const review = reviews.find(
                (candidate) => ownsReview(candidate) && candidate.commit_id === sha && candidate.body === body,
            );
            if (review) return review;
            if (attempt >= READ_RETRY_DELAYS_MS.length) throw error;
            await sleep(READ_RETRY_DELAYS_MS[attempt]);
        }
    }
}

function sleepSync(ms) {
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

export function githubApi(env, call = execFileSync, sleep = sleepSync) {
    const repo = env.GITHUB_REPOSITORY;
    if (!/^[\w.-]+\/[\w.-]+$/.test(repo ?? '')) throw new Error('Invalid GITHUB_REPOSITORY');
    const owner = repo.split('/')[0];
    const request = (
        endpoint,
        { method = 'GET', body, handoff = false, membership = false, review = false, pages = false } = {},
    ) => {
        const token = handoff
            ? env.GH_HANDOFF_TOKEN
            : membership
              ? env.GH_MEMBERS_TOKEN
              : review
                ? env.GH_REVIEW_TOKEN
                : env.GH_TOKEN;
        if (!token) throw new Error('Missing GitHub token');
        const args = ['api', endpoint, '--method', method];
        if (pages) args.push('--paginate', '--slurp');
        if (body) args.push('--input', '-');
        const send = () =>
            call('gh', args, {
                encoding: 'utf8',
                maxBuffer: 32 * 1024 * 1024,
                env: { ...env, GH_TOKEN: token },
                input: body ? JSON.stringify(body) : undefined,
            });
        // Only reads are retried: a write answered with a 5xx may still have been applied.
        const delays = method === 'GET' ? READ_RETRY_DELAYS_MS : [];
        for (let attempt = 0; ; attempt++) {
            try {
                const output = send();
                return output.trim() ? JSON.parse(output) : null;
            } catch (error) {
                if (attempt >= delays.length || !isTransientApiError(error)) throw error;
                console.warn(`Retrying ${method} ${endpoint} after a transient error: ${error.message}`);
                sleep(delays[attempt]);
            }
        }
    };
    const pages = (endpoint) => request(endpoint, { pages: true });
    const list = (endpoint) => pages(`${endpoint}${endpoint.includes('?') ? '&' : '?'}per_page=100`).flat();
    return {
        repo,
        pages,
        pr: (number) => request(`repos/${repo}/pulls/${number}`),
        // With a branch, GitHub filters server-side: one small page instead of every open PR in the repo.
        open: (branch) =>
            list(
                `repos/${repo}/pulls?state=open${
                    branch ? `&head=${repo.split('/')[0]}:${encodeURIComponent(branch)}` : ''
                }`,
            ),
        closed: (branch) =>
            list(`repos/${repo}/pulls?state=closed&head=${repo.split('/')[0]}:${encodeURIComponent(branch)}`),
        reviews: (number) => list(`repos/${repo}/pulls/${number}/reviews`),
        engineeringMember(login) {
            try {
                const membership = request(
                    `orgs/${owner}/teams/${ENGINEERING_TEAM_SLUG}/memberships/${encodeURIComponent(login)}`,
                    { membership: true },
                );
                return membership?.state === 'active';
            } catch (error) {
                // A missing membership is the expected negative response. Other failures are unsafe to ignore.
                if (/HTTP 404/i.test(`${error?.stderr ?? ''}\n${error?.message ?? ''}`)) return false;
                throw error;
            }
        },
        dismiss: (number, id, message) =>
            request(`repos/${repo}/pulls/${number}/reviews/${id}/dismissals`, {
                method: 'PUT',
                review: true,
                body: { message },
            }),
        dispatch(eventType, payload) {
            if (!/^ci-approved-[a-z0-9-]+$/.test(eventType)) throw new Error('Invalid automerge hand-off event');
            return request(`repos/${repo}/dispatches`, {
                method: 'POST',
                handoff: true,
                body: { event_type: eventType, client_payload: payload },
            });
        },
        approve: (number, sha, body) =>
            request(`repos/${repo}/pulls/${number}/reviews`, {
                method: 'POST',
                review: true,
                body: { event: 'APPROVE', commit_id: sha, body },
            }),
        async status(sha, state, description) {
            description = description.slice(0, 140);
            const old = list(`repos/${repo}/commits/${sha}/statuses`).find((status) => status.context === CONTEXT);
            if (old?.creator?.login === 'github-actions[bot]' && old.state === state && old.description === description)
                return;
            request(`repos/${repo}/statuses/${sha}`, {
                method: 'POST',
                body: {
                    state,
                    description,
                    context: CONTEXT,
                    target_url: `${env.GITHUB_SERVER_URL}/${repo}/actions/runs/${env.GITHUB_RUN_ID}`,
                },
            });
        },
    };
}

export async function targets(api, event, eventName, branch) {
    if (eventName === 'repository_dispatch' && event.client_payload?.pr_number) {
        const number = Number(event.client_payload.pr_number);
        if (!Number.isSafeInteger(number) || number <= 0) throw new Error('Invalid PR number');
        return [number];
    }
    if (event.workflow_run) {
        const run = event.workflow_run;
        if (run.head_repository?.full_name !== api.repo) return [];
        const sameBranch = (pr) => pr.head.repo?.full_name === api.repo && pr.head.ref === run.head_branch;
        // Query current PRs: event.pull_requests can be empty and events may arrive out of order.
        const numbers = (await api.open(run.head_branch)).filter(sameBranch).map((pr) => pr.number);
        if (run.path === PR_EVENT_WORKFLOW) {
            // A signal can come from closing a PR. Evaluating the closed PR withdraws the approval and
            // resets the status, so a reopening at the same commit starts pending instead of approved
            // until the gate runs. Only the PR at the signalled head counts: an older PR closed on the
            // same branch was settled when it closed. A merged PR cannot reopen.
            for (const pr of await api.closed(run.head_branch)) {
                if (sameBranch(pr) && !pr.merged_at && pr.head.sha === run.head_sha && !numbers.includes(pr.number)) {
                    numbers.push(pr.number);
                }
            }
        }
        return numbers;
    }
    return (await api.open(branch))
        .filter((pr) => pr.head.repo?.full_name === api.repo && supportedBase(pr.base.ref))
        .map((pr) => pr.number);
}

export async function targetBranches(api, event, eventName) {
    if (event.workflow_run) {
        return event.workflow_run.head_repository?.full_name === api.repo ? [event.workflow_run.head_branch] : [];
    }
    const branches = [];
    for (const number of await targets(api, event, eventName)) {
        const pr = await api.pr(number);
        if (pr.head.repo?.full_name === api.repo) branches.push(pr.head.ref);
    }
    return [...new Set(branches)];
}

export async function handOff(api, { workflow, eventType, inputs }) {
    await api.dispatch(eventType, inputs);
    console.log(`PR #${inputs.pr_number}: started ${workflow} for ${inputs.head_sha}.`);
}

export function verifyPrCi(api, pr, workflows, onReject) {
    if (!pr.base?.sha || !pr.base?.ref) throw new Error('Missing PR base revision');
    return verifyCi(
        {
            REPO: api.repo,
            HEAD_BRANCH: pr.head.ref,
            HEAD_SHA: pr.head.sha,
            BASE_BRANCH: pr.base.ref,
            BASE_SHA: pr.base.sha,
        },
        pr.number,
        workflows,
        api.pages,
        onReject,
    );
}

// pr-event.yaml runs on pull_request without secrets, so a PR change reaches the gate as the
// completion of that run. The completion carries no CI result to wait for, nor which change it was:
// the gate evaluates the PR as it is now, which also withdraws an approval for an older head.
export const PR_EVENT_WORKFLOW = '.github/workflows/pr-event.yaml';

export function reconcileOptions(env, event) {
    const ciCompletion = env.GITHUB_EVENT_NAME === 'workflow_run' && event.workflow_run.path !== PR_EVENT_WORKFLOW;
    return { settle: ciCompletion, settleOptions: { trigger: event.workflow_run }, branch: env.CI_APPROVAL_BRANCH };
}

export async function main(env, { discover = false } = {}) {
    const api = githubApi(env);
    const event = JSON.parse(readFileSync(env.GITHUB_EVENT_PATH, 'utf8'));
    if (discover) {
        const branches = await targetBranches(api, event, env.GITHUB_EVENT_NAME);
        appendFileSync(env.GITHUB_OUTPUT, `branches=${JSON.stringify(branches)}\n`);
        return;
    }
    const policy = JSON.parse(readFileSync(new URL('./automerge-ci-policy.json', import.meta.url), 'utf8'));
    const ci = (pr, onReject) => verifyPrCi(api, pr, Object.keys(policy), onReject);
    const handoffsUrl = new URL('./ci-approve-handoffs.json', import.meta.url);
    const handoffs = existsSync(handoffsUrl) ? JSON.parse(readFileSync(handoffsUrl, 'utf8')) : {};
    const errors = [];
    for (const number of await targets(api, event, env.GITHUB_EVENT_NAME, env.CI_APPROVAL_BRANCH)) {
        try {
            const result = await reconcile(api, number, ci, { ...reconcileOptions(env, event), handoffs });
            const summary = `PR #${number}: ${result.reason}`;
            console.log(summary);
            if (env.GITHUB_STEP_SUMMARY) appendFileSync(env.GITHUB_STEP_SUMMARY, `${summary}\n\n`);
            // A failed hand-off fails this job without touching the published status or approval.
            if (result.handoff) await handOff(api, result.handoff);
        } catch (error) {
            errors.push(new Error(`PR #${number}: ${error.message}`, { cause: error }));
        }
    }
    if (errors.length) throw new AggregateError(errors, 'Approval reconciliation failed');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    await main(process.env, { discover: process.argv.includes('--discover') });
}
