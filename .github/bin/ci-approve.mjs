import { execFileSync } from 'node:child_process';
import { appendFileSync, readFileSync } from 'node:fs';
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
        timeoutMs = CI_SETTLE_TIMEOUT_MS,
        pollMs = CI_SETTLE_POLL_MS,
        sleep = sleepAsync,
        now = () => performance.now(),
    } = {},
) {
    const deadline = now() + timeoutMs;
    let current = expected;
    for (;;) {
        const passed = await ci(current);
        current = await api.pr(number);
        if (!sameRevision(expected, current)) return { passed: false, changed: true };
        const remaining = deadline - now();
        if (passed || remaining <= 0) return { passed, changed: false, pr: current };
        await sleep(Math.min(pollMs, remaining));
    }
}

export async function reconcile(api, number, ci, { pushed = false, settle = false, settleOptions, branch } = {}) {
    // Without a current PR head, fail with the original API error before attempting writes.
    let pr = await api.pr(number);
    if (branch && pr.head.ref !== branch) {
        return { reason: 'PR head branch changed; waiting for another event with its branch lock.' };
    }
    // Only the marked reviews from this App are ever candidates for dismissal.
    let standing = [];
    async function withdraw(reason, all = true) {
        for (const review of standing) {
            if (all || review.commit_id !== pr.head.sha) await api.dismiss(number, review.id, reason);
        }
        standing = all ? [] : standing.filter((review) => review.commit_id === pr.head.sha);
    }
    try {
        standing = (await api.reviews(number)).filter(ownsReview);
        let settled;
        if (settle && !pushed) {
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
        if (pushed) {
            // A delayed push event must preserve a newer approval and its successful status.
            return { reason: 'Removed old-commit approvals; CI completion handles approval.' };
        }
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
            const review = await submitApproval(
                api,
                number,
                pr.head.sha,
                `${MARKER}\nApproved after lint, build, and selected tests passed for ${pr.head.sha}.`,
            );
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
                result = await evaluate(api, after, ci);
                if (!result.approve) await withdraw(result.reason);
            }
        }
        if (pr.head.repo?.full_name === api.repo) await api.status(pr.head.sha, result.state, result.reason);
        return result;
    } catch (error) {
        // Best-effort cleanup: status/review writes can also fail during an API outage.
        try {
            if (pr.head.repo?.full_name === api.repo) {
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

// Delays before each retry of a read. A CI-completion event is the only trigger that re-checks a
// finished head, so one transient 5xx on a read would otherwise leave its PR pending until a manual
// dispatch.
export const READ_RETRY_DELAYS_MS = [2000, 5000];

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
    const request = (endpoint, { method = 'GET', body, membership = false, review = false, pages = false } = {}) => {
        const token = membership ? env.GH_MEMBERS_TOKEN : review ? env.GH_REVIEW_TOKEN : env.GH_TOKEN;
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
    if (event.pull_request) return [event.pull_request.number];
    if (eventName === 'workflow_dispatch' && event.inputs?.pr_number) {
        const number = Number(event.inputs.pr_number);
        if (!Number.isSafeInteger(number) || number <= 0) throw new Error('Invalid PR number');
        return [number];
    }
    if (event.workflow_run) {
        const run = event.workflow_run;
        if (run.head_repository?.full_name !== api.repo) return [];
        // Query current PRs: event.pull_requests can be empty and events may arrive out of order.
        return (await api.open(run.head_branch))
            .filter((pr) => pr.head.repo?.full_name === api.repo && pr.head.ref === run.head_branch)
            .map((pr) => pr.number);
    }
    return (await api.open(branch))
        .filter((pr) => pr.head.repo?.full_name === api.repo && supportedBase(pr.base.ref))
        .map((pr) => pr.number);
}

export async function targetBranches(api, event, eventName) {
    if (event.pull_request) return [event.pull_request.head.ref];
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

export function verifyPrCi(api, pr, workflows) {
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
    );
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
    const ci = (pr) => verifyPrCi(api, pr, Object.keys(policy));
    const errors = [];
    for (const number of await targets(api, event, env.GITHUB_EVENT_NAME, env.CI_APPROVAL_BRANCH)) {
        try {
            const result = await reconcile(api, number, ci, {
                pushed: env.GITHUB_EVENT_NAME === 'pull_request_target' && event.action === 'synchronize',
                settle: env.GITHUB_EVENT_NAME === 'workflow_run',
                branch: env.CI_APPROVAL_BRANCH,
            });
            const summary = `PR #${number}: ${result.reason}`;
            console.log(summary);
            if (env.GITHUB_STEP_SUMMARY) appendFileSync(env.GITHUB_STEP_SUMMARY, `${summary}\n\n`);
        } catch (error) {
            errors.push(new Error(`PR #${number}: ${error.message}`, { cause: error }));
        }
    }
    if (errors.length) throw new AggregateError(errors, 'Approval reconciliation failed');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    await main(process.env, { discover: process.argv.includes('--discover') });
}
