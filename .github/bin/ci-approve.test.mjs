import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { verifyWorkflow } from './automerge-ci.mjs';
import {
    APP_LOGIN,
    approvalBody,
    approvedSha,
    CI_SETTLE_POLL_MS,
    CI_SETTLE_TIMEOUT_MS,
    CONTEXT,
    ENGINEERING_TEAM_SLUG,
    evaluate,
    githubApi,
    handOff,
    handoffFor,
    isAmbiguousApprovalError,
    isTransientApiError,
    MARKER,
    ownsReview,
    PR_EVENT_WORKFLOW,
    READ_RETRY_DELAYS_MS,
    reconcile,
    reconcileOptions,
    requiresHuman,
    submitApproval,
    targetBranches,
    targets,
    verifyPrCi,
    waitForCiToSettle,
} from './ci-approve.mjs';

const sha = 'a'.repeat(40);
const newer = 'b'.repeat(40);
const pr = {
    number: 12,
    state: 'open',
    draft: false,
    changed_files: 0,
    labels: [],
    user: { type: 'User', login: 'engineer' },
    base: { ref: 'main', sha: 'base' },
    head: { sha, ref: 'feature', repo: { full_name: 'vertesia/studio' } },
};
const approval = {
    id: 1,
    state: 'APPROVED',
    commit_id: sha,
    body: approvalBody(sha),
    user: { login: APP_LOGIN, type: 'Bot' },
};

function fixture({ pulls = [pr], closedPulls = [], reviews = [], files = [], engineeringMembers = ['engineer'] } = {}) {
    let reads = 0;
    let stored = [...reviews];
    const writes = [];
    return {
        repo: 'vertesia/studio',
        writes,
        pr: async () => ({
            ...structuredClone(pulls[Math.min(reads++, pulls.length - 1)]),
            changed_files: files.length,
        }),
        reviews: async () => structuredClone(stored),
        engineeringMember: async (login) => engineeringMembers.includes(login),
        files: async () => files,
        opened: [],
        async open(branch) {
            this.opened.push(branch);
            return pulls;
        },
        closedLookups: [],
        async closed(branch) {
            this.closedLookups.push(branch);
            return closedPulls;
        },
        status: async (...args) => writes.push(['status', ...args]),
        dismiss: async (_number, id) => {
            writes.push(['dismiss', id]);
            stored = stored.filter((review) => review.id !== id);
        },
        approve: async (number, commit, body) => {
            writes.push(['approve', number, commit, body]);
            const review = { ...approval, id: 100, commit_id: commit, body };
            stored.push(review);
            return review;
        },
    };
}

for (const base of ['main', 'release/1.6']) {
    test(`approves a tested PR onto ${base} with an explicit commit SHA`, async () => {
        const api = fixture({ pulls: [{ ...pr, base: { ...pr.base, ref: base } }] });
        assert.equal((await reconcile(api, 12, () => true)).approve, true);
        assert.deepEqual(api.writes[0].slice(0, 3), ['approve', 12, sha]);
        assert.equal(api.writes.at(-1)[2], 'success');
    });
}

test('an unchanged eligible approval is not submitted twice', async () => {
    const api = fixture({ reviews: [approval] });
    await reconcile(api, 12, () => true);
    assert.deepEqual(
        api.writes.map(([kind]) => kind),
        ['status'],
    );
});

test('push immediately dismisses only the gate review before inspecting CI', async () => {
    const unrelated = [
        { ...approval, id: 2, user: { type: 'User', login: 'reviewer' } },
        { ...approval, id: 3, body: 'Approved by the deployment gate' },
        { ...approval, id: 4, user: { type: 'Bot', login: 'another-app[bot]' } },
    ];
    const api = fixture({ pulls: [{ ...pr, head: { ...pr.head, sha: newer } }], reviews: [approval, ...unrelated] });
    await reconcile(api, 12, () => {
        assert.deepEqual(api.writes[0], ['dismiss', 1]);
        return false;
    });
    assert.deepEqual(
        api.writes.filter(([kind]) => kind === 'dismiss'),
        [['dismiss', 1]],
    );
    assert.equal(api.writes.at(-1)[2], 'pending');
});

test('failed or running CI on the same SHA withdraws its approval', async () => {
    const api = fixture({ reviews: [approval] });
    await reconcile(api, 12, () => false);
    assert.ok(api.writes.some(([kind, id]) => kind === 'dismiss' && id === 1));
    assert.ok(!api.writes.some(([kind]) => kind === 'approve'));
});

test('workflow completion waits for temporarily incomplete CI without publishing intermediate writes', async () => {
    const api = fixture();
    const verdicts = [false, false, true, true, true];
    const sleeps = [];
    const result = await reconcile(api, 12, () => verdicts.shift(), {
        settle: true,
        settleOptions: {
            timeoutMs: 20,
            pollMs: 10,
            now: () => sleeps.length * 10,
            sleep: async (ms) => sleeps.push(ms),
        },
    });
    assert.equal(result.approve, true);
    assert.deepEqual(sleeps, [10, 10]);
    assert.deepEqual(
        api.writes.map(([kind]) => kind),
        ['approve', 'status'],
    );
});

test('workflow completion publishes pending only after the CI settle deadline', async () => {
    const api = fixture({ reviews: [approval] });
    const sleeps = [];
    const result = await reconcile(api, 12, () => false, {
        settle: true,
        settleOptions: {
            timeoutMs: 20,
            pollMs: 10,
            now: () => sleeps.length * 10,
            sleep: async (ms) => sleeps.push(ms),
        },
    });
    assert.equal(result.approve, false);
    assert.deepEqual(sleeps, [10, 10]);
    assert.deepEqual(
        api.writes.map(([kind]) => kind),
        ['dismiss', 'status'],
    );
});

for (const [name, update] of [
    ['head', { head: { ...pr.head, sha: newer } }],
    ['base', { base: { ...pr.base, sha: 'new-base' } }],
]) {
    test(`workflow completion abandons reconciliation when the ${name} moves during settling`, async () => {
        const api = fixture({ pulls: [pr, { ...pr, ...update }], reviews: [approval] });
        const result = await reconcile(api, 12, () => false, {
            settle: true,
            settleOptions: { timeoutMs: 10, pollMs: 10, sleep: async () => {} },
        });
        assert.equal(result.approve, false);
        assert.match(result.reason, /changed while waiting/);
        assert.deepEqual(api.writes, []);
    });
}

const noWait = async () => assert.fail('waiting cannot change this verdict');

test('workflow completion for an older head checks CI once without waiting', async () => {
    const api = fixture({ reviews: [approval] });
    let checks = 0;
    const result = await reconcile(api, 12, () => ++checks > 1, {
        settle: true,
        settleOptions: { trigger: { id: 41, head_sha: newer }, sleep: noWait },
    });
    assert.equal(result.approve, false);
    assert.equal(checks, 1);
    assert.deepEqual(
        api.writes.map(([kind]) => kind),
        ['dismiss', 'status'],
    );
});

test('workflow completion stops waiting when CI is blocked by another unfinished run', async () => {
    let checks = 0;
    const result = await waitForCiToSettle(
        fixture(),
        12,
        pr,
        (_current, onReject) => {
            checks++;
            onReject({ reason: 'run 99 is in_progress', running: { id: 99, status: 'in_progress' } });
            return false;
        },
        { trigger: { id: 41, head_sha: sha }, sleep: noWait },
    );
    assert.equal(result.passed, false);
    assert.equal(checks, 1);
});

test('workflow completion keeps waiting while the triggering run itself reads as unfinished', async () => {
    const sleeps = [];
    let checks = 0;
    const result = await waitForCiToSettle(
        fixture(),
        12,
        pr,
        (_current, onReject) => {
            if (++checks > 2) return true;
            onReject({ reason: 'run 41 is in_progress', running: { id: 41, status: 'in_progress' } });
            return false;
        },
        {
            trigger: { id: 41, head_sha: sha },
            timeoutMs: 100,
            pollMs: 10,
            now: () => sleeps.length * 10,
            sleep: async (ms) => sleeps.push(ms),
        },
    );
    assert.equal(result.passed, true);
    assert.deepEqual(sleeps, [10, 10]);
});

const lintJob = 'Lint / Check codebase';
for (const [missing, partial] of [
    ['job', []],
    ['step', [{ name: lintJob, status: 'completed', conclusion: 'success', steps: [] }]],
]) {
    test(`workflow completion keeps waiting while the triggering run lacks a required ${missing}`, async () => {
        const complete = [
            {
                name: lintJob,
                status: 'completed',
                conclusion: 'success',
                steps: ['Test automerge CI policy', 'Check codebase'].map((name) => ({
                    name,
                    status: 'completed',
                    conclusion: 'success',
                })),
            },
        ];
        const validation = {
            id: 41,
            head_sha: sha,
            head_branch: 'feature',
            event: 'pull_request',
            status: 'completed',
            conclusion: 'success',
            path: '.github/workflows/pr-validation.yaml',
            pull_requests: [{ number: 12, base: { ref: 'main', sha: 'base' } }],
        };
        let jobReads = 0;
        const api = fixture();
        api.pages = (endpoint) => {
            if (endpoint.includes('/workflows/pr-validation.yaml/')) return [{ workflow_runs: [validation] }];
            if (endpoint.includes('/workflows/')) return [{ workflow_runs: [] }];
            return [{ jobs: jobReads++ === 0 ? partial : complete }];
        };
        const sleeps = [];
        const result = await waitForCiToSettle(
            api,
            12,
            pr,
            (current, onReject) => verifyPrCi(api, current, ['lint.yml'], onReject),
            {
                trigger: { id: 41, head_sha: sha },
                timeoutMs: 100,
                pollMs: 10,
                now: () => sleeps.length * 10,
                sleep: async (ms) => sleeps.push(ms),
            },
        );
        assert.equal(result.passed, true);
        assert.deepEqual(sleeps, [10]);
        assert.equal(jobReads, 2);
    });
}

test('a PR event signal evaluates the PR as it is now, and a CI completion waits for its data', () => {
    const env = { GITHUB_EVENT_NAME: 'workflow_run', CI_APPROVAL_BRANCH: 'feature' };
    const signal = { id: 40, path: PR_EVENT_WORKFLOW };
    const validation = { id: 41, path: '.github/workflows/pr-validation.yaml' };
    assert.deepEqual(reconcileOptions(env, { workflow_run: signal }), {
        settle: false,
        settleOptions: { trigger: signal },
        branch: 'feature',
    });
    assert.deepEqual(reconcileOptions(env, { workflow_run: validation }), {
        settle: true,
        settleOptions: { trigger: validation },
        branch: 'feature',
    });
    const manual = reconcileOptions({ GITHUB_EVENT_NAME: 'repository_dispatch' }, { client_payload: {} });
    assert.equal(manual.settle, false);
});

test('a PR event signal on a new head withdraws the old approval and waits for CI', async () => {
    const human = { ...approval, id: 2, user: { type: 'User', login: 'reviewer' } };
    const api = fixture({ pulls: [{ ...pr, head: { ...pr.head, sha: newer } }], reviews: [approval, human] });
    const result = await reconcile(
        api,
        12,
        () => false,
        reconcileOptions(
            { GITHUB_EVENT_NAME: 'workflow_run' },
            {
                workflow_run: { id: 40, path: PR_EVENT_WORKFLOW },
            },
        ),
    );
    assert.equal(result.approve, false);
    assert.deepEqual(
        api.writes.map(([kind, id]) => [kind, id]),
        [
            ['dismiss', 1],
            ['status', newer],
        ],
    );
    assert.equal(api.writes.at(-1)[2], 'pending');
});

test('CI settle defaults match the established three-minute automerge convergence window', () => {
    assert.equal(CI_SETTLE_TIMEOUT_MS, 180_000);
    assert.equal(CI_SETTLE_POLL_MS, 10_000);
    assert.equal(typeof waitForCiToSettle, 'function');
});

for (const [name, change] of [
    ['draft', { draft: true }],
    ['closed', { state: 'closed' }],
    ['retired branch', { base: { ref: 'preview', sha: 'base' } }],
    ['fork', { head: { ...pr.head, repo: { full_name: 'someone/studio' } } }],
    ['human label', { labels: [{ name: 'human-review-required' }] }],
    ['deployment gate', { labels: [{ name: 'deployment' }] }],
    ['bot author', { user: { type: 'Bot' } }],
]) {
    test(`${name} cannot retain or receive this gate's approval`, async () => {
        const api = fixture({ pulls: [{ ...pr, ...change }], reviews: [approval] });
        await reconcile(api, 12, () => true);
        assert.ok(api.writes.some(([kind]) => kind === 'dismiss'));
        assert.ok(!api.writes.some(([kind]) => kind === 'approve'));
    });
}

test('dependency, CI, and configuration changes receive approval after CI passes', async () => {
    for (const file of [
        { filename: '.github/workflows/lint.yaml' },
        { filename: 'package.json' },
        { filename: 'src/innocent.js', previous_filename: '.github/bin/automerge-ci.mjs' },
        { filename: 'packages/example/vitest.config.ts' },
        { filename: '.githooks/pre-commit' },
        { filename: 'scripts/build.mjs' },
        { filename: 'pnpm-workspace.yaml' },
        { filename: 'turbo.json' },
        { filename: 'biome.json' },
        { filename: 'packages/example/tsconfig.json' },
    ]) {
        const api = fixture({ files: [file] });
        const result = await reconcile(api, 12, () => true);
        assert.equal(result.approve, true);
        assert.equal(result.state, 'success');
        assert.ok(api.writes.some(([kind]) => kind === 'approve'));
    }
    assert.equal(requiresHuman(pr), false);
});

test('a tested PR from outside the engineering team cannot retain or receive approval', async () => {
    const api = fixture({ reviews: [approval], engineeringMembers: [] });
    const result = await reconcile(api, 12, () => true);
    assert.equal(result.approve, false);
    assert.match(result.reason, /not an active @vertesia\/engineering member/);
    assert.ok(api.writes.some(([kind, id]) => kind === 'dismiss' && id === approval.id));
    assert.ok(!api.writes.some(([kind]) => kind === 'approve'));
});

test('an engineering membership lookup failure fails closed and withdraws approval', async () => {
    const api = fixture({ reviews: [approval] });
    api.engineeringMember = async () => {
        throw new Error('membership API unavailable');
    };
    await assert.rejects(
        reconcile(api, 12, () => true),
        /membership API unavailable/,
    );
    assert.ok(api.writes.some(([kind, id]) => kind === 'dismiss' && id === approval.id));
    assert.ok(api.writes.some(([kind, _sha, state]) => kind === 'status' && state === 'error'));
});

for (const [name, update] of [
    ['head', { head: { ...pr.head, sha: newer } }],
    ['base', { base: { ...pr.base, sha: 'new-base' } }],
    ['draft', { draft: true }],
    ['opt-out label', { labels: [{ name: 'human-review-required' }] }],
]) {
    test(`${name} changing before publication prevents approval`, async () => {
        const api = fixture({ pulls: [pr, { ...pr, ...update }] });
        await reconcile(api, 12, () => true);
        assert.ok(!api.writes.some(([kind]) => kind === 'approve'));
    });
}

test('a rerun starting between evaluation and publication prevents approval', async () => {
    let calls = 0;
    const api = fixture();
    await reconcile(api, 12, () => ++calls === 1);
    assert.ok(!api.writes.some(([kind]) => kind === 'approve'));
});

test('a push during submission dismisses the review and leaves the new head pending', async () => {
    const api = fixture({ pulls: [pr, pr, { ...pr, head: { ...pr.head, sha: newer } }] });
    await reconcile(api, 12, () => true);
    assert.deepEqual(
        api.writes.map(([kind]) => kind),
        ['approve', 'dismiss', 'status'],
    );
    assert.deepEqual(api.writes.at(-1).slice(1, 3), [newer, 'pending']);
});

test('API failure withdraws the gate approval and fails the status', async () => {
    const api = fixture({ reviews: [approval] });
    await assert.rejects(
        reconcile(api, 12, () => {
            throw new Error('API unavailable');
        }),
        /API unavailable/,
    );
    assert.ok(api.writes.some(([kind, id]) => kind === 'dismiss' && id === 1));
    assert.ok(api.writes.some(([kind, _sha, state]) => kind === 'status' && state === 'error'));
});

test('review identity requires the exact App and marker at the start', () => {
    assert.equal(ownsReview(approval), true);
    for (const review of [
        { ...approval, body: `quoted ${MARKER}\n` },
        { ...approval, state: 'DISMISSED' },
        { ...approval, user: { type: 'User', login: APP_LOGIN } },
    ])
        assert.equal(Boolean(ownsReview(review)), false);
});

test('eligibility reads no Copilot reviews or review threads', async () => {
    const api = { repo: 'vertesia/studio', engineeringMember: async () => true };
    assert.equal((await evaluate(api, pr, () => true)).approve, true);
});

test('manual discovery excludes forks and unsupported bases', async () => {
    const api = fixture({
        pulls: [
            pr,
            { ...pr, number: 13, base: { ref: 'preview' } },
            { ...pr, number: 14, head: { ...pr.head, repo: { full_name: 'fork/studio' } } },
        ],
    });
    assert.deepEqual(await targets(api, {}, 'repository_dispatch'), [12]);
    assert.deepEqual(api.opened, [undefined]);
});

test('workflow events resolve current PRs even when the event has no PR list or an old SHA', async () => {
    const api = fixture();
    assert.deepEqual(
        await targets(
            api,
            {
                workflow_run: {
                    head_repository: { full_name: api.repo },
                    head_branch: 'feature',
                    head_sha: 'old',
                    pull_requests: [],
                },
            },
            'workflow_run',
        ),
        [12],
    );
    assert.deepEqual(api.opened, ['feature']);
    assert.deepEqual(
        await targets(
            api,
            {
                workflow_run: {
                    head_repository: { full_name: 'fork/studio' },
                    head_branch: 'feature',
                },
            },
            'workflow_run',
        ),
        [],
    );
});

test('a close signal targets the closed, unmerged PR at the signalled head only', async () => {
    const closed = { ...pr, state: 'closed', merged_at: null };
    const signal = (overrides = {}) => ({
        workflow_run: {
            head_repository: { full_name: fixture().repo },
            head_branch: 'feature',
            head_sha: sha,
            path: PR_EVENT_WORKFLOW,
            ...overrides,
        },
    });
    const run = async (closedPulls, event = signal()) => {
        const api = fixture({ pulls: [], closedPulls });
        return { numbers: await targets(api, event, 'workflow_run'), api };
    };
    assert.deepEqual((await run([closed])).numbers, [12]);
    assert.deepEqual((await run([{ ...closed, merged_at: '2026-11-24T00:00:00Z' }])).numbers, []);
    assert.deepEqual((await run([{ ...closed, head: { ...closed.head, sha: newer } }])).numbers, []);
    assert.deepEqual((await run([{ ...closed, head: { ...closed.head, ref: 'other' } }])).numbers, []);
    // A CI completion never re-evaluates a closed PR.
    const ci = await run([closed], signal({ path: '.github/workflows/test.yaml' }));
    assert.deepEqual(ci.numbers, []);
    assert.deepEqual(ci.api.closedLookups, []);
});

test('closing withdraws the approval, so a reopening at the same commit starts pending', async () => {
    const closed = { ...pr, state: 'closed', merged_at: null };
    const atClose = fixture({ pulls: [closed], reviews: [approval] });
    const options = reconcileOptions(
        { GITHUB_EVENT_NAME: 'workflow_run' },
        { workflow_run: { id: 40, path: PR_EVENT_WORKFLOW } },
    );
    assert.equal((await reconcile(atClose, 12, () => true, options)).approve, false);
    assert.deepEqual(
        atClose.writes.map(([kind, id]) => [kind, id]),
        [
            ['dismiss', 1],
            ['status', sha],
        ],
    );
    assert.equal(atClose.writes.at(-1)[2], 'pending');
    // Reopened at the same commit after a label made it a human review: nothing carries forward.
    const reopened = { ...pr, labels: [{ name: 'human-review-required' }] };
    const atReopen = fixture({ pulls: [reopened] });
    assert.equal((await reconcile(atReopen, 12, () => true, options)).approve, false);
    assert.ok(!atReopen.writes.some(([kind]) => kind === 'approve'));
});

test('a closed PR never overrides the status of an open PR at the same commit', async () => {
    const open = { ...pr, number: 13 };
    const closed = { ...pr, state: 'closed', merged_at: null };
    const byNumber = { 12: closed, 13: open };
    const reviewsOf = { 12: [approval], 13: [] };
    const writes = [];
    const api = {
        ...fixture(),
        writes,
        pr: async (number) => structuredClone(byNumber[number]),
        open: async () => [open],
        closed: async () => [closed],
        reviews: async (number) => structuredClone(reviewsOf[number]),
        status: async (...args) => writes.push(['status', ...args]),
        dismiss: async (number, id) => {
            writes.push(['dismiss', number, id]);
            reviewsOf[number] = reviewsOf[number].filter((review) => review.id !== id);
        },
        approve: async (number, commit, body) => {
            writes.push(['approve', number, commit]);
            const review = { ...approval, id: 100 + number, commit_id: commit, body };
            reviewsOf[number].push(review);
            return review;
        },
    };
    const event = {
        workflow_run: {
            head_repository: { full_name: api.repo },
            head_branch: 'feature',
            head_sha: sha,
            path: PR_EVENT_WORKFLOW,
        },
    };
    const numbers = await targets(api, event, 'workflow_run');
    assert.deepEqual(numbers, [13, 12]);
    const options = reconcileOptions({ GITHUB_EVENT_NAME: 'workflow_run' }, event);
    for (const number of numbers) await reconcile(api, number, () => true, options);
    // The closed PR loses its approval, and the open PR keeps the last word on the commit status.
    assert.ok(writes.some(([kind, number, id]) => kind === 'dismiss' && number === 12 && id === 1));
    assert.deepEqual(
        writes.filter(([kind]) => kind === 'status').map(([, commit, state]) => [commit, state]),
        [[sha, 'success']],
    );
    // Without an open PR at that commit, the closed PR resets it.
    const alone = fixture({ pulls: [closed], reviews: [approval] });
    alone.open = async () => [];
    await reconcile(alone, 12, () => true, options);
    assert.deepEqual(alone.writes.at(-1).slice(0, 3), ['status', sha, 'pending']);
});

test('a PR that closes during a run never overrides an open PR at the same commit', async () => {
    const closed = { ...pr, state: 'closed', merged_at: null };
    const sharing = { ...pr, number: 13 };
    const run = async (pulls, ci, options = {}, reviews = [], tamper = () => {}) => {
        const api = fixture({ pulls, reviews });
        api.open = async () => [sharing];
        tamper(api);
        let error;
        try {
            await reconcile(api, 12, ci, options);
        } catch (caught) {
            error = caught;
        }
        return { statuses: api.writes.filter(([kind]) => kind === 'status'), writes: api.writes, error };
    };
    const completion = reconcileOptions(
        { GITHUB_EVENT_NAME: 'workflow_run' },
        { workflow_run: { id: 41, head_sha: sha, path: '.github/workflows/test.yaml' } },
    );
    // Closed while the gate waited for CI data.
    const settling = await run([pr, closed], () => true, completion, [approval]);
    assert.deepEqual(settling.statuses, []);
    assert.ok(settling.writes.some(([kind, id]) => kind === 'dismiss' && id === 1));
    // Closed between the first verdict and the re-read before approving.
    const evaluating = await run([pr, closed], () => true);
    assert.deepEqual(evaluating.statuses, []);
    assert.ok(!evaluating.writes.some(([kind]) => kind === 'approve'));
    // Closed while the approval was being submitted: the new approval is withdrawn.
    const approving = await run([pr, pr, closed], () => true);
    assert.deepEqual(approving.statuses, []);
    assert.deepEqual(
        approving.writes.map(([kind]) => kind),
        ['approve', 'dismiss'],
    );
    // An error after the PR closed publishes no error status on the shared commit.
    const failing = await run(
        [pr, closed],
        () => true,
        {},
        [approval],
        (api) => {
            api.dismiss = async () => {
                throw new Error('dismissal failed');
            };
        },
    );
    assert.match(failing.error.message, /dismissal failed/);
    assert.deepEqual(failing.statuses, []);
    // Without an open PR at that commit, the PR that closed mid-run still resets it.
    const alone = fixture({ pulls: [pr, closed] });
    alone.open = async () => [];
    await reconcile(alone, 12, () => true);
    assert.deepEqual(alone.writes.at(-1).slice(0, 3), ['status', sha, 'pending']);
});

test('a failed review cleanup never publishes an error over an open PR at the same commit', async () => {
    const closed = { ...pr, state: 'closed', merged_at: null };
    const sharing = { ...pr, number: 13 };
    const run = async (pulls, reviews, sharers = [sharing], { reads = Infinity } = {}) => {
        const api = fixture({ pulls, reviews });
        api.open = async () => sharers;
        const read = api.pr;
        let count = 0;
        api.pr = async (number) => {
            if (++count > reads) throw new Error('gh: Server Error (HTTP 502)');
            return read(number);
        };
        api.dismiss = async () => {
            throw new Error('dismissal failed');
        };
        await assert.rejects(
            reconcile(api, 12, () => true),
            /dismissal failed/,
        );
        return api.writes.filter(([kind]) => kind === 'status').map(([, commit, state]) => [commit, state]);
    };
    // The fresh read sees the PR closed on a moved base; the dismissal of the approval fails.
    const movedBase = { ...closed, base: { ...closed.base, sha: 'moved' } };
    assert.deepEqual(await run([pr, movedBase], [approval]), []);
    // Same, with the error path's own read failing too: the run must already hold the fresh read.
    assert.deepEqual(await run([pr, movedBase], [approval], [sharing], { reads: 2 }), []);
    // Same, when the PR closes on a moved base while the new approval is being submitted.
    assert.deepEqual(await run([pr, pr, movedBase], [], [sharing], { reads: 3 }), []);
    // The first dismissal fails before the run reads the PR again, and the PR closed meanwhile.
    const olderApproval = { ...approval, body: approvalBody(newer) };
    assert.deepEqual(await run([pr, closed], [olderApproval]), []);
    // The PR closes during the failed dismissal, and the error path's read fails too: its last read
    // still shows it open, so the open PR at the same commit decides.
    assert.deepEqual(await run([pr, closed], [olderApproval], [sharing], { reads: 1 }), []);
    assert.deepEqual(await run([pr], [olderApproval], [], { reads: 1 }), [[sha, 'error']]);
    // An open PR, or a closed one that shares its commit with no open PR, still reports the error.
    assert.deepEqual(await run([pr, pr], [olderApproval]), [[sha, 'error']]);
    assert.deepEqual(await run([pr, closed], [olderApproval], []), [[sha, 'error']]);
});

test('manual reconciliation rejects malformed PR numbers', async () => {
    await assert.rejects(
        targets(fixture(), { client_payload: { pr_number: '-1' } }, 'repository_dispatch'),
        /Invalid PR/,
    );
});

test('membership transport treats only a 404 as a normal negative result', () => {
    const membershipApi = (error) => {
        const calls = [];
        const api = githubApi(
            {
                GITHUB_REPOSITORY: 'vertesia/llumiverse',
                GH_TOKEN: 'read',
                GH_REVIEW_TOKEN: 'app',
                GH_MEMBERS_TOKEN: 'members',
            },
            (_cmd, args) => {
                calls.push(args);
                throw error;
            },
            () => {},
        );
        return { api, calls };
    };

    const notFound = membershipApi(ghFailure('gh: Not Found (HTTP 404)'));
    assert.equal(notFound.api.engineeringMember('outsider'), false);
    assert.equal(notFound.calls.length, 1);

    const forbidden = membershipApi(ghFailure('gh: Forbidden (HTTP 403)'));
    assert.throws(
        () => forbidden.api.engineeringMember('engineer'),
        (error) => error.stderr === 'gh: Forbidden (HTTP 403)',
    );
    assert.equal(forbidden.calls.length, 1);

    const serverError = membershipApi(ghFailure('gh: Server Error (HTTP 502)'));
    assert.throws(
        () => serverError.api.engineeringMember('engineer'),
        (error) => error.stderr === 'gh: Server Error (HTTP 502)',
    );
    assert.equal(serverError.calls.length, READ_RETRY_DELAYS_MS.length + 1);
    for (const calls of [notFound.calls, forbidden.calls, serverError.calls]) {
        assert.ok(calls.every((args) => args[1].includes('/teams/engineering/memberships/')));
    }
});

test('read retries outlast the 5xx incidents seen in CI', () => {
    // Runs 42054194248 and 42061447529: 5xx responses kept coming for 28 s and 12 s after the first.
    const total = READ_RETRY_DELAYS_MS.reduce((sum, ms) => sum + ms, 0);
    assert.ok(total >= 30_000, `reads wait ${total} ms in all`);
    assert.ok(READ_RETRY_DELAYS_MS.every((ms, i) => i === 0 || ms >= READ_RETRY_DELAYS_MS[i - 1]));
});

test('API transport scopes approval and membership calls to dedicated App tokens', async () => {
    const calls = [];
    const api = githubApi(
        {
            GITHUB_REPOSITORY: 'vertesia/studio',
            GH_TOKEN: 'read',
            GH_REVIEW_TOKEN: 'app',
            GH_MEMBERS_TOKEN: 'members',
        },
        (_cmd, args, options) => {
            calls.push({ args, options });
            if (args.includes('--paginate')) return JSON.stringify([[approval], [{ ...approval, id: 2 }]]);
            if (args[1]?.includes('/memberships/')) return JSON.stringify({ state: 'active' });
            return '{}';
        },
    );
    assert.equal((await api.reviews(12)).length, 2);
    assert.ok(calls[0].args.includes('--paginate'));
    assert.equal(calls[0].options.env.GH_TOKEN, 'read');
    await api.approve(12, sha, 'review');
    assert.equal(calls[1].options.env.GH_TOKEN, 'app');
    assert.equal(JSON.parse(calls[1].options.input).commit_id, sha);
    assert.equal(await api.engineeringMember('engineer'), true);
    assert.equal(calls[2].options.env.GH_TOKEN, 'members');
    assert.match(calls[2].args[1], new RegExp(`/teams/${ENGINEERING_TEAM_SLUG}/memberships/engineer$`));
});

test('a missing engineering membership is an ineligible author, while other lookup failures surface', () => {
    const responses = [ghFailure('gh: Not Found (HTTP 404)'), ghFailure('gh: Forbidden (HTTP 403)')];
    const api = githubApi(
        { GITHUB_REPOSITORY: 'vertesia/studio', GH_TOKEN: 'read', GH_MEMBERS_TOKEN: 'members' },
        () => {
            throw responses.shift();
        },
    );
    assert.equal(api.engineeringMember('outsider'), false);
    assert.throws(() => api.engineeringMember('engineer'), /Command failed/);
});

test('a missing dedicated membership token fails closed without making a request', () => {
    const api = githubApi({ GITHUB_REPOSITORY: 'vertesia/studio', GH_TOKEN: 'read', GH_REVIEW_TOKEN: 'app' }, () => {
        assert.fail('membership request must not fall back to another token');
    });
    assert.throws(() => api.engineeringMember('engineer'), /Missing GitHub token/);
});

function ghFailure(stderr) {
    return Object.assign(new Error('Command failed: gh api'), { status: 1, stderr });
}

function internalApprovalError() {
    return Object.assign(ghFailure('gh: Unprocessable Entity (HTTP 422)'), {
        stdout: JSON.stringify({
            status: '422',
            errors: ['An internal error occurred, please try again.'],
        }),
    });
}

test('only internal or transport failures qualify for approval read-back', () => {
    assert.equal(isAmbiguousApprovalError(internalApprovalError()), true);
    assert.equal(isAmbiguousApprovalError(ghFailure('gh: Server Error (HTTP 502)')), true);
    assert.equal(isAmbiguousApprovalError(ghFailure('connection reset')), true);
    assert.equal(isAmbiguousApprovalError(ghFailure('gh: Forbidden (HTTP 403)')), false);
    assert.equal(isAmbiguousApprovalError(ghFailure('gh: Validation Failed (HTTP 422)')), false);
    const validation = Object.assign(ghFailure('gh: Unprocessable Entity (HTTP 422)'), {
        stdout: JSON.stringify({ status: '422', errors: ['Can not approve your own pull request'] }),
    });
    assert.equal(isAmbiguousApprovalError(validation), false);
});

for (const error of [
    internalApprovalError(),
    ghFailure('gh: Server Error (HTTP 502)'),
    ghFailure('connection reset'),
]) {
    test(`approval applied despite ${error.stderr} is recovered without another POST`, async () => {
        const api = fixture();
        const approve = api.approve;
        api.approve = async (...args) => {
            await approve(...args);
            throw error;
        };
        const result = await reconcile(api, 12, () => true);
        assert.equal(result.approve, true);
        assert.deepEqual(
            api.writes.map(([kind]) => kind),
            ['approve', 'status'],
        );
        assert.equal(api.writes.at(-1)[2], 'success');
    });
}

test('approval read-back waits for visibility without replaying the POST', async () => {
    const sleeps = [];
    let posts = 0;
    let reads = 0;
    const api = {
        approve: async () => {
            posts++;
            throw internalApprovalError();
        },
        reviews: async () => (++reads < 3 ? [] : [approval]),
    };
    assert.deepEqual(
        await submitApproval(api, 12, sha, approval.body, { sleep: async (ms) => sleeps.push(ms) }),
        approval,
    );
    assert.equal(posts, 1);
    assert.equal(reads, 3);
    assert.deepEqual(sleeps, READ_RETRY_DELAYS_MS.slice(0, 2));
});

test('unconfirmed approval fails after bounded reads and never retries its write', async () => {
    const error = internalApprovalError();
    let posts = 0;
    let reads = 0;
    const api = {
        approve: async () => {
            posts++;
            throw error;
        },
        reviews: async () => {
            reads++;
            return [];
        },
    };
    await assert.rejects(
        submitApproval(api, 12, sha, approval.body, { sleep: async () => {} }),
        (caught) => caught === error,
    );
    assert.equal(posts, 1);
    assert.equal(reads, READ_RETRY_DELAYS_MS.length + 1);
});

test('read-back cannot adopt another identity, commit, body, or review state', async () => {
    const error = internalApprovalError();
    const unrelated = [
        { ...approval, user: { login: APP_LOGIN, type: 'User' } },
        { ...approval, user: { login: 'other[bot]', type: 'Bot' } },
        { ...approval, commit_id: newer },
        { ...approval, body: `${MARKER}\nDifferent approval` },
        { ...approval, state: 'DISMISSED' },
    ];
    await assert.rejects(
        submitApproval(
            {
                approve: async () => {
                    throw error;
                },
                reviews: async () => unrelated,
            },
            12,
            sha,
            approval.body,
            { sleep: async () => {} },
        ),
        (caught) => caught === error,
    );
});

test('approval validation failures do not read back reviews', async () => {
    const error = ghFailure('gh: Validation Failed (HTTP 422)');
    await assert.rejects(
        submitApproval(
            {
                approve: async () => {
                    throw error;
                },
                reviews: async () => assert.fail('validation errors must not trigger recovery'),
            },
            12,
            sha,
            approval.body,
        ),
        (caught) => caught === error,
    );
});

test('a read-back failure publishes an error and keeps both API errors for diagnosis', async () => {
    const api = fixture();
    const writeError = internalApprovalError();
    const readError = new Error('review read unavailable');
    api.approve = async () => {
        api.reviews = async () => {
            throw readError;
        };
        throw writeError;
    };
    await assert.rejects(
        reconcile(api, 12, () => true),
        (error) => error instanceof AggregateError && error.errors[0] === writeError && error.errors[1] === readError,
    );
    assert.equal(api.writes.at(-1)[2], 'error');
});

test('a head move after recovering an ambiguous approval dismisses it', async () => {
    const api = fixture({ pulls: [pr, pr, { ...pr, head: { ...pr.head, sha: newer } }] });
    const approve = api.approve;
    api.approve = async (...args) => {
        await approve(...args);
        throw internalApprovalError();
    };
    const result = await reconcile(api, 12, () => true);
    assert.equal(result.approve, false);
    assert.deepEqual(
        api.writes.map(([kind]) => kind),
        ['approve', 'dismiss', 'status'],
    );
    assert.deepEqual(api.writes.at(-1).slice(1, 3), [newer, 'pending']);
});

test('CI restarting after recovered approval withdraws it through the normal recheck', async () => {
    const api = fixture();
    const approve = api.approve;
    let checks = 0;
    api.approve = async (...args) => {
        await approve(...args);
        throw internalApprovalError();
    };
    const result = await reconcile(api, 12, () => ++checks < 3);
    assert.equal(result.approve, false);
    assert.deepEqual(
        api.writes.map(([kind]) => kind),
        ['approve', 'dismiss', 'status'],
    );
    assert.equal(api.writes.at(-1)[2], 'pending');
});

function flakyApi(failures) {
    const calls = [];
    const sleeps = [];
    const api = githubApi(
        {
            GITHUB_REPOSITORY: 'vertesia/studio',
            GH_TOKEN: 'read',
            GH_REVIEW_TOKEN: 'app',
            GH_MEMBERS_TOKEN: 'members',
        },
        (_cmd, args) => {
            calls.push(args);
            if (calls.length <= failures.length) throw failures[calls.length - 1];
            return args.includes('--paginate') ? JSON.stringify([[pr]]) : JSON.stringify(approval);
        },
        (ms) => sleeps.push(ms),
    );
    return { api, calls, sleeps };
}

test('CI-completion discovery asks GitHub for the branch instead of listing every open PR', () => {
    const { api, calls } = flakyApi([]);
    api.open('feat/x#1');
    assert.equal(calls[0][1], 'repos/vertesia/studio/pulls?state=open&head=vertesia:feat%2Fx%231&per_page=100');
    api.open();
    assert.equal(calls[1][1], 'repos/vertesia/studio/pulls?state=open&per_page=100');
});

test('a transient read failure is retried with backoff', () => {
    const { api, calls, sleeps } = flakyApi([
        ghFailure('gh: Server Error (HTTP 502)'),
        ghFailure('error connecting to api.github.com'),
    ]);
    assert.deepEqual(api.open('feature'), [pr]);
    assert.equal(calls.length, 3);
    assert.deepEqual(sleeps, READ_RETRY_DELAYS_MS.slice(0, 2));
});

test('a read still failing after every retry surfaces the last error', () => {
    const failures = Array.from({ length: READ_RETRY_DELAYS_MS.length + 1 }, () =>
        ghFailure('gh: Server Error (HTTP 503)'),
    );
    const { api, calls } = flakyApi(failures);
    assert.throws(
        () => api.pr(12),
        (error) => error === failures.at(-1),
    );
    assert.equal(calls.length, failures.length);
});

test('client errors and writes are not retried', () => {
    const notFound = ghFailure('gh: Not Found (HTTP 404)');
    const reads = flakyApi([notFound]);
    assert.throws(
        () => reads.api.pr(12),
        (error) => error === notFound,
    );
    assert.equal(reads.calls.length, 1);

    const serverError = ghFailure('gh: Server Error (HTTP 502)');
    const writes = flakyApi([serverError]);
    assert.throws(
        () => writes.api.approve(12, sha, 'review'),
        (error) => error === serverError,
    );
    assert.equal(writes.calls.length, 1);
    assert.deepEqual(writes.sleeps, []);
});

test('only server errors and dropped connections count as transient', () => {
    assert.equal(isTransientApiError(ghFailure('gh: Bad Gateway (HTTP 502)')), true);
    assert.equal(isTransientApiError(ghFailure('Post "https://api.github.com/graphql": unexpected EOF')), true);
    assert.equal(isTransientApiError(ghFailure('gh: Validation Failed (HTTP 422)')), false);
    assert.equal(isTransientApiError(new Error('Missing GitHub token')), false);
});

test('workflow executes only trusted scripts and observes PR events and CI completion', () => {
    const workflow = readFileSync(new URL('../workflows/ci-approve.yaml', import.meta.url), 'utf8');
    const coordinator = readFileSync(new URL('../workflows/pr-validation.yaml', import.meta.url), 'utf8');
    const build = readFileSync(new URL('../workflows/build-and-test.yml', import.meta.url), 'utf8');
    const lint = readFileSync(new URL('../workflows/lint.yml', import.meta.url), 'utf8');
    assert.match(workflow, /ref: \$\{\{ github.workflow_sha \}\}/);
    assert.match(workflow, /types: \[completed\]/);
    assert.match(workflow, /permission-members: read/);
    assert.match(workflow, /GH_MEMBERS_TOKEN:/);
    assert.match(workflow, /continue-on-error: true/);
    assert.doesNotMatch(workflow, /schedule:|requested|in_progress/);
    assert.doesNotMatch(workflow, /^ {2}pull_request(_target)?:$/m);
    assert.match(workflow, /"\(ci\) PR event"/);
    assert.match(workflow, /github\.event\.workflow_run\.conclusion != 'skipped'/);
    assert.match(workflow, /permission-pull-requests: write/);
    assert.match(workflow, /Generate automerge hand-off token[\s\S]*permission-contents: write/);
    assert.doesNotMatch(workflow, /pull_request_review|checkout.*head|npm install/);
    assert.match(workflow, /cancel-in-progress: false/);
    assert.match(workflow, /queue: max/);
    assert.match(workflow, /matrix\.branch/);
    assert.match(workflow, /CI_APPROVAL_BRANCH:/);
    assert.match(workflow, /ci-approve\.mjs --discover/);
    assert.doesNotMatch(workflow, /github\.ref_name/);
    assert.match(workflow, /"\(ci\) PR validation"/);
    assert.match(coordinator, /uses: \.\/\.github\/workflows\/build-and-test\.yml/);
    assert.match(coordinator, /uses: \.\/\.github\/workflows\/lint\.yml/);
    assert.match(coordinator, /if: github\.event\.pull_request\.head\.repo\.full_name == github\.repository/);
    for (const reusable of [build, lint]) {
        assert.match(reusable, /^ {2}workflow_call:$/m);
        assert.doesNotMatch(reusable, /^ {2}pull_request:$/m);
    }
});

test('the PR event signal runs every change of a PR without secrets or code', () => {
    const file = readFileSync(new URL(`../../${PR_EVENT_WORKFLOW}`, import.meta.url), 'utf8');
    const signal = file.replace(/^\s*#.*$/gm, '');
    assert.match(signal, /^name: \(ci\) PR event$/m);
    const triggers = signal.split(/^on:$/m)[1].split(/^permissions:/m)[0];
    assert.deepEqual(triggers.match(/^ {2}\w+:$/gm), ['  pull_request:']);
    assert.doesNotMatch(triggers, /branches|paths/);
    const types = /^ {4}types: \[(.+)\]$/m.exec(triggers)[1].split(', ');
    assert.deepEqual(types.toSorted(), [
        'closed',
        'converted_to_draft',
        'edited',
        'labeled',
        'opened',
        'ready_for_review',
        'reopened',
        'synchronize',
        'unlabeled',
    ]);
    assert.match(signal, /^permissions: \{\}$/m);
    assert.doesNotMatch(signal, /secrets\.|github\.token|actions\/checkout|uses:/);
    assert.match(signal, /if: github\.event\.action != 'edited' \|\| github\.event\.changes\.base != null/);
});

test('the gate isolates repository dispatch permission in a dedicated App token', () => {
    const workflow = readFileSync(new URL('../workflows/ci-approve.yaml', import.meta.url), 'utf8');
    assert.match(workflow, /actions: read/);
    assert.match(workflow, /permission-contents: write/);
    assert.match(workflow, /GH_HANDOFF_TOKEN: \$\{\{ steps\.handoff-token\.outputs\.token \}\}/);
    assert.doesNotMatch(workflow, /^\s+contents: write/m);
});

test('additive ruleset requires CI without changing human review or thread rules', () => {
    const ruleset = JSON.parse(readFileSync(new URL('../rulesets/ci-approval.json', import.meta.url), 'utf8'));
    assert.equal(ruleset.rules.length, 1);
    assert.equal(ruleset.rules[0].type, 'required_status_checks');
    assert.equal(ruleset.rules[0].parameters.required_status_checks[0].context, CONTEXT);
    assert.deepEqual(ruleset.conditions.ref_name.include, ['refs/heads/main', 'refs/heads/release/**']);
    // Retired lines have no pr-event.yaml: requiring the gate there would leave their PRs without a signal.
    assert.deepEqual(ruleset.conditions.ref_name.exclude, ['refs/heads/release/1.3', 'refs/heads/release/1.4']);
});

test('approval does not depend on listing changed files', async () => {
    const api = fixture();
    api.pr = async () => ({ ...pr, changed_files: 3001 });
    api.files = async () => assert.fail('must not request changed files');
    assert.equal((await reconcile(api, 12, () => true)).approve, true);
    assert.ok(api.writes.some(([kind]) => kind === 'approve'));
});

test('failed dismissal still publishes a blocking error status', async () => {
    const api = fixture({ reviews: [approval] });
    api.dismiss = async () => {
        throw new Error('dismissal denied');
    };
    await assert.rejects(
        reconcile(api, 12, () => false),
        /dismissal denied/,
    );
    assert.ok(api.writes.some(([kind, _sha, state]) => kind === 'status' && state === 'error'));
});

test('failure after submission dismisses the newly created review', async () => {
    const api = fixture();
    let reads = 0;
    api.pr = async () => {
        if (++reads === 3) throw new Error('PR read failed');
        return pr;
    };
    await assert.rejects(
        reconcile(api, 12, () => true),
        /PR read failed/,
    );
    assert.ok(api.writes.some(([kind, id]) => kind === 'dismiss' && id === 100));
});

test('a delayed PR event preserves an approval already granted for the current head', async () => {
    const api = fixture({ reviews: [approval] });
    await reconcile(api, 12, () => true);
    assert.deepEqual(
        api.writes.map(([kind, commit, state]) => [kind, commit, state]),
        [['status', sha, 'success']],
    );
});

test('the tested commit comes from the review body, not the commit_id GitHub moves forward', () => {
    assert.equal(approvedSha(approval), sha);
    assert.equal(approvedSha({ ...approval, commit_id: newer }), sha);
    assert.equal(approvedSha({ ...approval, body: `${MARKER}\nApproved` }), undefined);
});

test('an approval whose commit_id moved to the new head is withdrawn on push', async () => {
    const moved = { ...approval, commit_id: newer };
    const api = fixture({ pulls: [{ ...pr, head: { ...pr.head, sha: newer } }], reviews: [moved] });
    await reconcile(api, 12, () => {
        assert.deepEqual(api.writes, [['dismiss', 1]]);
        return false;
    });
    assert.deepEqual(
        api.writes.filter(([kind]) => kind !== 'status'),
        [['dismiss', 1]],
    );
});

test('a moved approval is replaced by one for the tested head once CI passes', async () => {
    const moved = { ...approval, commit_id: newer };
    const api = fixture({ pulls: [{ ...pr, head: { ...pr.head, sha: newer } }], reviews: [moved] });
    assert.equal((await reconcile(api, 12, () => true)).approve, true);
    assert.deepEqual(
        api.writes.filter(([kind]) => kind !== 'status'),
        [
            ['dismiss', 1],
            ['approve', 12, newer, approvalBody(newer)],
        ],
    );
    assert.equal(api.writes.at(-1)[2], 'success');
});

test('an owned approval without a tested commit in its body is never treated as current', async () => {
    const unnamed = { ...approval, body: `${MARKER}\nApproved` };
    const api = fixture({ reviews: [unnamed] });
    await reconcile(api, 12, () => {
        assert.deepEqual(api.writes, [['dismiss', 1]]);
        return false;
    });
    assert.deepEqual(
        api.writes.filter(([kind]) => kind !== 'status'),
        [['dismiss', 1]],
    );
});

const ciPolicies = JSON.parse(readFileSync(new URL('./automerge-ci-policy.json', import.meta.url), 'utf8'));
for (const failure of ['verification', 'polling read']) {
    test(`settle ${failure} failures publish an error and withdraw owned approvals`, async () => {
        const api = fixture({ reviews: [approval] });
        const error = new Error('settle read failed');
        let reads = 0;
        const readPr = api.pr;
        api.pr = async () => {
            if (++reads > 1 && failure === 'polling read') throw error;
            return readPr();
        };
        await assert.rejects(
            reconcile(
                api,
                12,
                () => {
                    assert.deepEqual(api.writes, []);
                    if (failure === 'verification') throw error;
                    return false;
                },
                { settle: true },
            ),
            (caught) => caught === error,
        );
        assert.deepEqual(
            api.writes.map(([kind]) => kind),
            ['status', 'dismiss'],
        );
        assert.equal(api.writes[0][2], 'error');
    });
}

for (const passed of [false, true]) {
    test(`revision movement during terminal CI verdict ${passed} prevents every write`, async () => {
        const api = fixture({ reviews: [approval] });
        let moved = false;
        api.pr = async () => (moved ? { ...pr, base: { ...pr.base, sha: 'moved' } } : pr);
        const result = await reconcile(
            api,
            12,
            () => {
                moved = true;
                return passed;
            },
            { settle: true, settleOptions: { timeoutMs: 0 } },
        );
        assert.match(result.reason, /changed while waiting/);
        assert.deepEqual(api.writes, []);
    });
}

test('settle deadline includes verification and PR read time and caps the last sleep', async () => {
    let elapsed = 0;
    const sleeps = [];
    let checks = 0;
    const api = fixture();
    api.pr = async () => {
        elapsed += 3;
        return pr;
    };
    const result = await waitForCiToSettle(
        api,
        12,
        pr,
        () => {
            elapsed += 6;
            checks++;
            return false;
        },
        {
            timeoutMs: 20,
            pollMs: 10,
            now: () => elapsed,
            sleep: async (ms) => {
                sleeps.push(ms);
                elapsed += ms;
            },
        },
    );
    assert.equal(result.passed, false);
    assert.deepEqual(sleeps, [10]);
    assert.equal(checks, 2);
    assert.equal(elapsed, 28);
});

test('settle caps a sleep at the remaining deadline', async () => {
    let elapsed = 0;
    const sleeps = [];
    const result = await waitForCiToSettle(fixture(), 12, pr, () => false, {
        timeoutMs: 15,
        pollMs: 10,
        now: () => elapsed,
        sleep: async (ms) => {
            sleeps.push(ms);
            elapsed += ms;
        },
    });
    assert.equal(result.passed, false);
    assert.deepEqual(sleeps, [10, 5]);
});

test('manual reconciliation resolves a selected PR to its actual branch', async () => {
    assert.deepEqual(await targetBranches(fixture(), { client_payload: { pr_number: '12' } }, 'repository_dispatch'), [
        'feature',
    ]);
});

test('bulk manual reconciliation deduplicates branch jobs and scopes discovery to the locked branch', async () => {
    const api = fixture({ pulls: [pr, { ...pr, number: 13 }] });
    assert.deepEqual(await targetBranches(api, {}, 'repository_dispatch'), ['feature']);
    await targets(api, {}, 'repository_dispatch', 'feature');
    assert.equal(api.opened.at(-1), 'feature');
});

test('a branch job cannot write to a PR outside its lock', async () => {
    const api = fixture({ reviews: [approval] });
    await reconcile(api, 12, () => assert.fail('must not inspect CI'), { branch: 'other' });
    assert.deepEqual(api.writes, []);
});

const ciWorkflow = Object.keys(ciPolicies)[0];
function ciApi(runs) {
    return {
        repo: pr.head.repo.full_name,
        pages: (endpoint) =>
            endpoint.includes('/jobs?')
                ? [
                      {
                          jobs: ciPolicies[ciWorkflow].jobs.map((required) => ({
                              name: required.example ?? required.name.slice(1, -1),
                              status: 'completed',
                              conclusion: 'success',
                              steps: (required.steps ?? []).map((name) => ({
                                  name,
                                  status: 'completed',
                                  conclusion: 'success',
                              })),
                          })),
                      },
                  ]
                : [{ workflow_runs: runs }],
    };
}
const ciRun = {
    id: 1,
    head_sha: sha,
    head_branch: pr.head.ref,
    event: 'pull_request',
    status: 'completed',
    conclusion: 'success',
    pull_requests: [{ number: pr.number, base: pr.base }],
};

test('approval accepts successful CI for the current head and base', () => {
    assert.equal(verifyPrCi(ciApi([ciRun]), pr, [ciWorkflow]), true);
});

for (const [name, base] of [
    ['base advanced', { ...pr.base, sha: 'new-base' }],
    ['retargeted', { ...pr.base, ref: 'release/1.6' }],
]) {
    test(`approval rejects old CI after the PR is ${name}`, async () => {
        const changed = { ...pr, base };
        const api = fixture({ pulls: [changed], reviews: [approval] });
        await reconcile(api, pr.number, (current) => verifyPrCi(ciApi([ciRun]), current, [ciWorkflow]));
        assert.ok(api.writes.some(([kind, id]) => kind === 'dismiss' && id === approval.id));
        assert.ok(!api.writes.some(([kind]) => kind === 'approve'));
        assert.equal(api.writes.at(-1)[2], 'pending');
    });
}

test('missing base metadata and another PR base cannot authorize approval', () => {
    const run = { ...ciRun, pull_requests: [{ number: pr.number }, { number: 99, base: pr.base }] };
    assert.equal(verifyPrCi(ciApi([run]), pr, [ciWorkflow]), false);
    assert.throws(() => verifyPrCi(ciApi([ciRun]), { ...pr, base: {} }, [ciWorkflow]), /Missing PR base/);
});

test('a newer substantive run on another base prevents fallback to older CI', () => {
    const later = { ...ciRun, id: 2, pull_requests: [{ number: pr.number, base: { ...pr.base, sha: 'other' } }] };
    assert.equal(verifyPrCi(ciApi([ciRun, later]), pr, [ciWorkflow]), false);
});

test('lockfile changes retain approval after CI passes', async () => {
    for (const file of [
        { filename: 'pnpm-lock.yaml' },
        { filename: 'nested/pnpm-lock.yaml' },
        { filename: 'archived-lock.yaml', previous_filename: 'pnpm-lock.yaml' },
    ]) {
        const api = fixture({ files: [file], reviews: [approval] });
        const result = await reconcile(api, pr.number, () => true);
        assert.equal(result.approve, true);
        assert.equal(result.state, 'success');
        assert.ok(!api.writes.some(([kind]) => kind === 'dismiss'));
        assert.ok(!api.writes.some(([kind]) => kind === 'approve'));
    }
});

test('an initial PR read failure preserves the original error and performs no writes', async () => {
    const error = new Error('PR API unavailable');
    const api = fixture({ reviews: [approval] });
    api.pr = async () => {
        throw error;
    };
    await assert.rejects(
        reconcile(api, pr.number, () => assert.fail('must not inspect CI')),
        (caught) => caught === error,
    );
    assert.deepEqual(api.writes, []);
});

test('the required status ruleset blocks merging when the branch is behind its base', () => {
    const ruleset = JSON.parse(readFileSync(new URL('../rulesets/ci-approval.json', import.meta.url), 'utf8'));
    assert.equal(ruleset.rules[0].parameters.strict_required_status_checks_policy, true);
});

test('no-op runs with changed or missing base metadata cannot fall back to older CI', () => {
    const policy = { noOpJobs: ['Router'], noOp: { gate: 'Gate', skipped: 'Build' }, jobs: [{ name: '^Tests$' }] };
    const job = (name, conclusion) => ({ name, status: 'completed', conclusion });
    const context = { sha, branch: pr.head.ref, pr: pr.number, baseSha: pr.base.sha, baseBranch: pr.base.ref };
    const noOps = [[job('Gate', 'success'), job('Build', 'skipped')]];
    if (Object.values(ciPolicies).some((configured) => configured.noOpJobs)) noOps.push([job('Router', 'skipped')]);
    for (const jobs of noOps) {
        for (const base of [undefined, { ...pr.base, sha: 'older-base' }, { ...pr.base, ref: 'release/1.6' }]) {
            const latest = { ...ciRun, id: 2, pull_requests: [{ number: pr.number, base }] };
            assert.equal(
                verifyWorkflow([ciRun, latest], (id) => (id === 2 ? jobs : [job('Tests', 'success')]), policy, context),
                false,
            );
        }
        const sameBase = { ...ciRun, id: 2 };
        assert.equal(
            verifyWorkflow([ciRun, sameBase], (id) => (id === 2 ? jobs : [job('Tests', 'success')]), policy, context),
            true,
        );
    }
});

const handoffs = JSON.parse(readFileSync(new URL('./ci-approve-handoffs.json', import.meta.url), 'utf8'));
const renovate = { ...pr, head: { ...pr.head, ref: 'renovate/main-node-dependencies' } };

test('a successful gate hands the exact published revision to the specialized automerge workflow', async () => {
    const api = fixture({ pulls: [renovate] });
    const result = await reconcile(api, 12, () => true, { handoffs });
    assert.equal(api.writes.at(-1)[2], 'success');
    assert.deepEqual(result.handoff, {
        workflow: 'renovate-automerge.yaml',
        eventType: 'ci-approved-renovate',
        inputs: {
            pr_number: '12',
            head_branch: renovate.head.ref,
            head_sha: sha,
            base_ref: 'main',
            base_sha: 'base',
        },
    });
});

test('bot PRs routed to a specialized gate are handed off without this gate approving', async () => {
    const backport = {
        ...pr,
        user: { type: 'Bot', login: 'vertesia-release-bot[bot]' },
        head: { ...pr.head, ref: 'backport-800-to-release-1.5' },
    };
    const api = fixture({ pulls: [backport] });
    const result = await reconcile(api, 12, () => true, { handoffs });
    assert.equal(result.approve, false);
    assert.equal(result.handoff.workflow, 'backport-automerge.yaml');
    assert.ok(!api.writes.some(([kind]) => kind === 'approve'));
});

test('no hand-off without a successful status, for a draft, or for an unrouted branch', async () => {
    assert.equal((await reconcile(fixture({ pulls: [renovate] }), 12, () => false, { handoffs })).handoff, undefined);
    assert.equal((await reconcile(fixture(), 12, () => true, { handoffs })).handoff, undefined);
    assert.equal((await reconcile(fixture({ pulls: [renovate] }), 12, () => true)).handoff, undefined);
    const draft = { ...renovate, draft: true };
    assert.equal((await reconcile(fixture({ pulls: [draft] }), 12, () => true, { handoffs })).handoff, undefined);
});

test('a PR change during evaluation hands off nothing', async () => {
    const moved = { ...renovate, head: { ...renovate.head, sha: newer } };
    const result = await reconcile(fixture({ pulls: [renovate, moved] }), 12, () => true, { handoffs });
    assert.equal(result.state, 'pending');
    assert.equal(result.handoff, undefined);
});

test('a CI error publishes no success and hands off nothing', async () => {
    const api = fixture({ pulls: [renovate] });
    await assert.rejects(
        reconcile(
            api,
            12,
            () => {
                throw new Error('CI read failed');
            },
            { handoffs },
        ),
    );
    assert.equal(api.writes.at(-1)[2], 'error');
});

test('hand-off routing matches head-branch prefixes only', () => {
    assert.equal(handoffFor({ ...pr, head: { ...pr.head, ref: 'feat-renovate/x' } }, handoffs), undefined);
    assert.equal(handoffFor(renovate, {}), undefined);
});

test('the hand-off dispatches a default-branch event with the published revision', async () => {
    const calls = [];
    const handoff = handoffFor(renovate, handoffs);
    await handOff({ dispatch: async (...args) => calls.push(args) }, handoff);
    assert.deepEqual(calls, [['ci-approved-renovate', handoff.inputs]]);
});

test('dispatch transport uses the scoped hand-off token and cannot select a workflow ref', async () => {
    const calls = [];
    const api = githubApi(
        {
            GITHUB_REPOSITORY: 'vertesia/llumiverse',
            GH_TOKEN: 'actions',
            GH_REVIEW_TOKEN: 'review-app',
            GH_HANDOFF_TOKEN: 'handoff-app',
        },
        (_cmd, args, options) => {
            calls.push({ args, options });
            return '';
        },
    );
    const inputs = handoffFor(renovate, handoffs).inputs;
    assert.equal(await api.dispatch('ci-approved-renovate', inputs), null);
    assert.equal(calls[0].args[1], 'repos/vertesia/llumiverse/dispatches');
    assert.equal(calls[0].options.env.GH_TOKEN, 'handoff-app');
    assert.deepEqual(JSON.parse(calls[0].options.input), {
        event_type: 'ci-approved-renovate',
        client_payload: inputs,
    });
    assert.throws(() => api.dispatch('../workflow', inputs), /Invalid automerge hand-off event/);
});

test('every hand-off target runs from the default branch and verifies the revision before merging', () => {
    const eventTypes = {
        'renovate-automerge.yaml': 'ci-approved-renovate',
        'backport-automerge.yaml': 'ci-approved-backport',
        'release-automerge.yaml': 'ci-approved-release',
    };
    for (const [prefix, file] of Object.entries(handoffs)) {
        const workflow = readFileSync(new URL(`../workflows/${file}`, import.meta.url), 'utf8');
        for (const input of ['pr_number', 'head_branch', 'head_sha', 'base_ref', 'base_sha']) {
            assert.match(workflow, new RegExp(`github\\.event\\.client_payload\\.${input}`), `${file} ${input}`);
        }
        assert.match(workflow, new RegExp(`repository_dispatch:\\n    types: \\[${eventTypes[file]}\\]`), file);
        assert.doesNotMatch(workflow, /^\s+workflow_dispatch:/m, file);
        assert.ok(
            workflow.includes(
                `startsWith(github.event.workflow_run.head_branch || github.event.client_payload.head_branch, '${prefix}')`,
            ),
            file,
        );
        assert.match(workflow, /run: node \.github\/bin\/automerge-dispatch\.mjs/, file);
        assert.match(
            workflow,
            /if: github\.event_name != 'repository_dispatch' \|\| steps\.dispatched\.outputs\.current == 'true'/,
            file,
        );
        assert.match(workflow, /statuses: read/, file);
        const merge = workflow.split(/\n {6}- name: Approve and merge[^\n]*\n/)[1].split('\n')[0];
        assert.equal(
            merge,
            "        if: github.event_name == 'repository_dispatch' && steps.eligibility.outputs.should_merge == 'true'",
            file,
        );
    }
});

test('manual approval reconciliation runs only from the default branch', () => {
    const workflow = readFileSync(new URL('../workflows/ci-approve.yaml', import.meta.url), 'utf8');
    assert.match(workflow, /repository_dispatch:\n {4}types: \[ci-approve-reconcile\]/);
    assert.doesNotMatch(workflow, /^\s+workflow_dispatch:/m);
});
