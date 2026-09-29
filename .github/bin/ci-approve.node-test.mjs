import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { verifyWorkflow } from './automerge-ci.mjs';
import {
    APP_LOGIN,
    CONTEXT,
    evaluate,
    githubApi,
    MARKER,
    ownsReview,
    reconcile,
    requiresHuman,
    targets,
    verifyPrCi,
} from './ci-approve.mjs';

const sha = 'a'.repeat(40);
const newer = 'b'.repeat(40);
const pr = {
    number: 12,
    state: 'open',
    draft: false,
    changed_files: 0,
    labels: [],
    user: { type: 'User' },
    base: { ref: 'main', sha: 'base' },
    head: { sha, ref: 'feature', repo: { full_name: 'vertesia/studio' } },
};
const approval = {
    id: 1,
    state: 'APPROVED',
    commit_id: sha,
    body: `${MARKER}\nApproved`,
    user: { login: APP_LOGIN, type: 'Bot' },
};

function fixture({ pulls = [pr], reviews = [], files = [] } = {}) {
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
        files: async () => files,
        open: async () => pulls,
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
    const api = { repo: 'vertesia/studio', files: async () => [] };
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
    assert.deepEqual(await targets(api, {}, 'workflow_dispatch'), [12]);
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

test('manual dispatch rejects malformed PR numbers', async () => {
    await assert.rejects(targets(fixture(), { inputs: { pr_number: '-1' } }, 'workflow_dispatch'), /Invalid PR/);
});

test('API transport paginates reviews and scopes approval writes to the App token', async () => {
    const calls = [];
    const api = githubApi(
        { GITHUB_REPOSITORY: 'vertesia/studio', GH_TOKEN: 'read', GH_REVIEW_TOKEN: 'app' },
        (_cmd, args, options) => {
            calls.push({ args, options });
            return args.includes('--paginate') ? JSON.stringify([[approval], [{ ...approval, id: 2 }]]) : '{}';
        },
    );
    assert.equal((await api.reviews(12)).length, 2);
    assert.ok(calls[0].args.includes('--paginate'));
    assert.equal(calls[0].options.env.GH_TOKEN, 'read');
    await api.approve(12, sha, 'review');
    assert.equal(calls[1].options.env.GH_TOKEN, 'app');
    assert.equal(JSON.parse(calls[1].options.input).commit_id, sha);
});

test('workflow executes only trusted scripts and observes pushes and CI completion', () => {
    const workflow = readFileSync(new URL('../workflows/ci-approve.yaml', import.meta.url), 'utf8');
    assert.match(workflow, /ref: \$\{\{ github.workflow_sha \}\}/);
    assert.match(workflow, /types: \[completed\]/);
    assert.doesNotMatch(workflow, /schedule:|requested|in_progress/);
    assert.match(workflow, /synchronize/);
    assert.match(workflow, /converted_to_draft/);
    assert.match(workflow, /permission-pull-requests: write/);
    assert.doesNotMatch(workflow, /permission-contents: write|pull_request_review|checkout.*head|npm install/);
    assert.match(workflow, /cancel-in-progress: false/);
    assert.match(workflow, /queue: max/);
});

test('additive ruleset requires CI without changing human review or thread rules', () => {
    const ruleset = JSON.parse(readFileSync(new URL('../rulesets/ci-approval.json', import.meta.url), 'utf8'));
    assert.equal(ruleset.rules.length, 1);
    assert.equal(ruleset.rules[0].type, 'required_status_checks');
    assert.equal(ruleset.rules[0].parameters.required_status_checks[0].context, CONTEXT);
    assert.deepEqual(ruleset.conditions.ref_name.include, ['refs/heads/main', 'refs/heads/release/**']);
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

test('push withdraws only old approvals without reading CI or replacing the status', async () => {
    const human = { ...approval, id: 2, user: { type: 'User', login: 'reviewer' } };
    const api = fixture({
        pulls: [{ ...pr, head: { ...pr.head, sha: newer } }],
        reviews: [approval, human],
    });
    await reconcile(api, 12, () => assert.fail('push must not read CI'), { pushed: true });
    assert.deepEqual(api.writes, [['dismiss', 1]]);
});

test('a delayed push preserves an approval already granted for the current head', async () => {
    const api = fixture({ reviews: [approval] });
    await reconcile(api, 12, () => assert.fail('push must not read CI'), { pushed: true });
    assert.deepEqual(api.writes, []);
});

const ciPolicies = JSON.parse(readFileSync(new URL('./automerge-ci-policy.json', import.meta.url), 'utf8'));
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
