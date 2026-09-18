import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import test from 'node:test';
import { main, verifyWorkflow } from './automerge-ci.mjs';

const context = { sha: 'head', branch: 'fix', pr: 12 };
const policy = { noOp: { gate: 'Gate', skipped: 'Plan' }, jobs: [{ name: '^Tests$', steps: ['Run tests'] }] };
const run = (id, overrides = {}) => ({
    id,
    head_sha: 'head',
    head_branch: 'fix',
    event: 'pull_request',
    pull_requests: [{ number: 12 }],
    status: 'completed',
    conclusion: 'success',
    ...overrides,
});
const job = (name, conclusion = 'success') => ({ name, status: 'completed', conclusion });
const tested = [{ ...job('Tests'), steps: [job('Run tests')] }];
const noop = [job('Gate'), job('Plan', 'skipped'), job('Tests', 'skipped')];

test('a successful label-only run cannot authorize a merge', () => {
    assert.equal(
        verifyWorkflow([run(2)], () => noop, policy, context),
        false,
    );
});
test('a newer no-op cannot hide a pending real build', () => {
    assert.equal(
        verifyWorkflow(
            [run(1, { status: 'in_progress', conclusion: null }), run(2)],
            (id) => (id === 2 ? noop : tested),
            policy,
            context,
        ),
        false,
    );
});
test('a completed real run remains usable after an unrelated label update', () => {
    assert.equal(
        verifyWorkflow([run(1), run(2)], (id) => (id === 2 ? noop : tested), policy, context),
        true,
    );
});
for (const conclusion of ['failure', 'cancelled', 'skipped', 'neutral', 'timed_out']) {
    test(`does not fall back past a newer ${conclusion} real run`, () => {
        assert.equal(
            verifyWorkflow([run(1), run(2, { conclusion })], () => tested, policy, context),
            false,
        );
    });
}
test('missing and skipped jobs or steps fail closed', () => {
    for (const jobs of [
        [],
        [job('Tests', 'skipped')],
        [job('Tests')],
        [{ ...job('Tests'), steps: [job('Run tests', 'skipped')] }],
    ]) {
        assert.equal(
            verifyWorkflow([run(1)], () => jobs, policy, context),
            false,
        );
    }
});
test('all matrix jobs must succeed', () => {
    assert.equal(
        verifyWorkflow([run(1)], () => [...tested, job('Tests', 'failure')], policy, context),
        false,
    );
});
test('ignores other commits, branches, events and PRs', () => {
    for (const overrides of [
        { head_sha: 'old' },
        { head_branch: 'other' },
        { event: 'push' },
        { pull_requests: [{ number: 13 }] },
    ]) {
        assert.equal(
            verifyWorkflow([run(1, overrides)], () => tested, policy, context),
            false,
        );
    }
});
test('does not mistake a partially executed or failed run for a no-op', () => {
    assert.equal(
        verifyWorkflow([run(1), run(2)], (id) => (id === 2 ? [...noop, job('Other')] : tested), policy, context),
        false,
    );
});
test('API errors block merging', () => {
    assert.throws(() =>
        verifyWorkflow(
            [run(1)],
            () => {
                throw new Error('API failed');
            },
            policy,
            context,
        ),
    );
});
test('reads all pages and validates every configured workflow', () => {
    const policies = JSON.parse(readFileSync(new URL('./automerge-ci-policy.json', import.meta.url)));
    const names = Object.keys(policies);
    let index = -1;
    const pages = (endpoint) => {
        if (endpoint.includes('/workflows/')) {
            index++;
            return [{ workflow_runs: [] }, { workflow_runs: [run(1)] }];
        }
        assert.match(endpoint, /filter=latest&per_page=100/);
        return [
            { jobs: [] },
            {
                jobs: policies[names[index]].jobs.map((required) => ({
                    ...job(required.example ?? required.name.slice(1, -1)),
                    steps: (required.steps ?? []).map((name) => job(name)),
                })),
            },
        ];
    };
    assert.equal(main({ REPO: 'owner/repo', HEAD_BRANCH: 'fix', HEAD_SHA: 'head' }, 12, names, pages), true);
});
test('all automerge workflows use trusted policy and the shared gate', () => {
    const dir = new URL('../workflows/', import.meta.url);
    for (const file of readdirSync(dir).filter((name) => name.includes('automerge'))) {
        const yaml = readFileSync(new URL(file, dir), 'utf8');
        if (!yaml.includes('required_workflows=(')) continue;
        assert.match(yaml, /ref: \$\{\{ github.sha \}\}/, file);
        assert.match(yaml, /node \.github\/bin\/automerge-ci.mjs/, file);
        assert.doesNotMatch(yaml, /sort_by\(\.id\) \| last/, file);
    }
});
