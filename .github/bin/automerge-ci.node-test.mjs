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
const coordinatorPolicy = { ...policy, workflow: 'pr-validation.yaml' };
const coordinatorPath = '.github/workflows/pr-validation.yaml';
for (const conclusion of ['failure', 'cancelled', 'skipped', 'neutral', 'timed_out']) {
    test(`accepts required jobs when the unrelated coordinator half concluded ${conclusion}`, () => {
        assert.equal(
            verifyWorkflow(
                [run(1), run(2, { conclusion, path: coordinatorPath })],
                () => tested,
                coordinatorPolicy,
                context,
            ),
            true,
        );
    });
    test(`rejects a standalone run that concluded ${conclusion}`, () => {
        assert.equal(verifyWorkflow([run(2, { conclusion })], () => tested, coordinatorPolicy, context), false);
    });
}
test('does not accept required jobs from an active coordinator run', () => {
    assert.equal(
        verifyWorkflow(
            [run(1, { status: 'in_progress', conclusion: null, path: coordinatorPath })],
            () => tested,
            coordinatorPolicy,
            context,
        ),
        false,
    );
});
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
    const policies = JSON.parse(readFileSync(new URL('./automerge-ci-policy.json', import.meta.url), 'utf8'));
    const names = Object.keys(policies);
    const configuredJobs = Object.values(policies).flatMap((configured) =>
        configured.jobs.map((required) => ({
            ...job(required.example ?? required.name.slice(1, -1)),
            steps: (required.steps ?? []).map((name) => job(name)),
        })),
    );
    const pages = (endpoint) => {
        if (endpoint.includes('/workflows/')) {
            if (endpoint.includes('/pr-validation.yaml/')) {
                return [{ workflow_runs: [] }, { workflow_runs: [run(1, { path: coordinatorPath })] }];
            }
            return [{ workflow_runs: [] }];
        }
        assert.match(endpoint, /filter=latest&per_page=100/);
        return [{ jobs: [] }, { jobs: configuredJobs }];
    };
    assert.equal(main({ REPO: 'owner/repo', HEAD_BRANCH: 'fix', HEAD_SHA: 'head' }, 12, names, pages), true);
});

test('memoizes the shared coordinator run and jobs across policies', () => {
    const policies = JSON.parse(readFileSync(new URL('./automerge-ci-policy.json', import.meta.url), 'utf8'));
    const names = Object.keys(policies);
    const calls = new Map();
    const pages = (endpoint) => {
        calls.set(endpoint, (calls.get(endpoint) ?? 0) + 1);
        if (endpoint.includes('/pr-validation.yaml/')) {
            return [{ workflow_runs: [run(1, { path: coordinatorPath })] }];
        }
        if (endpoint.includes('/workflows/')) return [{ workflow_runs: [] }];
        if (endpoint.includes('/runs/1/')) {
            return [{
                jobs: Object.values(policies).flatMap((configured) =>
                    configured.jobs.map((required) => ({
                        ...job(required.example ?? required.name.slice(1, -1)),
                        steps: required.steps.map((name) => job(name)),
                    })),
                ),
            }];
        }
        throw new Error(`Unexpected endpoint: ${endpoint}`);
    };
    assert.equal(main({ REPO: 'owner/repo', HEAD_BRANCH: 'fix', HEAD_SHA: 'head' }, 12, names, pages), true);
    assert.equal([...calls].find(([endpoint]) => endpoint.includes('/pr-validation.yaml/'))?.[1], 1);
    assert.equal([...calls].find(([endpoint]) => endpoint.includes('/runs/1/jobs'))?.[1], 1);
});

test('legacy PR workflows remain valid while release branches adopt the coordinator', () => {
    const policies = JSON.parse(readFileSync(new URL('./automerge-ci-policy.json', import.meta.url), 'utf8'));
    for (const [name, configured] of Object.entries(policies)) {
        const pages = (endpoint) => {
            if (endpoint.includes('/pr-validation.yaml/')) return [{ workflow_runs: [] }];
            if (endpoint.includes(`/workflows/${name}/`)) return [{ workflow_runs: [run(1)] }];
            if (endpoint.includes('/runs/1/')) {
                return [{
                    jobs: configured.jobs.map((required) => ({
                        ...job(required.example ?? required.name.slice(1, -1)),
                        steps: required.steps.map((step) => job(step)),
                    })),
                }];
            }
            throw new Error(`Unexpected endpoint: ${endpoint}`);
        };
        assert.equal(main({ REPO: 'owner/repo', HEAD_BRANCH: 'fix', HEAD_SHA: 'head' }, 12, [name], pages), true);
    }
});
test('all automerge workflows use trusted policy and the shared gate', () => {
    const dir = new URL('../workflows/', import.meta.url);
    for (const file of readdirSync(dir).filter((name) => name.includes('automerge'))) {
        const yaml = readFileSync(new URL(file, dir), 'utf8');
        if (!yaml.includes('required_workflows=(')) continue;
        assert.match(yaml, /ref: \$\{\{ github.workflow_sha \}\}/, file);
        assert.match(yaml, /node \.github\/bin\/automerge-ci.mjs/, file);
        assert.match(yaml, /- "\(ci\) PR validation"/, file);
        assert.doesNotMatch(yaml, /sort_by\(\.id\) \| last/, file);
    }
});


test('every configured required step must actually run', () => {
    const policies = JSON.parse(readFileSync(new URL('./automerge-ci-policy.json', import.meta.url), 'utf8'));
    for (const configured of Object.values(policies)) {
        const jobs = configured.jobs.map((required) => ({
            ...job(required.example ?? required.name.slice(1, -1)),
            steps: required.steps.map((name) => job(name)),
        }));
        for (const required of jobs) {
            for (const step of required.steps) {
                step.conclusion = 'skipped';
                assert.equal(verifyWorkflow([run(1)], () => jobs, configured, context), false, step.name);
                step.conclusion = 'success';
            }
        }
    }
});

test('every rejection reports which check withheld approval', () => {
    const reasonFor = (runs, jobs, ctx = context) => {
        let reason;
        assert.equal(
            verifyWorkflow(runs, () => jobs, policy, ctx, (why) => {
                reason = why;
            }),
            false,
        );
        return reason;
    };
    assert.equal(reasonFor([], tested), 'no pull_request run for this head among 0 listed runs');
    assert.equal(
        reasonFor([run(3, { pull_requests: [] })], tested),
        'runs for this head are not associated with PR #12 (3: [])',
    );
    assert.equal(
        reasonFor([run(3, { pull_requests: [{ number: 12, base: { ref: 'main', sha: 'old' } }] })], tested, {
            ...context,
            baseBranch: 'main',
            baseSha: 'new',
        }),
        'run 3 tested base main@old, not main@new',
    );
    assert.equal(reasonFor([run(3, { status: 'in_progress' })], tested), 'run 3 is in_progress');
    assert.equal(reasonFor([run(3)], [job('Other')]), 'run 3 has no job matching ^Tests$');
    assert.equal(
        reasonFor([run(3)], [{ ...job('Tests', 'failure'), steps: [] }]),
        'run 3 job "Tests" is completed/failure',
    );
    assert.equal(
        reasonFor([run(3)], [{ ...job('Tests'), steps: [] }]),
        'run 3 job "Tests" did not pass step "Run tests"',
    );
    assert.equal(reasonFor([run(3)], noop), 'every run for this head is a no-op');
});
