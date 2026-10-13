import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { dispatchedRevisionReason, main } from './automerge-dispatch.mjs';
import { CONTEXT } from './ci-approve.mjs';

const head = 'a'.repeat(40);
const base = 'b'.repeat(40);
const repo = 'vertesia/llumiverse';
const expected = { repo, headBranch: 'renovate/main-node-dependencies', headSha: head, baseSha: base };
const pr = {
    state: 'open',
    head: { ref: expected.headBranch, sha: head, repo: { full_name: repo } },
    base: { sha: base },
};
const gate = (state, login = 'github-actions[bot]') => ({ context: CONTEXT, state, creator: { login } });
const other = { context: 'Vercel', state: 'success', creator: { login: 'vercel[bot]' } };

test('acts on the exact revision whose latest gate status succeeded', () => {
    assert.equal(dispatchedRevisionReason(pr, [other, gate('success'), gate('pending')], expected), undefined);
});

test('a closed PR, a foreign head, or a moved head, branch or base stops the merge', () => {
    for (const [changed, reason] of [
        [{ ...pr, state: 'closed' }, /no longer open/],
        [{ ...pr, head: { ...pr.head, repo: { full_name: 'fork/llumiverse' } } }, /not in this repository/],
        [{ ...pr, head: { ...pr.head, ref: 'renovate/other' } }, /head branch/],
        [{ ...pr, head: { ...pr.head, sha: 'c'.repeat(40) } }, /head moved/],
        [{ ...pr, base: { sha: 'c'.repeat(40) } }, /base moved/],
    ]) {
        assert.match(dispatchedRevisionReason(changed, [gate('success')], expected), reason);
    }
});

test('only the latest Actions-published gate status can authorize the merge', () => {
    for (const statuses of [
        [],
        [other],
        [gate('pending'), gate('success')],
        [gate('error'), gate('success')],
        [gate('success', 'someone')],
    ]) {
        assert.match(dispatchedRevisionReason(pr, statuses, expected), /status is not successful/);
    }
});

test('main reports whether the dispatched revision is current', () => {
    const dir = mkdtempSync(join(tmpdir(), 'automerge-dispatch-'));
    const output = join(dir, 'output');
    const env = {
        PR_NUMBER: '12',
        REPO: repo,
        HEAD_BRANCH: expected.headBranch,
        HEAD_SHA: head,
        BASE_SHA: base,
        GITHUB_OUTPUT: output,
    };
    const endpoints = [];
    const api = (statuses) => (endpoint, options) => {
        endpoints.push([endpoint, options]);
        return endpoint.includes('/statuses') ? [statuses] : pr;
    };
    try {
        assert.equal(main(env, api([gate('success')])), true);
        assert.equal(main(env, api([gate('pending')])), false);
        assert.equal(readFileSync(output, 'utf8'), 'current=true\ncurrent=false\n');
        assert.deepEqual(endpoints[0], [`repos/${repo}/pulls/12`, undefined]);
        assert.deepEqual(endpoints[1], [`repos/${repo}/commits/${head}/statuses?per_page=100`, { pages: true }]);
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

test('malformed dispatch inputs are rejected before any API call', () => {
    const env = { PR_NUMBER: '12', REPO: repo, HEAD_BRANCH: 'renovate/x', HEAD_SHA: head, BASE_SHA: base };
    const api = () => assert.fail('must not call the API');
    for (const changed of [
        { PR_NUMBER: '0' },
        { PR_NUMBER: '12abc' },
        { REPO: 'vertesia/llumiverse/../x' },
        { HEAD_BRANCH: '' },
        { HEAD_SHA: '../../../user' },
        { BASE_SHA: 'main' },
    ]) {
        assert.throws(() => main({ ...env, ...changed }, api), /Invalid PR number|Missing or malformed/);
    }
});
