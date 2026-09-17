import assert from 'node:assert/strict';
import { test } from 'node:test';
import { notification } from './automation-notification.mjs';

const sha = 'a'.repeat(40);
const childSha = 'b'.repeat(40);
const repo = 'vertesia/studio';
const branch = 'backport-7313-to-main';
const user = { login: 'ebarroca', type: 'User' };
const pr = {
    number: 7314, html_url: `https://github.com/${repo}/pull/7314`, draft: false,
    head: { sha, ref: branch, repo: { full_name: repo } }, base: { ref: 'main' },
    user: { login: 'vertesia-release-bot[bot]', type: 'Bot' },
};
const run = {
    id: 42, workflow_id: 7, run_attempt: 1, event: 'pull_request', conclusion: 'failure',
    head_sha: sha, head_branch: branch, name: 'Build and test', html_url: 'https://github.com/run/42',
};
const context = { repo, eventName: 'workflow_run', event: { workflow_run: run }, runUrl: 'https://github.com/run/43' };

function apiFor(overrides = {}) {
    return async (path) => {
        if (path in overrides) return overrides[path];
        if (path.includes('/actions/workflows/')) return { workflow_runs: [run] };
        if (path.includes('/pulls?')) return [pr];
        if (path === `repos/${repo}/pulls/7313`) return { user };
        throw new Error(`Unexpected API request: ${path}`);
    };
}

test('failed backport CI reaches the original author with PR and failure links', async () => {
    const result = await notification(context, apiFor());
    assert.equal(result.login, 'ebarroca');
    assert.match(result.message, /studio#7314/);
    assert.match(result.message, /CI workflow Build and test concluded failure/);
    assert.match(result.message, /https:\/\/github.com\/run\/42/);
});

test('a conflicted draft notifies before labels or assignees are added', async () => {
    const draft = { ...pr, draft: true };
    const result = await notification({ ...context, eventName: 'pull_request_target', event: { pull_request: draft } },
        async (path) => path.includes('/pulls?') ? [draft] : { user });
    assert.equal(result.login, 'ebarroca');
    assert.match(result.message, /draft with conflicts/);
});

test('composableai conflict draft resolves Leon from the source PR', async () => {
    const draft = { ...pr, number: 2213, draft: true, head: { ...pr.head, ref: 'backport-2212-to-main', repo: { full_name: 'vertesia/composableai' } } };
    const result = await notification({ ...context, repo: 'vertesia/composableai', eventName: 'pull_request_target', event: { pull_request: draft } },
        async (path) => path.includes('/pulls?') ? [draft] : { user: { login: 'LeonRuggiero', type: 'User' } });
    assert.equal(result.login, 'LeonRuggiero');
});

for (const conclusion of ['success', 'skipped', 'neutral']) {
    test(`${conclusion} CI does not notify`, async () => {
        assert.equal(await notification({ ...context, event: { workflow_run: { ...run, conclusion } } }, apiFor()), null);
    });
}

for (const conclusion of ['timed_out', 'cancelled', 'action_required']) {
    test(`${conclusion} CI requires attention`, async () => {
        assert.equal((await notification({ ...context, event: { workflow_run: { ...run, conclusion } } }, apiFor())).login, user.login);
    });
}

for (const [name, replacement] of [
    ['closed PR', []],
    ['ambiguous PRs', [pr, pr]],
    ['new head', [{ ...pr, head: { ...pr.head, sha: childSha } }]],
    ['fork', [{ ...pr, head: { ...pr.head, repo: { full_name: 'someone/studio' } } }]],
    ['wrong author', [{ ...pr, user }]],
    ['unsupported target', [{ ...pr, base: { ref: 'feature' } }]],
]) {
    test(`ignore ${name}`, async () => {
        const baseApi = apiFor();
        assert.equal(await notification(context, (path) => path.includes('/pulls?') ? replacement : baseApi(path)), null);
    });
}

for (const latest of [{ ...run, id: 43 }, { ...run, run_attempt: 2 }]) {
    test(`ignore superseded run/attempt ${latest.id}/${latest.run_attempt}`, async () => {
        assert.equal(await notification(context, async () => ({ workflow_runs: [latest] })), null);
    });
}

test('an automerge failure links the automerge run and does not inspect CI again', async () => {
    const result = await notification({ ...context, inputs: { branch, sha, reason: 'Merge failed' } }, apiFor());
    assert.match(result.message, /Merge failed/);
    assert.match(result.message, /run\/43/);
});

test('a delayed draft event is suppressed after the PR is ready', async () => {
    assert.equal(await notification({ ...context, eventName: 'pull_request_target', event: { pull_request: { ...pr, draft: true } } }, apiFor()), null);
});

test('submodule sync resolves the source PR author using the actual gitlink, not the actor', async () => {
    const syncBranch = 'sync-composableai-main-aaaaaaa';
    const syncPr = { ...pr, user: { login: 'vertesia-submodule-sync[bot]', type: 'Bot' }, head: { ...pr.head, ref: syncBranch } };
    const result = await notification({ ...context, inputs: { branch: syncBranch, sha, reason: 'Sync failed' } }, async (path) => {
        if (path.startsWith(`repos/${repo}/pulls?`)) return [syncPr];
        if (path === `repos/${repo}/contents/composableai?ref=${sha}`) return { sha: childSha };
        if (path === `repos/vertesia/composableai/commits/${childSha}/pulls?per_page=100`) return [{ merged_at: 'today', merge_commit_sha: childSha, user }];
        throw new Error(path);
    });
    assert.equal(result.login, user.login);
    assert.match(result.message, /composableai sync/);
});

test('source author resolution follows a bot backport through a nested sync', async () => {
    const inputs = { source_repository: 'vertesia/composableai', source_sha: sha, reason: 'Preparation failed' };
    const result = await notification({ ...context, inputs }, async (path) => {
        if (path.includes(`/commits/${sha}/pulls?`)) return [{ ...pr, merged_at: 'today', merge_commit_sha: sha }];
        if (path === 'repos/vertesia/composableai/pulls/7313') return { user };
        throw new Error(path);
    });
    assert.equal(result.login, user.login);
});

test('studio sync follows composableai sync and llumiverse backport to a human', async () => {
    const syncBranch = 'sync-composableai-main-aaaaaaa';
    const syncBot = { login: 'vertesia-submodule-sync[bot]', type: 'Bot' };
    const grandchildSha = 'c'.repeat(40);
    const syncPr = { ...pr, user: syncBot, head: { ...pr.head, ref: syncBranch } };
    const result = await notification({ ...context, inputs: { branch: syncBranch, sha, reason: 'Sync failed' } }, async (path) => {
        if (path.startsWith(`repos/${repo}/pulls?`)) return [syncPr];
        if (path === `repos/${repo}/contents/composableai?ref=${sha}`) return { sha: childSha };
        if (path === `repos/vertesia/composableai/commits/${childSha}/pulls?per_page=100`) {
            return [{ ...pr, merged_at: 'today', merge_commit_sha: childSha, user: syncBot, head: { sha: childSha, ref: 'sync-llumiverse-main-bbbbbbb' } }];
        }
        if (path === `repos/vertesia/composableai/contents/llumiverse?ref=${childSha}`) return { sha: grandchildSha };
        if (path === `repos/vertesia/llumiverse/commits/${grandchildSha}/pulls?per_page=100`) {
            return [{ ...pr, merged_at: 'today', merge_commit_sha: grandchildSha }];
        }
        if (path === 'repos/vertesia/llumiverse/pulls/7313') return { user };
        throw new Error(path);
    });
    assert.equal(result.login, user.login);
});

test('early sync failure without PR falls back to the commit author', async () => {
    const inputs = { source_repository: 'vertesia/composableai', source_sha: sha, reason: 'Install failed' };
    const result = await notification({ ...context, inputs }, async (path) => path.includes('/pulls?') ? [] : { author: user });
    assert.equal(result.login, user.login);
    assert.match(result.message, /Install failed/);
});

test('unresolvable bot author fails visibly rather than dispatching to the bot', async () => {
    const inputs = { source_repository: 'vertesia/composableai', source_sha: sha, reason: 'Sync failed' };
    await assert.rejects(notification({ ...context, inputs }, async (path) => path.includes('/pulls?') ? [] : { author: pr.user }), /Cannot resolve/);
});

test('reject unexpected source repository', async () => {
    await assert.rejects(notification({ ...context, inputs: { source_repository: 'someone/repo', source_sha: sha, reason: 'failure' } }, apiFor()), /Invalid sync source/);
});

test('escape Slack mentions in workflow names', async () => {
    const result = await notification({ ...context, event: { workflow_run: { ...run, name: '<!channel> & tests' } } }, apiFor());
    assert.match(result.message, /&lt;!channel&gt; &amp; tests/);
});

test('failed source build or downstream dispatch reaches the source author', async () => {
    const sourceRun = { ...run, event: 'push', head_branch: 'main' };
    const result = await notification({ ...context, repo: 'vertesia/composableai', event: { workflow_run: sourceRun } }, async (path) => {
        if (path.includes('/actions/')) return { workflow_runs: [sourceRun] };
        if (path.endsWith('/commits/main')) return { sha };
        if (path.includes('/pulls?')) return [{ merged_at: 'today', merge_commit_sha: sha, user }];
        throw new Error(path);
    });
    assert.equal(result.login, user.login);
    assert.match(result.message, /CI needs your attention/);
});

test('superseded source cancellation does not notify', async () => {
    const sourceRun = { ...run, event: 'push', head_branch: 'main', conclusion: 'cancelled' };
    assert.equal(await notification({ ...context, repo: 'vertesia/llumiverse', event: { workflow_run: sourceRun } }, async (path) =>
        path.includes('/actions/') ? { workflow_runs: [sourceRun] } : { sha: childSha }), null);
});

for (const repository of ['vertesia/studio', 'vertesia/composableai', 'vertesia/llumiverse']) {
    for (const target of ['main', 'release/1.6', 'release/hotfix']) {
        test(`failed push on ${repository} ${target} notifies the merged PR author`, async () => {
            const sourceRun = { ...run, event: 'push', head_branch: target };
            const result = await notification({ ...context, repo: repository, event: { workflow_run: sourceRun } }, async (path) => {
                if (path.includes('/actions/')) return { workflow_runs: [sourceRun] };
                if (path.includes('/pulls?')) return [{ merged_at: 'today', merge_commit_sha: sha, user }];
                throw new Error(path);
            });
            assert.equal(result.login, user.login);
            assert.ok(result.message.includes(`branch ${target}`));
            assert.ok(result.message.includes(`/commit/${sha}`));
            assert.ok(result.message.includes(sourceRun.html_url));
            assert.ok(!result.message.includes('Automatic merging'));
        });
    }
}

test('direct push failure notifies its author even after a newer commit lands', async () => {
    const sourceRun = { ...run, event: 'push', head_branch: 'main' };
    const result = await notification({ ...context, event: { workflow_run: sourceRun } }, async (path) => {
        if (path.includes('/actions/')) return { workflow_runs: [sourceRun] };
        if (path.includes('/pulls?')) return [];
        if (path.endsWith(`/commits/${sha}`)) return { author: user };
        // No branch-tip query: the failure must not be dropped because main advanced.
        throw new Error(path);
    });
    assert.equal(result.login, user.login);
});

test('feature branch push failures are outside the notification scope', async () => {
    const sourceRun = { ...run, event: 'push', head_branch: 'feature/example' };
    assert.equal(await notification({ ...context, event: { workflow_run: sourceRun } }, async () => ({ workflow_runs: [sourceRun] })), null);
});
