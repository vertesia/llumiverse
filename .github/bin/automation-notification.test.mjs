import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
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

// The cherry-pick of merged PR #7313, from its branch fix-foo, opened the draft #7314.
const backportRun = {
    id: 50, path: '.github/workflows/backport.yaml', name: '(backport) Cherry-pick merged PR', event: 'pull_request',
    conclusion: 'success', head_sha: sha, head_branch: 'fix-foo', head_repository: { full_name: repo },
    run_started_at: '2026-11-19T10:00:00Z', html_url: 'https://github.com/run/50',
};
const merged = { number: 7313, merged_at: '2026-11-19T09:59:00Z', head: { sha, ref: 'fix-foo' } };
const conflicted = { ...pr, draft: true, created_at: '2026-11-19T10:01:00Z' };
const backportContext = { ...context, event: { workflow_run: backportRun } };

function backportApi({ drafts = { [branch]: [conflicted] }, refs = Object.keys(drafts), closed = [merged], author = user } = {}) {
    return async (path) => {
        const open = /^repos\/vertesia\/\w+\/pulls\?state=open&head=vertesia:([^&]+)&per_page=100$/.exec(path);
        if (open) return drafts[decodeURIComponent(open[1])] ?? [];
        if (/^repos\/vertesia\/\w+\/pulls\?state=closed&head=vertesia:fix-foo&per_page=100$/.test(path)) return closed;
        if (/^repos\/vertesia\/\w+\/git\/matching-refs\/heads\/backport-7313-to-$/.test(path)) {
            return refs.map((name) => ({ ref: `refs/heads/${name}` }));
        }
        if (/^repos\/vertesia\/\w+\/pulls\/7313$/.test(path)) return { user: author };
        throw new Error(`Unexpected API request: ${path}`);
    };
}

test('a cherry-pick with conflicts notifies the original author of every draft it opened', async () => {
    const release = 'backport-7313-to-release-1.6';
    const second = { ...conflicted, number: 7315, html_url: `https://github.com/${repo}/pull/7315`,
        head: { ...conflicted.head, ref: release }, base: { ref: 'release/1.6' } };
    const result = await notification(backportContext, backportApi({ drafts: { [branch]: [conflicted], [release]: [second] } }));
    assert.equal(result.login, 'ebarroca');
    assert.match(result.message, /studio#7314>, <[^|]+\|vertesia\/studio#7315>/);
    assert.match(result.message, /draft with conflicts/);
    assert.match(result.message, /run\/50\|Workflow run/);
});

test('a failed cherry-pick still announces the drafts it opened', async () => {
    const failed = { ...backportContext, event: { workflow_run: { ...backportRun, conclusion: 'failure' } } };
    assert.equal((await notification(failed, backportApi())).login, 'ebarroca');
});

test('composableai conflict draft resolves Leon from the source PR', async () => {
    const draft = { ...conflicted, head: { ...conflicted.head, repo: { full_name: 'vertesia/composableai' } } };
    const run = { ...backportRun, head_repository: { full_name: 'vertesia/composableai' } };
    const result = await notification({ ...context, repo: 'vertesia/composableai', event: { workflow_run: run } },
        backportApi({ drafts: { [branch]: [draft] }, author: { login: 'LeonRuggiero', type: 'User' } }));
    assert.equal(result.login, 'LeonRuggiero');
});

for (const [name, options] of [
    ['clean backport', { drafts: { [branch]: [{ ...conflicted, draft: false }] } }],
    ['draft opened by an earlier run', { drafts: { [branch]: [{ ...conflicted, created_at: '2026-11-19T09:00:00Z' }] } }],
    ['closed draft', { drafts: { [branch]: [] } }],
    ['draft of another author', { drafts: { [branch]: [{ ...conflicted, user }] } }],
    ['fork draft', { drafts: { [branch]: [{ ...conflicted, head: { ...conflicted.head, repo: { full_name: 'someone/studio' } } }] } }],
    ['unsupported target', { drafts: { [branch]: [{ ...conflicted, base: { ref: 'feature' } }] } }],
    ['branch that only shares the prefix', { refs: ['backport-7313-to-feature'], drafts: {} }],
    ['PR merged from an older head of the branch', { closed: [{ ...merged, head: { ...merged.head, sha: childSha } }] }],
    ['closed PR that was not merged', { closed: [{ ...merged, merged_at: null }] }],
]) {
    test(`a cherry-pick ignores a ${name}`, async () => {
        assert.equal(await notification(backportContext, backportApi(options)), null);
    });
}

for (const [name, change] of [
    ['skipped', { conclusion: 'skipped' }],
    ['cancelled', { conclusion: 'cancelled' }],
    ['fork', { head_repository: { full_name: 'someone/studio' } }],
]) {
    test(`a ${name} cherry-pick run reads nothing`, async () => {
        const event = { workflow_run: { ...backportRun, ...change } };
        assert.equal(await notification({ ...backportContext, event }, async (path) => assert.fail(path)), null);
    });
}

test('the notifier follows the cherry-pick workflow by name, on any branch', () => {
    const workflow = readFileSync(new URL('../workflows/notify-automation.yaml', import.meta.url), 'utf8');
    const backport = readFileSync(new URL('../workflows/backport.yaml', import.meta.url), 'utf8');
    const name = /^name: (.+)$/m.exec(backport)[1];
    assert.ok(workflow.includes(`"${name}"`));
    assert.ok(workflow.includes(`github.event.workflow_run.path == '${backportRun.path}'`));
    const triggers = workflow.split(/^on:$/m)[1].split(/^concurrency:/m)[0].replace(/^\s*#.*$/gm, '');
    assert.deepEqual(triggers.match(/^ {2}\w+:$/gm), ['  workflow_run:', '  workflow_call:']);
    assert.doesNotMatch(triggers, /branches:/);
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
    ['retired preview target', [{ ...pr, base: { ref: 'preview' } }]],
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

test('unresolvable bot author skips notification instead of dispatching to the bot', async () => {
    const inputs = { source_repository: 'vertesia/composableai', source_sha: sha, reason: 'Sync failed' };
    assert.equal(await notification({ ...context, inputs }, async (path) => path.includes('/pulls?') ? [] : { author: pr.user }), null);
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

for (const login of ['app/vertesia-release-bot', 'vertesia-release-bot[bot]']) {
    test(`backport CI and draft notifications accept ${login}`, async () => {
        const draft = { ...pr, draft: true, user: { login } };
        const baseApi = apiFor();
        const api = (path) => path.includes('/pulls?') ? [draft] : baseApi(path);
        assert.equal((await notification(context, api)).login, user.login);
        const drafts = { [branch]: [{ ...conflicted, user: { login } }] };
        assert.equal((await notification(backportContext, backportApi({ drafts }))).login, user.login);
    });
}

test('unknown app aliases cannot impersonate the backport bot', async () => {
    const baseApi = apiFor();
    assert.equal(await notification(context, (path) => path.includes('/pulls?')
        ? [{ ...pr, user: { login: 'app/another-bot' } }] : baseApi(path)), null);
});

test('sync app aliases resolve through source backports instead of notifying a bot', async () => {
    const syncBranch = 'sync-composableai-main-aaaaaaa';
    const syncPr = { ...pr, user: { login: 'app/vertesia-submodule-sync' },
        head: { ...pr.head, ref: syncBranch } };
    const result = await notification({ ...context, inputs: { branch: syncBranch, sha, reason: 'Sync failed' } }, async (path) => {
        if (path.startsWith(`repos/${repo}/pulls?`)) return [syncPr];
        if (path === `repos/${repo}/contents/composableai?ref=${sha}`) return { sha: childSha };
        if (path === `repos/vertesia/composableai/commits/${childSha}/pulls?per_page=100`) {
            return [{ ...pr, merged_at: 'today', merge_commit_sha: childSha, user: { login: 'app/vertesia-release-bot' } }];
        }
        if (path === 'repos/vertesia/composableai/pulls/7313') return { user };
        throw new Error(path);
    });
    assert.equal(result.login, user.login);
});

for (const [repository, upstream] of [
    ['vertesia/studio', 'composableai'],
    ['vertesia/composableai', 'llumiverse'],
]) {
    for (const author of [null, { login: 'github-actions[bot]', type: 'Bot' }, { login: 'app/release' }]) {
        test(`${repository} skips sync notifications without a human source author: ${author?.login}`, async () => {
            const sourceApi = async (path) => {
                if (path === `repos/vertesia/${upstream}/commits/${sha}/pulls?per_page=100`) return [];
                if (path === `repos/vertesia/${upstream}/commits/${sha}`) return { author };
                throw new Error(path);
            };
            const inputs = { source_repository: `vertesia/${upstream}`, source_sha: sha, reason: 'Sync failed' };
            assert.equal(await notification({ ...context, repo: repository, inputs }, sourceApi), null);

            const syncBranch = `sync-${upstream}-main-aaaaaaa`;
            const syncPr = { ...pr, user: { login: 'vertesia-submodule-sync[bot]', type: 'Bot' },
                head: { sha, ref: syncBranch, repo: { full_name: repository } } };
            const api = async (path) => {
                if (path.startsWith(`repos/${repository}/pulls?`)) return [syncPr];
                if (path === `repos/${repository}/contents/${upstream}?ref=${sha}`) return { sha };
                return sourceApi(path);
            };
            assert.equal(await notification({ ...context, repo: repository,
                inputs: { branch: syncBranch, sha, reason: 'Sync failed' } }, api), null);
        });
    }
}

test('sync notification still fails on source API errors', async () => {
    const inputs = { source_repository: 'vertesia/composableai', source_sha: sha, reason: 'Sync failed' };
    await assert.rejects(notification({ ...context, inputs }, async () => {
        throw new Error('GitHub API unavailable');
    }), /GitHub API unavailable/);
});

for (const repository of ['vertesia/studio', 'vertesia/composableai', 'vertesia/llumiverse']) {
    test(`retired preview pushes do not notify in ${repository}`, async () => {
        const sourceRun = { ...run, event: 'push', head_branch: 'preview' };
        assert.equal(await notification({ ...context, repo: repository, event: { workflow_run: sourceRun } },
            async () => ({ workflow_runs: [sourceRun] })), null);
    });
}
