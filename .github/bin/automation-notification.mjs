// Kept identical in studio, composableai and llumiverse: public repos dispatch only
// GitHub identities; the private Studio relay owns the Slack mapping and token.
import { execFileSync } from 'node:child_process';
import { appendFileSync, readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const failureConclusions = new Set(['failure', 'timed_out', 'cancelled', 'action_required', 'startup_failure']);
const normalizeBotLogin = (login) => login?.startsWith('app/') ? `${login.slice(4)}[bot]` : login;
const human = (user) => user?.login && user.type !== 'Bot' && !normalizeBotLogin(user.login).endsWith('[bot]');
const escapeSlack = (value) => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');

function kindOf(repo, branch) {
    const backport = /^backport-(\d+)-to-(?:main|release-\d+\.\d+)$/.exec(branch);
    if (backport) return { kind: 'Backport', original: backport[1], bot: 'vertesia-release-bot[bot]' };
    const upstream = repo === 'vertesia/studio' ? 'composableai' : repo === 'vertesia/composableai' ? 'llumiverse' : '';
    if (upstream && branch.startsWith(`sync-${upstream}-`)) {
        return { kind: `${upstream} sync`, upstream, bot: 'vertesia-submodule-sync[bot]' };
    }
    return null;
}

async function prOwner(api, repo, pr, kind, depth = 0) {
    if (depth > 8) return null;
    if (kind.original) {
        const original = await api(`repos/${repo}/pulls/${kind.original}`);
        if (human(original.user)) return original.user.login;
        const originalKind = kindOf(repo, original.head.ref);
        return originalKind ? prOwner(api, repo, original, originalKind, depth + 1) : null;
    }
    const entry = await api(`repos/${repo}/contents/${kind.upstream}?ref=${pr.head.sha}`);
    return commitOwner(api, `vertesia/${kind.upstream}`, entry.sha, depth + 1);
}

async function commitOwner(api, repo, sha, depth = 0) {
    if (depth > 8) return null;
    // Prefer the merged PR author over the merge bot or the person who clicked Merge.
    const prs = await api(`repos/${repo}/commits/${encodeURIComponent(sha)}/pulls?per_page=100`);
    const source = prs.find((pr) => pr.merged_at && pr.merge_commit_sha === sha);
    if (source) {
        if (human(source.user)) return source.user.login;
        const kind = kindOf(repo, source.head.ref);
        if (kind) return prOwner(api, repo, source, kind, depth + 1);
    }
    const commit = await api(`repos/${repo}/commits/${encodeURIComponent(sha)}`);
    return human(commit.author) ? commit.author.login : null;
}

export async function notification({ repo, eventName, event, inputs = {}, runUrl }, api) {
    if (!['vertesia/studio', 'vertesia/composableai', 'vertesia/llumiverse'].includes(repo)) return null;
    if (inputs.source_repository) {
        const expected = repo === 'vertesia/studio' ? 'vertesia/composableai' : 'vertesia/llumiverse';
        if (inputs.source_repository !== expected || !/^[a-f0-9]{40}$/.test(inputs.source_sha)) {
            throw new Error('Invalid sync source repository or SHA');
        }
        const login = await commitOwner(api, inputs.source_repository, inputs.source_sha);
        if (!login) throw new Error('Cannot resolve a human author for the failed sync');
        return { login, message: `${escapeSlack(inputs.reason)}\n<${runUrl}|Failed sync run>` };
    }

    const called = Boolean(inputs.reason);
    const run = event.workflow_run;
    let branch = inputs.branch;
    let sha = inputs.sha;
    let reason = inputs.reason;
    let detailsUrl = runUrl;
    if (!called && eventName === 'workflow_run') {
        if (!['pull_request', 'push'].includes(run.event) || !failureConclusions.has(run.conclusion)) return null;
        branch = run.head_branch;
        sha = run.head_sha;
        reason = `CI workflow ${run.name} concluded ${run.conclusion}.`;
        detailsUrl = run.html_url;
        // A delayed event from an older run/attempt must not report a recovered failure.
        const latest = await api(`repos/${repo}/actions/workflows/${run.workflow_id}/runs?branch=${encodeURIComponent(branch)}&event=${run.event}&head_sha=${sha}&per_page=1`);
        const current = latest.workflow_runs[0];
        if (!current || current.id !== run.id || current.run_attempt !== run.run_attempt) return null;
        if (run.event === 'push') {
            const watched = branch === 'main' || /^release\/.+$/.test(branch)
                || (repo !== 'vertesia/studio' && branch === 'preview');
            if (!watched) return null;
            // A newer push can cancel old CI. Suppress those cancellations, but a real
            // failure still belongs to its author even if another commit has landed.
            if (run.conclusion === 'cancelled') {
                const tip = await api(`repos/${repo}/commits/${encodeURIComponent(branch)}`);
                if (tip.sha !== sha) return null;
            }
            const login = await commitOwner(api, repo, sha);
            if (!login) throw new Error('Cannot resolve a human author for the failed branch CI');
            return { login, message: `CI needs your attention on ${repo} branch ${escapeSlack(branch)}.\n${escapeSlack(reason)}\n<https://github.com/${repo}/commit/${sha}|Commit ${sha.slice(0, 7)}> · <${detailsUrl}|Failed workflow run>` };
        }
    } else if (!called && eventName === 'pull_request_target') {
        if (!event.pull_request.draft) return null;
        branch = event.pull_request.head.ref;
        sha = event.pull_request.head.sha;
        reason = 'Created as a draft with conflicts. Resolve the conflicts and mark the PR ready for review.';
    } else if (!called) {
        return null;
    }

    const kind = kindOf(repo, branch || '');
    if (!kind) return null;
    const prs = await api(`repos/${repo}/pulls?state=open&head=${repo.split('/')[0]}:${encodeURIComponent(branch)}&per_page=100`);
    if (prs.length !== 1) return null;
    const pr = prs[0];
    if (pr.head.repo?.full_name !== repo || pr.head.sha !== sha || normalizeBotLogin(pr.user.login) !== kind.bot) return null;
    if (!/^(main|preview|release\/\d+\.\d+)$/.test(pr.base.ref)) return null;
    // Creation precedes label/assignee updates; resolve provenance from GitHub instead.
    if (!called && eventName === 'pull_request_target' && !pr.draft) return null;
    const login = await prOwner(api, repo, pr, kind);
    if (!login) throw new Error(`Cannot resolve a human author for ${pr.html_url}`);
    return {
        login,
        message: `${kind.kind} needs your attention: <${pr.html_url}|${repo}#${pr.number}>\n${escapeSlack(reason)}\n<${detailsUrl}|Workflow run>`,
    };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    const event = JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8'));
    const result = await notification({
        repo: process.env.GITHUB_REPOSITORY,
        eventName: process.env.GITHUB_EVENT_NAME,
        event,
        inputs: JSON.parse(process.env.NOTIFICATION_INPUTS || '{}'),
        runUrl: `${process.env.GITHUB_SERVER_URL}/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID}`,
    }, async (path) => JSON.parse(execFileSync('gh', ['api', path], { encoding: 'utf8' })));
    if (result) {
        appendFileSync(process.env.GITHUB_OUTPUT, `login=${result.login}\nmessage=${JSON.stringify(result.message)}\n`);
    }
}
