import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

// A successful workflow can contain only skipped tests. Inspect the latest
// substantive run and require the jobs/steps declared by this repository.
export function verifyWorkflow(runs, loadJobs, policy, { sha, branch, pr }) {
    const candidates = runs
        .filter(
            (run) =>
                run.head_sha === sha &&
                run.head_branch === branch &&
                run.event === 'pull_request' &&
                run.pull_requests?.some((item) => item.number === pr),
        )
        .sort((a, b) => b.id - a.id);
    for (const run of candidates) {
        const jobs = loadJobs(run.id);
        const noOp = policy.noOp;
        if (
            noOp &&
            run.status === 'completed' &&
            run.conclusion === 'success' &&
            jobs.some((job) => job.name === noOp.gate && job.conclusion === 'success') &&
            jobs.some((job) => job.name === noOp.skipped && job.conclusion === 'skipped') &&
            jobs.every(
                (job) =>
                    job.status === 'completed' &&
                    (job.name === noOp.gate ? job.conclusion === 'success' : job.conclusion === 'skipped'),
            )
        ) {
            continue;
        }
        // Never fall back past a pending, failed, cancelled or malformed real run.
        if (run.status !== 'completed' || run.conclusion !== 'success') return false;
        return (
            policy.jobs.length > 0 &&
            policy.jobs.every((required) => {
                const matches = jobs.filter((job) => new RegExp(required.name).test(job.name));
                return (
                    matches.length > 0 &&
                    matches.every(
                        (job) =>
                            job.status === 'completed' &&
                            job.conclusion === 'success' &&
                            (required.steps ?? []).every((name) =>
                                job.steps?.some(
                                    (step) =>
                                        step.name === name &&
                                        step.status === 'completed' &&
                                        step.conclusion === 'success',
                                ),
                            ),
                    )
                );
            })
        );
    }
    return false;
}

export function ghPages(endpoint) {
    // Both runs and jobs can exceed one page. API failures throw and block merging.
    return JSON.parse(
        execFileSync('gh', ['api', '--paginate', '--slurp', endpoint], {
            encoding: 'utf8',
            maxBuffer: 32 * 1024 * 1024,
        }),
    );
}

export function main(env, pr, workflows, pages = ghPages) {
    const { REPO: repo, HEAD_BRANCH: branch, HEAD_SHA: sha } = env;
    if (!repo || !branch || !sha || !Number.isSafeInteger(pr) || pr <= 0 || workflows.length === 0) {
        throw new Error('Missing repository, branch, commit, PR or workflows');
    }
    const policies = JSON.parse(readFileSync(new URL('./automerge-ci-policy.json', import.meta.url), 'utf8'));
    for (const workflow of workflows) {
        const policy = policies[workflow];
        if (!policy) throw new Error(`No CI policy for ${workflow}`);
        const query = new URLSearchParams({ branch, head_sha: sha, event: 'pull_request', per_page: '100' });
        const runs = pages(`repos/${repo}/actions/workflows/${workflow}/runs?${query}`).flatMap(
            (page) => page.workflow_runs,
        );
        const passed = verifyWorkflow(
            runs,
            (id) =>
                pages(`repos/${repo}/actions/runs/${id}/jobs?filter=latest&per_page=100`).flatMap((page) => page.jobs),
            policy,
            { sha, branch, pr },
        );
        if (!passed) {
            console.log(`Required jobs have not passed in ${workflow} for ${sha}.`);
            return false;
        }
        console.log(`${workflow}: required CI jobs passed for ${sha}.`);
    }
    return true;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    try {
        process.exitCode = main(process.env, Number(process.argv[2]), process.argv.slice(3)) ? 0 : 1;
    } catch (error) {
        console.error(error);
        process.exitCode = 1;
    }
}
