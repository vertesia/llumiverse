import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

// A successful workflow can contain only skipped tests. Inspect the latest
// substantive run and require the jobs/steps declared by this repository.
export function verifyWorkflow(runs, loadJobs, policy, { sha, branch, pr, baseSha, baseBranch }, explain = () => {}) {
    // Every rejection names its cause, so a gate log shows which check withheld approval. A rejection
    // for a run that has not finished also passes that run, so a caller can tell CI that is still
    // running apart from missing or failed CI data.
    const reject = (reason, running) => {
        explain(reason, running);
        return false;
    };
    const candidates = runs
        .filter(
            (run) =>
                run.head_sha === sha &&
                run.head_branch === branch &&
                run.event === 'pull_request' &&
                run.pull_requests?.some((item) => item.number === pr),
        )
        .sort((a, b) => b.id - a.id);
    if (candidates.length === 0) {
        const unlinked = runs.filter(
            (run) => run.head_sha === sha && run.head_branch === branch && run.event === 'pull_request',
        );
        if (unlinked.length === 0) return reject(`no pull_request run for this head among ${runs.length} listed runs`);
        const listed = unlinked.map((run) => `${run.id}: [${(run.pull_requests ?? []).map((item) => item.number)}]`);
        return reject(`runs for this head are not associated with PR #${pr} (${listed.join('; ')})`);
    }
    for (const run of candidates) {
        // Approval may require CI for the current base as well as the head.
        // Validate even no-op runs before considering fallback to older CI.
        if (baseSha || baseBranch) {
            const testedPr = run.pull_requests.find((item) => item.number === pr);
            if (testedPr.base?.sha !== baseSha || testedPr.base?.ref !== baseBranch) {
                const tested = `${testedPr.base?.ref}@${testedPr.base?.sha}`;
                return reject(`run ${run.id} tested base ${tested}, not ${baseBranch}@${baseSha}`);
            }
        }
        const jobs = loadJobs(run.id);
        // Ignore an intentionally empty run, such as a label router that did not invoke the
        // pipeline, before evaluating the actual CI run.
        if (
            run.status === 'completed' &&
            (run.conclusion === 'success' || run.conclusion === 'skipped') &&
            jobs.length === 1 &&
            policy.noOpJobs?.includes(jobs[0].name) &&
            jobs[0].status === 'completed' &&
            jobs[0].conclusion === 'skipped'
        ) {
            continue;
        }
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
        // A coordinator run can fail or be cancelled because its *other* reusable workflow failed
        // or was superseded. Judge this policy's jobs independently, but only after the run is
        // terminal; missing, skipped, cancelled or failed required jobs still fail closed below.
        // Any other run must succeed as a whole: its required jobs do not cover every job in it.
        if (run.status !== 'completed') return reject(`run ${run.id} is ${run.status}`, run);
        const coordinator = Boolean(policy.workflow) && run.path === `.github/workflows/${policy.workflow}`;
        if (!coordinator && run.conclusion !== 'success') return reject(`run ${run.id} concluded ${run.conclusion}`);
        if (policy.jobs.length === 0) return reject('the CI policy declares no required jobs');
        // A job a policy requires only on some base branches (the license check, which maintained release
        // lines do not run) is required when the run's recorded base is unknown, so a gap fails closed.
        const testedBase = run.pull_requests.find((item) => item.number === pr)?.base?.ref ?? baseBranch;
        for (const required of policy.jobs) {
            if (required.baseBranches && testedBase && !required.baseBranches.includes(testedBase)) continue;
            const matches = jobs.filter((job) => new RegExp(required.name).test(job.name));
            if (matches.length === 0) return reject(`run ${run.id} has no job matching ${required.name}`);
            for (const job of matches) {
                if (job.status !== 'completed' || job.conclusion !== 'success') {
                    return reject(`run ${run.id} job "${job.name}" is ${job.status}/${job.conclusion}`);
                }
                const passedStep = (name) =>
                    job.steps?.some(
                        (step) => step.name === name && step.status === 'completed' && step.conclusion === 'success',
                    );
                const missing = (required.steps ?? []).find((name) => !passedStep(name));
                if (missing) return reject(`run ${run.id} job "${job.name}" did not pass step "${missing}"`);
            }
        }
        return true;
    }
    return reject('every run for this head is a no-op');
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

export function main(env, pr, workflows, pages = ghPages, onReject = () => {}) {
    const { REPO: repo, HEAD_BRANCH: branch, HEAD_SHA: sha, BASE_SHA: baseSha, BASE_BRANCH: baseBranch } = env;
    if (!repo || !branch || !sha || !Number.isSafeInteger(pr) || pr <= 0 || workflows.length === 0) {
        throw new Error('Missing repository, branch, commit, PR or workflows');
    }
    if (Boolean(baseSha) !== Boolean(baseBranch)) throw new Error('Both base branch and commit are required');
    const policies = JSON.parse(readFileSync(new URL('./automerge-ci-policy.json', import.meta.url), 'utf8'));
    const pageCache = new Map();
    const cachedPages = (endpoint) => {
        if (!pageCache.has(endpoint)) pageCache.set(endpoint, pages(endpoint));
        return pageCache.get(endpoint);
    };
    const jobsCache = new Map();
    const loadJobs = (id) => {
        if (!jobsCache.has(id)) {
            jobsCache.set(
                id,
                cachedPages(`repos/${repo}/actions/runs/${id}/jobs?filter=latest&per_page=100`).flatMap(
                    (page) => page.jobs,
                ),
            );
        }
        return jobsCache.get(id);
    };
    for (const workflow of workflows) {
        const policy = policies[workflow];
        if (!policy) throw new Error(`No CI policy for ${workflow}`);
        const query = new URLSearchParams({ branch, head_sha: sha, event: 'pull_request', per_page: '100' });
        const runs = [
            policy.workflow ?? workflow,
            ...(policy.fallbackWorkflows ?? []),
            ...(policy.companionWorkflows ?? []),
        ].flatMap(
            (source) => cachedPages(`repos/${repo}/actions/workflows/${source}/runs?${query}`).flatMap(
                (page) => page.workflow_runs,
            ),
        );
        let reason = '';
        let running;
        const passed = verifyWorkflow(runs, loadJobs, policy, { sha, branch, pr, baseSha, baseBranch }, (why, run) => {
            reason = why;
            running = run;
        });
        if (!passed) {
            console.log(`Required jobs have not passed in ${workflow} for ${sha}: ${reason}.`);
            onReject({ workflow, reason, running });
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
