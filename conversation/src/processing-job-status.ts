import type { ProcessingState } from './types.js';

/** Shared durable readiness predicate; indexed migration records only its bounded count. */
export function countUnresolvedProcessingJobs(
    processing: Pick<ProcessingState, 'jobs' | 'completions' | 'supersessions'>,
): number {
    let count = 0;
    for (const job of Object.values(processing.jobs ?? {})) {
        const completion = processing.completions?.[job.id];
        if ((!completion || completion.status === 'blocked') && !processing.supersessions?.[job.id]) count += 1;
    }
    return count;
}
