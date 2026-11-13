import type { Logger } from '@llumiverse/core';
import type OpenAI from 'openai';

/**
 * Report replayed reasoning items the API withheld from the model (`response.input_transformations`) because they no
 * longer match the request: another organization, end user or model, a changed prefix, or a turn boundary. The
 * Claude driver reports dropped thinking blocks the same way.
 */
export function logOpenAIInputTransformations(
    logger: Logger | undefined,
    transformations: OpenAI.Responses.Response.InputTransformation[] | null | undefined,
    context: { provider: string; model: string },
): void {
    if (!logger || !transformations?.length) return;
    const counts: Record<string, number> = {};
    for (const { type, action, reason } of transformations) {
        const key = `${type}:${action}:${reason}`;
        counts[key] = (counts[key] ?? 0) + 1;
    }
    logger.warn(
        {
            ...context,
            transformation_count: transformations.length,
            transformations: counts,
            first_id: transformations[0].id,
        },
        '[OpenAI] Replayed reasoning items were withheld from the model',
    );
}
