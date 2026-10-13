import type { BetaInputTransformation } from '@anthropic-ai/sdk/resources/beta.js';
import type { Logger } from '@llumiverse/core';

export function claudeFinishReason(reason: string | undefined): string | undefined {
    if (!reason) return undefined;
    switch (reason) {
        case 'end_turn':
            return 'stop';
        case 'max_tokens':
        case 'model_context_window_exceeded':
            return 'length';
        default:
            return reason;
    }
}

/** Preserve Claude's provider-native truncation cause after normalizing control flow to `length`. */
export function logClaudeTruncation(
    logger: Logger | undefined,
    reason: string | null | undefined,
    context: { provider: string; model: string },
): void {
    if (!logger) return;

    const details = {
        ...context,
        finish_reason: 'length',
        provider_finish_reason: reason,
    };
    if (reason === 'max_tokens') {
        logger.warn(details, '[Claude] Completion stopped at the output token limit');
    } else if (reason === 'model_context_window_exceeded') {
        logger.warn(details, '[Claude] Completion exceeded the model context window');
    }
}

/**
 * Report thinking blocks the API dropped or let through because the conversation before them changed
 * (`thinking.block_binding`). Each entry means earlier reasoning was not reused for this request.
 */
export function logClaudeInputTransformations(
    logger: Logger | undefined,
    transformations: BetaInputTransformation[] | null | undefined,
    context: { provider: string; model: string },
): void {
    if (!logger || !transformations?.length) return;

    logger.warn(
        {
            ...context,
            transformation_count: transformations.length,
            transformation_types: [...new Set(transformations.map((t) => t.type))],
            transformation_reasons: [...new Set(transformations.map((t) => t.reason))],
            first_path: transformations[0].path,
        },
        '[Claude] API transformed replayed thinking blocks',
    );
}
