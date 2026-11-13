/**
 * Server-side compaction for the OpenAI Responses API.
 *
 * Once the rendered input crosses `compact_threshold`, the response carries an opaque `compaction` output item that
 * replaces the context before it. The item is sent back with the rest of the output, and the items before the latest
 * one may be dropped, so the stored conversation is trimmed to start there.
 */

import type { ConversationCompactionOptions, Logger } from '@llumiverse/core';
import type OpenAI from 'openai';
import { logModelOptionException } from '../shared/model-option-exceptions.js';

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;
type ContextManagement = NonNullable<OpenAI.Responses.ResponseCreateParams['context_management']>;

/** The `context_management` field for the caller's compaction options; an omitted trigger keeps OpenAI's default. */
export function openAIResponsesContextManagement(
    compaction: ConversationCompactionOptions | undefined,
    logger: Logger | undefined,
    model: string,
): ContextManagement | undefined {
    if (!compaction) return undefined;
    // Compatibility exception: the Responses API has no summarization prompt to carry `instructions`.
    logModelOptionException(logger, model, compaction, ['instructions'], 'openai_compaction_instructions');
    return [
        {
            type: 'compaction',
            ...(compaction.trigger_tokens !== undefined && { compact_threshold: compaction.trigger_tokens }),
        },
    ];
}

/** Drop the items before the latest compaction item, which supersedes them. Conversations without one are unchanged. */
export function trimOpenAIConversationToCompaction(items: ResponseInputItem[]): ResponseInputItem[] {
    const start = items.findLastIndex((item) => item.type === 'compaction');
    return start > 0 ? items.slice(start) : items;
}
