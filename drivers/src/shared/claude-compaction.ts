/**
 * Server-side compaction for Claude (Messages API).
 *
 * Threshold compaction (beta `compact-2026-01-12`): when the request's input reaches the trigger, the API summarizes
 * the earlier turns and returns a `compaction` block first in the response content.
 *
 * On-demand compaction (beta `compact-2026-09-04`): a request with `compaction: { type: 'summarize' }` returns only the
 * `compaction` block, with stop reason `compaction`, and no answer.
 *
 * Either way the block must be sent back unchanged; everything before it is superseded, and the conversation may hold
 * only one compaction block with content, so the stored conversation is trimmed to start at the latest one.
 */

import type {
    BetaCompact20260112Edit,
    BetaCompactionConfig,
    BetaCompactionContentBlockDelta,
    BetaContextManagementConfig,
} from '@anthropic-ai/sdk/resources/beta.js';
import type {
    ContentBlock,
    Message,
    MessageParam,
    RawMessageStreamEvent,
} from '@anthropic-ai/sdk/resources/messages.js';
import type { CompactConversationOptions, CompletionResult, ConversationCompactionOptions } from '@llumiverse/core';

export const CLAUDE_COMPACTION_BETA = 'compact-2026-01-12';
export const CLAUDE_ON_DEMAND_COMPACTION_BETA = 'compact-2026-09-04';

/** The `compaction` field of an on-demand compaction request; an omitted prompt keeps the API's default. */
export function claudeOnDemandCompaction(request: CompactConversationOptions): BetaCompactionConfig {
    return {
        type: 'summarize',
        ...(request.instructions !== undefined && { instructions: request.instructions }),
    };
}

/** The summary text of the compaction blocks in a response, reported as the result of an on-demand compaction. */
export function claudeCompactionResults(content: readonly unknown[]): CompletionResult[] {
    return content.flatMap((block): CompletionResult[] =>
        isClaudeCompactionSummary(block) ? [{ type: 'text', value: (block as { content: string }).content }] : [],
    );
}

/** The `context_management` field for the caller's compaction options; omitted fields keep the API defaults. */
export function claudeCompactionContextManagement(
    compaction: ConversationCompactionOptions | undefined,
): BetaContextManagementConfig | undefined {
    if (!compaction) return undefined;
    const edit: BetaCompact20260112Edit = {
        type: 'compact_20260112',
        ...(compaction.trigger_tokens !== undefined && {
            trigger: { type: 'input_tokens', value: compaction.trigger_tokens },
        }),
        ...(compaction.instructions !== undefined && { instructions: compaction.instructions }),
    };
    return { edits: [edit] };
}

type CompactionDelta = Pick<BetaCompactionContentBlockDelta, 'content' | 'encrypted_content'>;

/**
 * The summaries streamed for compaction blocks, by content index.
 *
 * The block starts with no content and a single `compaction_delta` carries its final value. The stable SDK's message
 * accumulator, which every Claude client here uses, ignores that delta, so its final message holds the block without
 * the summary. Collect the deltas from the raw events and write them back onto the final message.
 */
export class ClaudeCompactionDeltas {
    private readonly deltas = new Map<number, CompactionDelta>();

    track(event: RawMessageStreamEvent): void {
        if (event.type !== 'content_block_delta') return;
        const delta = event.delta as { type: string } | BetaCompactionContentBlockDelta;
        if (delta.type !== 'compaction_delta') return;
        const { content, encrypted_content } = delta as BetaCompactionContentBlockDelta;
        this.deltas.set(event.index, { content, encrypted_content });
    }

    apply(message: Message): Message {
        if (this.deltas.size === 0) return message;
        // The stable SDK types do not declare compaction blocks, so the patched block is cast back to their union.
        const content = message.content.map((block, index): ContentBlock => {
            const delta = this.deltas.get(index);
            return delta && isClaudeCompactionBlock(block)
                ? ({ ...block, ...delta } as unknown as ContentBlock)
                : block;
        });
        return { ...message, content };
    }
}

export function isClaudeCompactionBlock(block: unknown): boolean {
    return typeof block === 'object' && block !== null && (block as { type?: unknown }).type === 'compaction';
}

/** A compaction block that carries a summary. A block with null content is a failed compaction, which the API ignores. */
function isClaudeCompactionSummary(block: unknown): boolean {
    const content = (block as { content?: unknown }).content;
    return isClaudeCompactionBlock(block) && typeof content === 'string' && content.length > 0;
}

/**
 * Drop everything before the latest compaction summary, which replaces it. The API ignores that history once the
 * block is in the conversation, but it would still be stored and sent with every request, and a second summary
 * alongside the first is rejected (`compaction_block_ambiguous`). Conversations without a summary are returned as is.
 */
export function trimClaudeConversationToCompaction<T extends { messages: MessageParam[] }>(conversation: T): T {
    const { messages } = conversation;
    for (let messageIndex = messages.length - 1; messageIndex >= 0; messageIndex--) {
        const message = messages[messageIndex];
        if (!Array.isArray(message.content)) continue;
        const blockIndex = message.content.findLastIndex(isClaudeCompactionSummary);
        if (blockIndex < 0) continue;
        if (messageIndex === 0 && blockIndex === 0) return conversation;
        const first = { ...message, content: message.content.slice(blockIndex) } as MessageParam;
        return { ...conversation, messages: [first, ...messages.slice(messageIndex + 1)] };
    }
    return conversation;
}

interface ClaudeIterationUsage {
    type?: string;
    input_tokens?: number | null;
    output_tokens?: number | null;
    cache_read_input_tokens?: number | null;
    cache_creation_input_tokens?: number | null;
    cache_creation?: { ephemeral_1h_input_tokens?: number | null } | null;
}

/** Token usage of the compaction iterations, which the API leaves out of the response's top-level usage. */
export function claudeCompactionUsage(iterations: readonly ClaudeIterationUsage[] | null | undefined) {
    const total = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cacheWrite1h: 0, count: 0 };
    for (const iteration of iterations ?? []) {
        if (iteration.type !== 'compaction') continue;
        total.count++;
        total.input += iteration.input_tokens ?? 0;
        total.output += iteration.output_tokens ?? 0;
        total.cacheRead += iteration.cache_read_input_tokens ?? 0;
        total.cacheWrite += iteration.cache_creation_input_tokens ?? 0;
        total.cacheWrite1h += iteration.cache_creation?.ephemeral_1h_input_tokens ?? 0;
    }
    return total;
}
