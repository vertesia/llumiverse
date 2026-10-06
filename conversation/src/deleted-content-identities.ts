import type { ContentBlock } from './types.js';

/** Includes nested result content: every deleted identity remains reserved without retaining public bodies. */
export function deletedContentIdentities(blocks: readonly ContentBlock[]): { block_ids: string[]; call_ids: string[] } {
    const block_ids: string[] = [];
    const call_ids: string[] = [];
    const visit = (block: ContentBlock): void => {
        block_ids.push(block.id);
        if (block.type === 'tool_call') call_ids.push(block.call_id);
        if (block.type === 'tool_result') for (const nested of block.content) visit(nested);
    };
    for (const block of blocks) visit(block);
    return { block_ids, call_ids };
}
