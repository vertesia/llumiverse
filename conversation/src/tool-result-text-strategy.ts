import { z } from 'zod';
import { preflightJsonInput } from './json-preflight.js';
import { IdentifierSchema } from './schemas/primitives.js';
import type { ProcessingJob } from './types.js';

export const TOOL_RESULT_TEXT_PROCESSOR_ID = 'externalize-tool-result-text';
export const TOOL_RESULT_TEXT_PROCESSOR_VERSION = '1';
export const TOOL_RESULT_TEXT_PARTIAL_PROCESSOR_VERSION = '2';
/** Explicit opt-in current-projection processing; never automatic re-ingestion of replacements. */
export const TOOL_RESULT_TEXT_CHAINED_PROCESSOR_VERSION = '3';
export const TOOL_RESULT_TEXT_MAX_BYTES = 32 * 1024 * 1024;
export const TOOL_RESULT_TEXT_MAX_BLOCKS = 4096;
const ExactTextBlockSchema = z.strictObject({
    turn_id: IdentifierSchema,
    result_block_id: IdentifierSchema,
    block_id: IdentifierSchema,
});
/** Content IDs nominate only parts of the already authenticated selected result; never an access grant. */
export const ToolResultTextConfigurationV2Schema = z.strictObject({
    selector: z.discriminatedUnion('kind', [
        z.strictObject({ kind: z.literal('all_text') }),
        z.strictObject({
            kind: z.literal('minimum_text_bytes'),
            minimum_text_bytes: z.number().int().nonnegative().max(TOOL_RESULT_TEXT_MAX_BYTES),
        }),
        z.strictObject({
            kind: z.literal('exact_blocks'),
            blocks: z.array(ExactTextBlockSchema).min(1).max(TOOL_RESULT_TEXT_MAX_BLOCKS),
        }),
    ]),
});
export type ToolResultTextConfigurationV2 = z.infer<typeof ToolResultTextConfigurationV2Schema>;
export type ToolResultTextStrategy = Pick<ProcessingJob, 'processor_id' | 'processor_version' | 'configuration'>;
export function isToolResultTextStrategy(id: string, version: string): boolean {
    return (
        id === TOOL_RESULT_TEXT_PROCESSOR_ID &&
        (version === TOOL_RESULT_TEXT_PROCESSOR_VERSION ||
            version === TOOL_RESULT_TEXT_PARTIAL_PROCESSOR_VERSION ||
            version === TOOL_RESULT_TEXT_CHAINED_PROCESSOR_VERSION)
    );
}
/** Manual v2 selects nested text through its registered configuration; v1 remains append-only. */
export function supportsToolResultTextProcessingScope(
    job: Pick<ProcessingJob, 'processor_id' | 'processor_version' | 'scope'>,
): boolean {
    return (
        isToolResultTextStrategy(job.processor_id, job.processor_version) &&
        ((job.scope === 'on_append' && job.processor_version !== TOOL_RESULT_TEXT_CHAINED_PROCESSOR_VERSION) ||
            (job.scope === 'manual' && job.processor_version !== TOOL_RESULT_TEXT_PROCESSOR_VERSION))
    );
}

export function parseToolResultTextStrategy(
    strategy: ToolResultTextStrategy,
): ToolResultTextConfigurationV2['selector'] {
    if (!isToolResultTextStrategy(strategy.processor_id, strategy.processor_version))
        throw new Error('Tool-result text strategy is not registered');
    if (!preflightJsonInput(strategy.configuration, { max_bytes: 512 * 1024 }).success)
        throw new TypeError('Tool-result text strategy exceeds its bounded configuration');
    if (strategy.processor_version === TOOL_RESULT_TEXT_PROCESSOR_VERSION) {
        if (Object.keys(strategy.configuration).length !== 0)
            throw new Error('Tool-result text v1 requires its original empty configuration');
        return { kind: 'all_text' };
    }
    return ToolResultTextConfigurationV2Schema.parse(strategy.configuration).selector;
}
function textIdentity(input: { turn_id: string; result_block_id: string; block_id: string }): string {
    return JSON.stringify([input.turn_id, input.result_block_id, input.block_id]);
}
/** Pure selection over complete authenticated candidate blocks. Unknown, duplicate and nontext IDs reject. */
export function selectToolResultTextBlocks<
    T extends { turn_id: string; result_block_id: string; block_id: string; text: string },
>(texts: readonly T[], strategy?: ToolResultTextStrategy): T[] {
    const selector = strategy ? parseToolResultTextStrategy(strategy) : { kind: 'all_text' as const };
    const candidates = new Map(texts.map((text) => [textIdentity(text), text]));
    if (texts.length > TOOL_RESULT_TEXT_MAX_BLOCKS || candidates.size !== texts.length)
        throw new Error('Tool-result text selection contains ambiguous or excessive original blocks');
    if (selector.kind === 'all_text') return [...texts];
    if (selector.kind === 'minimum_text_bytes') {
        const encoder = new TextEncoder();
        return texts.filter((text) => encoder.encode(text.text).byteLength >= selector.minimum_text_bytes);
    }
    const selected = new Set<string>();
    for (const nominated of selector.blocks) {
        const key = textIdentity(nominated);
        if (selected.has(key)) throw new Error('Tool-result text selector repeats an accepted block');
        if (!candidates.has(key))
            throw new Error('Tool-result text selector names unknown or nontext accepted content');
        selected.add(key);
    }
    return texts.filter((text) => selected.has(textIdentity(text)));
}
