import type {
    AgentContentBlock,
    ConversationTurn,
    DecodedConversationResponse,
    JsonObject,
    JsonValue,
    NativeReplayBlock,
} from '@llumiverse/conversation';
import type { CanonicalStructuredOutput } from '@llumiverse/core';

export interface CanonicalStructuredOutputEvidence extends JsonObject {
    type: 'canonical_structured_output';
    block_id: string;
    source_block_ids: string[];
    source_texts: string[];
    normalized_value: JsonValue;
}

export interface CanonicalStructuredOutputBinding {
    block_id: string;
    source_block_ids: string[];
    source_texts: string[];
    value: JsonValue;
}

interface StructuredOutputReplayInput {
    turn: Extract<ConversationTurn, { kind: 'agent' }>;
    semantic_blocks: AgentContentBlock[];
    replay_blocks: NativeReplayBlock[];
    binding: CanonicalStructuredOutputBinding;
}

export type StructuredOutputReplayRewriter = (
    input: StructuredOutputReplayInput,
) => NativeReplayBlock[] | Promise<NativeReplayBlock[]>;

export interface InvalidStructuredOutputEvidence {
    code: 'validation_error' | 'json_error';
    message: string;
}

/** Retain raw decoded evidence while marking an invalid required structured result as failed. */
export function rejectDecodedStructuredOutput(
    decoded: DecodedConversationResponse,
    error: InvalidStructuredOutputEvidence,
): DecodedConversationResponse {
    if (decoded.turns.length !== 1 || decoded.turns[0]?.kind !== 'agent') {
        throw new TypeError('Structured output rejection requires one decoded agent turn');
    }
    return {
        ...decoded,
        turns: [{ ...decoded.turns[0], status: 'failed' }],
        generation: {
            ...decoded.generation,
            status: 'failed',
            metadata: {
                ...decoded.generation.metadata,
                structured_output: {
                    status: 'invalid',
                    code: error.code,
                    message: error.message,
                },
            },
        },
    };
}

function stableJson(value: unknown): string {
    if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
    if (typeof value === 'object' && value !== null) {
        const record = value as Record<string, unknown>;
        return `{${Object.keys(record)
            .sort()
            .map((key) => `${JSON.stringify(key)}:${stableJson(record[key])}`)
            .join(',')}}`;
    }
    return JSON.stringify(value) ?? 'undefined';
}

export function structuredOutputEvidence(binding: CanonicalStructuredOutputBinding): CanonicalStructuredOutputEvidence {
    return {
        type: 'canonical_structured_output',
        block_id: binding.block_id,
        source_block_ids: [...binding.source_block_ids],
        source_texts: [...binding.source_texts],
        normalized_value: structuredClone(binding.value),
    };
}

export function parseStructuredOutputEvidence(value: unknown): CanonicalStructuredOutputEvidence {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        throw new TypeError('Structured output replay evidence is not an object');
    }
    const record = value as Record<string, unknown>;
    if (
        record.type !== 'canonical_structured_output' ||
        typeof record.block_id !== 'string' ||
        !Array.isArray(record.source_block_ids) ||
        !record.source_block_ids.every((id) => typeof id === 'string') ||
        !Array.isArray(record.source_texts) ||
        !record.source_texts.every((text) => typeof text === 'string') ||
        !Object.hasOwn(record, 'normalized_value')
    ) {
        throw new TypeError('Structured output replay evidence is malformed');
    }
    return record as CanonicalStructuredOutputEvidence;
}

export function assertStructuredOutputEvidence(
    turn: ConversationTurn,
    evidence: CanonicalStructuredOutputEvidence,
    nativeSourceTexts: readonly string[],
    replayId: string,
): void {
    const block = turn.blocks.find((candidate) => candidate.id === evidence.block_id);
    if (
        block?.type !== 'json' ||
        stableJson(block.value) !== stableJson(evidence.normalized_value) ||
        stableJson(nativeSourceTexts) !== stableJson(evidence.source_texts)
    ) {
        throw new TypeError(`Structured output replay block ${replayId} no longer matches canonical data`);
    }
}

export function remapStructuredOutputReplayDependencies(
    replay: NativeReplayBlock,
    binding: CanonicalStructuredOutputBinding,
): NativeReplayBlock {
    const sources = new Set(binding.source_block_ids);
    const blockIds = replay.dependencies.block_ids.map((id) => (sources.has(id) ? binding.block_id : id));
    return {
        ...replay,
        dependencies: {
            ...replay.dependencies,
            block_ids: [...new Set(blockIds)],
        },
    };
}

/**
 * Replace every answer-text block in one decoded generated turn with one validated JSON block.
 * Native replay is rewritten in the same operation so no receipt can observe a half-normalized turn.
 */
export async function normalizeDecodedStructuredOutput(
    decoded: DecodedConversationResponse,
    structuredOutput: CanonicalStructuredOutput,
    rewriteReplay: StructuredOutputReplayRewriter,
): Promise<DecodedConversationResponse> {
    if (structuredOutput.source_texts.length === 0) return decoded;
    if (decoded.turns.length !== 1 || decoded.turns[0]?.kind !== 'agent') {
        throw new TypeError('Structured output normalization requires one decoded agent turn');
    }
    const turn = decoded.turns[0];
    const sourceBlocks = turn.blocks.filter((block) => block.type === 'text');
    const sourceTexts = sourceBlocks.map((block) => block.text);
    if (stableJson(sourceTexts) !== stableJson(structuredOutput.source_texts)) {
        throw new TypeError('Structured output source partitions do not match decoded canonical text');
    }
    const firstSource = sourceBlocks[0];
    if (firstSource === undefined) {
        throw new TypeError('Structured output has no decoded canonical text block');
    }
    const sourceIds = new Set(sourceBlocks.map((block) => block.id));
    const binding: CanonicalStructuredOutputBinding = {
        block_id: firstSource.id,
        source_block_ids: [...sourceIds],
        source_texts: sourceTexts,
        value: structuredClone(structuredOutput.value),
    };
    let inserted = false;
    const semanticBlocks = turn.blocks.flatMap((block): AgentContentBlock[] => {
        if (block.type === 'native_replay') return [];
        if (!sourceIds.has(block.id)) return [block];
        if (inserted) return [];
        inserted = true;
        return [{ id: binding.block_id, type: 'json', value: binding.value }];
    });
    const replayBlocks = turn.blocks.filter((block): block is NativeReplayBlock => block.type === 'native_replay');
    const normalizedTurn: Extract<ConversationTurn, { kind: 'agent' }> = {
        ...turn,
        blocks: [
            ...semanticBlocks,
            ...(await rewriteReplay({ turn, semantic_blocks: semanticBlocks, replay_blocks: replayBlocks, binding })),
        ],
    };
    return { ...decoded, turns: [normalizedTurn] };
}
