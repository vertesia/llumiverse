import { canonicalJsonContentString } from './json-content-runtime.js';
import { fingerprintJson } from './runtime.js';
import {
    ConversationStreamCursorSchema,
    ConversationStreamEventSchema,
    ConversationStreamIdentitySchema,
} from './schemas/streaming.js';
import type {
    ConversationStreamIdentity,
    ConversationStreamTransformationProof,
    DecodedConversationResponse,
    JsonBlock,
    NativeStreamPosition,
    TextBlock,
} from './types.js';

export {
    CONVERSATION_STREAM_MAX_EVENT_BYTES,
    CONVERSATION_STREAM_MAX_EVENTS,
    CONVERSATION_STREAM_MAX_TOTAL_BYTES,
    type ConversationStreamAccumulatorOptions,
    type ConversationStreamDraftSnapshot,
    conversationStreamCursor,
    conversationStreamEventId,
} from './streaming-runtime.js';

import {
    type ConversationStreamAccumulatorOptions,
    ConversationStreamAccumulatorRuntime,
    type ConversationStreamRuntimeValidators,
} from './streaming-runtime.js';

const SCHEMA_STREAM_VALIDATORS: ConversationStreamRuntimeValidators = {
    identity: (input) => ConversationStreamIdentitySchema.parse(input),
    cursor: (input) => ConversationStreamCursorSchema.parse(input),
    event: (input) => ConversationStreamEventSchema.parse(input),
};

/** Schema-authoritative canonical stream accumulator used by hosts and provider adapters. */
export class ConversationStreamAccumulator extends ConversationStreamAccumulatorRuntime {
    constructor(identityInput: ConversationStreamIdentity, options: ConversationStreamAccumulatorOptions = {}) {
        super(identityInput, SCHEMA_STREAM_VALIDATORS, options);
    }
}

function positionKey(position: NativeStreamPosition): string {
    return canonicalJsonContentString(position);
}

export async function createStructuredOutputTransformationProof(input: {
    id: string;
    source_blocks: readonly TextBlock[];
    result_block: JsonBlock;
}): Promise<ConversationStreamTransformationProof> {
    if (input.source_blocks.length === 0) throw new Error('Structured output transformation requires source blocks');
    return {
        id: input.id,
        type: 'structured_output',
        source_block_ids: input.source_blocks.map((block) => block.id),
        source_texts: input.source_blocks.map((block) => block.text),
        result_block_id: input.result_block.id,
        source_fingerprint: await fingerprintJson(
            input.source_blocks.map((block) => ({ id: block.id, text: block.text })),
        ),
        result_fingerprint: await fingerprintJson({ id: input.result_block.id, value: input.result_block.value }),
    };
}

export async function assertStructuredOutputTransformationProof(
    proof: ConversationStreamTransformationProof,
    sourceBlocks: readonly TextBlock[],
    resultBlock: JsonBlock,
): Promise<void> {
    const actual = await createStructuredOutputTransformationProof({
        id: proof.id,
        source_blocks: sourceBlocks,
        result_block: resultBlock,
    });
    if (canonicalJsonContentString(actual) !== canonicalJsonContentString(proof)) {
        throw new Error(`Structured output transformation ${proof.id} does not match decoded blocks`);
    }
}

/** Validate transformation evidence against the blocks produced by this concrete native decode. */
export async function assertConversationStreamDecodeEvidence(decoded: DecodedConversationResponse): Promise<void> {
    const evidence = decoded.stream_evidence;
    if (evidence === undefined) return;
    const turns = decoded.turns;
    const turnIds = new Set(turns.map((turn) => turn.id));
    const blocks = new Map(turns.flatMap((turn) => turn.blocks.map((block) => [block.id, block] as const)));
    const callIds = new Set(
        turns.flatMap((turn) => turn.blocks.flatMap((block) => (block.type === 'tool_call' ? [block.call_id] : []))),
    );
    const transientSourceIds = new Set(evidence.transformations.flatMap((proof) => proof.source_block_ids));
    const mappedCanonicalIds = new Set<string>();
    const mappedPositionsByKind = new Set<string>();
    for (const mapping of evidence.item_mappings) {
        const resolves =
            (mapping.kind === 'turn' && turnIds.has(mapping.canonical_id)) ||
            (mapping.kind === 'block' &&
                (blocks.has(mapping.canonical_id) || transientSourceIds.has(mapping.canonical_id))) ||
            (mapping.kind === 'call' && callIds.has(mapping.canonical_id));
        if (!resolves) throw new Error(`Stream decode mapping cannot resolve ${mapping.kind} ${mapping.canonical_id}`);
        const canonicalKey = `${mapping.kind}:${mapping.canonical_id}`;
        if (mappedCanonicalIds.has(canonicalKey)) {
            throw new Error(`Stream decode maps ${mapping.kind} ${mapping.canonical_id} more than once`);
        }
        mappedCanonicalIds.add(canonicalKey);
        const position = `${mapping.kind}:${positionKey(mapping.native_position)}`;
        if (mappedPositionsByKind.has(position)) {
            throw new Error(`Stream decode maps one native position to more than one ${mapping.kind}`);
        }
        mappedPositionsByKind.add(position);
    }
    const transformationIds = new Set<string>();
    const transformedSourceIds = new Set<string>();
    for (const proof of evidence.transformations) {
        if (transformationIds.has(proof.id)) throw new Error(`Duplicate stream transformation ${proof.id}`);
        transformationIds.add(proof.id);
        if (proof.source_block_ids.length !== proof.source_texts.length) {
            throw new Error(`Structured-output transformation ${proof.id} has mismatched source evidence`);
        }
        const sourceBlocks = proof.source_block_ids.map((id, index) => {
            if (transformedSourceIds.has(id)) throw new Error(`Structured-output source ${id} is transformed twice`);
            transformedSourceIds.add(id);
            return { id, type: 'text' as const, text: proof.source_texts[index] ?? '', format: 'plain' as const };
        });
        const resultBlock = blocks.get(proof.result_block_id);
        if (resultBlock?.type !== 'json') {
            throw new Error(`Structured-output result ${proof.result_block_id} is not a decoded JSON block`);
        }
        await assertStructuredOutputTransformationProof(proof, sourceBlocks, resultBlock);
    }
}
