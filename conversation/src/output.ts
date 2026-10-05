import type { z } from 'zod';
import { canonicalJsonContentString } from './content-integrity.js';
import { ConversationValidationError } from './diagnostics.js';
import { isGeneratedAgentTurn } from './guards.js';
import { preflightJsonInput } from './json-preflight.js';
import { assertAcceptedOutputFragmentSemantics, ConversationOutputProjectionError } from './output-runtime.js';
import { CONVERSATION_EXPERIMENTAL_REVISION, CONVERSATION_SCHEMA_VERSION } from './runtime-constants.js';
import {
    CONVERSATION_ACCEPTED_OUTPUT_FORMAT,
    ConversationAcceptedOutputFragmentSchema,
    type ConversationOutputAssetSchema,
    type ConversationOutputBlockSchema,
    type ConversationOutputCompletenessSchema,
    ConversationOutputGenerationSchema,
    ConversationOutputReceiptSchema,
    ConversationOutputToolCallBlockSchema,
    ConversationOutputTurnSchema,
} from './schemas/output.js';
import { IdentifierSchema } from './schemas/primitives.js';
import { assertHistoricalToolArgumentExternalization } from './tool-arguments.js';
import type {
    Asset,
    ConversationDocument,
    ConversationRef,
    ExecutedGeneration,
    GeneratedAgentTurn,
    OperationReceipt,
} from './types.js';
import { diagnosticsFromZodError, parseConversationDocument } from './validation.js';

export type ConversationAcceptedOutputFragment = z.infer<typeof ConversationAcceptedOutputFragmentSchema>;
export type ConversationOutputAsset = z.infer<typeof ConversationOutputAssetSchema>;
export type ConversationOutputBlock = z.infer<typeof ConversationOutputBlockSchema>;
export type ConversationOutputCompleteness = z.infer<typeof ConversationOutputCompletenessSchema>;
export type ConversationOutputGeneration = z.infer<typeof ConversationOutputGenerationSchema>;
export type ConversationOutputReceipt = z.infer<typeof ConversationOutputReceiptSchema>;
export type ConversationOutputTurn = z.infer<typeof ConversationOutputTurnSchema>;
export {
    assertAcceptedOutputFragmentSemantics,
    ConversationOutputProjectionError,
    type ConversationOutputProjectionErrorCode,
    cloneSemanticallyValidAcceptedOutputFragment,
} from './output-runtime.js';

export type AcceptedOutputFragmentValidationResult =
    | { success: true; data: ConversationAcceptedOutputFragment }
    | { success: false; error: ConversationOutputProjectionError | ConversationValidationError };

function ownRecordValue<T>(record: Record<string, T>, id: string): T | undefined {
    return Object.hasOwn(record, id) ? record[id] : undefined;
}

function resolveAcceptedRecords(document: ConversationDocument, operationId: string) {
    const receipt = ownRecordValue(document.operation_receipts, operationId);
    if (
        !receipt ||
        receipt.id !== operationId ||
        receipt.conversation_id !== document.id ||
        receipt.result_revision !== document.revision ||
        receipt.accepted_turn_ids?.length !== 1 ||
        receipt.accepted_generation_ids?.length !== 1
    ) {
        throw new ConversationOutputProjectionError(
            'receipt_mismatch',
            'The operation does not identify one accepted canonical response',
        );
    }

    const turnId = receipt.accepted_turn_ids[0];
    const generationId = receipt.accepted_generation_ids[0];
    const turn = document.turns.find((candidate) => candidate.id === turnId);
    if (!turn || !isGeneratedAgentTurn(turn) || turn.generation_id !== generationId) {
        throw new ConversationOutputProjectionError(
            'turn_mismatch',
            'The accepted response turn does not match its operation receipt',
        );
    }

    const generation = ownRecordValue(document.generations, generationId);
    if (
        generation?.record_source !== 'executed' ||
        generation.id !== generationId ||
        generation.source.conversation_id !== document.id ||
        generation.source.revision !== receipt.base_revision
    ) {
        throw new ConversationOutputProjectionError(
            'generation_mismatch',
            'The accepted generation does not match its operation receipt',
        );
    }

    return { receipt, turn, generation };
}

/** Validates a standalone persisted or wire output fragment, including all cross-record references. */
export function parseAcceptedOutputFragment(input: unknown): ConversationAcceptedOutputFragment {
    const preflight = preflightJsonInput(input);
    if (!preflight.success) {
        throw new ConversationValidationError('Accepted output fragment failed JSON preflight', preflight.diagnostics);
    }
    const shape = ConversationAcceptedOutputFragmentSchema.safeParse(input);
    if (!shape.success) {
        throw new ConversationValidationError(
            'Accepted output fragment schema validation failed',
            diagnosticsFromZodError(shape.error),
        );
    }
    const fragment = structuredClone(input) as ConversationAcceptedOutputFragment;
    assertAcceptedOutputFragmentSemantics(fragment);
    return fragment;
}

/** Non-throwing companion to parseAcceptedOutputFragment for storage and SDK boundaries. */
export function validateAcceptedOutputFragment(input: unknown): AcceptedOutputFragmentValidationResult {
    try {
        return { success: true, data: parseAcceptedOutputFragment(input) };
    } catch (error) {
        if (error instanceof ConversationOutputProjectionError || error instanceof ConversationValidationError) {
            return { success: false, error };
        }
        throw error;
    }
}

function toolArgumentAssets(
    block: Extract<GeneratedAgentTurn['blocks'][number], { type: 'tool_call' }>,
): { asset_id: string; content_hash: string }[] {
    return block.arguments.type === 'externalized_json'
        ? block.arguments.hydration.map((hydration) => ({
              asset_id: hydration.asset_id,
              content_hash: hydration.content_hash,
          }))
        : [];
}

function referencedAssetId(block: GeneratedAgentTurn['blocks'][number]): string | undefined {
    switch (block.type) {
        case 'image':
        case 'document':
        case 'audio':
        case 'video':
            return block.asset_id;
        default:
            return undefined;
    }
}

function safeGeneratedAsset(asset: Asset | undefined, generationId: string): ConversationOutputAsset | undefined {
    if (asset?.provenance.type !== 'generated' || asset.provenance.generation_id !== generationId) return undefined;
    const { metadata: _metadata, provenance, ...safeAsset } = asset;
    return { ...safeAsset, provenance };
}

function projectReceipt(receipt: ReturnType<typeof resolveAcceptedRecords>['receipt']) {
    const {
        payload_fingerprint: _payloadFingerprint,
        accepted_tool_definition_ids: _acceptedToolDefinitionIds,
        accepted_execution_receipt_ids: _acceptedExecutionReceiptIds,
        accepted_context_entry_ids: _acceptedContextEntryIds,
        accepted_context_entries: _acceptedContextEntries,
        accepted_tool_selection: _acceptedToolSelection,
        ...projected
    } = receipt;
    return ConversationOutputReceiptSchema.parse(projected);
}

function projectGeneration(generation: ReturnType<typeof resolveAcceptedRecords>['generation']) {
    const {
        request_receipt: _requestReceipt,
        model_options: _modelOptions,
        context_fingerprint: _contextFingerprint,
        tool_set_fingerprint: _toolSetFingerprint,
        metadata: _metadata,
        usage,
        ...projected
    } = generation;
    const safeUsage = (() => {
        if (usage === undefined) return undefined;
        const { reported_usage: _reportedUsage, ...normalizedUsage } = usage;
        return Object.keys(normalizedUsage).length === 0 ? undefined : normalizedUsage;
    })();
    return ConversationOutputGenerationSchema.parse({ ...projected, ...(safeUsage ? { usage: safeUsage } : {}) });
}

function projectToolCall(block: Extract<GeneratedAgentTurn['blocks'][number], { type: 'tool_call' }>) {
    const { definition_id: _definitionId, native_id: _nativeId, ...projected } = block;
    return ConversationOutputToolCallBlockSchema.parse(projected);
}

/** Check an immutable accepted fragment against a later complete head without rewinding that head. */
export async function matchesRetainedAcceptedOutputFragment(
    documentInput: unknown,
    fragmentInput: unknown,
): Promise<boolean> {
    let document: ConversationDocument;
    let fragment: ConversationAcceptedOutputFragment;
    try {
        document = parseConversationDocument(documentInput);
        fragment = parseAcceptedOutputFragment(fragmentInput);
    } catch {
        return false;
    }
    const same = (left: unknown, right: unknown) =>
        canonicalJsonContentString(left) === canonicalJsonContentString(right);
    const receipt = ownRecordValue(document.operation_receipts, fragment.receipt.id);
    const generation = ownRecordValue(document.generations, fragment.generation.id);
    const turn = document.turns.find((candidate) => candidate.id === fragment.turn.id);
    if (!receipt || !generation || !turn || !isGeneratedAgentTurn(turn)) return false;
    if (document.revision === fragment.source.revision) {
        try {
            return same(createAcceptedOutputFragment(document, receipt.id), fragment);
        } catch {
            return false;
        }
    }
    if (
        document.id !== fragment.source.conversation_id ||
        document.revision < fragment.source.revision ||
        receipt.result_revision !== fragment.source.revision ||
        generation.record_source !== 'executed'
    )
        return false;
    try {
        if (
            !same(projectReceipt(receipt), fragment.receipt) ||
            !same(projectGeneration(generation), fragment.generation)
        )
            return false;
    } catch {
        return false;
    }
    const { blocks: _fragmentBlocks, ...fragmentTurn } = fragment.turn;
    const { blocks: _currentBlocks, ...currentTurn } = {
        id: turn.id,
        kind: turn.kind,
        authority: turn.authority,
        status: turn.status,
        timestamps: turn.timestamps,
        model_visibility: turn.model_visibility,
        provenance: turn.provenance,
        generation_id: turn.generation_id,
        blocks: turn.blocks,
    };
    if (!same(currentTurn, fragmentTurn)) return false;
    const included = new Set(fragment.turn.blocks.map((block) => block.id));
    const omitted = new Set(fragment.completeness.omitted_block_ids);
    if (turn.blocks.some((block) => !included.has(block.id) && !omitted.has(block.id))) return false;
    if (
        !same(
            turn.blocks.filter((block) => included.has(block.id)).map((block) => block.id),
            fragment.turn.blocks.map((block) => block.id),
        )
    )
        return false;
    const archivedReplayIds = new Set(
        turn.blocks.flatMap((block) =>
            block.type === 'tool_call' && block.arguments.type === 'externalized_json'
                ? (block.arguments.invalidated_replay_archives ?? []).map((archive) => archive.replay_block_id)
                : [],
        ),
    );
    for (const id of omitted) {
        const current = turn.blocks.find((block) => block.id === id);
        if (!current) {
            if (!archivedReplayIds.has(id)) return false;
            continue;
        }
        if (current.type === 'native_replay' || current.type === 'extension' || current.type === 'external_reference')
            continue;
        const assetId = referencedAssetId(current);
        if (
            assetId === undefined ||
            ((receipt.accepted_asset_ids ?? []).includes(assetId) &&
                safeGeneratedAsset(ownRecordValue(document.assets, assetId), generation.id))
        )
            return false;
    }
    for (const original of fragment.turn.blocks) {
        const current = turn.blocks.find((block) => block.id === original.id);
        if (!current || current.type !== original.type) return false;
        if (current.type === 'tool_call' && original.type === 'tool_call') {
            if (current.arguments.type === 'externalized_json') {
                if (original.arguments.type === 'externalized_json') {
                    if (!same(projectToolCall(current), original)) return false;
                    continue;
                }
                if (
                    original.arguments.type !== 'json' ||
                    original.arguments.value === null ||
                    typeof original.arguments.value !== 'object' ||
                    Array.isArray(original.arguments.value)
                )
                    return false;
                try {
                    await assertHistoricalToolArgumentExternalization(
                        document,
                        current,
                        original.arguments.value,
                        fragment.source.revision,
                        generation.id,
                    );
                } catch {
                    return false;
                }
                if (!same(projectToolCall({ ...current, arguments: original.arguments }), original)) return false;
            } else if (!same(projectToolCall(current), original)) return false;
        } else if (!same(current, original)) return false;
    }
    for (const [id, acceptedAsset] of Object.entries(fragment.assets)) {
        const current = safeGeneratedAsset(ownRecordValue(document.assets, id), generation.id);
        if (!current || !same(current, acceptedAsset)) return false;
    }
    return true;
}

/** Project one indexed, text-only accepted response from independently verified record bodies. */
export function createAcceptedIndexedTextOutputFragment(input: {
    source: { conversation_id: string; revision: number };
    receipt: OperationReceipt;
    turn: GeneratedAgentTurn;
    generation: ExecutedGeneration;
}): ConversationAcceptedOutputFragment {
    const { source, receipt, turn, generation } = input;
    if (
        receipt.conversation_id !== source.conversation_id ||
        receipt.result_revision !== source.revision ||
        receipt.base_revision + 1 !== receipt.result_revision ||
        receipt.accepted_turn_ids?.length !== 1 ||
        receipt.accepted_turn_ids[0] !== turn.id ||
        receipt.accepted_generation_ids?.length !== 1 ||
        receipt.accepted_generation_ids[0] !== generation.id ||
        (receipt.accepted_asset_ids?.length ?? 0) !== 0 ||
        turn.generation_id !== generation.id ||
        generation.record_source !== 'executed' ||
        generation.source.conversation_id !== source.conversation_id ||
        generation.source.revision !== receipt.base_revision ||
        turn.blocks.some((block) => block.type !== 'text' && block.type !== 'native_replay')
    ) {
        throw new ConversationOutputProjectionError(
            'receipt_mismatch',
            'Indexed text output differs from its accepted response receipt',
        );
    }
    return parseAcceptedOutputFragment({
        format: CONVERSATION_ACCEPTED_OUTPUT_FORMAT,
        schema_version: CONVERSATION_SCHEMA_VERSION,
        experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
        source,
        receipt: projectReceipt(receipt),
        turn: ConversationOutputTurnSchema.parse({
            id: turn.id,
            kind: turn.kind,
            authority: turn.authority,
            status: turn.status,
            timestamps: turn.timestamps,
            model_visibility: turn.model_visibility,
            provenance: turn.provenance,
            generation_id: turn.generation_id,
            blocks: turn.blocks.filter((block) => block.type === 'text'),
        }),
        generation: projectGeneration(generation),
        assets: {},
        completeness: {
            history: 'omitted',
            native_replay: 'omitted',
            metadata: 'omitted',
            semantic_content: 'complete',
            omitted_block_ids: turn.blocks.filter((block) => block.type === 'native_replay').map((block) => block.id),
            omitted_asset_ids: [],
        },
    });
}

/** Output-only semantic projection from exact indexed response acceptance; native replay stays cold. */
export function createAcceptedIndexedOutputFragment(input: {
    source: { conversation_id: string; revision: number };
    receipt: OperationReceipt;
    turn: GeneratedAgentTurn;
    generation: ExecutedGeneration;
}): ConversationAcceptedOutputFragment {
    const { source, receipt, turn, generation } = input;
    if (
        receipt.conversation_id !== source.conversation_id ||
        receipt.result_revision !== source.revision ||
        receipt.base_revision + 1 !== receipt.result_revision ||
        receipt.accepted_turn_ids?.length !== 1 ||
        receipt.accepted_turn_ids[0] !== turn.id ||
        receipt.accepted_generation_ids?.length !== 1 ||
        receipt.accepted_generation_ids[0] !== generation.id ||
        (receipt.accepted_asset_ids?.length ?? 0) !== 0 ||
        turn.generation_id !== generation.id ||
        generation.record_source !== 'executed' ||
        generation.source.conversation_id !== source.conversation_id ||
        generation.source.revision !== receipt.base_revision ||
        turn.blocks.some((block) => !['text', 'json', 'tool_call', 'native_replay'].includes(block.type))
    ) {
        throw new ConversationOutputProjectionError(
            'receipt_mismatch',
            'Indexed dependent output differs from its accepted response receipt',
        );
    }
    return parseAcceptedOutputFragment({
        format: CONVERSATION_ACCEPTED_OUTPUT_FORMAT,
        schema_version: CONVERSATION_SCHEMA_VERSION,
        experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
        source,
        receipt: projectReceipt(receipt),
        turn: ConversationOutputTurnSchema.parse({
            id: turn.id,
            kind: turn.kind,
            authority: turn.authority,
            status: turn.status,
            timestamps: turn.timestamps,
            model_visibility: turn.model_visibility,
            provenance: turn.provenance,
            generation_id: turn.generation_id,
            blocks: turn.blocks.flatMap((block): ConversationOutputBlock[] =>
                block.type === 'tool_call'
                    ? [projectToolCall(block)]
                    : block.type === 'text' || block.type === 'json'
                      ? [block]
                      : [],
            ),
        }),
        generation: projectGeneration(generation),
        assets: {},
        completeness: {
            history: 'omitted',
            native_replay: 'omitted',
            metadata: 'omitted',
            semantic_content: 'complete',
            omitted_block_ids: turn.blocks.filter((block) => block.type === 'native_replay').map((block) => block.id),
            omitted_asset_ids: [],
        },
    });
}

/**
 * Select the accepted semantic output of one response operation without retaining input history,
 * provider replay payloads, or opaque metadata.
 *
 * The returned value is an explicit output fragment. It preserves canonical entity IDs and lists
 * every omitted source block; it is not resumable history and must never be presented as a complete
 * conversation document.
 */
export function createAcceptedOutputFragment(documentInput: unknown, operationIdInput: string) {
    const operationId = IdentifierSchema.safeParse(operationIdInput);
    if (!operationId.success) {
        throw new ConversationOutputProjectionError('invalid_operation_id', 'The response operation ID is invalid');
    }

    const document = parseConversationDocument(documentInput);
    const { receipt, turn, generation } = resolveAcceptedRecords(document, operationId.data);
    return createAcceptedOutputFragmentFromRecords({
        source: { conversation_id: document.id, revision: document.revision },
        receipt,
        turn,
        generation,
        assets: document.assets,
    });
}

/** Project only exact retained accepted records. An indexed host point-loads this bounded set;
 * no full conversation document or unrelated historical content is needed. */
export function createAcceptedOutputFragmentFromRecords(input: {
    source: ConversationRef;
    receipt: OperationReceipt;
    turn: GeneratedAgentTurn;
    generation: ExecutedGeneration;
    assets: Readonly<Record<string, Asset>>;
}): ConversationAcceptedOutputFragment {
    const { source, receipt, turn, generation, assets } = input;
    if (
        receipt.conversation_id !== source.conversation_id ||
        receipt.result_revision !== source.revision ||
        receipt.accepted_turn_ids?.length !== 1 ||
        receipt.accepted_turn_ids[0] !== turn.id ||
        receipt.accepted_generation_ids?.length !== 1 ||
        receipt.accepted_generation_ids[0] !== generation.id ||
        turn.generation_id !== generation.id ||
        generation.record_source !== 'executed' ||
        generation.source.conversation_id !== source.conversation_id ||
        generation.source.revision !== receipt.base_revision
    )
        throw new ConversationOutputProjectionError(
            'receipt_mismatch',
            'Selected output records differ from acceptance',
        );
    const acceptedAssetIds = new Set(receipt.accepted_asset_ids ?? []);
    const assetEntries: [string, ConversationOutputAsset][] = [];
    const blocks: z.infer<typeof ConversationOutputTurnSchema>['blocks'] = [];
    const omittedBlockIds: string[] = [];
    const omittedAssetIds = new Set(receipt.accepted_asset_ids ?? []);
    let semanticContentPartial = false;

    for (const block of turn.blocks) {
        const assetId = referencedAssetId(block);
        if (assetId !== undefined) {
            const asset = safeGeneratedAsset(ownRecordValue(assets, assetId), generation.id);
            if (!asset || !acceptedAssetIds.has(assetId)) {
                omittedBlockIds.push(block.id);
                semanticContentPartial = true;
                continue;
            }
            if (!assetEntries.some(([id]) => id === assetId)) assetEntries.push([assetId, asset]);
            omittedAssetIds.delete(assetId);
        }

        switch (block.type) {
            case 'text':
            case 'json':
            case 'image':
            case 'document':
            case 'audio':
            case 'video':
            case 'reasoning':
                blocks.push(block);
                break;
            case 'tool_call':
                {
                    const hydrationAssets: [string, ConversationOutputAsset][] = [];
                    let hydrationIsSafe = true;
                    for (const hydration of toolArgumentAssets(block)) {
                        const asset = safeGeneratedAsset(ownRecordValue(assets, hydration.asset_id), generation.id);
                        if (
                            !acceptedAssetIds.has(hydration.asset_id) ||
                            asset?.content_hash !== hydration.content_hash
                        ) {
                            hydrationIsSafe = false;
                            break;
                        }
                        hydrationAssets.push([hydration.asset_id, asset]);
                    }
                    if (!hydrationIsSafe) {
                        omittedBlockIds.push(block.id);
                        semanticContentPartial = true;
                        break;
                    }
                    for (const [assetId, asset] of hydrationAssets) {
                        if (!assetEntries.some(([id]) => id === assetId)) assetEntries.push([assetId, asset]);
                        omittedAssetIds.delete(assetId);
                    }
                    blocks.push(projectToolCall(block));
                }
                break;
            case 'native_replay':
                omittedBlockIds.push(block.id);
                break;
            case 'external_reference':
            case 'extension':
                omittedBlockIds.push(block.id);
                semanticContentPartial = true;
                break;
        }
    }

    const projectedTurn = {
        id: turn.id,
        kind: turn.kind,
        authority: turn.authority,
        status: turn.status,
        timestamps: turn.timestamps,
        model_visibility: turn.model_visibility,
        provenance: turn.provenance,
        generation_id: turn.generation_id,
    };
    const fragment = {
        format: CONVERSATION_ACCEPTED_OUTPUT_FORMAT,
        schema_version: CONVERSATION_SCHEMA_VERSION,
        experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
        source: { conversation_id: source.conversation_id, revision: receipt.result_revision },
        receipt: projectReceipt(receipt),
        turn: ConversationOutputTurnSchema.parse({ ...projectedTurn, blocks }),
        generation: projectGeneration(generation),
        assets: Object.fromEntries(assetEntries),
        completeness: {
            history: 'omitted',
            native_replay: 'omitted',
            metadata: 'omitted',
            semantic_content: semanticContentPartial ? 'partial' : 'complete',
            omitted_block_ids: omittedBlockIds,
            omitted_asset_ids: [...omittedAssetIds],
        },
    } as const;

    return parseAcceptedOutputFragment(fragment);
}
