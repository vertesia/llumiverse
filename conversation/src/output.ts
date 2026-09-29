import type { z } from 'zod';
import { ConversationValidationError } from './diagnostics.js';
import { isGeneratedAgentTurn } from './guards.js';
import { preflightJsonInput } from './json-preflight.js';
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
import {
    CONVERSATION_EXPERIMENTAL_REVISION,
    CONVERSATION_FORMAT,
    CONVERSATION_SCHEMA_VERSION,
    IdentifierSchema,
} from './schemas/primitives.js';
import { validateConversationSemantics } from './semantic-validation.js';
import type { Asset, ConversationDocument, GeneratedAgentTurn } from './types.js';
import { diagnosticsFromZodError, parseConversationDocument } from './validation.js';

export type ConversationAcceptedOutputFragment = z.infer<typeof ConversationAcceptedOutputFragmentSchema>;
export type ConversationOutputAsset = z.infer<typeof ConversationOutputAssetSchema>;
export type ConversationOutputBlock = z.infer<typeof ConversationOutputBlockSchema>;
export type ConversationOutputCompleteness = z.infer<typeof ConversationOutputCompletenessSchema>;
export type ConversationOutputGeneration = z.infer<typeof ConversationOutputGenerationSchema>;
export type ConversationOutputReceipt = z.infer<typeof ConversationOutputReceiptSchema>;
export type ConversationOutputTurn = z.infer<typeof ConversationOutputTurnSchema>;

export type ConversationOutputProjectionErrorCode =
    | 'asset_mismatch'
    | 'generation_mismatch'
    | 'invalid_fragment'
    | 'invalid_operation_id'
    | 'receipt_mismatch'
    | 'turn_mismatch';

export class ConversationOutputProjectionError extends Error {
    constructor(
        readonly code: ConversationOutputProjectionErrorCode,
        message: string,
    ) {
        super(message);
        this.name = 'ConversationOutputProjectionError';
    }
}

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

function assertUnique(values: readonly string[]): boolean {
    return new Set(values).size === values.length;
}

function invalidFragment(): never {
    throw new ConversationOutputProjectionError(
        'invalid_fragment',
        'The accepted output fragment has inconsistent canonical references',
    );
}

function assertAcceptedOutputSemantics(fragment: ConversationAcceptedOutputFragment): void {
    const { source, receipt, turn, generation, assets, completeness } = fragment;
    if (
        source.conversation_id !== receipt.conversation_id ||
        source.revision !== receipt.result_revision ||
        receipt.result_revision !== receipt.base_revision + 1 ||
        !Number.isSafeInteger(receipt.result_revision) ||
        receipt.accepted_turn_ids.length !== 1 ||
        receipt.accepted_turn_ids[0] !== turn.id ||
        receipt.accepted_generation_ids.length !== 1 ||
        receipt.accepted_generation_ids[0] !== generation.id ||
        turn.generation_id !== generation.id ||
        generation.source.conversation_id !== receipt.conversation_id ||
        generation.source.revision !== receipt.base_revision
    ) {
        invalidFragment();
    }

    const includedBlockIds = turn.blocks.map((block) => block.id);
    const omittedBlockIds = completeness.omitted_block_ids;
    const includedBlockIdSet = new Set(includedBlockIds);
    if (
        includedBlockIdSet.size !== includedBlockIds.length ||
        !assertUnique(omittedBlockIds) ||
        omittedBlockIds.some((id) => includedBlockIdSet.has(id))
    ) {
        invalidFragment();
    }

    const acceptedAssetIds = receipt.accepted_asset_ids ?? [];
    const includedAssetIds = Object.keys(assets);
    const omittedAssetIds = completeness.omitted_asset_ids;
    const acceptedAssetIdSet = new Set(acceptedAssetIds);
    const includedAssetIdSet = new Set(includedAssetIds);
    const omittedAssetIdSet = new Set(omittedAssetIds);
    if (
        acceptedAssetIdSet.size !== acceptedAssetIds.length ||
        includedAssetIdSet.size !== includedAssetIds.length ||
        omittedAssetIdSet.size !== omittedAssetIds.length ||
        omittedAssetIds.some((id) => includedAssetIdSet.has(id)) ||
        acceptedAssetIds.length !== includedAssetIds.length + omittedAssetIds.length ||
        acceptedAssetIds.some((id) => !includedAssetIdSet.has(id) && !omittedAssetIdSet.has(id)) ||
        includedAssetIds.some((id) => !acceptedAssetIdSet.has(id)) ||
        omittedAssetIds.some((id) => !acceptedAssetIdSet.has(id))
    ) {
        invalidFragment();
    }

    const referencedAssets = new Set<string>();
    for (const block of turn.blocks) {
        const assetId = referencedAssetId(block);
        if (assetId !== undefined) {
            const asset = ownRecordValue(assets, assetId);
            if (asset?.kind !== block.type) invalidFragment();
            referencedAssets.add(assetId);
        }
        if (block.type === 'tool_call' && block.arguments.type === 'externalized_json') {
            for (const hydration of block.arguments.hydration) {
                const asset = ownRecordValue(assets, hydration.asset_id);
                if (
                    asset?.kind !== 'text' ||
                    asset.mime_type !== 'text/plain' ||
                    asset.content_hash !== hydration.content_hash
                ) {
                    invalidFragment();
                }
                referencedAssets.add(hydration.asset_id);
            }
        }
    }

    for (const [key, asset] of Object.entries(assets)) {
        if (
            key !== asset.id ||
            asset.provenance.generation_id !== generation.id ||
            (asset.provenance.source_turn_id !== undefined && asset.provenance.source_turn_id !== turn.id) ||
            !referencedAssets.has(key)
        ) {
            invalidFragment();
        }
    }

    // Reuse the document semantic validator for the retained records. Synthetic request-only fields
    // let the projected executed generation participate without putting those private fields on wire.
    const usedIds = new Set<string>([
        receipt.id,
        turn.id,
        generation.id,
        ...turn.blocks.flatMap((block) => (block.type === 'tool_call' ? [block.id, block.call_id] : [block.id])),
        ...Object.keys(assets),
    ]);
    let requestReceiptId = 'accepted-output-validation:request-receipt';
    while (usedIds.has(requestReceiptId)) requestReceiptId += ':';
    const validationDocument: ConversationDocument = {
        format: CONVERSATION_FORMAT,
        schema_version: CONVERSATION_SCHEMA_VERSION,
        experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
        id: source.conversation_id,
        revision: source.revision,
        created_at: turn.timestamps.recorded_at,
        updated_at: receipt.recorded_at,
        turns: [turn],
        generations: Object.fromEntries([
            [
                generation.id,
                {
                    ...generation,
                    request_receipt: {
                        id: requestReceiptId,
                        request_id: generation.request_id,
                        attempt_id: generation.attempt_id,
                        source: generation.source,
                        context_fingerprint: 'accepted-output-validation:context',
                        tool_set_fingerprint: 'accepted-output-validation:tools',
                        request_fingerprint: 'accepted-output-validation:request',
                        target: {
                            provider: generation.provider,
                            protocol: generation.protocol,
                            model: generation.requested_model,
                            adapter_version: generation.adapter_version,
                        },
                        tool_definition_ids: [],
                        asset_versions: [],
                        item_mappings: [],
                        recorded_at: generation.timestamps.recorded_at,
                    },
                },
            ],
        ]),
        operation_receipts: Object.fromEntries([
            [receipt.id, { ...receipt, payload_fingerprint: 'accepted-output-validation:operation' }],
        ]),
        execution_receipts: {},
        assets,
        tool_definitions: {},
        context: {
            revision: source.revision,
            entries: [],
            active_tool_definition_ids: [],
            protected_entry_ids: [],
            retrieval_requirements: [],
        },
        compactions: {},
        processing: { enabled: false, policy_revision: 0, processors: [] },
    };
    if (validateConversationSemantics(validationDocument).length > 0) invalidFragment();
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
    assertAcceptedOutputSemantics(fragment);
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
        return normalizedUsage;
    })();
    return ConversationOutputGenerationSchema.parse({ ...projected, ...(safeUsage ? { usage: safeUsage } : {}) });
}

function projectToolCall(block: Extract<GeneratedAgentTurn['blocks'][number], { type: 'tool_call' }>) {
    const { definition_id: _definitionId, native_id: _nativeId, ...projected } = block;
    return ConversationOutputToolCallBlockSchema.parse(projected);
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
    const acceptedAssetIds = new Set(receipt.accepted_asset_ids ?? []);
    const assetEntries: [string, ConversationOutputAsset][] = [];
    const blocks: z.infer<typeof ConversationOutputTurnSchema>['blocks'] = [];
    const omittedBlockIds: string[] = [];
    const omittedAssetIds = new Set(receipt.accepted_asset_ids ?? []);
    let semanticContentPartial = false;

    for (const block of turn.blocks) {
        const assetId = referencedAssetId(block);
        if (assetId !== undefined) {
            const asset = safeGeneratedAsset(ownRecordValue(document.assets, assetId), generation.id);
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
                        const asset = safeGeneratedAsset(
                            ownRecordValue(document.assets, hydration.asset_id),
                            generation.id,
                        );
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
        source: { conversation_id: document.id, revision: receipt.result_revision },
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
