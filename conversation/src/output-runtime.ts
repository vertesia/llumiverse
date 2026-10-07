import { canonicalJsonContentString } from './json-content-runtime.js';
import type {
    ConversationAcceptedOutputFragment,
    ConversationOutputBlock,
    ConversationOutputReceipt,
} from './output.js';
import {
    CONVERSATION_EXPERIMENTAL_REVISION,
    CONVERSATION_FORMAT,
    CONVERSATION_SCHEMA_VERSION,
} from './runtime-constants.js';
import { validateConversationSemantics } from './semantic-validation.js';
import type { ConversationDocument } from './types.js';

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

/** Compare complete accepted-output receipts without schema imports or object key-order sensitivity. */
export function conversationOutputReceiptsEqual(
    first: ConversationOutputReceipt,
    second: ConversationOutputReceipt,
): boolean {
    return canonicalJsonContentString(first) === canonicalJsonContentString(second);
}

function ownRecordValue<T>(record: Record<string, T>, id: string): T | undefined {
    return Object.hasOwn(record, id) ? record[id] : undefined;
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

function referencedOutputAssetId(block: ConversationOutputBlock): string | undefined {
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

/**
 * Validates cross-record semantics for a fragment whose JSON shape was already checked by the
 * versioned wire boundary. This function deliberately performs no schema parsing.
 */
export function assertAcceptedOutputFragmentSemantics(fragment: ConversationAcceptedOutputFragment): void {
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
        const assetId = referencedOutputAssetId(block);
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

/**
 * Clones a structurally validated wire fragment and verifies its canonical cross-record semantics.
 * Use parseAcceptedOutputFragment for unknown or otherwise untrusted values.
 */
export function cloneSemanticallyValidAcceptedOutputFragment(
    fragment: ConversationAcceptedOutputFragment,
): ConversationAcceptedOutputFragment {
    const clone = structuredClone(fragment);
    assertAcceptedOutputFragmentSemantics(clone);
    return clone;
}
