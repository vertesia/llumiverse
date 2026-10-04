import type { ExecutionOptions, ToolDefinition as LegacyToolDefinition } from '@llumiverse/common';
import {
    CanonicalProjectedRequestMeasurementSchema,
    deriveCanonicalProjectedMeasurementIdentity,
} from '@llumiverse/common/schemas';
import {
    type AppendConversationRecordsOptions,
    type AppendConversationRecordsResult,
    type AppendConversationRecordsWithProcessingResult,
    type Asset,
    acceptsProcessingMeasurement,
    adoptConversationPreparedRequestRecord,
    appendConversationRecordsWithProcessing,
    appendDecodedConversationResponse,
    appendDecodedConversationResponseWithProcessing,
    assertAcceptedResponseMatchesPreparedRecord,
    assertProcessingReady,
    type ContentBlock,
    type ContextEntry,
    type ConversationAcceptedOutputFragment,
    type ConversationDocument,
    type ConversationPreparedRequest,
    ConversationRuntimeContextSchema,
    type ConversationTurn,
    createConversationDocument,
    type DecodedConversationResponse,
    deriveConversationId,
    type ExecutedGeneration,
    type ExecutionReceipt,
    fingerprintJson,
    type GeneratedAgentTurn,
    type GenerationUsage,
    inlineAssetContentIntegrity,
    isConversationDocumentFormat,
    isGeneratedAgentTurn,
    type JsonObject,
    type JsonValue,
    type NativeItemMapping,
    type PreparedConversationRequest,
    ProcessingReadinessError,
    parseConversationDocument,
    parseConversationPreparedRequest,
    preflightJsonInput,
    processingContextFingerprint,
    type RequestReceipt,
    type ResolvedConversationRuntimeContext,
    type ToolDefinition,
    validateToolExecutionResult,
} from '@llumiverse/conversation';
import { ConversationToolExecutionResultSchema } from '@llumiverse/conversation/schemas';
import {
    assertDecodedCanonicalToolSelection,
    CanonicalAcceptedOutputRecovered,
    type CanonicalExecutionContextOptions,
    type CanonicalExecutionResponse,
    type CanonicalToolSelectionPolicy,
    canonicalRetainedPreparedRequest,
    canonicalToolDefinitions,
    canonicalToolSelectionPolicy,
    createCanonicalExecutionResponse,
    markCanonicalAcceptedRecovery,
    markCanonicalHostCallbackFailure,
    parseCanonicalToolSelectionPolicy,
    resolveCanonicalExecutionContextOptions,
} from '@llumiverse/core';

export type { ResolvedConversationRuntimeContext } from '@llumiverse/conversation';
export { canonicalToolDefinitions } from '@llumiverse/core';

type CanonicalRecoveredOutput = Awaited<ReturnType<NonNullable<ExecutionOptions['load_recovered_canonical_output']>>>;

/** Project the host's optional DEBUG history wrapper to the exact accepted fragment adapters consume. */
export function canonicalRecoveredOutputFragment(
    recovered: CanonicalRecoveredOutput,
): ConversationAcceptedOutputFragment | undefined {
    return recovered === undefined ? undefined : 'accepted_output' in recovered ? recovered.accepted_output : recovered;
}

export interface CanonicalPromptRecords {
    turns: ConversationTurn[];
    assets: Asset[];
    context_entries: ContextEntry[];
    item_mappings: NativeItemMapping[];
    execution_receipts?: ExecutionReceipt[];
}

export interface CanonicalPreparedStateBase {
    document: ConversationDocument;
    receipt: RequestReceipt;
    runtime: ResolvedConversationRuntimeContext;
    generation_id: string;
    response_turn_id: string;
    tool_definitions: ToolDefinition[];
    response_selection_policy?: CanonicalToolSelectionPolicy;
    accepted_response?: {
        turn: GeneratedAgentTurn;
        generation: ExecutedGeneration;
    };
}

export interface CanonicalPreparedState<NativeConversation> extends CanonicalPreparedStateBase {
    native_conversation: NativeConversation;
}

export interface CanonicalContextPreparation extends Omit<CanonicalPreparedStateBase, 'receipt'> {
    request_document: ConversationDocument;
}

export const CANONICAL_TOOL_SELECTION_TARGET_OPTION = 'canonical_tool_selection';

/** Add the reserved response-selection marker without overwriting provider-specific target options. */
export function canonicalToolSelectionTargetOptions(
    targetOptions: JsonObject | undefined,
    policy: CanonicalToolSelectionPolicy | undefined,
): JsonObject | undefined {
    if (policy === undefined) return targetOptions;
    if (targetOptions !== undefined && Object.hasOwn(targetOptions, CANONICAL_TOOL_SELECTION_TARGET_OPTION)) {
        throw new TypeError(`Target options reserve ${CANONICAL_TOOL_SELECTION_TARGET_OPTION}`);
    }
    return {
        ...targetOptions,
        [CANONICAL_TOOL_SELECTION_TARGET_OPTION]: policy,
    };
}

function sameToolSelectionPolicy(first: CanonicalToolSelectionPolicy, second: CanonicalToolSelectionPolicy): boolean {
    return (
        first.mode === second.mode &&
        (first.mode !== 'required' || second.mode !== 'required' || first.tool_name === second.tool_name)
    );
}

function assertAcceptedToolSelectionPolicy(
    receipt: RequestReceipt,
    current: CanonicalToolSelectionPolicy | undefined,
): void {
    const targetOptions = receipt.target.options;
    if (targetOptions === undefined || !Object.hasOwn(targetOptions, CANONICAL_TOOL_SELECTION_TARGET_OPTION)) return;
    const retained = parseCanonicalToolSelectionPolicy(targetOptions[CANONICAL_TOOL_SELECTION_TARGET_OPTION]);
    if (current === undefined || !sameToolSelectionPolicy(retained, current)) {
        throw new Error('Accepted response operation has incompatible canonical tool-selection policy');
    }
}

type AcceptedCanonicalResponse = NonNullable<CanonicalPreparedStateBase['accepted_response']>;

function selectedAssetIds(turns: readonly ConversationTurn[]): Set<string> {
    const assetIds = new Set<string>();
    for (const turn of turns) {
        for (const block of turn.blocks) {
            if (block.type === 'tool_call' && block.arguments.type === 'externalized_json') {
                for (const hydration of block.arguments.hydration) assetIds.add(hydration.asset_id);
            }
            if ('asset_id' in block) assetIds.add(block.asset_id);
            if (block.type === 'tool_result') {
                for (const nested of block.content) {
                    if ('asset_id' in nested) assetIds.add(nested.asset_id);
                }
            }
        }
    }
    return assetIds;
}

export async function requestContextFingerprint(
    document: CanonicalTurnSelectionSource & Pick<ConversationDocument, 'assets'>,
    protocol: string,
): Promise<{ fingerprint: string; assets: Asset[] }> {
    const selected = selectedCanonicalTurns(document, { allow_interrupted_with_replay_protocol: protocol });
    const assets = [...selectedAssetIds(selected)].sort().map((id) => {
        const asset = Object.hasOwn(document.assets, id) ? document.assets[id] : undefined;
        if (asset === undefined) throw new Error(`Selected context references missing asset ${id}`);
        return asset;
    });
    // A caller may supply a persisted document directly, so declared integrity is not proof
    // of the selected inline bytes. Check both new requests and accepted-response recovery.
    // External assets are the host resolver's responsibility; this boundary performs no I/O.
    for (const asset of assets) {
        const actual = await inlineAssetContentIntegrity(asset.storage);
        if (actual === undefined) continue;
        if (asset.content_hash !== undefined && asset.content_hash !== actual.content_hash) {
            throw new Error(`Selected inline asset ${asset.id} content hash does not match its bytes`);
        }
        if (asset.byte_length !== undefined && asset.byte_length !== actual.byte_length) {
            throw new Error(`Selected inline asset ${asset.id} byte length does not match its bytes`);
        }
    }
    return {
        fingerprint: await fingerprintJson({ turns: selected, assets }),
        assets,
    };
}

/** Reconstruct and verify the exact canonical context used by an already accepted provider request. */
export async function acceptedCanonicalRequestDocument(
    document: ConversationDocument,
    accepted: AcceptedCanonicalResponse,
): Promise<ConversationDocument> {
    const receipt = accepted.generation.request_receipt;
    const requestedTurnIds = new Set(
        receipt.item_mappings.flatMap((mapping) => (mapping.kind === 'turn' ? [mapping.canonical_id] : [])),
    );
    const entries = document.context.entries.filter((entry) => requestedTurnIds.has(entry.turn_id));
    const retainedTurnIds = new Set(entries.map((entry) => entry.turn_id));
    const missingTurnId = [...requestedTurnIds].find((id) => !retainedTurnIds.has(id));
    if (missingTurnId !== undefined) {
        throw new Error(`Accepted request context cannot resolve retained turn ${missingTurnId}`);
    }
    const entryIds = new Set(entries.map((entry) => entry.id));
    const candidate: ConversationDocument = {
        ...document,
        context: {
            ...document.context,
            revision: receipt.source.revision,
            entries,
            active_tool_definition_ids: [...receipt.tool_definition_ids],
            protected_entry_ids: document.context.protected_entry_ids.filter((id) => entryIds.has(id)),
        },
    };
    const { fingerprint } = await requestContextFingerprint(candidate, receipt.target.protocol);
    if (fingerprint !== receipt.context_fingerprint) {
        throw new Error(`Accepted response operation context does not match its retained request receipt`);
    }
    return candidate;
}

/** Verify a rebuilt native request before returning an accepted response without provider transport. */
export async function assertAcceptedCanonicalRequest(
    state: Pick<CanonicalPreparedStateBase, 'accepted_response' | 'response_selection_policy' | 'runtime'>,
    target: { provider: string; protocol: string; model: string },
    nativePayload: JsonValue,
): Promise<void> {
    const accepted = state.accepted_response;
    if (accepted === undefined) return;
    const generation = accepted.generation;
    const receipt = generation.request_receipt;
    assertAcceptedToolSelectionPolicy(receipt, state.response_selection_policy);
    if (
        generation.request_id !== state.runtime.request_id ||
        generation.provider !== target.provider ||
        generation.protocol !== target.protocol ||
        generation.requested_model !== target.model ||
        receipt.target.provider !== target.provider ||
        receipt.target.protocol !== target.protocol ||
        receipt.target.model !== target.model ||
        receipt.request_fingerprint !== (await fingerprintJson(nativePayload))
    ) {
        throw new Error(
            `Accepted response operation ${state.runtime.response_operation_id} has incompatible request identity`,
        );
    }
}

/** Validate response selection before appending any canonical response records or acceptance receipt. */
export function appendCanonicalDecodedResponse<NativePayload>(
    prepared: PreparedConversationRequest<NativePayload> &
        Pick<CanonicalPreparedStateBase, 'response_selection_policy'>,
    decoded: DecodedConversationResponse,
    options: Omit<AppendConversationRecordsOptions, 'expected_revision' | 'payload_fingerprint'>,
): AppendConversationRecordsResult {
    assertDecodedCanonicalToolSelection(decoded, prepared.response_selection_policy);
    return appendDecodedConversationResponse(prepared, decoded, options);
}

/** Preserve response-selection validation while staging the response and processing jobs together. */
export async function appendCanonicalDecodedResponseWithProcessing<NativePayload>(
    prepared: PreparedConversationRequest<NativePayload> &
        Pick<CanonicalPreparedStateBase, 'response_selection_policy'>,
    decoded: DecodedConversationResponse,
    options: Omit<AppendConversationRecordsOptions, 'expected_revision' | 'payload_fingerprint'>,
): Promise<AppendConversationRecordsWithProcessingResult> {
    assertDecodedCanonicalToolSelection(decoded, prepared.response_selection_policy);
    return appendDecodedConversationResponseWithProcessing(prepared, decoded, options);
}

/** Await the host durability barrier for an exact finalized provider request. */
export async function publishCanonicalPreparedRequest(
    state: CanonicalPreparedStateBase,
    options: ExecutionOptions,
    projection?: import('@llumiverse/common').CanonicalProjectedRequestMeasurement,
): Promise<ConversationPreparedRequest | undefined> {
    let ownedProjection = projection;
    if (state.document.processing.enabled && projection !== undefined) {
        try {
            if (!preflightJsonInput(projection, { max_bytes: 64 * 1024 }).success)
                throw new RangeError('Canonical processing projection exceeds its bounded envelope');
            ownedProjection = CanonicalProjectedRequestMeasurementSchema.parse(structuredClone(projection));
        } catch (error: unknown) {
            throw markCanonicalHostCallbackFailure(error);
        }
    }
    const prepared = await parseConversationPreparedRequest({
        document: state.document,
        record: {
            source: state.receipt.source,
            runtime: state.runtime,
            request_receipt: state.receipt,
            generation_id: state.generation_id,
            response_turn_id: state.response_turn_id,
        },
    });
    let recovered: CanonicalRecoveredOutput;
    let accepted = prepared;
    try {
        if (!prepared.document.processing.enabled) {
            // Disabled policy still drains accepted jobs. This shared branch ignores measurement/target
            // fingerprints; empty arguments assert no count authority and require no projection.
            await assertProcessingReady(prepared.document, '', '');
        }
        if (prepared.document.processing.enabled) {
            if (ownedProjection?.readiness === undefined)
                throw new ProcessingReadinessError(
                    'PROCESSING_PENDING',
                    'Current input requires appendConversationRecordsWithProcessing and fresh host processing readiness',
                );
            const readiness = ownedProjection.readiness;
            const owned = ownedProjection;
            const identity = await deriveCanonicalProjectedMeasurementIdentity(
                owned,
                prepared.record.request_receipt.target,
            );
            if (
                owned.measurement.input_tokens > readiness.context_limit - readiness.output_reserve_tokens ||
                owned.measurement.source_fingerprint !== (await processingContextFingerprint(prepared.document)) ||
                (await fingerprintJson(owned.measurement)) !==
                    (await fingerprintJson(prepared.record.request_receipt.measurement)) ||
                (prepared.document.processing.budget !== undefined &&
                    !acceptsProcessingMeasurement(prepared.document.processing.budget, owned.measurement))
            )
                throw new ProcessingReadinessError(
                    'PROCESSING_BLOCKED',
                    'Current processing measurement conflicts with prepared source or policy',
                );
            await assertProcessingReady(
                prepared.document,
                await fingerprintJson(prepared.record.request_receipt.target),
                identity,
            );
        }

        const returned = await options.on_canonical_request_prepared?.(
            structuredClone(prepared),
            ownedProjection === undefined ? undefined : structuredClone(ownedProjection),
        );
        if (returned !== undefined) {
            accepted = await adoptConversationPreparedRequestRecord(prepared, returned);
        }
        state.receipt = accepted.record.request_receipt;
        recovered = await options.load_recovered_canonical_output?.({
            conversation_id: accepted.document.id,
            response_operation_id: accepted.record.runtime.response_operation_id,
            prepared_request: structuredClone(accepted.record),
        });
    } catch (error: unknown) {
        throw markCanonicalHostCallbackFailure(error);
    }
    if (recovered !== undefined) {
        throw new CanonicalAcceptedOutputRecovered(
            'accepted_output' in recovered ? recovered : { accepted_output: recovered },
        );
    }
    return accepted;
}

/** Recover one already accepted response, optionally using a verified host-retained output fragment. */
export async function recoverCanonicalExecutionResponse(
    state: Pick<CanonicalPreparedStateBase, 'document' | 'runtime' | 'accepted_response'>,
    options: ExecutionOptions,
    metadata: Parameters<typeof createCanonicalExecutionResponse>[2] = {},
): Promise<CanonicalExecutionResponse> {
    if (state.accepted_response === undefined) throw new Error('No accepted canonical response is available');
    let recoveredOutput: CanonicalRecoveredOutput;
    try {
        recoveredOutput = await options.load_recovered_canonical_output?.({
            conversation_id: state.document.id,
            response_operation_id: state.runtime.response_operation_id,
        });
    } catch (error: unknown) {
        throw markCanonicalHostCallbackFailure(error);
    }
    return markCanonicalAcceptedRecovery(
        createCanonicalExecutionResponse(
            state.document,
            state.runtime.response_operation_id,
            metadata,
            canonicalRecoveredOutputFragment(recoveredOutput),
        ),
    );
}

function randomIdentity(prefix: string): string {
    return `${prefix}_${globalThis.crypto.randomUUID()}`;
}

export function resolveConversationRuntime(options: ExecutionOptions): ResolvedConversationRuntimeContext {
    const supplied = options.conversation_runtime;
    if (supplied !== undefined) {
        const parsed = ConversationRuntimeContextSchema.parse(supplied);
        return {
            ...parsed,
            conversation_id: parsed.conversation_id ?? randomIdentity('conversation'),
            purpose: parsed.purpose ?? 'conversation',
        };
    }

    const recordedAt = new Date().toISOString();
    return {
        conversation_id: randomIdentity('conversation'),
        request_id: randomIdentity('request'),
        attempt_id: randomIdentity('attempt'),
        input_operation_id: randomIdentity('input'),
        response_operation_id: randomIdentity('response'),
        recorded_at: recordedAt,
        started_at: recordedAt,
        purpose: 'conversation',
    };
}

export function newCanonicalConversation(runtime: ResolvedConversationRuntimeContext): ConversationDocument {
    return createConversationDocument({
        id: runtime.conversation_id,
        created_at: runtime.recorded_at,
    });
}

export function parseCanonicalConversation(input: unknown): ConversationDocument | undefined {
    return isConversationDocumentFormat(input) ? parseConversationDocument(input) : undefined;
}

/** Preserve legacy retention age while canonical generations take over the interaction counter. */
export function canonicalConversationTurnNumber(document: ConversationDocument): number {
    let importedTurnNumber = 0;
    for (const turn of document.turns) {
        if (turn.provenance.type === 'imported') {
            importedTurnNumber = Math.max(importedTurnNumber, turn.provenance.source_history_turn_number ?? 0);
        }
    }
    const executedGenerations = Object.values(document.generations).filter(
        (generation) => generation.record_source === 'executed',
    ).length;
    const total = importedTurnNumber + executedGenerations;
    return Number.isSafeInteger(total) ? total : Number.MAX_SAFE_INTEGER;
}

/** Only the selected turn lookup is needed here; a sparse source is not a ConversationDocument. */
export interface CanonicalTurnSelectionSource {
    turns: readonly ConversationTurn[];
    context: { entries: readonly ContextEntry[] };
    compactions: Record<string, { replacement_turns: readonly ConversationTurn[] }>;
}

export function selectedCanonicalTurns(
    document: CanonicalTurnSelectionSource,
    options?: {
        allow_interrupted_with_replay_protocol?: string;
        allow_interrupted_with_complete_tool_calls?: boolean;
    },
): ConversationTurn[] {
    const turnsById = new Map(document.turns.map((turn) => [turn.id, turn]));
    const selected: ConversationTurn[] = [];
    for (const entry of document.context.entries) {
        const turn =
            entry.type === 'source_turn'
                ? turnsById.get(entry.turn_id)
                : document.compactions[entry.compaction_id]?.replacement_turns.find(
                      (candidate) => candidate.id === entry.turn_id,
                  );
        if (turn === undefined) {
            throw new Error(
                entry.type === 'replacement_turn'
                    ? `Conversation context entry ${entry.id} references missing replacement turn ${entry.turn_id} in compaction ${entry.compaction_id}`
                    : `Conversation context entry ${entry.id} references missing turn ${entry.turn_id}`,
            );
        }
        if (turn.model_visibility === 'exclude') continue;
        const selectedBlockIds = entry.block_ids === undefined ? undefined : new Set(entry.block_ids);
        const hasSelectedInterruptedReplay =
            turn.status === 'interrupted' &&
            turn.blocks.some(
                (block) =>
                    block.type === 'native_replay' &&
                    block.protocol === options?.allow_interrupted_with_replay_protocol &&
                    (selectedBlockIds === undefined || selectedBlockIds.has(block.id)),
            );
        const hasSelectedCompleteToolCall =
            turn.status === 'interrupted' &&
            options?.allow_interrupted_with_complete_tool_calls === true &&
            turn.kind === 'agent' &&
            turn.blocks.some(
                (block) =>
                    block.type === 'tool_call' &&
                    block.executor === 'application' &&
                    block.arguments.type !== 'invalid' &&
                    (selectedBlockIds === undefined || selectedBlockIds.has(block.id)),
            );
        if (turn.status !== 'completed' && !hasSelectedInterruptedReplay && !hasSelectedCompleteToolCall) {
            throw new Error(
                `Conversation context turn ${turn.id} has status ${turn.status}, which ${'cannot be represented by a completed native history message'}`,
            );
        }
        if (entry.block_ids === undefined) {
            selected.push(turn);
            continue;
        }
        const blockIds = new Set(entry.block_ids);
        const selectedIds = new Set(turn.blocks.filter((block) => blockIds.has(block.id)).map((block) => block.id));
        const missingId = entry.block_ids.find((id) => !selectedIds.has(id));
        if (missingId !== undefined) {
            throw new Error(`Conversation context entry ${entry.id} references missing block ${missingId}`);
        }
        switch (turn.kind) {
            case 'user':
                selected.push({
                    ...turn,
                    blocks: turn.blocks.filter((block) => blockIds.has(block.id)),
                });
                break;
            case 'agent':
                selected.push({
                    ...turn,
                    blocks: turn.blocks.filter((block) => blockIds.has(block.id)),
                });
                break;
            case 'program':
                selected.push({
                    ...turn,
                    blocks: turn.blocks.filter((block) => blockIds.has(block.id)),
                });
                break;
            case 'tool': {
                const blocks = turn.blocks.filter((block) => blockIds.has(block.id));
                if (blocks.length !== 1) {
                    throw new Error(`Conversation context entry ${entry.id} cannot omit a tool result block`);
                }
                selected.push({ ...turn, blocks });
                break;
            }
        }
    }
    return selected;
}

export interface CanonicalContextProjectionPolicy {
    label: string;
    program_authorities: readonly ConversationTurn['authority'][];
    preserve_media_caption?: (
        block: Extract<ContentBlock, { type: 'image' | 'document' | 'audio' | 'video' }>,
        owner: ConversationTurn,
    ) => boolean;
}

/**
 * Reject selected semantics that a native request would otherwise lower or silently discard.
 *
 * Read-only legacy projections deliberately do not use this guard. Current canonical compilers call
 * it with their exact native authority capabilities before publishing a prepared request. Replay
 * dependencies are included because protected native payloads cannot make unsupported semantic
 * media fields safe merely by hiding their source block from the active top-level selection.
 */
export function assertCanonicalContextProjection(
    document: CanonicalTurnSelectionSource,
    selectedTurns: readonly ConversationTurn[],
    policy: CanonicalContextProjectionPolicy,
): void {
    const supportedProgramAuthorities = new Set(policy.program_authorities);
    const blocks = new Map<string, ContentBlock>();
    const blockOwners = new Map<string, ConversationTurn>();
    const indexBlock = (block: ContentBlock, turn: ConversationTurn): void => {
        blocks.set(block.id, block);
        blockOwners.set(block.id, turn);
        if (block.type === 'tool_result') {
            for (const nested of block.content) indexBlock(nested, turn);
        }
    };
    const indexedTurns = [
        ...document.turns,
        ...Object.values(document.compactions).flatMap((compaction) => compaction.replacement_turns),
    ];
    const turns = new Map(indexedTurns.map((turn) => [turn.id, turn]));
    for (const turn of indexedTurns) {
        for (const block of turn.blocks) indexBlock(block, turn);
    }

    const assertTurnAuthority = (turn: ConversationTurn): void => {
        if (turn.kind === 'program') {
            if (!supportedProgramAuthorities.has(turn.authority)) {
                throw new TypeError(
                    `${policy.label} cannot preserve program turn ${turn.id} authority ${turn.authority}`,
                );
            }
        } else if (turn.authority !== 'ordinary') {
            throw new TypeError(
                `${policy.label} cannot preserve ${turn.kind} turn ${turn.id} authority ${turn.authority}`,
            );
        }
    };
    for (const turn of selectedTurns) {
        assertTurnAuthority(turn);
    }

    const visited = new Set<string>();
    const assertBlock = (block: ContentBlock): void => {
        if (visited.has(block.id)) return;
        visited.add(block.id);
        if (block.type === 'image' || block.type === 'document' || block.type === 'audio' || block.type === 'video') {
            const owner = blockOwners.get(block.id);
            if (
                block.caption !== undefined &&
                (owner === undefined || policy.preserve_media_caption?.(block, owner) !== true)
            ) {
                throw new TypeError(`${policy.label} cannot preserve ${block.type} block ${block.id} caption`);
            }
            if (block.selection !== undefined) {
                throw new TypeError(`${policy.label} cannot preserve ${block.type} block ${block.id} selection`);
            }
            return;
        }
        if (block.type === 'tool_result') {
            for (const nested of block.content) assertBlock(nested);
            return;
        }
        if (block.type === 'native_replay') {
            for (const turnId of block.dependencies.turn_ids) {
                const dependency = turns.get(turnId);
                if (dependency !== undefined) {
                    assertTurnAuthority(dependency);
                    for (const dependencyBlock of dependency.blocks) assertBlock(dependencyBlock);
                }
            }
            for (const blockId of block.dependencies.block_ids) {
                const dependency = blocks.get(blockId);
                if (dependency !== undefined) {
                    const owner = blockOwners.get(blockId);
                    if (owner !== undefined) assertTurnAuthority(owner);
                    assertBlock(dependency);
                }
            }
        }
    };
    for (const turn of selectedTurns) {
        for (const block of turn.blocks) assertBlock(block);
    }
}

/**
 * Resolve the effective canonical tool catalog for one request.
 *
 * An explicit legacy tool array remains an exact replacement, including an empty array that clears the active set.
 * When the caller omits that compatibility input, an existing canonical document is authoritative and its active
 * definitions retain their exact identities, capabilities, and order.
 */
export async function resolveCanonicalToolDefinitions(
    document: ConversationDocument | undefined,
    tools: readonly LegacyToolDefinition[] | undefined,
): Promise<ToolDefinition[]> {
    if (tools !== undefined) return canonicalToolDefinitions(tools);
    if (document === undefined) return [];
    return document.context.active_tool_definition_ids.map((id) => {
        const definition = Object.hasOwn(document.tool_definitions, id) ? document.tool_definitions[id] : undefined;
        if (definition === undefined) {
            throw new Error(`Active canonical tool definition ${id} is missing`);
        }
        return structuredClone(definition);
    });
}

async function retainedToolDefinitionsMatch(
    document: ConversationDocument,
    toolDefinitions: readonly ToolDefinition[],
    activeIds: readonly string[],
): Promise<boolean> {
    const expectedIds = toolDefinitions.map((definition) => definition.id);
    if (activeIds.length !== expectedIds.length || activeIds.some((id, index) => id !== expectedIds[index])) {
        return false;
    }
    for (const definition of toolDefinitions) {
        const retained = Object.hasOwn(document.tool_definitions, definition.id)
            ? document.tool_definitions[definition.id]
            : undefined;
        if (
            retained === undefined ||
            (await fingerprintJson(retained as JsonValue)) !== (await fingerprintJson(definition as JsonValue))
        ) {
            return false;
        }
    }
    return true;
}

function assertMaterializedInputRecords(
    document: ConversationDocument,
    proof: NonNullable<ResolvedConversationRuntimeContext['materialized_input']>,
): void {
    const receipt = Object.hasOwn(document.operation_receipts, proof.operation_id)
        ? document.operation_receipts[proof.operation_id]
        : undefined;
    if (
        receipt === undefined ||
        receipt.id !== proof.operation_id ||
        receipt.conversation_id !== document.id ||
        receipt.result_revision !== proof.result_revision ||
        receipt.accepted_turn_ids === undefined ||
        receipt.accepted_turn_ids.length === 0 ||
        (receipt.accepted_generation_ids?.length ?? 0) > 0
    ) {
        throw new Error('Materialized canonical input does not identify an accepted input-only operation receipt');
    }
    const acceptedTurnIds = new Set(receipt.accepted_turn_ids);
    if (acceptedTurnIds.size !== receipt.accepted_turn_ids.length) {
        throw new Error('Materialized canonical input receipt contains duplicate accepted turns');
    }
    const acceptedContextEntryIds = new Set(receipt.accepted_context_entry_ids ?? []);
    if (acceptedContextEntryIds.size !== (receipt.accepted_context_entry_ids?.length ?? 0)) {
        throw new Error('Materialized canonical input receipt contains duplicate accepted context entries');
    }
    const acceptedTurns = new Map<string, ConversationTurn>();
    for (const turnId of acceptedTurnIds) {
        const turn = document.turns.find((candidate) => candidate.id === turnId);
        if (turn === undefined || isGeneratedAgentTurn(turn) || turn.status !== 'completed') {
            throw new Error(`Materialized canonical input turn ${turnId} is not a completed input turn`);
        }
        acceptedTurns.set(turnId, turn);
    }
    if (acceptedContextEntryIds.size === 0) {
        throw new Error('Materialized canonical input is not selected by an accepted context entry');
    }
    const coverage = new Map<string, number>();
    const representedTurns = new Set<string>();
    let originVisits = 0;
    const sourceTurns = new Map(document.turns.map((turn) => [turn.id, turn]));
    const replacementTurns = new Map(
        Object.values(document.compactions).flatMap((compaction) =>
            compaction.replacement_turns.map((turn) => [turn.id, { turn, compaction }] as const),
        ),
    );
    const activeEntries = new Map(document.context.entries.map((entry) => [entry.id, entry]));
    const visitedEntries = new Set<string>();
    const visitedChanges = new Set<string>();

    const acceptedCompaction = (id: string) => {
        const compaction = document.compactions[id];
        const change = compaction === undefined ? undefined : document.operation_receipts[compaction.operation_id];
        if (
            compaction === undefined ||
            change?.operation_kind !== 'context_change' ||
            change.context_change?.kind !== 'replace_with_compaction' ||
            change.context_change.source_fingerprint !== compaction.source.source_fingerprint ||
            change.payload_fingerprint !== compaction.metadata?.payload_fingerprint ||
            change.result_revision !== compaction.metadata?.applied_revision ||
            change.recorded_at !== compaction.created_at ||
            change.result_revision <= proof.result_revision ||
            change.result_revision > document.revision ||
            JSON.stringify(change.accepted_context_entry_ids) !==
                JSON.stringify(change.context_change.inserted_entry_ids)
        ) {
            throw new Error('Materialized canonical input replacement lacks its accepted compaction receipt');
        }
        return { compaction, change };
    };
    const includeOrigins = (turnId: string, blockIds: readonly string[] | undefined, path: Set<string>): void => {
        originVisits += 1;
        if (originVisits > 4096 || path.has(turnId) || path.size >= 64) {
            throw new Error('Materialized canonical input compaction lineage exceeds its acyclic bound');
        }
        const source = sourceTurns.get(turnId);
        if (source !== undefined) {
            const selected = blockIds ?? source.blocks.map((block) => block.id);
            if (selected.some((id) => !source.blocks.some((block) => block.id === id))) {
                throw new Error('Materialized canonical input selects an unknown original block');
            }
            if (acceptedTurns.has(turnId)) {
                if (selected.length > 0 || source.blocks.length === 0) representedTurns.add(turnId);
                for (const id of selected) coverage.set(id, (coverage.get(id) ?? 0) + 1);
            }
            return;
        }
        const derived = replacementTurns.get(turnId);
        if (derived === undefined || derived.turn.provenance.type !== 'derived') {
            throw new Error('Materialized canonical input has an unbound derived replacement');
        }
        if (
            blockIds !== undefined &&
            (blockIds.length !== derived.turn.blocks.length ||
                derived.turn.blocks.some((block) => !blockIds.includes(block.id)))
        ) {
            throw new Error('Materialized canonical input derived replacement is only partially selected');
        }
        const { compaction } = acceptedCompaction(derived.compaction.id);
        const provenance = derived.turn.provenance;
        if (
            provenance.derivation_id !== compaction.id ||
            provenance.source_hash !== compaction.source.source_fingerprint ||
            provenance.source_turn_ids.length === 0
        ) {
            throw new Error('Materialized canonical input replacement lost its exact compaction provenance');
        }
        if (
            compaction.source.block_ids !== undefined &&
            (provenance.source_block_ids === undefined ||
                provenance.source_block_ids.some((id) => !compaction.source.block_ids?.includes(id)))
        ) {
            throw new Error('Materialized canonical input replacement has a foreign selected source block');
        }
        if (
            provenance.source_block_ids?.some(
                (id) =>
                    !provenance.source_turn_ids.some((sourceId) => {
                        const source = sourceTurns.get(sourceId) ?? replacementTurns.get(sourceId)?.turn;
                        return source?.blocks.some((block) => block.id === id) === true;
                    }),
            )
        ) {
            throw new Error('Materialized canonical input replacement has an unknown source block');
        }
        const nextPath = new Set(path).add(turnId);
        for (const sourceId of provenance.source_turn_ids) {
            const original = sourceTurns.get(sourceId) ?? replacementTurns.get(sourceId)?.turn;
            if (original === undefined || !compaction.source.turn_ids.includes(sourceId)) {
                throw new Error('Materialized canonical input replacement has a foreign source turn');
            }
            const selected =
                provenance.source_block_ids === undefined
                    ? undefined
                    : original.blocks
                          .filter((block) => provenance.source_block_ids?.includes(block.id))
                          .map((block) => block.id);
            includeOrigins(sourceId, selected, nextPath);
        }
    };
    const visitEntry = (id: string): void => {
        if (visitedEntries.has(id)) return;
        if (visitedEntries.size >= 4096)
            throw new Error('Materialized canonical input context lineage exceeds its bound');
        visitedEntries.add(id);
        const active = activeEntries.get(id);
        if (active !== undefined) {
            if (active.type === 'replacement_turn') {
                const { change } = acceptedCompaction(active.compaction_id);
                if (!change.context_change?.inserted_entry_ids.includes(id)) {
                    throw new Error('Materialized canonical input replacement entry was not accepted');
                }
            }
            includeOrigins(active.turn_id, active.block_ids, new Set());
            return;
        }
        const changes = Object.values(document.operation_receipts).filter((candidate) =>
            candidate.context_change?.removed_entry_ids.includes(id),
        );
        if (changes.length !== 1 || visitedChanges.size >= 64) {
            throw new Error('Materialized canonical input is not selected by exactly one accepted context lineage');
        }
        const change = changes[0];
        const compactions = Object.values(document.compactions).filter(
            (candidate) => candidate.operation_id === change.id,
        );
        if (compactions.length !== 1) {
            throw new Error('Materialized canonical input removed entry has no unique accepted compaction');
        }
        acceptedCompaction(compactions[0].id);
        visitedChanges.add(change.id);
        for (const insertedId of change.context_change?.inserted_entry_ids ?? []) visitEntry(insertedId);
    };
    for (const id of acceptedContextEntryIds) {
        const active = activeEntries.get(id);
        if (active !== undefined && (active.type !== 'source_turn' || !acceptedTurnIds.has(active.turn_id))) {
            throw new Error('Materialized canonical input context entry does not select an accepted input turn');
        }
        if (
            Object.values(document.operation_receipts).some(
                (candidate) =>
                    candidate.result_revision > proof.result_revision &&
                    candidate.accepted_context_entry_ids?.includes(id),
            )
        ) {
            throw new Error('Materialized canonical input receipt nominates a later derived context entry');
        }
        visitEntry(id);
    }
    const originMemo = new Map<string, boolean>();
    let derivedOriginVisits = 0;
    const hasAcceptedOrigin = (turnId: string, path: Set<string>): boolean => {
        if (acceptedTurnIds.has(turnId)) return true;
        if (sourceTurns.has(turnId)) return false;
        if (path.has(turnId) || path.size >= 64)
            throw new Error('Materialized canonical input lineage is cyclic or unbounded');
        const cached = originMemo.get(turnId);
        if (cached !== undefined) return cached;
        derivedOriginVisits += 1;
        if (derivedOriginVisits > 4096)
            throw new Error('Materialized canonical input derived origin search exceeds its node bound');
        const turn = replacementTurns.get(turnId)?.turn;
        const result =
            turn?.provenance.type === 'derived' &&
            turn.provenance.source_turn_ids.some((sourceId) => hasAcceptedOrigin(sourceId, new Set(path).add(turnId)));
        originMemo.set(turnId, result);
        return result;
    };
    for (const entry of document.context.entries) {
        if (!visitedEntries.has(entry.id) && hasAcceptedOrigin(entry.turn_id, new Set())) {
            throw new Error('Materialized canonical input is selected outside its unique accepted context lineage');
        }
    }
    for (const turn of acceptedTurns.values()) {
        if (!representedTurns.has(turn.id)) {
            throw new Error(
                `Materialized canonical input turn ${turn.id} is not selected by its accepted context lineage`,
            );
        }
        for (const block of turn.blocks) {
            if (coverage.get(block.id) !== 1) {
                throw new Error(
                    `Materialized canonical input turn ${turn.id} is only partially selected or selected more than once`,
                );
            }
        }
    }
}

/** Native retained input must preserve exact application call/result evidence, not only turn IDs. */
async function assertMaterializedToolExecutions(
    document: ConversationDocument,
    proof: NonNullable<ResolvedConversationRuntimeContext['materialized_input']>,
): Promise<void> {
    const receipt = document.operation_receipts[proof.operation_id];
    const executionIds = new Set(receipt?.accepted_execution_receipt_ids ?? []);
    const matched = new Set<string>();
    for (const turnId of receipt?.accepted_turn_ids ?? []) {
        const turn = document.turns.find((candidate) => candidate.id === turnId);
        if (turn?.kind !== 'tool') continue;
        const execution = turn.execution_id === undefined ? undefined : document.execution_receipts[turn.execution_id];
        if (execution?.executor !== 'application' || !executionIds.has(execution.id) || matched.has(execution.id)) {
            throw new Error('Materialized canonical tool input has unbound application execution evidence');
        }
        await validateToolExecutionResult(
            document,
            ConversationToolExecutionResultSchema.parse({
                source: execution.call_source,
                turn,
                execution_receipt: execution,
            }),
        );
        matched.add(execution.id);
    }
    if (matched.size !== executionIds.size) {
        throw new Error('Materialized canonical input contains unmatched application execution evidence');
    }
}

async function materializedToolSetFingerprint(
    proof: NonNullable<ResolvedConversationRuntimeContext['materialized_input']>,
    toolDefinitions: readonly ToolDefinition[],
): Promise<string> {
    return fingerprintJson({ materialized_input: proof, tools: toolDefinitions });
}

async function assertAcceptedMaterializedResponse(
    document: ConversationDocument,
    runtime: ResolvedConversationRuntimeContext,
    toolDefinitions: readonly ToolDefinition[],
): Promise<boolean> {
    const proof = runtime.materialized_input;
    if (proof === undefined) return false;
    const accepted = acceptedCanonicalResponse(document, runtime.response_operation_id);
    if (accepted === undefined) return false;
    const responseReceipt = document.operation_receipts[runtime.response_operation_id];
    const requestReceipt = accepted.generation.request_receipt;
    if (
        accepted.generation.request_id !== runtime.request_id ||
        requestReceipt.request_id !== runtime.request_id ||
        requestReceipt.source.conversation_id !== document.id ||
        accepted.generation.source.conversation_id !== document.id ||
        accepted.generation.source.revision !== requestReceipt.source.revision ||
        responseReceipt?.base_revision !== requestReceipt.source.revision ||
        responseReceipt.result_revision > document.revision
    ) {
        throw new Error('Accepted canonical response does not originate from the materialized request');
    }
    if (
        requestReceipt.tool_definition_ids.length !== toolDefinitions.length ||
        requestReceipt.tool_definition_ids.some((id, index) => id !== toolDefinitions[index]?.id) ||
        !(await retainedToolDefinitionsMatch(document, toolDefinitions, requestReceipt.tool_definition_ids))
    ) {
        throw new Error('Accepted canonical response tool definitions do not match the materialized request');
    }
    if (requestReceipt.source.revision === proof.result_revision) return true;

    const toolSetReceipt = Object.hasOwn(document.operation_receipts, runtime.input_operation_id)
        ? document.operation_receipts[runtime.input_operation_id]
        : undefined;
    const expectedFingerprint = await materializedToolSetFingerprint(proof, toolDefinitions);
    if (
        toolSetReceipt === undefined ||
        toolSetReceipt.conversation_id !== document.id ||
        toolSetReceipt.base_revision !== proof.result_revision ||
        toolSetReceipt.result_revision !== requestReceipt.source.revision ||
        toolSetReceipt.payload_fingerprint !== expectedFingerprint ||
        (toolSetReceipt.accepted_turn_ids?.length ?? 0) > 0 ||
        (toolSetReceipt.accepted_generation_ids?.length ?? 0) > 0 ||
        (toolSetReceipt.accepted_asset_ids?.length ?? 0) > 0 ||
        (toolSetReceipt.accepted_execution_receipt_ids?.length ?? 0) > 0 ||
        (toolSetReceipt.accepted_context_entry_ids?.length ?? 0) > 0 ||
        (toolSetReceipt.accepted_tool_definition_ids?.length ?? 0) !== toolDefinitions.length ||
        toolSetReceipt.accepted_tool_definition_ids?.some((id, index) => id !== toolDefinitions[index]?.id)
    ) {
        throw new Error('Accepted canonical response does not originate from the materialized tool-set operation');
    }
    return true;
}

export async function appendCanonicalPrompt(
    document: ConversationDocument,
    records: CanonicalPromptRecords,
    runtime: ResolvedConversationRuntimeContext,
    tools: readonly LegacyToolDefinition[] | undefined,
    semanticPayload: JsonValue,
): Promise<{ document: ConversationDocument; tool_definitions: ToolDefinition[] }> {
    const toolDefinitions = await resolveCanonicalToolDefinitions(document, tools);
    if (runtime.materialized_input !== undefined) {
        const suppliedRecords = [
            ...records.turns,
            ...records.assets,
            ...records.context_entries,
            ...records.item_mappings,
            ...(records.execution_receipts ?? []),
        ];
        if (suppliedRecords.length > 0) {
            throw new Error('A materialized canonical input cannot include new prompt records');
        }
        const proof = runtime.materialized_input;
        assertMaterializedInputRecords(document, proof);
        if (await assertAcceptedMaterializedResponse(document, runtime, toolDefinitions)) {
            return { document, tool_definitions: toolDefinitions };
        }

        const toolDefinitionsMatch = await retainedToolDefinitionsMatch(
            document,
            toolDefinitions,
            document.context.active_tool_definition_ids,
        );
        if (proof.result_revision === document.revision && toolDefinitionsMatch) {
            return { document, tool_definitions: toolDefinitions };
        }

        const payloadFingerprint = await materializedToolSetFingerprint(proof, toolDefinitions);
        const toolSetReceipt = Object.hasOwn(document.operation_receipts, runtime.input_operation_id)
            ? document.operation_receipts[runtime.input_operation_id]
            : undefined;
        if (proof.result_revision !== document.revision && toolSetReceipt === undefined) {
            throw new Error(
                `Materialized canonical input revision ${proof.result_revision} does not match current revision ${document.revision}`,
            );
        }
        const appended = await appendConversationRecordsWithProcessing(
            document,
            {
                tool_definitions: toolDefinitions,
                active_tool_definition_ids: toolDefinitions.map((definition) => definition.id),
            },
            {
                expected_revision: proof.result_revision,
                operation_id: runtime.input_operation_id,
                payload_fingerprint: payloadFingerprint,
                recorded_at: runtime.recorded_at,
            },
        );
        if (
            !(await retainedToolDefinitionsMatch(
                appended.document,
                toolDefinitions,
                appended.document.context.active_tool_definition_ids,
            ))
        ) {
            throw new Error('Materialized canonical input tool-set operation did not activate the requested tools');
        }
        if (toolSetReceipt !== undefined && toolSetReceipt.result_revision !== document.revision) {
            throw new Error('Materialized canonical input has unrelated operations after its tool-set operation');
        }
        return { document: appended.document, tool_definitions: toolDefinitions };
    }
    const payloadFingerprint = await fingerprintJson({
        prompt: semanticPayload,
        tools: toolDefinitions,
    });
    const inputReceipt = Object.hasOwn(document.operation_receipts, runtime.input_operation_id)
        ? document.operation_receipts[runtime.input_operation_id]
        : undefined;
    const historicalAcceptedRetry = inputReceipt !== undefined && inputReceipt.accepted_tool_selection === undefined;
    if (historicalAcceptedRetry) {
        const acceptedResponse = acceptedCanonicalResponse(document, runtime.response_operation_id);
        const responseReceipt = Object.hasOwn(document.operation_receipts, runtime.response_operation_id)
            ? document.operation_receipts[runtime.response_operation_id]
            : undefined;
        const acceptedRequest = acceptedResponse?.generation.request_receipt;
        const toolIds = toolDefinitions.map((tool) => tool.id);
        if (
            inputReceipt.conversation_id !== document.id ||
            inputReceipt.payload_fingerprint !== payloadFingerprint ||
            inputReceipt.operation_kind !== undefined ||
            acceptedResponse?.generation.request_id !== runtime.request_id ||
            acceptedResponse.generation.record_source !== 'executed' ||
            acceptedResponse.generation.source.conversation_id !== document.id ||
            acceptedResponse.generation.source.revision !== inputReceipt.result_revision ||
            acceptedRequest?.request_id !== runtime.request_id ||
            acceptedRequest.source.conversation_id !== document.id ||
            acceptedRequest.source.revision !== inputReceipt.result_revision ||
            responseReceipt?.conversation_id !== document.id ||
            responseReceipt.base_revision !== inputReceipt.result_revision ||
            responseReceipt.result_revision > document.revision ||
            acceptedRequest.tool_definition_ids.length !== toolIds.length ||
            acceptedRequest.tool_definition_ids.some((id, index) => id !== toolIds[index])
        ) {
            throw new Error('Historical canonical input cannot prove its exact accepted response and tool selection');
        }
    }
    const appended = await appendConversationRecordsWithProcessing(
        document,
        {
            turns: records.turns,
            assets: records.assets,
            tool_definitions: toolDefinitions,
            context_entries: records.context_entries,
            ...(records.execution_receipts === undefined ? {} : { execution_receipts: records.execution_receipts }),
            ...(historicalAcceptedRetry ? {} : { active_tool_definition_ids: toolDefinitions.map((tool) => tool.id) }),
        },
        {
            expected_revision: document.revision,
            operation_id: runtime.input_operation_id,
            payload_fingerprint: payloadFingerprint,
            recorded_at: runtime.recorded_at,
        },
    );
    if (historicalAcceptedRetry && appended.applied) {
        throw new Error('Historical canonical input recovery unexpectedly appended a new operation');
    }
    return { document: appended.document, tool_definitions: toolDefinitions };
}

/**
 * Prepare an already materialized canonical context without importing native prompt content or
 * appending an empty input operation. The document's ordered active catalog is the only tool authority.
 */
export async function prepareCanonicalContext(input: {
    options: CanonicalExecutionContextOptions;
    provider: string;
    protocol: string;
    adapter_version: string;
}): Promise<CanonicalContextPreparation> {
    // Capture native input and host-selected record before the first await. Direct adapter callers
    // receive the same ownership guarantee as the public Driver context boundary.
    const options = resolveCanonicalExecutionContextOptions(input.options);
    const document = options.conversation;
    const runtime = options.conversation_runtime;
    const retained = canonicalRetainedPreparedRequest(options);
    const model = options.model;
    const { provider, protocol, adapter_version: adapterVersion } = input;
    if (runtime.conversation_id !== document.id) {
        throw new Error('conversation_runtime.conversation_id does not match the canonical document');
    }
    const toolDefinitions = await resolveCanonicalToolDefinitions(document, undefined);
    if (runtime.materialized_input !== undefined) {
        assertMaterializedInputRecords(document, runtime.materialized_input);
        await assertMaterializedToolExecutions(document, runtime.materialized_input);
    }
    const acceptedResponse = acceptedCanonicalResponse(document, runtime.response_operation_id);
    if (retained !== undefined && acceptedResponse === undefined) {
        throw new Error('Retained canonical prepared recovery requires its accepted response operation');
    }
    if (acceptedResponse !== undefined) {
        const requestReceipt = acceptedResponse.generation.request_receipt;
        if (
            acceptedResponse.generation.request_id !== runtime.request_id ||
            acceptedResponse.generation.provider !== provider ||
            acceptedResponse.generation.protocol !== protocol ||
            acceptedResponse.generation.adapter_version !== adapterVersion ||
            acceptedResponse.generation.requested_model !== model ||
            requestReceipt.request_id !== runtime.request_id ||
            requestReceipt.source.conversation_id !== document.id
        ) {
            throw new Error(
                `Accepted response operation ${runtime.response_operation_id} has incompatible request identity`,
            );
        }
        if (retained !== undefined) {
            if (
                (await fingerprintJson(retained.runtime)) !== (await fingerprintJson(runtime)) ||
                retained.source.conversation_id !== document.id ||
                retained.source.revision < (runtime.materialized_input?.result_revision ?? 0)
            ) {
                throw new Error('Retained canonical prepared record changed its exact materialized runtime');
            }
            await assertAcceptedResponseMatchesPreparedRecord(document, retained);
            // Full receipt equality above includes source_view, native/target/context fingerprints,
            // ordered tools and assets. The adapters also recompile and compare actual native bytes
            // before their existing accepted-output recovery branch can run.
            if (
                retained.request_receipt.tool_definition_ids.length !== toolDefinitions.length ||
                retained.request_receipt.tool_definition_ids.some((id, index) => id !== toolDefinitions[index]?.id) ||
                !(await retainedToolDefinitionsMatch(
                    document,
                    toolDefinitions,
                    retained.request_receipt.tool_definition_ids,
                ))
            ) {
                throw new Error('Retained canonical prepared record changed its materialized tool definitions');
            }
        } else if (runtime.materialized_input !== undefined) {
            // Authored compatibility still requires its exact tool-set operation. No record or
            // ancestry-only assertion can silently bypass that branch.
            await assertAcceptedMaterializedResponse(document, runtime, toolDefinitions);
        }
    }
    const requestDocument =
        acceptedResponse === undefined ? document : await acceptedCanonicalRequestDocument(document, acceptedResponse);
    const identities =
        acceptedResponse === undefined
            ? await canonicalResponseIdentities(runtime)
            : { generation_id: acceptedResponse.generation.id, response_turn_id: acceptedResponse.turn.id };
    const responseSelectionPolicy = canonicalToolSelectionPolicy(options);
    return {
        document,
        request_document: requestDocument,
        runtime,
        generation_id: identities.generation_id,
        response_turn_id: identities.response_turn_id,
        tool_definitions: toolDefinitions,
        ...(responseSelectionPolicy === undefined ? {} : { response_selection_policy: responseSelectionPolicy }),
        ...(acceptedResponse === undefined ? {} : { accepted_response: acceptedResponse }),
    };
}

export async function createRequestReceipt(
    document: ConversationDocument,
    runtime: ResolvedConversationRuntimeContext,
    target: { provider: string; protocol: string; model: string; adapter_version: string; options?: JsonObject },
    nativePayload: JsonValue,
    itemMappings: readonly NativeItemMapping[],
    toolDefinitions: readonly ToolDefinition[],
): Promise<RequestReceipt> {
    return createRequestReceiptFromSelectedSource(
        {
            ...document,
            source_tail_turn_id: document.turns.at(-1)?.id,
        },
        runtime,
        target,
        nativePayload,
        itemMappings,
        toolDefinitions,
    );
}

/** The same receipt builder for an authenticated selected working set, without a sparse document cast. */
export async function createRequestReceiptFromSelectedSource(
    source: CanonicalTurnSelectionSource &
        Pick<ConversationDocument, 'id' | 'revision' | 'assets'> & { source_tail_turn_id?: string },
    runtime: ResolvedConversationRuntimeContext,
    target: { provider: string; protocol: string; model: string; adapter_version: string; options?: JsonObject },
    nativePayload: JsonValue,
    itemMappings: readonly NativeItemMapping[],
    toolDefinitions: readonly ToolDefinition[],
): Promise<RequestReceipt> {
    const selected = selectedCanonicalTurns(source, { allow_interrupted_with_replay_protocol: target.protocol });
    const turnIds = new Set(selected.map((turn) => turn.id));
    const blockIds = new Set<string>();
    const callIds = new Set<string>();
    for (const turn of selected) {
        for (const block of turn.blocks) {
            blockIds.add(block.id);
            if (block.type === 'tool_call') {
                callIds.add(block.call_id);
            }
            if (block.type === 'tool_result') {
                for (const nested of block.content) {
                    blockIds.add(nested.id);
                }
            }
        }
    }
    // Historical receipts may outlive deleted records. A freshly prepared request must still
    // resolve every mapping against the selected canonical content it actually projects.
    for (const mapping of itemMappings) {
        const ids = mapping.kind === 'turn' ? turnIds : mapping.kind === 'block' ? blockIds : callIds;
        if (!ids.has(mapping.canonical_id)) {
            throw new Error(`Request mapping references unselected ${mapping.kind} ${mapping.canonical_id}`);
        }
    }
    const { fingerprint: contextFingerprint, assets } = await requestContextFingerprint(source, target.protocol);
    const toolSetFingerprint = await fingerprintJson(toolDefinitions);
    const requestFingerprint = await fingerprintJson(nativePayload);
    return {
        id: await deriveConversationId('request_receipt', runtime.request_id, runtime.attempt_id),
        request_id: runtime.request_id,
        attempt_id: runtime.attempt_id,
        source: { conversation_id: source.id, revision: source.revision },
        ...(source.source_tail_turn_id === undefined ? {} : { source_tail_turn_id: source.source_tail_turn_id }),
        context_fingerprint: contextFingerprint,
        tool_set_fingerprint: toolSetFingerprint,
        request_fingerprint: requestFingerprint,
        target: {
            provider: target.provider,
            protocol: target.protocol,
            model: target.model,
            adapter_version: target.adapter_version,
            ...(target.options === undefined ? {} : { options: target.options }),
        },
        tool_definition_ids: toolDefinitions.map((tool) => tool.id),
        asset_versions: assets.flatMap((asset) =>
            asset.content_hash === undefined ? [] : [{ asset_id: asset.id, content_hash: asset.content_hash }],
        ),
        item_mappings: [...itemMappings],
        recorded_at: runtime.recorded_at,
    };
}

export async function canonicalResponseIdentities(runtime: ResolvedConversationRuntimeContext): Promise<{
    generation_id: string;
    response_turn_id: string;
}> {
    const generationId = await deriveConversationId('generation', runtime.request_id, runtime.attempt_id);
    return {
        generation_id: generationId,
        response_turn_id: await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
    };
}

export function acceptedCanonicalResponse(
    document: ConversationDocument,
    responseOperationId: string,
): CanonicalPreparedStateBase['accepted_response'] {
    const receipt = Object.hasOwn(document.operation_receipts, responseOperationId)
        ? document.operation_receipts[responseOperationId]
        : undefined;
    if (receipt === undefined) return undefined;
    if (receipt.accepted_turn_ids?.length !== 1 || receipt.accepted_generation_ids?.length !== 1) {
        throw new Error(`Accepted response operation ${responseOperationId} has no recoverable record identities`);
    }
    const turn = document.turns.find((candidate) => candidate.id === receipt.accepted_turn_ids?.[0]);
    const generation = document.generations[receipt.accepted_generation_ids[0]];
    if (
        turn === undefined ||
        !isGeneratedAgentTurn(turn) ||
        generation?.record_source !== 'executed' ||
        turn.generation_id !== generation.id
    ) {
        throw new Error(`Accepted response operation ${responseOperationId} cannot resolve its generated response`);
    }
    return { turn, generation };
}

export async function createExecutedGeneration(input: {
    id: string;
    runtime: ResolvedConversationRuntimeContext;
    receipt: RequestReceipt;
    provider: string;
    protocol: string;
    adapter_version: string;
    requested_model: string;
    resolved_model?: string;
    provider_response_id?: string;
    finish_reason?: string;
    usage?: GenerationUsage;
}): Promise<ExecutedGeneration> {
    const completedAt = input.runtime.completed_at ?? new Date().toISOString();
    return {
        id: input.id,
        record_source: 'executed',
        request_id: input.runtime.request_id,
        attempt_id: input.runtime.attempt_id,
        ...(input.provider_response_id === undefined ? {} : { provider_response_id: input.provider_response_id }),
        purpose: input.runtime.purpose,
        requested_model: input.requested_model,
        ...(input.resolved_model === undefined ? {} : { resolved_model: input.resolved_model }),
        provider: input.provider,
        protocol: input.protocol,
        adapter_version: input.adapter_version,
        status: 'completed',
        ...(input.finish_reason === undefined ? {} : { finish_reason: input.finish_reason }),
        timestamps: {
            recorded_at: completedAt,
            ...(input.runtime.started_at === undefined ? {} : { started_at: input.runtime.started_at }),
            completed_at: completedAt,
        },
        source: input.receipt.source,
        context_fingerprint: input.receipt.context_fingerprint,
        tool_set_fingerprint: input.receipt.tool_set_fingerprint,
        ...(input.usage === undefined ? {} : { usage: input.usage }),
        request_receipt: input.receipt,
    };
}

export function jsonClone<T extends JsonValue>(value: T): T {
    return structuredClone(value);
}

/** Project an internally-built provider payload onto its actual JSON transport representation. */
export function providerJsonValue(value: unknown): JsonValue {
    const text = JSON.stringify(value);
    if (text === undefined) throw new TypeError('Provider payload has no JSON representation');
    return JSON.parse(text) as JsonValue;
}

/** Protected native state requires recorded invocation evidence, never the newly requested target. */
export function assertProtectedReplayCompatibility(
    document: Pick<ConversationDocument, 'generations'>,
    turn: ConversationTurn,
    protocol: string,
    target?: { provider?: string; model?: string },
): void {
    for (const block of turn.blocks.flatMap((item): ContentBlock[] =>
        item.type === 'tool_result' ? [item, ...item.content] : [item],
    )) {
        if (
            block.type !== 'native_replay' ||
            block.protocol !== protocol ||
            block.dependency_policy === 'discard_on_dependency_change'
        )
            continue;
        const scope = block.compatibility_scope;
        let originModel = scope.model;
        // Previously persisted generated turns may omit the replay model but retain the exact executed receipt.
        if (originModel === undefined && isGeneratedAgentTurn(turn)) {
            const generation = document.generations[turn.generation_id];
            if (generation?.record_source === 'executed') {
                const receipt = generation.request_receipt;
                if (
                    generation.provider === scope.provider &&
                    generation.protocol === scope.protocol &&
                    generation.adapter_version === scope.adapter_version &&
                    receipt.target.provider === scope.provider &&
                    receipt.target.protocol === scope.protocol &&
                    receipt.target.adapter_version === scope.adapter_version &&
                    receipt.target.model === generation.requested_model &&
                    receipt.request_id === generation.request_id &&
                    block.dependencies.request_ids.includes(receipt.request_id)
                )
                    originModel = receipt.target.model;
            }
        }
        if (originModel === undefined) {
            throw new TypeError(
                `Protected replay block ${block.id} has unknown recorded model origin; an explicit checkpoint is required`,
            );
        }
        if (
            target?.provider === undefined ||
            target.model === undefined ||
            scope.provider !== target.provider ||
            originModel !== target.model
        ) {
            throw new TypeError(
                `Protected replay block ${block.id} is outside its compatibility scope; an explicit checkpoint is required`,
            );
        }
    }
}
