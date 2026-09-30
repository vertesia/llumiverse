import type { ExecutionOptions, ToolDefinition as LegacyToolDefinition } from '@llumiverse/common';
import {
    type Asset,
    appendConversationRecords,
    type ContextEntry,
    type ConversationAcceptedOutputFragment,
    type ConversationDocument,
    type ConversationPreparedRequest,
    ConversationRuntimeContextSchema,
    type ConversationTurn,
    createConversationDocument,
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
    parseConversationDocument,
    parseConversationPreparedRequest,
    type RequestReceipt,
    type ResolvedConversationRuntimeContext,
    type ToolDefinition,
} from '@llumiverse/conversation';
import {
    CanonicalAcceptedOutputRecovered,
    type CanonicalExecutionResponse,
    createCanonicalExecutionResponse,
    markCanonicalAcceptedRecovery,
} from '@llumiverse/core';

export type { ResolvedConversationRuntimeContext } from '@llumiverse/conversation';

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

export interface CanonicalPreparedState<NativeConversation> {
    document: ConversationDocument;
    native_conversation: NativeConversation;
    receipt: RequestReceipt;
    runtime: ResolvedConversationRuntimeContext;
    generation_id: string;
    response_turn_id: string;
    tool_definitions: ToolDefinition[];
    accepted_response?: {
        turn: GeneratedAgentTurn;
        generation: ExecutedGeneration;
    };
}

type AcceptedCanonicalResponse = NonNullable<CanonicalPreparedState<unknown>['accepted_response']>;

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

async function requestContextFingerprint(
    document: ConversationDocument,
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
    state: Pick<CanonicalPreparedState<unknown>, 'accepted_response' | 'runtime'>,
    target: { provider: string; protocol: string; model: string },
    nativePayload: JsonValue,
): Promise<void> {
    const accepted = state.accepted_response;
    if (accepted === undefined) return;
    const generation = accepted.generation;
    const receipt = generation.request_receipt;
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

/** Await the host durability barrier for an exact finalized provider request. */
export async function publishCanonicalPreparedRequest(
    state: CanonicalPreparedState<unknown>,
    options: ExecutionOptions,
): Promise<ConversationPreparedRequest | undefined> {
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
    await options.on_canonical_request_prepared?.(prepared);
    const recovered = await options.load_recovered_canonical_output?.({
        conversation_id: prepared.document.id,
        response_operation_id: prepared.record.runtime.response_operation_id,
        prepared_request: prepared.record,
    });
    if (recovered !== undefined) {
        throw new CanonicalAcceptedOutputRecovered(
            'accepted_output' in recovered ? recovered : { accepted_output: recovered },
        );
    }
    return prepared;
}

/** Recover one already accepted response, optionally using a verified host-retained output fragment. */
export async function recoverCanonicalExecutionResponse(
    state: Pick<CanonicalPreparedState<unknown>, 'document' | 'runtime' | 'accepted_response'>,
    options: ExecutionOptions,
    metadata: Parameters<typeof createCanonicalExecutionResponse>[2] = {},
): Promise<CanonicalExecutionResponse> {
    if (state.accepted_response === undefined) throw new Error('No accepted canonical response is available');
    const recoveredOutput = await options.load_recovered_canonical_output?.({
        conversation_id: state.document.id,
        response_operation_id: state.runtime.response_operation_id,
    });
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

export function selectedCanonicalTurns(
    document: ConversationDocument,
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

function jsonSchemaValue(tool: LegacyToolDefinition): JsonObject | boolean {
    return structuredClone(tool.input_schema) as JsonObject;
}

export async function canonicalToolDefinitions(
    tools: readonly LegacyToolDefinition[] | undefined,
): Promise<ToolDefinition[]> {
    const definitions: ToolDefinition[] = [];
    for (const tool of tools ?? []) {
        const versionHash = await fingerprintJson({
            name: tool.name,
            description: tool.description ?? null,
            input_schema: jsonSchemaValue(tool),
        });
        definitions.push({
            id: await deriveConversationId('tool_definition', tool.name, versionHash),
            name: tool.name,
            version: versionHash,
            ...(tool.description === undefined ? {} : { description: tool.description }),
            input_schema: jsonSchemaValue(tool),
        });
    }
    return definitions;
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
    for (const turnId of acceptedTurnIds) {
        const turn = document.turns.find((candidate) => candidate.id === turnId);
        if (turn === undefined || isGeneratedAgentTurn(turn) || turn.status !== 'completed') {
            throw new Error(`Materialized canonical input turn ${turnId} is not a completed input turn`);
        }
        const entries = document.context.entries.filter(
            (entry) => entry.type === 'source_turn' && entry.turn_id === turnId,
        );
        if (entries.length !== 1) {
            throw new Error(`Materialized canonical input turn ${turnId} is not selected by exactly one context entry`);
        }
        const [entry] = entries;
        if (entry === undefined || !acceptedContextEntryIds.has(entry.id)) {
            throw new Error(`Materialized canonical input turn ${turnId} is not selected by an accepted context entry`);
        }
        const selectedBlockIds = entry.block_ids;
        if (
            selectedBlockIds !== undefined &&
            (selectedBlockIds.length !== turn.blocks.length ||
                turn.blocks.some((block) => !selectedBlockIds.includes(block.id)))
        ) {
            throw new Error(`Materialized canonical input turn ${turnId} is only partially selected`);
        }
    }
    for (const entryId of acceptedContextEntryIds) {
        const entry = document.context.entries.find((candidate) => candidate.id === entryId);
        if (entry === undefined || entry.type !== 'source_turn' || !acceptedTurnIds.has(entry.turn_id)) {
            throw new Error(
                `Materialized canonical input context entry ${entryId} does not select an accepted input turn`,
            );
        }
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
    const toolDefinitions = await canonicalToolDefinitions(tools);
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
        const appended = appendConversationRecords(
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
    const appended = appendConversationRecords(
        document,
        {
            turns: records.turns,
            assets: records.assets,
            tool_definitions: toolDefinitions,
            context_entries: records.context_entries,
            ...(records.execution_receipts === undefined ? {} : { execution_receipts: records.execution_receipts }),
            active_tool_definition_ids: toolDefinitions.map((tool) => tool.id),
        },
        {
            expected_revision: document.revision,
            operation_id: runtime.input_operation_id,
            payload_fingerprint: payloadFingerprint,
            recorded_at: runtime.recorded_at,
        },
    );
    return { document: appended.document, tool_definitions: toolDefinitions };
}

export async function createRequestReceipt(
    document: ConversationDocument,
    runtime: ResolvedConversationRuntimeContext,
    target: { provider: string; protocol: string; model: string; adapter_version: string; options?: JsonObject },
    nativePayload: JsonValue,
    itemMappings: readonly NativeItemMapping[],
    toolDefinitions: readonly ToolDefinition[],
): Promise<RequestReceipt> {
    const selected = selectedCanonicalTurns(document, { allow_interrupted_with_replay_protocol: target.protocol });
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
    const { fingerprint: contextFingerprint, assets } = await requestContextFingerprint(document, target.protocol);
    const toolSetFingerprint = await fingerprintJson(toolDefinitions);
    const requestFingerprint = await fingerprintJson(nativePayload);
    return {
        id: await deriveConversationId('request_receipt', runtime.request_id, runtime.attempt_id),
        request_id: runtime.request_id,
        attempt_id: runtime.attempt_id,
        source: { conversation_id: document.id, revision: document.revision },
        ...(document.turns.length === 0 ? {} : { source_tail_turn_id: document.turns.at(-1)?.id }),
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
): CanonicalPreparedState<unknown>['accepted_response'] {
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
