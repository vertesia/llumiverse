import type {
    ConversationRuntimeContext,
    ExecutionOptions,
    ToolDefinition as LegacyToolDefinition,
} from '@llumiverse/common';
import {
    type Asset,
    appendConversationRecords,
    type ContextEntry,
    type ConversationDocument,
    ConversationRuntimeContextSchema,
    type ConversationTurn,
    createConversationDocument,
    deriveConversationId,
    type ExecutedGeneration,
    type ExecutionReceipt,
    fingerprintJson,
    type GeneratedAgentTurn,
    type GenerationUsage,
    isConversationDocumentFormat,
    isGeneratedAgentTurn,
    type JsonObject,
    type JsonValue,
    type NativeItemMapping,
    parseConversationDocument,
    type RequestReceipt,
    type ToolDefinition,
} from '@llumiverse/conversation';

export interface ResolvedConversationRuntimeContext extends ConversationRuntimeContext {
    conversation_id: string;
    purpose: string;
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

export function selectedCanonicalTurns(document: ConversationDocument): ConversationTurn[] {
    const turnsById = new Map(document.turns.map((turn) => [turn.id, turn]));
    const selected: ConversationTurn[] = [];
    for (const entry of document.context.entries) {
        const turn = turnsById.get(entry.turn_id);
        if (turn === undefined) {
            throw new Error(`Conversation context entry ${entry.id} references missing turn ${entry.turn_id}`);
        }
        if (turn.model_visibility === 'exclude') continue;
        if (turn.status !== 'completed') {
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

export async function appendCanonicalPrompt(
    document: ConversationDocument,
    records: CanonicalPromptRecords,
    runtime: ResolvedConversationRuntimeContext,
    tools: readonly LegacyToolDefinition[] | undefined,
    semanticPayload: JsonValue,
): Promise<{ document: ConversationDocument; tool_definitions: ToolDefinition[] }> {
    const toolDefinitions = await canonicalToolDefinitions(tools);
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
    const contextFingerprint = await fingerprintJson(document.context);
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
        asset_versions: Object.values(document.assets).flatMap((asset) =>
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
