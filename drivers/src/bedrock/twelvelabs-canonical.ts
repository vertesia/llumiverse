import type { InvokeModelCommandOutput, ServiceTierType } from '@aws-sdk/client-bedrock-runtime';
import {
    type Asset,
    appendDecodedConversationResponseWithProcessing,
    type ConversationTurn,
    createGeneratedAgentTurn,
    createProgramTurn,
    createStructuredOutputTransformationProof,
    createTextBlock,
    createUserTurn,
    type DecodedConversationResponse,
    deriveConversationId,
    fingerprintJson,
    inlineAssetContentIntegrity,
    isConversationDocumentFormat,
    type JsonObject,
    type NativeItemMapping,
    type NativeStreamPosition,
    parseConversationDocument,
    type UserContentBlock,
} from '@llumiverse/conversation';
import {
    type CanonicalExecutionContextOptions,
    type CanonicalExecutionEventStream,
    type CanonicalExecutionResponse,
    type CanonicalStreamOpenOptions,
    createCanonicalExecutionResponse,
    type ExecutionOptions,
    FallbackCanonicalExecutionEventStream,
    normalizeCanonicalStructuredOutput,
    PromptRole,
} from '@llumiverse/core';
import { canonicalNativeExecutionEventStream } from '../conversation/canonical-execution-event-stream.js';
import {
    acceptedCanonicalResponse,
    appendCanonicalPrompt,
    assertAcceptedCanonicalRequest,
    assertCanonicalContextProjection,
    type CanonicalPreparedState,
    canonicalResponseIdentities,
    createExecutedGeneration,
    createRequestReceipt,
    newCanonicalConversation,
    prepareCanonicalContext,
    providerJsonValue,
    publishCanonicalPreparedRequest,
    recoverCanonicalExecutionResponse,
    resolveConversationRuntime,
    selectedCanonicalTurns,
} from '../conversation/canonical-runtime.js';
import { rejectDecodedStructuredOutput } from '../conversation/structured-output.js';
import {
    TWELVELABS_PEGASUS_PROMPT_SOURCE,
    type TwelvelabsPegasusCanonicalPrompt,
    type TwelvelabsPegasusRequest,
    type TwelvelabsPegasusResponse,
} from './twelvelabs.js';

export const TWELVELABS_PEGASUS_PROTOCOL = 'aws.bedrock.invoke_model.twelvelabs_pegasus';
const TWELVELABS_PEGASUS_ADAPTER_VERSION = '2026-09-30.canonical.1';
const MAX_INLINE_VIDEO_BYTES = 25 * 1024 * 1024;
const MAX_NATIVE_STREAM_BYTES = 8 * 1024 * 1024;

export interface TwelvelabsPegasusInvokeRequest {
    modelId: string;
    contentType: 'application/json';
    accept: 'application/json';
    body: string;
    serviceTier?: ServiceTierType;
}

export interface TwelvelabsPegasusStreamEvent {
    chunk?: { bytes?: Uint8Array };
    internalServerException?: unknown;
    modelStreamErrorException?: unknown;
    modelTimeoutException?: unknown;
    serviceUnavailableException?: unknown;
    throttlingException?: unknown;
    validationException?: unknown;
    $unknown?: unknown;
}

export interface TwelvelabsPegasusTransport {
    invoke(request: TwelvelabsPegasusInvokeRequest, signal?: AbortSignal): Promise<InvokeModelCommandOutput>;
    stream(
        request: TwelvelabsPegasusInvokeRequest,
        signal: AbortSignal,
    ): Promise<{
        body: AsyncIterable<TwelvelabsPegasusStreamEvent>;
        provider_response_id?: string;
        service_tier?: string;
    }>;
}

interface PreparedTwelvelabsPegasus extends CanonicalPreparedState<TwelvelabsPegasusRequest> {
    request: TwelvelabsPegasusInvokeRequest;
    request_json: ReturnType<typeof providerJsonValue>;
}

interface TwelvelabsPegasusNativeResponse extends TwelvelabsPegasusResponse {
    provider_response_id?: string;
    service_tier?: string;
}

interface FinalizedTwelvelabsPegasus {
    raw_decoded: DecodedConversationResponse;
    decoded: DecodedConversationResponse;
    response: CanonicalExecutionResponse;
    normalized: ReturnType<typeof normalizeCanonicalStructuredOutput> | undefined;
}

function requestBody(prompt: TwelvelabsPegasusRequest): ReturnType<typeof providerJsonValue> {
    return providerJsonValue(prompt);
}

function invokeRequest(
    prompt: TwelvelabsPegasusRequest,
    options: ExecutionOptions,
    serviceTier: ServiceTierType | undefined,
): TwelvelabsPegasusInvokeRequest {
    return {
        modelId: options.model,
        contentType: 'application/json',
        accept: 'application/json',
        body: JSON.stringify(requestBody(prompt)),
        ...(serviceTier === undefined ? {} : { serviceTier }),
    };
}

function serviceTier(options: ExecutionOptions): ServiceTierType | undefined {
    const value = (options.model_options as { service_tier?: unknown } | undefined)?.service_tier;
    if (value === undefined) return undefined;
    if (typeof value !== 'string' || value.length === 0) {
        throw new TypeError('TwelveLabs Pegasus service_tier must be a nonempty string');
    }
    // Public model options may carry a provider value released before this SDK's string union.
    return value as ServiceTierType;
}

function targetOptions(
    region: string,
    prompt: TwelvelabsPegasusCanonicalPrompt,
    options: ExecutionOptions,
): JsonObject {
    const source = prompt[TWELVELABS_PEGASUS_PROMPT_SOURCE];
    if (source === undefined) throw new Error('TwelveLabs Pegasus prompt has no canonical source metadata');
    return providerJsonValue({
        region,
        service_tier: serviceTier(options),
        parameters: {
            temperature: prompt.temperature,
            maxOutputTokens: prompt.maxOutputTokens,
            responseFormat: prompt.responseFormat,
        },
        video: {
            segment_index: source.video.segment_index,
            mime_type: source.video.mime_type,
            source: prompt.mediaSource.s3Location === undefined ? 'inline_base64' : 'aws.s3',
        },
    }) as JsonObject;
}

async function sourceRecords(
    prompt: TwelvelabsPegasusCanonicalPrompt,
    runtime: ReturnType<typeof resolveConversationRuntime>,
) {
    const source = prompt[TWELVELABS_PEGASUS_PROMPT_SOURCE];
    if (source === undefined) throw new Error('TwelveLabs Pegasus prompt has no canonical source metadata');
    const turns: ConversationTurn[] = [];
    const assets: Asset[] = [];
    const mappings: NativeItemMapping[] = [];
    for (const segment of source.segments) {
        const turnId = await deriveConversationId('turn', runtime.input_operation_id, String(segment.index));
        const blocks: UserContentBlock[] = [];
        if (segment.content.length > 0) {
            const block = createTextBlock({
                id: await deriveConversationId('block', runtime.input_operation_id, String(segment.index), 'text'),
                text: segment.content,
                format: 'plain',
            });
            blocks.push(block);
            mappings.push({
                canonical_id: block.id,
                native_id: `body/inputPrompt/segments/${segment.index}`,
                kind: 'block',
            });
        }
        if (segment.index === source.video.segment_index) {
            const assetId = await deriveConversationId('asset', runtime.input_operation_id, 'video');
            const block = {
                id: await deriveConversationId('block', runtime.input_operation_id, String(segment.index), 'video'),
                type: 'video' as const,
                asset_id: assetId,
            };
            blocks.push(block);
            const storage: Asset['storage'] =
                prompt.mediaSource.s3Location !== undefined
                    ? {
                          type: 'external',
                          resolver: 'aws.s3',
                          locator: providerJsonValue(prompt.mediaSource.s3Location) as JsonObject,
                      }
                    : { type: 'inline_base64', data: prompt.mediaSource.base64String ?? '' };
            const integrity = await inlineAssetContentIntegrity(storage);
            if (storage.type === 'inline_base64' && integrity === undefined) {
                throw new Error('TwelveLabs Pegasus inline video integrity is unavailable');
            }
            if (integrity !== undefined && integrity.byte_length > MAX_INLINE_VIDEO_BYTES) {
                throw new Error('TwelveLabs Pegasus inline video exceeds the 25MB limit');
            }
            assets.push({
                id: assetId,
                kind: 'video',
                mime_type: source.video.mime_type,
                storage,
                provenance: { type: 'received', source_turn_id: turnId },
                ...(integrity === undefined
                    ? {}
                    : { byte_length: integrity.byte_length, content_hash: integrity.content_hash }),
                created_at: runtime.recorded_at,
                metadata: { twelvelabs_pegasus: { name: source.video.name, segment_index: segment.index } },
            });
            mappings.push({ canonical_id: block.id, native_id: 'body/mediaSource', kind: 'block' });
        }
        if (blocks.length === 0) continue;
        const common = {
            id: turnId,
            blocks,
            status: 'completed' as const,
            timestamps: { recorded_at: runtime.recorded_at },
            model_visibility: 'include' as const,
            provenance: { type: 'received' as const },
        };
        const turn =
            segment.role === PromptRole.system
                ? createProgramTurn({ ...common, authority: 'system' })
                : createUserTurn({ ...common, authority: 'ordinary' });
        turns.push(turn);
        mappings.push({ canonical_id: turn.id, native_id: `source/segments/${segment.index}`, kind: 'turn' });
    }
    return { turns, assets, mappings };
}

function pegasusAssetName(asset: Asset): string {
    const metadata = asset.metadata?.twelvelabs_pegasus;
    if (typeof metadata === 'object' && metadata !== null && !Array.isArray(metadata)) {
        const name = (metadata as Record<string, unknown>).name;
        if (typeof name === 'string' && name.length > 0) return name;
    }
    return asset.id;
}

function pegasusS3Location(asset: Asset): NonNullable<TwelvelabsPegasusRequest['mediaSource']['s3Location']> {
    if (asset.storage.type !== 'external' || asset.storage.resolver !== 'aws.s3') {
        throw new Error(`TwelveLabs Pegasus retained video ${asset.id} must use inline_base64 or aws.s3 storage`);
    }
    const uri = asset.storage.locator.uri;
    if (typeof uri !== 'string' || !uri.startsWith('s3://')) {
        throw new Error(`TwelveLabs Pegasus retained video ${asset.id} has no valid S3 URI`);
    }
    const bucketOwner = asset.storage.locator.bucketOwner;
    if (bucketOwner !== undefined && typeof bucketOwner !== 'string') {
        throw new Error(`TwelveLabs Pegasus retained video ${asset.id} has an invalid bucket owner`);
    }
    return { uri, ...(bucketOwner === undefined ? {} : { bucketOwner }) };
}

async function retainedContextPrompt(
    document: ReturnType<typeof parseConversationDocument>,
    options: CanonicalExecutionContextOptions,
): Promise<{ prompt: TwelvelabsPegasusCanonicalPrompt; mappings: NativeItemMapping[] }> {
    const modelOptions = options.model_options as Record<string, unknown> | undefined;
    const allowedOptions = new Set(['_option_id', 'temperature', 'max_tokens', 'service_tier']);
    for (const [key, value] of Object.entries(modelOptions ?? {})) {
        if (value !== undefined && !allowedOptions.has(key)) {
            throw new TypeError(`TwelveLabs Pegasus canonical execution does not support model option ${key}`);
        }
    }
    if (modelOptions?._option_id !== undefined && modelOptions._option_id !== 'bedrock-twelvelabs-pegasus') {
        throw new TypeError(`TwelveLabs Pegasus does not support option set ${String(modelOptions._option_id)}`);
    }
    const sourceSegments: Array<{ index: number; role: PromptRole.user; content: string }> = [];
    const mappings: NativeItemMapping[] = [];
    let mediaSource: TwelvelabsPegasusRequest['mediaSource'] | undefined;
    let videoSource: { segment_index: number; name: string; mime_type: string } | undefined;
    const selected = selectedCanonicalTurns(document);
    assertCanonicalContextProjection(document, selected, {
        label: 'TwelveLabs Pegasus',
        program_authorities: ['ordinary'],
    });
    for (let turnIndex = 0; turnIndex < selected.length; turnIndex += 1) {
        const turn = selected[turnIndex];
        if (turn.kind !== 'program' && turn.kind !== 'user') {
            throw new TypeError(`TwelveLabs Pegasus retained context does not support ${turn.kind} turns`);
        }
        const text: string[] = [];
        mappings.push({ canonical_id: turn.id, native_id: `source/segments/${turnIndex}`, kind: 'turn' });
        for (let blockIndex = 0; blockIndex < turn.blocks.length; blockIndex += 1) {
            const block = turn.blocks[blockIndex];
            if (block.type === 'text') {
                text.push(block.text);
                mappings.push({
                    canonical_id: block.id,
                    native_id: `body/inputPrompt/segments/${turnIndex}/blocks/${blockIndex}`,
                    kind: 'block',
                });
                continue;
            }
            if (block.type !== 'video') {
                throw new TypeError(`TwelveLabs Pegasus retained context does not support ${block.type} input`);
            }
            if (mediaSource !== undefined) {
                throw new TypeError('TwelveLabs Pegasus canonical execution requires exactly one video');
            }
            const asset = Object.hasOwn(document.assets, block.asset_id) ? document.assets[block.asset_id] : undefined;
            if (asset === undefined)
                throw new Error(`TwelveLabs Pegasus retained context is missing ${block.asset_id}`);
            if (asset.kind !== 'video') {
                throw new TypeError(`TwelveLabs Pegasus retained asset ${asset.id} is not a video`);
            }
            if (!asset.mime_type.startsWith('video/')) {
                throw new TypeError(`TwelveLabs Pegasus does not support ${asset.mime_type || 'untyped'} input files`);
            }
            if (asset.storage.type === 'inline_base64') {
                const integrity = await inlineAssetContentIntegrity(asset.storage);
                if (integrity === undefined) {
                    throw new Error('TwelveLabs Pegasus inline video integrity is unavailable');
                }
                if (integrity.byte_length > MAX_INLINE_VIDEO_BYTES) {
                    throw new Error('TwelveLabs Pegasus inline video exceeds the 25MB limit');
                }
                if (asset.content_hash !== undefined && asset.content_hash !== integrity.content_hash) {
                    throw new Error(
                        `TwelveLabs Pegasus retained video ${asset.id} content hash does not match its bytes`,
                    );
                }
                if (asset.byte_length !== undefined && asset.byte_length !== integrity.byte_length) {
                    throw new Error(
                        `TwelveLabs Pegasus retained video ${asset.id} byte length does not match its bytes`,
                    );
                }
                mediaSource = { base64String: asset.storage.data };
            } else {
                mediaSource = { s3Location: pegasusS3Location(asset) };
            }
            videoSource = {
                segment_index: turnIndex,
                name: pegasusAssetName(asset),
                mime_type: asset.mime_type,
            };
            mappings.push({ canonical_id: block.id, native_id: 'body/mediaSource', kind: 'block' });
        }
        sourceSegments.push({
            index: turnIndex,
            // Pegasus receives a single inputPrompt string and has no native system-role channel. Preflight
            // accepts only ordinary program authority, so the retained source metadata records this as the
            // user-role projection that is actually sent.
            role: PromptRole.user,
            content: text.join(''),
        });
    }
    const inputPrompt = sourceSegments
        .map((segment) => segment.content)
        .filter(Boolean)
        .join('\n')
        .trim();
    if (inputPrompt.length === 0) throw new TypeError('TwelveLabs Pegasus requires a text prompt');
    if (mediaSource === undefined || videoSource === undefined) {
        throw new TypeError('TwelveLabs Pegasus canonical execution requires exactly one video');
    }
    const prompt: TwelvelabsPegasusCanonicalPrompt = {
        inputPrompt,
        mediaSource,
        ...(typeof modelOptions?.temperature === 'number' ? { temperature: modelOptions.temperature } : {}),
        ...(typeof modelOptions?.max_tokens === 'number' ? { maxOutputTokens: modelOptions.max_tokens } : {}),
        ...(options.result_schema === undefined ? {} : { responseFormat: { jsonSchema: options.result_schema } }),
    };
    Object.defineProperty(prompt, TWELVELABS_PEGASUS_PROMPT_SOURCE, {
        configurable: false,
        enumerable: false,
        value: { segments: sourceSegments, video: videoSource },
        writable: false,
    });
    return { prompt, mappings };
}

function assertCanonicalConversationInput(options: ExecutionOptions): void {
    if (options.conversation !== undefined && !isConversationDocumentFormat(options.conversation)) {
        throw new TypeError('TwelveLabs Pegasus canonical execution does not support legacy conversation input');
    }
    if (options.conversation_runtime?.materialized_input !== undefined) {
        throw new TypeError('TwelveLabs Pegasus does not support materialized canonical input');
    }
}

function sameIds(actual: readonly string[] | undefined, expected: readonly string[]): boolean {
    return (
        actual !== undefined && actual.length === expected.length && actual.every((id, index) => id === expected[index])
    );
}

function isExactPreparedInputDocument(
    document: ReturnType<typeof parseConversationDocument>,
    inputOperationId: string,
): boolean {
    const receipt = Object.hasOwn(document.operation_receipts, inputOperationId)
        ? document.operation_receipts[inputOperationId]
        : undefined;
    return (
        receipt !== undefined &&
        receipt.base_revision === 0 &&
        receipt.result_revision === document.revision &&
        Object.keys(document.operation_receipts).length === 1 &&
        Object.keys(document.generations).length === 0 &&
        Object.keys(document.compactions).length === 0 &&
        document.lineage === undefined &&
        sameIds(
            receipt.accepted_turn_ids,
            document.turns.map((turn) => turn.id),
        ) &&
        sameIds(receipt.accepted_asset_ids, Object.keys(document.assets)) &&
        sameIds(
            receipt.accepted_context_entry_ids,
            document.context.entries.map((entry) => entry.id),
        ) &&
        sameIds(receipt.accepted_tool_definition_ids, Object.keys(document.tool_definitions)) &&
        sameIds(receipt.accepted_execution_receipt_ids, Object.keys(document.execution_receipts)) &&
        (receipt.accepted_generation_ids?.length ?? 0) === 0 &&
        document.context.active_tool_definition_ids.length === 0 &&
        document.context.protected_entry_ids.length === 0 &&
        document.context.retrieval_requirements.length === 0
    );
}

async function prepareTwelvelabsPegasusCanonical(input: {
    provider: string;
    region: string;
    prompt: TwelvelabsPegasusCanonicalPrompt;
    options: ExecutionOptions;
}): Promise<PreparedTwelvelabsPegasus> {
    assertCanonicalConversationInput(input.options);
    const runtime = resolveConversationRuntime(input.options);
    let document = isConversationDocumentFormat(input.options.conversation)
        ? parseConversationDocument(input.options.conversation)
        : newCanonicalConversation(runtime);
    if (
        input.options.conversation_runtime?.conversation_id !== undefined &&
        input.options.conversation_runtime.conversation_id !== document.id
    ) {
        throw new Error('conversation_runtime.conversation_id does not match the canonical document');
    }
    const acceptedBefore = acceptedCanonicalResponse(document, runtime.response_operation_id);
    if (
        document.turns.length > 0 &&
        acceptedBefore === undefined &&
        !isExactPreparedInputDocument(document, runtime.input_operation_id)
    ) {
        throw new Error('TwelveLabs Pegasus does not support conversation continuation');
    }
    const records = await sourceRecords(input.prompt, runtime);
    const contextEntries = await Promise.all(
        records.turns.map(async (turn, index) => ({
            id: await deriveConversationId('context', runtime.input_operation_id, String(index)),
            type: 'source_turn' as const,
            turn_id: turn.id,
        })),
    );
    const semanticPayload = requestBody(input.prompt);
    const appended = await appendCanonicalPrompt(
        document,
        {
            turns: records.turns,
            assets: records.assets,
            context_entries: contextEntries,
            item_mappings: records.mappings,
        },
        { ...runtime, conversation_id: document.id },
        undefined,
        semanticPayload,
    );
    document = appended.document;
    if (appended.tool_definitions.length > 0) {
        throw new Error('TwelveLabs Pegasus canonical execution does not support tool definitions');
    }
    const tier = serviceTier(input.options);
    const request = invokeRequest(input.prompt, input.options, tier);
    const requestJson = providerJsonValue({
        modelId: request.modelId,
        contentType: request.contentType,
        accept: request.accept,
        body: semanticPayload,
        serviceTier: request.serviceTier,
    });
    const expectedTargetOptions = targetOptions(input.region, input.prompt, input.options);
    const accepted = acceptedCanonicalResponse(document, runtime.response_operation_id);
    if (accepted !== undefined) {
        await assertAcceptedCanonicalRequest(
            { accepted_response: accepted, runtime },
            { provider: input.provider, protocol: TWELVELABS_PEGASUS_PROTOCOL, model: input.options.model },
            requestJson,
        );
        if (
            accepted.generation.adapter_version !== TWELVELABS_PEGASUS_ADAPTER_VERSION ||
            accepted.generation.request_receipt.target.adapter_version !== TWELVELABS_PEGASUS_ADAPTER_VERSION ||
            accepted.generation.request_receipt.target.options === undefined ||
            (await fingerprintJson(accepted.generation.request_receipt.target.options)) !==
                (await fingerprintJson(expectedTargetOptions))
        ) {
            throw new Error(
                `Accepted response operation ${runtime.response_operation_id} has incompatible TwelveLabs Pegasus target options`,
            );
        }
    }
    const identities =
        accepted === undefined
            ? await canonicalResponseIdentities(runtime)
            : { generation_id: accepted.generation.id, response_turn_id: accepted.turn.id };
    const receipt =
        accepted?.generation.request_receipt ??
        (await createRequestReceipt(
            document,
            { ...runtime, conversation_id: document.id },
            {
                provider: input.provider,
                protocol: TWELVELABS_PEGASUS_PROTOCOL,
                model: input.options.model,
                adapter_version: TWELVELABS_PEGASUS_ADAPTER_VERSION,
                options: expectedTargetOptions,
            },
            requestJson,
            records.mappings,
            appended.tool_definitions,
        ));
    return {
        document,
        native_conversation: input.prompt,
        receipt,
        runtime: { ...runtime, conversation_id: document.id },
        generation_id: identities.generation_id,
        response_turn_id: identities.response_turn_id,
        tool_definitions: appended.tool_definitions,
        request,
        request_json: requestJson,
        ...(accepted === undefined ? {} : { accepted_response: accepted }),
    };
}

async function prepareTwelvelabsPegasusCanonicalContext(input: {
    provider: string;
    region: string;
    options: CanonicalExecutionContextOptions;
}): Promise<PreparedTwelvelabsPegasus> {
    const prepared = await prepareCanonicalContext({
        options: input.options,
        provider: input.provider,
        protocol: TWELVELABS_PEGASUS_PROTOCOL,
        adapter_version: TWELVELABS_PEGASUS_ADAPTER_VERSION,
    });
    if (prepared.tool_definitions.length > 0) {
        throw new TypeError('TwelveLabs Pegasus does not support active canonical tool definitions');
    }
    const compiled = await retainedContextPrompt(prepared.request_document, input.options);
    const semanticPayload = requestBody(compiled.prompt);
    const tier = serviceTier(input.options);
    const request = invokeRequest(compiled.prompt, input.options, tier);
    const requestJson = providerJsonValue({
        modelId: request.modelId,
        contentType: request.contentType,
        accept: request.accept,
        body: semanticPayload,
        serviceTier: request.serviceTier,
    });
    const expectedTargetOptions = targetOptions(input.region, compiled.prompt, input.options);
    await assertAcceptedCanonicalRequest(
        prepared,
        { provider: input.provider, protocol: TWELVELABS_PEGASUS_PROTOCOL, model: input.options.model },
        requestJson,
    );
    if (prepared.accepted_response !== undefined) {
        const receipt = prepared.accepted_response.generation.request_receipt;
        if (
            receipt.target.adapter_version !== TWELVELABS_PEGASUS_ADAPTER_VERSION ||
            receipt.target.options === undefined ||
            (await fingerprintJson(receipt.target.options)) !== (await fingerprintJson(expectedTargetOptions))
        ) {
            throw new Error(
                `Accepted response operation ${prepared.runtime.response_operation_id} has incompatible TwelveLabs Pegasus target options`,
            );
        }
    }
    const receipt =
        prepared.accepted_response?.generation.request_receipt ??
        (await createRequestReceipt(
            prepared.document,
            prepared.runtime,
            {
                provider: input.provider,
                protocol: TWELVELABS_PEGASUS_PROTOCOL,
                model: input.options.model,
                adapter_version: TWELVELABS_PEGASUS_ADAPTER_VERSION,
                options: expectedTargetOptions,
            },
            requestJson,
            compiled.mappings,
            prepared.tool_definitions,
        ));
    const { request_document: _requestDocument, ...base } = prepared;
    return {
        ...base,
        native_conversation: compiled.prompt,
        receipt,
        request,
        request_json: requestJson,
    };
}

interface PegasusOutcome {
    generation_status: 'completed' | 'cancelled';
    turn_status: 'completed' | 'interrupted';
    draft_outcome: 'completed' | 'interrupted';
    block_outcome: 'native_complete' | 'interrupted';
}

function pegasusOutcome(reason: TwelvelabsPegasusResponse['finishReason']): PegasusOutcome {
    return reason === 'stop'
        ? {
              generation_status: 'completed',
              turn_status: 'completed',
              draft_outcome: 'completed',
              block_outcome: 'native_complete',
          }
        : {
              generation_status: 'cancelled',
              turn_status: 'interrupted',
              draft_outcome: 'interrupted',
              block_outcome: 'interrupted',
          };
}

function boundedNativeJson(bytes: Uint8Array, label: string): unknown {
    if (bytes.byteLength > MAX_NATIVE_STREAM_BYTES) {
        throw new Error(`TwelveLabs Pegasus ${label} exceeds the native response byte limit`);
    }
    const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
    return JSON.parse(text) as unknown;
}

function parseNativeResponse(value: unknown): TwelvelabsPegasusResponse {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        throw new Error('TwelveLabs Pegasus response is not an object');
    }
    const record = value as Record<string, unknown>;
    if (typeof record.message !== 'string' || record.message.length === 0) {
        throw new Error('TwelveLabs Pegasus response has no message');
    }
    if (record.finishReason !== 'stop' && record.finishReason !== 'length') {
        throw new Error('TwelveLabs Pegasus response has no supported terminal finish reason');
    }
    return { message: record.message, finishReason: record.finishReason };
}

function nativeResponseFromInvoke(response: InvokeModelCommandOutput): TwelvelabsPegasusNativeResponse {
    const parsed = parseNativeResponse(boundedNativeJson(response.body, 'response'));
    return {
        ...parsed,
        ...(response.$metadata.requestId === undefined ? {} : { provider_response_id: response.$metadata.requestId }),
        ...(response.serviceTier === undefined ? {} : { service_tier: response.serviceTier }),
    };
}

async function finalizeTwelvelabsPegasusCanonical(
    prepared: PreparedTwelvelabsPegasus,
    nativeResponse: TwelvelabsPegasusNativeResponse,
    options: ExecutionOptions,
): Promise<FinalizedTwelvelabsPegasus> {
    const outcome = pegasusOutcome(nativeResponse.finishReason);
    const completedAt = prepared.runtime.completed_at ?? prepared.runtime.recorded_at;
    const generation = await createExecutedGeneration({
        id: prepared.generation_id,
        runtime: { ...prepared.runtime, completed_at: completedAt },
        receipt: prepared.receipt,
        provider: prepared.receipt.target.provider,
        protocol: TWELVELABS_PEGASUS_PROTOCOL,
        adapter_version: TWELVELABS_PEGASUS_ADAPTER_VERSION,
        requested_model: options.model,
        resolved_model: prepared.request.modelId,
        ...(nativeResponse.provider_response_id === undefined
            ? {}
            : { provider_response_id: nativeResponse.provider_response_id }),
        finish_reason: nativeResponse.finishReason,
    });
    generation.status = outcome.generation_status;
    const responseBlockId = await deriveConversationId('block', prepared.runtime.response_operation_id, '0');
    const rawBlock = createTextBlock({ id: responseBlockId, text: nativeResponse.message, format: 'plain' });
    const rawTurn = createGeneratedAgentTurn({
        id: prepared.response_turn_id,
        authority: 'ordinary',
        blocks: [rawBlock],
        status: outcome.turn_status,
        timestamps: { recorded_at: completedAt, completed_at: completedAt },
        model_visibility: 'include',
        provenance: { type: 'generated' },
        generation_id: generation.id,
    });
    const rawDecoded: DecodedConversationResponse = {
        turns: [rawTurn],
        assets: [],
        generation,
        diagnostics: [],
        payload_fingerprint: await fingerprintJson(
            providerJsonValue({ message: nativeResponse.message, finishReason: nativeResponse.finishReason }),
        ),
    };
    const normalized =
        options.result_schema === undefined
            ? undefined
            : normalizeCanonicalStructuredOutput(
                  { type: 'text', source_texts: [nativeResponse.message] },
                  options.result_schema,
              );
    let decoded = rawDecoded;
    if (normalized?.status === 'valid') {
        const structuredBlockId = await deriveConversationId(
            'block',
            prepared.runtime.response_operation_id,
            'structured',
        );
        decoded = {
            ...rawDecoded,
            turns: [
                createGeneratedAgentTurn({
                    ...rawTurn,
                    blocks: [{ id: structuredBlockId, type: 'json', value: normalized.structured_output.value }],
                }),
            ],
        };
    } else if (normalized?.status === 'invalid') {
        decoded = rejectDecodedStructuredOutput(rawDecoded, normalized.error);
    }
    const finalDocument = (
        await appendDecodedConversationResponseWithProcessing(
            {
                document: prepared.document,
                generation_id: prepared.generation_id,
                response_turn_id: prepared.response_turn_id,
                receipt: prepared.receipt,
                payload: prepared.request_json,
                diagnostics: [],
            },
            decoded,
            { operation_id: prepared.runtime.response_operation_id, recorded_at: completedAt },
        )
    ).document;
    return {
        raw_decoded: rawDecoded,
        decoded,
        response: createCanonicalExecutionResponse(finalDocument, prepared.runtime.response_operation_id, {
            ...(nativeResponse.service_tier === undefined ? {} : { service_tier: nativeResponse.service_tier }),
            ...(options.include_original_response
                ? {
                      original_response: {
                          message: nativeResponse.message,
                          finishReason: nativeResponse.finishReason,
                      },
                  }
                : {}),
        }),
        normalized,
    };
}

function assertRecoverableTwelvelabsPegasus(prepared: PreparedTwelvelabsPegasus, options: ExecutionOptions): void {
    if (prepared.accepted_response !== undefined && options.include_original_response) {
        throw new Error('An idempotently recovered TwelveLabs Pegasus response cannot reconstruct original_response');
    }
}

export async function executeTwelvelabsPegasusCanonical(input: {
    provider: string;
    region: string;
    prompt: TwelvelabsPegasusCanonicalPrompt;
    options: ExecutionOptions;
    signal?: AbortSignal;
    transport: TwelvelabsPegasusTransport;
}): Promise<CanonicalExecutionResponse> {
    const prepared = await prepareTwelvelabsPegasusCanonical(input);
    return executePreparedTwelvelabsPegasus(prepared, input.options, input.signal, input.transport);
}

export async function executeTwelvelabsPegasusCanonicalContext(input: {
    provider: string;
    region: string;
    options: CanonicalExecutionContextOptions;
    signal?: AbortSignal;
    transport: TwelvelabsPegasusTransport;
}): Promise<CanonicalExecutionResponse> {
    const prepared = await prepareTwelvelabsPegasusCanonicalContext(input);
    return executePreparedTwelvelabsPegasus(prepared, input.options, input.signal, input.transport);
}

async function executePreparedTwelvelabsPegasus(
    prepared: PreparedTwelvelabsPegasus,
    options: ExecutionOptions,
    signal: AbortSignal | undefined,
    transport: TwelvelabsPegasusTransport,
): Promise<CanonicalExecutionResponse> {
    assertRecoverableTwelvelabsPegasus(prepared, options);
    if (prepared.accepted_response !== undefined) return recoverCanonicalExecutionResponse(prepared, options);
    await publishCanonicalPreparedRequest(prepared, options);
    signal?.throwIfAborted();
    const response = await transport.invoke(prepared.request, signal);
    signal?.throwIfAborted();
    return (await finalizeTwelvelabsPegasusCanonical(prepared, nativeResponseFromInvoke(response), options)).response;
}

export interface PegasusStreamChunk {
    fragment: string;
    finish_reason?: 'stop' | 'length';
}

const STREAM_ERROR_KEYS = [
    'internalServerException',
    'modelStreamErrorException',
    'modelTimeoutException',
    'serviceUnavailableException',
    'throttlingException',
    'validationException',
] as const;

export class TwelvelabsPegasusNativeStreamAccumulator {
    private text = '';
    private terminal: 'stop' | 'length' | undefined;
    private nativeBytes = 0;

    accept(event: TwelvelabsPegasusStreamEvent): PegasusStreamChunk {
        if (this.terminal !== undefined) {
            throw new Error('TwelveLabs Pegasus stream emitted content after its terminal event');
        }
        for (const key of STREAM_ERROR_KEYS) {
            if (event[key] !== undefined) {
                throw new Error(`TwelveLabs Pegasus stream reported ${key}`);
            }
        }
        if (event.$unknown !== undefined) throw new Error('TwelveLabs Pegasus stream reported an unknown event');
        const bytes = event.chunk?.bytes;
        if (bytes === undefined) throw new Error('TwelveLabs Pegasus stream event has no payload bytes');
        this.nativeBytes += bytes.byteLength;
        if (this.nativeBytes > MAX_NATIVE_STREAM_BYTES) {
            throw new Error('TwelveLabs Pegasus stream exceeds the native response byte limit');
        }
        const parsed = boundedNativeJson(bytes, 'stream event');
        if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
            throw new Error('TwelveLabs Pegasus stream event is not an object');
        }
        const record = parsed as Record<string, unknown>;
        const delta = record.delta;
        const message = record.message;
        const finishReason = record.finishReason;
        if (delta !== undefined && typeof delta !== 'string') {
            throw new Error('TwelveLabs Pegasus stream delta is not text');
        }
        if (message !== undefined && typeof message !== 'string') {
            throw new Error('TwelveLabs Pegasus stream message is not text');
        }
        if (finishReason !== undefined && finishReason !== 'stop' && finishReason !== 'length') {
            throw new Error('TwelveLabs Pegasus stream has an unsupported finish reason');
        }
        let fragment = '';
        if (finishReason === undefined) {
            fragment = (delta as string | undefined) ?? (message as string | undefined) ?? '';
        } else if (delta !== undefined) {
            fragment = delta as string;
            const afterDelta = this.text + fragment;
            if (message !== undefined && message !== afterDelta) {
                throw new Error('TwelveLabs Pegasus terminal message diverges from streamed text');
            }
        } else if (message !== undefined) {
            if (this.text.length === 0) fragment = message;
            else if (message === this.text) fragment = '';
            else if (message.startsWith(this.text)) fragment = message.slice(this.text.length);
            else throw new Error('TwelveLabs Pegasus terminal message diverges from streamed text');
        } else if (this.text.length === 0) {
            throw new Error('TwelveLabs Pegasus terminal event has no message');
        }
        this.text += fragment;
        if (finishReason !== undefined) this.terminal = finishReason;
        return { fragment, ...(finishReason === undefined ? {} : { finish_reason: finishReason }) };
    }

    response(input: { provider_response_id?: string; service_tier?: string } = {}): TwelvelabsPegasusNativeResponse {
        if (this.terminal === undefined) {
            throw new Error('TwelveLabs Pegasus stream ended without a terminal finish reason');
        }
        if (this.text.length === 0) throw new Error('TwelveLabs Pegasus stream ended without a response message');
        return {
            message: this.text,
            finishReason: this.terminal,
            ...input,
        };
    }
}

function streamPosition(): NativeStreamPosition {
    return { protocol: TWELVELABS_PEGASUS_PROTOCOL, path: ['body', 'message'] };
}

export async function streamTwelvelabsPegasusCanonicalEvents(input: {
    provider: string;
    region: string;
    prompt: TwelvelabsPegasusCanonicalPrompt;
    options: ExecutionOptions;
    signal: AbortSignal | undefined;
    open: CanonicalStreamOpenOptions;
    transport: TwelvelabsPegasusTransport;
}): Promise<CanonicalExecutionEventStream> {
    const prepared = await prepareTwelvelabsPegasusCanonical(input);
    return streamPreparedTwelvelabsPegasusEvents(prepared, input.options, input.signal, input.open, input.transport);
}

export async function streamTwelvelabsPegasusCanonicalContextEvents(input: {
    provider: string;
    region: string;
    options: CanonicalExecutionContextOptions;
    signal: AbortSignal | undefined;
    open: CanonicalStreamOpenOptions;
    transport: TwelvelabsPegasusTransport;
}): Promise<CanonicalExecutionEventStream> {
    const prepared = await prepareTwelvelabsPegasusCanonicalContext(input);
    return streamPreparedTwelvelabsPegasusEvents(prepared, input.options, input.signal, input.open, input.transport);
}

async function streamPreparedTwelvelabsPegasusEvents(
    prepared: PreparedTwelvelabsPegasus,
    options: ExecutionOptions,
    signal: AbortSignal | undefined,
    open: CanonicalStreamOpenOptions,
    transport: TwelvelabsPegasusTransport,
): Promise<CanonicalExecutionEventStream> {
    assertRecoverableTwelvelabsPegasus(prepared, options);
    const accepted = prepared.accepted_response;
    const identity = {
        request_id: accepted?.generation.request_id ?? prepared.runtime.request_id,
        attempt_id: accepted?.generation.attempt_id ?? prepared.runtime.attempt_id,
        response_operation_id: prepared.runtime.response_operation_id,
        generation_id: prepared.generation_id,
        draft_turn_id: prepared.response_turn_id,
    };
    if (accepted !== undefined) {
        return new FallbackCanonicalExecutionEventStream(
            identity,
            () => recoverCanonicalExecutionResponse(prepared, options),
            { ...open, origin: 'accepted_recovery' },
        );
    }
    const abortController = new AbortController();
    const forwardAbort = () => abortController.abort(signal?.reason);
    const accumulator = new TwelvelabsPegasusNativeStreamAccumulator();
    const position = streamPosition();
    const draftBlockId = `${prepared.response_turn_id}:twelvelabs:text`;
    let draftStarted = false;
    let responseMetadata: { provider_response_id?: string; service_tier?: string } = {};
    const eventStream = canonicalNativeExecutionEventStream({
        identity,
        open,
        openSource: async () => {
            const opened = await transport.stream(prepared.request, abortController.signal);
            responseMetadata = {
                provider_response_id: opened.provider_response_id,
                service_tier: opened.service_tier,
            };
            return opened.body;
        },
        map: async (event, writer) => {
            const chunk = accumulator.accept(event);
            if (chunk.fragment.length === 0) return;
            if (!draftStarted) {
                draftStarted = true;
                await writer.startBlock({
                    draft_block_id: draftBlockId,
                    native_position: position,
                    block: { type: 'text' },
                });
            }
            await writer.text({ draft_block_id: draftBlockId, native_position: position, text: chunk.fragment });
        },
        finalize: async () => {
            const nativeResponse = accumulator.response(responseMetadata);
            const finalized = await finalizeTwelvelabsPegasusCanonical(prepared, nativeResponse, options);
            return {
                decoded: finalized.decoded,
                response: finalized.response,
                prepare_reconciliation: async () => {
                    const rawBlock = finalized.raw_decoded.turns[0]?.blocks[0];
                    const committedBlock = finalized.decoded.turns[0]?.blocks[0];
                    if (rawBlock?.type !== 'text' || committedBlock === undefined) {
                        throw new Error('TwelveLabs Pegasus stream finalization has no semantic response block');
                    }
                    const transformations = [];
                    const reconciliations = [];
                    if (draftStarted && finalized.normalized?.status === 'valid') {
                        if (committedBlock.type !== 'json') {
                            throw new Error('TwelveLabs Pegasus structured finalization has no JSON result');
                        }
                        const proof = await createStructuredOutputTransformationProof({
                            id: `${prepared.generation_id}:structured-output`,
                            source_blocks: [rawBlock],
                            result_block: committedBlock,
                        });
                        transformations.push(proof);
                        reconciliations.push({
                            draft_block_ids: [draftBlockId],
                            native_positions: [position],
                            committed_block_ids: [committedBlock.id],
                            disposition: 'structured_output' as const,
                            transformation_id: proof.id,
                        });
                    } else if (draftStarted) {
                        reconciliations.push({
                            draft_block_ids: [draftBlockId],
                            native_positions: [position],
                            committed_block_ids: [committedBlock.id],
                            disposition: 'direct' as const,
                        });
                    }
                    const decoded: DecodedConversationResponse = {
                        ...finalized.decoded,
                        stream_evidence: {
                            item_mappings: draftStarted
                                ? [{ canonical_id: rawBlock.id, native_position: position, kind: 'block' as const }]
                                : [],
                            transformations,
                        },
                    };
                    const outcome = pegasusOutcome(nativeResponse.finishReason);
                    return {
                        decoded,
                        reconciliations,
                        deliver_final_events: async (writer) => {
                            if (draftStarted) {
                                await writer.finishBlock({
                                    draft_block_id: draftBlockId,
                                    native_position: position,
                                    outcome: outcome.block_outcome,
                                });
                            }
                            await writer.finish({
                                outcome: outcome.draft_outcome,
                                finish_reason: nativeResponse.finishReason,
                            });
                        },
                    };
                },
                ...(finalized.normalized?.status === 'valid' && options.result_schema !== undefined
                    ? { result_schema: options.result_schema }
                    : {}),
            };
        },
        abort: () => abortController.abort(),
        close: () => signal?.removeEventListener('abort', forwardAbort),
    });
    await publishCanonicalPreparedRequest(prepared, options);
    if (signal?.aborted) forwardAbort();
    else signal?.addEventListener('abort', forwardAbort, { once: true });
    return eventStream;
}
