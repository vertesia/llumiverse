import {
    type AgentContentBlock,
    type Asset,
    appendDecodedConversationResponse,
    buildConversationTurn,
    type ConversationTurn,
    createGeneratedAgentTurn,
    createProgramTurn,
    createStructuredOutputTransformationProof,
    createTextBlock,
    createUserTurn,
    type DecodedConversationResponse,
    deriveConversationId,
    fingerprintJson,
    type GenerationUsage,
    hashContentBytes,
    isConversationDocumentFormat,
    type JsonObject,
    type JsonValue,
    type NativeItemMapping,
    type NativeStreamPosition,
    parseConversationDocument,
} from '@llumiverse/conversation';
import {
    type CanonicalExecutionEventStream,
    type CanonicalExecutionResponse,
    type CanonicalExecutionStream,
    type CanonicalStreamOpenOptions,
    type CompletionChunkObject,
    createCanonicalExecutionResponse,
    type ExecutionOptions,
    FallbackCanonicalExecutionEventStream,
    normalizeCompletionResult,
    PromptRole,
    type PromptSegment,
    type TextFallbackOptions,
} from '@llumiverse/core';
import { EventStream } from '@llumiverse/core/async';
import { EventSource } from 'eventsource';
import type { Prediction } from 'replicate';
import { canonicalNativeExecutionEventStream } from './conversation/canonical-execution-event-stream.js';
import {
    type CanonicalFinalizingDriverStream,
    canonicalExecutionStreamFromDriver,
} from './conversation/canonical-execution-stream.js';
import {
    acceptedCanonicalResponse,
    appendCanonicalPrompt,
    assertAcceptedCanonicalRequest,
    type CanonicalPreparedState,
    canonicalResponseIdentities,
    createExecutedGeneration,
    createRequestReceipt,
    newCanonicalConversation,
    providerJsonValue,
    publishCanonicalPreparedRequest,
    recoverCanonicalExecutionResponse,
    resolveConversationRuntime,
} from './conversation/canonical-runtime.js';
import { rejectDecodedStructuredOutput } from './conversation/structured-output.js';
import type { ReplicateDriver } from './replicate.js';

const REPLICATE_PREDICTIONS_PROTOCOL = 'replicate.predictions';
const REPLICATE_PREDICTIONS_ADAPTER_VERSION = '2026-09-30.canonical.1';
const MAX_REPLICATE_ASSET_BYTES = 50_000_000;
const MAX_REPLICATE_ASSET_CHUNKS = 8_192;
const POLL_INTERVAL_MS = 500;
const TERMINAL_STATUSES = new Set(['succeeded', 'failed', 'canceled', 'aborted']);

interface ReplicateRequest {
    version: string;
    input: {
        prompt: string;
        max_new_tokens?: number;
        temperature?: number;
    };
    [property: string]: unknown;
}

interface PreparedReplicateCanonical extends CanonicalPreparedState<string> {
    request: ReplicateRequest;
    request_json: JsonValue;
}

interface ReplicateSseEvent {
    type: 'output' | 'done';
    data: string;
}

interface FinalizedReplicateCanonical {
    raw_decoded: DecodedConversationResponse;
    decoded: DecodedConversationResponse;
    response: CanonicalExecutionResponse;
    normalized: ReturnType<typeof normalizeCompletionResult> | undefined;
}

function boundedDiagnostic(value: unknown): string {
    const text = value instanceof Error ? value.message : typeof value === 'string' ? value : 'unknown provider error';
    return text.length <= 160 ? text : `${text.slice(0, 157)}...`;
}

function requestPayload(prompt: string, options: ExecutionOptions): ReplicateRequest {
    const modelOptions = options.model_options as TextFallbackOptions | undefined;
    if (modelOptions?._option_id !== undefined && modelOptions._option_id !== 'text-fallback') {
        throw new TypeError(`Replicate canonical execution does not support option set ${modelOptions._option_id}`);
    }
    if (options.tools?.length) throw new TypeError('Replicate predictions do not support tools');
    const { version } = parseReplicateModel(options.model);
    return {
        version,
        input: {
            prompt,
            max_new_tokens: modelOptions?.max_tokens,
            temperature: modelOptions?.temperature,
        },
    };
}

function parseReplicateModel(modelId: string): { owner: string; model: string; version: string } {
    const slash = modelId.indexOf('/');
    const colon = modelId.lastIndexOf(':');
    if (slash <= 0 || colon <= slash + 1 || colon === modelId.length - 1) {
        throw new TypeError('Invalid Replicate model id. Expected owner/model:version');
    }
    return {
        owner: modelId.slice(0, slash),
        model: modelId.slice(slash + 1, colon),
        version: modelId.slice(colon + 1),
    };
}

export function validateReplicateCanonicalInput(segments: PromptSegment[], options: ExecutionOptions): void {
    parseReplicateModel(options.model);
    if (options.tools?.length) throw new TypeError('Replicate predictions do not support tools');
    if (options.conversation_runtime?.materialized_input !== undefined) {
        throw new TypeError('Replicate predictions do not support materialized canonical input');
    }
    for (const segment of segments) {
        if (segment.files?.length) throw new TypeError('Replicate text predictions do not support media input');
        if (
            segment.role === PromptRole.tool ||
            segment.role === PromptRole.negative ||
            segment.role === PromptRole.mask
        ) {
            throw new TypeError(`Replicate text predictions do not support ${segment.role} prompt segments`);
        }
    }
}

async function sourceRecords(
    segments: PromptSegment[],
    runtime: ReturnType<typeof resolveConversationRuntime>,
): Promise<{ turns: ConversationTurn[]; mappings: NativeItemMapping[] }> {
    const turns: ConversationTurn[] = [];
    const mappings: NativeItemMapping[] = [];
    for (let index = 0; index < segments.length; index += 1) {
        const segment = segments[index];
        if (segment.content.length === 0) continue;
        const turnId = await deriveConversationId('turn', runtime.input_operation_id, String(index));
        const block = createTextBlock({
            id: await deriveConversationId('block', runtime.input_operation_id, String(index), '0'),
            text: segment.content,
            format: 'plain',
        });
        const common = {
            id: turnId,
            blocks: [block],
            status: 'completed' as const,
            timestamps: { recorded_at: runtime.recorded_at },
            model_visibility: 'include' as const,
            provenance: { type: 'received' as const },
        };
        const turn =
            segment.role === PromptRole.system || segment.role === PromptRole.safety
                ? createProgramTurn({ ...common, authority: 'system' })
                : segment.role === PromptRole.assistant
                  ? buildConversationTurn({ ...common, kind: 'agent', authority: 'ordinary' })
                  : createUserTurn({ ...common, authority: 'ordinary' });
        turns.push(turn);
        mappings.push(
            { canonical_id: turn.id, native_id: `source/segments/${index}`, kind: 'turn' },
            { canonical_id: block.id, native_id: `source/segments/${index}/text`, kind: 'block' },
        );
    }
    return { turns, mappings };
}

async function prepareReplicateCanonical(input: {
    driver: ReplicateDriver;
    segments: PromptSegment[];
    prompt: string;
    options: ExecutionOptions;
}): Promise<PreparedReplicateCanonical> {
    const runtime = resolveConversationRuntime(input.options);
    if (runtime.materialized_input !== undefined) {
        throw new TypeError('Replicate predictions do not support materialized canonical input');
    }
    let document = isConversationDocumentFormat(input.options.conversation)
        ? parseConversationDocument(input.options.conversation)
        : newCanonicalConversation(runtime);
    if (
        input.options.conversation !== undefined &&
        input.options.conversation !== null &&
        !isConversationDocumentFormat(input.options.conversation)
    ) {
        throw new TypeError('Replicate canonical execution does not support legacy conversation input');
    }
    if (input.options.conversation_runtime?.conversation_id !== undefined && runtime.conversation_id !== document.id) {
        throw new Error('conversation_runtime.conversation_id does not match the canonical document');
    }
    const acceptedBefore = acceptedCanonicalResponse(document, runtime.response_operation_id);
    const records = await sourceRecords(input.segments, runtime);
    if (
        document.turns.length > 0 &&
        acceptedBefore === undefined &&
        (Object.keys(document.generations).length > 0 ||
            document.turns.some((turn) => !records.turns.some((record) => record.id === turn.id)))
    ) {
        throw new Error('Replicate predictions do not support conversation continuation');
    }
    const contextEntries = await Promise.all(
        records.turns.map(async (turn, index) => ({
            id: await deriveConversationId('context', runtime.input_operation_id, String(index)),
            type: 'source_turn' as const,
            turn_id: turn.id,
        })),
    );
    const appended = await appendCanonicalPrompt(
        document,
        {
            turns: records.turns,
            assets: [],
            context_entries: contextEntries,
            item_mappings: records.mappings,
        },
        { ...runtime, conversation_id: document.id },
        undefined,
        input.prompt,
    );
    document = appended.document;
    if (appended.tool_definitions.length > 0) {
        throw new TypeError('Replicate predictions do not support active tool definitions');
    }
    const request = requestPayload(input.prompt, input.options);
    const requestJson = providerJsonValue(request);
    const accepted = acceptedCanonicalResponse(document, runtime.response_operation_id);
    const model = parseReplicateModel(input.options.model);
    const targetOptions = providerJsonValue({
        owner: model.owner,
        model: model.model,
        version: model.version,
        ...(input.options.result_schema === undefined ? {} : { result_schema: input.options.result_schema }),
    }) as JsonObject;
    if (accepted !== undefined) {
        await assertAcceptedCanonicalRequest(
            { accepted_response: accepted, runtime },
            { provider: input.driver.provider, protocol: REPLICATE_PREDICTIONS_PROTOCOL, model: input.options.model },
            requestJson,
        );
        if (
            accepted.generation.adapter_version !== REPLICATE_PREDICTIONS_ADAPTER_VERSION ||
            accepted.generation.request_receipt.target.adapter_version !== REPLICATE_PREDICTIONS_ADAPTER_VERSION ||
            (await fingerprintJson(accepted.generation.request_receipt.target.options ?? {})) !==
                (await fingerprintJson(targetOptions))
        ) {
            throw new Error(
                `Accepted response operation ${runtime.response_operation_id} has incompatible Replicate target options`,
            );
        }
    }
    const identities =
        accepted === undefined
            ? await canonicalResponseIdentities(runtime)
            : {
                  generation_id: accepted.generation.id,
                  response_turn_id: accepted.turn.id,
              };
    const receipt =
        accepted?.generation.request_receipt ??
        (await createRequestReceipt(
            document,
            { ...runtime, conversation_id: document.id },
            {
                provider: input.driver.provider,
                protocol: REPLICATE_PREDICTIONS_PROTOCOL,
                model: input.options.model,
                adapter_version: REPLICATE_PREDICTIONS_ADAPTER_VERSION,
                options: targetOptions,
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

function isTerminal(prediction: Prediction): boolean {
    return TERMINAL_STATUSES.has(prediction.status);
}

function waitForPoll(signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    return new Promise((resolve, reject) => {
        const abort = () => {
            clearTimeout(timer);
            reject(signal?.reason);
        };
        const timer = setTimeout(() => {
            signal?.removeEventListener('abort', abort);
            resolve();
        }, POLL_INTERVAL_MS);
        signal?.addEventListener('abort', abort, { once: true });
    });
}

async function finalPrediction(
    driver: ReplicateDriver,
    initial: Prediction,
    signal?: AbortSignal,
): Promise<Prediction> {
    let current = initial;
    while (!isTerminal(current)) {
        signal?.throwIfAborted();
        current = await driver.service.predictions.get(initial.id, signal === undefined ? undefined : { signal });
        if (!isTerminal(current)) await waitForPoll(signal);
    }
    return current;
}

function assertSucceededPrediction(prediction: Prediction, requestedModel: string): void {
    if (prediction.status === 'failed')
        throw new Error(`Replicate prediction failed: ${boundedDiagnostic(prediction.error)}`);
    if (prediction.status === 'canceled') throw new Error('Replicate prediction was canceled');
    if (prediction.status === 'aborted') throw new Error('Replicate prediction was aborted');
    if (prediction.status !== 'succeeded')
        throw new Error(`Replicate prediction ended with unsupported status ${prediction.status}`);
    const model = parseReplicateModel(requestedModel);
    if (prediction.version !== undefined && prediction.version !== 'hidden' && prediction.version !== model.version) {
        throw new Error('Replicate prediction version does not match the requested model');
    }
    if (prediction.model !== undefined && prediction.model !== `${model.owner}/${model.model}`) {
        throw new Error('Replicate prediction model does not match the requested model');
    }
}

function trustedReplicateFileUrl(value: string): URL | undefined {
    let url: URL;
    try {
        url = new URL(value);
    } catch {
        return undefined;
    }
    if (url.protocol !== 'https:') return undefined;
    const hostname = url.hostname.toLowerCase();
    if (hostname === 'replicate.delivery' || hostname.endsWith('.replicate.delivery')) return url;
    return undefined;
}

const REPLICATE_MEDIA_MIME_TYPES = new Map<string, 'image' | 'audio' | 'video'>([
    ['image/jpeg', 'image'],
    ['image/png', 'image'],
    ['image/webp', 'image'],
    ['image/gif', 'image'],
    ['audio/wav', 'audio'],
    ['audio/x-wav', 'audio'],
    ['audio/mpeg', 'audio'],
    ['audio/ogg', 'audio'],
    ['audio/flac', 'audio'],
    ['video/mp4', 'video'],
    ['video/quicktime', 'video'],
    ['video/webm', 'video'],
    ['video/mpeg', 'video'],
]);

function mediaKind(mimeType: string): 'image' | 'audio' | 'video' | undefined {
    return REPLICATE_MEDIA_MIME_TYPES.get(mimeType);
}

function hasPrefix(bytes: Uint8Array, prefix: readonly number[], offset = 0): boolean {
    return prefix.every((value, index) => bytes[offset + index] === value);
}

function assertMediaSignature(bytes: Uint8Array, mimeType: string): void {
    const valid = (() => {
        switch (mimeType) {
            case 'image/jpeg':
                return hasPrefix(bytes, [0xff, 0xd8, 0xff]);
            case 'image/png':
                return hasPrefix(bytes, [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
            case 'image/webp':
                return hasPrefix(bytes, [0x52, 0x49, 0x46, 0x46]) && hasPrefix(bytes, [0x57, 0x45, 0x42, 0x50], 8);
            case 'image/gif':
                return (
                    hasPrefix(bytes, [0x47, 0x49, 0x46, 0x38, 0x37, 0x61]) ||
                    hasPrefix(bytes, [0x47, 0x49, 0x46, 0x38, 0x39, 0x61])
                );
            case 'audio/wav':
            case 'audio/x-wav':
                return hasPrefix(bytes, [0x52, 0x49, 0x46, 0x46]) && hasPrefix(bytes, [0x57, 0x41, 0x56, 0x45], 8);
            case 'audio/mpeg':
                return hasPrefix(bytes, [0x49, 0x44, 0x33]) || (bytes[0] === 0xff && (bytes[1] ?? 0) >= 0xe0);
            case 'audio/ogg':
                return hasPrefix(bytes, [0x4f, 0x67, 0x67, 0x53]);
            case 'audio/flac':
                return hasPrefix(bytes, [0x66, 0x4c, 0x61, 0x43]);
            case 'video/mp4':
            case 'video/quicktime':
                return hasPrefix(bytes, [0x66, 0x74, 0x79, 0x70], 4);
            case 'video/webm':
                return hasPrefix(bytes, [0x1a, 0x45, 0xdf, 0xa3]);
            case 'video/mpeg':
                return hasPrefix(bytes, [0x00, 0x00, 0x01, 0xba]) || hasPrefix(bytes, [0x00, 0x00, 0x01, 0xb3]);
            default:
                return false;
        }
    })();
    if (!valid) throw new Error(`Replicate generated asset MIME type ${mimeType} does not match its bytes`);
}

async function readReplicateAsset(response: Response, signal?: AbortSignal) {
    if (!response.ok) throw new Error(`Replicate generated asset download failed with status ${response.status}`);
    const mimeType = response.headers.get('content-type')?.split(';', 1)[0]?.trim().toLowerCase();
    if (mimeType === undefined || mediaKind(mimeType) === undefined) {
        throw new Error(`Replicate generated asset has unsupported MIME type ${mimeType ?? 'missing'}`);
    }
    const declaredLength = response.headers.get('content-length');
    let expectedLength: number | undefined;
    if (declaredLength !== null) {
        const parsed = Number(declaredLength);
        if (!Number.isSafeInteger(parsed) || parsed <= 0 || parsed > MAX_REPLICATE_ASSET_BYTES) {
            throw new Error(`Replicate generated asset exceeds the ${MAX_REPLICATE_ASSET_BYTES} byte limit`);
        }
        expectedLength = parsed;
    }
    if (response.body === null) throw new Error('Replicate generated asset response has no body');
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let byteLength = 0;
    let chunkCount = 0;
    let completed = false;
    try {
        while (true) {
            signal?.throwIfAborted();
            const item = await reader.read();
            if (item.done) {
                completed = true;
                break;
            }
            chunkCount += 1;
            if (chunkCount > MAX_REPLICATE_ASSET_CHUNKS)
                throw new Error('Replicate generated asset has too many chunks');
            byteLength += item.value.byteLength;
            if (byteLength > MAX_REPLICATE_ASSET_BYTES) {
                throw new Error(`Replicate generated asset exceeds the ${MAX_REPLICATE_ASSET_BYTES} byte limit`);
            }
            if (item.value.byteLength > 0) chunks.push(item.value);
        }
    } finally {
        if (!completed) await reader.cancel().catch(() => {});
        reader.releaseLock();
    }
    if (byteLength === 0) throw new Error('Replicate generated asset is empty');
    if (expectedLength !== undefined && expectedLength !== byteLength) {
        throw new Error('Replicate generated asset byte length does not match its response header');
    }
    const bytes = new Uint8Array(byteLength);
    let offset = 0;
    for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
    }
    assertMediaSignature(bytes, mimeType);
    return {
        bytes,
        mime_type: mimeType,
        kind: mediaKind(mimeType) as 'image' | 'audio' | 'video',
        ...(await hashContentBytes(bytes)),
    };
}

async function decodeOutput(
    driver: ReplicateDriver,
    prediction: Prediction,
    options: ExecutionOptions,
    prepared: PreparedReplicateCanonical,
    signal?: AbortSignal,
): Promise<{ blocks: AgentContentBlock[]; assets: Asset[] }> {
    const values = Array.isArray(prediction.output) ? prediction.output : [prediction.output];
    if (values.length === 0 || values.every((value) => value === undefined || value === null || value === '')) {
        throw new Error('Replicate succeeded prediction has no output');
    }
    const blocks: AgentContentBlock[] = [];
    const assets: Asset[] = [];
    let pendingText = '';
    const flushText = async () => {
        if (pendingText.length === 0) return;
        blocks.push(
            createTextBlock({
                id: await deriveConversationId('block', prepared.runtime.response_operation_id, String(blocks.length)),
                text: pendingText,
                format: 'plain',
            }),
        );
        pendingText = '';
    };
    for (let index = 0; index < values.length; index += 1) {
        const value = values[index];
        if (typeof value === 'string') {
            const fileUrl = trustedReplicateFileUrl(value);
            if (fileUrl === undefined) {
                pendingText += value;
                continue;
            }
            await flushText();
            if (options.store_generated_asset === undefined) {
                throw new Error('Replicate generated media requires a durable canonical asset sink');
            }
            const loaded = await readReplicateAsset(await driver.fetchGeneratedAsset(fileUrl, signal), signal);
            const source = new ReadableStream<Uint8Array>({
                start(controller) {
                    controller.enqueue(loaded.bytes.slice());
                    controller.close();
                },
            });
            const stored = await options.store_generated_asset(
                source,
                { kind: loaded.kind, mime_type: loaded.mime_type },
                signal,
            );
            signal?.throwIfAborted();
            if (
                stored.storage.type !== 'external' ||
                stored.byte_length !== loaded.bytes.byteLength ||
                stored.content_hash !== loaded.content_hash
            ) {
                throw new Error('Generated asset storage did not preserve the exact Replicate output bytes');
            }
            const assetId = await deriveConversationId('asset', prepared.runtime.response_operation_id, String(index));
            const blockId = await deriveConversationId(
                'block',
                prepared.runtime.response_operation_id,
                String(blocks.length),
            );
            assets.push({
                id: assetId,
                kind: loaded.kind,
                mime_type: loaded.mime_type,
                storage: stored.storage,
                provenance: { type: 'generated', generation_id: prepared.generation_id },
                byte_length: loaded.bytes.byteLength,
                content_hash: loaded.content_hash,
                created_at: prepared.runtime.completed_at ?? prepared.runtime.recorded_at,
            });
            blocks.push({ id: blockId, type: loaded.kind, asset_id: assetId });
            continue;
        }
        await flushText();
        const json = providerJsonValue(value);
        blocks.push({
            id: await deriveConversationId('block', prepared.runtime.response_operation_id, String(blocks.length)),
            type: 'json',
            value: json,
        });
    }
    await flushText();
    if (blocks.length === 0) throw new Error('Replicate succeeded prediction has no semantic output');
    return { blocks, assets };
}

function usage(prediction: Prediction): GenerationUsage | undefined {
    const metrics = prediction.metrics;
    if (metrics === undefined) return undefined;
    const payload: Record<string, number> = {};
    for (const [key, value] of [
        ['predict_time', metrics.predict_time],
        ['total_time', metrics.total_time],
    ] as const) {
        if (value === undefined) continue;
        if (!Number.isFinite(value) || value < 0) throw new Error(`Replicate prediction contains invalid ${key}`);
        payload[key] = value;
    }
    if (Object.keys(payload).length === 0) return undefined;
    return {
        reported_usage: [
            {
                source: 'provider',
                protocol: REPLICATE_PREDICTIONS_PROTOCOL,
                accounting_basis: 'replicate_seconds',
                payload: providerJsonValue(payload),
            },
        ],
    };
}

async function finalizeReplicateCanonical(
    prepared: PreparedReplicateCanonical,
    prediction: Prediction,
    options: ExecutionOptions,
    driver: ReplicateDriver,
    signal?: AbortSignal,
): Promise<FinalizedReplicateCanonical> {
    assertSucceededPrediction(prediction, options.model);
    const completedAt = prepared.runtime.completed_at ?? prepared.runtime.recorded_at;
    const decodedOutput = await decodeOutput(driver, prediction, options, prepared, signal);
    const generation = await createExecutedGeneration({
        id: prepared.generation_id,
        runtime: { ...prepared.runtime, completed_at: completedAt },
        receipt: prepared.receipt,
        provider: prepared.receipt.target.provider,
        protocol: REPLICATE_PREDICTIONS_PROTOCOL,
        adapter_version: REPLICATE_PREDICTIONS_ADAPTER_VERSION,
        requested_model: options.model,
        resolved_model: options.model,
        provider_response_id: prediction.id,
        finish_reason: 'stop',
        usage: usage(prediction),
    });
    const rawTurn = createGeneratedAgentTurn({
        id: prepared.response_turn_id,
        authority: 'ordinary',
        blocks: decodedOutput.blocks,
        status: 'completed',
        timestamps: { recorded_at: completedAt, completed_at: completedAt },
        model_visibility: 'include',
        provenance: { type: 'generated' },
        generation_id: generation.id,
    });
    const rawDecoded: DecodedConversationResponse = {
        turns: [rawTurn],
        assets: decodedOutput.assets,
        generation,
        diagnostics: [],
        payload_fingerprint: await fingerprintJson(
            providerJsonValue({
                id: prediction.id,
                model: prediction.model,
                version: prediction.version,
                status: prediction.status,
                output: prediction.output,
                metrics: prediction.metrics,
            }),
        ),
    };
    let decoded = rawDecoded;
    let normalized: ReturnType<typeof normalizeCompletionResult> | undefined;
    if (options.result_schema !== undefined) {
        const block = rawTurn.blocks[0];
        if (rawTurn.blocks.length !== 1 || (block?.type !== 'text' && block?.type !== 'json')) {
            decoded = rejectDecodedStructuredOutput(rawDecoded, {
                code: 'validation_error',
                message: 'Replicate structured output requires one text or JSON result',
            });
        } else {
            normalized = normalizeCompletionResult(
                [block.type === 'text' ? { type: 'text', value: block.text } : { type: 'json', value: block.value }],
                options.result_schema,
            );
            if (normalized.status === 'valid') {
                const structuredBlockId = await deriveConversationId(
                    'block',
                    prepared.runtime.response_operation_id,
                    'structured',
                );
                decoded = {
                    ...rawDecoded,
                    turns: [
                        {
                            ...rawTurn,
                            blocks: [
                                { id: structuredBlockId, type: 'json', value: normalized.structured_output.value },
                            ],
                        },
                    ],
                };
            } else {
                decoded = rejectDecodedStructuredOutput(rawDecoded, normalized.error);
            }
        }
    }
    const final = appendDecodedConversationResponse(
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
    ).document;
    return {
        raw_decoded: rawDecoded,
        decoded,
        response: createCanonicalExecutionResponse(final, prepared.runtime.response_operation_id, {
            ...(options.include_original_response ? { original_response: prediction } : {}),
        }),
        normalized,
    };
}

function assertRecoverable(prepared: PreparedReplicateCanonical, options: ExecutionOptions): void {
    if (prepared.accepted_response !== undefined && options.include_original_response) {
        throw new Error('An idempotently recovered Replicate response cannot reconstruct original_response');
    }
}

export async function executeReplicateCanonical(input: {
    driver: ReplicateDriver;
    segments: PromptSegment[];
    prompt: string;
    options: ExecutionOptions;
    signal?: AbortSignal;
}): Promise<CanonicalExecutionResponse> {
    const prepared = await prepareReplicateCanonical(input);
    assertRecoverable(prepared, input.options);
    if (prepared.accepted_response !== undefined) return recoverCanonicalExecutionResponse(prepared, input.options);
    await publishCanonicalPreparedRequest(prepared, input.options);
    input.signal?.throwIfAborted();
    const prediction = await input.driver.service.predictions.create({
        ...prepared.request,
        ...(input.signal === undefined ? {} : { signal: input.signal }),
    });
    let cancellation: Promise<void> | undefined;
    const abort = () => {
        cancellation ??= input.driver.cancelPrediction(prediction);
    };
    input.signal?.addEventListener('abort', abort, { once: true });
    try {
        const final = await finalPrediction(input.driver, prediction, input.signal);
        input.signal?.throwIfAborted();
        return (await finalizeReplicateCanonical(prepared, final, input.options, input.driver, input.signal)).response;
    } finally {
        input.signal?.removeEventListener('abort', abort);
        if (input.signal?.aborted) abort();
        await cancellation;
    }
}

class ReplicateStreamingSession {
    prediction: Prediction | undefined;
    private source: EventSource | undefined;
    private cancellation: Promise<void> | undefined;
    private events: EventStream<ReplicateSseEvent> | undefined;
    private closed = false;

    constructor(
        private readonly driver: ReplicateDriver,
        private readonly request: ReplicateRequest,
        private readonly signal: AbortSignal,
    ) {}

    async open(): Promise<AsyncIterable<ReplicateSseEvent>> {
        this.signal.throwIfAborted();
        const prediction = await this.driver.service.predictions.create({ ...this.request, signal: this.signal });
        this.prediction = prediction;
        this.signal.addEventListener('abort', this.abort, { once: true });
        if (this.signal.aborted) {
            this.abort();
            await this.cancellation;
            this.signal.throwIfAborted();
        }
        if (prediction.urls.stream === undefined || isTerminal(prediction)) {
            return {
                async *[Symbol.asyncIterator]() {
                    // A non-streaming model still uses the authoritative Prediction polling finalizer.
                },
            };
        }
        const events = new EventStream<ReplicateSseEvent>();
        this.events = events;
        const source = new EventSource(prediction.urls.stream, {
            fetch: (url, init) =>
                this.driver.service.fetch(url instanceof URL ? url.toString() : url, init as RequestInit),
        });
        this.source = source;
        source.addEventListener('output', (event: MessageEvent) => events.push({ type: 'output', data: event.data }));
        source.addEventListener('error', (event: MessageEvent) => {
            let reason: unknown = event.data || 'Replicate SSE connection failed';
            try {
                reason = JSON.parse(event.data);
            } catch {
                // Plain provider error text remains a bounded diagnostic below.
            }
            events.fail(new Error(`Replicate stream failed: ${boundedDiagnostic(reason)}`));
            this.source?.close();
            this.cancelRemote();
        });
        source.addEventListener('done', (event: MessageEvent) => {
            events.push({ type: 'done', data: event.data });
            events.close();
            this.source?.close();
        });
        return events;
    }

    private readonly abort = () => {
        this.source?.close();
        this.events?.close();
        this.cancelRemote();
    };

    private cancelRemote(): void {
        const prediction = this.prediction;
        if (prediction !== undefined) this.cancellation ??= this.driver.cancelPrediction(prediction);
    }

    async close(): Promise<void> {
        if (this.closed) return this.cancellation;
        this.closed = true;
        this.signal.removeEventListener('abort', this.abort);
        this.source?.close();
        this.events?.close();
        await this.cancellation;
    }

    async final(): Promise<Prediction> {
        if (this.prediction === undefined) throw new Error('Replicate stream has no prediction');
        return finalPrediction(this.driver, this.prediction, this.signal);
    }
}

class ReplicateStreamAccumulator {
    private done = false;
    private text = '';

    accept(event: ReplicateSseEvent): string {
        if (this.done) throw new Error('Replicate stream emitted an event after done');
        if (event.type === 'output') {
            this.text += event.data;
            return event.data;
        }
        let reason: unknown;
        try {
            reason = event.data.length === 0 ? {} : JSON.parse(event.data);
        } catch {
            throw new Error('Replicate stream emitted malformed done data');
        }
        if (typeof reason !== 'object' || reason === null || Array.isArray(reason)) {
            throw new Error('Replicate stream emitted malformed done data');
        }
        this.done = true;
        return '';
    }

    preview(): string {
        return this.text;
    }
}

class ReplicateCanonicalDriverStream implements CanonicalFinalizingDriverStream {
    constructor(
        private readonly source: AsyncIterable<ReplicateSseEvent>,
        private readonly accumulator: ReplicateStreamAccumulator,
        private readonly finalizeResponse: () => Promise<CanonicalExecutionResponse>,
    ) {}

    async *[Symbol.asyncIterator](): AsyncIterator<CompletionChunkObject> {
        for await (const event of this.source) {
            const fragment = this.accumulator.accept(event);
            if (fragment.length > 0) yield { result: [{ type: 'text', value: fragment }] };
        }
    }

    finalizeCanonicalExecution(): Promise<CanonicalExecutionResponse> {
        return this.finalizeResponse();
    }
}

function streamPosition(): NativeStreamPosition {
    return { protocol: REPLICATE_PREDICTIONS_PROTOCOL, path: ['output'] };
}

export async function streamReplicateCanonical(input: {
    driver: ReplicateDriver;
    segments: PromptSegment[];
    prompt: string;
    options: ExecutionOptions;
    signal?: AbortSignal;
}): Promise<CanonicalExecutionStream> {
    const prepared = await prepareReplicateCanonical(input);
    assertRecoverable(prepared, input.options);
    if (prepared.accepted_response !== undefined) {
        const completion = await recoverCanonicalExecutionResponse(prepared, input.options);
        return { completion, cancel: async () => {}, async *[Symbol.asyncIterator]() {} };
    }
    await publishCanonicalPreparedRequest(prepared, input.options);
    const abortController = new AbortController();
    const forwardAbort = () => abortController.abort(input.signal?.reason);
    if (input.signal?.aborted) forwardAbort();
    else input.signal?.addEventListener('abort', forwardAbort, { once: true });
    const session = new ReplicateStreamingSession(input.driver, prepared.request, abortController.signal);
    try {
        const source = await session.open();
        const accumulator = new ReplicateStreamAccumulator();
        return canonicalExecutionStreamFromDriver(
            new ReplicateCanonicalDriverStream(source, accumulator, async () => {
                const final = await session.final();
                return (
                    await finalizeReplicateCanonical(
                        prepared,
                        final,
                        input.options,
                        input.driver,
                        abortController.signal,
                    )
                ).response;
            }),
            {
                abort: () => abortController.abort(),
                close: async () => {
                    input.signal?.removeEventListener('abort', forwardAbort);
                    await session.close();
                },
            },
        );
    } catch (error: unknown) {
        input.signal?.removeEventListener('abort', forwardAbort);
        abortController.abort(error);
        await session.close();
        throw error;
    }
}

export async function streamReplicateCanonicalEvents(input: {
    driver: ReplicateDriver;
    segments: PromptSegment[];
    prompt: string;
    options: ExecutionOptions;
    signal?: AbortSignal;
    open: CanonicalStreamOpenOptions;
}): Promise<CanonicalExecutionEventStream> {
    if (input.options.conversation_runtime === undefined) {
        throw new Error('Canonical typed streaming requires conversation_runtime');
    }
    const prepared = await prepareReplicateCanonical(input);
    assertRecoverable(prepared, input.options);
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
            () => recoverCanonicalExecutionResponse(prepared, input.options),
            { ...input.open, origin: 'accepted_recovery' },
        );
    }
    const abortController = new AbortController();
    const forwardAbort = () => abortController.abort(input.signal?.reason);
    const session = new ReplicateStreamingSession(input.driver, prepared.request, abortController.signal);
    const accumulator = new ReplicateStreamAccumulator();
    const position = streamPosition();
    const draftBlockId = `${prepared.response_turn_id}:replicate:text`;
    let draftStarted = false;
    const eventStream = canonicalNativeExecutionEventStream({
        identity,
        open: input.open,
        openSource: () => session.open(),
        map: async (event, writer) => {
            const fragment = accumulator.accept(event);
            if (fragment.length === 0) return;
            if (!draftStarted) {
                draftStarted = true;
                await writer.startBlock({
                    draft_block_id: draftBlockId,
                    native_position: position,
                    block: { type: 'text' },
                });
            }
            await writer.text({ draft_block_id: draftBlockId, native_position: position, text: fragment });
        },
        finalize: async () => {
            const final = await session.final();
            const finalized = await finalizeReplicateCanonical(
                prepared,
                final,
                input.options,
                input.driver,
                abortController.signal,
            );
            return {
                decoded: finalized.decoded,
                response: finalized.response,
                prepare_reconciliation: async () => {
                    const rawBlock = finalized.raw_decoded.turns[0]?.blocks[0];
                    const committedBlock = finalized.decoded.turns[0]?.blocks[0];
                    const reconciliations = [];
                    const transformations = [];
                    if (draftStarted) {
                        if (
                            finalized.raw_decoded.turns[0]?.blocks.length !== 1 ||
                            rawBlock?.type !== 'text' ||
                            committedBlock === undefined
                        ) {
                            throw new Error('Replicate streamed text does not match its authoritative final output');
                        }
                        if (accumulator.preview() !== rawBlock.text) {
                            throw new Error('Replicate streamed text differs from its authoritative final output');
                        }
                        if (finalized.normalized?.status === 'valid') {
                            if (committedBlock.type !== 'json')
                                throw new Error('Replicate structured output has no JSON result');
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
                        } else {
                            reconciliations.push({
                                draft_block_ids: [draftBlockId],
                                native_positions: [position],
                                committed_block_ids: [committedBlock.id],
                                disposition: 'direct' as const,
                            });
                        }
                    }
                    const decoded: DecodedConversationResponse = {
                        ...finalized.decoded,
                        stream_evidence: {
                            item_mappings:
                                draftStarted && rawBlock !== undefined
                                    ? [{ canonical_id: rawBlock.id, native_position: position, kind: 'block' as const }]
                                    : [],
                            transformations,
                        },
                    };
                    return {
                        decoded,
                        reconciliations,
                        deliver_final_events: async (writer) => {
                            if (decoded.generation.usage !== undefined) await writer.usage(decoded.generation.usage);
                            if (draftStarted) {
                                await writer.finishBlock({
                                    draft_block_id: draftBlockId,
                                    native_position: position,
                                    outcome: 'native_complete',
                                });
                            }
                            await writer.finish({ outcome: 'completed', finish_reason: 'stop' });
                        },
                    };
                },
                ...(finalized.normalized?.status === 'valid' && input.options.result_schema !== undefined
                    ? { result_schema: input.options.result_schema }
                    : {}),
            };
        },
        abort: () => abortController.abort(),
        close: async () => {
            input.signal?.removeEventListener('abort', forwardAbort);
            await session.close();
        },
    });
    await publishCanonicalPreparedRequest(prepared, input.options);
    if (input.signal?.aborted) forwardAbort();
    else input.signal?.addEventListener('abort', forwardAbort, { once: true });
    return eventStream;
}
