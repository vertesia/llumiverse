import {
    type AgentContentBlock,
    type Asset,
    type AssetStorage,
    appendDecodedConversationResponse,
    createTextBlock,
    createUserTurn,
    deriveConversationId,
    fingerprintJson,
    type GenerationUsage,
    isConversationDocumentFormat,
    type JsonObject,
    type JsonValue,
    parseConversationDocument,
} from '@llumiverse/conversation';
import {
    type CanonicalExecutionResponse,
    createCanonicalExecutionResponse,
    type ExecutionOptions,
    type ExecutionTokenUsage,
    type OpenAiDalleOptions,
    type OpenAiGptImageOptions,
    PromptRole,
    type PromptSegment,
} from '@llumiverse/core';
import type OpenAI from 'openai';
import {
    acceptedCanonicalResponse,
    appendCanonicalPrompt,
    assertAcceptedCanonicalRequest,
    canonicalResponseIdentities,
    createExecutedGeneration,
    createRequestReceipt,
    newCanonicalConversation,
    providerJsonValue,
    publishCanonicalPreparedRequest,
    resolveConversationRuntime,
} from '../conversation/canonical-runtime.js';

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;

const OPENAI_IMAGES_PROTOCOL = 'openai.images.generate';
const OPENAI_IMAGES_ADAPTER_VERSION = '2026-09-30.canonical.1';
const MAX_GENERATED_IMAGE_BYTES = 50_000_000;
const MAX_INLINE_GENERATED_IMAGE_BYTES = 8_000_000;
const MAX_GENERATED_IMAGE_CHUNKS = 8_192;
const MAX_CANONICAL_JSON_BYTES = 32 * 1024 * 1024;
const CANONICAL_RESPONSE_JSON_RESERVE_BYTES = 1024 * 1024;
const SUPPORTED_IMAGE_MIME_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);

interface DecodedImage {
    bytes: Uint8Array;
    mime_type: string;
    content_hash: string;
    revised_prompt?: string;
    source_url?: string;
}

function imagePromptText(prompt: ResponseInputItem[]): string {
    const text: string[] = [];
    for (const item of prompt) {
        if (!('role' in item) || item.role !== 'user' || !('content' in item)) {
            throw new Error('OpenAI standalone image generation supports user text input only');
        }
        if (typeof item.content === 'string') {
            if (item.content.trim().length > 0) text.push(item.content);
            continue;
        }
        if (!Array.isArray(item.content)) throw new Error('OpenAI image input content is malformed');
        for (const part of item.content) {
            if (!('type' in part) || part.type !== 'input_text' || !('text' in part)) {
                throw new Error('OpenAI standalone image generation does not support input media');
            }
            if (part.text.trim().length > 0) text.push(part.text);
        }
    }
    const value = text.join('\n').trim();
    if (value.length === 0) throw new Error('OpenAI standalone image generation requires nonempty user text');
    return value;
}

export function validateOpenAICanonicalImageInput(segments: PromptSegment[], options: ExecutionOptions): void {
    if (options.tools && options.tools.length > 0) {
        throw new Error('OpenAI standalone image generation does not support tools');
    }
    if (options.result_schema !== undefined) {
        throw new Error('OpenAI standalone image generation does not support structured output');
    }
    if (options.conversation_runtime?.materialized_input !== undefined) {
        throw new Error('OpenAI standalone image generation does not support materialized conversation input');
    }
    if (segments.length === 0) throw new Error('OpenAI standalone image generation requires user text');
    for (const segment of segments) {
        if (segment.role !== PromptRole.user) {
            throw new Error(`OpenAI standalone image generation does not support ${segment.role} input`);
        }
        if ((segment.files?.length ?? 0) > 0) {
            throw new Error('OpenAI standalone image generation does not support input files');
        }
        if (segment.content.trim().length === 0) {
            throw new Error('OpenAI standalone image generation requires nonempty user text');
        }
    }
}

export function openAIImageRequest(
    prompt: ResponseInputItem[],
    options: ExecutionOptions,
): OpenAI.Images.ImageGenerateParamsNonStreaming {
    const promptText = imagePromptText(prompt);
    const modelOptions = options.model_options as OpenAiDalleOptions | OpenAiGptImageOptions | undefined;
    const request: OpenAI.Images.ImageGenerateParamsNonStreaming = {
        model: options.model,
        prompt: promptText,
        size: modelOptions?.size ?? '1024x1024',
    };
    if (options.model.includes('dall-e') || modelOptions?._option_id === 'openai-dalle') {
        const dalle = modelOptions as OpenAiDalleOptions | undefined;
        request.n = dalle?.n ?? 1;
        request.response_format = dalle?.response_format ?? 'b64_json';
        if (options.model.includes('dall-e-3')) {
            request.quality = dalle?.image_quality ?? 'standard';
            if (dalle?.style !== undefined) request.style = dalle.style;
        }
    } else {
        const image = modelOptions as OpenAiGptImageOptions | undefined;
        request.n = 1;
        if (image?.image_quality !== undefined) request.quality = image.image_quality;
        if (image?.background !== undefined) request.background = image.background;
        if (image?.output_format !== undefined) request.output_format = image.output_format;
    }
    return request;
}

function effectiveImageOptions(request: OpenAI.Images.ImageGenerateParamsNonStreaming): JsonObject {
    const { model: _model, prompt: _prompt, ...options } = request;
    return providerJsonValue(options) as JsonObject;
}

function isBase64Alphabet(code: number): boolean {
    return (
        (code >= 0x41 && code <= 0x5a) ||
        (code >= 0x61 && code <= 0x7a) ||
        (code >= 0x30 && code <= 0x39) ||
        code === 0x2b ||
        code === 0x2f
    );
}

function strictBase64(value: string, maximumBytes: number): Uint8Array {
    const maximumEncodedLength = Math.ceil(maximumBytes / 3) * 4;
    if (value.length === 0 || value.length > maximumEncodedLength || value.length % 4 !== 0) {
        throw new Error('OpenAI image response contains malformed base64 data');
    }
    let padding = 0;
    if (value.endsWith('==')) padding = 2;
    else if (value.endsWith('=')) padding = 1;
    for (let index = 0; index < value.length - padding; index += 1) {
        if (!isBase64Alphabet(value.charCodeAt(index))) {
            throw new Error('OpenAI image response contains malformed base64 data');
        }
    }
    for (let index = value.length - padding; index < value.length; index += 1) {
        if (value.charCodeAt(index) !== 0x3d) throw new Error('OpenAI image response contains malformed base64 data');
    }
    const bytes = new Uint8Array(Buffer.from(value, 'base64'));
    if (bytes.byteLength === 0 || bytes.byteLength > maximumBytes || Buffer.from(bytes).toString('base64') !== value) {
        throw new Error('OpenAI image response contains malformed base64 data');
    }
    return bytes;
}

async function sha256Bytes(bytes: Uint8Array): Promise<string> {
    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    const digest = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', copy));
    return `sha256:${Array.from(digest, (byte) => byte.toString(16).padStart(2, '0')).join('')}`;
}

async function readBoundedImageResponse(response: Response, maximumBytes: number, signal?: AbortSignal) {
    if (!response.ok) throw new Error(`OpenAI generated image download failed with status ${response.status}`);
    const mimeType = response.headers.get('content-type')?.split(';', 1)[0]?.trim().toLowerCase();
    if (mimeType === undefined || !SUPPORTED_IMAGE_MIME_TYPES.has(mimeType)) {
        throw new Error(`OpenAI generated image has unsupported MIME type ${mimeType ?? 'missing'}`);
    }
    const declaredLength = response.headers.get('content-length');
    if (declaredLength !== null) {
        const parsed = Number(declaredLength);
        if (!Number.isSafeInteger(parsed) || parsed <= 0 || parsed > maximumBytes) {
            throw new Error(`OpenAI generated image exceeds the ${maximumBytes} byte limit`);
        }
    }
    if (response.body === null) throw new Error('OpenAI generated image response has no body');
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
            if (chunkCount > MAX_GENERATED_IMAGE_CHUNKS) {
                throw new Error('OpenAI generated image response contains too many chunks');
            }
            if (item.value.byteLength === 0) continue;
            byteLength += item.value.byteLength;
            if (byteLength > maximumBytes) {
                throw new Error(`OpenAI generated image exceeds the ${maximumBytes} byte limit`);
            }
            chunks.push(item.value.slice());
        }
    } finally {
        if (!completed) await reader.cancel().catch(() => undefined);
        reader.releaseLock();
    }
    if (byteLength === 0) throw new Error('OpenAI generated image is empty');
    const bytes = new Uint8Array(byteLength);
    let offset = 0;
    for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
    }
    const detectedMimeType = detectImageMimeType(bytes);
    if (detectedMimeType !== mimeType) {
        throw new Error(`OpenAI generated image MIME type ${mimeType} does not match its bytes`);
    }
    return { bytes, mime_type: detectedMimeType };
}

function detectImageMimeType(bytes: Uint8Array): string {
    if (
        bytes.byteLength >= 8 &&
        bytes[0] === 0x89 &&
        bytes[1] === 0x50 &&
        bytes[2] === 0x4e &&
        bytes[3] === 0x47 &&
        bytes[4] === 0x0d &&
        bytes[5] === 0x0a &&
        bytes[6] === 0x1a &&
        bytes[7] === 0x0a
    ) {
        return 'image/png';
    }
    if (bytes.byteLength >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
        return 'image/jpeg';
    }
    if (
        bytes.byteLength >= 12 &&
        bytes[0] === 0x52 &&
        bytes[1] === 0x49 &&
        bytes[2] === 0x46 &&
        bytes[3] === 0x46 &&
        bytes[8] === 0x57 &&
        bytes[9] === 0x45 &&
        bytes[10] === 0x42 &&
        bytes[11] === 0x50
    ) {
        return 'image/webp';
    }
    throw new Error('OpenAI generated image bytes have an unsupported format');
}

async function decodeImage(
    image: OpenAI.Images.Image,
    fetchImage: typeof fetch,
    maximumBytes: number,
    signal?: AbortSignal,
): Promise<DecodedImage> {
    let bytes: Uint8Array;
    let mimeType: string;
    if (image.b64_json !== undefined) {
        bytes = strictBase64(image.b64_json, maximumBytes);
        mimeType = detectImageMimeType(bytes);
    } else if (image.url !== undefined) {
        const url = new URL(image.url);
        if (url.protocol !== 'https:' && url.protocol !== 'http:') {
            throw new Error('OpenAI generated image URL must use HTTP or HTTPS');
        }
        const decoded = await readBoundedImageResponse(await fetchImage(url, { signal }), maximumBytes, signal);
        bytes = decoded.bytes;
        mimeType = decoded.mime_type;
    } else {
        throw new Error('OpenAI image response item contains neither base64 data nor a URL');
    }
    if (bytes.byteLength > maximumBytes) {
        throw new Error(`OpenAI generated image exceeds the ${maximumBytes} byte limit`);
    }
    return {
        bytes,
        mime_type: mimeType,
        content_hash: await sha256Bytes(bytes),
        ...(image.revised_prompt === undefined ? {} : { revised_prompt: image.revised_prompt }),
        ...(image.url === undefined ? {} : { source_url: image.url }),
    };
}

async function imageStorage(
    decoded: DecodedImage,
    options: ExecutionOptions,
    signal?: AbortSignal,
): Promise<AssetStorage> {
    if (options.store_generated_asset === undefined) {
        if (decoded.bytes.byteLength > MAX_INLINE_GENERATED_IMAGE_BYTES) {
            throw new Error(
                `OpenAI generated image exceeds the ${MAX_INLINE_GENERATED_IMAGE_BYTES} byte inline asset limit`,
            );
        }
        return { type: 'inline_base64', data: Buffer.from(decoded.bytes).toString('base64') };
    }
    const source = new ReadableStream<Uint8Array>({
        start(controller) {
            controller.enqueue(decoded.bytes.slice());
            controller.close();
        },
    });
    const stored = await options.store_generated_asset(source, { kind: 'image', mime_type: decoded.mime_type }, signal);
    signal?.throwIfAborted();
    if (stored.byte_length !== decoded.bytes.byteLength || stored.content_hash !== decoded.content_hash) {
        throw new Error('Generated asset storage did not preserve the exact OpenAI image bytes');
    }
    if (stored.storage.type !== 'external') {
        throw new Error('Generated asset storage must return external canonical storage');
    }
    return stored.storage;
}

function maximumImageOutputBytes(document: unknown, options: ExecutionOptions): number {
    if (options.store_generated_asset !== undefined) return MAX_GENERATED_IMAGE_BYTES;
    const currentBytes = new TextEncoder().encode(JSON.stringify(document)).byteLength;
    const remainingJsonBytes = MAX_CANONICAL_JSON_BYTES - currentBytes - CANONICAL_RESPONSE_JSON_RESERVE_BYTES;
    const remainingDecodedBytes = Math.floor((remainingJsonBytes * 3) / 4);
    const maximum = Math.min(MAX_INLINE_GENERATED_IMAGE_BYTES, remainingDecodedBytes);
    if (maximum <= 0) {
        throw new Error('Canonical conversation has no safe capacity for inline generated image output');
    }
    return maximum;
}

export function mapOpenAIImagesUsage(
    usage: OpenAI.Images.ImagesResponse.Usage | undefined,
): ExecutionTokenUsage | undefined {
    if (usage === undefined) return undefined;
    return {
        prompt: usage.input_tokens,
        prompt_new: usage.input_tokens,
        result: usage.output_tokens,
        result_image: usage.output_tokens,
        total: usage.total_tokens,
    };
}

function canonicalImageUsage(usage: OpenAI.Images.ImagesResponse.Usage | undefined): GenerationUsage | undefined {
    if (usage === undefined) return undefined;
    const basis = 'openai_images_tokens';
    return {
        input_tokens: usage.input_tokens,
        input_new_tokens: usage.input_tokens,
        cache_read_tokens: 0,
        output_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
        accounting_provenance: {
            input_tokens: { method: 'reported', accounting_basis: basis },
            input_new_tokens: { method: 'derived', accounting_basis: basis },
            cache_read_tokens: { method: 'derived', accounting_basis: basis },
            output_tokens: { method: 'reported', accounting_basis: basis },
            total_tokens: { method: 'reported', accounting_basis: basis },
        },
        input_partition: { type: 'complete_disjoint', cache_write_bucket: 'inapplicable' },
        reported_usage: [
            {
                source: 'provider',
                protocol: OPENAI_IMAGES_PROTOCOL,
                accounting_basis: basis,
                payload: providerJsonValue(usage),
            },
        ],
    };
}

export async function executeOpenAIImageCanonical(input: {
    service: OpenAI;
    provider: string;
    prompt: ResponseInputItem[];
    options: ExecutionOptions;
    fetch_image: typeof fetch;
    request_options?: { signal?: AbortSignal; timeout?: number };
    signal?: AbortSignal;
}): Promise<CanonicalExecutionResponse> {
    const promptText = imagePromptText(input.prompt);
    const request = openAIImageRequest(input.prompt, input.options);
    const requestJson = providerJsonValue(request);
    const runtime = resolveConversationRuntime(input.options);
    let document = isConversationDocumentFormat(input.options.conversation)
        ? parseConversationDocument(input.options.conversation)
        : newCanonicalConversation(runtime);
    if (
        input.options.conversation !== undefined &&
        input.options.conversation !== null &&
        !isConversationDocumentFormat(input.options.conversation)
    ) {
        throw new TypeError('OpenAI standalone image generation does not support legacy conversation input');
    }
    if (
        input.options.conversation_runtime?.conversation_id !== undefined &&
        input.options.conversation_runtime.conversation_id !== document.id
    ) {
        throw new Error('conversation_runtime.conversation_id does not match the canonical document');
    }
    const acceptedBefore = acceptedCanonicalResponse(document, runtime.response_operation_id);
    const turnId = await deriveConversationId('turn', runtime.input_operation_id, 'image-prompt');
    const blockId = await deriveConversationId('block', runtime.input_operation_id, 'image-prompt');
    if (
        document.turns.length > 0 &&
        acceptedBefore === undefined &&
        (Object.keys(document.generations).length > 0 || document.turns.some((turn) => turn.id !== turnId))
    ) {
        throw new Error('OpenAI standalone image generation does not support conversation continuation');
    }
    const turn = createUserTurn({
        id: turnId,
        authority: 'ordinary',
        model_visibility: 'include',
        status: 'completed',
        timestamps: { recorded_at: runtime.recorded_at },
        provenance: { type: 'received' },
        blocks: [createTextBlock({ id: blockId, text: promptText, format: 'plain' })],
    });
    const contextEntryId = await deriveConversationId('context', runtime.input_operation_id, 'image-prompt');
    const appended = await appendCanonicalPrompt(
        document,
        {
            turns: [turn],
            assets: [],
            context_entries: [{ id: contextEntryId, type: 'source_turn', turn_id: turn.id }],
            item_mappings: [],
        },
        { ...runtime, conversation_id: document.id },
        undefined,
        { prompt: promptText },
    );
    document = appended.document;
    const accepted = acceptedCanonicalResponse(document, runtime.response_operation_id);
    if (accepted !== undefined) {
        await assertAcceptedCanonicalRequest(
            { accepted_response: accepted, runtime },
            { provider: input.provider, protocol: OPENAI_IMAGES_PROTOCOL, model: input.options.model },
            requestJson,
        );
        if (input.options.include_original_response) {
            throw new Error('An idempotently recovered image response cannot reconstruct original_response');
        }
        return createCanonicalExecutionResponse(
            document,
            runtime.response_operation_id,
            {},
            await input.options.load_recovered_canonical_output?.({
                conversation_id: document.id,
                response_operation_id: runtime.response_operation_id,
            }),
        );
    }
    const receipt = await createRequestReceipt(
        document,
        { ...runtime, conversation_id: document.id },
        {
            provider: input.provider,
            protocol: OPENAI_IMAGES_PROTOCOL,
            model: input.options.model,
            adapter_version: OPENAI_IMAGES_ADAPTER_VERSION,
            options: effectiveImageOptions(request),
        },
        requestJson,
        [
            { canonical_id: turn.id, native_id: 'prompt', kind: 'turn' },
            { canonical_id: blockId, native_id: 'prompt/text', kind: 'block' },
        ],
        appended.tool_definitions,
    );
    const identities = await canonicalResponseIdentities(runtime);
    const maximumBytes = maximumImageOutputBytes(document, input.options);
    await publishCanonicalPreparedRequest(
        {
            document,
            native_conversation: input.prompt,
            receipt,
            runtime: { ...runtime, conversation_id: document.id },
            generation_id: identities.generation_id,
            response_turn_id: identities.response_turn_id,
            tool_definitions: appended.tool_definitions,
        },
        input.options,
    );
    input.signal?.throwIfAborted();
    const response = input.request_options
        ? await input.service.images.generate(request, input.request_options)
        : await input.service.images.generate(request);
    input.signal?.throwIfAborted();
    if (!Array.isArray(response.data) || response.data.length === 0) {
        throw new Error('OpenAI image response contains no images');
    }
    const decodedImages: DecodedImage[] = [];
    let totalBytes = 0;
    for (const image of response.data) {
        if (totalBytes >= maximumBytes) {
            throw new Error(`OpenAI generated images exceed the ${maximumBytes} byte total limit`);
        }
        const decoded = await decodeImage(
            image,
            input.fetch_image,
            Math.min(maximumBytes - totalBytes, maximumBytes),
            input.signal,
        );
        totalBytes += decoded.bytes.byteLength;
        if (totalBytes > maximumBytes) {
            throw new Error(`OpenAI generated images exceed the ${maximumBytes} byte total limit`);
        }
        decodedImages.push(decoded);
    }
    const completedAt = runtime.completed_at ?? runtime.recorded_at;
    const completedRuntime = { ...runtime, completed_at: completedAt };
    const assets: Asset[] = [];
    const blocks: AgentContentBlock[] = [];
    for (let index = 0; index < decodedImages.length; index += 1) {
        const decoded = decodedImages[index];
        const assetId = await deriveConversationId('asset', runtime.response_operation_id, String(index));
        const responseBlockId = await deriveConversationId('block', runtime.response_operation_id, String(index));
        assets.push({
            id: assetId,
            kind: 'image',
            mime_type: decoded.mime_type,
            storage: await imageStorage(decoded, input.options, input.signal),
            provenance: {
                type: 'generated',
                generation_id: identities.generation_id,
                source_turn_id: identities.response_turn_id,
            },
            byte_length: decoded.bytes.byteLength,
            content_hash: decoded.content_hash,
            created_at: completedAt,
            ...(decoded.revised_prompt === undefined && decoded.source_url === undefined
                ? {}
                : {
                      metadata: {
                          openai_image: {
                              ...(decoded.revised_prompt === undefined
                                  ? {}
                                  : { revised_prompt: decoded.revised_prompt }),
                              ...(decoded.source_url === undefined ? {} : { source_url: decoded.source_url }),
                          },
                      },
                  }),
        });
        blocks.push({
            id: responseBlockId,
            type: 'image',
            asset_id: assetId,
            ...(decoded.revised_prompt === undefined ? {} : { caption: decoded.revised_prompt }),
        });
    }
    const generation = await createExecutedGeneration({
        id: identities.generation_id,
        runtime: completedRuntime,
        receipt,
        provider: input.provider,
        protocol: OPENAI_IMAGES_PROTOCOL,
        adapter_version: OPENAI_IMAGES_ADAPTER_VERSION,
        requested_model: input.options.model,
        resolved_model: input.options.model,
        finish_reason: 'stop',
        usage: canonicalImageUsage(response.usage),
    });
    const responseTurn = {
        id: identities.response_turn_id,
        kind: 'agent' as const,
        authority: 'ordinary' as const,
        blocks,
        status: 'completed' as const,
        timestamps: { recorded_at: completedAt, completed_at: completedAt },
        model_visibility: 'include' as const,
        provenance: { type: 'generated' as const },
        generation_id: generation.id,
    };
    const responseEvidence: JsonValue = {
        created: response.created,
        data: decodedImages.map((image) => ({
            content_hash: image.content_hash,
            mime_type: image.mime_type,
            byte_length: image.bytes.byteLength,
            ...(image.revised_prompt === undefined ? {} : { revised_prompt: image.revised_prompt }),
            ...(image.source_url === undefined ? {} : { source_url: image.source_url }),
        })),
        ...(response.usage === undefined ? {} : { usage: providerJsonValue(response.usage) }),
    };
    const finalDocument = appendDecodedConversationResponse(
        {
            document,
            generation_id: identities.generation_id,
            response_turn_id: identities.response_turn_id,
            receipt,
            payload: requestJson,
            diagnostics: [],
        },
        {
            turns: [responseTurn],
            assets,
            generation,
            diagnostics: [],
            payload_fingerprint: await fingerprintJson(responseEvidence),
        },
        { operation_id: runtime.response_operation_id, recorded_at: completedAt },
    ).document;
    return createCanonicalExecutionResponse(finalDocument, runtime.response_operation_id, {
        ...(input.options.include_original_response ? { original_response: response } : {}),
    });
}
