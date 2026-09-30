import {
    type AgentContentBlock,
    type Asset,
    appendDecodedConversationResponse,
    buildConversationTurn,
    type ConversationTurn,
    createProgramTurn,
    createTextBlock,
    createUserTurn,
    deriveConversationId,
    fingerprintJson,
    type GenerationUsage,
    isConversationDocumentFormat,
    type JsonObject,
    type JsonValue,
    type NativeItemMapping,
    parseConversationDocument,
    type TextBlock,
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
    recoverCanonicalExecutionResponse,
    resolveConversationRuntime,
} from '../conversation/canonical-runtime.js';
import {
    generatedImageStorage,
    maximumGeneratedImageOutputBytes,
    readBoundedGeneratedImageResponse,
    verifiedBase64GeneratedImage,
} from '../shared/generated-image.js';

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;

const OPENAI_IMAGES_PROTOCOL = 'openai.images.generate';
const OPENAI_IMAGES_ADAPTER_VERSION = '2026-09-30.canonical.2';

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
        if (
            !('role' in item) ||
            (item.role !== 'system' &&
                item.role !== 'developer' &&
                item.role !== 'user' &&
                item.role !== 'assistant') ||
            !('content' in item)
        ) {
            throw new Error('OpenAI standalone image generation received an unsupported prompt item');
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
    if (value.length === 0) throw new Error('OpenAI standalone image generation requires nonempty text input');
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
    if (segments.length === 0) throw new Error('OpenAI standalone image generation requires text input');
    let hasText = false;
    for (const segment of segments) {
        if (
            segment.role !== PromptRole.system &&
            segment.role !== PromptRole.safety &&
            segment.role !== PromptRole.user &&
            segment.role !== PromptRole.assistant
        ) {
            throw new Error(`OpenAI standalone image generation does not support ${segment.role} input`);
        }
        if (segment.content.trim().length > 0) hasText = true;
        for (const file of segment.files ?? []) {
            if (!file.mime_type.startsWith('text/')) {
                throw new Error(
                    `OpenAI standalone image generation does not support ${file.mime_type || 'untyped'} input files`,
                );
            }
            hasText = true;
        }
    }
    if (!hasText) throw new Error('OpenAI standalone image generation requires nonempty text input');
}

async function imagePromptRecords(
    prompt: ResponseInputItem[],
    runtime: ReturnType<typeof resolveConversationRuntime>,
): Promise<{ turns: ConversationTurn[]; mappings: NativeItemMapping[] }> {
    const turns: ConversationTurn[] = [];
    const mappings: NativeItemMapping[] = [];
    let textIndex = 0;
    for (let itemIndex = 0; itemIndex < prompt.length; itemIndex += 1) {
        const item = prompt[itemIndex];
        if (!('role' in item) || !('content' in item)) {
            throw new Error(`OpenAI standalone image prompt item ${itemIndex} is unsupported`);
        }
        const parts =
            typeof item.content === 'string'
                ? [{ type: 'input_text' as const, text: item.content }]
                : Array.isArray(item.content)
                  ? item.content
                  : [];
        const blocks: TextBlock[] = [];
        for (let partIndex = 0; partIndex < parts.length; partIndex += 1) {
            const part = parts[partIndex];
            if (part.type !== 'input_text') {
                throw new Error('OpenAI standalone image generation does not support input media');
            }
            if (part.text.trim().length === 0) continue;
            const block = createTextBlock({
                id: await deriveConversationId(
                    'block',
                    runtime.input_operation_id,
                    String(itemIndex),
                    String(partIndex),
                ),
                text: part.text,
                format: 'plain',
            });
            blocks.push(block);
            mappings.push({ canonical_id: block.id, native_id: `prompt/parts/${textIndex}`, kind: 'block' });
            textIndex += 1;
        }
        if (blocks.length === 0) continue;
        const turnId = await deriveConversationId('turn', runtime.input_operation_id, String(itemIndex));
        const common = {
            id: turnId,
            authority: 'ordinary' as const,
            model_visibility: 'include' as const,
            status: 'completed' as const,
            timestamps: { recorded_at: runtime.recorded_at },
            provenance: { type: 'received' as const },
            blocks,
        };
        const turn =
            item.role === 'system' || item.role === 'developer'
                ? createProgramTurn({ ...common, authority: item.role })
                : item.role === 'assistant'
                  ? buildConversationTurn({ ...common, kind: 'agent' })
                  : createUserTurn(common);
        turns.push(turn);
        mappings.push({ canonical_id: turn.id, native_id: `source/items/${itemIndex}`, kind: 'turn' });
    }
    return { turns, mappings };
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

async function decodeImage(
    image: OpenAI.Images.Image,
    fetchImage: typeof fetch,
    maximumBytes: number,
    signal?: AbortSignal,
): Promise<DecodedImage> {
    let bytes: Uint8Array;
    let mimeType: string;
    let contentHash: string;
    if (image.b64_json !== undefined) {
        const verified = await verifiedBase64GeneratedImage(image.b64_json, maximumBytes, 'OpenAI');
        bytes = verified.bytes;
        mimeType = verified.mime_type;
        contentHash = verified.content_hash;
    } else if (image.url !== undefined) {
        const url = new URL(image.url);
        if (url.protocol !== 'https:' && url.protocol !== 'http:') {
            throw new Error('OpenAI generated image URL must use HTTP or HTTPS');
        }
        const decoded = await readBoundedGeneratedImageResponse(
            await fetchImage(url, { signal }),
            maximumBytes,
            'OpenAI',
            signal,
        );
        bytes = decoded.bytes;
        mimeType = decoded.mime_type;
        contentHash = decoded.content_hash;
    } else {
        throw new Error('OpenAI image response item contains neither base64 data nor a URL');
    }
    if (bytes.byteLength > maximumBytes) {
        throw new Error(`OpenAI generated image exceeds the ${maximumBytes} byte limit`);
    }
    return {
        bytes,
        mime_type: mimeType,
        content_hash: contentHash,
        ...(image.revised_prompt === undefined ? {} : { revised_prompt: image.revised_prompt }),
        ...(image.url === undefined ? {} : { source_url: image.url }),
    };
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
    if (
        acceptedBefore !== undefined &&
        (acceptedBefore.generation.adapter_version !== OPENAI_IMAGES_ADAPTER_VERSION ||
            acceptedBefore.generation.request_receipt?.target.adapter_version !== OPENAI_IMAGES_ADAPTER_VERSION)
    ) {
        throw new Error(
            `Accepted OpenAI image response operation ${runtime.response_operation_id} uses unsupported adapter version ${acceptedBefore.generation.adapter_version}`,
        );
    }
    const promptRecords = await imagePromptRecords(input.prompt, runtime);
    if (
        document.turns.length > 0 &&
        acceptedBefore === undefined &&
        (Object.keys(document.generations).length > 0 ||
            document.turns.some((turn) => !promptRecords.turns.some((record) => record.id === turn.id)))
    ) {
        throw new Error('OpenAI standalone image generation does not support conversation continuation');
    }
    const contextEntries = await Promise.all(
        promptRecords.turns.map(async (turn, index) => ({
            id: await deriveConversationId('context', runtime.input_operation_id, String(index)),
            type: 'source_turn' as const,
            turn_id: turn.id,
        })),
    );
    const appended = await appendCanonicalPrompt(
        document,
        {
            turns: promptRecords.turns,
            assets: [],
            context_entries: contextEntries,
            item_mappings: promptRecords.mappings,
        },
        { ...runtime, conversation_id: document.id },
        undefined,
        providerJsonValue(input.prompt),
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
        return recoverCanonicalExecutionResponse({ document, runtime, accepted_response: accepted }, input.options);
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
        promptRecords.mappings,
        appended.tool_definitions,
    );
    const identities = await canonicalResponseIdentities(runtime);
    const maximumBytes = maximumGeneratedImageOutputBytes(document, input.options);
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
            storage: await generatedImageStorage(decoded, input.options, 'OpenAI', input.signal),
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
