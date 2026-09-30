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
    type ImageBlock,
    isConversationDocumentFormat,
    type JsonObject,
    type NativeItemMapping,
    parseConversationDocument,
    type TextBlock,
} from '@llumiverse/conversation';
import {
    type CanonicalExecutionResponse,
    createCanonicalExecutionResponse,
    type ExecutionOptions,
    PromptRole,
    type PromptSegment,
    type XAIGrokImageOptions,
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
import {
    generatedImageStorage,
    maximumGeneratedImageOutputBytes,
    readBoundedGeneratedImageResponse,
    type VerifiedGeneratedImage,
    verifiedBase64GeneratedImage,
} from '../shared/generated-image.js';

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;

const XAI_IMAGES_PROTOCOL = 'xai.images';
const XAI_IMAGES_ADAPTER_VERSION = '2026-09-30.canonical.1';
const MAX_XAI_INPUT_IMAGE_BYTES = 20 * 1024 * 1024;
const SUPPORTED_INPUT_MIME_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);

export type XAIImageInput = { file_id: string } | { type: 'image_url'; url: string };

export interface XAIImageRequest {
    aspect_ratio?: XAIGrokImageOptions['aspect_ratio'];
    image?: XAIImageInput;
    images?: XAIImageInput[];
    model: string;
    n?: number;
    prompt: string;
    quality?: XAIGrokImageOptions['quality'];
    resolution?: XAIGrokImageOptions['resolution'];
    response_format?: XAIGrokImageOptions['response_format'];
}

export interface XAIImageResponse {
    created?: number;
    data?: Array<{
        b64_json?: string;
        mime_type?: string;
        revised_prompt?: string;
        url?: string;
    }>;
    usage?: { cost_in_usd_ticks?: number };
}

interface CanonicalImageInput {
    text: string;
    images: Array<{ byte_length: number; content_hash: string; data: string; mime_type: string }>;
}

interface DecodedXAIImage extends VerifiedGeneratedImage {
    revised_prompt?: string;
    source_url?: string;
}

function extractImageRequest(prompt: ResponseInputItem[]): { promptText: string; images: XAIImageInput[] } {
    const text: string[] = [];
    const images: XAIImageInput[] = [];
    for (const item of prompt) {
        if (!('content' in item)) continue;
        if (typeof item.content === 'string') {
            text.push(item.content);
            continue;
        }
        if (!Array.isArray(item.content)) continue;
        for (const part of item.content) {
            if (part.type === 'input_text') text.push(part.text);
            else if (part.type === 'input_image') {
                if (part.image_url) images.push({ type: 'image_url', url: part.image_url });
                else if (part.file_id) images.push({ file_id: part.file_id });
            }
        }
    }
    return { promptText: text.join('\n').trim(), images };
}

export function xAIImageRequest(prompt: ResponseInputItem[], options: ExecutionOptions): XAIImageRequest {
    const { promptText, images } = extractImageRequest(prompt);
    const modelOptions = options.model_options as XAIGrokImageOptions | undefined;
    const payload: XAIImageRequest = {
        model: options.model,
        prompt: promptText,
        ...(modelOptions?.aspect_ratio ? { aspect_ratio: modelOptions.aspect_ratio } : {}),
        ...(modelOptions?.resolution ? { resolution: modelOptions.resolution } : {}),
        ...(modelOptions?.quality ? { quality: modelOptions.quality } : {}),
        ...(modelOptions?.response_format ? { response_format: modelOptions.response_format } : {}),
        ...(modelOptions?.n ? { n: modelOptions.n } : {}),
    };
    if (images.length === 1) payload.image = images[0];
    else if (images.length > 1) payload.images = images;
    return payload;
}

export function xAIImageEndpoint(request: XAIImageRequest): '/images/generations' | '/images/edits' {
    return request.image === undefined && request.images === undefined ? '/images/generations' : '/images/edits';
}

export function validateXAICanonicalImageInput(segments: PromptSegment[], options: ExecutionOptions): void {
    if (options.tools && options.tools.length > 0) throw new Error('xAI image generation does not support tools');
    if (isConversationDocumentFormat(options.conversation)) {
        const document = parseConversationDocument(options.conversation);
        if (document.context.active_tool_definition_ids.length > 0) {
            throw new Error('xAI image generation does not support active canonical tool definitions');
        }
    }
    if (options.result_schema !== undefined) throw new Error('xAI image generation does not support structured output');
    if (options.conversation_runtime?.materialized_input !== undefined) {
        throw new Error('xAI image generation does not support materialized conversation input');
    }
    if (segments.length === 0) throw new Error('xAI image generation requires text input');
    let imageCount = 0;
    let hasText = false;
    for (const segment of segments) {
        if (
            segment.role !== PromptRole.system &&
            segment.role !== PromptRole.safety &&
            segment.role !== PromptRole.user &&
            segment.role !== PromptRole.assistant
        ) {
            throw new Error(`xAI image generation does not support ${segment.role} input`);
        }
        if (segment.content.trim().length > 0) hasText = true;
        for (const file of segment.files ?? []) {
            if (file.mime_type.startsWith('text/')) {
                hasText = true;
                continue;
            }
            if (!SUPPORTED_INPUT_MIME_TYPES.has(file.mime_type)) {
                throw new Error(`xAI image generation does not support ${file.mime_type || 'untyped'} input files`);
            }
            if (segment.role === PromptRole.system || segment.role === PromptRole.safety) {
                throw new Error(`xAI image generation does not support image files in ${segment.role} input`);
            }
            imageCount += 1;
        }
    }
    if (!hasText) throw new Error('xAI image generation requires nonempty text input');
    if (imageCount > 5) throw new Error('xAI image editing supports at most five input images');
    const modelOptions = options.model_options as XAIGrokImageOptions | undefined;
    if (modelOptions?._option_id !== undefined && modelOptions._option_id !== 'xai-grok-image') {
        throw new Error(`xAI image generation does not support model options ${modelOptions._option_id}`);
    }
    if (
        modelOptions?.n !== undefined &&
        (!Number.isSafeInteger(modelOptions.n) || modelOptions.n < 1 || modelOptions.n > 10)
    ) {
        throw new Error('xAI image generation count must be an integer from one through ten');
    }
}

async function canonicalImageInput(prompt: ResponseInputItem[]): Promise<CanonicalImageInput> {
    const text: string[] = [];
    const images: CanonicalImageInput['images'] = [];
    for (const item of prompt) {
        if (
            !('role' in item) ||
            (item.role !== 'system' &&
                item.role !== 'developer' &&
                item.role !== 'user' &&
                item.role !== 'assistant') ||
            !('content' in item)
        ) {
            throw new Error('xAI canonical image generation received an unsupported prompt item');
        }
        if (typeof item.content === 'string') {
            if (item.content.trim().length > 0) text.push(item.content);
            continue;
        }
        if (!Array.isArray(item.content)) throw new Error('xAI canonical image input content is malformed');
        for (const part of item.content) {
            if (part.type === 'input_text') {
                if (part.text.trim().length > 0) text.push(part.text);
                continue;
            }
            if (part.type !== 'input_image' || !part.image_url) {
                throw new Error('xAI canonical image generation supports text and inline image inputs only');
            }
            const separator = part.image_url.indexOf(',');
            if (separator < 0 || !part.image_url.slice(0, separator).endsWith(';base64')) {
                throw new Error('xAI canonical image inputs must be base64 data URLs');
            }
            const mimeType = part.image_url.slice(5, part.image_url.indexOf(';', 5)).toLowerCase();
            if (!SUPPORTED_INPUT_MIME_TYPES.has(mimeType)) {
                throw new Error(`xAI canonical image generation does not support ${mimeType || 'untyped'} input`);
            }
            const data = part.image_url.slice(separator + 1);
            const verified = await verifiedBase64GeneratedImage(data, MAX_XAI_INPUT_IMAGE_BYTES, 'xAI input', mimeType);
            images.push({
                byte_length: verified.bytes.byteLength,
                content_hash: verified.content_hash,
                data,
                mime_type: mimeType,
            });
        }
    }
    const value = text.join('\n').trim();
    if (!value) throw new Error('xAI image generation requires nonempty user text');
    if (images.length > 5) throw new Error('xAI image editing supports at most five input images');
    return { text: value, images };
}

async function inputRecords(
    input: CanonicalImageInput,
    prompt: ResponseInputItem[],
    runtime: ReturnType<typeof resolveConversationRuntime>,
): Promise<{ turns: ConversationTurn[]; assets: Asset[]; mappings: NativeItemMapping[] }> {
    const turns: ConversationTurn[] = [];
    const assets: Asset[] = [];
    const mappings: NativeItemMapping[] = [];
    let textIndex = 0;
    let imageIndex = 0;
    for (let itemIndex = 0; itemIndex < prompt.length; itemIndex += 1) {
        const item = prompt[itemIndex];
        if (!('role' in item) || !('content' in item)) {
            throw new Error(`xAI canonical image prompt item ${itemIndex} is unsupported`);
        }
        const turnId = await deriveConversationId('turn', runtime.input_operation_id, String(itemIndex));
        const blocks: Array<TextBlock | ImageBlock> = [];
        const parts =
            typeof item.content === 'string'
                ? [{ type: 'input_text' as const, text: item.content }]
                : Array.isArray(item.content)
                  ? item.content
                  : [];
        for (let partIndex = 0; partIndex < parts.length; partIndex += 1) {
            const part = parts[partIndex];
            if (part.type === 'input_text') {
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
                mappings.push({
                    canonical_id: block.id,
                    native_id: `prompt/parts/${textIndex}`,
                    kind: 'block',
                });
                textIndex += 1;
                continue;
            }
            if (part.type !== 'input_image' || !part.image_url) {
                throw new Error('xAI canonical image generation supports text and inline image inputs only');
            }
            const image = input.images[imageIndex];
            if (image === undefined) throw new Error('xAI canonical image input mapping is incomplete');
            const assetId = await deriveConversationId('asset', runtime.input_operation_id, String(imageIndex));
            const blockId = await deriveConversationId(
                'block',
                runtime.input_operation_id,
                String(itemIndex),
                String(partIndex),
            );
            blocks.push({ id: blockId, type: 'image', asset_id: assetId });
            assets.push({
                id: assetId,
                kind: 'image',
                mime_type: image.mime_type,
                storage: { type: 'inline_base64', data: image.data },
                provenance: { type: 'received', source_turn_id: turnId },
                byte_length: image.byte_length,
                content_hash: image.content_hash,
                created_at: runtime.recorded_at,
            });
            mappings.push({
                canonical_id: blockId,
                native_id: input.images.length === 1 ? 'image' : `images/${imageIndex}`,
                kind: 'block',
            });
            imageIndex += 1;
        }
        if (blocks.length === 0) continue;
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
                ? createProgramTurn({
                      ...common,
                      authority: item.role,
                      blocks,
                  })
                : item.role === 'assistant'
                  ? buildConversationTurn({ ...common, kind: 'agent' })
                  : createUserTurn(common);
        turns.push(turn);
        mappings.push({ canonical_id: turn.id, native_id: `source/items/${itemIndex}`, kind: 'turn' });
    }
    if (imageIndex !== input.images.length) throw new Error('xAI canonical image input mapping is incomplete');
    return { turns, assets, mappings };
}

function effectiveOptions(endpoint: string, request: XAIImageRequest): JsonObject {
    const { model: _model, prompt: _prompt, image: _image, images: _images, ...parameters } = request;
    const imageCount = request.image === undefined ? (request.images?.length ?? 0) : 1;
    const endpointUrl = new URL(endpoint);
    if (endpointUrl.search.length > 0) {
        throw new Error('xAI canonical image endpoint must not contain query parameters');
    }
    endpointUrl.username = '';
    endpointUrl.password = '';
    endpointUrl.hash = '';
    return providerJsonValue({
        endpoint: endpointUrl.toString().replace(/\/+$/, ''),
        route: xAIImageEndpoint(request),
        parameters,
        input_image_count: imageCount,
    }) as JsonObject;
}

async function decodeResponseImage(
    image: NonNullable<XAIImageResponse['data']>[number],
    fetchImage: typeof fetch,
    maximumBytes: number,
    signal?: AbortSignal,
): Promise<DecodedXAIImage> {
    let verified: VerifiedGeneratedImage;
    if (image.b64_json !== undefined) {
        verified = await verifiedBase64GeneratedImage(image.b64_json, maximumBytes, 'xAI', image.mime_type);
    } else if (image.url !== undefined) {
        const url = new URL(image.url);
        if (url.protocol !== 'https:' && url.protocol !== 'http:') {
            throw new Error('xAI generated image URL must use HTTP or HTTPS');
        }
        verified = await readBoundedGeneratedImageResponse(
            await fetchImage(url, { signal }),
            maximumBytes,
            'xAI',
            signal,
        );
        if (image.mime_type !== undefined && image.mime_type !== verified.mime_type) {
            throw new Error(`xAI generated image MIME type ${image.mime_type} does not match its bytes`);
        }
    } else {
        throw new Error('xAI image response item contains neither base64 data nor a URL');
    }
    return {
        ...verified,
        ...(image.revised_prompt === undefined ? {} : { revised_prompt: image.revised_prompt }),
        ...(image.url === undefined ? {} : { source_url: image.url }),
    };
}

function costAmount(ticks: number): string {
    const fixed = (ticks / 10_000_000_000).toFixed(10);
    const withoutZeros = fixed.replace(/0+$/, '');
    return withoutZeros.endsWith('.') ? withoutZeros.slice(0, -1) : withoutZeros;
}

function canonicalUsage(usage: XAIImageResponse['usage']): GenerationUsage | undefined {
    const ticks = usage?.cost_in_usd_ticks;
    if (ticks === undefined) return undefined;
    if (!Number.isSafeInteger(ticks) || ticks < 0) throw new Error('xAI image response contains invalid usage cost');
    return {
        cost: { amount: costAmount(ticks), currency: 'USD', provenance: 'reported' },
        reported_usage: [
            {
                source: 'provider',
                protocol: XAI_IMAGES_PROTOCOL,
                accounting_basis: 'xai_image_cost_ticks',
                payload: providerJsonValue(usage),
            },
        ],
    };
}

export async function executeXAIImageCanonical(input: {
    endpoint: string;
    fetch_image: typeof fetch;
    invoke(request: XAIImageRequest, signal?: AbortSignal): Promise<XAIImageResponse>;
    options: ExecutionOptions;
    prompt: ResponseInputItem[];
    provider: string;
    signal?: AbortSignal;
}): Promise<CanonicalExecutionResponse> {
    const canonicalInput = await canonicalImageInput(input.prompt);
    const request = xAIImageRequest(input.prompt, input.options);
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
        throw new TypeError('xAI image generation does not support legacy conversation input');
    }
    if (
        input.options.conversation_runtime?.conversation_id !== undefined &&
        input.options.conversation_runtime.conversation_id !== document.id
    ) {
        throw new Error('conversation_runtime.conversation_id does not match the canonical document');
    }
    const acceptedBefore = acceptedCanonicalResponse(document, runtime.response_operation_id);
    const records = await inputRecords(canonicalInput, input.prompt, runtime);
    if (
        document.turns.length > 0 &&
        acceptedBefore === undefined &&
        (Object.keys(document.generations).length > 0 ||
            document.turns.some((turn) => !records.turns.some((record) => record.id === turn.id)))
    ) {
        throw new Error('xAI image generation does not support conversation continuation');
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
            assets: records.assets,
            context_entries: contextEntries,
            item_mappings: records.mappings,
        },
        { ...runtime, conversation_id: document.id },
        undefined,
        { prompt: canonicalInput.text, input_image_count: canonicalInput.images.length },
    );
    document = appended.document;
    const accepted = acceptedCanonicalResponse(document, runtime.response_operation_id);
    if (accepted !== undefined) {
        await assertAcceptedCanonicalRequest(
            { accepted_response: accepted, runtime },
            { provider: input.provider, protocol: XAI_IMAGES_PROTOCOL, model: input.options.model },
            requestJson,
        );
        const expectedOptions = effectiveOptions(input.endpoint, request);
        if (
            accepted.generation.request_receipt.target.options === undefined ||
            (await fingerprintJson(accepted.generation.request_receipt.target.options)) !==
                (await fingerprintJson(expectedOptions))
        ) {
            throw new Error(
                `Accepted response operation ${runtime.response_operation_id} has incompatible xAI target options`,
            );
        }
        if (input.options.include_original_response) {
            throw new Error('An idempotently recovered xAI image response cannot reconstruct original_response');
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
            protocol: XAI_IMAGES_PROTOCOL,
            model: input.options.model,
            adapter_version: XAI_IMAGES_ADAPTER_VERSION,
            options: effectiveOptions(input.endpoint, request),
        },
        requestJson,
        records.mappings,
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
    const response = await input.invoke(request, input.signal);
    input.signal?.throwIfAborted();
    if (!Array.isArray(response.data) || response.data.length === 0) {
        throw new Error('xAI image response contains no images');
    }
    const decodedImages: DecodedXAIImage[] = [];
    let totalBytes = 0;
    for (const image of response.data) {
        if (totalBytes >= maximumBytes)
            throw new Error(`xAI generated images exceed the ${maximumBytes} byte total limit`);
        const decoded = await decodeResponseImage(image, input.fetch_image, maximumBytes - totalBytes, input.signal);
        totalBytes += decoded.bytes.byteLength;
        decodedImages.push(decoded);
    }
    const completedAt = runtime.completed_at ?? runtime.recorded_at;
    const completedRuntime = { ...runtime, completed_at: completedAt };
    const assets: Asset[] = [];
    const blocks: AgentContentBlock[] = [];
    for (let index = 0; index < decodedImages.length; index += 1) {
        const decoded = decodedImages[index];
        const assetId = await deriveConversationId('asset', runtime.response_operation_id, String(index));
        assets.push({
            id: assetId,
            kind: 'image',
            mime_type: decoded.mime_type,
            storage: await generatedImageStorage(decoded, input.options, 'xAI', input.signal),
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
                          xai_image: {
                              ...(decoded.revised_prompt === undefined
                                  ? {}
                                  : { revised_prompt: decoded.revised_prompt }),
                              ...(decoded.source_url === undefined ? {} : { source_url: decoded.source_url }),
                          },
                      },
                  }),
        });
        blocks.push({
            id: await deriveConversationId('block', runtime.response_operation_id, String(index)),
            type: 'image',
            asset_id: assetId,
            ...(decoded.revised_prompt === undefined ? {} : { caption: decoded.revised_prompt }),
        });
    }
    const usage = canonicalUsage(response.usage);
    const generation = await createExecutedGeneration({
        id: identities.generation_id,
        runtime: completedRuntime,
        receipt,
        provider: input.provider,
        protocol: XAI_IMAGES_PROTOCOL,
        adapter_version: XAI_IMAGES_ADAPTER_VERSION,
        requested_model: input.options.model,
        resolved_model: input.options.model,
        finish_reason: 'stop',
        usage,
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
    const responseEvidence = providerJsonValue({
        created: response.created,
        data: decodedImages.map((image) => ({
            content_hash: image.content_hash,
            mime_type: image.mime_type,
            byte_length: image.bytes.byteLength,
            ...(image.revised_prompt === undefined ? {} : { revised_prompt: image.revised_prompt }),
            ...(image.source_url === undefined ? {} : { source_url: image.source_url }),
        })),
        ...(response.usage === undefined ? {} : { usage: response.usage }),
    });
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
