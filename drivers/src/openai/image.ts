import {
    type AgentContentBlock,
    type Asset,
    appendDecodedConversationResponseWithProcessing,
    buildConversationTurn,
    type ConversationDocument,
    type ConversationTurn,
    createProgramTurn,
    createTextBlock,
    createUserTurn,
    deriveConversationId,
    fingerprintJson,
    type GenerationUsage,
    inlineAssetContentIntegrity,
    isConversationDocumentFormat,
    type JsonObject,
    type JsonValue,
    type NativeItemMapping,
    parseConversationDocument,
    type RequestReceipt,
    type ResolvedConversationRuntimeContext,
    type ToolDefinition,
    type UserContentBlock,
} from '@llumiverse/conversation';
import {
    type CanonicalExecutionContextOptions,
    type CanonicalExecutionResponse,
    type CanonicalHostCapabilities,
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
    assertCanonicalContextProjection,
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
import {
    generatedImageStorage,
    maximumGeneratedImageOutputBytes,
    readBoundedGeneratedImageResponse,
    verifiedBase64GeneratedImage,
} from '../shared/generated-image.js';
import { imagePromptInputIdentity, imageRequest, imageRequestIdentity } from './images.js';
import { getImageMasks, setImageMasks } from './openai_format.js';
import {
    compileOpenAIResponsesConversation,
    type OpenAIResponsesMediaCaptionProjection,
    openAIResponsesImageInput,
} from './openai-responses-conversation-adapter.js';

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;

const OPENAI_IMAGES_PROTOCOL = 'openai.images.generate';
const OPENAI_IMAGES_ADAPTER_VERSION = '2026-09-30.canonical.3';
const OPENAI_IMAGES_COMPATIBLE_GENERATE_ADAPTER_VERSION = '2026-09-30.canonical.2';

interface DecodedImage {
    bytes: Uint8Array;
    mime_type: string;
    content_hash: string;
    revised_prompt?: string;
    source_url?: string;
}

interface OpenAIImageContextAssets {
    images: Asset[];
    mask?: Asset;
}

const OPENAI_IMAGE_EDIT_MASK_NAMESPACE = 'openai.images.edit_mask';

export interface OpenAIImageEditMaskBinding {
    turn_id: string;
    binding_block_id: string;
    image_block_id: string;
    asset: Asset;
}

function imageInputStorage(input: OpenAI.Responses.ResponseInputImage): {
    storage: Asset['storage'];
    mime_type: string;
    source_field: 'file_id' | 'image_url';
} {
    const inline = typeof input.image_url === 'string' ? /^data:([^;,]+);base64,(.*)$/s.exec(input.image_url) : null;
    if (inline !== null) {
        return {
            storage: { type: 'inline_base64', data: inline[2] },
            mime_type: inline[1],
            source_field: 'image_url',
        };
    }
    if (typeof input.image_url === 'string') {
        return {
            storage: { type: 'external', resolver: 'url', locator: { url: input.image_url } },
            mime_type: 'application/octet-stream',
            source_field: 'image_url',
        };
    }
    if (typeof input.file_id === 'string') {
        return {
            storage: { type: 'external', resolver: 'openai_file', locator: { file_id: input.file_id } },
            mime_type: 'application/octet-stream',
            source_field: 'file_id',
        };
    }
    throw new Error('OpenAI standalone image input has no supported source');
}

async function canonicalImageInput(input: {
    image: OpenAI.Responses.ResponseInputImage;
    runtime: ReturnType<typeof resolveConversationRuntime>;
    turn_id: string;
    identity: readonly string[];
}): Promise<{ block: UserContentBlock; asset: Asset }> {
    const blockId = await deriveConversationId('block', input.runtime.input_operation_id, ...input.identity);
    const assetId = await deriveConversationId('asset', input.runtime.input_operation_id, ...input.identity);
    const source = imageInputStorage(input.image);
    const integrity = await inlineAssetContentIntegrity(source.storage);
    return {
        block: { id: blockId, type: 'image', asset_id: assetId },
        asset: {
            id: assetId,
            kind: 'image',
            mime_type: source.mime_type,
            storage: source.storage,
            provenance: { type: 'received', source_turn_id: input.turn_id },
            ...(integrity ?? {}),
            created_at: input.runtime.recorded_at,
            metadata: {
                openai_images: {
                    source_field: source.source_field,
                    ...(input.image.detail === undefined ? {} : { detail: input.image.detail }),
                },
            },
        },
    };
}

/** Resolve the selected, provider-registered mask binding from JSON-persisted canonical state. */
export function openAIImageEditMaskBinding(input: unknown): OpenAIImageEditMaskBinding | undefined {
    const document = parseConversationDocument(input);
    let binding: OpenAIImageEditMaskBinding | undefined;
    for (const turn of selectedCanonicalTurns(document)) {
        const blocks = new Map(turn.blocks.map((block) => [block.id, block]));
        for (const block of turn.blocks) {
            if (block.type !== 'extension' || block.namespace !== OPENAI_IMAGE_EDIT_MASK_NAMESPACE) continue;
            if (binding !== undefined) throw new Error('OpenAI image context contains multiple edit mask bindings');
            if (
                block.version !== '1' ||
                block.model_projection !== 'registered' ||
                block.payload === null ||
                Array.isArray(block.payload) ||
                typeof block.payload !== 'object' ||
                Object.keys(block.payload).length !== 1 ||
                typeof block.payload.image_block_id !== 'string'
            ) {
                throw new Error('OpenAI image context contains an invalid edit mask binding');
            }
            const imageBlock = blocks.get(block.payload.image_block_id);
            if (imageBlock?.type !== 'image') {
                throw new Error('OpenAI image edit mask binding does not reference a selected image block');
            }
            const asset = Object.hasOwn(document.assets, imageBlock.asset_id)
                ? document.assets[imageBlock.asset_id]
                : undefined;
            if (asset?.kind !== 'image') {
                throw new Error('OpenAI image edit mask binding does not reference an image asset');
            }
            binding = {
                turn_id: turn.id,
                binding_block_id: block.id,
                image_block_id: imageBlock.id,
                asset,
            };
        }
    }
    return binding;
}

function isTextOnlyImageGenerationPrompt(prompt: ResponseInputItem[]): boolean {
    return prompt.every((item) => {
        if (!('role' in item) || !('content' in item)) return false;
        if (typeof item.content === 'string') return true;
        return Array.isArray(item.content) && item.content.every((part) => part.type === 'input_text');
    });
}

function assertAcceptedOpenAIImageAdapter(input: {
    accepted: NonNullable<ReturnType<typeof acceptedCanonicalResponse>>;
    document: unknown;
    prompt: ResponseInputItem[];
    response_operation_id: string;
}): void {
    const generationVersion = input.accepted.generation.adapter_version;
    const receiptVersion = input.accepted.generation.request_receipt.target.adapter_version;
    if (generationVersion !== receiptVersion) {
        throw new Error(
            `Accepted OpenAI image response operation ${input.response_operation_id} has mismatched adapter versions`,
        );
    }
    if (generationVersion === OPENAI_IMAGES_ADAPTER_VERSION) return;
    if (
        generationVersion === OPENAI_IMAGES_COMPATIBLE_GENERATE_ADAPTER_VERSION &&
        getImageMasks(input.prompt).length === 0 &&
        isTextOnlyImageGenerationPrompt(input.prompt) &&
        openAIImageEditMaskBinding(input.document) === undefined
    ) {
        // The v2 generate representation is byte-compatible for text-only requests. The append
        // receipt and native request fingerprint are still checked below before accepted recovery.
        return;
    }
    throw new Error(
        `Accepted OpenAI image response operation ${input.response_operation_id} uses unsupported adapter version ${generationVersion}`,
    );
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
            segment.role !== PromptRole.assistant &&
            segment.role !== PromptRole.mask
        ) {
            throw new Error(`OpenAI standalone image generation does not support ${segment.role} input`);
        }
        if (segment.role !== PromptRole.mask && segment.content.trim().length > 0) hasText = true;
        for (const file of segment.files ?? []) {
            if (!file.mime_type.startsWith('text/') && !file.mime_type.startsWith('image/')) {
                throw new Error(
                    `OpenAI standalone image generation does not support ${file.mime_type || 'untyped'} input files`,
                );
            }
            if (segment.role !== PromptRole.mask && file.mime_type.startsWith('text/')) hasText = true;
        }
    }
    if (!hasText) throw new Error('OpenAI standalone image generation requires nonempty text input');
}

function validateOpenAIImageContextPrompt(prompt: ResponseInputItem[]): void {
    let hasText = false;
    for (let itemIndex = 0; itemIndex < prompt.length; itemIndex += 1) {
        const item = prompt[itemIndex];
        if (!('role' in item) || !('content' in item)) {
            throw new TypeError(`OpenAI standalone image context item ${itemIndex} is unsupported`);
        }
        const parts =
            typeof item.content === 'string'
                ? [{ type: 'input_text' as const, text: item.content }]
                : Array.isArray(item.content)
                  ? item.content
                  : [];
        for (const part of parts) {
            if (part.type === 'input_text') {
                if (part.text.trim().length > 0) hasText = true;
                continue;
            }
            if (part.type !== 'input_image') {
                throw new TypeError(`OpenAI standalone image context does not support ${part.type} input`);
            }
        }
    }
    if (!hasText) throw new TypeError('OpenAI standalone image generation requires nonempty text input');
}

function generatedRevisedPrompt(
    document: ConversationDocument,
    turn: ConversationTurn,
    block: Extract<UserContentBlock | AgentContentBlock, { type: 'image' }>,
    asset: Asset,
): string | undefined {
    if (
        block.caption === undefined ||
        turn.kind !== 'agent' ||
        turn.provenance.type !== 'generated' ||
        !('generation_id' in turn) ||
        asset.kind !== 'image' ||
        asset.provenance.type !== 'generated' ||
        asset.provenance.generation_id !== turn.generation_id ||
        asset.provenance.source_turn_id !== turn.id
    ) {
        return undefined;
    }
    const generation = document.generations[turn.generation_id];
    if (
        generation?.record_source !== 'executed' ||
        generation.status !== 'completed' ||
        generation.protocol !== OPENAI_IMAGES_PROTOCOL ||
        generation.request_receipt.target.protocol !== OPENAI_IMAGES_PROTOCOL ||
        generation.request_receipt.target.provider !== generation.provider ||
        generation.request_receipt.target.model !== generation.requested_model
    ) {
        return undefined;
    }
    const metadata = asset.metadata?.openai_image;
    if (metadata === null || Array.isArray(metadata) || typeof metadata !== 'object') return undefined;
    const revisedPrompt = metadata.revised_prompt;
    return revisedPrompt === block.caption ? block.caption : undefined;
}

function imageCaptionProjections(
    document: ConversationDocument,
    selected: readonly ConversationTurn[],
): Map<string, OpenAIResponsesMediaCaptionProjection> {
    const replayDependencyBlockIds = new Set<string>();
    for (const turn of selected) {
        for (const block of turn.blocks) {
            if (block.type !== 'native_replay') continue;
            for (const blockId of block.dependencies.block_ids) replayDependencyBlockIds.add(blockId);
            for (const turnId of block.dependencies.turn_ids) {
                const dependency = document.turns.find((candidate) => candidate.id === turnId);
                for (const dependencyBlock of dependency?.blocks ?? [])
                    replayDependencyBlockIds.add(dependencyBlock.id);
            }
        }
    }
    const projections = new Map<string, OpenAIResponsesMediaCaptionProjection>();
    for (const turn of selected) {
        for (const block of turn.blocks) {
            if (block.type !== 'image' || block.caption === undefined || replayDependencyBlockIds.has(block.id))
                continue;
            const asset = document.assets[block.asset_id];
            if (asset === undefined) continue;
            if (generatedRevisedPrompt(document, turn, block, asset) !== undefined) {
                projections.set(block.id, { type: 'provenance' });
                continue;
            }
            if (
                turn.provenance.type === 'received' &&
                asset.provenance.type === 'received' &&
                (asset.provenance.source_turn_id === undefined || asset.provenance.source_turn_id === turn.id)
            ) {
                projections.set(block.id, { type: 'semantic_text', text: block.caption });
            }
        }
    }
    return projections;
}

function imageInputMatchesAsset(
    input: OpenAI.Responses.ResponseInputImage,
    asset: Asset,
    target: { provider: string; model: string },
): boolean {
    const expected = openAIResponsesImageInput(asset, target);
    return (
        input.file_id === expected.file_id && input.image_url === expected.image_url && input.detail === expected.detail
    );
}

function canonicalContextImageAssets(
    document: ConversationDocument,
    selected: readonly ConversationTurn[],
    prompt: readonly ResponseInputItem[],
    target: { provider: string; model: string },
    mask?: Asset,
): OpenAIImageContextAssets {
    const assets = selected.flatMap((turn) =>
        turn.blocks.flatMap((block) => {
            if (block.type !== 'image') return [];
            const asset = document.assets[block.asset_id];
            if (asset === undefined)
                throw new Error(`OpenAI standalone image references missing asset ${block.asset_id}`);
            return [asset];
        }),
    );
    const inputs = prompt.flatMap((item): OpenAI.Responses.ResponseInputImage[] => {
        if (!('content' in item) || !Array.isArray(item.content)) return [];
        return item.content.flatMap((part) => (part.type === 'input_image' ? [part] : []));
    });
    if (
        inputs.length !== assets.length ||
        inputs.some((input, index) => !imageInputMatchesAsset(input, assets[index], target))
    ) {
        throw new TypeError('OpenAI standalone image context media does not match its canonical assets');
    }
    return { images: assets, ...(mask === undefined ? {} : { mask }) };
}

function imageContextPrompt(
    document: ConversationDocument,
    target: { provider: string; model: string },
    options: { accepted_historical_flattened_authority: boolean },
): { prompt: ResponseInputItem[]; mappings: NativeItemMapping[]; assets: OpenAIImageContextAssets } {
    const selected = selectedCanonicalTurns(document);
    const captions = imageCaptionProjections(document, selected);
    if (!options.accepted_historical_flattened_authority) {
        assertCanonicalContextProjection(document, selected, {
            label: 'OpenAI standalone image',
            // The Images API flattens all text into one prompt string and has no privileged role channel.
            program_authorities: ['ordinary'],
            preserve_media_caption: (block) => captions.has(block.id),
        });
    }
    const maskBinding = openAIImageEditMaskBinding(document);
    let projectedDocument = document;
    if (maskBinding !== undefined) {
        const maskTurn = document.turns.find((turn) => turn.id === maskBinding.turn_id);
        const maskBlockIds = new Set([maskBinding.binding_block_id, maskBinding.image_block_id]);
        if (
            maskTurn === undefined ||
            maskTurn.blocks.length !== maskBlockIds.size ||
            maskTurn.blocks.some((block) => !maskBlockIds.has(block.id))
        ) {
            throw new TypeError('OpenAI image edit mask must use its dedicated canonical binding turn');
        }
        const removedEntries = document.context.entries.filter(
            (entry) => entry.type === 'source_turn' && entry.turn_id === maskBinding.turn_id,
        );
        const removedEntryIds = new Set(removedEntries.map((entry) => entry.id));
        if (document.context.protected_entry_ids.some((id) => removedEntryIds.has(id))) {
            throw new TypeError('OpenAI image edit mask binding cannot be a protected context entry');
        }
        projectedDocument = structuredClone(document);
        projectedDocument.context.entries = projectedDocument.context.entries.filter(
            (entry) => !(entry.type === 'source_turn' && entry.turn_id === maskBinding.turn_id),
        );
    }
    const projectedSelected = selectedCanonicalTurns(projectedDocument);
    const compiled = compileOpenAIResponsesConversation(projectedDocument, target, {
        project_media_caption: (block) => captions.get(block.id),
    });
    if (maskBinding !== undefined) {
        setImageMasks(compiled.conversation, [openAIResponsesImageInput(maskBinding.asset, target)]);
        compiled.mappings.push(
            { canonical_id: maskBinding.turn_id, native_id: 'request/mask', kind: 'turn' },
            { canonical_id: maskBinding.binding_block_id, native_id: 'request/mask/binding', kind: 'block' },
            { canonical_id: maskBinding.image_block_id, native_id: 'request/mask/image', kind: 'block' },
        );
    }
    validateOpenAIImageContextPrompt(compiled.conversation);
    return {
        prompt: compiled.conversation,
        mappings: compiled.mappings,
        assets: canonicalContextImageAssets(
            projectedDocument,
            projectedSelected,
            compiled.conversation,
            target,
            maskBinding?.asset,
        ),
    };
}

async function imagePromptRecords(
    prompt: ResponseInputItem[],
    runtime: ReturnType<typeof resolveConversationRuntime>,
): Promise<{ turns: ConversationTurn[]; assets: Asset[]; mappings: NativeItemMapping[] }> {
    const turns: ConversationTurn[] = [];
    const assets: Asset[] = [];
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
        const turnId = await deriveConversationId('turn', runtime.input_operation_id, String(itemIndex));
        const blocks: UserContentBlock[] = [];
        for (let partIndex = 0; partIndex < parts.length; partIndex += 1) {
            const part = parts[partIndex];
            const nativeId = `prompt/parts/${textIndex}`;
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
                mappings.push({ canonical_id: block.id, native_id: nativeId, kind: 'block' });
                textIndex += 1;
                continue;
            }
            if (part.type !== 'input_image') {
                throw new Error('OpenAI standalone image generation received unsupported input content');
            }
            const record = await canonicalImageInput({
                image: part,
                runtime,
                turn_id: turnId,
                identity: [String(itemIndex), String(partIndex)],
            });
            assets.push(record.asset);
            blocks.push(record.block);
            mappings.push({ canonical_id: record.block.id, native_id: nativeId, kind: 'block' });
            textIndex += 1;
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
                ? createProgramTurn({ ...common, authority: item.role })
                : item.role === 'assistant'
                  ? buildConversationTurn({ ...common, kind: 'agent' })
                  : createUserTurn(common);
        turns.push(turn);
        mappings.push({ canonical_id: turn.id, native_id: `source/items/${itemIndex}`, kind: 'turn' });
    }

    const masks = getImageMasks(prompt);
    if (masks.length > 1) throw new Error('Only one image mask is supported');
    const mask = masks[0];
    if (mask !== undefined) {
        const turnId = await deriveConversationId('turn', runtime.input_operation_id, 'openai-image-mask');
        const record = await canonicalImageInput({
            image: mask,
            runtime,
            turn_id: turnId,
            identity: ['openai-image-mask'],
        });
        const bindingBlock = {
            id: await deriveConversationId('block', runtime.input_operation_id, 'openai-image-mask-binding'),
            type: 'extension' as const,
            namespace: OPENAI_IMAGE_EDIT_MASK_NAMESPACE,
            version: '1',
            payload: { image_block_id: record.block.id },
            model_projection: 'registered' as const,
        };
        const turn = createUserTurn({
            id: turnId,
            authority: 'ordinary',
            model_visibility: 'include',
            status: 'completed',
            timestamps: { recorded_at: runtime.recorded_at },
            provenance: { type: 'received' },
            blocks: [bindingBlock, record.block],
        });
        turns.push(turn);
        assets.push(record.asset);
        mappings.push(
            { canonical_id: turn.id, native_id: 'request/mask', kind: 'turn' },
            { canonical_id: bindingBlock.id, native_id: 'request/mask/binding', kind: 'block' },
            { canonical_id: record.block.id, native_id: 'request/mask/image', kind: 'block' },
        );
    }
    return { turns, assets, mappings };
}

function effectiveImageOptions(request: JsonValue): JsonObject {
    if (request === null || typeof request !== 'object' || Array.isArray(request)) return {};
    const { model: _model, prompt: _prompt, images: _images, mask: _mask, ...options } = request;
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

interface PreparedOpenAIImageExecution {
    document: ConversationDocument;
    runtime: ResolvedConversationRuntimeContext;
    receipt: RequestReceipt;
    generation_id: string;
    response_turn_id: string;
    tool_definitions: ToolDefinition[];
}

async function executePreparedOpenAIImageCanonical(
    input: {
        service: OpenAI;
        provider: string;
        prompt: ResponseInputItem[];
        options: ExecutionOptions | CanonicalExecutionContextOptions;
        request_model: string;
        source_model: string;
        fetch_image: typeof fetch;
        request_options?: { signal?: AbortSignal; timeout?: number };
        signal?: AbortSignal;
        request_json: JsonValue;
        context_assets?: OpenAIImageContextAssets;
        host_capabilities?: CanonicalHostCapabilities;
    },
    prepared: PreparedOpenAIImageExecution,
): Promise<CanonicalExecutionResponse> {
    const modelOptions = input.options.model_options as OpenAiDalleOptions | OpenAiGptImageOptions | undefined;
    const maximumBytes = maximumGeneratedImageOutputBytes(prepared.document, input.options);
    await publishCanonicalPreparedRequest(prepared, input.options);
    input.signal?.throwIfAborted();
    const request = await imageRequest(
        input.service,
        input.prompt,
        input.request_model,
        modelOptions,
        input.source_model,
        input.request_options,
        input.fetch_image,
        { preserve_omitted_defaults: true },
        input.context_assets === undefined
            ? undefined
            : {
                  ...input.context_assets,
                  resolve_asset:
                      input.host_capabilities?.resolve_canonical_asset ??
                      ('resolve_canonical_asset' in input.options ? input.options.resolve_canonical_asset : undefined),
              },
    );
    input.signal?.throwIfAborted();
    const response = request.edit
        ? input.request_options
            ? await input.service.images.edit(request.edit, input.request_options)
            : await input.service.images.edit(request.edit)
        : input.request_options
          ? await input.service.images.generate(request.generate, input.request_options)
          : await input.service.images.generate(request.generate);
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
    const completedAt = prepared.runtime.completed_at ?? prepared.runtime.recorded_at;
    const completedRuntime = { ...prepared.runtime, completed_at: completedAt };
    const assets: Asset[] = [];
    const blocks: AgentContentBlock[] = [];
    for (let index = 0; index < decodedImages.length; index += 1) {
        const decoded = decodedImages[index];
        const assetId = await deriveConversationId('asset', prepared.runtime.response_operation_id, String(index));
        const responseBlockId = await deriveConversationId(
            'block',
            prepared.runtime.response_operation_id,
            String(index),
        );
        assets.push({
            id: assetId,
            kind: 'image',
            mime_type: decoded.mime_type,
            storage: await generatedImageStorage(decoded, input.options, 'OpenAI', input.signal),
            provenance: {
                type: 'generated',
                generation_id: prepared.generation_id,
                source_turn_id: prepared.response_turn_id,
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
        id: prepared.generation_id,
        runtime: completedRuntime,
        receipt: prepared.receipt,
        provider: input.provider,
        protocol: OPENAI_IMAGES_PROTOCOL,
        adapter_version: OPENAI_IMAGES_ADAPTER_VERSION,
        requested_model: input.options.model,
        resolved_model: input.options.model,
        finish_reason: 'stop',
        usage: canonicalImageUsage(response.usage),
    });
    const responseTurn = {
        id: prepared.response_turn_id,
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
    const finalDocument = (
        await appendDecodedConversationResponseWithProcessing(
            {
                document: prepared.document,
                generation_id: prepared.generation_id,
                response_turn_id: prepared.response_turn_id,
                receipt: prepared.receipt,
                payload: input.request_json,
                diagnostics: [],
            },
            {
                turns: [responseTurn],
                assets,
                generation,
                diagnostics: [],
                payload_fingerprint: await fingerprintJson(responseEvidence),
            },
            { operation_id: prepared.runtime.response_operation_id, recorded_at: completedAt },
        )
    ).document;
    return createCanonicalExecutionResponse(finalDocument, prepared.runtime.response_operation_id, {
        ...(input.options.include_original_response ? { original_response: response } : {}),
    });
}

export async function executeOpenAIImageCanonical(input: {
    service: OpenAI;
    provider: string;
    prompt: ResponseInputItem[];
    options: ExecutionOptions;
    request_model: string;
    source_model: string;
    fetch_image: typeof fetch;
    request_options?: { signal?: AbortSignal; timeout?: number };
    signal?: AbortSignal;
}): Promise<CanonicalExecutionResponse> {
    const modelOptions = input.options.model_options as OpenAiDalleOptions | OpenAiGptImageOptions | undefined;
    const requestJson = providerJsonValue(
        imageRequestIdentity(input.prompt, input.request_model, modelOptions, input.source_model, {
            preserve_omitted_defaults: true,
        }),
    );
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
    if (acceptedBefore !== undefined) {
        assertAcceptedOpenAIImageAdapter({
            accepted: acceptedBefore,
            document,
            prompt: input.prompt,
            response_operation_id: runtime.response_operation_id,
        });
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
            assets: promptRecords.assets,
            context_entries: contextEntries,
            item_mappings: promptRecords.mappings,
        },
        { ...runtime, conversation_id: document.id },
        undefined,
        providerJsonValue(imagePromptInputIdentity(input.prompt)),
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
            options: effectiveImageOptions(requestJson),
        },
        requestJson,
        promptRecords.mappings,
        appended.tool_definitions,
    );
    const identities = await canonicalResponseIdentities(runtime);
    return executePreparedOpenAIImageCanonical(
        { ...input, request_json: requestJson },
        {
            document,
            runtime: { ...runtime, conversation_id: document.id },
            receipt,
            generation_id: identities.generation_id,
            response_turn_id: identities.response_turn_id,
            tool_definitions: appended.tool_definitions,
        },
    );
}

/** Execute a standalone image request from retained canonical context without authoring prompt segments. */
export async function executeOpenAIImageCanonicalContext(input: {
    service: OpenAI;
    provider: string;
    options: CanonicalExecutionContextOptions;
    host_capabilities?: CanonicalHostCapabilities;
    request_model: string;
    source_model: string;
    fetch_image: typeof fetch;
    request_options?: { signal?: AbortSignal; timeout?: number };
    signal?: AbortSignal;
}): Promise<CanonicalExecutionResponse> {
    if (input.options.result_schema !== undefined) {
        throw new TypeError('OpenAI standalone image generation does not support structured output');
    }
    const suppliedDocument = parseConversationDocument(input.options.conversation);
    const suppliedAccepted = acceptedCanonicalResponse(
        suppliedDocument,
        input.options.conversation_runtime.response_operation_id,
    );
    const acceptedAdapterVersion = suppliedAccepted?.generation.adapter_version;
    const preparationAdapterVersion =
        acceptedAdapterVersion === OPENAI_IMAGES_COMPATIBLE_GENERATE_ADAPTER_VERSION
            ? OPENAI_IMAGES_COMPATIBLE_GENERATE_ADAPTER_VERSION
            : OPENAI_IMAGES_ADAPTER_VERSION;
    const prepared = await prepareCanonicalContext({
        options: input.options,
        provider: input.provider,
        protocol: OPENAI_IMAGES_PROTOCOL,
        adapter_version: preparationAdapterVersion,
    });
    if (prepared.tool_definitions.length > 0) {
        throw new TypeError('OpenAI standalone image generation does not support active tool definitions');
    }
    const compiled = imageContextPrompt(
        prepared.request_document,
        {
            provider: input.provider,
            model: input.options.model,
        },
        {
            // Published image executions accepted privileged authoring segments even though the Images API flattened
            // them into one prompt. Permit only exact accepted-response recovery; the retained request fingerprint below
            // proves the same historical native request and no provider transport is opened.
            accepted_historical_flattened_authority: prepared.accepted_response !== undefined,
        },
    );
    const modelOptions = input.options.model_options as OpenAiDalleOptions | OpenAiGptImageOptions | undefined;
    const requestJson = providerJsonValue(
        imageRequestIdentity(compiled.prompt, input.request_model, modelOptions, input.source_model, {
            preserve_omitted_defaults: true,
        }),
    );
    if (prepared.accepted_response !== undefined) {
        assertAcceptedOpenAIImageAdapter({
            accepted: prepared.accepted_response,
            document: prepared.request_document,
            prompt: compiled.prompt,
            response_operation_id: prepared.runtime.response_operation_id,
        });
    }
    await assertAcceptedCanonicalRequest(
        prepared,
        { provider: input.provider, protocol: OPENAI_IMAGES_PROTOCOL, model: input.options.model },
        requestJson,
    );
    if (prepared.accepted_response !== undefined) {
        if (input.options.include_original_response) {
            throw new Error('An idempotently recovered image response cannot reconstruct original_response');
        }
        return recoverCanonicalExecutionResponse(prepared, input.options);
    }
    const receipt = await createRequestReceipt(
        prepared.document,
        prepared.runtime,
        {
            provider: input.provider,
            protocol: OPENAI_IMAGES_PROTOCOL,
            model: input.options.model,
            adapter_version: OPENAI_IMAGES_ADAPTER_VERSION,
            options: effectiveImageOptions(requestJson),
        },
        requestJson,
        compiled.mappings,
        prepared.tool_definitions,
    );
    return executePreparedOpenAIImageCanonical(
        { ...input, prompt: compiled.prompt, request_json: requestJson, context_assets: compiled.assets },
        {
            document: prepared.document,
            runtime: prepared.runtime,
            receipt,
            generation_id: prepared.generation_id,
            response_turn_id: prepared.response_turn_id,
            tool_definitions: prepared.tool_definitions,
        },
    );
}
