import { type Asset, type ResolveConversationAsset, readBoundedConversationAsset } from '@llumiverse/conversation';
import { isOpenAIImageVersionGTE, type OpenAiDalleOptions, type OpenAiGptImageOptions } from '@llumiverse/core';
import type OpenAI from 'openai';
import { toFile } from 'openai';
import { getImageMasks } from './openai_format.js';

type Input = OpenAI.Responses.ResponseInputItem[];
type ImageInput = Pick<OpenAI.Responses.ResponseInputImage, 'file_id' | 'image_url'>;
type ImageOptions = OpenAiDalleOptions | OpenAiGptImageOptions;

interface ImageRequestPlan {
    common: OpenAI.Images.ImageGenerateParamsNonStreaming;
    images: ImageInput[];
    mask?: ImageInput;
    input_fidelity?: OpenAI.Images.ImageEditParamsNonStreaming['input_fidelity'];
}

export interface CanonicalImageRequestAssets {
    images: readonly Asset[];
    mask?: Asset;
    resolve_asset?: ResolveConversationAsset;
}

const MAX_CANONICAL_IMAGE_INPUT_BYTES = 50 * 1024 * 1024;
const MAX_CANONICAL_IMAGE_INPUT_CHUNKS = 16_384;

export interface ImageRequestPlanningOptions {
    /** Keep the older canonical generate payload byte-compatible when callers omit newer SDK defaults. */
    preserve_omitted_defaults?: boolean;
}

export function imageDataUrl(data: string, returned?: string | null, requested?: string | null): string {
    return data.startsWith('data:') ? data : `data:image/${returned ?? requested ?? 'png'};base64,${data}`;
}

async function imageFile(
    image: ImageInput,
    service: OpenAI,
    requestOptions: OpenAI.RequestOptions,
    fetcher: typeof fetch,
) {
    const deadline = requestOptions.timeout === undefined ? undefined : AbortSignal.timeout(requestOptions.timeout);
    const signals = [requestOptions.signal, deadline].filter((signal): signal is AbortSignal => !!signal);
    const signal = signals.length ? AbortSignal.any(signals) : undefined;
    if (image.file_id) {
        const response = await service.files.content(image.file_id, { ...requestOptions, signal });
        return toFile(await response.blob(), `${image.file_id}.png`);
    }
    if (!image.image_url) throw new Error('Image input requires a URL or file ID');
    const response = await fetcher(image.image_url, { signal });
    if (!response.ok) throw new Error(`Could not read image input: HTTP ${response.status}`);
    const blob = await response.blob();
    return toFile(blob, `image.${blob.type.split('/')[1] || 'png'}`, { type: blob.type });
}

function requestSignal(requestOptions: OpenAI.RequestOptions): AbortSignal | undefined {
    const deadline = requestOptions.timeout === undefined ? undefined : AbortSignal.timeout(requestOptions.timeout);
    const signals = [requestOptions.signal, deadline].filter((signal): signal is AbortSignal => !!signal);
    return signals.length ? AbortSignal.any(signals) : undefined;
}

async function* responseBody(response: Response, label: string, signal?: AbortSignal): AsyncIterable<Uint8Array> {
    if (!response.ok) {
        await response.body?.cancel().catch(() => undefined);
        throw new Error(`${label} failed with HTTP ${response.status}`);
    }
    if (response.body === null) throw new Error(`${label} returned no body`);
    const reader = response.body.getReader();
    let completed = false;
    const abort = () => {
        void reader.cancel(signal?.reason).catch(() => undefined);
    };
    signal?.addEventListener('abort', abort, { once: true });
    try {
        while (true) {
            signal?.throwIfAborted();
            const item = await reader.read();
            if (item.done) {
                completed = true;
                break;
            }
            yield item.value;
        }
    } finally {
        signal?.removeEventListener('abort', abort);
        if (!completed) await reader.cancel().catch(() => undefined);
        reader.releaseLock();
    }
}

function decodeInlineAsset(asset: Asset): Uint8Array {
    if (asset.storage.type !== 'inline_base64') throw new TypeError(`Canonical image asset ${asset.id} is not inline`);
    const data = asset.storage.data;
    const bytes = new Uint8Array(Buffer.from(data, 'base64'));
    if (bytes.byteLength === 0 || Buffer.from(bytes).toString('base64') !== data) {
        throw new Error(`Canonical image asset ${asset.id} contains malformed base64 data`);
    }
    return bytes;
}

function assertCanonicalImageAssets(input: CanonicalImageRequestAssets, plan: ImageRequestPlan): readonly Asset[] {
    if (input.images.length !== plan.images.length) {
        throw new TypeError('Canonical image input assets do not match the native edit request');
    }
    if ((input.mask === undefined) !== (plan.mask === undefined)) {
        throw new TypeError('Canonical image mask asset does not match the native edit request');
    }
    const assets = [...input.images, ...(input.mask === undefined ? [] : [input.mask])];
    let declaredBytes = 0;
    for (const asset of assets) {
        if (asset.kind !== 'image') throw new TypeError(`Canonical image input asset ${asset.id} is not an image`);
        if (!asset.mime_type.startsWith('image/')) {
            throw new TypeError(`Canonical image input asset ${asset.id} has unsupported MIME type ${asset.mime_type}`);
        }
        if (asset.byte_length === undefined || asset.content_hash === undefined) {
            throw new Error(`Canonical image input asset ${asset.id} requires declared byte_length and content_hash`);
        }
        if (asset.byte_length > MAX_CANONICAL_IMAGE_INPUT_BYTES - declaredBytes) {
            throw new RangeError(
                `Canonical image inputs exceed the ${MAX_CANONICAL_IMAGE_INPUT_BYTES} byte aggregate limit`,
            );
        }
        declaredBytes += asset.byte_length;
    }
    return assets;
}

function canonicalAssetResolver(
    service: OpenAI,
    requestOptions: OpenAI.RequestOptions,
    fetcher: typeof fetch,
    fallback?: ResolveConversationAsset,
): ResolveConversationAsset {
    return async (asset, suppliedSignal) => {
        const signal = suppliedSignal ?? requestSignal(requestOptions);
        if (asset.storage.type === 'inline_base64') {
            const bytes = decodeInlineAsset(asset);
            return (async function* () {
                yield bytes;
            })();
        }
        if (asset.storage.type === 'external' && asset.storage.resolver === 'openai_file') {
            const fileId = asset.storage.locator.file_id;
            if (typeof fileId !== 'string') {
                throw new TypeError(`Canonical image asset ${asset.id} has no OpenAI file ID`);
            }
            const response = await service.files.content(fileId, { ...requestOptions, signal });
            return responseBody(response, `Canonical image asset ${asset.id} download`, signal);
        }
        if (asset.storage.type === 'external' && asset.storage.resolver === 'url') {
            const url = asset.storage.locator.url;
            if (typeof url !== 'string') throw new TypeError(`Canonical image asset ${asset.id} has no URL`);
            if (url.startsWith('https:') || url.startsWith('http:') || url.startsWith('data:')) {
                return responseBody(
                    await fetcher(url, { signal }),
                    `Canonical image asset ${asset.id} download`,
                    signal,
                );
            }
        }
        if (fallback === undefined) {
            throw new TypeError(`Canonical image asset ${asset.id} requires a host asset resolver`);
        }
        return fallback(structuredClone(asset), signal);
    };
}

async function canonicalImageFiles(
    assets: CanonicalImageRequestAssets,
    plan: ImageRequestPlan,
    service: OpenAI,
    requestOptions: OpenAI.RequestOptions,
    fetcher: typeof fetch,
): Promise<{ images: File[]; mask?: File }> {
    assertCanonicalImageAssets(assets, plan);
    const signal = requestSignal(requestOptions);
    const resolveAsset = canonicalAssetResolver(service, requestOptions, fetcher, assets.resolve_asset);
    let remainingBytes = MAX_CANONICAL_IMAGE_INPUT_BYTES;
    const read = async (asset: Asset): Promise<File> => {
        const bytes = await readBoundedConversationAsset(asset, resolveAsset, {
            max_bytes: remainingBytes,
            max_chunks: MAX_CANONICAL_IMAGE_INPUT_CHUNKS,
            require_integrity: true,
            label: 'Canonical image input asset',
            signal,
        });
        remainingBytes -= bytes.byteLength;
        const extension = asset.mime_type.split('/')[1]?.split('+')[0] || 'bin';
        return toFile(bytes, `${asset.id}.${extension}`, { type: asset.mime_type });
    };
    const images: File[] = [];
    for (const asset of assets.images) images.push(await read(asset));
    const mask = assets.mask === undefined ? undefined : await read(assets.mask);
    return { images, ...(mask === undefined ? {} : { mask }) };
}

function imageInputReference(image: ImageInput): ImageInput {
    return {
        ...(image.file_id === undefined ? {} : { file_id: image.file_id }),
        ...(image.image_url === undefined ? {} : { image_url: image.image_url }),
    };
}

/**
 * Canonical input-operation identity for the authored image prompt.
 *
 * Masks are carried outside the Responses prompt array by the legacy formatter. Include their
 * stable references when present, while retaining the historical prompt-only identity otherwise.
 */
export function imagePromptInputIdentity(prompt: Input): Input | { prompt: Input; masks: ImageInput[] } {
    const masks = getImageMasks(prompt);
    return masks.length === 0 ? prompt : { prompt, masks: masks.map(imageInputReference) };
}

function imageRequestPlan(
    prompt: Input,
    model: string,
    options: ImageOptions | undefined,
    sourceModel: string,
    planning: ImageRequestPlanningOptions = {},
): ImageRequestPlan {
    const texts: string[] = [];
    const images: ImageInput[] = [];
    for (const item of prompt) {
        const content =
            'content' in item
                ? item.content
                : 'type' in item && item.type === 'function_call_output'
                  ? item.output
                  : undefined;
        if (typeof content === 'string') texts.push(content);
        else if (Array.isArray(content)) {
            for (const part of content) {
                if (part.type === 'input_text') texts.push(part.text);
                else if (part.type === 'input_image') images.push(part);
            }
        }
    }
    const masks = getImageMasks(prompt);
    if (masks.length > 1) throw new Error('Only one image mask is supported');
    if (masks.length && !images.length) throw new Error('An image mask requires a reference image');
    const gpt = options as OpenAiGptImageOptions | undefined;
    const legacy = sourceModel.toLowerCase().includes('dall-e');
    const preserveOmittedDefaults = planning.preserve_omitted_defaults === true;
    const common: OpenAI.Images.ImageGenerateParamsNonStreaming = {
        model,
        prompt: texts.join('\n'),
        size:
            !legacy && (gpt?.width !== undefined || gpt?.height !== undefined)
                ? `${gpt?.width ?? 1024}x${gpt?.height ?? 1024}`
                : (options?.size ?? '1024x1024'),
        n: options?.n ?? 1,
        ...(legacy
            ? {
                  quality: (options as OpenAiDalleOptions | undefined)?.image_quality ?? 'standard',
                  response_format: (options as OpenAiDalleOptions | undefined)?.response_format ?? 'b64_json',
                  style: (options as OpenAiDalleOptions | undefined)?.style,
              }
            : {
                  quality: gpt?.image_quality ?? (preserveOmittedDefaults ? undefined : 'auto'),
                  background: gpt?.background,
                  output_format: gpt?.output_format ?? (preserveOmittedDefaults ? undefined : 'png'),
                  output_compression: gpt?.output_compression,
                  moderation: gpt?.moderation,
                  partial_images: gpt?.partial_images ?? (preserveOmittedDefaults ? undefined : 0),
              }),
    };
    return {
        common,
        images,
        ...(masks[0] === undefined ? {} : { mask: masks[0] }),
        ...(!legacy && (!isOpenAIImageVersionGTE(sourceModel, 2) || isOpenAIImageVersionGTE(sourceModel, 2, 5))
            ? { input_fidelity: gpt?.input_fidelity }
            : {}),
    };
}

/**
 * JSON-native semantic image request used for canonical request identity.
 * Multipart uploads remain references here so accepted recovery never downloads media again.
 */
export function imageRequestIdentity(
    prompt: Input,
    model: string,
    options: ImageOptions | undefined,
    sourceModel: string,
    planning: ImageRequestPlanningOptions = {},
): unknown {
    const plan = imageRequestPlan(prompt, model, options, sourceModel, planning);
    if (plan.images.length === 0) return plan.common;
    const { moderation: _moderation, response_format: _format, style: _style, ...editCommon } = plan.common;
    return {
        ...editCommon,
        quality: plan.common.quality === 'hd' ? 'standard' : plan.common.quality,
        images: plan.images.map(imageInputReference),
        ...(plan.mask === undefined ? {} : { mask: imageInputReference(plan.mask) }),
        ...(plan.input_fidelity === undefined ? {} : { input_fidelity: plan.input_fidelity }),
    };
}

/** Translate Responses inputs to SDK image parameters, retaining reference order. */
export async function imageRequest(
    service: OpenAI,
    prompt: Input,
    model: string,
    options: ImageOptions | undefined,
    sourceModel: string,
    requestOptions: OpenAI.RequestOptions = {},
    fetcher: typeof fetch = fetch,
    planning: ImageRequestPlanningOptions = {},
    canonicalAssets?: CanonicalImageRequestAssets,
): Promise<
    | { generate: OpenAI.Images.ImageGenerateParamsNonStreaming; edit?: never }
    | { edit: OpenAI.Images.ImageEditParamsNonStreaming; generate?: never }
> {
    const plan = imageRequestPlan(prompt, model, options, sourceModel, planning);
    if (!plan.images.length) return { generate: plan.common };
    const { moderation: _moderation, response_format: _format, style: _style, ...editCommon } = plan.common;
    const resolved =
        canonicalAssets === undefined
            ? undefined
            : await canonicalImageFiles(canonicalAssets, plan, service, requestOptions, fetcher);
    const edit: OpenAI.Images.ImageEditParamsNonStreaming = {
        ...editCommon,
        quality: plan.common.quality === 'hd' ? 'standard' : plan.common.quality,
        image:
            resolved?.images ??
            (await Promise.all(plan.images.map((image) => imageFile(image, service, requestOptions, fetcher)))),
        mask: resolved?.mask ?? (plan.mask ? await imageFile(plan.mask, service, requestOptions, fetcher) : undefined),
        input_fidelity: plan.input_fidelity,
    };
    return { edit };
}
