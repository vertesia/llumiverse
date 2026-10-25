import { isOpenAIImageVersionGTE, type OpenAiDalleOptions, type OpenAiGptImageOptions } from '@llumiverse/core';
import type OpenAI from 'openai';
import { toFile } from 'openai';
import { getImageMasks } from './openai_format.js';

type Input = OpenAI.Responses.ResponseInputItem[];
type ImageInput = Pick<OpenAI.Responses.ResponseInputImage, 'file_id' | 'image_url'>;
type ImageOptions = OpenAiDalleOptions | OpenAiGptImageOptions;

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

/** Translate Responses inputs to SDK image parameters, retaining reference order. */
export async function imageRequest(
    service: OpenAI,
    prompt: Input,
    model: string,
    options: ImageOptions | undefined,
    sourceModel: string,
    requestOptions: OpenAI.RequestOptions = {},
    fetcher: typeof fetch = fetch,
): Promise<
    | { generate: OpenAI.Images.ImageGenerateParamsNonStreaming; edit?: never }
    | { edit: OpenAI.Images.ImageEditParamsNonStreaming; generate?: never }
> {
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
                  quality: gpt?.image_quality ?? 'auto',
                  background: gpt?.background,
                  output_format: gpt?.output_format ?? 'png',
                  output_compression: gpt?.output_compression,
                  moderation: gpt?.moderation,
                  partial_images: gpt?.partial_images ?? 0,
              }),
    };
    if (!images.length) return { generate: common };
    const { moderation: _moderation, response_format: _format, style: _style, ...editCommon } = common;
    const edit: OpenAI.Images.ImageEditParamsNonStreaming = {
        ...editCommon,
        quality: common.quality === 'hd' ? 'standard' : common.quality,
        image: await Promise.all(images.map((image) => imageFile(image, service, requestOptions, fetcher))),
        mask: masks[0] ? await imageFile(masks[0], service, requestOptions, fetcher) : undefined,
        input_fidelity:
            isOpenAIImageVersionGTE(sourceModel, 2) && !isOpenAIImageVersionGTE(sourceModel, 2, 5)
                ? undefined
                : gpt?.input_fidelity,
    };
    return { edit };
}
