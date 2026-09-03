import type { Interactions } from '@google/genai';
import {
    type AIModel,
    type Completion,
    type CompletionResult,
    type DriverCompletionStream,
    type ExecutionOptions,
    type ExecutionTokenUsage,
    LlumiverseError,
    type LlumiverseErrorContext,
    ModelType,
    type PromptSegment,
    type VertexAIGeminiOmniVideoOptions,
} from '@llumiverse/core';
import type { VertexAIDriver } from '../index.js';
import type { ModelDefinition } from '../models.js';

export const GEMINI_OMNI_VIDEO_MODEL = 'gemini-omni-flash-preview';
export const GEMINI_OMNI_1_1_VIDEO_MODEL = 'gemini-omni-1.1-flash-preview';
export const GEMINI_OMNI_VIDEO_MODELS = [GEMINI_OMNI_VIDEO_MODEL, GEMINI_OMNI_1_1_VIDEO_MODEL] as const;

export type GeminiOmniVideoModel = (typeof GEMINI_OMNI_VIDEO_MODELS)[number];

export function isGeminiOmniVideoModel(model: string): model is GeminiOmniVideoModel {
    return GEMINI_OMNI_VIDEO_MODELS.some((candidate) => candidate === model);
}

const SUPPORTED_IMAGE_MIME_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/heic', 'image/heif']);
const SUPPORTED_VIDEO_MIME_TYPES = new Set([
    'video/3gpp',
    'video/mp4',
    'video/mpeg',
    'video/mpegs',
    'video/mpg',
    'video/quicktime',
    'video/webm',
    'video/wmv',
    'video/x-flv',
    'video/x-ms-wmv',
]);

type OmniVideoMediaInput = Extract<Interactions.Content, { type: 'image' | 'video' }>;

export interface OmniVideoPrompt {
    text: string;
    media: OmniVideoMediaInput[];
}

class OmniVideoTerminalError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'OmniVideoTerminalError';
    }
}

class OmniVideoInteractionError extends Error {
    constructor(
        message: string,
        readonly retryable: boolean | undefined,
    ) {
        super(message);
        this.name = 'OmniVideoInteractionError';
    }
}

function isGcsUri(value: string, requireObject = false): boolean {
    const match = /^gs:\/\/([^/]+)\/(.*)$/.exec(value);
    return !!match && (!requireObject || match[2].length > 0);
}

function normalizeOutputPrefix(value: string | undefined): string {
    if (!value || !isGcsUri(value)) {
        throw new OmniVideoTerminalError('Gemini Omni video generation requires a valid GCS output prefix');
    }
    return value.endsWith('/') ? value : `${value}/`;
}

function resolveTask(
    model: GeminiOmniVideoModel,
    options: VertexAIGeminiOmniVideoOptions | undefined,
    media: OmniVideoMediaInput[],
) {
    const imageCount = media.filter((item) => item.type === 'image').length;
    const videoCount = media.filter((item) => item.type === 'video').length;
    const task = options?.task ?? (media.length === 0 ? 'text_to_video' : undefined);
    if (!task) {
        throw new OmniVideoTerminalError('Gemini Omni video generation with media requires an explicit task');
    }
    if (task === 'extend' && model !== GEMINI_OMNI_1_1_VIDEO_MODEL) {
        throw new OmniVideoTerminalError(`${model} does not support extend`);
    }
    if (options?.resolution && model === GEMINI_OMNI_VIDEO_MODEL && options.resolution !== '720p') {
        throw new OmniVideoTerminalError(`${model} only supports 720p output`);
    }

    switch (task) {
        case 'text_to_video':
            if (media.length !== 0) throw new OmniVideoTerminalError('text_to_video does not accept media inputs');
            break;
        case 'image_to_video': {
            const maximumImages = model === GEMINI_OMNI_1_1_VIDEO_MODEL ? 2 : 1;
            if (videoCount !== 0 || imageCount < 1 || imageCount > maximumImages) {
                const expected = maximumImages === 1 ? 'exactly one image input' : 'one or two image inputs';
                throw new OmniVideoTerminalError(`image_to_video requires ${expected}`);
            }
            break;
        }
        case 'reference_to_video':
            if (media.length === 0 || imageCount > 10 || videoCount > 3) {
                throw new OmniVideoTerminalError(
                    'reference_to_video requires media with at most ten images and three videos',
                );
            }
            break;
        case 'edit':
            if (imageCount !== 0 || videoCount !== 1) {
                throw new OmniVideoTerminalError('edit requires exactly one video input');
            }
            break;
        case 'extend':
            if (imageCount !== 0 || videoCount !== 1) {
                throw new OmniVideoTerminalError('extend requires exactly one video input');
            }
            break;
    }
    return task;
}

function interactionFailureRetryable(errors: Interactions.Error[] | undefined): boolean | undefined {
    const details = JSON.stringify(errors ?? []).toLowerCase();
    if (
        details.includes('invalid_argument') ||
        details.includes('unauthenticated') ||
        details.includes('permission_denied') ||
        details.includes('not_found') ||
        details.includes('failed_precondition') ||
        details.includes('out_of_range') ||
        details.includes('unimplemented')
    ) {
        return false;
    }
    if (
        details.includes('resource_exhausted') ||
        details.includes('rate_limit') ||
        details.includes('throttl') ||
        details.includes('aborted') ||
        details.includes('internal') ||
        details.includes('unavailable') ||
        details.includes('deadline_exceeded') ||
        details.includes('timeout')
    ) {
        return true;
    }
    return undefined;
}

function parseOmniResults(response: Interactions.Interaction, outputPrefix: string) {
    if (response.status !== 'completed') {
        const details = response.errors?.length ? `: ${JSON.stringify(response.errors)}` : '';
        const permanent = new Set(['requires_action', 'cancelled', 'budget_exceeded']);
        const transient = new Set(['queued', 'in_progress', 'incomplete']);
        throw new OmniVideoInteractionError(
            `Gemini Omni video interaction did not complete (status: ${response.status})${details}`,
            permanent.has(response.status)
                ? false
                : transient.has(response.status)
                  ? true
                  : interactionFailureRetryable(response.errors),
        );
    }

    const outputParts = (response.steps ?? [])
        .filter((step) => step.type === 'model_output')
        .flatMap((step) => step.content ?? []);

    const results: CompletionResult[] = [];
    let hasVideoOutput = false;
    for (const part of outputParts) {
        if (part.type === 'text') {
            if (part.text) results.push({ type: 'text', value: part.text });
            continue;
        }
        if (part.type === 'video') {
            hasVideoOutput = true;
            if (part.data !== undefined) {
                throw new OmniVideoTerminalError('Gemini Omni returned inline video data instead of URI delivery');
            }
            if (!part.uri || !isGcsUri(part.uri, true)) {
                throw new OmniVideoTerminalError('Gemini Omni returned a missing or invalid GCS video URI');
            }
            if (!part.uri.startsWith(outputPrefix)) {
                throw new OmniVideoTerminalError(
                    'Gemini Omni returned a video URI outside the requested output prefix',
                );
            }
            if (part.mime_type && !part.mime_type.startsWith('video/')) {
                throw new OmniVideoTerminalError(`Gemini Omni returned an invalid video MIME type: ${part.mime_type}`);
            }
            results.push({ type: 'video', value: part.uri });
        }
    }
    if (!hasVideoOutput) {
        throw new OmniVideoTerminalError('Gemini Omni completed without a video output URI');
    }
    return results;
}

export class GeminiOmniVideoModelDefinition implements ModelDefinition<OmniVideoPrompt> {
    readonly model: AIModel;

    constructor(private readonly modelId: GeminiOmniVideoModel = GEMINI_OMNI_VIDEO_MODEL) {
        this.model = {
            id: modelId,
            name: modelId,
            provider: 'vertexai',
            type: ModelType.Video,
            can_stream: false,
        } satisfies AIModel;
    }

    async createPrompt(
        _driver: VertexAIDriver,
        segments: PromptSegment[],
        options: ExecutionOptions,
    ): Promise<OmniVideoPrompt> {
        if (options.conversation)
            throw new OmniVideoTerminalError('Gemini Omni video does not support conversation resume');
        if (options.tools?.length) throw new OmniVideoTerminalError('Gemini Omni video does not support tools');
        if (options.result_schema)
            throw new OmniVideoTerminalError('Gemini Omni video does not support result schemas');

        const text = segments
            .map((segment) => segment.content.trim())
            .filter(Boolean)
            .join('\n');
        if (!text) throw new OmniVideoTerminalError('Gemini Omni video requires a non-empty text prompt');

        const media: OmniVideoMediaInput[] = [];
        for (const segment of segments) {
            for (const file of segment.files ?? []) {
                const type = SUPPORTED_IMAGE_MIME_TYPES.has(file.mime_type)
                    ? 'image'
                    : SUPPORTED_VIDEO_MIME_TYPES.has(file.mime_type)
                      ? 'video'
                      : undefined;
                if (!type) {
                    throw new OmniVideoTerminalError(
                        `Gemini Omni video does not support input MIME type ${file.mime_type}`,
                    );
                }
                const uri = await file.getURI();
                if (!isGcsUri(uri, true)) {
                    throw new OmniVideoTerminalError('Gemini Omni video media inputs must expose a GCS object URI');
                }
                media.push({ type, uri, mime_type: file.mime_type });
            }
        }

        resolveTask(this.modelId, options.model_options as VertexAIGeminiOmniVideoOptions | undefined, media);
        return { text, media };
    }

    async requestTextCompletion(
        driver: VertexAIDriver,
        prompt: OmniVideoPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        const modelOptions = options.model_options as VertexAIGeminiOmniVideoOptions | undefined;
        const task = resolveTask(this.modelId, modelOptions, prompt.media);
        const outputPrefix = normalizeOutputPrefix(options.output_storage_uri);
        const responseFormat = {
            type: 'video' as const,
            delivery: 'uri' as const,
            gcs_uri: outputPrefix,
            duration: `${modelOptions?.duration_seconds ?? 5}s`,
            ...(modelOptions?.aspect_ratio ? { aspect_ratio: modelOptions.aspect_ratio } : {}),
            ...(modelOptions?.resolution ? { resolution: modelOptions.resolution } : {}),
        } satisfies Interactions.VideoResponseFormat;
        const payload = {
            model: this.modelId,
            input: [{ type: 'text' as const, text: prompt.text }, ...prompt.media],
            response_format: [responseFormat],
            generation_config: { video_config: { task } },
        } satisfies Interactions.CreateModelInteractionParamsNonStreaming;
        const response = await driver
            .getFetchClientForRegion('global', 'v1beta1')
            .post<Interactions.Interaction>('interactions', {
                payload,
                signal,
                timeoutMs: driver.getRequestTimeoutMs(options.httpTimeout),
            });

        const tokenUsage: ExecutionTokenUsage = {
            total: response.usage?.total_tokens,
            prompt: response.usage?.total_input_tokens,
            result: response.usage?.total_output_tokens,
        };
        return {
            result: parseOmniResults(response, outputPrefix),
            token_usage: tokenUsage,
            finish_reason: 'stop',
            ...(options.include_original_response ? { original_response: response } : {}),
        };
    }

    requestTextCompletionStream(): Promise<DriverCompletionStream> {
        return Promise.reject(new OmniVideoTerminalError('Gemini Omni video does not support streaming'));
    }

    formatLlumiverseError(_driver: VertexAIDriver, error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        if (!(error instanceof OmniVideoTerminalError) && !(error instanceof OmniVideoInteractionError)) {
            if (!(error instanceof Error) || error.name !== 'AbortError') {
                // Let VertexAIDriver fall back to the shared HTTP/network classifier. In particular,
                // timeouts and all 5xx responses are retryable while ordinary 4xx responses are not.
                throw error;
            }
        }
        const retryable = error instanceof OmniVideoInteractionError ? error.retryable : false;
        const name = error instanceof Error ? error.name : 'GeminiOmniVideoError';
        return new LlumiverseError(
            error instanceof Error ? error.message : String(error),
            retryable,
            context,
            error,
            undefined,
            name,
        );
    }
}
