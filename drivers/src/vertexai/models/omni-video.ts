import type { Interactions } from '@google/genai';
import {
    type AIModel,
    type Completion,
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

const SUPPORTED_IMAGE_MIME_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/heic', 'image/heif']);

interface OmniVideoImageInput {
    type: 'image';
    uri: string;
    mime_type: string;
}

export interface OmniVideoPrompt {
    text: string;
    images: OmniVideoImageInput[];
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

function resolveTask(options: VertexAIGeminiOmniVideoOptions | undefined, imageCount: number) {
    const task = options?.task ?? (imageCount === 0 ? 'text_to_video' : undefined);
    if (!task) {
        throw new OmniVideoTerminalError(
            'Gemini Omni video generation with images requires an explicit image_to_video or reference_to_video task',
        );
    }
    if (task === 'text_to_video' && imageCount !== 0) {
        throw new OmniVideoTerminalError('text_to_video does not accept image inputs');
    }
    if (task === 'image_to_video' && imageCount !== 1) {
        throw new OmniVideoTerminalError('image_to_video requires exactly one image input');
    }
    if (task === 'reference_to_video' && (imageCount < 1 || imageCount > 3)) {
        throw new OmniVideoTerminalError('reference_to_video requires between one and three image inputs');
    }
    return task;
}

function parseVideoResults(response: Interactions.Interaction, outputPrefix: string) {
    if (response.status !== 'completed') {
        const details = response.errors?.length ? `: ${JSON.stringify(response.errors)}` : '';
        const permanent = new Set(['requires_action', 'cancelled', 'budget_exceeded']);
        const transient = new Set(['queued', 'in_progress', 'incomplete']);
        throw new OmniVideoInteractionError(
            `Gemini Omni video interaction did not complete (status: ${response.status})${details}`,
            permanent.has(response.status) ? false : transient.has(response.status) ? true : undefined,
        );
    }

    const videoParts = (response.steps ?? [])
        .filter((step) => step.type === 'model_output')
        .flatMap((step) => step.content ?? [])
        .filter((part) => part.type === 'video');
    if (videoParts.length === 0) {
        throw new OmniVideoTerminalError('Gemini Omni completed without a video output URI');
    }

    return videoParts.map((part) => {
        if (part.data !== undefined) {
            throw new OmniVideoTerminalError('Gemini Omni returned inline video data instead of URI delivery');
        }
        if (!part.uri || !isGcsUri(part.uri, true)) {
            throw new OmniVideoTerminalError('Gemini Omni returned a missing or invalid GCS video URI');
        }
        if (!part.uri.startsWith(outputPrefix)) {
            throw new OmniVideoTerminalError('Gemini Omni returned a video URI outside the requested output prefix');
        }
        if (part.mime_type && !part.mime_type.startsWith('video/')) {
            throw new OmniVideoTerminalError(`Gemini Omni returned an invalid video MIME type: ${part.mime_type}`);
        }
        return { type: 'video' as const, value: part.uri };
    });
}

export class GeminiOmniVideoModelDefinition implements ModelDefinition<OmniVideoPrompt> {
    readonly model: AIModel;

    constructor() {
        this.model = {
            id: GEMINI_OMNI_VIDEO_MODEL,
            name: GEMINI_OMNI_VIDEO_MODEL,
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

        const images: OmniVideoImageInput[] = [];
        for (const segment of segments) {
            for (const file of segment.files ?? []) {
                if (!SUPPORTED_IMAGE_MIME_TYPES.has(file.mime_type)) {
                    throw new OmniVideoTerminalError(
                        `Gemini Omni video does not support input MIME type ${file.mime_type}`,
                    );
                }
                const uri = await file.getURI();
                if (!isGcsUri(uri, true)) {
                    throw new OmniVideoTerminalError('Gemini Omni video image inputs must expose a GCS object URI');
                }
                images.push({ type: 'image', uri, mime_type: file.mime_type });
            }
        }

        resolveTask(options.model_options as VertexAIGeminiOmniVideoOptions | undefined, images.length);
        return { text, images };
    }

    async requestTextCompletion(
        driver: VertexAIDriver,
        prompt: OmniVideoPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        const modelOptions = options.model_options as VertexAIGeminiOmniVideoOptions | undefined;
        const task = resolveTask(modelOptions, prompt.images.length);
        const outputPrefix = normalizeOutputPrefix(options.output_storage_uri);
        const responseFormat = {
            type: 'video' as const,
            delivery: 'uri' as const,
            gcs_uri: outputPrefix,
            duration: `${modelOptions?.duration_seconds ?? 5}s`,
            ...(modelOptions?.aspect_ratio ? { aspect_ratio: modelOptions.aspect_ratio } : {}),
        };
        const payload = {
            model: GEMINI_OMNI_VIDEO_MODEL,
            input: [{ type: 'text' as const, text: prompt.text }, ...prompt.images],
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
            result: parseVideoResults(response, outputPrefix),
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
