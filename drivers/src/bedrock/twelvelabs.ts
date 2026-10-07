import { isConversationDocumentFormat, parseConversationDocument } from '@llumiverse/conversation';
import {
    type DataSource,
    type ExecutionOptions,
    type JSONSchema,
    PromptRole,
    type PromptSegment,
    readStreamAsBase64,
    type TextFallbackOptions,
} from '@llumiverse/core';
import { isAmazonS3Hostname, parseS3UrlToUri } from './s3.js';

// TwelveLabs Pegasus Request/Response Types
export interface TwelvelabsPegasusRequest {
    inputPrompt: string;
    temperature?: number;
    responseFormat?: {
        jsonSchema: JSONSchema;
    };
    mediaSource: {
        base64String?: string;
        s3Location?: {
            uri: string;
            bucketOwner?: string;
        };
    };
    maxOutputTokens?: number;
}

export interface TwelvelabsPegasusPromptSource {
    segments: Array<{ index: number; role: PromptRole.system | PromptRole.user; content: string }>;
    video: { segment_index: number; name: string; mime_type: string };
}

export const TWELVELABS_PEGASUS_PROMPT_SOURCE = Symbol('twelvelabs.pegasus.prompt_source');

export type TwelvelabsPegasusCanonicalPrompt = TwelvelabsPegasusRequest & {
    [TWELVELABS_PEGASUS_PROMPT_SOURCE]?: TwelvelabsPegasusPromptSource;
};

export interface TwelvelabsPegasusResponse {
    message: string;
    finishReason: 'stop' | 'length';
}

export interface TwelvelabsPegasusFormatOptions {
    max_video_bytes?: number;
}

async function readVideoAsBase64(
    stream: ReadableStream<string | Uint8Array>,
    maxBytes: number | undefined,
): Promise<string> {
    if (maxBytes === undefined) return readStreamAsBase64(stream);
    const reader = stream.getReader();
    const chunks: Uint8Array[] = [];
    let byteLength = 0;
    try {
        while (true) {
            const next = await reader.read();
            if (next.done) break;
            if (!(next.value instanceof Uint8Array)) {
                await reader.cancel('TwelveLabs Pegasus video stream must contain binary chunks');
                throw new TypeError('TwelveLabs Pegasus video stream must contain binary chunks');
            }
            byteLength += next.value.byteLength;
            if (byteLength > maxBytes) {
                await reader.cancel('TwelveLabs Pegasus inline video exceeds the 25MB limit');
                throw new Error('TwelveLabs Pegasus inline video exceeds the 25MB limit');
            }
            chunks.push(next.value);
        }
    } finally {
        reader.releaseLock();
    }
    const bytes = new Uint8Array(byteLength);
    let offset = 0;
    for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
    }
    return Buffer.from(bytes).toString('base64');
}

export function validateTwelvelabsPegasusCanonicalInput(segments: PromptSegment[], options: ExecutionOptions): void {
    if (options.conversation !== undefined && !isConversationDocumentFormat(options.conversation)) {
        throw new TypeError('TwelveLabs Pegasus canonical execution does not support legacy conversation input');
    }
    if (isConversationDocumentFormat(options.conversation)) {
        const document = parseConversationDocument(options.conversation);
        if (document.context.active_tool_definition_ids.length > 0) {
            throw new TypeError('TwelveLabs Pegasus does not support active canonical tool definitions');
        }
    }
    if (options.tools?.length) throw new TypeError('TwelveLabs Pegasus does not support tools');
    if (options.format !== undefined) throw new TypeError('TwelveLabs Pegasus does not support custom formatting');
    if (options.conversation_runtime?.materialized_input !== undefined) {
        throw new TypeError('TwelveLabs Pegasus does not support materialized canonical input');
    }
    if (options.output_modality !== undefined && options.output_modality !== 'text') {
        throw new TypeError(`TwelveLabs Pegasus does not support ${options.output_modality} output`);
    }
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
    let videoCount = 0;
    let hasText = false;
    for (const segment of segments) {
        if (segment.role !== PromptRole.system && segment.role !== PromptRole.user) {
            throw new TypeError(`TwelveLabs Pegasus canonical execution does not support ${segment.role} input`);
        }
        if (segment.content.trim().length > 0) hasText = true;
        for (const file of segment.files ?? []) {
            if (!file.mime_type.startsWith('video/')) {
                throw new TypeError(`TwelveLabs Pegasus does not support ${file.mime_type || 'untyped'} input files`);
            }
            videoCount += 1;
        }
    }
    if (!hasText) throw new TypeError('TwelveLabs Pegasus requires a text prompt');
    if (videoCount !== 1) throw new TypeError('TwelveLabs Pegasus canonical execution requires exactly one video');
}

// TwelveLabs Marengo Request/Response Types
export interface TwelvelabsMarengoRequest {
    inputType: 'text' | 'image' | 'video' | 'audio';
    inputText?: string;
    textTruncate?: 'start' | 'end';
    mediaSource?: {
        base64String?: string;
        s3Location?: {
            uri: string;
            bucketOwner?: string;
        };
    };
    embeddingOption?: 'visual-text' | 'visual-image' | 'audio';
    startSec?: number;
    lengthSec?: number;
    useFixedLengthSec?: boolean;
    minClipSec?: number;
}

export interface TwelvelabsMarengoResponse {
    embedding: number[];
    embeddingOption: 'visual-text' | 'visual-image' | 'audio';
    startSec: number;
    endSec: number;
}

// Convert prompt segments to TwelveLabs Pegasus request
export async function formatTwelvelabsPegasusPrompt(
    segments: PromptSegment[],
    options: ExecutionOptions,
    formatOptions: TwelvelabsPegasusFormatOptions = {},
): Promise<TwelvelabsPegasusCanonicalPrompt> {
    let inputPrompt = '';
    let videoFile: DataSource | undefined;
    let videoSegmentIndex = -1;
    const sourceSegments: TwelvelabsPegasusPromptSource['segments'] = [];

    // Extract text content and video files from segments
    for (const [segmentIndex, segment] of segments.entries()) {
        if (segment.role === PromptRole.system || segment.role === PromptRole.user) {
            sourceSegments.push({ index: segmentIndex, role: segment.role, content: segment.content });
            if (segment.content) {
                inputPrompt += `${segment.content}\n`;
            }

            // Look for video files
            for (const file of segment.files ?? []) {
                if (file.mime_type?.startsWith('video/')) {
                    videoFile = file;
                    videoSegmentIndex = segmentIndex;
                    break; // Use the first video file found
                }
            }
        }
    }

    if (!videoFile) {
        throw new Error('TwelveLabs Pegasus requires a video file input');
    }

    // Prepare media source
    let mediaSource: TwelvelabsPegasusRequest['mediaSource'];

    let sourceUrl: { raw: string; parsed: URL } | undefined;
    try {
        const raw = await videoFile.getURL();
        sourceUrl = { raw, parsed: new URL(raw) };
    } catch {
        // A data source need not expose a provider-readable URL. Its stream remains the supported fallback.
    }
    if (
        sourceUrl !== undefined &&
        (sourceUrl.parsed.protocol === 's3:' || isAmazonS3Hostname(sourceUrl.parsed.hostname))
    ) {
        mediaSource = {
            s3Location: {
                uri: sourceUrl.parsed.protocol === 's3:' ? sourceUrl.raw : parseS3UrlToUri(sourceUrl.parsed),
            },
        };
    } else {
        const stream = await videoFile.getStream();
        const base64String = await readVideoAsBase64(stream, formatOptions.max_video_bytes);

        mediaSource = {
            base64String,
        };
    }

    const request: TwelvelabsPegasusRequest = {
        inputPrompt: inputPrompt.trim(),
        mediaSource,
    };

    // Add optional parameters from model options
    const modelOptions = options.model_options as TextFallbackOptions | undefined;
    if (modelOptions?.temperature !== undefined) {
        request.temperature = modelOptions.temperature;
    }
    if (modelOptions?.max_tokens !== undefined) {
        request.maxOutputTokens = modelOptions.max_tokens;
    }

    // Add response format if result schema is specified
    if (options.result_schema) {
        request.responseFormat = {
            jsonSchema: options.result_schema,
        };
    }
    Object.defineProperty(request, TWELVELABS_PEGASUS_PROMPT_SOURCE, {
        configurable: false,
        enumerable: false,
        value: {
            segments: sourceSegments,
            video: { segment_index: videoSegmentIndex, name: videoFile.name, mime_type: videoFile.mime_type },
        } satisfies TwelvelabsPegasusPromptSource,
        writable: false,
    });
    return request;
}
