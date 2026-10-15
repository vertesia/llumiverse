import {
    type DataSource,
    type ExecutionOptions,
    type ExecutionTokenUsage,
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
    // Pegasus takes the schema directly under `jsonSchema`, not the OpenAI `json_schema` wrapper. Pegasus 1.2 rejects
    // a `type` field, while Pegasus 1.5 requires `type: 'json_schema'` (verified against both models).
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
    responseFormat?: {
        type?: 'json_schema';
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

export interface TwelvelabsPegasusResponse {
    message: string;
    /** Pegasus 1.2 */
    finishReason?: 'stop' | 'length';
    /** Pegasus 1.5 */
    stopReason?: 'stop' | 'length';
    input_tokens?: number;
    output_tokens?: number;
}

/** Pegasus responses report token counts alongside the message. */
export function pegasusTokenUsage(response: TwelvelabsPegasusResponse): ExecutionTokenUsage | undefined {
    if (response.input_tokens === undefined && response.output_tokens === undefined) return undefined;
    return {
        prompt: response.input_tokens,
        result: response.output_tokens,
        total: (response.input_tokens ?? 0) + (response.output_tokens ?? 0),
    };
}

function isPegasus15OrLater(model: string): boolean {
    const match = model.match(/pegasus-(\d+)-(\d+)/);
    return !!match && (Number(match[1]) > 1 || (Number(match[1]) === 1 && Number(match[2]) >= 5));
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
): Promise<TwelvelabsPegasusRequest> {
    let inputPrompt = '';
    let videoFile: DataSource | undefined;

    // Extract text content and video files from segments
    for (const segment of segments) {
        if (segment.role === PromptRole.system || segment.role === PromptRole.user) {
            if (segment.content) {
                inputPrompt += `${segment.content}\n`;
            }

            // Look for video files
            for (const file of segment.files ?? []) {
                if (file.mime_type?.startsWith('video/')) {
                    videoFile = file;
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

    try {
        // Try to get S3 URL first
        const url = await videoFile.getURL();
        const parsedUrl = new URL(url);

        if (parsedUrl.protocol === 's3:' || isAmazonS3Hostname(parsedUrl.hostname)) {
            // Convert S3 URL to s3:// format
            mediaSource = {
                s3Location: {
                    uri: parsedUrl.protocol === 's3:' ? url : parseS3UrlToUri(parsedUrl),
                },
            };
        } else {
            // Fall back to base64 encoding
            const stream = await videoFile.getStream();
            const base64String = await readStreamAsBase64(stream);

            mediaSource = {
                base64String,
            };
        }
    } catch {
        // If getting URL fails, use base64 encoding
        const stream = await videoFile.getStream();
        const base64String = await readStreamAsBase64(stream);

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
        request.responseFormat = isPegasus15OrLater(options.model)
            ? { type: 'json_schema', jsonSchema: options.result_schema }
            : { jsonSchema: options.result_schema };
    }

    return request;
}
