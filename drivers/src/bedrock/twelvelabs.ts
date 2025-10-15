import { DataSource, ExecutionOptions, readStreamAsUint8Array } from "@llumiverse/core";
import { PromptSegment, PromptRole } from "@llumiverse/core";

// TwelveLabs Pegasus Request/Response Types
export interface TwelvelabsPegasusRequest {
    inputPrompt: string;
    temperature?: number;
    responseFormat?: {
        type: "json_schema";
        json_schema: {
            name: string;
            schema: any;
        };
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
    finishReason: "stop" | "length";
}

// TwelveLabs Marengo Request/Response Types
export interface TwelvelabsMarengoRequest {
    inputType: "text" | "image" | "video" | "audio";
    inputText?: string;
    textTruncate?: "start" | "end";
    mediaSource?: {
        base64String?: string;
        s3Location?: {
            uri: string;
            bucketOwner?: string;
        };
    };
    embeddingOption?: "visual-text" | "visual-image" | "audio";
    startSec?: number;
    lengthSec?: number;
    useFixedLengthSec?: boolean;
    minClipSec?: number;
}

export interface TwelvelabsMarengoResponse {
    embedding: number[];
    embeddingOption: "visual-text" | "visual-image" | "audio";
    startSec: number;
    endSec: number;
}

// Convert prompt segments to TwelveLabs Pegasus request
export async function formatTwelvelabsPegasusPrompt(
    segments: PromptSegment[],
    options: ExecutionOptions
): Promise<TwelvelabsPegasusRequest> {
    let inputPrompt = "";
    let videoFile: DataSource | undefined;

    // Extract text content and video files from segments
    for (const segment of segments) {
        if (segment.role === PromptRole.system || segment.role === PromptRole.user) {
            if (segment.content) {
                inputPrompt += segment.content + "\n";
            }

            // Look for video files
            for (const file of segment.files ?? []) {
                if (file.mime_type && file.mime_type.startsWith("video/")) {
                    videoFile = file;
                    break; // Use the first video file found
                }
            }
        }
    }

    if (!videoFile) {
        throw new Error("TwelveLabs Pegasus requires a video file input");
    }

    // Prepare media source
    let mediaSource: TwelvelabsPegasusRequest["mediaSource"];

    try {
        // Try to get S3 URL first
        const url = await videoFile.getURL();
        const parsedUrl = new URL(url);

        if (parsedUrl.hostname.endsWith("amazonaws.com") &&
            (parsedUrl.hostname.startsWith("s3.") || parsedUrl.hostname.includes(".s3."))) {
            // Convert S3 URL to s3:// format
            const bucketMatch = parsedUrl.hostname.match(/^(?:s3\.)?([^.]+)\.s3\./);
            const bucket = bucketMatch ? bucketMatch[1] : parsedUrl.hostname.split('.')[0];
            const key = parsedUrl.pathname.substring(1); // Remove leading slash

            mediaSource = {
                s3Location: {
                    uri: `s3://${bucket}/${key}`,
                }
            };
        } else {
            // Fall back to base64 encoding
            const stream = await videoFile.getStream();
            const buffer = await readStreamAsUint8Array(stream);
            const base64String = Buffer.from(buffer).toString('base64');

            mediaSource = {
                base64String
            };
        }
    } catch (error) {
        // If getting URL fails, use base64 encoding
        const stream = await videoFile.getStream();
        const buffer = await readStreamAsUint8Array(stream);
        const base64String = Buffer.from(buffer).toString('base64');

        mediaSource = {
            base64String
        };
    }

    const request: TwelvelabsPegasusRequest = {
        inputPrompt: inputPrompt.trim(),
        mediaSource
    };

    // Add optional parameters from model options
    const modelOptions = options.model_options as any;
    if (modelOptions?.temperature !== undefined) {
        request.temperature = modelOptions.temperature;
    }
    if (modelOptions?.max_tokens !== undefined) {
        request.maxOutputTokens = modelOptions.max_tokens;
    }

    // Add response format if result schema is specified
    if (options.result_schema) {
        request.responseFormat = {
            type: "json_schema",
            json_schema: {
                name: "response",
                schema: options.result_schema
            }
        };
    }

    return request;
}
