import { BEDROCK_DEFAULT_EMBEDDING_MODEL } from "@llumiverse/common";
import {
    Base64DataSource,
    buildEmbeddingsResult,
    type DataSource,
    dataSourceToBase64,
    type EmbeddingInput,
    type EmbeddingOutput,
    type EmbeddingResultItem,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type EmbeddingTaskType,
    type ImageEmbeddingInput,
    LlumiverseError,
    type TextEmbeddingInput,
} from "@llumiverse/core";
import type { BedrockDriver } from "./index.js";
import type { TwelvelabsMarengoRequest, TwelvelabsMarengoResponse } from "./twelvelabs.js";

interface BedrockMediaSource {
    base64String?: string;
    s3Location?: {
        uri: string;
        bucketOwner?: string;
    };
}

/** Matches s3[.region].amazonaws.com or bucket.s3[.region].amazonaws.com — anchored to prevent subdomain spoofing. */
const S3_HOSTNAME_RE = /^(?:[^.]+\.)?s3(?:\.[a-z0-9-]+)?\.amazonaws\.com$/;

function isS3Url(url: string): boolean {
    if (url.startsWith("s3://")) return true;
    try {
        const parsed = new URL(url);
        return S3_HOSTNAME_RE.test(parsed.hostname);
    } catch {
        return false;
    }
}

function toS3Uri(url: string): string {
    if (url.startsWith("s3://")) return url;
    const parsed = new URL(url);
    const bucketMatch = parsed.hostname.match(/^(?:s3\.)?([^.]+)\.s3\./);
    const bucket = bucketMatch ? bucketMatch[1] : parsed.hostname.split(".")[0];
    const key = parsed.pathname.replace(/^\/+/, "");
    return `s3://${bucket}/${key}`;
}

async function dataSourceToBedrockMediaSource(ds: DataSource): Promise<BedrockMediaSource> {
    const url = await ds.getURL().catch(() => undefined);
    if (url && isS3Url(url)) {
        return { s3Location: { uri: toS3Uri(url) } };
    }
    if (ds instanceof Base64DataSource) {
        return { base64String: ds.getBase64() };
    }
    return { base64String: await dataSourceToBase64(ds) };
}

async function readDataSourceAsBase64(ds: DataSource): Promise<string> {
    if (ds instanceof Base64DataSource) return ds.getBase64();
    return dataSourceToBase64(ds);
}

async function invokeJson<R>(
    driver: BedrockDriver,
    modelId: string,
    body: unknown,
): Promise<R> {
    const executor = driver.getExecutor();
    const res = await executor.invokeModel({
        modelId,
        contentType: "application/json",
        accept: "application/json",
        body: JSON.stringify(body),
    });
    const decoder = new TextDecoder();
    return JSON.parse(decoder.decode(res.body)) as R;
}

// ── Shared format lookup maps ───────────────────────────────────────────────

const MIME_TO_IMAGE_FORMAT: Record<string, string> = {
    "image/jpeg": "jpeg", "image/jpg": "jpeg",
    "image/png": "png", "image/gif": "gif", "image/webp": "webp",
};

const MIME_TO_AUDIO_FORMAT: Record<string, string> = {
    "audio/mpeg": "mp3", "audio/mp3": "mp3",
    "audio/wav": "wav", "audio/wave": "wav", "audio/x-wav": "wav",
    "audio/ogg": "ogg",
};

const MIME_TO_VIDEO_FORMAT: Record<string, string> = {
    "video/mp4": "mp4", "video/quicktime": "mov",
    "video/x-matroska": "mkv", "video/webm": "webm",
    "video/x-flv": "flv", "video/mpeg": "mpeg", "video/mpg": "mpg",
    "video/x-ms-wmv": "wmv", "video/3gpp": "3gp",
};

function requireFormat(map: Record<string, string>, mime: string, label: string): string {
    const fmt = map[mime.toLowerCase()];
    if (!fmt) throw new Error(`Unsupported ${label} MIME type for Bedrock: '${mime}'`);
    return fmt;
}

// =================== Nova multimodal ===================

/**
 * Synchronous InvokeModel request body for amazon.nova-2-multimodal-embeddings-v1:0.
 * Schema: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
 *
 * Sync limits (InvokeModel): text 8K tokens, image 50 MB S3 / 25 MB inline,
 * audio/video 30 s / 100 MB S3 / 25 MB inline base64.
 *
 * For inputs exceeding these limits, the Bedrock Runtime exposes StartAsyncInvoke:
 * the caller submits a job and receives an invocationArn, then polls GetAsyncInvoke
 * until the job completes and reads segmented embeddings from the S3 output bucket.
 * This is incompatible with a simple await-style interface and is not implemented here.
 */

type NovaEmbeddingPurpose =
    | "GENERIC_INDEX" | "GENERIC_RETRIEVAL"
    | "TEXT_RETRIEVAL" | "IMAGE_RETRIEVAL" | "VIDEO_RETRIEVAL"
    | "DOCUMENT_RETRIEVAL" | "AUDIO_RETRIEVAL"
    | "CLASSIFICATION" | "CLUSTERING";

type NovaSource = { bytes: string } | { s3Location: { uri: string } };

interface NovaSingleEmbeddingParams {
    embeddingPurpose: NovaEmbeddingPurpose;
    embeddingDimension?: number;
    text?: { truncationMode: "START" | "END" | "NONE"; value: string };
    image?: { format: string; source: NovaSource; detailLevel?: "STANDARD_IMAGE" | "DOCUMENT_IMAGE" };
    audio?: { format: string; source: NovaSource };
    video?: { format: string; source: NovaSource; embeddingMode: "AUDIO_VIDEO_COMBINED" | "AUDIO_VIDEO_SEPARATE" };
}

interface NovaEmbeddingRequest {
    taskType: "SINGLE_EMBEDDING";
    singleEmbeddingParams: NovaSingleEmbeddingParams;
}

interface NovaEmbeddingResponse {
    embeddings: Array<{
        embeddingType: "TEXT" | "IMAGE" | "VIDEO" | "AUDIO" | "AUDIO_VIDEO_COMBINED";
        embedding: number[];
    }>;
}

function toNovaEmbeddingPurpose(taskType: EmbeddingTaskType | undefined): NovaEmbeddingPurpose {
    return taskType === "query" ? "GENERIC_RETRIEVAL" : "GENERIC_INDEX";
}

async function toNovaSource(ds: DataSource): Promise<NovaSource> {
    const url = await ds.getURL().catch(() => undefined);
    if (url && isS3Url(url)) return { s3Location: { uri: toS3Uri(url) } };
    const bytes = ds instanceof Base64DataSource ? ds.getBase64() : await dataSourceToBase64(ds);
    return { bytes };
}

async function buildNovaParams(input: EmbeddingInput): Promise<Omit<NovaSingleEmbeddingParams, "embeddingPurpose" | "embeddingDimension">> {
    switch (input.type) {
        case "text":
            return { text: { truncationMode: "END", value: input.text } };
        case "image": {
            const source = await toNovaSource(input.source);
            return { image: { format: requireFormat(MIME_TO_IMAGE_FORMAT, input.source.mime_type, "image"), source } };
        }
        case "audio": {
            const source = await toNovaSource(input.source);
            return { audio: { format: requireFormat(MIME_TO_AUDIO_FORMAT, input.source.mime_type, "audio"), source } };
        }
        case "video": {
            // Sync path supports video up to 30 s / 100 MB via S3 (or 25 MB inline base64).
            // Longer video requires StartAsyncInvoke with segmentationConfig — not implemented.
            const source = await toNovaSource(input.source);
            return {
                video: {
                    format: requireFormat(MIME_TO_VIDEO_FORMAT, input.source.mime_type, "video"),
                    source,
                    embeddingMode: "AUDIO_VIDEO_COMBINED",
                },
            };
        }
    }
}

async function generateNovaEmbeddings(
    driver: BedrockDriver,
    options: EmbeddingsOptions,
    modelId: string,
): Promise<EmbeddingsResult> {
    const items: EmbeddingResultItem[] = [];
    const purpose = toNovaEmbeddingPurpose(options.task_type);

    for (const input of options.inputs) {
        const modalParams = await buildNovaParams(input);
        const request: NovaEmbeddingRequest = {
            taskType: "SINGLE_EMBEDDING",
            singleEmbeddingParams: {
                embeddingPurpose: purpose,
                ...(options.dimensions ? { embeddingDimension: options.dimensions } : {}),
                ...modalParams,
            },
        };
        const response = await invokeJson<NovaEmbeddingResponse>(driver, modelId, request);
        const item = response.embeddings?.[0];
        if (!item?.embedding) {
            throw new Error(`Nova embeddings response missing 'embeddings[0].embedding' for input type '${input.type}'`);
        }
        items.push({ outputs: [{ values: item.embedding, modality: input.type }] });
    }

    return buildEmbeddingsResult(modelId, items);
}

// =================== Titan ===================

interface TitanTextRequest { inputText: string; dimensions?: number; normalize?: boolean }
interface TitanTextResponse { embedding: number[]; inputTextTokenCount?: number }
interface TitanImageRequest { inputText?: string; inputImage?: string; embeddingConfig?: { outputEmbeddingLength?: number } }
interface TitanImageResponse { embedding?: number[]; inputTextTokenCount?: number; message?: string }

async function generateTitanEmbeddings(
    driver: BedrockDriver,
    options: EmbeddingsOptions,
    modelId: string,
): Promise<EmbeddingsResult> {
    const isImageModel = modelId.includes("titan-embed-image");
    const items: EmbeddingResultItem[] = [];
    let totalTokens: number | undefined;

    for (const input of options.inputs) {
        if (input.type === "video" || input.type === "audio") {
            throw new Error(`Titan embeddings do not support '${input.type}' input`);
        }

        if (isImageModel) {
            const body: TitanImageRequest = {};
            if (input.type === "text") body.inputText = input.text;
            if (input.type === "image") body.inputImage = await readDataSourceAsBase64(input.source);
            if (options.dimensions) body.embeddingConfig = { outputEmbeddingLength: options.dimensions };
            const res = await invokeJson<TitanImageResponse>(driver, modelId, body);
            if (res.message) throw new Error(`Titan image embedding error: ${res.message}`);
            if (!res.embedding) throw new Error(`Titan image embedding response missing 'embedding'`);
            items.push({ outputs: [{ values: res.embedding, modality: input.type }], input_tokens: res.inputTextTokenCount });
            if (typeof res.inputTextTokenCount === "number") totalTokens = (totalTokens ?? 0) + res.inputTextTokenCount;
        } else {
            if (input.type !== "text") {
                throw new Error(`Titan text embeddings model '${modelId}' only supports text input`);
            }
            const body: TitanTextRequest = { inputText: input.text };
            if (options.dimensions) body.dimensions = options.dimensions;
            const res = await invokeJson<TitanTextResponse>(driver, modelId, body);
            items.push({ outputs: [{ values: res.embedding, modality: "text" }], input_tokens: res.inputTextTokenCount });
            if (typeof res.inputTextTokenCount === "number") totalTokens = (totalTokens ?? 0) + res.inputTextTokenCount;
        }
    }

    const usage = totalTokens !== undefined ? { input_tokens: totalTokens } : undefined;
    return buildEmbeddingsResult(modelId, items, usage);
}

// =================== Cohere ===================

interface CohereEmbeddingRequest {
    texts?: string[];
    images?: string[];
    input_type?: string;
}

interface CohereEmbeddingResponse {
    embeddings: number[][];
    texts?: string[];
    response_type?: string;
}

function cohereInputType(taskType: EmbeddingTaskType | undefined): string | undefined {
    switch (taskType) {
        case "query": return "search_query";
        case "document": return "search_document";
        default: return undefined;
    }
}

async function generateCohereEmbeddings(
    driver: BedrockDriver,
    options: EmbeddingsOptions,
    modelId: string,
): Promise<EmbeddingsResult> {
    const textInputs: { index: number; input: TextEmbeddingInput }[] = [];
    const imageInputs: { index: number; input: ImageEmbeddingInput }[] = [];
    options.inputs.forEach((input, index) => {
        if (input.type === "text") textInputs.push({ index, input });
        else if (input.type === "image") imageInputs.push({ index, input });
        else throw new Error(`Cohere embeddings do not support '${input.type}' input`);
    });

    const items = new Array<EmbeddingResultItem>(options.inputs.length);
    const inputType = cohereInputType(options.task_type);

    if (textInputs.length > 0) {
        const body: CohereEmbeddingRequest = {
            texts: textInputs.map((t) => t.input.text),
            input_type: inputType,
        };
        const res = await invokeJson<CohereEmbeddingResponse>(driver, modelId, body);
        textInputs.forEach((entry, i) => {
            items[entry.index] = { outputs: [{ values: res.embeddings[i], modality: "text" }] };
        });
    }

    // Cohere accepts exactly one image per call; images must be data URIs.
    for (const entry of imageInputs) {
        const base64 = await readDataSourceAsBase64(entry.input.source);
        const dataUri = `data:${entry.input.source.mime_type};base64,${base64}`;
        const body: CohereEmbeddingRequest = { images: [dataUri], input_type: "image" };
        const res = await invokeJson<CohereEmbeddingResponse>(driver, modelId, body);
        items[entry.index] = { outputs: [{ values: res.embeddings[0], modality: "image" }] };
    }

    return buildEmbeddingsResult(modelId, items);
}

// =================== TwelveLabs Marengo ===================

async function generateMarengoEmbeddings(
    driver: BedrockDriver,
    options: EmbeddingsOptions,
    modelId: string,
): Promise<EmbeddingsResult> {
    const items: EmbeddingResultItem[] = [];

    for (const input of options.inputs) {
        const request = await buildMarengoRequest(input);
        const response = await invokeJson<TwelvelabsMarengoResponse | { embeddings?: TwelvelabsMarengoResponse[] }>(
            driver,
            modelId,
            request,
        );

        const segments = Array.isArray((response as { embeddings?: TwelvelabsMarengoResponse[] }).embeddings)
            ? (response as { embeddings: TwelvelabsMarengoResponse[] }).embeddings
            : [response as TwelvelabsMarengoResponse];

        const outputs: EmbeddingOutput[] = segments
            .filter((seg) => Array.isArray(seg.embedding))
            .map((seg) => ({
                values: seg.embedding,
                modality: input.type === "text" ? "text" : input.type,
                start_sec: seg.startSec,
                end_sec: seg.endSec,
                embedding_option: seg.embeddingOption,
            } satisfies EmbeddingOutput));

        if (outputs.length === 0) {
            throw new Error(`Marengo response did not contain embedding values for input type '${input.type}'`);
        }
        items.push({ outputs });
    }

    return buildEmbeddingsResult(modelId, items);
}

async function buildMarengoRequest(input: EmbeddingInput): Promise<TwelvelabsMarengoRequest> {
    switch (input.type) {
        case "text":
            return { inputType: "text", inputText: input.text };
        case "image": {
            const media = await dataSourceToBedrockMediaSource(input.source);
            return { inputType: "image", mediaSource: media };
        }
        case "video": {
            const media = await dataSourceToBedrockMediaSource(input.source);
            const request: TwelvelabsMarengoRequest = { inputType: "video", mediaSource: media };
            if (input.start_sec !== undefined) request.startSec = input.start_sec;
            if (input.length_sec !== undefined) request.lengthSec = input.length_sec;
            if (input.use_fixed_length_sec !== undefined) request.useFixedLengthSec = input.use_fixed_length_sec;
            if (input.min_clip_sec !== undefined) request.minClipSec = input.min_clip_sec;
            if (input.embedding_option && input.embedding_option.length === 1) {
                request.embeddingOption = input.embedding_option[0];
            }
            return request;
        }
        case "audio": {
            const media = await dataSourceToBedrockMediaSource(input.source);
            const request: TwelvelabsMarengoRequest = { inputType: "audio", mediaSource: media };
            if (input.start_sec !== undefined) request.startSec = input.start_sec;
            if (input.length_sec !== undefined) request.lengthSec = input.length_sec;
            return request;
        }
    }
}

// =================== Routing ===================

export async function generateBedrockEmbeddings(
    driver: BedrockDriver,
    options: EmbeddingsOptions,
): Promise<EmbeddingsResult> {
    const modelId = options.model ?? BEDROCK_DEFAULT_EMBEDDING_MODEL;
    try {
        if (modelId.includes("twelvelabs.marengo")) {
            return await generateMarengoEmbeddings(driver, options, modelId);
        }
        if (modelId.startsWith("cohere.embed")) {
            return await generateCohereEmbeddings(driver, options, modelId);
        }
        if (modelId.includes("titan-embed")) {
            return await generateTitanEmbeddings(driver, options, modelId);
        }
        return await generateNovaEmbeddings(driver, options, modelId);
    } catch (error) {
        if (LlumiverseError.isLlumiverseError(error)) throw error;
        throw driver.formatLlumiverseError(error, {
            provider: 'bedrock',
            model: modelId,
            operation: 'execute',
        });
    }
}
