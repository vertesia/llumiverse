import {
    type AudioEmbeddingInput,
    Base64DataSource,
    type DataSource,
    type EmbeddingInput,
    type EmbeddingOutput,
    type EmbeddingResultItem,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type EmbeddingTaskType,
    type ImageEmbeddingInput,
    type TextEmbeddingInput,
    type VideoEmbeddingInput,
    buildEmbeddingsResult,
    dataSourceToBase64,
} from "@llumiverse/core";
import type { BedrockDriver } from "./index.js";
import type { TwelvelabsMarengoRequest, TwelvelabsMarengoResponse } from "./twelvelabs.js";

const DEFAULT_MODEL = "amazon.nova-2-multimodal-embeddings-v1:0";

interface BedrockMediaSource {
    base64String?: string;
    s3Location?: {
        uri: string;
        bucketOwner?: string;
    };
}

function isS3Url(url: string): boolean {
    if (url.startsWith("s3://")) return true;
    try {
        const parsed = new URL(url);
        if (!parsed.hostname.endsWith("amazonaws.com")) return false;
        return parsed.hostname.startsWith("s3.") || parsed.hostname.includes(".s3.");
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

// =================== Nova multimodal ===================

interface NovaEmbeddingRequest {
    inputType: "text" | "image" | "video" | "audio";
    inputText?: string;
    inputImage?: { format?: string; source: { bytes: string } };
    inputVideo?: { format?: string; source: { bytes?: string; s3Location?: { uri: string } } };
    inputAudio?: { format?: string; source: { bytes?: string; s3Location?: { uri: string } } };
    embeddingTypes?: ("float" | "binary")[];
    embeddingOutputDimensions?: number;
    segmentationConfig?: { startOffsetSec?: number; endOffsetSec?: number; intervalSec?: number };
}

interface NovaEmbeddingResponse {
    embedding?: number[];
    embeddingsByType?: { float?: number[]; binary?: string };
    inputTextTokenCount?: number;
}

function novaEmbeddingTypes(options: EmbeddingsOptions): ("float" | "binary")[] | undefined {
    if (options.output_dtype === "float") return ["float"];
    if (options.output_dtype === "binary") return ["binary"];
    return undefined;
}

async function novaInputForText(input: TextEmbeddingInput): Promise<NovaEmbeddingRequest> {
    return { inputType: "text", inputText: input.text };
}

async function novaInputForImage(input: ImageEmbeddingInput): Promise<NovaEmbeddingRequest> {
    const base64 = await readDataSourceAsBase64(input.source);
    return {
        inputType: "image",
        inputImage: { format: mimeToImageFormat(input.source.mime_type), source: { bytes: base64 } },
    };
}

async function novaInputForVideo(input: VideoEmbeddingInput): Promise<NovaEmbeddingRequest> {
    const media = await dataSourceToBedrockMediaSource(input.source);
    const segmentationConfig: NovaEmbeddingRequest["segmentationConfig"] = {};
    if (input.start_sec !== undefined) segmentationConfig.startOffsetSec = input.start_sec;
    if (input.start_sec !== undefined && input.length_sec !== undefined) {
        segmentationConfig.endOffsetSec = input.start_sec + input.length_sec;
    }
    if (input.interval_sec !== undefined) segmentationConfig.intervalSec = input.interval_sec;
    return {
        inputType: "video",
        inputVideo: {
            format: mimeToVideoFormat(input.source.mime_type),
            source: media.s3Location ? { s3Location: media.s3Location } : { bytes: media.base64String },
        },
        ...(Object.keys(segmentationConfig).length > 0 ? { segmentationConfig } : {}),
    };
}

async function novaInputForAudio(input: AudioEmbeddingInput): Promise<NovaEmbeddingRequest> {
    const media = await dataSourceToBedrockMediaSource(input.source);
    return {
        inputType: "audio",
        inputAudio: {
            format: mimeToAudioFormat(input.source.mime_type),
            source: media.s3Location ? { s3Location: media.s3Location } : { bytes: media.base64String },
        },
    };
}

function mimeToImageFormat(mime: string): string | undefined {
    if (!mime) return undefined;
    const sub = mime.split("/")[1]?.split("+")[0];
    return sub === "jpg" ? "jpeg" : sub;
}

function mimeToVideoFormat(mime: string): string | undefined {
    if (!mime) return undefined;
    return mime.split("/")[1]?.split(";")[0];
}

function mimeToAudioFormat(mime: string): string | undefined {
    if (!mime) return undefined;
    return mime.split("/")[1]?.split(";")[0];
}

async function generateNovaEmbeddings(
    driver: BedrockDriver,
    options: EmbeddingsOptions,
    modelId: string,
): Promise<EmbeddingsResult> {
    const embeddingTypes = novaEmbeddingTypes(options);
    const items: EmbeddingResultItem[] = [];
    let totalTextTokens: number | undefined;

    for (const input of options.inputs) {
        const baseRequest = await buildNovaInput(input);
        const request: NovaEmbeddingRequest = {
            ...baseRequest,
            ...(embeddingTypes ? { embeddingTypes } : {}),
            ...(options.dimensions ? { embeddingOutputDimensions: options.dimensions } : {}),
        };
        const response = await invokeJson<NovaEmbeddingResponse>(driver, modelId, request);
        const values = response.embedding;
        if (!values) {
            throw new Error(`Nova embeddings response missing 'embedding' for input type '${input.type}'`);
        }
        items.push({
            outputs: [{ values, modality: input.type }],
            input_tokens: response.inputTextTokenCount,
        });
        if (typeof response.inputTextTokenCount === "number") {
            totalTextTokens = (totalTextTokens ?? 0) + response.inputTextTokenCount;
        }
    }

    const usage = totalTextTokens !== undefined
        ? { input_text_tokens: totalTextTokens, input_tokens: totalTextTokens }
        : undefined;
    return buildEmbeddingsResult(modelId, items, usage);
}

async function buildNovaInput(input: EmbeddingInput): Promise<NovaEmbeddingRequest> {
    switch (input.type) {
        case "text": return novaInputForText(input);
        case "image": return novaInputForImage(input);
        case "video": return novaInputForVideo(input);
        case "audio": return novaInputForAudio(input);
    }
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
    embedding_types?: string[];
    truncate?: "NONE" | "LEFT" | "RIGHT";
}

interface CohereEmbeddingResponse {
    embeddings:
        | number[][]
        | { float?: number[][]; int8?: number[][]; uint8?: number[][]; binary?: number[][]; ubinary?: number[][] };
    texts?: string[];
    response_type?: string;
}

function cohereInputType(taskType: EmbeddingTaskType | undefined): string | undefined {
    switch (taskType) {
        case "RETRIEVAL_QUERY": return "search_query";
        case "RETRIEVAL_DOCUMENT": return "search_document";
        case "CLASSIFICATION": return "classification";
        case "CLUSTERING": return "clustering";
        default: return undefined;
    }
}

function cohereEmbeddingTypes(dtype: EmbeddingsOptions["output_dtype"]): string[] | undefined {
    if (!dtype) return undefined;
    return [dtype];
}

function cohereTruncate(truncate: EmbeddingsOptions["truncate"]): "NONE" | "LEFT" | "RIGHT" | undefined {
    switch (truncate) {
        case "NONE": return "NONE";
        case "START": return "LEFT";
        case "END": return "RIGHT";
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
    const embeddingTypes = cohereEmbeddingTypes(options.output_dtype);
    const truncate = cohereTruncate(options.truncate);

    if (textInputs.length > 0) {
        const body: CohereEmbeddingRequest = {
            texts: textInputs.map((t) => t.input.text),
            input_type: inputType,
            embedding_types: embeddingTypes,
            truncate,
        };
        const res = await invokeJson<CohereEmbeddingResponse>(driver, modelId, body);
        const vectors = pickCohereVectors(res, options.output_dtype);
        textInputs.forEach((entry, i) => {
            items[entry.index] = { outputs: [{ values: vectors[i], modality: "text" }] };
        });
    }

    if (imageInputs.length > 0) {
        const base64Images = await Promise.all(imageInputs.map((i) => readDataSourceAsBase64(i.input.source)));
        const body: CohereEmbeddingRequest = {
            images: base64Images,
            input_type: inputType ?? "image",
            embedding_types: embeddingTypes,
        };
        const res = await invokeJson<CohereEmbeddingResponse>(driver, modelId, body);
        const vectors = pickCohereVectors(res, options.output_dtype);
        imageInputs.forEach((entry, i) => {
            items[entry.index] = { outputs: [{ values: vectors[i], modality: "image" }] };
        });
    }

    return buildEmbeddingsResult(modelId, items);
}

function pickCohereVectors(
    response: CohereEmbeddingResponse,
    dtype: EmbeddingsOptions["output_dtype"],
): number[][] {
    if (Array.isArray(response.embeddings)) return response.embeddings;
    const map = response.embeddings;
    if (dtype === "binary" && map.binary) return map.binary;
    if (dtype === "int8" && map.int8) return map.int8;
    if (dtype === "float" && map.float) return map.float;
    return map.float ?? map.int8 ?? map.uint8 ?? map.binary ?? map.ubinary ?? [];
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
    const modelId = options.model ?? DEFAULT_MODEL;
    if (modelId.includes("twelvelabs.marengo")) {
        return generateMarengoEmbeddings(driver, options, modelId);
    }
    if (modelId.startsWith("cohere.embed")) {
        return generateCohereEmbeddings(driver, options, modelId);
    }
    if (modelId.includes("titan-embed")) {
        return generateTitanEmbeddings(driver, options, modelId);
    }
    return generateNovaEmbeddings(driver, options, modelId);
}
