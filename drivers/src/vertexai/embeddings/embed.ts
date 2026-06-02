import { VERTEX_DEFAULT_EMBEDDING_MODEL, VERTEX_MULTIMODAL_EMBEDDING_MODEL } from '@llumiverse/common';
import {
    buildEmbeddingsResult,
    type DataSource,
    type EmbeddingInput,
    type EmbeddingResultItem,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type EmbeddingsTokenUsage,
    type EmbeddingTaskType,
    LlumiverseError,
    normalizeEmbeddingsOptions,
    type TextEmbeddingInput,
} from '@llumiverse/core';
import type { VertexAIDriver } from '../index.js';
import type { VertexContent, VertexPart } from '../types.js';
import { generateLegacyMultimodalEmbeddings } from './embed-legacy-multimodal.js';
import { dataSourceToVertexSourceData } from './source-utils.js';

const TASK_TYPE_PREFIX_MODELS = new Set<string>(['gemini-embedding-2']);
const GLOBAL_ONLY_MODELS = new Set<string>(['gemini-embedding-2']);
const LEGACY_MULTIMODAL_MODELS = new Set<string>([VERTEX_MULTIMODAL_EMBEDDING_MODEL]);

function shortModelName(model: string): string {
    return model.split('/').pop() ?? model;
}

function usesTextPredict(model: string): boolean {
    const shortName = shortModelName(model).toLowerCase();
    return (
        shortName === 'gemini-embedding-001' ||
        shortName === 'embedding-001' ||
        shortName.startsWith('text-embedding-') ||
        shortName.startsWith('text-multilingual-embedding-')
    );
}

function buildPrefixText(input: TextEmbeddingInput): string {
    if (!input.task_type) return input.text;
    if (input.task_type === 'query') {
        return `task: search result | query: ${input.text}`;
    }
    const title = input.title ?? 'none';
    return `title: ${title} | text: ${input.text}`;
}

type GoogleEmbedTaskType = 'RETRIEVAL_QUERY' | 'RETRIEVAL_DOCUMENT';

function toGoogleTaskType(taskType: EmbeddingTaskType | undefined): GoogleEmbedTaskType | undefined {
    switch (taskType) {
        case 'query':
            return 'RETRIEVAL_QUERY';
        case 'document':
            return 'RETRIEVAL_DOCUMENT';
        default:
            return undefined;
    }
}

async function dataSourceToPart(ds: DataSource): Promise<VertexPart> {
    const source = await dataSourceToVertexSourceData(ds);
    if (source.gcsUri) {
        return { fileData: { fileUri: source.gcsUri, mimeType: ds.mime_type } };
    }

    if (!source.bytesBase64Encoded) {
        throw new Error('Data source conversion produced neither GCS URI nor inline bytes');
    }

    return { inlineData: { data: source.bytesBase64Encoded, mimeType: ds.mime_type } };
}

async function inputToContent(input: EmbeddingInput, viaPrefix: boolean): Promise<VertexContent> {
    if (input.type === 'text') {
        const text = viaPrefix ? buildPrefixText(input) : input.text;
        return { role: 'user', parts: [{ text }] };
    }
    return { role: 'user', parts: [await dataSourceToPart(input.source)] };
}

function addTextTokenUsage(usage: EmbeddingsTokenUsage, tokenCount: number): void {
    usage.input_text_tokens = (usage.input_text_tokens ?? 0) + tokenCount;
    usage.input_tokens = (usage.input_tokens ?? 0) + tokenCount;
}

function addEmbeddingUsage(
    usage: EmbeddingsTokenUsage,
    input: EmbeddingInput,
    metadata: EmbedContentResponse['usageMetadata'],
): void {
    const tokenCount = metadata?.promptTokenCount;
    if (!tokenCount) {
        return;
    }
    usage.input_tokens = (usage.input_tokens ?? 0) + tokenCount;

    const details = metadata?.promptTokensDetails ?? metadata?.promptTokensDetail ?? [];
    if (details.length === 0) {
        if (input.type === 'image') {
            usage.input_image_tokens = (usage.input_image_tokens ?? 0) + tokenCount;
        } else {
            usage.input_text_tokens = (usage.input_text_tokens ?? 0) + tokenCount;
        }
        return;
    }

    for (const detail of details) {
        if (detail.modality?.toUpperCase() === 'IMAGE') {
            usage.input_image_tokens = (usage.input_image_tokens ?? 0) + detail.tokenCount;
        } else {
            usage.input_text_tokens = (usage.input_text_tokens ?? 0) + detail.tokenCount;
        }
    }
}

interface TextPredictResponse {
    predictions?: Array<{
        embeddings?: {
            values?: number[];
            statistics?: {
                token_count?: number;
                tokenCount?: number;
            };
        };
    }>;
}

interface EmbedContentResponse {
    embedding?: {
        values?: number[];
    };
    usageMetadata?: {
        promptTokenCount?: number;
        promptTokensDetails?: Array<{ modality?: string; tokenCount: number }>;
        promptTokensDetail?: Array<{ modality?: string; tokenCount: number }>;
    };
}

async function generateTextPredictEmbeddings(
    driver: VertexAIDriver,
    options: EmbeddingsOptions,
    model: string,
): Promise<EmbeddingsResult> {
    for (const input of options.inputs) {
        if (input.type !== 'text') {
            throw new Error(`Vertex AI text embedding model '${model}' only supports text inputs`);
        }
    }

    const entries = options.inputs.map((input, index) => ({
        index,
        input: input as TextEmbeddingInput,
    }));
    const batches =
        shortModelName(model).toLowerCase() === 'gemini-embedding-001' ? entries.map((entry) => [entry]) : [entries];

    const items = new Array<EmbeddingResultItem>(options.inputs.length);
    const usage: EmbeddingsTokenUsage = {};

    for (const batch of batches) {
        const payload = {
            instances: batch.map(({ input }) => ({
                content: input.text,
                task_type: toGoogleTaskType(input.task_type),
                title: input.title,
            })),
            parameters: options.dimensions !== undefined ? { outputDimensionality: options.dimensions } : undefined,
        };

        try {
            const response = await driver.postVertexModel<TextPredictResponse>(model, 'predict', payload);
            const predictions = response.predictions ?? [];
            if (predictions.length !== batch.length) {
                throw new Error(
                    `Vertex AI predict returned ${predictions.length} embeddings for ${batch.length} inputs (model ${model})`,
                );
            }

            predictions.forEach((prediction, i) => {
                const entry = batch[i];
                const values = prediction.embeddings?.values;
                if (!values) {
                    throw new Error(
                        `Vertex AI predict returned an empty embedding for input ${entry.index} (model ${model})`,
                    );
                }
                const tokenCount =
                    prediction.embeddings?.statistics?.token_count ?? prediction.embeddings?.statistics?.tokenCount;
                items[entry.index] = {
                    outputs: [{ values, modality: 'text' }],
                    input_tokens: tokenCount,
                };
                if (typeof tokenCount === 'number') {
                    addTextTokenUsage(usage, tokenCount);
                }
            });
        } catch (error) {
            if (LlumiverseError.isLlumiverseError(error)) throw error;
            if (error instanceof Error && typeof (error as { status?: unknown }).status !== 'number') throw error;
            throw driver.formatLlumiverseError(error, {
                provider: 'vertexai',
                model,
                operation: 'execute',
            });
        }
    }

    return buildEmbeddingsResult(model, items, Object.keys(usage).length > 0 ? usage : undefined);
}

async function generateEmbedContentEmbeddings(
    driver: VertexAIDriver,
    options: EmbeddingsOptions,
    model: string,
): Promise<EmbeddingsResult> {
    const viaPrefix = TASK_TYPE_PREFIX_MODELS.has(shortModelName(model));
    const region = GLOBAL_ONLY_MODELS.has(shortModelName(model)) ? 'global' : undefined;
    const items = new Array<EmbeddingResultItem>(options.inputs.length);
    const usage: EmbeddingsTokenUsage = {};

    for (const [index, input] of options.inputs.entries()) {
        const content = await inputToContent(input, viaPrefix);
        const payload: Record<string, unknown> = {
            content,
        };
        if (input.type === 'text') {
            if (!viaPrefix) {
                payload.taskType = toGoogleTaskType(input.task_type);
            }
            payload.title = input.title;
        }
        if (options.dimensions !== undefined) {
            payload.outputDimensionality = options.dimensions;
        }

        try {
            const response = await driver.postVertexModel<EmbedContentResponse>(model, 'embedContent', payload, {
                region,
            });
            const values = response.embedding?.values;
            if (!values) {
                throw new Error(
                    `Vertex AI embedContent returned an empty embedding for input ${index} (model ${model})`,
                );
            }
            items[index] = {
                outputs: [{ values, modality: input.type }],
                input_tokens: response.usageMetadata?.promptTokenCount,
            };
            addEmbeddingUsage(usage, input, response.usageMetadata);
        } catch (error) {
            if (LlumiverseError.isLlumiverseError(error)) throw error;
            if (error instanceof Error && typeof (error as { status?: unknown }).status !== 'number') throw error;
            throw driver.formatLlumiverseError(error, {
                provider: 'vertexai',
                model,
                operation: 'execute',
            });
        }
    }

    return buildEmbeddingsResult(model, items, Object.keys(usage).length > 0 ? usage : undefined);
}

export async function generateVertexAiEmbeddings(
    driver: VertexAIDriver,
    options: EmbeddingsOptions,
): Promise<EmbeddingsResult> {
    const normalized = normalizeEmbeddingsOptions(options);
    const model = normalized.model ?? VERTEX_DEFAULT_EMBEDDING_MODEL;

    if (LEGACY_MULTIMODAL_MODELS.has(shortModelName(model))) {
        return generateLegacyMultimodalEmbeddings(driver, normalized);
    }
    if (usesTextPredict(model)) {
        return generateTextPredictEmbeddings(driver, normalized, model);
    }
    return generateEmbedContentEmbeddings(driver, normalized, model);
}
