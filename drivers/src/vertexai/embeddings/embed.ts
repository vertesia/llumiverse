import type { EmbedContentConfig } from '@google/genai';
import { VERTEX_DEFAULT_EMBEDDING_MODEL, VERTEX_MULTIMODAL_EMBEDDING_MODEL } from '@llumiverse/common';
import {
    buildEmbeddingsResult,
    type EmbeddingInput,
    type EmbeddingResultItem,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type EmbeddingsTokenUsage,
    LlumiverseError,
    normalizeEmbeddingsOptions,
    type TextEmbeddingInput,
} from '@llumiverse/core';
import type { VertexAIDriver } from '../index.js';
import { generateLegacyMultimodalEmbeddings } from './embed-legacy-multimodal.js';
import { toGoogleTaskType, vertexEmbeddingInputToContent } from './format.js';

export { buildVertexEmbeddingText, toGoogleTaskType, vertexEmbeddingInputToContent } from './format.js';

/**
 * Models that do not accept task_type as an API parameter and instead expect
 * the task to be conveyed by a documented prompt prefix.
 */
const TASK_TYPE_PREFIX_MODELS = new Set<string>(['gemini-embedding-2']);

/**
 * Apply the documented prompt prefix for gemini-embedding-2 (prefix-only model).
 *
 * Prefixes per Google docs:
 *   query    → "task: search result | query: {text}"
 *   document → "title: {title} | text: {text}"  (uses "none" when title is absent)
 */
/**
 * Models only available in the Vertex "global" location.
 */
const GLOBAL_ONLY_MODELS = new Set<string>(['gemini-embedding-2']);

/**
 * Models that only support one input content per embedContent request.
 */
const NON_GROUPING_MODELS = new Set<string>(['gemini-embedding-001', 'gemini-embedding-2']);

type TextConfig = Pick<EmbedContentConfig, 'taskType' | 'title'>;

function textConfig(input: TextEmbeddingInput, viaPrefix: boolean): TextConfig {
    const config: TextConfig = {};
    if (!viaPrefix && input.task_type) config.taskType = toGoogleTaskType(input.task_type);
    if (input.title) config.title = input.title;
    return config;
}

function configSignature(input: EmbeddingInput, viaPrefix: boolean): string {
    if (input.type !== 'text') return '{}';
    return JSON.stringify(textConfig(input, viaPrefix));
}

function configForGroup(
    representative: EmbeddingInput,
    viaPrefix: boolean,
    options: EmbeddingsOptions,
): EmbedContentConfig | undefined {
    const config: EmbedContentConfig = {};
    if (representative.type === 'text') {
        Object.assign(config, textConfig(representative, viaPrefix));
    }
    if (options.dimensions !== undefined) config.outputDimensionality = options.dimensions;
    return Object.keys(config).length > 0 ? config : undefined;
}

function addInputTokenUsage(usage: EmbeddingsTokenUsage, tokenCount: number): void {
    usage.input_text_tokens = (usage.input_text_tokens ?? 0) + tokenCount;
    usage.input_tokens = (usage.input_tokens ?? 0) + tokenCount;
}

/**
 * Models that use the legacy multimodal predict API instead of embedContent.
 */
const LEGACY_MULTIMODAL_MODELS = new Set<string>([VERTEX_MULTIMODAL_EMBEDDING_MODEL]);

/**
 * Generate Vertex AI embeddings via @google/genai's embedContent API.
 * Text inputs are sent as Content with a text part; task_type and title
 * are applied via the SDK config (or via prompt prefix for models that
 * don't accept them as API parameters).
 * Image/video/audio inputs are sent as inlineData (base64) or fileData (gs://).
 * Inputs with the same config signature are batched in a single call;
 * differing configs produce separate calls. Results preserve input order.
 */
export async function generateVertexAiEmbeddings(
    driver: VertexAIDriver,
    options: EmbeddingsOptions,
): Promise<EmbeddingsResult> {
    const normalized = normalizeEmbeddingsOptions(options);
    const model = normalized.model ?? VERTEX_DEFAULT_EMBEDDING_MODEL;

    if (LEGACY_MULTIMODAL_MODELS.has(model)) {
        return generateLegacyMultimodalEmbeddings(driver, normalized);
    }

    const viaPrefix = TASK_TYPE_PREFIX_MODELS.has(model);
    const region = GLOBAL_ONLY_MODELS.has(model) ? 'global' : undefined;
    const disableGrouping = NON_GROUPING_MODELS.has(model);

    const groups = new Map<string, { index: number; input: EmbeddingInput }[]>();
    normalized.inputs.forEach((input, index) => {
        const key = disableGrouping ? `single:${index}` : configSignature(input, viaPrefix);
        const group = groups.get(key);
        if (group) {
            group.push({ index, input });
            return;
        }
        groups.set(key, [{ index, input }]);
    });

    const ai = region ? driver.getGoogleGenAIClient(region) : driver.getGoogleGenAIClient();
    const items = new Array<EmbeddingResultItem>(normalized.inputs.length);
    const usage: EmbeddingsTokenUsage = {};

    for (const group of groups.values()) {
        const contents = await Promise.all(group.map((entry) => vertexEmbeddingInputToContent(entry.input, viaPrefix)));
        const config = configForGroup(group[0].input, viaPrefix, normalized);

        try {
            const response = await ai.models.embedContent({ model, contents, config });
            const embeddings = response.embeddings ?? [];
            if (embeddings.length !== group.length) {
                throw new Error(
                    `Vertex AI embedContent returned ${embeddings.length} embeddings for ${group.length} inputs (model ${model})`,
                );
            }

            embeddings.forEach((embedding, i) => {
                const entry = group[i];
                const values = embedding.values;
                if (!values) {
                    throw new Error(
                        `Vertex AI embedContent returned an empty embedding for input ${entry.index} (model ${model})`,
                    );
                }
                const tokenCount = embedding.statistics?.tokenCount;
                items[entry.index] = {
                    outputs: [{ values, modality: entry.input.type }],
                    input_tokens: tokenCount,
                };
                if (typeof tokenCount === 'number') {
                    addInputTokenUsage(usage, tokenCount);
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
