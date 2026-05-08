import type { Content, EmbedContentConfig, Part } from "@google/genai";
import {
    applyTaskTypePrefix,
    buildEmbeddingsResult,
    type DataSource,
    type EmbeddingInput,
    type EmbeddingResultItem,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type EmbeddingsTokenUsage,
    type EmbeddingTaskType,
    type TextEmbeddingInput,
} from "@llumiverse/core";
import type { VertexAIDriver } from "../index.js";
import { generateLegacyMultimodalEmbeddings } from "./embed-legacy-multimodal.js";
import { dataSourceToVertexSourceData } from "./source-utils.js";

const DEFAULT_MODEL = "gemini-embedding-2";

/**
 * Models that do not accept task_type as an API parameter and instead expect
 * the task to be conveyed by a documented prompt prefix.
 */
const TASK_TYPE_PREFIX_MODELS = new Set<string>([
    "gemini-embedding-2",
]);

/**
 * Documented prompt prefixes for prefix-only models.
 * See Vertex AI text embeddings docs.
 */
const TASK_TYPE_PREFIX_MAP: Partial<Record<EmbeddingTaskType, string>> = {
    RETRIEVAL_QUERY: "task: search result | query: ",
    RETRIEVAL_DOCUMENT: "task: search result | ",
    SEMANTIC_SIMILARITY: "task: sentence similarity | query: ",
    CLASSIFICATION: "task: classification | query: ",
    CLUSTERING: "task: clustering | query: ",
    QUESTION_ANSWERING: "task: question answering | query: ",
    FACT_VERIFICATION: "task: fact checking | query: ",
    CODE_RETRIEVAL_QUERY: "task: code retrieval | query: ",
};

/**
 * Models only available in the Vertex "global" location.
 */
const GLOBAL_ONLY_MODELS = new Set<string>([
    "gemini-embedding-2",
]);

/**
 * Models that only support one input content per embedContent request.
 */
const NON_GROUPING_MODELS = new Set<string>([
    "gemini-embedding-001",
    "gemini-embedding-2",
]);

async function dataSourceToPart(ds: DataSource): Promise<Part> {
    const source = await dataSourceToVertexSourceData(ds);
    if (source.gcsUri) {
        return { fileData: { fileUri: source.gcsUri, mimeType: ds.mime_type } };
    }

    if (!source.bytesBase64Encoded) {
        throw new Error("Data source conversion produced neither GCS URI nor inline bytes");
    }

    return { inlineData: { data: source.bytesBase64Encoded, mimeType: ds.mime_type } };
}

type TextConfig = Pick<EmbedContentConfig, "taskType" | "title" | "autoTruncate">;

function textConfig(input: TextEmbeddingInput, viaPrefix: boolean): TextConfig {
    const config: TextConfig = {};
    if (!viaPrefix && input.task_type) config.taskType = input.task_type;
    if (input.title) config.title = input.title;
    if (input.truncate !== undefined) config.autoTruncate = input.truncate !== "NONE";
    return config;
}

function configSignature(input: EmbeddingInput, viaPrefix: boolean): string {
    if (input.type !== "text") return "{}";
    return JSON.stringify(textConfig(input, viaPrefix));
}

async function inputToContent(input: EmbeddingInput, viaPrefix: boolean): Promise<Content> {
    if (input.type === "text") {
        const text = viaPrefix
            ? applyTaskTypePrefix(input.text, input.task_type, TASK_TYPE_PREFIX_MAP)
            : input.text;
        return { role: "user", parts: [{ text }] };
    }
    return { role: "user", parts: [await dataSourceToPart(input.source)] };
}

function configForGroup(
    representative: EmbeddingInput,
    viaPrefix: boolean,
    options: EmbeddingsOptions,
): EmbedContentConfig | undefined {
    const config: EmbedContentConfig = {};
    if (representative.type === "text") {
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
const LEGACY_MULTIMODAL_MODELS = new Set<string>([
    "multimodalembedding@001",
]);

/**
 * Generate Vertex AI embeddings via @google/genai's embedContent API.
 * Unified path for text and multimodal inputs:
 * - Text inputs are sent as Content with a text part. task_type/title/truncate
 *   are applied via the SDK's config (or via prompt prefix for allow-listed
 *   models that don't accept the API parameter).
 * - Image/video/audio inputs are sent as Content with inlineData (base64) or
 *   fileData (gs:// URL) parts.
 *
 * Inputs sharing the same (task_type, title, truncate) signature are batched
 * in a single embedContent call; mismatched configs split into multiple calls.
 * Results are returned in the same order as options.inputs.
 */
export async function generateVertexAiEmbeddings(
    driver: VertexAIDriver,
    options: EmbeddingsOptions,
): Promise<EmbeddingsResult> {
    const model = options.model ?? DEFAULT_MODEL;

    if (LEGACY_MULTIMODAL_MODELS.has(model)) {
        return generateLegacyMultimodalEmbeddings(driver, options);
    }

    const viaPrefix = TASK_TYPE_PREFIX_MODELS.has(model);
    const region = GLOBAL_ONLY_MODELS.has(model) ? "global" : undefined;
    const disableGrouping = NON_GROUPING_MODELS.has(model);

    const groups = new Map<string, { index: number; input: EmbeddingInput }[]>();
    options.inputs.forEach((input, index) => {
        const key = disableGrouping ? `single:${index}` : configSignature(input, viaPrefix);
        const group = groups.get(key);
        if (group) {
            group.push({ index, input });
            return;
        }
        groups.set(key, [{ index, input }]);
    });

    const ai = region ? driver.getGoogleGenAIClient(region) : driver.getGoogleGenAIClient();
    const items = new Array<EmbeddingResultItem>(options.inputs.length);
    const usage: EmbeddingsTokenUsage = {};

    for (const group of groups.values()) {
        const contents = await Promise.all(group.map((entry) => inputToContent(entry.input, viaPrefix)));
        const config = configForGroup(group[0].input, viaPrefix, options);

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
                throw new Error(`Vertex AI embedContent returned an empty embedding for input ${entry.index} (model ${model})`);
            }
            const tokenCount = embedding.statistics?.tokenCount;
            items[entry.index] = {
                outputs: [{ values, modality: entry.input.type }],
                input_tokens: tokenCount,
            };
            if (typeof tokenCount === "number") {
                addInputTokenUsage(usage, tokenCount);
            }
        });
    }

    return buildEmbeddingsResult(model, items, Object.keys(usage).length > 0 ? usage : undefined);
}
