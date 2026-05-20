import {
    type AIModel,
    type EmbeddingInput,
    type EmbeddingModelCapabilities,
    type EmbeddingTaskType,
    ModelType,
    type Providers,
} from "../types.js";

/** Default embedding model ids, one per provider. */

// ── Vertex AI ──────────────────────────────────────────────────────────────
/** Default Vertex AI text/multimodal embedding model (supports text + image/video/audio via embedContent). */
export const VERTEX_DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2";
/** Legacy Vertex AI multimodal predict API model (joint text+image+video in one vector). */
export const VERTEX_MULTIMODAL_EMBEDDING_MODEL = "multimodalembedding@001";

// ── AWS Bedrock ────────────────────────────────────────────────────────────
/** Default Bedrock embedding model — Nova 2 multimodal (text, image, video, audio). */
export const BEDROCK_DEFAULT_EMBEDDING_MODEL = "amazon.nova-2-multimodal-embeddings-v1:0";
/** Bedrock Titan text embedding model (text only). */
export const BEDROCK_TITAN_TEXT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0";
/** Bedrock Titan image embedding model (text + image). */
export const BEDROCK_TITAN_IMAGE_EMBEDDING_MODEL = "amazon.titan-embed-image-v1";
/** Bedrock Cohere English embedding model (text + image, native batch). */
export const BEDROCK_COHERE_ENGLISH_EMBEDDING_MODEL = "cohere.embed-english-v3";
/** Bedrock Cohere multilingual embedding model (text + image, native batch). */
export const BEDROCK_COHERE_MULTILINGUAL_EMBEDDING_MODEL = "cohere.embed-multilingual-v3";

// ── OpenAI ─────────────────────────────────────────────────────────────────
/** Default OpenAI embedding model (text only, supports dimensions). */
export const OPENAI_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small";
/** OpenAI large embedding model (text only, higher capacity). */
export const OPENAI_LARGE_EMBEDDING_MODEL = "text-embedding-3-large";

// ── Mistral AI ─────────────────────────────────────────────────────────────
/** Default Mistral AI embedding model (text only). */
export const MISTRAL_DEFAULT_EMBEDDING_MODEL = "mistral-embed";

// ── IBM Watsonx ────────────────────────────────────────────────────────────
/** Default Watsonx embedding model (text only). */
export const WATSONX_DEFAULT_EMBEDDING_MODEL = "ibm/slate-125m-english-rtrvr";

// ── Azure Foundry ──────────────────────────────────────────────────────────
/**
 * Azure Foundry has no global default — the model is deployment-specific and
 * must always be specified explicitly.
 */
export const AZURE_FOUNDRY_DEFAULT_EMBEDDING_MODEL: undefined = undefined;

// ── Provider → default model map ───────────────────────────────────────────

type EmbeddingModality = EmbeddingInput["type"];

/**
 * Per-provider default embedding model, keyed first by provider then by the
 * primary modality of the input. When no modality-specific default exists for
 * a provider, the `text` entry is used as the fallback.
 *
 * Providers that have a single multimodal model (Bedrock Nova, Vertex) return
 * the same model for all modalities.
 */
const DEFAULT_EMBEDDING_MODELS: Partial<Record<Providers, Partial<Record<EmbeddingModality, string>>>> = {
    vertexai: {
        text: VERTEX_DEFAULT_EMBEDDING_MODEL,
        image: VERTEX_DEFAULT_EMBEDDING_MODEL,
        video: VERTEX_DEFAULT_EMBEDDING_MODEL,
        audio: VERTEX_DEFAULT_EMBEDDING_MODEL,
    },
    bedrock: {
        text: BEDROCK_DEFAULT_EMBEDDING_MODEL,
        image: BEDROCK_DEFAULT_EMBEDDING_MODEL,
        video: BEDROCK_DEFAULT_EMBEDDING_MODEL,
        audio: BEDROCK_DEFAULT_EMBEDDING_MODEL,
    },
    openai: {
        text: OPENAI_DEFAULT_EMBEDDING_MODEL,
    },
    mistralai: {
        text: MISTRAL_DEFAULT_EMBEDDING_MODEL,
    },
    watsonx: {
        text: WATSONX_DEFAULT_EMBEDDING_MODEL,
    },
};

/**
 * Returns the default embedding model for a provider and modality.
 * Falls back to the provider's `text` default when no modality-specific entry exists.
 * Returns `undefined` for providers without a global default (e.g. Azure Foundry).
 */
export function getDefaultEmbeddingModel(
    provider: Providers,
    modality: EmbeddingModality = "text",
): string | undefined {
    const byProvider = DEFAULT_EMBEDDING_MODELS[provider];
    if (!byProvider) return undefined;
    return byProvider[modality] ?? byProvider["text"];
}

// ── Embedding model catalog ────────────────────────────────────────────────

/**
 * Static descriptor for an embedding model. Sourced from provider documentation
 * because the corresponding list-models APIs do not return dimensions/modalities.
 */
export interface EmbeddingModelDescriptor {
    id: string;
    name: string;
    description?: string;
    /** Modalities the model accepts as input. */
    input_modalities: EmbeddingModality[];
    /** Capability block populated onto `AIModel.embedding`. */
    embedding: EmbeddingModelCapabilities;
    /** Free-form tags surfaced as `AIModel.tags` (e.g. "default", "legacy"). */
    tags?: string[];
}

const COMMON_TASK_TYPES: EmbeddingTaskType[] = ["query", "document"];

/**
 * Per-provider catalog of known embedding models and their capabilities.
 *
 * Note: values below are sourced from provider documentation at the time of
 * authoring. Verify against the provider's current docs before depending on
 * specific dimension values for production use.
 */
export const EMBEDDING_MODEL_CATALOG: Partial<Record<Providers, EmbeddingModelDescriptor[]>> = {
    openai: [
        {
            id: OPENAI_DEFAULT_EMBEDDING_MODEL,
            name: "OpenAI text-embedding-3-small",
            description: "Small, low-latency OpenAI text embedding model with MRL truncation.",
            input_modalities: ["text"],
            tags: ["default"],
            embedding: {
                default_dimensions: 1536,
                supported_dimensions: [512, 1024, 1536],
                supports_dimension_truncation: true,
                max_input_tokens: 8191,
            },
        },
        {
            id: OPENAI_LARGE_EMBEDDING_MODEL,
            name: "OpenAI text-embedding-3-large",
            description: "Higher-capacity OpenAI text embedding model with MRL truncation.",
            input_modalities: ["text"],
            embedding: {
                default_dimensions: 3072,
                supported_dimensions: [256, 1024, 3072],
                supports_dimension_truncation: true,
                max_input_tokens: 8191,
            },
        },
        {
            id: "text-embedding-ada-002",
            name: "OpenAI text-embedding-ada-002",
            description: "Legacy OpenAI text embedding model with fixed 1536 dimensions.",
            input_modalities: ["text"],
            tags: ["legacy"],
            embedding: {
                default_dimensions: 1536,
                max_input_tokens: 8191,
            },
        },
    ],
    vertexai: [
        {
            id: VERTEX_DEFAULT_EMBEDDING_MODEL,
            name: "Vertex AI Gemini Embedding 2",
            description: "Vertex AI Gemini multimodal embedding model (text, image, video, audio).",
            input_modalities: ["text", "image", "video", "audio"],
            tags: ["default"],
            embedding: {
                default_dimensions: 3072,
                supported_dimensions: [768, 1536, 3072],
                supports_dimension_truncation: true,
                supported_task_types: COMMON_TASK_TYPES,
            },
        },
        {
            id: VERTEX_MULTIMODAL_EMBEDDING_MODEL,
            name: "Vertex AI multimodalembedding@001",
            description: "Legacy Vertex AI multimodal embedding (joint text/image/video vectors).",
            input_modalities: ["text", "image", "video"],
            tags: ["legacy"],
            embedding: {
                default_dimensions: 1408,
                supported_dimensions: [128, 256, 512, 1408],
            },
        },
        {
            id: "text-embedding-005",
            name: "Vertex AI text-embedding-005",
            description: "Vertex AI English text embedding model.",
            input_modalities: ["text"],
            embedding: {
                default_dimensions: 768,
                supports_dimension_truncation: true,
                supported_task_types: COMMON_TASK_TYPES,
            },
        },
    ],
    bedrock: [
        {
            id: BEDROCK_DEFAULT_EMBEDDING_MODEL,
            name: "Amazon Nova 2 Multimodal Embeddings",
            description: "Amazon Nova 2 multimodal embedding model (text, image, video, audio).",
            input_modalities: ["text", "image", "video", "audio"],
            tags: ["default"],
            embedding: {
                default_dimensions: 3072,
                supported_dimensions: [256, 384, 1024, 3072],
            },
        },
        {
            id: BEDROCK_TITAN_TEXT_EMBEDDING_MODEL,
            name: "Amazon Titan Text Embeddings v2",
            description: "Amazon Titan text embedding model with discrete dimension options.",
            input_modalities: ["text"],
            embedding: {
                default_dimensions: 1024,
                supported_dimensions: [256, 512, 1024],
            },
        },
        {
            id: BEDROCK_TITAN_IMAGE_EMBEDDING_MODEL,
            name: "Amazon Titan Multimodal Embeddings G1",
            description: "Amazon Titan multimodal embedding model (text + image).",
            input_modalities: ["text", "image"],
            embedding: {
                default_dimensions: 1024,
                supported_dimensions: [256, 384, 1024],
            },
        },
        {
            id: BEDROCK_COHERE_ENGLISH_EMBEDDING_MODEL,
            name: "Cohere Embed English v3",
            description: "Cohere English text embedding model with fixed 1024 dimensions.",
            input_modalities: ["text", "image"],
            embedding: {
                default_dimensions: 1024,
                supported_task_types: COMMON_TASK_TYPES,
            },
        },
        {
            id: BEDROCK_COHERE_MULTILINGUAL_EMBEDDING_MODEL,
            name: "Cohere Embed Multilingual v3",
            description: "Cohere multilingual text embedding model with fixed 1024 dimensions.",
            input_modalities: ["text", "image"],
            embedding: {
                default_dimensions: 1024,
                supported_task_types: COMMON_TASK_TYPES,
            },
        },
    ],
    mistralai: [
        {
            id: MISTRAL_DEFAULT_EMBEDDING_MODEL,
            name: "Mistral Embed",
            description: "Mistral AI text embedding model with fixed 1024 dimensions.",
            input_modalities: ["text"],
            tags: ["default"],
            embedding: {
                default_dimensions: 1024,
                max_input_tokens: 8192,
            },
        },
    ],
    watsonx: [
        {
            id: WATSONX_DEFAULT_EMBEDDING_MODEL,
            name: "IBM Slate 125m English Retriever",
            description: "IBM Slate English text embedding model with fixed 768 dimensions.",
            input_modalities: ["text"],
            tags: ["default"],
            embedding: {
                default_dimensions: 768,
                max_input_tokens: 512,
            },
        },
    ],
};

/** Output modality for embedding models — they always emit numeric vectors. */
const EMBEDDING_OUTPUT_MODALITIES = ["vector"];

/**
 * Build an `AIModel` from a static descriptor for the given provider.
 */
export function embeddingDescriptorToAIModel(
    provider: Providers,
    descriptor: EmbeddingModelDescriptor,
): AIModel {
    return {
        id: descriptor.id,
        name: descriptor.name,
        provider,
        description: descriptor.description,
        type: ModelType.Embedding,
        tags: descriptor.tags,
        is_multimodal: descriptor.input_modalities.length > 1,
        input_modalities: [...descriptor.input_modalities],
        output_modalities: [...EMBEDDING_OUTPUT_MODALITIES],
        embedding: descriptor.embedding,
    };
}

/**
 * Returns the full static catalog of embedding models for a provider as `AIModel[]`.
 * Returns an empty array for providers without a catalog entry.
 */
export function getEmbeddingModelsForProvider(provider: Providers): AIModel[] {
    const descriptors = EMBEDDING_MODEL_CATALOG[provider] ?? [];
    return descriptors.map((d) => embeddingDescriptorToAIModel(provider, d));
}

/**
 * Returns the static catalog descriptor for a single (provider, model id) pair, or
 * `undefined` when the model is not in the catalog.
 */
export function getEmbeddingModelDescriptor(
    provider: Providers,
    modelId: string,
): EmbeddingModelDescriptor | undefined {
    return EMBEDDING_MODEL_CATALOG[provider]?.find((d) => d.id === modelId);
}

/**
 * Heuristic that decides whether an `AIModel` is an embedding model.
 *
 * Prefers the principled `type === ModelType.Embedding` check (set by drivers that have
 * a real type from the provider, e.g. Azure Foundry's `capabilities.embeddings`, or after
 * `enrichWithEmbeddingCatalog` has tagged a known id). Falls back to a substring match on
 * id/name — every embedding model we expose by id contains "embed" except WatsonX's
 * `ibm/slate-...-rtrvr`, which always arrives with the type already set by the driver.
 */
export function isEmbeddingModel(m: AIModel): boolean {
    if (m.type === ModelType.Embedding) return true;
    const id = m.id?.toLowerCase() ?? "";
    const name = m.name?.toLowerCase() ?? "";
    return id.includes("embed") || name.includes("embed");
}

/**
 * Merge static catalog metadata onto an array of `AIModel`s returned by a provider's
 * list-models API. Models whose id matches a catalog entry are enriched with the
 * `embedding`, `input_modalities`, `output_modalities`, `is_multimodal`, and `tags`
 * fields from the catalog. Unknown ids are kept (with `type: ModelType.Embedding` set)
 * only when they look like embedding models — non-embedding leaks from a driver-side
 * filter are dropped here as a defensive backstop.
 */
export function enrichWithEmbeddingCatalog(provider: Providers, models: AIModel[]): AIModel[] {
    const byId = new Map<string, EmbeddingModelDescriptor>();
    for (const d of EMBEDDING_MODEL_CATALOG[provider] ?? []) {
        byId.set(d.id, d);
    }
    const enriched: AIModel[] = [];
    for (const m of models) {
        const d = byId.get(m.id);
        if (d) {
            enriched.push({
                ...m,
                type: ModelType.Embedding,
                description: m.description ?? d.description,
                tags: m.tags ?? d.tags,
                is_multimodal: m.is_multimodal ?? d.input_modalities.length > 1,
                input_modalities: m.input_modalities ?? [...d.input_modalities],
                output_modalities: m.output_modalities ?? [...EMBEDDING_OUTPUT_MODALITIES],
                embedding: m.embedding ?? d.embedding,
            });
            continue;
        }
        // Unknown id — keep only when the driver already tagged it OR the id/name look like an embedding model.
        // The check intentionally runs on the *incoming* model so we don't paper over a non-embedding leak
        // by setting `type` to Embedding ourselves.
        if (isEmbeddingModel(m)) {
            enriched.push(m.type ? m : { ...m, type: ModelType.Embedding });
        }
    }
    return enriched;
}
