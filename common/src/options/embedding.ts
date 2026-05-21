import type { EmbeddingInput, Providers } from "../types.js";

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
