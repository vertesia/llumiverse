import { ModelModalities } from "../types.js";

// Model pattern lists for input modalities

/**
 * List of models that don't accept text input
 * These models should have text: false in their input modalities
 */
const NON_TEXT_INPUT_MODELS: string[] = [
    // Currently all Vertex AI models support text input
];

const IMAGE_INPUT_MODELS = [
    // Gemini models
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.5-flash-preview",
    "gemini-2.5-pro-preview",
    // Veo model
    "veo-2.0-generate"
];

const VIDEO_INPUT_MODELS = [
    // Gemini models
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.5-flash-preview",
    "gemini-2.5-pro-preview",
    // Live models
    "gemini-2.0-flash-live"
];

const AUDIO_INPUT_MODELS = [
    // Gemini models
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.5-flash-preview",
    "gemini-2.5-pro-preview",
    // Live models
    "gemini-2.0-flash-live"
];

// Model pattern lists for output modalities
const TEXT_OUTPUT_MODELS = [
    // Gemini general models
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.5-flash-preview",
    "gemini-2.5-pro-preview",
    // Live models
    "gemini-2.0-flash-live",
    // Claude models (on Vertex AI)
    "claude",
    // Llama models (on Vertex AI)
    "llama"
];

const IMAGE_OUTPUT_MODELS = [
    // Imagen models
    "imagen-3.0-generate",
    // Gemini image generation model
    "gemini-2.0-flash-preview-image-generation"
];

const VIDEO_OUTPUT_MODELS = [
    // Veo models
    "veo-2.0-generate"
];

const AUDIO_OUTPUT_MODELS = [
    // Live models with audio output
    "gemini-2.0-flash-live"
];

const EMBEDDING_OUTPUT_MODELS = [
    // Gemini embedding model
    "gemini-embedding"
];

/**
 * Normalizes the model name to handle different formats
 * 
 * @param modelName The full model name
 * @returns The normalized model identifier
 */
function normalizeModelName(modelName: string): string {
    const modelLower = modelName.toLowerCase();

    // Extract just the base model name without version specifics if needed
    // This is less critical for Vertex AI than for Bedrock's ARNs,
    // but still helpful for consistency

    return modelLower;
}

/**
 * Check if a model name contains any of the pattern strings
 * 
 * @param modelName The normalized model name
 * @param patterns Array of pattern strings to match against
 * @returns true if the model name matches any pattern
 */
function modelMatches(modelName: string, patterns: string[]): boolean {
    return patterns.some(pattern => modelName.includes(pattern));
}

/**
 * Get the output modalities supported by a Vertex AI model
 * 
 * @param model The model name
 * @returns Object containing the supported output modalities
 */
export function getOutputModalityVertexAI(model: string): ModelModalities {
    const normalizedModel = normalizeModelName(model);

    const modalities: ModelModalities = {
        text: modelMatches(normalizedModel, TEXT_OUTPUT_MODELS),
        image: modelMatches(normalizedModel, IMAGE_OUTPUT_MODELS),
        video: modelMatches(normalizedModel, VIDEO_OUTPUT_MODELS),
        audio: modelMatches(normalizedModel, AUDIO_OUTPUT_MODELS),
        embedding: modelMatches(normalizedModel, EMBEDDING_OUTPUT_MODELS)
    };

    return modalities;
}

/**
 * Get the input modalities supported by a Vertex AI model
 * 
 * @param model The model name
 * @returns Object containing the supported input modalities
 */
export function getInputModalityVertexAI(model: string): ModelModalities {
    const normalizedModel = normalizeModelName(model);

    // Most models accept text input by default, unless they're in NON_TEXT_INPUT_MODELS
    const modalities: ModelModalities = {
        text: !modelMatches(normalizedModel, NON_TEXT_INPUT_MODELS),
        image: modelMatches(normalizedModel, IMAGE_INPUT_MODELS),
        video: modelMatches(normalizedModel, VIDEO_INPUT_MODELS),
        audio: modelMatches(normalizedModel, AUDIO_INPUT_MODELS),
        embedding: false // No models support embedding input
    };

    return modalities;
}

/**
 * Check if a model supports a specific output modality
 * 
 * @param model The model name
 * @param modality The modality to check for
 * @returns Boolean indicating if the model supports the specified modality
 */
export function supportsOutputModalityVertexAI(
    model: string,
    modality: keyof ModelModalities
): boolean {
    return !!getOutputModalityVertexAI(model)[modality];
}

/**
 * Check if a model supports a specific input modality
 * 
 * @param model The model name
 * @param modality The modality to check for
 * @returns Boolean indicating if the model supports the specified modality
 */
export function supportsInputModalityVertexAI(
    model: string,
    modality: keyof ModelModalities
): boolean {
    return !!getInputModalityVertexAI(model)[modality];
}