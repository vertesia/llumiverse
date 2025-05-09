import { ModelModalities } from "../types.js";

// Model pattern lists for input modalities

/**
 * List of models that don't accept text input
 * These models should have text: false in their input modalities
 */
const NON_TEXT_INPUT_MODELS = [
    "nova-sonic"  // Only accepts audio input
];

const IMAGE_INPUT_MODELS = [
    // Amazon models
    "nova-canvas",
    "nova-lite",
    "nova-premier",
    "nova-pro",
    "nova-reel",
    "titan-embed-image",
    "titan-image-generator",
    // Anthropic models
    "claude-3-sonnet",
    "claude-3-opus",
    "claude-3-5-sonnet",
    "claude-3-7-sonnet",
    // Meta models
    "llama3-2-11b",
    "llama3-2-90b",
    "llama4-maverick",
    "llama4-scout",
    // Mistral models
    "pixtral",
    // Stability AI models
    "stability.sd3-5",
    "stability.sd3-large",
    "stability.stable-diffusion-xl"
];

const VIDEO_INPUT_MODELS = [
    "nova-lite",
    "nova-premier",
    "nova-pro"
];

const AUDIO_INPUT_MODELS = [
    "nova-sonic"
];

// Model pattern lists for output modalities
const TEXT_OUTPUT_MODELS = [
    // AI21 models
    "jamba",
    // Amazon models
    "nova-lite",
    "nova-micro",
    "nova-premier",
    "nova-pro",
    "nova-sonic",
    "titan-text",
    "rerank",
    // Anthropic models
    "claude",
    // Cohere models
    "command",
    // DeepSeek models
    "deepseek",
    // Meta models
    "llama",
    // Mistral models
    "mistral",
    "pixtral",
    // Writer models
    "palmyra"
];

const IMAGE_OUTPUT_MODELS = [
    "nova-canvas",
    "titan-image-generator",
    "stability.sd",
    "stability.stable-image"
];

const VIDEO_OUTPUT_MODELS = [
    "nova-reel",
    "luma.ray"
];

const AUDIO_OUTPUT_MODELS = [
    "nova-sonic"
];

const EMBEDDING_OUTPUT_MODELS = [
    "titan-embed",
    "cohere.embed"
];

/**
 * Extract the model identifier from an ARN or inference profile
 * 
 * @param modelName The full model ARN or name
 * @returns The normalized model identifier
 */
function normalizeModelName(modelName: string): string {
    const modelLower = modelName.toLowerCase();

    // Handle inference profiles by extracting the base model
    if (modelLower.includes("inference-profile")) {
        const parts = modelLower.split("/");
        if (parts.length > 1) {
            // Extract model from profile, e.g. "us.anthropic.claude-3-sonnet" → "claude-3-sonnet"
            const providerModel = parts[parts.length - 1];
            const modelParts = providerModel.split(".");
            return modelParts.length > 2 ? modelParts.slice(2).join(".") : providerModel;
        }
    }

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
 * Get the output modalities supported by a Bedrock model
 * 
 * @param model The model name or ARN
 * @returns Object containing the supported output modalities
 */
export function getOutputModalityBedrock(model: string): ModelModalities {
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
 * Get the input modalities supported by a Bedrock model
 * 
 * @param model The model name or ARN
 * @returns Object containing the supported input modalities
 */
export function getInputModalityBedrock(model: string): ModelModalities {
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
 * @param model The model name or ARN
 * @param modality The modality to check for
 * @returns Boolean indicating if the model supports the specified modality
 */
export function supportsOutputModalityBedrock(
    model: string,
    modality: keyof ModelModalities
): boolean {
    return !!getOutputModalityBedrock(model)[modality];
}

/**
 * Check if a model supports a specific input modality
 * 
 * @param model The model name or ARN
 * @param modality The modality to check for
 * @returns Boolean indicating if the model supports the specified modality
 */
export function supportsInputModalityBedrock(
    model: string,
    modality: keyof ModelModalities
): boolean {
    return !!getInputModalityBedrock(model)[modality];
}