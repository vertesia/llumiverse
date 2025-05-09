import { ModelModalities } from "../types.js";

// Model pattern lists for input modalities

/**
 * List of models that don't accept text input
 * These models should have text: false in their input modalities
 */
const NON_TEXT_INPUT_MODELS: string[] = [
    "whisper-1"  // Audio-only input (for transcription/translation)
];

const IMAGE_INPUT_MODELS = [
    // Models that can process images
    "gpt-4o",
    "gpt-4-vision",
    "gpt-4-turbo",
    "gpt-4o-mini",
    "gpt-4.5-preview",
    "o1",
    "o1-pro",
    "o3-mini",
    "o4-mini",
    "gpt-image-1",
    "dall-e-2",  // For image edits and variations
    // Note: o1-mini and o1-preview are text-only input
];

const AUDIO_INPUT_MODELS = [
    // Models that accept audio input
    "whisper-1",
    "gpt-4o-audio",
    "gpt-4o-audio-mini",
    "gpt-4o-audio-preview",
    "gpt-4o-realtime",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe"
];

// No models support video input for now
// @ts-ignore
const VIDEO_INPUT_MODELS: string[] = [];

// Model pattern lists for output modalities
const TEXT_OUTPUT_MODELS = [
    // GPT-4.x models
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4.5-preview",
    // GPT-4 models
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-search-preview",
    "gpt-4o-realtime-preview",
    "gpt-4o-realtime",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4-vision",
    // GPT-4o audio models
    "gpt-4o-audio",
    "gpt-4o-audio-mini",
    "gpt-4o-audio-preview",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
    // GPT-3.5 models
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-instruct",
    // O family models
    "o1",
    "o1-mini",
    "o1-pro",
    "o1-preview",
    "o3-mini",
    "o4-mini",
    "chatgpt-4o-latest",
    // Legacy completion models
    "davinci-002",
    "babbage-002",
    // Transcription/translation
    "whisper-1",
    // Moderation models (correct treatment)
    "text-moderation-latest",
    "text-moderation-stable",
    "omni-moderation-latest",
    "omni-moderation-2024-09-26"
];

const IMAGE_OUTPUT_MODELS = [
    "dall-e-2",
    "dall-e-3",
    "gpt-image-1"
];

const AUDIO_OUTPUT_MODELS = [
    "tts-1",
    "tts-1-hd",
    "gpt-4o-audio",
    "gpt-4o-audio-mini",
    "gpt-4o-audio-preview",
    "gpt-4o-realtime"
];

const EMBEDDING_OUTPUT_MODELS = [
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large"
];

/**
 * Normalizes the model name to handle different formats and versions
 * 
 * @param modelName The full model name
 * @returns The normalized model identifier
 */
function normalizeModelName(modelName: string): string {
    const modelLower = modelName.toLowerCase();

    // Strip date suffixes from model names for easier matching
    // Example: o1-mini-2024-09-12 → o1-mini
    const stripDatePattern = /-\d{4}-\d{2}-\d{2}$/;
    if (stripDatePattern.test(modelLower)) {
        return modelLower.replace(stripDatePattern, "");
    }

    // Handle O family models
    if (modelLower.startsWith("o1-")) {
        if (modelLower.includes("mini")) return "o1-mini";
        if (modelLower.includes("pro")) return "o1-pro";
        if (modelLower.includes("preview")) return "o1-preview";
        return "o1";
    }

    if (modelLower.startsWith("o3-")) {
        if (modelLower.includes("mini")) return "o3-mini";
        return "o3";
    }

    if (modelLower.startsWith("o4-")) {
        if (modelLower.includes("mini")) return "o4-mini";
        return "o4";
    }

    // Handle GPT-4.x models
    if (modelLower.startsWith("gpt-4.1-") || modelLower.startsWith("gpt-4.5-")) {
        if (modelLower.includes("nano")) return "gpt-4.1-nano";
        if (modelLower.includes("mini")) return "gpt-4.1-mini";
        if (modelLower.includes("4.5")) return "gpt-4.5-preview";
        return modelLower.split("-").slice(0, 2).join("-"); // Get base model name
    }

    if (modelLower === "gpt-4.1") return "gpt-4.1";
    if (modelLower === "gpt-4.5") return "gpt-4.5-preview";

    // Handle GPT-4o variants with updated audio/transcribe models
    if (modelLower.startsWith("gpt-4o-")) {
        if (modelLower.includes("mini") && modelLower.includes("transcribe")) {
            return "gpt-4o-mini-transcribe";
        } else if (modelLower.includes("mini") && modelLower.includes("search")) {
            return "gpt-4o-mini-search-preview";
        } else if (modelLower.includes("mini") && modelLower.includes("audio")) {
            return "gpt-4o-audio-mini";
        } else if (modelLower.includes("mini")) {
            return "gpt-4o-mini";
        } else if (modelLower.includes("audio")) {
            return "gpt-4o-audio";
        } else if (modelLower.includes("transcribe")) {
            return "gpt-4o-transcribe";
        } else if (modelLower.includes("search")) {
            return "gpt-4o-search-preview";
        } else if (modelLower.includes("realtime")) {
            return "gpt-4o-realtime";
        }
        return "gpt-4o";
    }

    if (modelLower === "gpt-4o") return "gpt-4o";

    // Handle GPT-4 variants
    if (modelLower.startsWith("gpt-4-")) {
        if (modelLower.includes("turbo")) {
            if (modelLower.includes("preview")) {
                return "gpt-4-turbo-preview";
            }
            return "gpt-4-turbo";
        } else if (modelLower.includes("32k")) {
            return "gpt-4-32k";
        } else if (modelLower.includes("vision")) {
            return "gpt-4-vision";
        } else if (modelLower.includes("preview")) {
            return "gpt-4-turbo-preview";
        }
        return "gpt-4";
    }

    if (modelLower === "gpt-4") return "gpt-4";

    // Handle GPT-3.5 variants
    if (modelLower.startsWith("gpt-3.5-turbo")) {
        if (modelLower.includes("16k")) {
            return "gpt-3.5-turbo-16k";
        } else if (modelLower.includes("instruct")) {
            return "gpt-3.5-turbo-instruct";
        }
        return "gpt-3.5-turbo";
    }

    // Other model families
    if (modelLower.startsWith("dall-e-")) {
        if (modelLower.includes("3")) {
            return "dall-e-3";
        }
        return "dall-e-2";
    }

    if (modelLower.includes("embedding")) {
        if (modelLower.includes("3-large")) {
            return "text-embedding-3-large";
        } else if (modelLower.includes("3-small")) {
            return "text-embedding-3-small";
        }
        return "text-embedding-ada-002";
    }

    if (modelLower.includes("moderation")) {
        if (modelLower.includes("omni")) {
            return "omni-moderation-latest";
        }
        return "text-moderation-latest";
    }

    if (modelLower.startsWith("tts-")) {
        if (modelLower.includes("hd")) {
            return "tts-1-hd";
        }
        return "tts-1";
    }

    if (modelLower.includes("chatgpt-4o")) {
        return "chatgpt-4o-latest";
    }

    if (modelLower === "gpt-image-1") {
        return "gpt-image-1";
    }

    // For other models, return as is
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
 * Get the output modalities supported by an OpenAI model
 * 
 * @param model The model name
 * @returns Object containing the supported output modalities
 */
export function getOutputModalityOpenAI(model: string): ModelModalities {
    const normalizedModel = normalizeModelName(model);

    const modalities: ModelModalities = {
        text: modelMatches(normalizedModel, TEXT_OUTPUT_MODELS),
        image: modelMatches(normalizedModel, IMAGE_OUTPUT_MODELS),
        audio: modelMatches(normalizedModel, AUDIO_OUTPUT_MODELS),
        embedding: modelMatches(normalizedModel, EMBEDDING_OUTPUT_MODELS),
        video: false  // OpenAI doesn't currently have video output models
    };

    // Special handling for embedding models
    if (modelMatches(normalizedModel, EMBEDDING_OUTPUT_MODELS)) {
        modalities.text = false;  // Embedding models don't generate text
    }

    return modalities;
}

/**
 * Get the input modalities supported by an OpenAI model
 * 
 * @param model The model name
 * @returns Object containing the supported input modalities
 */
export function getInputModalityOpenAI(model: string): ModelModalities {
    const normalizedModel = normalizeModelName(model);

    // Default - most models accept text input unless in NON_TEXT_INPUT_MODELS
    const modalities: ModelModalities = {
        text: !modelMatches(normalizedModel, NON_TEXT_INPUT_MODELS),
        image: modelMatches(normalizedModel, IMAGE_INPUT_MODELS),
        audio: modelMatches(normalizedModel, AUDIO_INPUT_MODELS),
        video: false,  // No OpenAI models support video input
        embedding: false  // No models take embeddings as input
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
export function supportsOutputModalityOpenAI(
    model: string,
    modality: keyof ModelModalities
): boolean {
    return !!getOutputModalityOpenAI(model)[modality];
}

/**
 * Check if a model supports a specific input modality
 * 
 * @param model The model name
 * @param modality The modality to check for
 * @returns Boolean indicating if the model supports the specified modality
 */
export function supportsInputModalityOpenAI(
    model: string,
    modality: keyof ModelModalities
): boolean {
    return !!getInputModalityOpenAI(model)[modality];
}