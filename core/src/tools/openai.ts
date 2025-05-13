/**
 * Models that explicitly do NOT support tool use in OpenAI
 */
const MODELS_WITHOUT_TOOL_USE = [
    // GPT-3 models
    "davinci",
    "curie",
    "babbage",
    "ada",
    // GPT-3.5 models
    "gpt-3.5-turbo",
    // Base GPT-4 models without tool support
    "gpt-4",
    "gpt-4-vision",
    // Transcribe models
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    // TTS models
    "gpt-4o-mini-tts",
    "tts",
    // Image models
    "gpt-image",
    "dall-e",
    // o1-mini
    "o1-mini",
    // Moderation models
    "omni-moderation",
    "text-moderation",
    // Embeddings models
    "text-embedding",
    "text-similarity",
    "text-search",
    // Whisper models
    "whisper"
];

/**
 * Models that explicitly do NOT support streaming tool use in OpenAI
 */
const MODELS_WITHOUT_STREAMING_TOOL_USE = [
    // Include all models that don't support tool use at all
    ...MODELS_WITHOUT_TOOL_USE
];

/**
 * Models that are known to support tool use in OpenAI
 */
const MODELS_WITH_TOOL_USE = [
    // GPT-4 turbo models
    "gpt-4-turbo",
    // GPT-4o models
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-audio",
    "gpt-4o-mini-audio",
    "gpt-4o-realtime",
    "gpt-4o-mini-realtime",
    // GPT-4.1 models
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    // o-series models
    "o1",
    "o1-pro",
    "o3-mini",
    "o3",
    "o4-mini"
];

/**
 * Models that are known to support streaming tool use in OpenAI
 */
const MODELS_WITH_STREAMING_TOOL_USE = [
    // All tool-supporting models can stream in OpenAI
    ...MODELS_WITH_TOOL_USE
];

/**
 * Checks if a model name matches a base model exactly or with a version suffix
 * 
 * @param modelName The model name to check
 * @param baseModel The base model name
 * @returns Whether the model matches the base model
 */
function modelMatches(modelName: string, baseModel: string): boolean {
    // Exact match
    if (modelName === baseModel) {
        return true;
    }

    // Check if it's a variant with any suffix (e.g., "gpt-4o-mini-search-preview-2025-03-11")
    if (modelName.startsWith(baseModel + '-')) {
        return true;
    }

    // Handle date-suffixed models without hyphens (e.g., "o1-2024-12-17")
    if (baseModel === modelName.split('-')[0]) {
        const parts = modelName.split('-');
        // Check if remaining parts form a date (e.g., 2024-12-17) or version
        if (parts.length >= 2) {
            // Match date patterns or version numbers
            return /^\d/.test(parts[1]);
        }
    }

    return false;
}

/**
 * Determines if an OpenAI model supports tool use based on the model name/ID.
 * 
 * @param model The model name to check for tool use support
 * @param streaming Optional parameter to check if the model supports streaming tool use
 * @returns true if the model supports tool use, false if it doesn't, undefined if unknown
 */
export function supportsToolUseOpenAI(model: string, streaming: boolean = false): boolean | undefined {
    // Normalize the model string for easier matching
    const modelLower = model.toLowerCase();

    // Determine which lists to check against based on the streaming parameter
    const supportList = streaming ? MODELS_WITH_STREAMING_TOOL_USE : MODELS_WITH_TOOL_USE;
    const noSupportList = streaming ?
        MODELS_WITHOUT_STREAMING_TOOL_USE :
        MODELS_WITHOUT_TOOL_USE;

    // First check if the model is explicitly known not to support tool use
    // Prioritize the exclusion list for safety
    if (noSupportList.some(baseModel => modelMatches(modelLower, baseModel))) {
        return false;
    }
    
    // Then check if the model is explicitly known to support tool use
    if (supportList.some(baseModel => modelMatches(modelLower, baseModel))) {
        return true;
    }

    // If neither match, return undefined (unknown status)
    return undefined;
}