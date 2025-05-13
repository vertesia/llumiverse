/**
 * Models that explicitly do NOT support tool use in Vertex AI
 */
const MODELS_WITHOUT_TOOL_USE = [
    // Gemini models that do not support tool use
    "gemini-pro",
    "gemini-ultra",
    "gemini-pro-vision",
    "gemini-ultra-vision",
    "gemini-2.0-flash-lite",
    // Imagen models
    "imagen-3.0-generate",
    "imagen-3.0-capability",
    // Claude models
    "claude-2",
    "claude-instant",
    "claude-1",
    // Llama models
    "llama-2"
];

/**
 * Models that explicitly do NOT support streaming tool use in Vertex AI
 */
const MODELS_WITHOUT_STREAMING_TOOL_USE = [
    // Include all models that don't support tool use at all
    ...MODELS_WITHOUT_TOOL_USE
];

/**
 * Models that are known to support tool use in Vertex AI
 */
const MODELS_WITH_TOOL_USE = [
    // Gemini models that support tool use
    "gemini-1.5",
    "gemini-2.0",
    "gemini-2.5",
    // Claude models
    "claude-3-5",
    "claude-3-7",
    // Llama models
    "llama-3.1",
    "llama-3.2"
];

/**
 * Models that are known to support streaming tool use in Vertex AI
 */
const MODELS_WITH_STREAMING_TOOL_USE = [
    // Gemini models that support streaming tool use
    "gemini-1.5",
    "gemini-2.0",
    "gemini-2.5",
    // Claude models
    "claude-3-5",
    "claude-3-7"
];

/**
 * Checks if a model name matches a base model exactly or with a feature suffix or a version suffix,
 * accounting for path prefixes like "publishers/anthropic/models/" and similar
 * 
 * @param modelName The model name to check
 * @param baseModel The base model name
 * @returns Whether the model matches the base model
 */
function modelMatches(modelName: string, baseModel: string): boolean {
    // Extract just the model part from paths like "publishers/anthropic/models/claude-3-5-sonnet-v2"
    const getModelPart = (name: string): string => {
        const segments = name.split('/');
        return segments[segments.length - 1]; // Get last segment
    };

    // Get the model part without path prefixes
    const modelPart = getModelPart(modelName);

    // Exact match with either full name or extracted model part
    if (modelName === baseModel || modelPart === baseModel) {
        return true;
    }

    // Check if it's a variant (e.g., "claude-3-5-sonnet-v2" matches "claude-3-5")
    if (modelPart.startsWith(baseModel + '-') || modelName.startsWith(baseModel + '-')) {
        return true;
    }

    // Check if the model part contains the base model as a substring
    // This handles cases like "claude-3-5-sonnet-v2" matching "claude-3-5"
    if (modelPart.includes(baseModel)) {
        // Make sure it's not a partial match (e.g. "claude-3" shouldn't match "claude-3-5")
        // Check if the character after the baseModel is either end of string, dash or other delimiter
        const baseModelEndIndex = modelPart.indexOf(baseModel) + baseModel.length;
        if (baseModelEndIndex === modelPart.length ||
            modelPart[baseModelEndIndex] === '-' ||
            modelPart[baseModelEndIndex] === '_') {
            return true;
        }
    }

    return false;
}

/**
 * Determines if a Vertex AI model supports tool use based on the model name/ID.
 * 
 * @param model The model name to check for tool use support
 * @param streaming Optional parameter to check if the model supports streaming tool use
 * @returns true if the model supports tool use, false if it doesn't, undefined if unknown
 */
export function supportsToolUseVertexAI(model: string, streaming: boolean = false): boolean | undefined {
    // Normalize the model string for easier matching
    const modelLower = model.toLowerCase();

    // Determine which lists to check against based on the streaming parameter
    const supportList = streaming ? MODELS_WITH_STREAMING_TOOL_USE : MODELS_WITH_TOOL_USE;
    const noSupportList = streaming ?
        MODELS_WITHOUT_STREAMING_TOOL_USE :
        MODELS_WITHOUT_TOOL_USE;

    // First check if the model is explicitly known not to support tool use
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