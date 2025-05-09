/**
 * Models that explicitly do NOT support tool use in Amazon Bedrock
 */
const MODELS_WITHOUT_TOOL_USE = [
    // AI21 models except Jamba 1.5
    "jamba-instruct",
    "jurassic-2",
    // Amazon Titan models
    "titan-text",
    "titan-text-express",
    "titan-text-lite",
    "titan-text-premier",
    // Anthropic Claude 2.x and earlier
    "claude-2",
    "claude-instant",
    "claude-1",
    // Cohere models except Command R
    "command",
    "command-light",
    // DeepSeek models
    "deepseek-r1",
    // Meta Llama 2
    "llama-2",
    // Meta Llama 3 models not supporting tool use
    "llama-3.2-1b",
    "llama-3.2-3b",
    // Mistral AI Instruct
    "mistral-instruct"
];

/**
 * Models that explicitly do NOT support streaming tool use in Amazon Bedrock
 * This includes all models without tool use plus models that only support non-streaming tools
 */
const MODELS_WITHOUT_STREAMING_TOOL_USE = [
    // Include all models that don't support tool use at all
    ...MODELS_WITHOUT_TOOL_USE,

    // Add models that support tool use but NOT streaming tool use
    // These are models in MODELS_WITH_TOOL_USE but not in MODELS_WITH_STREAMING_TOOL_USE
    "llama-3.1",
    "llama-3.2-11b",
    "llama-3.2-90b",
    "mistral-large",
    "mistral-small",
    "pixtral",
    "palmyra"
];

/**
 * Models that are known to support tool use in Amazon Bedrock
 */
const MODELS_WITH_TOOL_USE = [
    // AI21 Jamba models
    "jamba-1.5",
    // Amazon Nova models
    "nova",
    // Anthropic Claude 3 models
    "claude-3-5",
    "claude-3-7",
    // Cohere Command R models
    "command-r",
    // Meta Llama 3.1 and some 3.2 variants
    "llama-3.1",
    "llama-3.2-11b",
    "llama-3.2-90b",
    // Mistral models
    "mistral-large",
    "mistral-small",
    // Pixtral models
    "pixtral",
    // Writer Palmyra models
    "palmyra"
];

/**
 * Models that are known to support streaming tool use in Amazon Bedrock
 */
const MODELS_WITH_STREAMING_TOOL_USE = [
    // AI21 Jamba models
    "jamba-1.5",
    // Amazon Nova models
    "nova",
    // Anthropic Claude 3 models
    "claude-3-5",
    "claude-3-7",
    // Cohere Command R models
    "command-r"
];

/**
 * Determines if a Bedrock model supports tool use based on the model name/ID.
 * 
 * @param model The model name or ARN to check for tool use support
 * @param streaming Optional parameter to check if the model supports streaming tool use
 * @returns true if the model supports tool use, false if it doesn't, undefined if unknown
 */
export function supportsToolUseBedrock(model: string, streaming: boolean = false): boolean | undefined {
    // Normalize the model string for easier matching
    const modelLower = model.toLowerCase();

    // Determine which lists to check against based on the streaming parameter
    const supportList = streaming ? MODELS_WITH_STREAMING_TOOL_USE : MODELS_WITH_TOOL_USE;
    const noSupportList = streaming ?
        MODELS_WITHOUT_STREAMING_TOOL_USE :
        MODELS_WITHOUT_TOOL_USE;

    // First check if the model is explicitly known not to support tool use
    if (noSupportList.some(unsupportedModel => modelLower.includes(unsupportedModel))) {
        return false;
    }

    // Then check if the model is explicitly known to support tool use
    if (supportList.some(supportedModel => modelLower.includes(supportedModel))) {
        return true;
    }

    // If neither match, return undefined (unknown status)
    return undefined;
}