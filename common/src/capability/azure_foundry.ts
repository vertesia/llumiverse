import { ModelModalities, ModelCapabilities } from "../types.js";

// Global feature flags - temporarily disable tool support for non-OpenAI models
const ENABLE_TOOL_SUPPORT_NON_OPENAI = false;

// Record of Azure Foundry model capabilities keyed by model ID (lowercased) 
// Only include models with specific exceptions that differ from their family patterns
const RECORD_MODEL_CAPABILITIES: Record<string, ModelCapabilities> = {
    // OpenAI O-series exceptions - o1-mini doesn't have tool support like other o1 models
    "o1-mini": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    // OpenAI o3 is text-only unlike other o-series models
    "o3": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },

    // Models with special properties not covered by family patterns
    "deepseek-r1-0528": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "deepseek-v3-0324": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "mistral-medium-2505": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "mistral-nemo": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "llama-4-scout-17b-16e-instruct": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
};

// Populate RECORD_FAMILY_CAPABILITIES as a const record (lowest common denominator for each family)
const RECORD_FAMILY_CAPABILITIES: Record<string, ModelCapabilities> = {
    // OpenAI GPT families
    "gpt-3.5-turbo": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "gpt-35-turbo": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "gpt-35": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "gpt-4": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "gpt-4.1": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "gpt-4.5": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "gpt-4o": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },

    // OpenAI O-series families
    "o1": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "o1-preview": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "o1-pro": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "o3-mini": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "o4-mini": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "o4": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },

    // Llama families
    "llama-3.1": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "llama-3.2": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "llama-3.3": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "llama-3": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "llama-4": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "llama": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },

    // Mistral families
    "mistral-large": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "mistral-small": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "mistral": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },

    // Microsoft Phi families
    "phi-4": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "phi": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },

    // DeepSeek families
    "deepseek-r1": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "deepseek-v3": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "deepseek": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },

    // AI21 families
    "ai21-jamba": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "ai21": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "jamba": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },

    // Cohere families
    "cohere-command": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "cohere": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "command": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },

    // xAI families
    "grok-3": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "grok": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true }
};

// Fallback pattern lists for inferring modalities and tool support
const IMAGE_INPUT_MODELS = ["image", "vision"];
const VIDEO_INPUT_MODELS = ["video"];
const AUDIO_INPUT_MODELS = ["audio"];
const TEXT_INPUT_MODELS = ["text"];
const IMAGE_OUTPUT_MODELS = ["image"];
const VIDEO_OUTPUT_MODELS = ["video"];
const AUDIO_OUTPUT_MODELS = ["audio"];
const TEXT_OUTPUT_MODELS = ["text"];
const EMBEDDING_OUTPUT_MODELS = ["embed"];
const TOOL_SUPPORT_MODELS = ["tool", "gpt-4", "gpt-4o", "o1", "o3", "o4", "llama-3", "mistral-large", "mistral-small", "jamba", "cohere", "command", "grok"];

function modelMatches(modelName: string, patterns: string[]): boolean {
    return patterns.some(pattern => modelName.includes(pattern));
}

/**
 * Get the full ModelCapabilities for an Azure Foundry model.
 * Checks RECORD_MODEL_CAPABILITIES first, then falls back to family pattern matching.
 */
export function getModelCapabilitiesAzureFoundry(model: string): ModelCapabilities {
    // Extract base model from composite ID (deployment::baseModel)
    const { baseModel } = parseAzureFoundryModelId(model);
    const normalized = baseModel.toLowerCase();

    // 1. Exact match in record
    const record = RECORD_MODEL_CAPABILITIES[normalized];
    if (record) {
        return applyGlobalToolSupportDisable(record, normalized);
    }

    // 2. Fallback: find the longest matching family prefix in RECORD_FAMILY_CAPABILITIES
    let bestFamilyKey = undefined;
    let bestFamilyLength = 0;
    for (const key of Object.keys(RECORD_FAMILY_CAPABILITIES)) {
        if (normalized.startsWith(key) && key.length > bestFamilyLength) {
            bestFamilyKey = key;
            bestFamilyLength = key.length;
        }
    }
    if (bestFamilyKey) {
        return applyGlobalToolSupportDisable(RECORD_FAMILY_CAPABILITIES[bestFamilyKey], normalized);
    }

    // 3. Fallback: infer from normalized name using patterns
    const input: ModelModalities = {
        text: modelMatches(normalized, TEXT_INPUT_MODELS) || true, // Default to text input
        image: modelMatches(normalized, IMAGE_INPUT_MODELS) || undefined,
        video: modelMatches(normalized, VIDEO_INPUT_MODELS) || undefined,
        audio: modelMatches(normalized, AUDIO_INPUT_MODELS) || undefined,
        embed: false
    };
    const output: ModelModalities = {
        text: modelMatches(normalized, TEXT_OUTPUT_MODELS) || true, // Default to text output
        image: modelMatches(normalized, IMAGE_OUTPUT_MODELS) || undefined,
        video: modelMatches(normalized, VIDEO_OUTPUT_MODELS) || undefined,
        audio: modelMatches(normalized, AUDIO_OUTPUT_MODELS) || undefined,
        embed: modelMatches(normalized, EMBEDDING_OUTPUT_MODELS) || undefined
    };
    const tool_support = modelMatches(normalized, TOOL_SUPPORT_MODELS) || undefined;
    const tool_support_streaming = tool_support || undefined;

    const inferredCapabilities = { input, output, tool_support, tool_support_streaming };
    return applyGlobalToolSupportDisable(inferredCapabilities, normalized);
}

/**
 * Apply global tool support disable for non-OpenAI models.
 * Preserves model-specific information for future use while temporarily disabling tool support.
 */
function applyGlobalToolSupportDisable(capabilities: ModelCapabilities, modelName: string): ModelCapabilities {
    // Check if this is an OpenAI model
    const isOpenAIModel = modelName.startsWith('gpt-') || modelName.startsWith('o1') || modelName.startsWith('o3') || modelName.startsWith('o4');

    if (!ENABLE_TOOL_SUPPORT_NON_OPENAI && !isOpenAIModel) {
        // Disable tool support for non-OpenAI models while preserving other capabilities
        return {
            ...capabilities,
            tool_support: false,
            tool_support_streaming: false
        };
    }

    return capabilities;
}

// Helper function to parse composite model IDs
function parseAzureFoundryModelId(compositeId: string): { deploymentName: string; baseModel: string } {
    const parts = compositeId.split('::');
    if (parts.length === 2) {
        return {
            deploymentName: parts[0],
            baseModel: parts[1]
        };
    }

    // Backwards compatibility: if no delimiter found, treat as deployment name
    return {
        deploymentName: compositeId,
        baseModel: compositeId
    };
}