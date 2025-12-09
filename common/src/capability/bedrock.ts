import { ModelCapabilities, ModelModalities } from "../types.js";

// Record of Bedrock model capabilities keyed by model ID.
const RECORD_FOUNDATION_MODEL_CAPABILITIES: Record<string, ModelCapabilities> = {
    "ai21.jamba-1-5-large-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "ai21.jamba-1-5-mini-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "ai21.jamba-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "amazon.nova-canvas-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { image: true, text: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "amazon.nova-lite-v1:0": { input: { text: true, image: true, video: true, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "amazon.nova-micro-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "amazon.nova-pro-v1:0": { input: { text: true, image: true, video: true, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "amazon.titan-text-express-v1": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "amazon.titan-text-lite-v1": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "amazon.titan-text-premier-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "amazon.titan-tg1-large": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "anthropic.claude-3-5-haiku-20241022-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "anthropic.claude-3-5-sonnet-20240620-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "anthropic.claude-3-5-sonnet-20241022-v2:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "anthropic.claude-3-haiku-20240307-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "anthropic.claude-3-opus-20240229-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "anthropic.claude-3-sonnet-20240229-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "anthropic.claude-instant-v1": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "anthropic.claude-v2": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "anthropic.claude-v2:1": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "cohere.command-light-text-v14": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "cohere.command-r-plus-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "cohere.command-r-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "cohere.command-text-v14": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "meta.llama3-1-405b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "meta.llama3-1-70b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "meta.llama3-1-8b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "meta.llama3-70b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "meta.llama3-8b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "mistral.mixtral-8x7b-instruct-v0:1": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "mistral.mistral-7b-instruct-v0:2": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "mistral.mistral-large-2402-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "mistral.mistral-large-2407-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "mistral.mistral-small-2402-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "openai.gpt-oss-20b-1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "openai.gpt-oss-120b-1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
};

const RECORD_PROFILE_MODEL_CAPABILITIES: Record<string, ModelCapabilities> = {
    "amazon.nova-lite-v1:0": { input: { text: true, image: true, video: true, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "amazon.nova-micro-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "amazon.nova-premier-v1:0": { input: { text: true, image: true, video: true, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "amazon.nova-pro-v1:0": { input: { text: true, image: true, video: true, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "anthropic.claude-3-5-haiku-20241022-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "anthropic.claude-3-5-sonnet-20240620-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "anthropic.claude-3-5-sonnet-20241022-v2:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "anthropic.claude-3-7-sonnet-20250219-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "anthropic.claude-3-haiku-20240307-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "anthropic.claude-3-opus-20240229-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "anthropic.claude-3-sonnet-20240229-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "anthropic.claude-opus-4-20250514-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "anthropic.claude-sonnet-4-20250514-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "deepseek.r1-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "meta.llama3-1-70b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "meta.llama3-1-8b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "meta.llama3-2-1b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "meta.llama3-2-11b-instruct-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "meta.llama3-2-3b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "meta.llama3-2-90b-instruct-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "meta.llama3-3-70b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "meta.llama4-maverick-17b-instruct-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "meta.llama4-scout-17b-instruct-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "mistral.pixtral-large-2502-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "writer.palmyra-x4-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "writer.palmyra-x5-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
};

// Explicit exception lists keyed by the model identifier (last segment after the prefix)
const RECORD_FOUNDATION_EXCEPTIONS: Record<string, ModelCapabilities> = {};
const RECORD_PROFILE_EXCEPTIONS: Record<string, ModelCapabilities> = {
    "meta.llama3-1-70b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "meta.llama3-1-8b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
};

// Populate RECORD_FAMILY_CAPABILITIES as a const record (lowest common denominator for each family)
const RECORD_FAMILY_CAPABILITIES: Record<string, ModelCapabilities> = {
    "ai21.jamba": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "amazon.nova": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: false, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "amazon.titan": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "anthropic.claude": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "anthropic.claude-3": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "anthropic.claude-3-5": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "anthropic.claude-3-7": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "cohere.command": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "meta.llama3": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "mistral.mistral": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "mistral.mistral-large": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "mistral.mixtral": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "qwen.": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "openai.gpt-oss": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "anthropic.claude-3-haiku": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "anthropic.claude-3-5-sonnet": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "anthropic.claude-3-opus": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "anthropic.claude-3-sonnet": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "deepseek.r1": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "meta.llama4-maverick-17b": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "meta.llama4-scout-17b": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "mistral.pixtral": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "writer.palmyra": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "twelvelabs.": { input: { text: true, image: false, video: true, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
};

function extractModelLookupKey(modelName: string): string {
    const lower = modelName.toLowerCase();
    const lastSlashIdx = lower.lastIndexOf("/");
    let key = lastSlashIdx === -1 ? lower : lower.slice(lastSlashIdx + 1);
    if (lower.includes("inference-profile/")) {
        key = key.replace(/^[^.]+\./, "");
    }
    return key;
}

const FOUNDATION_MODEL_CAPABILITIES_BY_LOOKUP_KEY = RECORD_FOUNDATION_MODEL_CAPABILITIES;
const PROFILE_MODEL_CAPABILITIES_BY_LOOKUP_KEY = RECORD_PROFILE_MODEL_CAPABILITIES;
const MODEL_CAPABILITIES_BY_LOOKUP_KEY = {
    ...FOUNDATION_MODEL_CAPABILITIES_BY_LOOKUP_KEY,
    ...PROFILE_MODEL_CAPABILITIES_BY_LOOKUP_KEY,
};

const FOUNDATION_FAMILY_CAPABILITIES_BY_LOOKUP_KEY = RECORD_FAMILY_CAPABILITIES;
const PROFILE_FAMILY_CAPABILITIES_BY_LOOKUP_KEY = RECORD_FAMILY_CAPABILITIES;
const FAMILY_CAPABILITIES_BY_LOOKUP_KEY = RECORD_FAMILY_CAPABILITIES;

function findFamilyCapability(
    lookupKey: string,
    families: Record<string, ModelCapabilities>
): ModelCapabilities | undefined {
    let bestKey: string | undefined;
    for (const key of Object.keys(families)) {
        if (lookupKey.startsWith(key) && (!bestKey || key.length > bestKey.length)) {
            bestKey = key;
        }
    }
    return bestKey ? families[bestKey] : undefined;
}

/**
 * Extract the model identifier from an ARN or inference profile
 * @param modelName The full model ARN or name
 * @returns The normalized model identifier
 */
function normalizeModelName(modelName: string): string {
    const modelLower = modelName.toLowerCase();
    if (modelLower.includes("inference-profile")) {
        const parts = modelLower.split("/");
        if (parts.length > 1) {
            const providerModel = parts[parts.length - 1];
            const modelParts = providerModel.split(".");
            if (modelParts.length > 1 && modelParts[1] === "deepseek") {
                return `deepseek-${modelParts.slice(2).join(".")}`;
            }
            return modelParts.length > 2 ? modelParts.slice(2).join(".") : providerModel;
        }
    }
    return modelLower;
}

// Fallback pattern lists for inferring modalities and tool support
const IMAGE_INPUT_MODELS = ["image"]; // fallback: if model id contains 'image', supports image input
const VIDEO_INPUT_MODELS = ["video"];
const AUDIO_INPUT_MODELS = ["audio"];
const TEXT_INPUT_MODELS = ["text"];
const IMAGE_OUTPUT_MODELS = ["image"];
const VIDEO_OUTPUT_MODELS = ["video"];
const AUDIO_OUTPUT_MODELS = ["audio"];
const TEXT_OUTPUT_MODELS = ["text"];
const EMBEDDING_OUTPUT_MODELS = ["embed"];
const TOOL_SUPPORT_MODELS = ["tool", "sonnet", "opus", "nova", "palmyra", "command-r", "mistral-large", "pixtral"];

function modelMatches(modelName: string, patterns: string[]): boolean {
    return patterns.some(pattern => modelName.includes(pattern));
}

/**
 * Get the full ModelCapabilities for a Bedrock model.
 * Checks RECORD_MODEL_CAPABILITIES first, then falls back to pattern-based inference.
 */
export function getModelCapabilitiesBedrock(model: string): ModelCapabilities {
    // Normalize ARN or inference-profile to model ID
    const modelLower = model.toLowerCase();
    let normalized = modelLower;
    const arnPattern = /^arn:aws:bedrock:[^:]+:[^:]*:(inference-profile|foundation-model)\/.+/i;
    if (arnPattern.test(modelLower)) {
        // Extract after last occurrence of 'foundation-model/' or 'inference-profile/'
        const foundationIdx = modelLower.lastIndexOf('foundation-model/');
        const inferenceIdx = modelLower.lastIndexOf('inference-profile/');
        if (foundationIdx !== -1) {
            normalized = modelLower.substring(foundationIdx);
        } else if (inferenceIdx !== -1) {
            normalized = modelLower.substring(inferenceIdx);
        }
    }
    const isInferenceProfile = normalized.startsWith("inference-profile/");
    const isFoundationModel = normalized.startsWith("foundation-model/");
    const lookupKey = extractModelLookupKey(normalized);

    if (isFoundationModel) {
        const exception = RECORD_FOUNDATION_EXCEPTIONS[lookupKey];
        if (exception) return exception;
    } else if (isInferenceProfile) {
        const exception = RECORD_PROFILE_EXCEPTIONS[lookupKey];
        if (exception) return exception;
    }

    const capabilityLookups: Array<Record<string, ModelCapabilities>> = isFoundationModel
        ? [FOUNDATION_MODEL_CAPABILITIES_BY_LOOKUP_KEY]
        : isInferenceProfile
            ? [PROFILE_MODEL_CAPABILITIES_BY_LOOKUP_KEY]
            : [FOUNDATION_MODEL_CAPABILITIES_BY_LOOKUP_KEY, PROFILE_MODEL_CAPABILITIES_BY_LOOKUP_KEY];
    for (const lookup of capabilityLookups) {
        const capability = lookup[lookupKey];
        if (capability) return capability;
    }

    const fallbackCapability = MODEL_CAPABILITIES_BY_LOOKUP_KEY[lookupKey];
    if (fallbackCapability) return fallbackCapability;

    const familyLookups: Array<Record<string, ModelCapabilities>> = isFoundationModel
        ? [FOUNDATION_FAMILY_CAPABILITIES_BY_LOOKUP_KEY]
        : isInferenceProfile
            ? [PROFILE_FAMILY_CAPABILITIES_BY_LOOKUP_KEY]
            : [FOUNDATION_FAMILY_CAPABILITIES_BY_LOOKUP_KEY, PROFILE_FAMILY_CAPABILITIES_BY_LOOKUP_KEY];
    for (const familyLookup of familyLookups) {
        const familyCapability = findFamilyCapability(lookupKey, familyLookup);
        if (familyCapability) return familyCapability;
    }

    const fallbackFamilyCapability = findFamilyCapability(lookupKey, FAMILY_CAPABILITIES_BY_LOOKUP_KEY);
    if (fallbackFamilyCapability) return fallbackFamilyCapability;

    // 3. Fallback: infer from normalized name
    const inferredName = normalizeModelName(lookupKey);
    const input: ModelModalities = {
        text: modelMatches(inferredName, TEXT_INPUT_MODELS) || undefined,
        image: modelMatches(inferredName, IMAGE_INPUT_MODELS) || undefined,
        video: modelMatches(inferredName, VIDEO_INPUT_MODELS) || undefined,
        audio: modelMatches(inferredName, AUDIO_INPUT_MODELS) || undefined,
        embed: false
    };
    const output: ModelModalities = {
        text: modelMatches(inferredName, TEXT_OUTPUT_MODELS) || undefined,
        image: modelMatches(inferredName, IMAGE_OUTPUT_MODELS) || undefined,
        video: modelMatches(inferredName, VIDEO_OUTPUT_MODELS) || undefined,
        audio: modelMatches(inferredName, AUDIO_OUTPUT_MODELS) || undefined,
        embed: modelMatches(inferredName, EMBEDDING_OUTPUT_MODELS) || undefined
    };
    const tool_support = modelMatches(inferredName, TOOL_SUPPORT_MODELS) || undefined;
    return { input, output, tool_support };
}