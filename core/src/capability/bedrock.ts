import { ModelModalities, ModelCapabilities } from "@llumiverse/common";

// Record of Bedrock model capabilities keyed by model ID.
const RECORD_MODEL_CAPABILITIES: Record<string, ModelCapabilities> = {
    "foundation-model/ai21.jamba-1-5-large-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "foundation-model/ai21.jamba-1-5-mini-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "foundation-model/ai21.jamba-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/amazon.nova-canvas-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { image: true, text: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/amazon.nova-lite-v1:0": { input: { text: true, image: true, video: true, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "foundation-model/amazon.nova-micro-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "foundation-model/amazon.nova-pro-v1:0": { input: { text: true, image: true, video: true, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "foundation-model/amazon.titan-text-express-v1": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/amazon.titan-text-lite-v1": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/amazon.titan-text-premier-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/amazon.titan-tg1-large": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/anthropic.claude-3-5-haiku-20241022-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "foundation-model/anthropic.claude-3-haiku-20240307-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "foundation-model/anthropic.claude-3-opus-20240229-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "foundation-model/anthropic.claude-3-sonnet-20240229-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "foundation-model/anthropic.claude-instant-v1": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/anthropic.claude-v2": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/anthropic.claude-v2:1": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/cohere.command-light-text-v14": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/cohere.command-r-plus-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "foundation-model/cohere.command-r-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "foundation-model/cohere.command-text-v14": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/meta.llama3-1-405b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "foundation-model/meta.llama3-1-70b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "foundation-model/meta.llama3-1-8b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "foundation-model/meta.llama3-70b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/meta.llama3-8b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/mistral.mixtral-8x7b-instruct-v0:1": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/mistral.mistral-7b-instruct-v0:2": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/mistral.mistral-large-2402-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "foundation-model/mistral.mistral-large-2407-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "foundation-model/mistral.mistral-small-2402-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.amazon.nova-lite-v1:0": { input: { text: true, image: true, video: true, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "inference-profile/us.amazon.nova-micro-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "inference-profile/us.amazon.nova-premier-v1:0": { input: { text: true, image: true, video: true, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "inference-profile/us.amazon.nova-pro-v1:0": { input: { text: true, image: true, video: true, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "inference-profile/us.anthropic.claude-3-haiku-20240307-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.anthropic.claude-3-opus-20240229-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.anthropic.claude-3-sonnet-20240229-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.anthropic.claude-opus-4-20250514-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.deepseek.r1-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "inference-profile/us.meta.llama3-1-70b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "inference-profile/us.meta.llama3-1-8b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "inference-profile/us.meta.llama3-2-1b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "inference-profile/us.meta.llama3-2-11b-instruct-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.meta.llama3-2-3b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "inference-profile/us.meta.llama3-2-90b-instruct-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.meta.llama3-3-70b-instruct-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "inference-profile/us.meta.llama4-maverick-17b-instruct-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "inference-profile/us.meta.llama4-scout-17b-instruct-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "inference-profile/us.mistral.pixtral-large-2502-v1:0": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.writer.palmyra-x4-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.writer.palmyra-x5-v1:0": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false }
};

// Populate RECORD_FAMILY_CAPABILITIES as a const record (lowest common denominator for each family)
const RECORD_FAMILY_CAPABILITIES: Record<string, ModelCapabilities> = {
    "foundation-model/ai21.jamba": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "foundation-model/amazon.nova": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: false, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "foundation-model/amazon.titan": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/anthropic.claude": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/anthropic.claude-3": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "foundation-model/anthropic.claude-3-5": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "foundation-model/anthropic.claude-3-7": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "foundation-model/cohere.command": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/meta.llama3": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "foundation-model/mistral.mistral": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "foundation-model/mistral.mistral-large": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "foundation-model/mistral.mixtral": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "inference-profile/us.anthropic.claude-3-haiku": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.anthropic.claude-3-5-sonnet": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "inference-profile/us.anthropic.claude-3-opus": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.anthropic.claude-3-sonnet": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.anthropic.claude": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.deepseek.r1": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "inference-profile/us.meta.llama3": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "inference-profile/us.meta.llama4-maverick-17b": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "inference-profile/us.meta.llama4-scout-17b": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false, tool_support_streaming: false },
    "inference-profile/us.mistral.pixtral": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false },
    "inference-profile/us.writer.palmyra": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: false }
};

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
    let normalized = model;
    const arnPattern = /^arn:aws:bedrock:[^:]+:[^:]*:(inference-profile|foundation-model)\/.+/i;
    if (arnPattern.test(model)) {
        // Extract after last occurrence of 'foundation-model/' or 'inference-profile/'
        const foundationIdx = model.lastIndexOf('foundation-model/');
        const inferenceIdx = model.lastIndexOf('inference-profile/');
        if (foundationIdx !== -1) {
            normalized = model.substring(foundationIdx);
        } else if (inferenceIdx !== -1) {
            normalized = model.substring(inferenceIdx);
        }
    }
    // Standardize region for inference-profile to 'us' for record lookup
    // This allows support for different AWS regions by mapping all to 'us'
    if (normalized.startsWith("inference-profile/")) {
        normalized = normalized.replace(/^inference-profile\/[^.]+\./, "inference-profile/us.");
    }

    // 1. Exact match in record
    const record = RECORD_MODEL_CAPABILITIES[normalized];
    if (record) return record;

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
        return RECORD_FAMILY_CAPABILITIES[bestFamilyKey];
    }

    // 3. Fallback: infer from normalized name
    normalized = normalizeModelName(normalized);
    const input: ModelModalities = {
        text: modelMatches(normalized, TEXT_INPUT_MODELS) || undefined,
        image: modelMatches(normalized, IMAGE_INPUT_MODELS) || undefined,
        video: modelMatches(normalized, VIDEO_INPUT_MODELS) || undefined,
        audio: modelMatches(normalized, AUDIO_INPUT_MODELS) || undefined,
        embed: false
    };
    const output: ModelModalities = {
        text: modelMatches(normalized, TEXT_OUTPUT_MODELS) || undefined,
        image: modelMatches(normalized, IMAGE_OUTPUT_MODELS) || undefined,
        video: modelMatches(normalized, VIDEO_OUTPUT_MODELS) || undefined,
        audio: modelMatches(normalized, AUDIO_OUTPUT_MODELS) || undefined,
        embed: modelMatches(normalized, EMBEDDING_OUTPUT_MODELS) || undefined
    };
    const tool_support = modelMatches(normalized, TOOL_SUPPORT_MODELS) || undefined;
    return { input, output, tool_support };
}