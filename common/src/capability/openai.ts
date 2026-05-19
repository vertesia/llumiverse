import { ModelModalities } from "../types.js";

// OpenAI model capability type - tool_support_streaming defaults to tool_support for OpenAI models
type OpenAIModelCapability = { input: ModelModalities; output: ModelModalities; tool_support?: boolean; tool_support_streaming?: boolean };

// Record of OpenAI model capabilities keyed by model ID (lowercased)
const RECORD_MODEL_CAPABILITIES: Record<string, OpenAIModelCapability> = {
    "chatgpt-4o-latest": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true }
};

// Populate RECORD_FAMILY_CAPABILITIES as a const record (lowest common denominator for each family)
// For OpenAI, tool_support_streaming matches tool_support since the API supports tools while streaming
const RECORD_FAMILY_CAPABILITIES: Record<string, OpenAIModelCapability> = {
    "gpt": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gpt-3.5": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false },
    "gpt-4": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gpt-4-turbo": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gpt-4o": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gpt-4.1": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gpt-4.5": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gpt-5": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gpt-image": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: false, image: true, video: false, audio: false, embed: false }, tool_support: false },
    "chatgpt-image": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: false, image: true, video: false, audio: false, embed: false }, tool_support: false },
    "dall-e": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: false, image: true, video: false, audio: false, embed: false }, tool_support: false },
    "gpt-oss": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false },
    "o": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "o1-mini": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false },
    "o1-preview": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false },
    "omni-moderation": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false },
    "sora": { input: { text: true, image: true, video: true, audio: false, embed: false }, output: { text: false, image: false, video: true, audio: true, embed: false }, tool_support: false },
    "text-embedding": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: false, image: false, video: false, audio: false, embed: true }, tool_support: false },
    "text-moderation": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false },
    "whisper": { input: { text: false, image: false, video: false, audio: true, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false }
};

// Fallback pattern lists for inferring modalities and tool support
const IMAGE_INPUT_MODELS = ["image"];
const VIDEO_INPUT_MODELS = ["video"];
const AUDIO_INPUT_MODELS = ["audio"];
const TEXT_INPUT_MODELS = ["text"];
const IMAGE_OUTPUT_MODELS = ["image"];
const VIDEO_OUTPUT_MODELS = ["video"];
const AUDIO_OUTPUT_MODELS = ["audio"];
const TEXT_OUTPUT_MODELS = ["text"];
const EMBEDDING_OUTPUT_MODELS = ["embed"];
const TOOL_SUPPORT_MODELS = ["tool", "gpt", "gpt-5", "o1", "o3", "o4"];

function modelMatches(modelName: string, patterns: string[]): boolean {
    return patterns.some(pattern => modelName.includes(pattern));
}

function normalizeOpenAIModelName(modelName: string): string {
    return modelName.toLowerCase();
}

/**
 * Get the full ModelCapabilities for an OpenAI model.
 * Checks RECORD_MODEL_CAPABILITIES first, then falls back to pattern-based inference.
 * For OpenAI models, tool_support_streaming defaults to tool_support since the API supports tools while streaming.
 */
export function getModelCapabilitiesOpenAI(model: string): { input: ModelModalities; output: ModelModalities; tool_support?: boolean; tool_support_streaming?: boolean } {
    const normalized = normalizeOpenAIModelName(model);
    const record = RECORD_MODEL_CAPABILITIES[normalized];
    if (record) {
        // Default tool_support_streaming to tool_support for OpenAI models
        return {
            ...record,
            tool_support_streaming: record.tool_support_streaming ?? record.tool_support
        };
    }
    let bestFamilyKey = undefined;
    let bestFamilyLength = 0;
    for (const key of Object.keys(RECORD_FAMILY_CAPABILITIES)) {
        if (normalized.startsWith(key) && key.length > bestFamilyLength) {
            bestFamilyKey = key;
            bestFamilyLength = key.length;
        }
    }
    if (bestFamilyKey) {
        const family = RECORD_FAMILY_CAPABILITIES[bestFamilyKey];
        // Default tool_support_streaming to tool_support for OpenAI models
        return {
            ...family,
            tool_support_streaming: family.tool_support_streaming ?? family.tool_support
        };
    }
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
    // Default tool_support_streaming to tool_support for OpenAI models
    return { input, output, tool_support, tool_support_streaming: tool_support };
}