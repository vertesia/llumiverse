import { ModelModalities } from "../types.js";

// Record of Vertex AI model capabilities keyed by model ID (last path segment, lowercased)
const RECORD_MODEL_CAPABILITIES: Record<string, { input: ModelModalities; output: ModelModalities; tool_support?: boolean }> = {
    "gemini-1.5-flash-002": { input: { text: true, image: true, video: true, audio: true, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gemini-1.5-pro-002": { input: { text: true, image: true, video: true, audio: true, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gemini-2.0-flash-001": { input: { text: true, image: true, video: true, audio: true, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gemini-2.0-flash-lite-001": { input: { text: true, image: true, video: true, audio: true, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false },
    "gemini-2.5-flash-preview-04-17": { input: { text: true, image: true, video: true, audio: true, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gemini-2.5-pro-preview-05-06": { input: { text: true, image: true, video: true, audio: true, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "imagen-3.0-generate-002": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: false, image: true, video: false, audio: false, embed: false }, tool_support: false },
    "imagen-3.0-capability-001": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: false, image: true, video: false, audio: false, embed: false }, tool_support: false },
    "imagen-4.0-generate-preview-05-20": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: false, image: true, video: false, audio: false, embed: false }, tool_support: false },
    "claude-3-opus": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "claude-3-haiku": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "claude-3-5-sonnet": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "claude-3-5-haiku": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "claude-3-5-sonnet-v2": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "claude-3-7-sonnet": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "claude-opus-4": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "claude-sonnet-4": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
};

// Populate RECORD_FAMILY_CAPABILITIES as a const record (lowest common denominator for each family)
const RECORD_FAMILY_CAPABILITIES: Record<string, { input: ModelModalities; output: ModelModalities; tool_support?: boolean }> = {
    "gemini": { input: { text: true, image: true, video: true, audio: true, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gemini-1.5": { input: { text: true, image: true, video: true, audio: true, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gemini-2.0": { input: { text: true, image: true, video: true, audio: true, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gemini-2.5": { input: { text: true, image: true, video: true, audio: true, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gemini-3.0": { input: { text: true, image: true, video: true, audio: true, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "gemini-2.5-flash-image": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: true, video: false, audio: false, embed: false }, tool_support: false },
    "gemini-3.0-pro-image": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: true, video: false, audio: false, embed: false }, tool_support: false },
    "imagen-3.0-generate": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: false, image: true, video: false, audio: false, embed: false }, tool_support: false },
    "imagen-3.0-fast-generate": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: false, image: true, video: false, audio: false, embed: false }, tool_support: false },
    "imagen-3.0-capability": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: false, image: true, video: false, audio: false, embed: false }, tool_support: false },
    "imagen-4.0-generate": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: false, image: true, video: false, audio: false, embed: false }, tool_support: false },
    "imagen-4.0-ultra-generate": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: false, image: true, video: false, audio: false, embed: false }, tool_support: false },
    "imagen-4.0-fast-generate": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: false, image: true, video: false, audio: false, embed: false }, tool_support: false },
    "imagen-4.0-capability": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: false, image: true, video: false, audio: false, embed: false }, tool_support: false },
    "claude-3-5": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "claude-3-7": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "claude-3": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "claude": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true },
    "llama": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: false },
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
const TOOL_SUPPORT_MODELS = ["tool", "sonnet", "opus", "gemini", "claude-3-5", "claude-3-7"];

function modelMatches(modelName: string, patterns: string[]): boolean {
    return patterns.some(pattern => modelName.includes(pattern));
}

function normalizeVertexAIModelName(modelName: string): string {
    const segments = modelName.toLowerCase().split("/");
    return segments[segments.length - 1];
}

/**
 * Get the full ModelCapabilities for a Vertex AI model.
 * Checks RECORD_MODEL_CAPABILITIES first, then falls back to pattern-based inference.
 */
export function getModelCapabilitiesVertexAI(model: string): { input: ModelModalities; output: ModelModalities; tool_support?: boolean } {
    const normalized = normalizeVertexAIModelName(model);
    const record = RECORD_MODEL_CAPABILITIES[normalized];
    if (record) return record;
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