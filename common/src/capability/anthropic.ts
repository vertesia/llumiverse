import type { ModelCapabilities } from "../types.js";

// Explicit model exceptions
const RECORD_MODEL_CAPABILITIES: Record<string, ModelCapabilities> = {
    // claude-3-5-haiku: no image input
    "claude-3-5-haiku-20241022": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
};

// Family-level capabilities (longest prefix match)
const RECORD_FAMILY_CAPABILITIES: Record<string, ModelCapabilities> = {
    "claude-3-5-haiku": { input: { text: true, image: false, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "claude-3-haiku": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "claude-3-sonnet": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "claude-3-opus": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "claude-3-5": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "claude-3-7": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "claude-3": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "claude-4": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
    "claude-": { input: { text: true, image: true, video: false, audio: false, embed: false }, output: { text: true, image: false, video: false, audio: false, embed: false }, tool_support: true, tool_support_streaming: true },
};

const DEFAULT_CLAUDE_CAPABILITIES: ModelCapabilities = {
    input: { text: true, image: true, video: false, audio: false, embed: false },
    output: { text: true, image: false, video: false, audio: false, embed: false },
    tool_support: true,
    tool_support_streaming: true,
};

export function getModelCapabilitiesAnthropic(model: string): ModelCapabilities {
    const lower = model.toLowerCase();

    // Exact match first
    if (lower in RECORD_MODEL_CAPABILITIES) {
        return RECORD_MODEL_CAPABILITIES[lower];
    }

    // Longest prefix family match
    let bestKey: string | undefined;
    for (const key of Object.keys(RECORD_FAMILY_CAPABILITIES)) {
        if (lower.startsWith(key) && (!bestKey || key.length > bestKey.length)) {
            bestKey = key;
        }
    }
    if (bestKey) {
        return RECORD_FAMILY_CAPABILITIES[bestKey];
    }

    return DEFAULT_CLAUDE_CAPABILITIES;
}
