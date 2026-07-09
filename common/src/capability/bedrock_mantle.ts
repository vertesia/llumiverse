import type { ModelCapabilities } from '../types.js';

const BEDROCK_MANTLE_MODEL_CAPABILITIES: Record<string, ModelCapabilities> = {
    'openai.gpt-5.5': {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },
    'openai.gpt-5.4': {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },
    'xai.grok-4.3': {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    },
};

export function getModelCapabilitiesBedrockMantle(model: string): ModelCapabilities {
    return BEDROCK_MANTLE_MODEL_CAPABILITIES[model.toLowerCase()] ?? { input: {}, output: {} };
}
