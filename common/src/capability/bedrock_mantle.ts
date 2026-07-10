import { getBedrockMantleModelFamily } from '../options/bedrock_mantle.js';
import type { ModelCapabilities } from '../types.js';

export function getModelCapabilitiesBedrockMantle(model: string): ModelCapabilities {
    if (!getBedrockMantleModelFamily(model)) {
        return { input: {}, output: {} };
    }
    return {
        input: { text: true, image: true, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    };
}
