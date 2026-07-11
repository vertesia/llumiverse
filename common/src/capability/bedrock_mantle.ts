import { getBedrockMantleModelInfo } from '../options/bedrock_mantle.js';
import type { ModelCapabilities } from '../types.js';

export function getModelCapabilitiesBedrockMantle(model: string): ModelCapabilities {
    const info = getBedrockMantleModelInfo(model);
    if (!info) {
        return { input: {}, output: {} };
    }
    return {
        input: { text: true, image: info.input_image, video: false, audio: false, embed: false },
        output: { text: true, image: false, video: false, audio: false, embed: false },
        tool_support: true,
        tool_support_streaming: true,
    };
}
