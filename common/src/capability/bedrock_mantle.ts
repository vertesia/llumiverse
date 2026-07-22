import { getBedrockMantleModelInfo } from '../options/bedrock_mantle.js';
import type { ModelCapabilities } from '../types.js';
import { getBedrockModelCapabilities } from './bedrock-models.js';

export function getModelCapabilitiesBedrockMantle(model: string): ModelCapabilities {
    const info = getBedrockMantleModelInfo(model);
    if (!info) {
        return { input: {}, output: {} };
    }
    return getBedrockModelCapabilities(model, 'mantle');
}
