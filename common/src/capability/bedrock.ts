import { getBedrockModelCapabilities } from './bedrock-models.js';

export function getModelCapabilitiesBedrock(model: string) {
    return getBedrockModelCapabilities(model, 'runtime');
}
