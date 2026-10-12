import { isModelFamilyVersionGTE, isSingleDigitGrokVersionGte } from '../options/version-parsing.js';

interface VertexAIModelLimits {
    context_window?: number;
    max_output_tokens?: number;
}

// Exact records preserve documented model limits. Newer generations inherit known Vertex
// limits below, rather than assuming source-provider limits apply to the hosted version.
const MODEL_LIMITS: Readonly<Record<string, VertexAIModelLimits>> = {
    'glm-5.2-maas': { context_window: 1_000_000, max_output_tokens: 64_000 },
    'deepseek-v3.1-maas': { context_window: 163_840, max_output_tokens: 32_768 },
    'grok-4.7': { context_window: 524_288 },
    'grok-4.6': { context_window: 524_288 },
    'grok-4.3': { context_window: 200_000 },
    'grok-4.20-reasoning': { context_window: 2_000_000 },
    'grok-4.20-non-reasoning': { context_window: 2_000_000 },
    'grok-4.1-fast-reasoning': { context_window: 128_000 },
    'grok-4.1-fast-non-reasoning': { context_window: 128_000 },
    'mistral-small-2503': { context_window: 128_000 },
    'mistral-medium-3': { context_window: 128_000 },
    'codestral-2': { context_window: 128_000 },
};

export function getVertexAIModelLimits(model: string): VertexAIModelLimits {
    const modelName = model.toLowerCase().split('/').pop() ?? '';
    const exact = MODEL_LIMITS[modelName];
    if (exact) return exact;
    if (modelName.endsWith('-maas') && isModelFamilyVersionGTE(modelName, 'glm-', 5, 2)) {
        return MODEL_LIMITS['glm-5.2-maas'];
    }
    if (isSingleDigitGrokVersionGte(modelName, 4, 6)) return MODEL_LIMITS['grok-4.7'];
    if (isSingleDigitGrokVersionGte(modelName, 4, 3)) return MODEL_LIMITS['grok-4.3'];
    if (isModelFamilyVersionGTE(modelName, 'mistral-small-', 2503, 0)) return MODEL_LIMITS['mistral-small-2503'];
    if (isModelFamilyVersionGTE(modelName, 'mistral-medium-', 3, 0)) return MODEL_LIMITS['mistral-medium-3'];
    if (isModelFamilyVersionGTE(modelName, 'codestral-', 2, 0)) return MODEL_LIMITS['codestral-2'];
    return {};
}
