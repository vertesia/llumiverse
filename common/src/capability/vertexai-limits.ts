interface VertexAIModelLimits {
    context_window?: number;
    max_output_tokens?: number;
}

// Vertex limits can differ from the source provider's API. Keep overrides exact so new
// versions retain the canonical family fallback until their Vertex limits are documented.
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
    return MODEL_LIMITS[modelName] ?? {};
}
