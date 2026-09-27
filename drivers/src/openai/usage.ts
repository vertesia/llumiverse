import type { ExecutionTokenUsage } from '@llumiverse/core';
import type OpenAI from 'openai';

/**
 * Chat Completions usage plus the fields OpenAI-compatible gateways add: OpenRouter reports cache writes
 * under `prompt_tokens_details` and the amount it charged as `cost` (not charged on bring-your-own-key).
 */
export type ChatCompletionsUsage = Omit<OpenAI.CompletionUsage, 'prompt_tokens_details'> & {
    prompt_tokens_details?:
        | (OpenAI.CompletionUsage.PromptTokensDetails & { cache_write_tokens?: number | null })
        | null;
    cost?: number | null;
    is_byok?: boolean | null;
};

export function mapOpenAIChatCompletionsUsage(usage?: ChatCompletionsUsage | null): ExecutionTokenUsage | undefined {
    if (!usage) {
        return undefined;
    }
    const cachedTokens = usage.prompt_tokens_details?.cached_tokens ?? 0;
    const cacheWriteTokens = usage.prompt_tokens_details?.cache_write_tokens ?? 0;
    const providerCost = usage.is_byok !== true && typeof usage.cost === 'number' ? usage.cost : undefined;
    return {
        prompt: usage.prompt_tokens,
        result: usage.completion_tokens,
        total: usage.total_tokens,
        prompt_cached: cachedTokens || undefined,
        prompt_cache_write: cacheWriteTokens || undefined,
        prompt_new: Math.max(0, usage.prompt_tokens - cachedTokens - cacheWriteTokens),
        ...(providerCost !== undefined ? { provider_cost_usd: providerCost } : {}),
    };
}

/** Duration-billed transcription has no token counts; do not invent them. */
export function mapOpenAITranscriptionUsage(
    usage: OpenAI.Audio.Transcription['usage'],
): ExecutionTokenUsage | undefined {
    if (usage?.type !== 'tokens') return undefined;
    return { prompt: usage.input_tokens, result: usage.output_tokens, total: usage.total_tokens };
}
