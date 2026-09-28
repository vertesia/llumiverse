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
    const providerCost = usage.is_byok !== true && typeof usage.cost === 'number' ? usage.cost : undefined;
    return {
        ...openAIPromptUsage(
            usage.prompt_tokens,
            usage.prompt_tokens_details?.cached_tokens,
            usage.prompt_tokens_details?.cache_write_tokens,
        ),
        result: usage.completion_tokens,
        total: usage.total_tokens,
        ...(providerCost !== undefined ? { provider_cost_usd: providerCost } : {}),
    };
}

/**
 * Prompt tokens split by cache use. OpenAI-compatible APIs count cache reads and writes in the prompt tokens, so
 * the new prompt tokens are what remains.
 */
export function openAIPromptUsage(
    promptTokens: number,
    cachedTokens: number | null | undefined,
    cacheWriteTokens: number | null | undefined,
): Pick<ExecutionTokenUsage, 'prompt' | 'prompt_new' | 'prompt_cached' | 'prompt_cache_write'> {
    return {
        prompt: promptTokens,
        prompt_cached: cachedTokens || undefined,
        prompt_cache_write: cacheWriteTokens || undefined,
        prompt_new: Math.max(0, promptTokens - (cachedTokens ?? 0) - (cacheWriteTokens ?? 0)),
    };
}

/** Duration-billed transcription has no token counts; do not invent them. */
export function mapOpenAITranscriptionUsage(
    usage: OpenAI.Audio.Transcription['usage'],
): ExecutionTokenUsage | undefined {
    if (usage?.type !== 'tokens') return undefined;
    return { prompt: usage.input_tokens, result: usage.output_tokens, total: usage.total_tokens };
}
