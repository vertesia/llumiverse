import type { ExecutionTokenUsage } from '@llumiverse/core';
import type OpenAI from 'openai';

export function mapOpenAIChatCompletionsUsage(usage?: OpenAI.CompletionUsage | null): ExecutionTokenUsage | undefined {
    if (!usage) {
        return undefined;
    }
    const cachedTokens = usage.prompt_tokens_details?.cached_tokens ?? 0;
    return {
        prompt: usage.prompt_tokens,
        result: usage.completion_tokens,
        total: usage.total_tokens,
        prompt_cached: cachedTokens || undefined,
        prompt_new: Math.max(0, usage.prompt_tokens - cachedTokens),
    };
}

/** Duration-billed transcription has no token counts; do not invent them. */
export function mapOpenAITranscriptionUsage(
    usage: OpenAI.Audio.Transcription['usage'],
): ExecutionTokenUsage | undefined {
    if (usage?.type !== 'tokens') return undefined;
    return { prompt: usage.input_tokens, result: usage.output_tokens, total: usage.total_tokens };
}
