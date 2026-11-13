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

/** Image token counts as the Images API and the Responses `tool_usage.image_gen` report them. */
interface OpenAIImageTokenUsage {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    cached_tokens?: number;
    output_tokens_details?: { image_tokens: number } | null;
}

/** Image output is mostly image tokens; text output tokens are counted apart when the details are reported. */
function imageResultTokens(usage: OpenAIImageTokenUsage): number {
    return usage.output_tokens_details?.image_tokens ?? usage.output_tokens;
}

/**
 * Images API usage. An SVG output also reports the tokens of the model that authored it (`svg_usage`), billed apart
 * from the image tokens, so it goes to `tool_usage.svg` rather than into the main counts.
 */
export function mapOpenAIImageUsage(
    usage: OpenAIImageTokenUsage | null | undefined,
    svgUsage?: OpenAI.Images.ImagesResponse.SvgUsage | null,
): ExecutionTokenUsage | undefined {
    if (!usage && !svgUsage) return undefined;
    return {
        ...(usage
            ? {
                  prompt: usage.input_tokens,
                  prompt_new: usage.input_tokens,
                  result: usage.output_tokens,
                  result_image: imageResultTokens(usage),
                  total: usage.total_tokens,
              }
            : {}),
        ...(svgUsage
            ? {
                  tool_usage: {
                      svg: {
                          model: svgUsage.model,
                          prompt: svgUsage.input_tokens,
                          result: svgUsage.output_tokens,
                          total: svgUsage.total_tokens,
                      },
                  },
              }
            : {}),
    };
}

/**
 * Responses usage, with the cache writes split by retention and the hosted tools' usage, which OpenAI reports apart
 * from the model's (`response.tool_usage`). Tools that report zero use are left out.
 */
export function mapOpenAIResponsesUsage(
    usage: ResponsesUsage | null | undefined,
    toolUsage?: OpenAI.Responses.Response.ToolUsage | null,
): ExecutionTokenUsage | undefined {
    if (!usage) return undefined;
    const cachedTokens =
        usage.input_tokens_details?.cached_tokens ?? usage.prompt_tokens_details?.cached_tokens ?? usage.cached_tokens;
    const cacheWriteTokens =
        usage.input_tokens_details?.cache_write_tokens ??
        usage.prompt_tokens_details?.cache_write_tokens ??
        usage.cache_write_tokens;
    const cacheWrite12h = usage.input_tokens_details?.cache_write_tokens_details?.ttl_12h_tokens;
    const tools = mapOpenAIToolUsage(toolUsage);
    return {
        ...openAIPromptUsage(usage.input_tokens, cachedTokens, cacheWriteTokens),
        ...(cacheWrite12h ? { prompt_cache_write_12h: cacheWrite12h } : {}),
        result: usage.output_tokens,
        total: usage.total_tokens,
        ...(tools ? { tool_usage: tools } : {}),
    };
}

/** Responses usage plus the fields some OpenAI-compatible Responses endpoints report instead. */
export type ResponsesUsage = OpenAI.Responses.ResponseUsage & {
    cached_tokens?: number | null;
    cache_write_tokens?: number | null;
    prompt_tokens_details?: {
        cached_tokens?: number | null;
        cache_write_tokens?: number | null;
    } | null;
};

function mapOpenAIToolUsage(
    toolUsage: OpenAI.Responses.Response.ToolUsage | null | undefined,
): ExecutionTokenUsage['tool_usage'] {
    const image = toolUsage?.image_gen;
    const searches = toolUsage?.web_search?.num_requests;
    const imageUsed = !!image && (image.total_tokens > 0 || image.input_tokens > 0 || image.output_tokens > 0);
    if (!imageUsed && !searches) return undefined;
    return {
        ...(imageUsed
            ? {
                  image_generation: {
                      prompt: image.input_tokens,
                      // An omitted cache count means the cache use is unknown, so it stays omitted.
                      ...(image.cached_tokens != null ? { prompt_cached: image.cached_tokens } : {}),
                      result: image.output_tokens,
                      result_image: imageResultTokens(image),
                      total: image.total_tokens,
                  },
              }
            : {}),
        ...(searches ? { web_search_requests: searches } : {}),
    };
}
