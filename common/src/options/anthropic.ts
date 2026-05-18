import { type ModelOptions, type ModelOptionsInfo, OptionType, SharedOptions } from "../types.js";
import { textOptionsFallback } from "./fallback.js";
import {
    buildClaudeCacheOptions,
    buildClaudeCacheTtlOptions,
    buildClaudeEffortOptions,
    buildClaudeIncludeThoughtsOption,
    buildClaudeThinkingBudgetOption,
    getClaudeMaxTokensLimit,
} from "./shared-parsing.js";
import { hasSamplingParameterRestriction } from "./version-parsing.js";

export interface AnthropicClaudeOptions {
    _option_id: "anthropic-claude";
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    stop_sequence?: string[];
    effort?: 'low' | 'medium' | 'high' | 'xhigh' | 'max';
    thinking_budget_tokens?: number;
    include_thoughts?: boolean;
    cache_enabled?: boolean;
    cache_ttl?: '5m' | '1h';
}

export function getAnthropicOptions(model: string, option?: ModelOptions): ModelOptionsInfo {
    const max_tokens_limit = getClaudeMaxTokensLimit(model);
    const excludeOptions = ["max_tokens", "presence_penalty", "frequency_penalty"];
    let commonOptions = textOptionsFallback.options.filter((o) => !excludeOptions.includes(o.name));

    const hasSamplingRestriction = hasSamplingParameterRestriction(model);
    if (hasSamplingRestriction) {
        commonOptions = commonOptions.filter((o) =>
            o.name !== SharedOptions.temperature &&
            o.name !== SharedOptions.top_p &&
            o.name !== "top_k"
        );
    }

    return {
        _option_id: "anthropic-claude",
        options: [
            {
                name: SharedOptions.max_tokens, type: OptionType.numeric, min: 1, max: max_tokens_limit,
                integer: true, step: 200, description: "The maximum number of tokens to generate",
            },
            ...commonOptions,
            ...buildClaudeEffortOptions(model),
            ...buildClaudeThinkingBudgetOption(model),
            ...buildClaudeIncludeThoughtsOption(model),
            ...buildClaudeCacheOptions(),
            ...buildClaudeCacheTtlOptions((option as unknown as AnthropicClaudeOptions)?.cache_enabled),
        ],
    };
}
