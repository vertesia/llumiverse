import type { z } from 'zod';
import type { AnthropicClaudeOptionsSchema } from '../schemas/model-options.js';
import { type ModelOptions, type ModelOptionsInfo, OptionType, SharedOptions } from '../types.js';
import { textOptionsFallback } from './fallback.js';
import {
    buildClaudeCacheOptions,
    buildClaudeCacheTtlOptions,
    buildClaudeEffortOptions,
    buildClaudeIncludeThoughtsOption,
    buildClaudeThinkingBudgetOption,
    getClaudeMaxTokensLimit,
} from './shared-parsing.js';
import { hasSamplingParameterRestriction } from './version-parsing.js';

export type AnthropicClaudeOptions = z.infer<typeof AnthropicClaudeOptionsSchema>;

export function getAnthropicOptions(model: string, option?: ModelOptions): ModelOptionsInfo {
    const max_tokens_limit = getClaudeMaxTokensLimit(model);
    const excludeOptions = ['max_tokens', 'presence_penalty', 'frequency_penalty', 'include_thoughts'];
    let commonOptions = textOptionsFallback.options.filter((o) => !excludeOptions.includes(o.name));

    const hasSamplingRestriction = hasSamplingParameterRestriction(model);
    if (hasSamplingRestriction) {
        commonOptions = commonOptions.filter(
            (o) => o.name !== SharedOptions.temperature && o.name !== SharedOptions.top_p && o.name !== 'top_k',
        );
    }

    return {
        _option_id: 'anthropic-claude',
        options: [
            {
                name: SharedOptions.max_tokens,
                type: OptionType.numeric,
                min: 1,
                max: max_tokens_limit,
                integer: true,
                step: 200,
                description: 'The maximum number of tokens to generate',
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
