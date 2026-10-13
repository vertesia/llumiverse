import type {
    AnthropicBeta,
    BetaInputTransformation,
    BetaThinkingConfigParam,
} from '@anthropic-ai/sdk/resources/beta.js';
import type { OutputConfig } from '@anthropic-ai/sdk/resources/messages.js';
import {
    type AnthropicClaudeOptions,
    type ClaudeVersion,
    hasSamplingParameterRestriction,
    isClaudeVersionGTE,
    type Logger,
    parseClaudeVersion,
    supportsAdaptiveThinking,
} from '@llumiverse/core';

import { logModelOptionException } from './model-option-exceptions.js';

/**
 * Common Claude model options relevant to thinking/effort configuration.
 * Works with both VertexAIClaudeOptions and BedrockClaudeOptions.
 */
export interface ClaudeThinkingInput {
    thinking_budget_tokens?: number;
    thinking_mode?: AnthropicClaudeOptions['thinking_mode'];
    effort?: NonNullable<OutputConfig['effort']>;
    /** Controls whether thinking content is included in the response. Does not enable thinking. */
    include_thoughts?: boolean;
}

/** Beta that enables `thinking.block_binding` and the response's `input_transformations`. */
export const THINKING_BINDING_BETA: AnthropicBeta = 'thinking-binding-controls-2026-08-01';

/**
 * Whether the model binds each thinking block to the conversation before it (system, tools, earlier
 * messages) and rejects a replayed block once that prefix changes: Claude Fable 5.1 and Claude 5.5+.
 */
function enforcesThinkingBinding(version: ClaudeVersion | null): boolean {
    if (!version) return false;
    const { major, minor, variant } = version;
    return major > 5 || (major === 5 && (minor >= 5 || (variant === 'fable' && minor >= 1)));
}

/**
 * Result of resolving Claude thinking and effort configuration.
 */
export interface ClaudeThinkingResult {
    /** Thinking/reasoning config to include in the API payload. */
    thinking: BetaThinkingConfigParam | undefined;
    /** Beta values the thinking config requires; empty when it needs none. */
    betas: AnthropicBeta[];
    /** Output config (effort) to include in the API payload, if applicable. */
    outputConfig: OutputConfig | undefined;
    /** Whether sampling parameters (temperature, top_p, top_k) should be stripped. */
    hasSamplingRestriction: boolean;
    /** Whether the model supports thinking at all (>= Claude 3.7). */
    supportsThinking: boolean;
}

/**
 * Resolve thinking and effort configuration for a Claude model.
 *
 * - Explicit thinking_mode overrides inferred mode; between_tools omits display and budget.
 * - Extended thinking: enabled by setting `thinking_budget_tokens`.
 * - Adaptive thinking: enabled by setting `effort` on models that support it (Opus 4.6+, Sonnet 4.6+).
 * - `include_thoughts`: display-only; does not enable thinking.
 *
 * @param model - The model identifier string
 * @param options - User-provided Claude options (thinking_budget_tokens, effort, include_thoughts)
 */
export function resolveClaudeThinking(
    model: string,
    options?: ClaudeThinkingInput,
    logger?: Logger,
): ClaudeThinkingResult {
    const supportsAdaptive = supportsAdaptiveThinking(model);
    const samplingRestriction = hasSamplingParameterRestriction(model);
    const supportsThinking = isClaudeVersionGTE(model, 3, 7);
    const budgetTokens = options?.thinking_budget_tokens;
    const version = parseClaudeVersion(model);
    const defaultAdaptiveThinking =
        (version?.major === 5 && ['opus', 'sonnet', 'fable', 'mythos'].includes(version.variant)) ||
        (version?.variant === 'haiku' && isClaudeVersionGTE(model, 5, 5));
    // Adaptive thinking is active when the caller supplies an effort level on a
    // model that supports it. Extended thinking is active when a budget is set.
    const adaptiveEnabled = supportsAdaptive && options?.effort != null;
    const extendedEnabled = budgetTokens != null && !samplingRestriction;

    let thinking: BetaThinkingConfigParam | undefined;

    if (options?.thinking_mode === 'between_tools') {
        // This mode accepts only type: no display or budget, even if stale settings are present.
        thinking = { type: 'between_tools' };
    } else if (options?.thinking_mode === 'adaptive') {
        thinking = { type: 'adaptive', display: options.include_thoughts ? 'summarized' : 'omitted' };
    } else if (!supportsThinking) {
        // Pre-3.7 models: no thinking support
        thinking = undefined;
    } else if (adaptiveEnabled) {
        // Prefer adaptive thinking when an effort level is supplied. This also
        // ignores stale legacy budgets that may remain on a migrated model config.
        thinking = { type: 'adaptive' as const, display: options?.include_thoughts ? 'summarized' : 'omitted' };
    } else if (extendedEnabled) {
        // Preserve explicitly budgeted configurations, including dual-mode models
        // such as Sonnet 4.6 when no effort level was supplied.
        thinking = {
            type: 'enabled' as const,
            budget_tokens: budgetTokens,
        };
    } else if (defaultAdaptiveThinking) {
        // Set display explicitly for models with default adaptive thinking so include_thoughts requests summaries.
        thinking = { type: 'adaptive' as const, display: options?.include_thoughts ? 'summarized' : 'omitted' };
    } else if (supportsAdaptive) {
        // Other adaptive models: leave the provider default unchanged when effort is omitted.
        // display controls whether thinking blocks are returned; defaults to omitted.
        thinking = undefined;
    } else {
        // Older thinking models (3.7, 4.5): no adaptive support, thinking is always disabled
        // unless an explicit budget is provided (handled above).
        thinking = { type: 'disabled' as const };
    }

    // Compatibility exception: existing Claude mode selection can discard a legacy thinking budget.
    if (thinking?.type !== 'enabled') {
        logModelOptionException(logger, model, options, ['thinking_budget_tokens'], 'claude_thinking_mode');
    }
    if (thinking?.type === 'between_tools') {
        // Compatibility exception: between-tools mode has no caller-controlled display setting.
        logModelOptionException(logger, model, options, ['include_thoughts'], 'claude_between_tools_display');
    }

    // Output config for effort parameter (Opus 4.5+, Sonnet 4.6+, all 4.7+)
    const outputConfig: OutputConfig | undefined = options?.effort ? { effort: options.effort } : undefined;

    // Compatibility exception: models that bind thinking blocks to the conversation reject a replayed block
    // with a 400 once the system prompt, tool set or earlier messages change, which agents do when a skill
    // unlocks tools. Ask the API to drop those blocks instead of failing the request. Applies only to adaptive
    // thinking: between_tools rejects block_binding, and these models accept no thinking budget.
    const betas: AnthropicBeta[] = [];
    if (thinking?.type === 'adaptive' && enforcesThinkingBinding(version)) {
        thinking = { ...thinking, block_binding: { prefix_mismatch_behavior: 'drop_block' } };
        betas.push(THINKING_BINDING_BETA);
    }

    return {
        thinking,
        betas,
        outputConfig,
        hasSamplingRestriction: samplingRestriction,
        supportsThinking,
    };
}

/**
 * Report replayed thinking blocks the API dropped or let through because they no longer match the request
 * (`thinking.block_binding`): a changed conversation prefix, another model, or another account.
 */
export function logClaudeInputTransformations(
    logger: Logger | undefined,
    transformations: BetaInputTransformation[] | null | undefined,
    context: { provider: string; model: string },
): void {
    if (!logger || !transformations?.length) return;

    const counts: Record<string, number> = {};
    for (const { type, reason } of transformations) {
        const key = `${type}:${reason}`;
        counts[key] = (counts[key] ?? 0) + 1;
    }
    logger.warn(
        {
            ...context,
            transformation_count: transformations.length,
            transformations: counts,
            first_path: transformations[0].path,
        },
        '[Claude] API transformed replayed thinking blocks',
    );
}
