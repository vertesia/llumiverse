import type { z } from 'zod';
import { getBedrockModelKnowledge } from '../capability/bedrock-models.js';
import type {
    BedrockMantleChatCompletionsOptionsSchema,
    BedrockMantleClaudeOptionsSchema,
    BedrockMantleResponsesOptionsSchema,
} from '../schemas/model-options.js';
import { type ModelOptionInfoItem, type ModelOptions, type ModelOptionsInfo, OptionType } from '../types.js';
import { getAnthropicOptions } from './anthropic.js';
import { textOptionsFallback } from './fallback.js';
import { getOpenAIReasoningEffortLevels, isModelFamilyVersionGTE } from './version-parsing.js';

// The option shapes are DERIVED, not declared. Each schema in `../schemas/model-options.js` is the
// single definition of its option set: it is what the OpenAPI document publishes, what AJV enforces,
// and — through `z.infer` below — what TypeScript sees. There is nothing here to keep in step with
// anything, because there is only one statement of the shape.
//
// The OpenAPI scanner short-circuits these aliases to the component of the same name rather than
// trying to expand `z.infer`, which it cannot do. That is why the alias name, the schema variable
// and the published component id must all agree; generation fails loudly if they do not.
export type BedrockMantleResponsesOptions = z.infer<typeof BedrockMantleResponsesOptionsSchema>;
export type BedrockMantleChatCompletionsOptions = z.infer<typeof BedrockMantleChatCompletionsOptionsSchema>;
export type BedrockMantleClaudeOptions = z.infer<typeof BedrockMantleClaudeOptionsSchema>;

export type BedrockMantleProtocol = 'responses' | 'chat_completions' | 'messages';

/**
 * @discriminator _option_id
 */
export type BedrockMantleOptions =
    | BedrockMantleResponsesOptions
    | BedrockMantleChatCompletionsOptions
    | BedrockMantleClaudeOptions;

const CHAT_COMPLETIONS_PUBLISHERS = new Set([
    'deepseek',
    'minimax',
    'mistral',
    'moonshot',
    'moonshotai',
    'nvidia',
    'qwen',
    'writer',
    'zai',
]);

const PUBLISHERS: Record<string, string> = {
    anthropic: 'Anthropic',
    deepseek: 'DeepSeek',
    google: 'Google',
    minimax: 'MiniMax',
    mistral: 'Mistral AI',
    moonshot: 'Moonshot AI',
    moonshotai: 'Moonshot AI',
    nvidia: 'NVIDIA',
    openai: 'OpenAI',
    qwen: 'Qwen',
    writer: 'Writer',
    xai: 'xAI',
    zai: 'Z.AI',
};

export interface BedrockMantleModelInfo {
    protocol: BedrockMantleProtocol;
    owner: string;
}

function getPublisher(model: string): string {
    return model.split('.')[0];
}

function supportsImageInput(model: string): boolean {
    return getBedrockModelKnowledge(model).input.image === true;
}

export function getBedrockMantleProtocol(model: string): BedrockMantleProtocol | undefined {
    const normalized = model.toLowerCase();
    const publisher = getPublisher(normalized);

    // Intentional protocol allow-list: Mantle hosts families across three wire protocols. A new version of a known
    // family inherits its latest rule; a brand-new family stays unclassified until its working endpoint is known.
    if (publisher === 'anthropic' && normalized.includes('.claude-')) return 'messages';
    if (publisher === 'openai') {
        // GPT-OSS works through Bedrock Runtime Converse, but on Bedrock Mantle the
        // Responses route has proven unavailable/unreliable. Keep the whole OSS
        // family on Chat Completions, including future and safeguard variants.
        if (normalized.includes('.gpt-oss')) return 'chat_completions';
        if (normalized.includes('.gpt-')) return 'responses';
        return undefined;
    }
    if (publisher === 'xai' && normalized.includes('.grok-')) return 'responses';
    if (publisher === 'google' && normalized.includes('.gemma-')) {
        // Gemma 3 uses /v1/chat/completions. Gemma 4 moved to the OpenAI-compatible
        // /openai/v1 endpoint and supports Responses, which is our preferred API.
        // Later numeric generations inherit the latest known Gemma behavior.
        if (isModelFamilyVersionGTE(normalized, 'google.gemma-', 4, 0)) return 'responses';
        if (isModelFamilyVersionGTE(normalized, 'google.gemma-', 3, 0)) return 'chat_completions';
        return undefined;
    }
    if (CHAT_COMPLETIONS_PUBLISHERS.has(publisher)) return 'chat_completions';
    return undefined;
}

export function getBedrockMantleModelInfo(model: string): BedrockMantleModelInfo | undefined {
    const normalized = model.toLowerCase();
    const protocol = getBedrockMantleProtocol(normalized);
    if (!protocol) return undefined;
    const publisher = getPublisher(normalized);
    return {
        protocol,
        owner: PUBLISHERS[publisher] ?? publisher,
    };
}

/**
 * Backwards-compatible family helper retained for callers that distinguish the
 * OpenAI and Grok Responses option variants.
 */
export type BedrockMantleModelFamily = 'openai' | 'grok';

export function getBedrockMantleModelFamily(model: string): BedrockMantleModelFamily | undefined {
    if (getBedrockMantleProtocol(model) !== 'responses') return undefined;
    const normalized = model.toLowerCase();
    if (normalized.startsWith('openai.')) return 'openai';
    if (normalized.startsWith('xai.grok-')) return 'grok';
    return undefined;
}

function maxTokensOption(model: string): ModelOptionInfoItem {
    return {
        name: 'max_tokens',
        type: OptionType.numeric,
        min: 1,
        max: getBedrockModelKnowledge(model).max_output_tokens,
        integer: true,
        step: 200,
        description: 'The maximum number of tokens to generate',
    };
}

function getResponsesOptions(model: string): ModelOptionsInfo {
    const normalized = model.toLowerCase();
    const isGrok = normalized.startsWith('xai.grok-');
    const isOpenAI = normalized.startsWith('openai.gpt-');
    const isGemma4 = isModelFamilyVersionGTE(normalized, 'google.gemma-', 4, 0);
    const openAiEffortLevels = getOpenAIReasoningEffortLevels(normalized);
    const reasoningEffortEnum: Record<string, string> | undefined = isGrok
        ? { none: 'none', low: 'low', medium: 'medium', high: 'high' }
        : isOpenAI && openAiEffortLevels
          ? Object.fromEntries(Object.values(openAiEffortLevels).map((value) => [value, value]))
          : isGemma4
            ? { low: 'low', medium: 'medium', high: 'high' }
            : undefined;
    const reasoningEffortDefault = isGrok ? 'low' : isGemma4 ? 'high' : 'medium';
    const options: ModelOptionInfoItem[] = [maxTokensOption(model)];

    if (isGrok) {
        options.push(
            {
                name: 'temperature',
                type: OptionType.numeric,
                min: 0,
                default: 0.7,
                integer: false,
                step: 0.1,
                description: 'A higher temperature biases toward less likely tokens, making the model more creative',
            },
            {
                name: 'top_p',
                type: OptionType.numeric,
                min: 0,
                max: 1,
                default: 0.95,
                integer: false,
                step: 0.1,
                description: 'Limits the model to the most probable tokens whose cumulative probability is top_p',
            },
        );
    }

    if (reasoningEffortEnum) {
        options.push({
            name: 'effort',
            type: OptionType.enum,
            enum: reasoningEffortEnum,
            default: reasoningEffortDefault,
            description: 'The reasoning effort of the model, which affects the quality and speed of the response',
        });
        options.push({
            name: 'reasoning_effort',
            type: OptionType.enum,
            enum: reasoningEffortEnum,
            default: reasoningEffortDefault,
            description: 'Alias for effort; controls how much reasoning the model performs before responding',
        });
    }

    if (isOpenAI) {
        options.push({
            name: 'verbosity',
            type: OptionType.enum,
            enum: { low: 'low', medium: 'medium', high: 'high' },
            default: 'medium',
            description: 'Controls how concise or verbose the model response should be',
        });
    }

    if (supportsImageInput(normalized)) {
        options.push({
            name: 'image_detail',
            type: OptionType.enum,
            enum: { Low: 'low', High: 'high', Auto: 'auto' },
            default: 'auto',
            description: 'Controls how the model processes an input image',
        });
    }

    return { _option_id: 'bedrock-mantle-responses', options };
}

function getChatCompletionsOptions(model: string): ModelOptionsInfo {
    const allowedOptions = new Set(['max_tokens', 'temperature', 'top_p', 'stop_sequence', 'include_thoughts']);
    const maxOutputTokens = getBedrockModelKnowledge(model).max_output_tokens;
    const options = textOptionsFallback.options
        .filter((option) => allowedOptions.has(option.name))
        .map((option) => (option.name === 'max_tokens' ? { ...option, max: maxOutputTokens } : option));
    if (model.toLowerCase().includes('gpt-oss')) {
        const effortOption: ModelOptionInfoItem = {
            name: 'effort',
            type: OptionType.enum,
            enum: { low: 'low', medium: 'medium', high: 'high' },
            default: 'medium',
            description: 'Controls how much reasoning the model performs before responding',
        };
        options.push(effortOption, { ...effortOption, name: 'reasoning_effort' });
    }
    return {
        _option_id: 'bedrock-mantle-chat-completions',
        options,
    };
}

export function getBedrockMantleOptions(model: string, option?: ModelOptions): ModelOptionsInfo {
    switch (getBedrockMantleProtocol(model)) {
        case 'responses':
            return getResponsesOptions(model);
        case 'chat_completions':
            return getChatCompletionsOptions(model);
        case 'messages': {
            const anthropicOptions = getAnthropicOptions(model, option);
            return { ...anthropicOptions, _option_id: 'bedrock-mantle-claude' };
        }
        default:
            return textOptionsFallback;
    }
}
