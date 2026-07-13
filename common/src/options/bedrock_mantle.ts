import { getBedrockModelKnowledge } from '../capability/bedrock-models.js';
import { type ModelOptionInfoItem, type ModelOptions, type ModelOptionsInfo, OptionType } from '../types.js';
import { getAnthropicOptions } from './anthropic.js';
import { textOptionsFallback } from './fallback.js';
import { isModelFamilyVersionGTE } from './version-parsing.js';

export type BedrockMantleProtocol = 'responses' | 'chat_completions' | 'messages';

export interface BedrockMantleResponsesOptions {
    _option_id: 'bedrock-mantle-responses';
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    effort?: 'none' | 'low' | 'medium' | 'high' | 'xhigh';
    reasoning_effort?: 'none' | 'low' | 'medium' | 'high' | 'xhigh';
    verbosity?: 'low' | 'medium' | 'high';
    image_detail?: 'low' | 'high' | 'auto';
    include_thoughts?: boolean;
}

export interface BedrockMantleChatCompletionsOptions {
    _option_id: 'bedrock-mantle-chat-completions';
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    stop_sequence?: string[];
    include_thoughts?: boolean;
}

export interface BedrockMantleClaudeOptions {
    _option_id: 'bedrock-mantle-claude';
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
    const reasoningEffortEnum: Record<string, string> = isGrok
        ? { none: 'none', low: 'low', medium: 'medium', high: 'high' }
        : { low: 'low', medium: 'medium', high: 'high', xhigh: 'xhigh' };
    const reasoningEffortDefault = isGrok ? 'low' : 'medium';
    const options: ModelOptionInfoItem[] = [
        maxTokensOption(model),
        {
            name: 'include_thoughts',
            type: OptionType.boolean,
            default: true,
            description: 'Include visible model reasoning as separate thoughts results.',
        },
    ];

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

    options.push(
        {
            name: 'effort',
            type: OptionType.enum,
            enum: reasoningEffortEnum,
            default: reasoningEffortDefault,
            description: 'The reasoning effort of the model, which affects the quality and speed of the response',
        },
        {
            name: 'reasoning_effort',
            type: OptionType.enum,
            enum: reasoningEffortEnum,
            default: reasoningEffortDefault,
            description: 'Alias for effort; controls how much reasoning the model performs before responding',
        },
    );

    if (!isGrok) {
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
    return {
        _option_id: 'bedrock-mantle-chat-completions',
        options: textOptionsFallback.options
            .filter((option) => allowedOptions.has(option.name))
            .map((option) => (option.name === 'max_tokens' ? { ...option, max: maxOutputTokens } : option)),
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
