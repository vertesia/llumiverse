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
}

export interface BedrockMantleChatCompletionsOptions {
    _option_id: 'bedrock-mantle-chat-completions';
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    stop_sequence?: string[];
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
    input_image: boolean;
}

function getPublisher(model: string): string {
    return model.split('.')[0];
}

function isKimiMultimodal(model: string): boolean {
    const match = /kimi-k(\d+)(?:[.-](\d+))?/i.exec(model);
    if (!match) return false;
    const major = Number(match[1]);
    const minor = Number(match[2] ?? 0);
    return major > 2 || (major === 2 && minor >= 5);
}

function supportsImageInput(model: string, protocol: BedrockMantleProtocol): boolean {
    if (protocol === 'messages') return true;

    const publisher = getPublisher(model);
    switch (publisher) {
        case 'openai':
            return !model.includes('gpt-oss');
        case 'xai':
        case 'google':
            return true;
        case 'mistral':
            return (
                model.includes('magistral-') ||
                model.includes('ministral-3-') ||
                model.includes('mistral-large-3-') ||
                model.includes('pixtral-')
            );
        case 'moonshot':
        case 'moonshotai':
            return isKimiMultimodal(model);
        case 'nvidia':
            return model.includes('nemotron-nano-12b');
        case 'qwen':
            return model.includes('-vl-');
        case 'writer':
            return model.includes('vision');
        default:
            return false;
    }
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
        input_image: supportsImageInput(normalized, protocol),
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

function maxTokensOption(): ModelOptionInfoItem {
    return {
        name: 'max_tokens',
        type: OptionType.numeric,
        min: 1,
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
    const options: ModelOptionInfoItem[] = [maxTokensOption()];

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

    if (supportsImageInput(normalized, 'responses')) {
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

function getChatCompletionsOptions(): ModelOptionsInfo {
    const allowedOptions = new Set(['max_tokens', 'temperature', 'top_p', 'stop_sequence']);
    return {
        _option_id: 'bedrock-mantle-chat-completions',
        options: textOptionsFallback.options.filter((option) => allowedOptions.has(option.name)),
    };
}

export function getBedrockMantleOptions(model: string, option?: ModelOptions): ModelOptionsInfo {
    switch (getBedrockMantleProtocol(model)) {
        case 'responses':
            return getResponsesOptions(model);
        case 'chat_completions':
            return getChatCompletionsOptions();
        case 'messages': {
            const anthropicOptions = getAnthropicOptions(model, option);
            return { ...anthropicOptions, _option_id: 'bedrock-mantle-claude' };
        }
        default:
            return textOptionsFallback;
    }
}
