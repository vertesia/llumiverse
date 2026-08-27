import type { z } from 'zod';
import { type ModelProfile, resolveModelProfile } from '../model-directory.js';
import type { OpenRouterTextOptionsSchema } from '../schemas/model-options.js';
import { type ModelOptionInfoItem, type ModelOptions, type ModelOptionsInfo, OptionType, Providers } from '../types.js';
import { getOpenAiCompatibleOptions } from './openai.js';

export type OpenRouterTextOptions = z.infer<typeof OpenRouterTextOptionsSchema>;

const providerRoutingOptions: ModelOptionInfoItem[] = [
    {
        name: 'provider_sort',
        type: OptionType.enum,
        enum: { Price: 'price', Throughput: 'throughput', Latency: 'latency', Exacto: 'exacto' },
        description: 'Sort OpenRouter endpoints by price, throughput, latency, or tool-calling quality.',
    },
    {
        name: 'provider_order',
        type: OptionType.string_list,
        description: 'Provider slugs to try in order. When set, OpenRouter disables load balancing.',
    },
    {
        name: 'provider_only',
        type: OptionType.string_list,
        description: 'Only route through these OpenRouter provider slugs.',
    },
    {
        name: 'provider_ignore',
        type: OptionType.string_list,
        description: 'Do not route through these OpenRouter provider slugs.',
    },
    {
        name: 'provider_allow_fallbacks',
        type: OptionType.boolean,
        default: true,
        description: 'Allow OpenRouter to try backup providers when the preferred provider is unavailable.',
    },
    {
        name: 'provider_require_parameters',
        type: OptionType.boolean,
        default: false,
        description: 'Only use providers that support every parameter in the request.',
    },
    {
        name: 'provider_data_collection',
        type: OptionType.enum,
        enum: { Allow: 'allow', Deny: 'deny' },
        default: 'allow',
        description: 'Control whether routed providers may collect request data.',
    },
    {
        name: 'provider_zdr',
        type: OptionType.boolean,
        description: 'Only use zero-data-retention endpoints.',
    },
    {
        name: 'provider_quantizations',
        type: OptionType.string_list,
        description: 'Only use endpoints with one of these quantization levels.',
    },
];

export function getOpenRouterOptions(
    model: string,
    options?: ModelOptions,
    profile: ModelProfile = resolveModelProfile(model, Providers.openrouter),
): ModelOptionsInfo {
    const compatible = getOpenAiCompatibleOptions(model, options, profile);
    return {
        _option_id: 'openrouter-text',
        options: [...compatible.options.filter((option) => option.name !== 'extra_body'), ...providerRoutingOptions],
    };
}
