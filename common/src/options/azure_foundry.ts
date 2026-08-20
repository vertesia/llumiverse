import type { z } from 'zod';
import { resolveModelProfile } from '../model-directory.js';
import type { AzureFoundryChatOptionsSchema } from '../schemas/model-options.js';
import {
    type ModelOptionInfoItem,
    type ModelOptions,
    type ModelOptionsInfo,
    OptionType,
    Providers,
    SharedOptions,
} from '../types.js';
import { getMaxOutputTokens } from './context-windows.js';
import { getOpenAiOptions } from './openai.js';
import { isOpenAIGptVersionGTE } from './version-parsing.js';

// Helper function to parse composite model IDs
function parseAzureFoundryModelId(compositeId: string): { deploymentName: string; baseModel: string } {
    const parts = compositeId.split('::');
    if (parts.length === 2) {
        return {
            deploymentName: parts[0],
            baseModel: parts[1],
        };
    }

    // Backwards compatibility: if no delimiter found, treat as deployment name
    return {
        deploymentName: compositeId,
        baseModel: compositeId,
    };
}

export type AzureFoundryChatOptions = z.infer<typeof AzureFoundryChatOptionsSchema>;

export function getMaxTokensLimitAzureFoundry(model: string): number | undefined {
    // Extract base model from composite ID (deployment::baseModel)
    const { baseModel } = parseAzureFoundryModelId(model);
    const modelLower = baseModel.toLowerCase();
    // GPT models
    if (modelLower.includes('gpt-4o')) {
        if (modelLower.includes('mini')) {
            return 16384;
        }
        return 16384;
    }
    if (modelLower.includes('gpt-4')) {
        if (modelLower.includes('turbo')) {
            return 4096;
        }
        if (modelLower.includes('32k')) {
            return 32768;
        }
        return 8192;
    }
    if (modelLower.includes('gpt-35') || modelLower.includes('gpt-3.5')) {
        return 4096;
    }
    if (isOpenAIGptVersionGTE(modelLower, 5, 0)) {
        return 128000;
    }
    // O-series models
    if (modelLower.includes('o1')) {
        if (modelLower.includes('preview')) {
            return 32768;
        }
        if (modelLower.includes('mini')) {
            return 65536;
        }
        return 100000;
    }
    if (modelLower.includes('o3')) {
        if (modelLower.includes('mini')) {
            return 100000;
        }
        return 100000;
    }
    if (modelLower.includes('o4')) {
        return 100000;
    }
    // DeepSeek models
    if (modelLower.includes('deepseek')) {
        if (modelLower.includes('r1')) {
            return 163840;
        }
        if (modelLower.includes('v3')) {
            return 131072;
        }
    }
    // Claude models — delegate to provider-agnostic limits
    if (modelLower.includes('claude')) {
        return getMaxOutputTokens(modelLower);
    }
    // Llama models
    if (modelLower.includes('llama')) {
        return getMaxOutputTokens(modelLower);
    }
    // Mistral models
    if (modelLower.includes('mistral')) {
        if (modelLower.includes('large')) {
            return 4096;
        }
        if (modelLower.includes('small')) {
            return 4096;
        }
        return 4096;
    }
    // Phi models
    if (modelLower.includes('phi')) {
        return 4096;
    }
    // AI21 Jamba models
    if (modelLower.includes('jamba')) {
        return 4096;
    }
    // Cohere models
    if (modelLower.includes('cohere')) {
        if (modelLower.includes('command-a')) {
            return 8000;
        }
        return 4096;
    }
    // Grok models
    if (modelLower.includes('grok')) {
        return 131072;
    }
    return undefined;
}

export function getAzureFoundryOptions(model: string, _option?: ModelOptions): ModelOptionsInfo {
    // Extract base model from composite ID (deployment::baseModel)
    const { baseModel } = parseAzureFoundryModelId(model);
    const modelLower = baseModel.toLowerCase();
    const max_tokens_limit = getMaxTokensLimitAzureFoundry(model);
    const profile = resolveModelProfile(model, Providers.azure_foundry);
    if (modelLower.includes('gpt-') || modelLower.includes('dall-e') || /(?:^|[~/.])o\d+(?:[-_.]|$)/.test(modelLower)) {
        return getOpenAiOptions(baseModel, _option, profile);
    }
    // Vision model options
    const visionOptions: ModelOptionInfoItem[] =
        profile.capabilities.input.image === true
            ? [
                  {
                      name: 'image_detail',
                      type: OptionType.enum,
                      enum: { Low: 'low', High: 'high', Auto: 'auto' },
                      default: 'auto',
                      description: 'Controls how the model processes input images',
                  },
              ]
            : [];
    // DeepSeek R1 models
    if (modelLower.includes('deepseek') && modelLower.includes('r1')) {
        return {
            _option_id: 'azure-foundry-chat',
            options: [
                {
                    name: SharedOptions.max_tokens,
                    type: OptionType.numeric,
                    min: 1,
                    max: max_tokens_limit,
                    integer: true,
                    description: 'The maximum number of tokens to generate',
                },
                {
                    name: SharedOptions.temperature,
                    type: OptionType.numeric,
                    min: 0.0,
                    max: 1.0,
                    default: 0.7,
                    step: 0.1,
                    description: 'Lower temperatures recommended for DeepSeek R1 (0.3-0.7)',
                },
                {
                    name: SharedOptions.top_p,
                    type: OptionType.numeric,
                    min: 0,
                    max: 1,
                    step: 0.1,
                    description: 'Nucleus sampling parameter',
                },
                {
                    name: SharedOptions.stop_sequence,
                    type: OptionType.string_list,
                    value: [],
                    description: 'Sequences where the model will stop generating',
                },
                {
                    name: SharedOptions.seed,
                    type: OptionType.numeric,
                    integer: true,
                    description: 'Random seed for reproducible generation',
                },
                {
                    name: 'include_thoughts',
                    type: OptionType.boolean,
                    default: true,
                    description: 'Include visible model reasoning as separate thoughts results',
                },
            ],
        };
    }
    // General text models (Claude, Llama, Mistral, Phi, etc.)
    const baseOptions: ModelOptionInfoItem[] = [
        {
            name: SharedOptions.max_tokens,
            type: OptionType.numeric,
            min: 1,
            max: max_tokens_limit,
            integer: true,
            step: 200,
            description: 'The maximum number of tokens to generate',
        },
        {
            name: SharedOptions.temperature,
            type: OptionType.numeric,
            min: 0.0,
            max: 1.0,
            default: 0.7,
            step: 0.1,
            description: 'Controls randomness in the output',
        },
        {
            name: SharedOptions.top_p,
            type: OptionType.numeric,
            min: 0,
            max: 1,
            step: 0.1,
            description: 'Nucleus sampling parameter',
        },
        {
            name: SharedOptions.stop_sequence,
            type: OptionType.string_list,
            value: [],
            description: 'Sequences where the model will stop generating',
        },
    ];
    // Add model-specific options
    const additionalOptions: ModelOptionInfoItem[] = [];
    // Add penalty options for certain models
    if (modelLower.includes('claude') || modelLower.includes('jamba') || modelLower.includes('cohere')) {
        additionalOptions.push(
            {
                name: SharedOptions.presence_penalty,
                type: OptionType.numeric,
                min: -2.0,
                max: 2.0,
                step: 0.1,
                description: 'Penalize new tokens based on their presence in the text',
            },
            {
                name: SharedOptions.frequency_penalty,
                type: OptionType.numeric,
                min: -2.0,
                max: 2.0,
                step: 0.1,
                description: 'Penalize new tokens based on their frequency in the text',
            },
        );
    }
    additionalOptions.push({
        name: SharedOptions.seed,
        type: OptionType.numeric,
        integer: true,
        description: 'Random seed for reproducible generation',
    });
    additionalOptions.push({
        name: 'include_thoughts',
        type: OptionType.boolean,
        default: true,
        description: 'Include visible model reasoning as separate thoughts results',
    });
    return {
        _option_id: 'azure-foundry-chat',
        options: [...baseOptions, ...additionalOptions, ...visionOptions],
    };
}
