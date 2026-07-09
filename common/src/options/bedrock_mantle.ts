import { type ModelOptionInfoItem, type ModelOptions, type ModelOptionsInfo, OptionType } from '../types.js';
import { textOptionsFallback } from './fallback.js';

export interface BedrockMantleOptions {
    _option_id: 'bedrock-mantle-responses';
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    effort?: 'none' | 'low' | 'medium' | 'high';
    reasoning_effort?: 'none' | 'low' | 'medium' | 'high';
    verbosity?: 'low' | 'medium' | 'high';
    image_detail?: 'low' | 'high' | 'auto';
}

function isBedrockMantleResponsesModel(model: string): boolean {
    return model.includes('openai.gpt-5.5') || model.includes('openai.gpt-5.4') || model.includes('xai.grok-4.3');
}

function isBedrockMantleGrokModel(model: string): boolean {
    return model.includes('xai.grok-4.3');
}

export function getBedrockMantleOptions(model: string, _option?: ModelOptions): ModelOptionsInfo {
    if (!isBedrockMantleResponsesModel(model)) {
        return textOptionsFallback;
    }

    const maxTokensOption: ModelOptionInfoItem = {
        name: 'max_tokens',
        type: OptionType.numeric,
        min: 1,
        integer: true,
        step: 200,
        description: 'The maximum number of tokens to generate',
    };
    const samplingOptions: ModelOptionInfoItem[] = isBedrockMantleGrokModel(model)
        ? [
              {
                  name: 'temperature',
                  type: OptionType.numeric,
                  min: 0.0,
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
          ]
        : [];
    const reasoningEffortEnum: Record<string, string> = isBedrockMantleGrokModel(model)
        ? {
              none: 'none',
              low: 'low',
              medium: 'medium',
              high: 'high',
          }
        : {
              low: 'low',
              medium: 'medium',
              high: 'high',
          };
    const reasoningEffortDefault = isBedrockMantleGrokModel(model) ? 'low' : 'medium';
    const mantleOptions: ModelOptionInfoItem[] = [
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
        {
            name: 'image_detail',
            type: OptionType.enum,
            enum: { Low: 'low', High: 'high', Auto: 'auto' },
            default: 'auto',
            description: 'Controls how the model processes an input image',
        },
    ];
    if (!isBedrockMantleGrokModel(model)) {
        mantleOptions.splice(2, 0, {
            name: 'verbosity',
            type: OptionType.enum,
            enum: {
                low: 'low',
                medium: 'medium',
                high: 'high',
            },
            default: 'medium',
            description: 'Controls how concise or verbose the model response should be',
        });
    }

    return {
        _option_id: 'bedrock-mantle-responses',
        options: [maxTokensOption, ...samplingOptions, ...mantleOptions],
    };
}
