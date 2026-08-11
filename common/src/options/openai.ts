import type { z } from 'zod';
import type {
    OpenAiDalleOptionsSchema,
    OpenAiGptImageOptionsSchema,
    OpenAiTextOptionsSchema,
    OpenAiThinkingOptionsSchema,
} from '../schemas/model-options.js';
import {
    type ModelOptionInfoItem,
    type ModelOptions,
    type ModelOptionsInfo,
    OptionType,
    SharedOptions,
} from '../types.js';
import { getMaxOutputTokens } from './context-windows.js';
import { getOpenAIReasoningEffortLevels, isOpenAIGptVersionGTE } from './version-parsing.js';

// The option shapes are DERIVED, not declared. Each schema in `../schemas/model-options.js` is the
// single definition of its option set: it is what the OpenAPI document publishes, what AJV enforces,
// and — through `z.infer` below — what TypeScript sees. There is nothing here to keep in step with
// anything, because there is only one statement of the shape.
//
// The OpenAPI scanner short-circuits these aliases to the component of the same name rather than
// trying to expand `z.infer`, which it cannot do. That is why the alias name, the schema variable
// and the published component id must all agree; generation fails loudly if they do not.
export type OpenAiThinkingOptions = z.infer<typeof OpenAiThinkingOptionsSchema>;
export type OpenAiTextOptions = z.infer<typeof OpenAiTextOptionsSchema>;
export type OpenAiDalleOptions = z.infer<typeof OpenAiDalleOptionsSchema>;
export type OpenAiGptImageOptions = z.infer<typeof OpenAiGptImageOptionsSchema>;

// Union type of all OpenAI options
/**
 * @discriminator _option_id
 */
export type OpenAiOptions = OpenAiThinkingOptions | OpenAiTextOptions | OpenAiDalleOptions | OpenAiGptImageOptions;

/** OpenAI model families with published Flex processing support. */
export function isFlexSupportedOpenAIModel(model: string): boolean {
    const modelName = (model.split('/').pop() ?? model).toLowerCase();
    const unsupportedVariant = ['chat', 'codex', 'pro', 'deep-research'].some((variant) => modelName.includes(variant));
    if (unsupportedVariant) return false;

    return (
        /^gpt-5(?:[.-]|$)/.test(modelName) ||
        /^o3(?:-\d{4}|$)/.test(modelName) ||
        /^o4-mini(?:-\d{4}|$)/.test(modelName)
    );
}

export function getOpenAiOptions(model: string, _option?: ModelOptions): ModelOptionsInfo {
    const visionOptions: ModelOptionInfoItem[] = isVisionModel(model)
        ? [
              {
                  name: 'image_detail',
                  type: OptionType.enum,
                  enum: { Low: 'low', High: 'high', Auto: 'auto' },
                  default: 'auto',
                  description: 'Controls how the model processes an input image.',
              },
          ]
        : [];
    const serviceTiers: Record<string, string> = {
        Auto: 'auto',
        Default: 'default',
        Priority: 'priority',
    };
    if (isFlexSupportedOpenAIModel(model)) {
        serviceTiers.Flex = 'flex';
    }
    const serviceTierOptions: ModelOptionInfoItem[] = [
        {
            name: 'service_tier',
            type: OptionType.enum,
            enum: serviceTiers,
            default: 'auto',
            description: 'Select the OpenAI processing tier for this request.',
        },
    ];

    // Image generation models
    if (isImageModel(model)) {
        const isGPTImage = model.includes('gpt-image') || model.includes('chatgpt-image');
        const isDallE2 = model.includes('dall-e-2');
        const isDallE3 = model.includes('dall-e-3');

        const sizeOptions: Record<string, string> = {};
        if (isGPTImage) {
            sizeOptions['1024x1024'] = '1024x1024';
            sizeOptions['1024x1536'] = '1024x1536';
            sizeOptions['1536x1024'] = '1536x1024';
            sizeOptions.Auto = 'auto';
        } else if (isDallE2) {
            sizeOptions['256x256'] = '256x256';
            sizeOptions['512x512'] = '512x512';
            sizeOptions['1024x1024'] = '1024x1024';
        } else if (isDallE3) {
            sizeOptions['1024x1024'] = '1024x1024';
            sizeOptions['1792x1024'] = '1792x1024';
            sizeOptions['1024x1792'] = '1024x1792';
        }

        const baseImageOptions: ModelOptionInfoItem[] = [
            {
                name: 'size',
                type: OptionType.enum,
                enum: sizeOptions,
                default: '1024x1024',
                description: 'The size of the generated image',
            },
        ];

        const gptImageOptions: ModelOptionInfoItem[] = isGPTImage
            ? [
                  {
                      name: 'image_quality',
                      type: OptionType.enum,
                      enum: { Low: 'low', Medium: 'medium', High: 'high', Auto: 'auto' },
                      default: 'auto',
                      description: 'The quality of the generated image',
                  },
                  {
                      name: 'background',
                      type: OptionType.enum,
                      enum: { Transparent: 'transparent', Opaque: 'opaque', Auto: 'auto' },
                      default: 'auto',
                      description: 'The background setting for the image',
                  },
                  {
                      name: 'output_format',
                      type: OptionType.enum,
                      enum: { PNG: 'png', WebP: 'webp', JPEG: 'jpeg' },
                      default: 'png',
                      description: 'The output format for the image',
                  },
              ]
            : [];

        const dalleOptions: ModelOptionInfoItem[] =
            isDallE2 || isDallE3
                ? [
                      {
                          name: 'image_quality',
                          type: OptionType.enum,
                          enum: isDallE3 ? { Standard: 'standard', HD: 'hd' } : { Standard: 'standard' },
                          default: 'standard',
                          description: 'The quality of the generated image',
                      },
                      {
                          name: 'style',
                          type: OptionType.enum,
                          enum: { Vivid: 'vivid', Natural: 'natural' },
                          default: 'vivid',
                          description: 'The style of the generated image (DALL-E 3 only)',
                      },
                      {
                          name: 'response_format',
                          type: OptionType.enum,
                          enum: { URL: 'url', 'Base64 JSON': 'b64_json' },
                          default: 'b64_json',
                          description: 'The format of the response',
                      },
                  ]
                : [];

        const nImagesOption: ModelOptionInfoItem[] = isDallE2
            ? [
                  {
                      name: 'n',
                      type: OptionType.numeric,
                      min: 1,
                      max: 10,
                      default: 1,
                      integer: true,
                      description: 'Number of images to generate (DALL-E 2 only)',
                  },
              ]
            : [];

        return {
            _option_id: isGPTImage ? 'openai-gpt-image' : 'openai-dalle',
            options: [...baseImageOptions, ...gptImageOptions, ...dalleOptions, ...nImagesOption],
        };
    }

    if (isReasoningModel(model)) {
        //Is thinking text model
        let max_tokens_limit = 4096;
        if (model.includes('o1')) {
            if (model.includes('preview')) {
                max_tokens_limit = 32768;
            } else if (model.includes('mini')) {
                max_tokens_limit = 65536;
            } else {
                max_tokens_limit = 100000;
            }
        } else if (model.includes('o3')) {
            max_tokens_limit = 100000;
        } else if (model.includes('o4')) {
            max_tokens_limit = 100000;
        } else if (isOpenAIGptVersionGTE(model, 5, 0)) {
            max_tokens_limit = getMaxOutputTokens(model);
        }

        const commonOptions: ModelOptionInfoItem[] = [
            {
                name: SharedOptions.max_tokens,
                type: OptionType.numeric,
                min: 1,
                max: max_tokens_limit,
                integer: true,
                description: 'The maximum number of tokens to generate',
            },
            {
                name: SharedOptions.stop_sequence,
                type: OptionType.string_list,
                value: [],
                description: 'The stop sequence of the generated image',
            },
        ];

        const gptEffortLevels = getOpenAIReasoningEffortLevels(model);
        const reasoningOptions: ModelOptionInfoItem[] =
            gptEffortLevels || model.includes('o3') || model.includes('o4') || isO1Full(model)
                ? [
                      {
                          name: SharedOptions.effort,
                          type: OptionType.enum,
                          enum: gptEffortLevels ?? { Low: 'low', Medium: 'medium', High: 'high' },
                          description:
                              'How much effort the model should put into reasoning, lower values result in faster responses and less tokens used.',
                      },
                  ]
                : [];

        return {
            _option_id: 'openai-thinking',
            options: [...commonOptions, ...reasoningOptions, ...visionOptions, ...serviceTierOptions],
        };
    } else {
        let max_tokens_limit = 4096;
        if (model.includes('gpt-4o')) {
            max_tokens_limit = 16384;
            if (model.includes('gpt-4o-2024-05-13') || model.includes('realtime')) {
                max_tokens_limit = 4096;
            }
        } else if (model.includes('gpt-4')) {
            if (model.includes('turbo')) {
                max_tokens_limit = 4096;
            } else {
                max_tokens_limit = 8192;
            }
        } else if (model.includes('gpt-3-5')) {
            max_tokens_limit = 4096;
        } else if (model.includes('gpt-5')) {
            max_tokens_limit = 128000;
        }

        //Is non-thinking text model
        const commonOptions: ModelOptionInfoItem[] = [
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
                name: 'temperature',
                type: OptionType.numeric,
                min: 0.0,
                max: 2.0,
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
                integer: false,
                step: 0.1,
                description: 'Limits token sampling to the cumulative probability of the top p tokens',
            },
            {
                name: 'presence_penalty',
                type: OptionType.numeric,
                min: -2.0,
                max: 2.0,
                integer: false,
                step: 0.1,
                description: 'Penalise tokens if they appear at least once in the text',
            },
            {
                name: 'frequency_penalty',
                type: OptionType.numeric,
                min: -2.0,
                max: 2.0,
                integer: false,
                step: 0.1,
                description: 'Penalise tokens based on their frequency in the text',
            },
            {
                name: SharedOptions.stop_sequence,
                type: OptionType.string_list,
                value: [],
                description: 'The generation will halt if one of the stop sequences is output',
            },
        ];

        return {
            _option_id: 'openai-text',
            options: [...commonOptions, ...visionOptions, ...serviceTierOptions],
        };
    }
}

/** Azure OpenAI currently documents request-level Default and Priority processing, selected through auto. */
export function getAzureOpenAiOptions(model: string, option?: ModelOptions): ModelOptionsInfo {
    const options = getOpenAiOptions(model, option);
    return {
        ...options,
        options: options.options.map((item) =>
            item.name === 'service_tier' && item.type === OptionType.enum
                ? {
                      ...item,
                      enum: { Auto: 'auto', Default: 'default', Priority: 'priority' },
                      description: 'Select the Azure OpenAI processing tier for this request.',
                  }
                : item,
        ),
    };
}

/** OpenAI-compatible endpoints own model capability detection, so expose effort for any text model. */
export function getOpenAiCompatibleOptions(model: string, option?: ModelOptions): ModelOptionsInfo {
    const options = getOpenAiOptions(model, option);
    const compatibleOptions = options.options.filter((item) => item.name !== 'service_tier');
    if (compatibleOptions.some((item) => item.name === SharedOptions.effort) || options._option_id !== 'openai-text') {
        return { ...options, options: compatibleOptions };
    }
    return {
        ...options,
        options: [
            ...compatibleOptions,
            {
                name: SharedOptions.effort,
                type: OptionType.enum,
                enum: {
                    None: 'none',
                    Minimal: 'minimal',
                    Low: 'low',
                    Medium: 'medium',
                    High: 'high',
                    XHigh: 'xhigh',
                    Max: 'max',
                },
                description: 'How much effort the model should put into reasoning, when supported by the endpoint.',
            },
        ],
    };
}

function isO1Full(model: string): boolean {
    if (model.includes('o1')) {
        if (model.includes('mini') || model.includes('preview')) {
            return false;
        }
        return true;
    }
    return false;
}

function isReasoningModel(model: string): boolean {
    const normalized = model.toLowerCase();
    return (
        normalized.includes('o1') ||
        normalized.includes('o3') ||
        normalized.includes('o4') ||
        isOpenAIGptVersionGTE(model, 5, 0)
    );
}

function isVisionModel(model: string): boolean {
    return model.includes('gpt-4o') || isO1Full(model) || model.includes('gpt-4-turbo');
}

function isImageModel(model: string): boolean {
    return model.includes('dall-e') || model.includes('gpt-image') || model.includes('chatgpt-image');
}
