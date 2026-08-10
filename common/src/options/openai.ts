import type { z } from 'zod';
import { type ModelProfile, resolveModelProfile } from '../model-directory.js';
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
    Providers,
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

export function getOpenAiOptions(
    model: string,
    _option?: ModelOptions,
    profile: ModelProfile = resolveModelProfile(model, Providers.openai),
): ModelOptionsInfo {
    // Option matching follows the resolved source ID so provider/path-qualified and uppercase IDs expose the same
    // controls as their canonical model.
    model = profile.canonical_id;
    const visionOptions: ModelOptionInfoItem[] =
        profile.capabilities.input.image === true
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
        } else if (isOSeriesModel(model)) {
            max_tokens_limit = 100000;
        } else if (isOpenAIGptVersionGTE(model, 5, 0)) {
            max_tokens_limit = profile.max_output_tokens ?? getMaxOutputTokens(model);
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
            gptEffortLevels || isOSeriesModel(model)
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
            options: [...commonOptions, ...reasoningOptions, ...visionOptions],
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
        } else if (isOpenAIGptVersionGTE(model, 5, 0)) {
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
            options: [...commonOptions, ...visionOptions],
        };
    }
}

/** Expose only effort values verified for the source model and this transport. */
export function getOpenAiCompatibleOptions(
    model: string,
    option?: ModelOptions,
    profile: ModelProfile = resolveModelProfile(model, Providers.openai_compatible),
): ModelOptionsInfo {
    const options = getOpenAiOptions(model, option, profile);
    const maxOutputTokens = profile.max_output_tokens;
    const profileEffortLevels = profile.reasoning_effort_levels?.length
        ? new Set(profile.reasoning_effort_levels)
        : undefined;
    const profileOptions: ModelOptionInfoItem[] = options.options
        .map((item): ModelOptionInfoItem | null => {
            if (item.name === SharedOptions.max_tokens && maxOutputTokens !== undefined) {
                return { ...item, max: maxOutputTokens } as ModelOptionInfoItem;
            }
            if (item.name === SharedOptions.effort && item.type === OptionType.enum) {
                if (!profileEffortLevels) return null;
                const enumValues = item.enum as Record<string, string>;
                return {
                    ...item,
                    enum: Object.fromEntries(
                        Object.entries(enumValues).filter(([, value]) => profileEffortLevels.has(value)),
                    ) as Record<string, string>,
                } as ModelOptionInfoItem;
            }
            return item;
        })
        .filter((item): item is ModelOptionInfoItem => item !== null);
    if (profileOptions.some((item) => item.name === SharedOptions.effort) || options._option_id !== 'openai-text') {
        return { ...options, options: profileOptions };
    }
    if (!profileEffortLevels) return { ...options, options: profileOptions };
    return {
        ...options,
        options: [
            ...profileOptions,
            {
                name: SharedOptions.effort,
                type: OptionType.enum,
                enum: Object.fromEntries([...profileEffortLevels].map((value) => [value, value])),
                description: 'How much effort the model should put into reasoning, when supported by the endpoint.',
            },
        ],
    };
}

function isReasoningModel(model: string): boolean {
    return isOSeriesModel(model) || isOpenAIGptVersionGTE(model, 5, 0);
}

function isOSeriesModel(model: string): boolean {
    return /(?:^|[~/.])o\d+(?:[-_.]|$)/.test(model.toLowerCase());
}

function isImageModel(model: string): boolean {
    return model.includes('dall-e') || model.includes('gpt-image') || model.includes('chatgpt-image');
}
