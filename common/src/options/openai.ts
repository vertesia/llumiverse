import type { z } from 'zod';
import { type ModelProfile, resolveModelProfile } from '../model-directory.js';
import type {
    OpenAiAudioOptionsSchema,
    OpenAiDalleOptionsSchema,
    OpenAiGptImageOptionsSchema,
    OpenAiImageGenerationMaskSchema,
    OpenAiImageGenerationOptionsSchema,
    OpenAiImageSvgOptionsSchema,
    OpenAiSpeechOptionsSchema,
    OpenAiTextOptionsSchema,
    OpenAiThinkingOptionsSchema,
    OpenAiTranscriptionOptionsSchema,
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
import { buildNativeCompactionOption } from './shared-parsing.js';
import { getOpenAIReasoningEffortLevels, isOpenAIGptVersionGTE, isOpenAIImageVersionGTE } from './version-parsing.js';

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
export type OpenAiImageGenerationOptions = z.infer<typeof OpenAiImageGenerationOptionsSchema>;
export type OpenAiImageGenerationMask = z.infer<typeof OpenAiImageGenerationMaskSchema>;
export type OpenAiImageSvgOptions = z.infer<typeof OpenAiImageSvgOptionsSchema>;
export type OpenAiGptImageOptions = z.infer<typeof OpenAiGptImageOptionsSchema>;

export type OpenAiTranscriptionOptions = z.infer<typeof OpenAiTranscriptionOptionsSchema>;
export type OpenAiSpeechOptions = z.infer<typeof OpenAiSpeechOptionsSchema>;
export type OpenAiAudioOptions = z.infer<typeof OpenAiAudioOptionsSchema>;

// Union type of all OpenAI options
/**
 * @discriminator _option_id
 */
export type OpenAiOptions =
    | OpenAiThinkingOptions
    | OpenAiTextOptions
    | OpenAiDalleOptions
    | OpenAiGptImageOptions
    | OpenAiTranscriptionOptions
    | OpenAiSpeechOptions
    | OpenAiAudioOptions;

/** A prompt-cache lifetime: `30m`/`12h` for GPT-5.6 and later, `in_memory`/`24h` for earlier models. */
export type OpenAIPromptCacheTtl = NonNullable<OpenAiThinkingOptions['cache_ttl']>;

/**
 * Earlier models that OpenAI documents as accepting extended (`24h`) prompt-cache retention as well as `in_memory`.
 * GPT-5.5 and GPT-5.5 Pro accept `24h` only, so they offer no choice and are not listed.
 * https://developers.openai.com/api/docs/guides/prompt-caching
 */
const OPENAI_EXTENDED_CACHE_RETENTION_MODELS: ReadonlySet<string> = new Set([
    'gpt-5.4',
    'gpt-5.2',
    'gpt-5.1-codex-max',
    'gpt-5.1',
    'gpt-5.1-codex',
    'gpt-5.1-codex-mini',
    'gpt-5.1-chat-latest',
    'gpt-5',
    'gpt-5-codex',
    'gpt-4.1',
]);

/**
 * The prompt-cache lifetimes the OpenAI API accepts for a model, or an empty list when the model offers no choice.
 * GPT-5.6 and later take `prompt_cache_options.ttl`; the earlier models take the legacy `prompt_cache_retention`.
 */
export function getOpenAIPromptCacheTtls(model: string): readonly OpenAIPromptCacheTtl[] {
    if (isOpenAIGptVersionGTE(model, 5, 6)) return ['30m', '12h'];
    const modelName = (model.split('/').pop() ?? model)
        .toLowerCase()
        .replace(/^openai\./, '')
        .replace(/-\d{4}-\d{2}-\d{2}$/, '');
    return OPENAI_EXTENDED_CACHE_RETENTION_MODELS.has(modelName) ? ['in_memory', '24h'] : [];
}

function getOpenAIPromptCacheOptions(model: string, provider: Providers): ModelOptionInfoItem[] {
    // The lifetimes are documented for the OpenAI API only; Azure and compatible endpoints keep their defaults.
    if (provider !== Providers.openai) return [];
    const ttls = getOpenAIPromptCacheTtls(model);
    if (ttls.includes('12h')) {
        return [
            {
                name: 'cache_ttl',
                type: OptionType.enum,
                enum: { '30 minutes (default)': '30m', '12 hours': '12h' },
                default: '30m',
                description:
                    'Minimum lifetime of prompt-cache entries. Cache writes cost 1.25× the input rate for 30 minutes ' +
                    'and 2.5× for 12 hours; cache reads cost the same for both.',
            },
        ];
    }
    if (ttls.includes('24h')) {
        return [
            {
                name: 'cache_ttl',
                type: OptionType.enum,
                enum: { 'In memory': 'in_memory', '24 hours': '24h' },
                description:
                    'Prompt-cache retention. Leave unset for the organization default: 24 hours, or in memory with ' +
                    'Zero Data Retention. Cache writes have no extra charge on these models.',
            },
        ];
    }
    return [];
}

/** OpenAI model families with published Flex processing support. */
export function isFlexSupportedOpenAIModel(model: string): boolean {
    const modelName = (model.split('/').pop() ?? model).toLowerCase();
    const unsupportedVariant = ['chat', 'codex', 'pro', 'deep-research'].some((variant) => modelName.includes(variant));
    if (unsupportedVariant) return false;

    return (
        isOpenAIGptVersionGTE(modelName, 5, 0) ||
        /^o3(?:-\d{4}|$)/.test(modelName) ||
        /^o4-mini(?:-\d{4}|$)/.test(modelName)
    );
}

export function getOpenAiOptions(
    model: string,
    _option?: ModelOptions,
    profile: ModelProfile = resolveModelProfile(model, Providers.openai),
    provider: Providers = Providers.openai,
): ModelOptionsInfo {
    // Option matching follows the resolved source ID so provider/path-qualified and uppercase IDs expose the same
    // controls as their canonical model.
    model = profile.canonical_id;
    if (profile.family === 'transcription') {
        return { _option_id: 'openai-transcription', options: [] };
    }
    if (profile.family === 'speech') {
        return {
            _option_id: 'openai-speech',
            options: [
                {
                    name: 'voice',
                    type: OptionType.enum,
                    enum: { Alloy: 'alloy', Coral: 'coral', Nova: 'nova', Onyx: 'onyx', Shimmer: 'shimmer' },
                    default: 'alloy',
                    description: 'Provider voice name; additional voices can be supplied through the API.',
                },
                {
                    name: 'response_format',
                    type: OptionType.enum,
                    enum: { MP3: 'mp3', WAV: 'wav', Opus: 'opus', AAC: 'aac', FLAC: 'flac', PCM: 'pcm' },
                    default: 'mp3',
                },
                { name: 'speed', type: OptionType.numeric, min: 0.25, max: 4, default: 1 },
            ],
        };
    }
    if (profile.family === 'audio') {
        return {
            _option_id: 'openai-audio',
            options: [
                {
                    name: 'voice',
                    type: OptionType.enum,
                    enum: {
                        Alloy: 'alloy',
                        Ash: 'ash',
                        Coral: 'coral',
                        Marin: 'marin',
                        Sage: 'sage',
                        Shimmer: 'shimmer',
                    },
                    default: 'alloy',
                    description: 'Provider voice name; additional voices can be supplied through the API.',
                },
                {
                    name: 'response_format',
                    type: OptionType.enum,
                    enum: { WAV: 'wav', MP3: 'mp3', FLAC: 'flac', Opus: 'opus', PCM16: 'pcm16' },
                    default: 'wav',
                },
            ],
        };
    }
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
    const serviceTiers: Record<string, string> = {
        Auto: 'auto',
        Default: 'default',
        Priority: 'priority',
    };
    if (provider === Providers.openai && isOpenAIGptVersionGTE(model, 6, 0)) {
        serviceTiers.Fast = 'fast';
        if (model.includes('astra') || (model.includes('sol') && isOpenAIGptVersionGTE(model, 6, 1))) {
            serviceTiers.Ultrafast = 'ultrafast';
        }
    }
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
        // GPT Image 2.6 adds planning complexity and GIF/SVG outputs; later generations inherit them. Azure OpenAI
        // documents only PNG, JPEG and WebP, so it does not advertise them; explicit values are still forwarded.
        const hasGptImage26Controls = provider !== Providers.azure_openai && isOpenAIImageVersionGTE(model, 2, 6);

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

        const baseImageOptions: ModelOptionInfoItem[] = isOpenAIImageVersionGTE(model, 2)
            ? ['width', 'height'].map((name) => ({
                  name,
                  type: OptionType.numeric,
                  integer: true,
                  description:
                      `The ${name} of the generated image in pixels. Set both width and height, or leave both ` +
                      'empty to let the model choose the size.',
              }))
            : [
                  {
                      name: 'size',
                      type: OptionType.enum,
                      enum: sizeOptions,
                      default: isGPTImage ? 'auto' : '1024x1024',
                      description: 'The size of the generated image',
                  },
              ];

        const gptImageOptions: ModelOptionInfoItem[] = isGPTImage
            ? [
                  {
                      name: 'image_quality',
                      type: OptionType.enum,
                      enum: {
                          Low: 'low',
                          Medium: 'medium',
                          High: 'high',
                          Auto: 'auto',
                          ...(isOpenAIImageVersionGTE(model, 2, 5) ? { 'Extra high': 'xhigh', Max: 'max' } : {}),
                      },
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
                      enum: {
                          PNG: 'png',
                          WebP: 'webp',
                          JPEG: 'jpeg',
                          ...(hasGptImage26Controls ? { GIF: 'gif', SVG: 'svg' } : {}),
                      },
                      default: 'png',
                      description: hasGptImage26Controls
                          ? 'The output format for the image. GIF and SVG outputs do not support streaming.'
                          : 'The output format for the image',
                  },
                  ...(hasGptImage26Controls
                      ? ([
                            {
                                name: 'complexity',
                                type: OptionType.enum,
                                enum: { Low: 'low', Medium: 'medium', High: 'high' },
                                description:
                                    'Planning effort for the image, independent of rendering quality. ' +
                                    'The provider default applies when omitted.',
                            },
                            {
                                name: 'frame_count',
                                type: OptionType.numeric,
                                min: 2,
                                max: 16,
                                integer: true,
                                description: 'Number of GIF frames (GIF output only)',
                            },
                            {
                                name: 'frame_duration_ms',
                                type: OptionType.numeric,
                                min: 20,
                                integer: true,
                                description: 'Delay per GIF frame in milliseconds, rounded to 10 ms (GIF output only)',
                            },
                            {
                                name: 'svg',
                                type: OptionType.json_object,
                                description:
                                    'SVG author settings (SVG output only), e.g. {"model": "<model>", ' +
                                    '"reasoning": {"effort": "low"}}. Required when SVG model selection is enabled.',
                            },
                        ] satisfies ModelOptionInfoItem[])
                      : []),
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

        const nImagesOption: ModelOptionInfoItem[] =
            isDallE2 || isGPTImage
                ? [
                      {
                          name: 'n',
                          type: OptionType.numeric,
                          min: 1,
                          max: 10,
                          default: 1,
                          integer: true,
                          description: 'Number of images to generate',
                      },
                  ]
                : [];

        return {
            _option_id: isGPTImage ? 'openai-gpt-image' : 'openai-dalle',
            options: [
                ...baseImageOptions,
                ...gptImageOptions,
                ...dalleOptions,
                ...nImagesOption,
                ...(isGPTImage
                    ? [
                          {
                              name: 'output_compression',
                              type: OptionType.numeric as const,
                              min: 0,
                              max: 100,
                              integer: true,
                          },
                          { name: 'moderation', type: OptionType.enum as const, enum: { Auto: 'auto', Low: 'low' } },
                          {
                              name: 'partial_images',
                              type: OptionType.numeric as const,
                              min: 0,
                              max: 3,
                              integer: true,
                              default: 0,
                          },
                          ...(!isOpenAIImageVersionGTE(model, 2) || isOpenAIImageVersionGTE(model, 2, 5)
                              ? [
                                    {
                                        name: 'input_fidelity',
                                        type: OptionType.enum as const,
                                        enum: { Low: 'low', High: 'high' },
                                    },
                                ]
                              : []),
                      ]
                    : []),
            ],
        };
    }

    if (isReasoningModel(model)) {
        //Is thinking text model
        let max_tokens_limit: number | undefined = 4096;
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

        const reasoningContextOptions: ModelOptionInfoItem[] =
            provider === Providers.openai && isOpenAIGptVersionGTE(model, 5, 6)
                ? [
                      {
                          name: SharedOptions.reasoning_context,
                          type: OptionType.enum,
                          enum: { Auto: 'auto', 'Current turn': 'current_turn', 'All turns': 'all_turns' },
                          description:
                              'Controls whether compatible reasoning from earlier turns is rendered into the next request. Leave unset to use the model default.',
                      },
                  ]
                : [];

        return {
            _option_id: 'openai-thinking',
            options: [
                ...commonOptions.filter(
                    (item) => provider === Providers.openai_compatible || item.name !== SharedOptions.stop_sequence,
                ),
                ...reasoningOptions,
                ...reasoningContextOptions,
                ...(isOpenAIGptVersionGTE(model, 5, 0)
                    ? [
                          {
                              name: 'verbosity',
                              type: OptionType.enum as const,
                              enum: { Low: 'low', Medium: 'medium', High: 'high' },
                              description: 'Controls the detail of the final answer independently of reasoning effort.',
                          },
                      ]
                    : []),
                ...visionOptions,
                ...serviceTierOptions,
                ...getOpenAIPromptCacheOptions(model, provider),
                // The Responses drivers (OpenAI, Azure OpenAI) implement on-demand compaction; Chat Completions do not.
                ...(provider === Providers.openai_compatible ? [] : buildNativeCompactionOption()),
                {
                    name: 'image_generation',
                    type: OptionType.json_object,
                    description: 'Optional image generation tool configuration; requires an image model.',
                },
            ],
        };
    } else {
        let max_tokens_limit: number | undefined = 4096;
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
            options: [
                ...commonOptions.filter(
                    (item) =>
                        provider === Providers.openai_compatible ||
                        ![
                            SharedOptions.stop_sequence,
                            SharedOptions.presence_penalty,
                            SharedOptions.frequency_penalty,
                        ].includes(item.name as SharedOptions),
                ),
                ...visionOptions,
                ...serviceTierOptions,
                ...getOpenAIPromptCacheOptions(model, provider),
                // The Responses drivers (OpenAI, Azure OpenAI) implement on-demand compaction; Chat Completions do not.
                ...(provider === Providers.openai_compatible ? [] : buildNativeCompactionOption()),
                {
                    name: 'image_generation',
                    type: OptionType.json_object,
                    description: 'Optional image generation tool configuration; requires an image model.',
                },
            ],
        };
    }
}

/** Azure OpenAI currently documents request-level Default and Priority processing, selected through auto. */
export function getAzureOpenAiOptions(
    model: string,
    option?: ModelOptions,
    profile: ModelProfile = resolveModelProfile(model, Providers.azure_openai),
): ModelOptionsInfo {
    const options = getOpenAiOptions(model, option, profile, Providers.azure_openai);
    // Azure Flex availability is model-specific; do not inherit direct OpenAI tier support.
    // https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/flex-processing
    const supportsAzureFlex = /^gpt-(?:5\.6-(?:sol|luna|terra)|6-astra)(?:$|-\d{4}-\d{2}-\d{2}$)/i.test(
        profile.canonical_id,
    );
    return {
        ...options,
        options: options.options.map((item) =>
            item.name === 'service_tier' && item.type === OptionType.enum
                ? {
                      ...item,
                      enum: {
                          Auto: 'auto',
                          Default: 'default',
                          Priority: 'priority',
                          ...(supportsAzureFlex ? { Flex: 'flex' } : {}),
                      },
                      description: 'Select the Azure OpenAI processing tier for this request.',
                  }
                : item,
        ),
    };
}

/** Expose only effort values verified for the source model and this transport. */
export function getOpenAiCompatibleOptions(
    model: string,
    option?: ModelOptions,
    profile: ModelProfile = resolveModelProfile(model, Providers.openai_compatible),
): ModelOptionsInfo {
    const options = getOpenAiOptions(model, option, profile, Providers.openai_compatible);
    if (profile.family === 'speech' || profile.family === 'transcription') return options;
    const compatibleOptions = options.options.filter(
        (item) => item.name !== 'service_tier' && item.name !== 'image_generation',
    );
    const maxOutputTokens = profile.max_output_tokens;
    const profileEffortLevels = profile.reasoning_effort_levels?.length
        ? new Set(profile.reasoning_effort_levels)
        : undefined;
    const profileOptions: ModelOptionInfoItem[] = compatibleOptions
        .map((item): ModelOptionInfoItem | null => {
            if (item.name === SharedOptions.max_tokens && maxOutputTokens !== undefined) {
                return { ...item, max: maxOutputTokens } as ModelOptionInfoItem;
            }
            if (item.name === SharedOptions.max_tokens && profile.context_window !== undefined) {
                // A known model without a published output limit must not inherit the legacy GPT text-model cap.
                const { max: _legacyMax, ...withoutMax } = item as ModelOptionInfoItem & { max?: number };
                return withoutMax as ModelOptionInfoItem;
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
        return withCompatibleExtraBody({ ...options, options: profileOptions });
    }
    if (!profileEffortLevels) return withCompatibleExtraBody({ ...options, options: profileOptions });
    return withCompatibleExtraBody({
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
    });
}

function withCompatibleExtraBody(options: ModelOptionsInfo): ModelOptionsInfo {
    return {
        ...options,
        options: [
            ...options.options,
            {
                name: 'extra_body',
                type: OptionType.json_object,
                description:
                    'Additional provider-specific fields merged into the request body. Standard model options take precedence.',
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
