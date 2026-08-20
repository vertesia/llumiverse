import type { z } from 'zod';
import { type ModelProfile, resolveModelProfile } from '../model-directory.js';
import type { XAIGrokImageOptionsSchema } from '../schemas/model-options.js';
import { type ModelOptionInfoItem, type ModelOptions, type ModelOptionsInfo, OptionType, Providers } from '../types.js';
import { getOpenAiCompatibleOptions } from './openai.js';

export type XAIGrokImageOptions = z.infer<typeof XAIGrokImageOptionsSchema>;

export function isXAIGrokImageModel(model: string): boolean {
    const modelId = (model.split('/').pop() ?? model).toLowerCase();
    return modelId.includes('grok') && modelId.includes('image');
}

export function getXAIOptions(
    model: string,
    options?: ModelOptions,
    profile: ModelProfile = resolveModelProfile(model, Providers.xai),
): ModelOptionsInfo {
    if (!isXAIGrokImageModel(model)) {
        return getOpenAiCompatibleOptions(model, options, profile);
    }

    const imageOptions: ModelOptionInfoItem[] = [
        {
            name: 'aspect_ratio',
            type: OptionType.enum,
            enum: {
                Auto: 'auto',
                Square: '1:1',
                Widescreen: '16:9',
                Portrait: '9:16',
                Landscape: '4:3',
                'Portrait 3:4': '3:4',
                Photo: '3:2',
                'Portrait 2:3': '2:3',
                Banner: '2:1',
                'Portrait 1:2': '1:2',
                Smartphone: '19.5:9',
                'Portrait 9:19.5': '9:19.5',
                Ultrawide: '20:9',
                'Portrait 9:20': '9:20',
            },
            default: 'auto',
            description: 'The aspect ratio of generated images.',
        },
        {
            name: 'resolution',
            type: OptionType.enum,
            enum: { '1K': '1k', '2K': '2k' },
            default: '1k',
            description: 'The output resolution of generated images.',
        },
        ...(model.split('/').pop()?.toLowerCase() === 'grok-imagine-image-2.0'
            ? [
                  {
                      name: 'quality',
                      type: OptionType.enum,
                      enum: { Low: 'low', Medium: 'medium' },
                      default: 'medium',
                      description: 'The generation quality. Supported by Grok Imagine Image 2.0.',
                  } satisfies ModelOptionInfoItem,
              ]
            : []),
        {
            name: 'response_format',
            type: OptionType.enum,
            enum: { URL: 'url', 'Base64 JSON': 'b64_json' },
            default: 'url',
            description: 'The format used to return generated images.',
        },
        {
            name: 'n',
            type: OptionType.numeric,
            min: 1,
            max: 10,
            default: 1,
            integer: true,
            description: 'The number of images to generate.',
        },
    ];

    return { _option_id: 'xai-grok-image', options: imageOptions };
}
