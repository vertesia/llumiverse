import type { z } from 'zod';
import type { GroqOptionsSchema } from '../schemas/model-options.js';
import {
    type ModelOptionInfoItem,
    type ModelOptions,
    type ModelOptionsInfo,
    OptionType,
    SharedOptions,
} from '../types.js';
import { textOptionsFallback } from './fallback.js';

// The option shapes are DERIVED, not declared. Each schema in `../schemas/model-options.js` is the
// single definition of its option set: it is what the OpenAPI document publishes, what AJV enforces,
// and — through `z.infer` below — what TypeScript sees. There is nothing here to keep in step with
// anything, because there is only one statement of the shape.
//
// The OpenAPI scanner short-circuits these aliases to the component of the same name rather than
// trying to expand `z.infer`, which it cannot do. That is why the alias name, the schema variable
// and the published component id must all agree; generation fails loudly if they do not.
export type GroqOptions = z.infer<typeof GroqOptionsSchema>;

/** The only member of the Groq union today; kept as its own name because the driver narrows to it. */
export type GroqDeepseekThinkingOptions = GroqOptions;

export function getGroqOptions(model: string, _option?: ModelOptions): ModelOptionsInfo {
    if (model.includes('deepseek') && model.includes('r1')) {
        const commonOptions: ModelOptionInfoItem[] = [
            {
                name: SharedOptions.max_tokens,
                type: OptionType.numeric,
                min: 1,
                max: 131072,
                integer: true,
                description: 'The maximum number of tokens to generate',
            },
            {
                name: SharedOptions.temperature,
                type: OptionType.numeric,
                min: 0.0,
                default: 0.7,
                max: 2.0,
                integer: false,
                step: 0.1,
                description:
                    'A higher temperature biases toward less likely tokens, making the model more creative. A lower temperature than other models is recommended for deepseek R1, 0.3-0.7 approximately.',
            },
            {
                name: SharedOptions.top_p,
                type: OptionType.numeric,
                min: 0,
                max: 1,
                integer: false,
                step: 0.1,
                description: 'Limits token sampling to the cumulative probability of the top p tokens',
            },
            {
                name: SharedOptions.stop_sequence,
                type: OptionType.string_list,
                value: [],
                description: 'The generation will halt if one of the stop sequences is output',
            },
            {
                name: 'reasoning_format',
                type: OptionType.enum,
                enum: { Parsed: 'parsed', Raw: 'raw', Hidden: 'hidden' },
                default: 'parsed',
                description: 'Controls how the reasoning is returned.',
            },
        ];

        return {
            _option_id: 'groq-deepseek-thinking',
            options: commonOptions,
        };
    }
    return textOptionsFallback;
}
