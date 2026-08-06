import type { z } from 'zod';
import type { TextFallbackOptionsSchema } from '../schemas/model-options.js';
import { type ModelOptionsInfo, OptionType, SharedOptions } from '../types.js';

// The option shapes are DERIVED, not declared. Each schema in `../schemas/model-options.js` is the
// single definition of its option set: it is what the OpenAPI document publishes, what AJV enforces,
// and — through `z.infer` below — what TypeScript sees. There is nothing here to keep in step with
// anything, because there is only one statement of the shape.
//
// The OpenAPI scanner short-circuits these aliases to the component of the same name rather than
// trying to expand `z.infer`, which it cannot do. That is why the alias name, the schema variable
// and the published component id must all agree; generation fails loudly if they do not.
export type TextFallbackOptions = z.infer<typeof TextFallbackOptionsSchema>;

export const textOptionsFallback: ModelOptionsInfo = {
    _option_id: 'text-fallback',
    options: [
        {
            name: SharedOptions.max_tokens,
            type: OptionType.numeric,
            min: 1,
            integer: true,
            step: 200,
            description: 'The maximum number of tokens to generate',
        },
        {
            name: SharedOptions.temperature,
            type: OptionType.numeric,
            min: 0.0,
            default: 0.7,
            integer: false,
            step: 0.1,
            description: 'A higher temperature biases toward less likely tokens, making the model more creative',
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
            name: SharedOptions.top_k,
            type: OptionType.numeric,
            min: 1,
            integer: true,
            step: 1,
            description: 'Limits token sampling to the top k tokens',
        },
        {
            name: SharedOptions.presence_penalty,
            type: OptionType.numeric,
            min: -2.0,
            max: 2.0,
            integer: false,
            step: 0.1,
            description: 'Penalise tokens if they appear at least once in the text',
        },
        {
            name: SharedOptions.frequency_penalty,
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
        {
            name: 'include_thoughts',
            type: OptionType.boolean,
            default: true,
            description: 'Include visible model reasoning as separate thoughts results.',
        },
    ],
};
