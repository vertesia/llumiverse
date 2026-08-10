import type { z } from 'zod';
import { type ModelProfile, resolveModelProfile } from '../model-directory.js';
import type { MistralTextOptionsSchema } from '../schemas/model-options.js';
import {
    type ModelOptionInfoItem,
    type ModelOptions,
    type ModelOptionsInfo,
    OptionType,
    Providers,
    SharedOptions,
} from '../types.js';
import { getOpenAiCompatibleOptions } from './openai.js';

export type MistralTextOptions = z.infer<typeof MistralTextOptionsSchema>;

export function getMistralOptions(
    model: string,
    options?: ModelOptions,
    profile: ModelProfile = resolveModelProfile(model, Providers.mistralai),
): ModelOptionsInfo {
    const compatible = getOpenAiCompatibleOptions(model, options, profile);
    const supportsReasoning = profile.reasoning_effort_levels?.length;
    const compatibleOptions = compatible.options.map((item): ModelOptionInfoItem => {
        if (item.name !== SharedOptions.max_tokens || profile.max_output_tokens !== undefined) return item;
        const { max: _unverifiedMax, ...withoutMax } = item as ModelOptionInfoItem & { max?: number };
        return withoutMax as ModelOptionInfoItem;
    });
    const mistralOptions: ModelOptionInfoItem[] = [
        ...compatibleOptions,
        {
            name: 'random_seed',
            type: OptionType.numeric,
            integer: true,
            description: 'Seed used for deterministic random sampling.',
        },
        {
            name: 'safe_prompt',
            type: OptionType.boolean,
            default: false,
            description: 'Inject the Mistral safety prompt before the conversation.',
        },
        ...(profile.capabilities.tool_support
            ? [
                  {
                      name: 'parallel_tool_calls',
                      type: OptionType.boolean,
                      default: true,
                      description: 'Allow the model to request multiple tool calls in parallel.',
                  } satisfies ModelOptionInfoItem,
                  {
                      name: 'tool_choice',
                      type: OptionType.enum,
                      enum: { Auto: 'auto', None: 'none', Any: 'any', Required: 'required' },
                      default: 'auto',
                      description: 'Control whether the model may or must call a supplied tool.',
                  } satisfies ModelOptionInfoItem,
              ]
            : []),
        ...(supportsReasoning || model.toLowerCase().includes('magistral')
            ? [
                  {
                      name: 'prompt_mode',
                      type: OptionType.enum,
                      enum: { Reasoning: 'reasoning' },
                      description: 'Apply Mistral reasoning prompt instructions.',
                  } satisfies ModelOptionInfoItem,
                  {
                      name: 'include_thoughts',
                      type: OptionType.boolean,
                      default: true,
                      description: 'Include visible model reasoning as separate thoughts results.',
                  } satisfies ModelOptionInfoItem,
              ]
            : []),
    ];

    // The discriminator only carries native controls this execution abstraction can faithfully represent. Streaming,
    // tools, response schemas and prompt caching are already execution-level fields; n remains fixed at one because
    // Completion represents a single choice.
    return {
        _option_id: 'mistral-text',
        options: mistralOptions,
    };
}
