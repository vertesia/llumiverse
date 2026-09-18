import { describe, expect, it } from 'vitest';
import type { z } from 'zod';
import { getOptions } from '../options.js';
import { ModelOptionsSchema } from '../schemas/model-options.js';
import { type ModelOptionInfoItem, type ModelOptions, OptionType, Providers } from '../types.js';
import { ImagenTaskType } from './vertexai.js';

// Representative routing boundaries, not a model catalog. Future models inherit family rules.
const routes: Record<Providers, readonly (readonly [string, ModelOptions['_option_id']])[]> = {
    anthropic: [
        ['claude-3-7-sonnet', 'anthropic-claude'],
        ['claude-sonnet-4-6', 'anthropic-claude'],
        ['claude-opus-4-7', 'anthropic-claude'],
        ['claude-fable-5', 'anthropic-claude'],
    ],
    openai: [
        ['gpt-4.1', 'openai-text'],
        ['gpt-5.6-sol', 'openai-thinking'],
        ['o3', 'openai-thinking'],
        ['dall-e-2', 'openai-dalle'],
        ['dall-e-3', 'openai-dalle'],
        ['gpt-image-1', 'openai-gpt-image'],
    ],
    openrouter: [['openai/gpt-4o', 'openrouter-text']],
    azure_openai: [['deployment::gpt-5.6-sol', 'openai-thinking']],
    openai_compatible: [
        ['custom-model', 'openai-text'],
        ['openai/gpt-oss-120b', 'openai-text'],
    ],
    azure_foundry: [
        ['deployment::gpt-5.6-sol', 'openai-thinking'],
        ['DeepSeek-R1', 'azure-foundry-chat'],
    ],
    bedrock: [
        ['amazon.nova-pro-v1:0', 'bedrock-nova'],
        ['amazon.titan-text-premier-v1:0', 'bedrock-converse'],
        ['mistral.mixtral-8x7b-instruct-v0:1', 'bedrock-mistral'],
        ['mistral.mistral-large-2407-v1:0', 'bedrock-converse'],
        ['ai21.j2-ultra-v1', 'bedrock-ai21'],
        ['ai21.jamba-1-5-large-v1:0', 'bedrock-ai21'],
        ['cohere.command-r-v1:0', 'bedrock-cohere-command'],
        ['cohere.command-text-v14', 'bedrock-cohere-command'],
        ['anthropic.claude-sonnet-4-6-v1:0', 'bedrock-claude'],
        ['anthropic.claude-opus-4-7-v1:0', 'bedrock-claude'],
        ['writer.palmyra-x5-v1:0', 'bedrock-palmyra'],
        ['openai.gpt-oss-120b-1:0', 'bedrock-gpt-oss'],
        ['twelvelabs.pegasus-1-2-v1:0', 'bedrock-twelvelabs-pegasus'],
        ['amazon.nova-canvas-v1:0', 'bedrock-nova-canvas'],
    ],
    bedrock_mantle: [
        ['anthropic.claude-sonnet-4-6', 'bedrock-mantle-claude'],
        ['openai.gpt-5.6', 'bedrock-mantle-responses'],
        ['openai.gpt-oss-120b', 'bedrock-mantle-chat-completions'],
        ['xai.grok-4', 'bedrock-mantle-responses'],
        ['google.gemma-4-27b', 'bedrock-mantle-responses'],
        ['mistral.mistral-large', 'bedrock-mantle-chat-completions'],
    ],
    vertexai: [
        ['imagen-3.0-generate-002', 'vertexai-imagen'],
        ['imagen-3.0-capability-001', 'vertexai-imagen'],
        ['gemini-2.5-flash', 'vertexai-gemini'],
        ['gemini-3.5-flash', 'vertexai-gemini'],
        ['gemini-3.1-flash-image', 'vertexai-gemini'],
        ['gemini-omni-1.1-flash-preview', 'vertexai-gemini-omni-video'],
        ['claude-sonnet-4-6', 'vertexai-claude'],
        ['grok-4', 'openai-text'],
        ['qwen4-instruct-maas', 'openai-text'],
    ],
    togetherai: [['meta-llama/Llama-4-Maverick', 'openai-text']],
    mistralai: [['mistral-small-latest', 'mistral-text']],
    groq: [
        ['deepseek-r1-distill-llama-70b', 'groq-deepseek-thinking'],
        ['llama-3.3-70b-versatile', 'text-fallback'],
    ],
    xai: [
        ['grok-4', 'openai-text'],
        ['grok-imagine-image', 'xai-grok-image'],
    ],
    // These transports deliberately retain the generic option surface.
    huggingface_ie: [['custom-model', 'text-fallback']],
    replicate: [['custom-model', 'text-fallback']],
    watsonx: [['custom-model', 'text-fallback']],
};

function sampleValues(option: ModelOptionInfoItem): unknown[] {
    switch (option.type) {
        case OptionType.enum:
            return Object.values(option.enum);
        case OptionType.boolean:
            return [false, true];
        case OptionType.numeric:
            return [option.min ?? option.default ?? 1, option.max].filter((v) => v !== undefined);
        case OptionType.string_list:
            return [option.name === 'provider_quantizations' ? ['int4'] : ['example']];
        case OptionType.numeric_list:
            return [[1]];
        case OptionType.json_object:
            return [{ provider_extension: { enabled: true } }];
    }
}

describe('option factory contracts', () => {
    it('allows fractional Imagen mask dilation', () => {
        const info = getOptions('imagen-3.0-capability-001', Providers.vertexai, {
            _option_id: 'vertexai-imagen',
            edit_mode: ImagenTaskType.EDIT_MODE_INPAINT_INSERTION,
        });
        const dilation = info.options.find((option) => option.name === 'mask_dilation');
        expect(dilation?.type).toBe(OptionType.numeric);
        if (dilation?.type !== OptionType.numeric) throw new Error('Expected numeric dilation metadata');
        expect(dilation.integer).not.toBe(true);
        expect(ModelOptionsSchema.safeParse({ _option_id: 'vertexai-imagen', mask_dilation: 0.01 }).success).toBe(true);
    });

    it('exercises every registered option family', () => {
        const exercised = new Set(
            Object.values(routes)
                .flat()
                .map(([, id]) => id as string),
        );
        // Kept for saved payloads; Vertex Grok now routes through the OpenAI-compatible surface.
        exercised.add('vertexai-grok');
        expect([...exercised].sort()).toEqual(ModelOptionsSchema.options.map((s) => s.shape._option_id.value).sort());
        expect(ModelOptionsSchema.safeParse({ _option_id: 'vertexai-grok', max_tokens: 100 }).success).toBe(true);
    });

    for (const provider of Object.values(Providers)) {
        it.each(routes[provider])(`${provider} / %s routes to %s and exposes valid metadata`, (model, id) => {
            const schema = ModelOptionsSchema.options.find((s) => s.shape._option_id.value === id);
            if (!schema) throw new Error(`Missing schema for ${id}`);
            const fields: Record<string, z.ZodType> = schema.shape;
            const states: (ModelOptions | undefined)[] = [undefined];
            const expanded = new Set<string>();
            for (const state of states) {
                const info = getOptions(model, provider, state);
                expect(info._option_id).toBe(id);
                const defaults = Object.fromEntries(
                    info.options.filter((o) => o.default !== undefined).map((o) => [o.name, o.default]),
                );
                const payload = { _option_id: id, ...defaults, ...state };
                expect(ModelOptionsSchema.safeParse(payload).success, JSON.stringify(payload)).toBe(true);
                for (const option of info.options) {
                    const field = fields[option.name];
                    expect(field, option.name).toBeDefined();
                    for (const value of [option.default, option.value, ...sampleValues(option)]) {
                        if (value === undefined) continue;
                        expect(field.safeParse(value).success, `${option.name}: ${JSON.stringify(value)}`).toBe(true);
                    }
                    if (option.type !== OptionType.enum && option.type !== OptionType.boolean) continue;
                    for (const value of sampleValues(option)) {
                        const key = `${option.name}:${value}`;
                        if (expanded.has(key)) continue;
                        expanded.add(key);
                        states.push({ ...payload, [option.name]: value } as ModelOptions);
                    }
                }
            }
        });
    }
});
