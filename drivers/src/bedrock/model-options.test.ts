import type { ModelOptions } from '@llumiverse/common';
import { describe, expect, it } from 'vitest';
import { BedrockDriver } from './index.js';

describe('Bedrock provider-specific model options', () => {
    it('serializes Claude between-tools mode through Converse reasoning_config', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const model = 'anthropic.claude-sonnet-5-5';
        const request = driver.preparePayload(
            { modelId: model, messages: [{ role: 'user', content: [{ text: 'hello' }] }] },
            {
                model,
                model_options: {
                    _option_id: 'bedrock-claude',
                    thinking_mode: 'between_tools',
                    effort: 'high',
                    include_thoughts: true,
                    thinking_budget_tokens: 8000,
                },
            },
        );
        expect(request.additionalModelRequestFields).toEqual({
            reasoning_config: { type: 'between_tools' },
            output_config: { effort: 'high' },
        });
    });

    it.each([
        ['amazon.nova-pro-v1:0', { _option_id: 'bedrock-nova', top_k: 12 }, { inferenceConfig: { topK: 12 } }],
        ['mistral.mixtral-8x7b-instruct-v0:1', { _option_id: 'bedrock-mistral', top_k: 12 }, { top_k: 12 }],
        ['cohere.command-text-v14', { _option_id: 'bedrock-cohere-command', top_k: 12 }, { k: 12 }],
        [
            'cohere.command-r-v1:0',
            { _option_id: 'bedrock-cohere-command', top_k: 12, presence_penalty: 0.4, frequency_penalty: 0.2 },
            { k: 12, presence_penalty: 0.4, frequency_penalty: 0.2 },
        ],
        [
            'ai21.jamba-1-5-large-v1:0',
            { _option_id: 'bedrock-ai21', presence_penalty: 0.4, frequency_penalty: 0.2 },
            { presence_penalty: 0.4, frequency_penalty: 0.2 },
        ],
        [
            'ai21.j2-ultra-v1',
            { _option_id: 'bedrock-ai21', presence_penalty: 0.4, frequency_penalty: 0.2 },
            { presencePenalty: { scale: 0.4 }, frequencyPenalty: { scale: 0.2 } },
        ],
    ] satisfies [string, ModelOptions, object][])(
        'serializes supported fields for %s',
        (model, model_options, expected) => {
            const driver = new BedrockDriver({ region: 'us-east-1' });
            const request = driver.preparePayload(
                { modelId: model, messages: [{ role: 'user', content: [{ text: 'hello' }] }] },
                { model, model_options },
            );
            expect(request.additionalModelRequestFields).toEqual(expected);
        },
    );
});

describe('Bedrock Converse closed-weight GPT options', () => {
    it.each(['openai.gpt-5.5', 'us.openai.gpt-5.6-sol', 'global.openai.gpt-6-luna', 'us.openai.gpt-6.1-sol'])(
        'transports effort and strict output together for %s',
        (model) => {
            const driver = new BedrockDriver({ region: 'us-east-1' });
            const result_schema = {
                type: 'object' as const,
                properties: { answer: { type: 'string' as const } },
                required: ['answer'],
                additionalProperties: false,
            };
            const request = driver.preparePayload(
                { modelId: model, messages: [{ role: 'user', content: [{ text: 'hello' }] }] },
                {
                    model,
                    result_schema,
                    model_options: {
                        _option_id: 'bedrock-converse',
                        effort: 'high',
                        reasoning_effort: 'low',
                        verbosity: 'low',
                        temperature: 0.7,
                        top_p: 0.9,
                        stop_sequence: ['STOP'],
                        max_tokens: 100,
                    },
                },
            );
            expect(request.modelId).toBe(model);
            expect(request.inferenceConfig).toEqual({ maxTokens: 100 });
            expect(request.additionalModelRequestFields).toEqual({
                reasoning: { effort: 'high' },
                text: { verbosity: 'low', format: { strict: true } },
            });
            expect(request.outputConfig?.textFormat?.structure?.jsonSchema?.schema).toBe(JSON.stringify(result_schema));
        },
    );

    it.each(['none', 'max'] as const)('preserves the reasoning_effort alias value %s', (reasoning_effort) => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const request = driver.preparePayload(
            { modelId: undefined, messages: [] },
            {
                model: 'us.openai.gpt-6-luna',
                model_options: { _option_id: 'bedrock-converse', reasoning_effort },
            },
        );
        expect(request.additionalModelRequestFields).toMatchObject({ reasoning: { effort: reasoning_effort } });
    });

    it('leaves reasoning to provider defaults when effort is unset', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const request = driver.preparePayload({ modelId: undefined, messages: [] }, { model: 'us.openai.gpt-6.1-sol' });
        expect(request.additionalModelRequestFields).toBeUndefined();
    });
});
