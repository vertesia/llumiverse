import type { ModelOptions } from '@llumiverse/common';
import { describe, expect, it, vi } from 'vitest';
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

    it('asks Converse to drop mismatched thinking blocks for adaptive Claude Sonnet 5.5', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const model = 'us.anthropic.claude-sonnet-5-5';
        const request = driver.preparePayload(
            { modelId: model, messages: [{ role: 'user', content: [{ text: 'hello' }] }] },
            { model, model_options: { _option_id: 'bedrock-claude', effort: 'medium' } },
        );
        expect(request.additionalModelRequestFields).toEqual({
            reasoning_config: {
                type: 'adaptive',
                display: 'omitted',
                block_binding: { prefix_mismatch_behavior: 'drop_block' },
            },
            anthropic_beta: ['thinking-binding-controls-2026-08-01'],
            output_config: { effort: 'medium' },
        });
    });

    it('sends no thinking binding to Claude Opus 4.6, which rejects block_binding on Bedrock', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const model = 'us.anthropic.claude-opus-4-6-v1';
        const request = driver.preparePayload(
            { modelId: model, messages: [{ role: 'user', content: [{ text: 'hello' }] }] },
            { model, model_options: { _option_id: 'bedrock-claude', effort: 'medium' } },
        );
        expect(request.additionalModelRequestFields).toMatchObject({ reasoning_config: { type: 'adaptive' } });
        expect(request.additionalModelRequestFields).not.toHaveProperty('anthropic_beta');
        expect(request.additionalModelRequestFields).not.toHaveProperty('reasoning_config.block_binding');
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
    it('warns when canonical GPT effort overrides a conflicting alias', () => {
        const logger = { warn: vi.fn(), info: vi.fn(), debug: vi.fn(), error: vi.fn() };
        const driver = new BedrockDriver({ region: 'us-east-1', logger });
        const model = 'openai.gpt-6.1-sol';
        const request = driver.preparePayload(
            { modelId: model, messages: [{ role: 'user', content: [{ text: 'hello' }] }] },
            { model, model_options: { _option_id: 'bedrock-converse', effort: 'low', reasoning_effort: 'high' } },
        );
        expect(request.additionalModelRequestFields).toMatchObject({ reasoning: { effort: 'low' } });
        expect(logger.warn).toHaveBeenCalledWith(
            { model, option_names: ['reasoning_effort'], reason: 'openai_effort_alias_precedence' },
            'Model option compatibility exception changed caller input',
        );
    });

    it.each([
        'openai.gpt-5.5',
        'us.openai.gpt-5.6-sol',
        'global.openai.gpt-6-luna',
        'us.openai.gpt-6.1-sol',
        'arn:aws:bedrock:us-east-1:000000000000:inference-profile/global.openai.gpt-6-luna',
    ])('transports effort and strict output together for %s', (model) => {
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
        expect(request.inferenceConfig).toEqual({
            maxTokens: 100,
            temperature: 0.7,
            topP: 0.9,
            stopSequences: ['STOP'],
        });
        expect(request.additionalModelRequestFields).toEqual({
            reasoning: { effort: 'high' },
            text: { verbosity: 'low', format: { strict: true } },
        });
        expect(request.outputConfig?.textFormat?.structure?.jsonSchema?.schema).toBe(JSON.stringify(result_schema));
    });

    it.each([undefined, 0])('preserves omitted and zero GPT sampling values (%s) without warnings', (value) => {
        const logger = { warn: vi.fn(), info: vi.fn(), debug: vi.fn(), error: vi.fn() };
        const driver = new BedrockDriver({ region: 'us-east-1', logger });
        const model = 'global.openai.gpt-6-luna';
        const model_options: ModelOptions = {
            _option_id: 'bedrock-converse',
            temperature: value,
            top_p: value,
            stop_sequence: value === undefined ? undefined : ['STOP'],
        };
        const original = { ...model_options };
        const request = driver.preparePayload(
            { modelId: model, messages: [{ role: 'user', content: [{ text: 'hello' }] }] },
            { model, model_options },
        );
        expect(request.inferenceConfig).toEqual(
            value === undefined ? undefined : { temperature: 0, topP: 0, stopSequences: ['STOP'] },
        );
        expect(model_options).toEqual(original);
        expect(logger.warn).not.toHaveBeenCalled();
    });

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

describe('Bedrock Nova extended thinking', () => {
    it.each(['none', 'low', 'medium', 'high'] as const)('transports %s effort', (effort) => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const request = driver.preparePayload(
            { modelId: undefined, messages: [] },
            {
                model: 'us.amazon.nova-2-lite-v1:0',
                model_options: { _option_id: 'bedrock-nova', effort, max_tokens: 1000, temperature: 0.7, top_k: 12 },
            },
        );
        expect(request.additionalModelRequestFields).toEqual({
            inferenceConfig: { topK: 12 },
            reasoningConfig: effort === 'none' ? { type: 'disabled' } : { type: 'enabled', maxReasoningEffort: effort },
        });
        expect(request.inferenceConfig).toEqual({ maxTokens: 1000, temperature: 0.7 });
    });

    it('passes explicit effort through for Nova versions without known reasoning support', () => {
        const request = new BedrockDriver({ region: 'us-east-1' }).preparePayload(
            { modelId: undefined, messages: [] },
            { model: 'amazon.nova-pro-v1:0', model_options: { _option_id: 'bedrock-nova', effort: 'high' } },
        );
        expect(request.additionalModelRequestFields).toEqual({
            reasoningConfig: { type: 'enabled', maxReasoningEffort: 'high' },
        });
    });

    it('preserves Nova 1.5 high-effort output and sampling controls', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const request = driver.preparePayload(
            { modelId: undefined, messages: [] },
            {
                model: 'amazon.nova-lite-1-5-v1:0',
                model_options: { _option_id: 'bedrock-nova', effort: 'high', max_tokens: 40000, temperature: 0 },
            },
        );
        expect(request.additionalModelRequestFields).toMatchObject({
            reasoningConfig: { type: 'enabled', maxReasoningEffort: 'high' },
        });
        expect(request.inferenceConfig).toEqual({ maxTokens: 40000, temperature: 0 });
    });

    it('preserves provider defaults when effort is unset', () => {
        const request = new BedrockDriver({ region: 'us-east-1' }).preparePayload(
            { modelId: undefined, messages: [] },
            {
                model: 'amazon.nova-2-lite-v1:0',
                model_options: { _option_id: 'bedrock-nova', max_tokens: 1000 },
            },
        );
        expect(request.additionalModelRequestFields).toBeUndefined();
        expect(request.inferenceConfig).toEqual({ maxTokens: 1000 });
    });
});
