import { describe, expect, it, vi } from 'vitest';
import { BedrockDriver } from '../bedrock/index.js';
import { getGeminiPayload } from '../vertexai/models/gemini.js';
import { getClaudePayload } from './claude-messages.js';
import { logModelOptionException } from './model-option-exceptions.js';

function logger() {
    return { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
}

describe('model option compatibility exceptions', () => {
    it('reports only supplied option names, without values or unrelated input', () => {
        const log = logger();
        logModelOptionException(
            log,
            'model',
            { temperature: 0, top_p: undefined, extra_body: 'private' },
            ['temperature', 'top_p'],
            'sampling',
        );
        expect(log.warn).toHaveBeenCalledExactlyOnceWith(
            { model: 'model', option_names: ['temperature'], reason: 'sampling' },
            'Model option compatibility exception changed caller input',
        );
    });

    it('does not warn for absent options', () => {
        const log = logger();
        logModelOptionException(log, 'model', undefined, ['temperature'], 'sampling');
        logModelOptionException(log, 'model', {}, ['temperature'], 'sampling');
        expect(log.warn).not.toHaveBeenCalled();
    });

    it('reports existing Claude sampling and legacy-budget omissions', () => {
        const log = logger();
        const { payload } = getClaudePayload(
            {
                model: 'claude-sonnet-5-5',
                model_options: {
                    _option_id: 'anthropic-claude',
                    temperature: 0,
                    top_p: 0.8,
                    top_k: 5,
                    effort: 'high',
                    thinking_budget_tokens: 8000,
                },
            },
            { messages: [{ role: 'user', content: 'hello' }] },
            log,
        );
        expect(payload.temperature).toBeUndefined();
        expect(payload.top_p).toBeUndefined();
        expect(payload.top_k).toBeUndefined();
        expect(log.warn).toHaveBeenCalledWith(
            expect.objectContaining({
                option_names: ['temperature', 'top_p', 'top_k'],
                reason: 'claude_sampling_restriction',
            }),
            expect.any(String),
        );
        expect(log.warn).toHaveBeenCalledWith(
            expect.objectContaining({
                option_names: ['thinking_budget_tokens'],
                reason: 'claude_thinking_mode',
            }),
            expect.any(String),
        );
    });

    it('reports existing Bedrock Claude clamping and sampling omissions', () => {
        const log = logger();
        const payload = new BedrockDriver({ region: 'us-east-1', logger: log }).preparePayload(
            { modelId: undefined, messages: [] },
            {
                model: 'anthropic.claude-sonnet-5-5',
                model_options: {
                    _option_id: 'bedrock-claude',
                    max_tokens: 1_000_000,
                    temperature: 0,
                    top_p: 0.8,
                    top_k: 5,
                },
            },
        );
        expect(payload.inferenceConfig?.temperature).toBeUndefined();
        expect(log.warn).toHaveBeenCalledWith(
            expect.objectContaining({
                option_names: ['max_tokens'],
                reason: 'claude_output_limit',
            }),
            expect.any(String),
        );
        expect(log.warn).toHaveBeenCalledWith(
            expect.objectContaining({
                option_names: ['temperature', 'top_p', 'top_k'],
                reason: 'converse_sampling_restriction',
            }),
            expect.any(String),
        );
    });

    it('reports existing DeepSeek option omissions', () => {
        const log = logger();
        const payload = new BedrockDriver({ region: 'us-east-1', logger: log }).preparePayload(
            { modelId: undefined, messages: [] },
            {
                model: 'us.deepseek.r1-v1:0',
                model_options: {
                    _option_id: 'bedrock-converse',
                    stop_sequence: ['stop'],
                    top_p: 0.8,
                },
            },
        );
        expect(payload.inferenceConfig).toBeUndefined();
        expect(log.warn).toHaveBeenCalledWith(
            expect.objectContaining({
                option_names: ['stop_sequence', 'top_p'],
                reason: 'deepseek_converse_options',
            }),
            expect.any(String),
        );
    });

    it('reports existing Nano Banana sampling omissions', () => {
        const log = logger();
        getGeminiPayload(
            {
                model: 'gemini-nano-banana-2.1',
                model_options: { _option_id: 'vertexai-gemini', temperature: 0, top_p: 0.8 },
            },
            { contents: [] },
            log,
        );
        expect(log.warn).toHaveBeenCalledExactlyOnceWith(
            expect.objectContaining({ option_names: ['temperature', 'top_p'], reason: 'nano_banana_sampling' }),
            expect.any(String),
        );
    });

    it.each(['gemini-3.1-flash-image', 'gemini-nano-banana-2.1'])(
        'reports existing image generation option omissions for %s',
        (model) => {
            const log = logger();
            const { config } = getGeminiPayload(
                {
                    model,
                    model_options: {
                        _option_id: 'vertexai-gemini',
                        top_k: 5,
                        seed: 0,
                        presence_penalty: 0,
                        frequency_penalty: 0,
                    },
                },
                { contents: [] },
                log,
            );
            expect(config?.topK).toBeUndefined();
            expect(config?.seed).toBeUndefined();
            expect(config?.presencePenalty).toBeUndefined();
            expect(config?.frequencyPenalty).toBeUndefined();
            expect(log.warn).toHaveBeenCalledExactlyOnceWith(
                {
                    model,
                    option_names: ['top_k', 'seed', 'presence_penalty', 'frequency_penalty'],
                    reason: 'gemini_image_generation_options',
                },
                'Model option compatibility exception changed caller input',
            );
        },
    );
});
