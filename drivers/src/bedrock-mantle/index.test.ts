import { getTokenProvider } from '@aws/bedrock-token-generator';
import type { AwsCredentialIdentity } from '@aws-sdk/types';
import { getBedrockMantleProtocol, PromptRole, type PromptSegment, Providers } from '@llumiverse/core';
import type OpenAI from 'openai';
import type { BedrockOpenAI } from 'openai';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { BedrockDriver } from '../bedrock/index.js';
import { OpenAIResponsesDriverBase } from '../openai/index.js';
import { BedrockMantleDriver, isBedrockMantleModel } from './index.js';

vi.mock('@aws/bedrock-token-generator', () => ({
    getTokenProvider: vi.fn(() => async () => 'bedrock-api-key-test'),
}));

const promptSegments: PromptSegment[] = [{ role: PromptRole.user, content: 'hello' }];

describe('Bedrock Mantle model routing', () => {
    it('routes model families through their publisher protocol', () => {
        expect(getBedrockMantleProtocol('openai.gpt-oss-120b')).toBe('chat_completions');
        expect(getBedrockMantleProtocol('xai.grok-4.3')).toBe('responses');
        expect(getBedrockMantleProtocol('openai.gpt-oss-safeguard-120b')).toBe('chat_completions');
        expect(getBedrockMantleProtocol('anthropic.claude-opus-4-7')).toBe('messages');
        expect(getBedrockMantleProtocol('google.gemma-3-27b-it')).toBe('chat_completions');
        expect(getBedrockMantleProtocol('google.gemma-4-31b')).toBe('responses');
        expect(getBedrockMantleProtocol('google.gemma-5-70b')).toBe('responses');
        expect(isBedrockMantleModel('openai.gpt-6.1')).toBe(true);
        expect(isBedrockMantleModel('qwen.qwen4-coder')).toBe(true);
        expect(isBedrockMantleModel('openai.o7')).toBe(false);
    });

    it('formats Mantle prompts as OpenAI Responses input items', async () => {
        const driver = new BedrockMantleDriver({ region: 'us-west-2' });

        const prompt = await driver.createPrompt(promptSegments, { model: 'openai.gpt-5.5' });

        expect(Array.isArray(prompt)).toBe(true);
        expect(prompt).toEqual([{ type: 'message', role: 'user', content: [{ type: 'input_text', text: 'hello' }] }]);
    });

    it('keeps result schemas out of Chat Completions prompts', async () => {
        const driver = new BedrockMantleDriver({ region: 'us-west-2' });

        const prompt = await driver.createPrompt(promptSegments, {
            model: 'google.gemma-3-4b-it',
            result_schema: {
                type: 'object',
                properties: { answer: { type: 'string' } },
                required: ['answer'],
            },
        });

        expect(prompt).toMatchObject({
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'hello' }],
        });
    });

    it('formats Gemma 4 and later models as Responses input', async () => {
        const driver = new BedrockMantleDriver({ region: 'us-west-2' });

        for (const model of ['google.gemma-4-31b', 'google.gemma-5-70b']) {
            const prompt = await driver.createPrompt(promptSegments, { model });

            expect(prompt).toEqual([
                { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'hello' }] },
            ]);
        }
    });

    it('formats every GPT-OSS Mantle model as Chat Completions', async () => {
        const driver = new BedrockMantleDriver({ region: 'us-west-2' });

        const prompt = await driver.createPrompt(promptSegments, { model: 'openai.gpt-oss-200b' });

        expect(prompt).toMatchObject({
            _is_openai_chat_completions: true,
            messages: [{ role: 'user', content: 'hello' }],
        });
    });

    it('formats Claude models with the shared Anthropic Messages prompt shape', async () => {
        const driver = new BedrockMantleDriver({ region: 'us-west-2' });

        const prompt = await driver.createPrompt(promptSegments, { model: 'anthropic.claude-haiku-4-5' });

        expect(prompt).toEqual({ messages: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }] });
    });

    it('keeps Bedrock GPT-OSS on the existing Converse prompt shape', async () => {
        const driver = new BedrockDriver({ region: 'us-east-2' });

        const prompt = await driver.createPrompt(promptSegments, { model: 'openai.gpt-oss-120b-1:0' });

        expect(Array.isArray(prompt)).toBe(false);
        expect(prompt).toMatchObject({ messages: [{ role: 'user' }] });
    });

    it('does not route Mantle-only model ids through Bedrock prompt formatting', async () => {
        const driver = new BedrockDriver({ region: 'us-east-2' });

        const prompt = await driver.createPrompt(promptSegments, { model: 'openai.gpt-5.5' });

        expect(Array.isArray(prompt)).toBe(false);
        expect(prompt).toMatchObject({ messages: [{ role: 'user' }] });
    });
});

describe('BedrockMantleDriver auth', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('builds a bearer token provider from explicit Bedrock credentials', () => {
        const credentials: AwsCredentialIdentity = {
            accessKeyId: 'test-key',
            secretAccessKey: 'test-secret',
        };

        const driver = new BedrockMantleDriver({ region: 'us-west-2', credentials });

        expect(getTokenProvider).toHaveBeenCalledWith({ region: 'us-west-2', credentials });
        expect(
            (driver as unknown as { anthropicService: { awsAccessKey: string; awsSecretAccessKey: string } })
                .anthropicService,
        ).toMatchObject({ awsAccessKey: 'test-key', awsSecretAccessKey: 'test-secret' });
    });

    it('builds a bearer token provider from the default AWS credential chain when credentials are absent', () => {
        const driver = new BedrockMantleDriver({ region: 'us-west-2' });

        expect(getTokenProvider).toHaveBeenCalledWith({ region: 'us-west-2' });
        expect(driver.service.baseURL).toBe('https://bedrock-mantle.us-west-2.api.aws/v1');
        expect(
            (driver as unknown as { responsesDelegate: { service: OpenAI } }).responsesDelegate.service.baseURL,
        ).toBe('https://bedrock-mantle.us-west-2.api.aws/openai/v1');
    });
});

describe('BedrockMantleDriver model listing', () => {
    it('lists only the Mantle models that the Bedrock Mantle driver can route', async () => {
        const driver = new BedrockMantleDriver({ region: 'us-west-2' });
        const list = vi.fn(async () => ({
            data: [
                { id: 'unverified.model-1', object: 'model', created: 1783669008, owned_by: 'system' },
                { id: 'openai.gpt-5.6', object: 'model', created: 1783669008, owned_by: 'system' },
                { id: 'openai.gpt-5.5', owned_by: 'system' },
                { id: 'openai.gpt-oss-120b', owned_by: 'system' },
                { id: 'xai.grok-4.3', owned_by: 'system' },
                { id: 'openai.gpt-oss-120b-1:0', owned_by: 'system' },
                { id: 'anthropic.claude-opus-4-7', owned_by: 'system' },
                { id: 'google.gemma-3-4b-it', owned_by: 'system' },
                { id: 'google.gemma-4-31b', owned_by: 'system' },
                { id: 'google.gemma-5-70b', owned_by: 'system' },
                { id: 'zai.glm-5', owned_by: 'system' },
            ],
        }));
        driver.service = { models: { list } } as unknown as BedrockOpenAI;

        const models = await driver.listModels();
        const modelIds = models.map((model) => model.id);
        for (const expectedModel of [
            'anthropic.claude-opus-4-7',
            'google.gemma-3-4b-it',
            'google.gemma-4-31b',
            'google.gemma-5-70b',
            'openai.gpt-5.5',
            'openai.gpt-5.6',
            'openai.gpt-oss-120b',
            'openai.gpt-oss-120b-1:0',
            'xai.grok-4.3',
            'zai.glm-5',
        ]) {
            expect(modelIds).toContain(expectedModel);
        }
        expect(modelIds).not.toContain('unverified.model-1');
        expect(models).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    id: 'anthropic.claude-opus-4-7',
                    owner: 'Anthropic',
                    input_modalities: ['text', 'image'],
                }),
                expect.objectContaining({
                    id: 'google.gemma-3-4b-it',
                    owner: 'Google',
                    input_modalities: ['text', 'image'],
                }),
                expect.objectContaining({
                    id: 'google.gemma-4-31b',
                    owner: 'Google',
                    input_modalities: ['text', 'image', 'video'],
                }),
                expect.objectContaining({
                    id: 'openai.gpt-oss-120b',
                    owner: 'OpenAI',
                    input_modalities: ['text'],
                }),
                expect.objectContaining({
                    id: 'zai.glm-5',
                    owner: 'Z.AI',
                    provider: Providers.bedrock_mantle,
                    can_stream: true,
                    output_modalities: ['text'],
                    tool_support: true,
                }),
            ]),
        );
    });
});

describe('BedrockMantleDriver protocol execution', () => {
    it('executes Chat Completions through the shared OpenAI SDK protocol', async () => {
        const driver = new BedrockMantleDriver({ region: 'us-west-2' });
        const create = vi.fn(async () => ({
            id: 'chatcmpl-test',
            object: 'chat.completion',
            created: 1,
            model: 'google.gemma-3-4b-it',
            choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
            usage: { prompt_tokens: 4, completion_tokens: 1, total_tokens: 5 },
        }));
        driver.service = { chat: { completions: { create } } } as unknown as BedrockOpenAI;
        const prompt = await driver.createPrompt(promptSegments, { model: 'google.gemma-3-4b-it' });

        const completion = await driver.requestTextCompletion(prompt, {
            model: 'google.gemma-3-4b-it',
            model_options: { _option_id: 'bedrock-mantle-chat-completions', max_tokens: 100 },
        });

        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({
                model: 'google.gemma-3-4b-it',
                messages: [{ role: 'user', content: 'hello' }],
                max_tokens: 100,
            }),
        );
        expect(completion.result).toEqual([{ type: 'text', value: 'ok' }]);
    });

    it('uses response_format for Chat Completions schema requests', async () => {
        for (const model of [
            'google.gemma-3-4b-it',
            'mistral.ministral-3-3b-instruct',
            'openai.gpt-oss-20b',
            'qwen.qwen3-vl-235b-a22b-instruct',
            'writer.palmyra-vision-7b',
        ]) {
            const driver = new BedrockMantleDriver({ region: 'us-west-2' });
            const create = vi.fn(async () => ({
                id: 'chatcmpl-test',
                object: 'chat.completion',
                created: 1,
                model,
                choices: [
                    {
                        index: 0,
                        message: { role: 'assistant', content: '{"answer":"ok"}' },
                        finish_reason: 'stop',
                    },
                ],
            }));
            driver.service = { chat: { completions: { create } } } as unknown as BedrockOpenAI;
            const result_schema = {
                type: 'object' as const,
                properties: { answer: { type: 'string' as const } },
                required: ['answer'],
            };
            const prompt = await driver.createPrompt(promptSegments, { model, result_schema });

            await driver.requestTextCompletion(prompt, {
                model,
                result_schema,
                model_options: { _option_id: 'bedrock-mantle-chat-completions' },
            });

            expect(create).toHaveBeenCalledWith(
                expect.objectContaining({
                    model,
                    messages: [{ role: 'user', content: 'hello' }],
                    response_format: expect.objectContaining({ type: 'json_schema' }),
                }),
            );
        }
    });

    it('supplements Magistral response_format with schema prompt alignment', async () => {
        const driver = new BedrockMantleDriver({ region: 'us-west-2' });
        const create = vi.fn(async () => ({
            id: 'chatcmpl-test',
            object: 'chat.completion',
            created: 1,
            model: 'mistral.magistral-small-2509',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: '{"answer":"ok"}' },
                    finish_reason: 'stop',
                },
            ],
        }));
        driver.service = { chat: { completions: { create } } } as unknown as BedrockOpenAI;
        const result_schema = {
            type: 'object' as const,
            properties: { answer: { type: 'string' as const } },
            required: ['answer'],
        };
        const prompt = await driver.createPrompt(promptSegments, {
            model: 'mistral.magistral-small-2509',
            result_schema,
        });

        await driver.requestTextCompletion(prompt, {
            model: 'mistral.magistral-small-2509',
            result_schema,
            model_options: { _option_id: 'bedrock-mantle-chat-completions' },
        });

        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({
                messages: [
                    { role: 'system', content: expect.stringContaining('<response_schema>') },
                    { role: 'user', content: 'hello' },
                ],
                response_format: expect.objectContaining({ type: 'json_schema' }),
            }),
        );
    });

    it('executes Claude through the shared Messages helpers', async () => {
        const driver = new BedrockMantleDriver({ region: 'us-west-2' });
        const finalMessage = vi.fn(async () => ({
            id: 'msg-test',
            type: 'message',
            role: 'assistant',
            model: 'anthropic.claude-haiku-4-5',
            content: [{ type: 'text', text: 'ok', citations: null }],
            stop_reason: 'end_turn',
            stop_sequence: null,
            usage: { input_tokens: 4, output_tokens: 1 },
        }));
        const stream = vi.fn(() => ({ finalMessage }));
        (driver as unknown as { anthropicService: { messages: { stream: typeof stream } } }).anthropicService = {
            messages: { stream },
        };
        const prompt = await driver.createPrompt(promptSegments, { model: 'anthropic.claude-haiku-4-5' });

        const completion = await driver.requestTextCompletion(prompt, {
            model: 'anthropic.claude-haiku-4-5',
            model_options: { _option_id: 'bedrock-mantle-claude', max_tokens: 100 },
        });

        expect(stream).toHaveBeenCalledWith(
            expect.objectContaining({
                model: 'anthropic.claude-haiku-4-5',
                messages: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }],
                max_tokens: 100,
                stream: true,
            }),
            undefined,
        );
        expect(completion.result).toEqual([{ type: 'text', value: 'ok' }]);
    });
});

class TestResponsesDriver extends OpenAIResponsesDriverBase {
    provider = Providers.bedrock_mantle as const;
    service: OpenAI;

    constructor(create: ReturnType<typeof vi.fn>) {
        super({});
        this.service = { responses: { create } } as unknown as OpenAI;
    }
}

describe('Bedrock Mantle Responses options', () => {
    it('forwards verbosity alongside structured output format', async () => {
        const create = vi.fn(async () => ({
            output: [
                {
                    type: 'message',
                    role: 'assistant',
                    content: [{ type: 'output_text', text: 'ok', annotations: [] }],
                },
            ],
        }));
        const driver = new TestResponsesDriver(create);
        const prompt = [{ type: 'message', role: 'user', content: 'hello' }] as OpenAI.Responses.ResponseInputItem[];

        await driver.requestTextCompletion(prompt, {
            model: 'openai.gpt-5.5',
            model_options: {
                _option_id: 'bedrock-mantle-responses',
                max_tokens: 100,
                effort: 'low',
                verbosity: 'low',
            },
            result_schema: {
                type: 'object',
                properties: { answer: { type: 'string' } },
                required: ['answer'],
            },
        });

        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({
                model: 'openai.gpt-5.5',
                max_output_tokens: 100,
                reasoning: { effort: 'low' },
                text: expect.objectContaining({
                    verbosity: 'low',
                    format: expect.objectContaining({ type: 'json_schema', name: 'format_output' }),
                }),
            }),
        );
    });

    it('forwards Grok reasoning effort through the Responses API', async () => {
        const create = vi.fn(async () => ({
            output: [
                {
                    type: 'message',
                    role: 'assistant',
                    content: [{ type: 'output_text', text: 'ok', annotations: [] }],
                },
            ],
        }));
        const driver = new TestResponsesDriver(create);
        const prompt = [{ type: 'message', role: 'user', content: 'hello' }] as OpenAI.Responses.ResponseInputItem[];

        await driver.requestTextCompletion(prompt, {
            model: 'xai.grok-4.3',
            model_options: {
                _option_id: 'bedrock-mantle-responses',
                reasoning_effort: 'none',
                temperature: 0.4,
                top_p: 0.9,
            },
        });

        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({
                model: 'xai.grok-4.3',
                reasoning: { effort: 'none' },
                temperature: 0.4,
                top_p: 0.9,
            }),
        );
    });
});
