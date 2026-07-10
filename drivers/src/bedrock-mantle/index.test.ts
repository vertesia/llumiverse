import { getTokenProvider } from '@aws/bedrock-token-generator';
import type { AwsCredentialIdentity } from '@aws-sdk/types';
import { PromptRole, type PromptSegment, Providers } from '@llumiverse/core';
import type OpenAI from 'openai';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { BedrockDriver } from '../bedrock/index.js';
import { OpenAIResponsesDriverBase } from '../openai/index.js';
import { BedrockMantleDriver, type BedrockMantlePrompt, isBedrockMantleModel } from './index.js';

vi.mock('@aws/bedrock-token-generator', () => ({
    getTokenProvider: vi.fn(() => async () => 'bedrock-api-key-test'),
}));

const promptSegments: PromptSegment[] = [{ role: PromptRole.user, content: 'hello' }];

describe('Bedrock Mantle model routing', () => {
    it('matches OpenAI and Grok models without requiring an exact model allowlist', () => {
        expect(isBedrockMantleModel('openai.gpt-5.6')).toBe(true);
        expect(isBedrockMantleModel('openai.gpt-5.5')).toBe(true);
        expect(isBedrockMantleModel('xai.grok-4.3')).toBe(true);
        expect(isBedrockMantleModel('xai.grok-4.4')).toBe(true);
        expect(isBedrockMantleModel('openai.gpt-oss-120b-1:0')).toBe(false);
        expect(isBedrockMantleModel('anthropic.claude-opus-4-7')).toBe(false);
    });

    it('formats Mantle prompts as OpenAI Responses input items', async () => {
        const driver = new BedrockMantleDriver({ region: 'us-west-2' });

        const prompt = await driver.createPrompt(promptSegments, { model: 'openai.gpt-5.5' });

        expect(Array.isArray(prompt)).toBe(true);
        expect(prompt).toEqual([{ type: 'message', role: 'user', content: [{ type: 'input_text', text: 'hello' }] }]);
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

        new BedrockMantleDriver({ region: 'us-west-2', credentials });

        expect(getTokenProvider).toHaveBeenCalledWith({ region: 'us-west-2', credentials });
    });

    it('builds a bearer token provider from the default AWS credential chain when credentials are absent', () => {
        new BedrockMantleDriver({ region: 'us-west-2' });

        expect(getTokenProvider).toHaveBeenCalledWith({ region: 'us-west-2' });
    });
});

describe('BedrockMantleDriver model listing', () => {
    it('lists only the Mantle models that the Bedrock Mantle driver can route', async () => {
        const driver = new BedrockMantleDriver({ region: 'us-west-2' });
        const list = vi.fn(async () => ({
            data: [
                { id: 'openai.gpt-5.6', object: 'model', created: 1783669008, owned_by: 'system' },
                { id: 'openai.gpt-5.5', owned_by: 'system' },
                { id: 'openai.gpt-5.4', owned_by: 'system' },
                { id: 'xai.grok-4.3', owned_by: 'system' },
                { id: 'openai.gpt-oss-120b-1:0', owned_by: 'system' },
                { id: 'anthropic.claude-opus-4-7', owned_by: 'system' },
            ],
        }));
        (driver as unknown as { modelsService: OpenAI }).modelsService = { models: { list } } as unknown as OpenAI;

        const models = await driver.listModels();

        expect(models).toEqual([
            expect.objectContaining({
                id: 'openai.gpt-5.4',
                name: 'OpenAI GPT-5.4',
                provider: Providers.bedrock_mantle,
                owner: 'OpenAI',
                can_stream: true,
                tool_support: true,
            }),
            expect.objectContaining({
                id: 'openai.gpt-5.5',
                name: 'OpenAI GPT-5.5',
                provider: Providers.bedrock_mantle,
                owner: 'OpenAI',
                can_stream: true,
                tool_support: true,
            }),
            expect.objectContaining({
                id: 'openai.gpt-5.6',
                name: 'OpenAI GPT-5.6',
                provider: Providers.bedrock_mantle,
                owner: 'OpenAI',
                can_stream: true,
            }),
            expect.objectContaining({
                id: 'xai.grok-4.3',
                name: 'xAI Grok 4.3',
                provider: Providers.bedrock_mantle,
                owner: 'xAI',
                can_stream: true,
                input_modalities: ['text', 'image'],
                output_modalities: ['text'],
                tool_support: true,
            }),
        ]);
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
        const prompt = [{ type: 'message', role: 'user', content: 'hello' }] as BedrockMantlePrompt;

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
        const prompt = [{ type: 'message', role: 'user', content: 'hello' }] as BedrockMantlePrompt;

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
