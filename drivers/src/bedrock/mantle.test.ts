import { getTokenProvider } from '@aws/bedrock-token-generator';
import type { AwsCredentialIdentity } from '@aws-sdk/types';
import {
    type AIModel,
    type Completion,
    type CompletionChunkObject,
    PromptRole,
    type PromptSegment,
    Providers,
} from '@llumiverse/core';
import type OpenAI from 'openai';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { OpenAIResponsesDriverBase } from '../openai/index.js';
import { BedrockDriver, type BedrockPrompt } from './index.js';
import { BedrockMantleDriver, type BedrockMantlePrompt, isBedrockMantleModel } from './mantle.js';

vi.mock('@aws/bedrock-token-generator', () => ({
    getTokenProvider: vi.fn(() => async () => 'bedrock-api-key-test'),
}));

const promptSegments: PromptSegment[] = [{ role: PromptRole.user, content: 'hello' }];
const gpt55Options = { model: 'openai.gpt-5.5' };

type MantleStub = {
    requestTextCompletion: (prompt: BedrockMantlePrompt, options: typeof gpt55Options) => Promise<Completion>;
    requestTextCompletionStream: (
        prompt: BedrockMantlePrompt,
        options: typeof gpt55Options,
    ) => Promise<AsyncIterable<CompletionChunkObject>>;
    buildStreamingConversation: (
        prompt: BedrockMantlePrompt,
        result: unknown[],
        toolUse: unknown[] | undefined,
        options: typeof gpt55Options,
    ) => BedrockMantlePrompt;
    supportsStreaming: (options: typeof gpt55Options) => Promise<boolean>;
    listModels: () => Promise<AIModel[]>;
    destroy: () => void;
};

function setMantleStub(driver: BedrockDriver, mantle: MantleStub): void {
    (driver as unknown as { _mantleDriver: MantleStub })._mantleDriver = mantle;
}

async function* emptyStream(): AsyncIterable<CompletionChunkObject> {
    yield { result: [] };
}

describe('Bedrock Mantle model routing', () => {
    it('matches Bedrock Mantle models but not GPT-OSS', () => {
        expect(isBedrockMantleModel('openai.gpt-5.5')).toBe(true);
        expect(isBedrockMantleModel('openai.gpt-5.4')).toBe(true);
        expect(isBedrockMantleModel('xai.grok-4.3')).toBe(true);
        expect(isBedrockMantleModel('openai.gpt-oss-120b-1:0')).toBe(false);
    });

    it('formats Mantle prompts as OpenAI Responses input items', async () => {
        const driver = new BedrockDriver({ region: 'us-east-2' });

        const prompt = await driver.createPrompt(promptSegments, gpt55Options);

        expect(Array.isArray(prompt)).toBe(true);
        expect(prompt).toEqual([{ type: 'message', role: 'user', content: [{ type: 'input_text', text: 'hello' }] }]);
    });

    it('keeps GPT-OSS on the existing Converse prompt shape', async () => {
        const driver = new BedrockDriver({ region: 'us-east-2' });

        const prompt = await driver.createPrompt(promptSegments, { model: 'openai.gpt-oss-120b-1:0' });

        expect(Array.isArray(prompt)).toBe(false);
        expect(prompt).toMatchObject({ messages: [{ role: 'user' }] });
    });

    it('delegates Mantle execution paths to the internal subdriver', async () => {
        const driver = new BedrockDriver({ region: 'us-east-2' });
        const prompt = [{ type: 'message', role: 'user', content: 'hello' }] as BedrockMantlePrompt;
        const completion = { result: [{ type: 'text' as const, value: 'ok' }] };
        const streamingConversation = [{ type: 'message', role: 'assistant', content: 'ok' }] as BedrockMantlePrompt;
        const mantle: MantleStub = {
            requestTextCompletion: vi.fn(async () => completion),
            requestTextCompletionStream: vi.fn(async () => emptyStream()),
            buildStreamingConversation: vi.fn(() => streamingConversation),
            supportsStreaming: vi.fn(async () => true),
            listModels: vi.fn(async () => []),
            destroy: vi.fn(),
        };
        setMantleStub(driver, mantle);

        await expect(driver.requestTextCompletion(prompt, gpt55Options)).resolves.toBe(completion);
        await expect(driver.requestTextCompletionStream(prompt, gpt55Options)).resolves.toBeDefined();
        await expect(
            (driver as unknown as { canStream(options: typeof gpt55Options): Promise<boolean> }).canStream(
                gpt55Options,
            ),
        ).resolves.toBe(true);
        expect(driver.buildStreamingConversation(prompt, [], undefined, gpt55Options)).toBe(streamingConversation);

        expect(mantle.requestTextCompletion).toHaveBeenCalledWith(prompt, gpt55Options);
        expect(mantle.requestTextCompletionStream).toHaveBeenCalledWith(prompt, gpt55Options);
        expect(mantle.supportsStreaming).toHaveBeenCalledWith(gpt55Options);
        expect(mantle.buildStreamingConversation).toHaveBeenCalledWith(prompt, [], undefined, gpt55Options);
    });

    it('merges Mantle-only models into the Bedrock model listing', async () => {
        const driver = new BedrockDriver({ region: 'us-east-2' });
        const service = {
            listFoundationModels: vi.fn(async () => ({
                modelSummaries: [
                    {
                        modelId: 'openai.gpt-oss-120b-1:0',
                        modelArn: 'arn:aws:bedrock:us-east-2::foundation-model/openai.gpt-oss-120b-1:0',
                        providerName: 'OpenAI',
                        modelName: 'gpt-oss-120b',
                        inferenceTypesSupported: ['ON_DEMAND'],
                        inputModalities: ['TEXT'],
                        outputModalities: ['TEXT'],
                        responseStreamingSupported: true,
                    },
                ],
            })),
            listCustomModels: vi.fn(async () => ({ modelSummaries: [] })),
            listInferenceProfiles: vi.fn(async () => ({ inferenceProfileSummaries: [] })),
        };
        const mantleModel: AIModel = {
            id: 'openai.gpt-5.5',
            name: 'OpenAI GPT-5.5',
            provider: Providers.bedrock,
            owner: 'OpenAI',
            can_stream: true,
            input_modalities: ['text', 'image'],
            output_modalities: ['text'],
            tool_support: true,
        };
        const mantle: MantleStub = {
            requestTextCompletion: vi.fn(async () => ({ result: [] })),
            requestTextCompletionStream: vi.fn(async () => emptyStream()),
            buildStreamingConversation: vi.fn(() => []),
            supportsStreaming: vi.fn(async () => true),
            listModels: vi.fn(async () => [mantleModel]),
            destroy: vi.fn(),
        };
        (driver as unknown as { getService: () => typeof service }).getService = () => service;
        setMantleStub(driver, mantle);

        const models = await driver.listModels();

        expect(models.map((model) => model.id)).toEqual([
            'arn:aws:bedrock:us-east-2::foundation-model/openai.gpt-oss-120b-1:0',
            'openai.gpt-5.5',
        ]);
        expect(mantle.listModels).toHaveBeenCalled();
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

        new BedrockMantleDriver({ region: 'us-east-2', credentials });

        expect(getTokenProvider).toHaveBeenCalledWith({ region: 'us-east-2', credentials });
    });

    it('builds a bearer token provider from the default AWS credential chain when credentials are absent', () => {
        new BedrockMantleDriver({ region: 'us-east-2' });

        expect(getTokenProvider).toHaveBeenCalledWith({ region: 'us-east-2' });
    });
});

describe('BedrockMantleDriver model listing', () => {
    it('lists only the Mantle models that the Bedrock driver can route', async () => {
        const driver = new BedrockMantleDriver({ region: 'us-east-2' });
        const list = vi.fn(async () => ({
            data: [
                { id: 'openai.gpt-5.5', owned_by: 'system' },
                { id: 'openai.gpt-5.4', owned_by: 'system' },
                { id: 'xai.grok-4.3', owned_by: 'system' },
                { id: 'openai.gpt-oss-120b-1:0', owned_by: 'system' },
            ],
        }));
        (driver as unknown as { modelsService: OpenAI }).modelsService = { models: { list } } as unknown as OpenAI;

        const models = await driver.listModels();

        expect(models).toEqual([
            expect.objectContaining({
                id: 'openai.gpt-5.4',
                name: 'OpenAI GPT-5.4',
                provider: Providers.bedrock,
                owner: 'OpenAI',
                can_stream: true,
                tool_support: true,
            }),
            expect.objectContaining({
                id: 'openai.gpt-5.5',
                name: 'OpenAI GPT-5.5',
                provider: Providers.bedrock,
                owner: 'OpenAI',
                can_stream: true,
                tool_support: true,
            }),
            expect.objectContaining({
                id: 'xai.grok-4.3',
                name: 'xAI Grok 4.3',
                provider: Providers.bedrock,
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
    provider = Providers.bedrock as const;
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
        const prompt = [{ type: 'message', role: 'user', content: 'hello' }] as BedrockPrompt;

        await driver.requestTextCompletion(prompt as BedrockMantlePrompt, {
            model: 'openai.gpt-5.5',
            model_options: {
                _option_id: 'bedrock-openai-responses',
                max_tokens: 100,
                effort: 'medium',
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
                reasoning: { effort: 'medium' },
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
        const prompt = [{ type: 'message', role: 'user', content: 'hello' }] as BedrockPrompt;

        await driver.requestTextCompletion(prompt as BedrockMantlePrompt, {
            model: 'xai.grok-4.3',
            model_options: {
                _option_id: 'bedrock-openai-responses',
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
