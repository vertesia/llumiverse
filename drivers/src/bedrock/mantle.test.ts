import { getTokenProvider } from '@aws/bedrock-token-generator';
import type { AwsCredentialIdentity } from '@aws-sdk/types';
import {
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
    destroy: () => void;
};

function setMantleStub(driver: BedrockDriver, mantle: MantleStub): void {
    (driver as unknown as { _mantleDriver: MantleStub })._mantleDriver = mantle;
}

async function* emptyStream(): AsyncIterable<CompletionChunkObject> {
    yield { result: [] };
}

describe('Bedrock Mantle model routing', () => {
    it('matches GPT-5.5 and GPT-5.4 but not GPT-OSS', () => {
        expect(isBedrockMantleModel('openai.gpt-5.5')).toBe(true);
        expect(isBedrockMantleModel('openai.gpt-5.4')).toBe(true);
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
});
