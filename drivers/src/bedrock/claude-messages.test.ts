import type Anthropic from '@anthropic-ai/sdk';
import { APIError, BadRequestError } from '@anthropic-ai/sdk/error';
import { type ExecutionOptions, PromptRole, type PromptSegment } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { isConverseConversation } from './converse.js';
import { BedrockDriver } from './index.js';

const MODEL = 'us.anthropic.claude-sonnet-5-5';
const segments: PromptSegment[] = [{ role: PromptRole.user, content: 'hello' }];

const legacyConversation = {
    messages: [
        { role: 'user', content: [{ text: 'earlier question' }] },
        { role: 'assistant', content: [{ reasoningContent: { reasoningText: { text: '', signature: 's' } } }] },
    ],
};

function message(): Anthropic.Message {
    return {
        id: 'msg-test',
        type: 'message',
        role: 'assistant',
        model: MODEL,
        content: [{ type: 'text', text: 'ok', citations: null }],
        stop_reason: 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: 4, output_tokens: 1 },
    } as unknown as Anthropic.Message;
}

function driverWithClaudeClient(logger?: ConstructorParameters<typeof BedrockDriver>[0]['logger']) {
    const driver = new BedrockDriver({ region: 'us-east-1', logger });
    const stream = vi.fn((_payload: unknown, _options?: unknown) => ({ finalMessage: async () => message() }));
    Reflect.set(driver, '_anthropicService', { messages: { stream } });
    return { driver, stream };
}

describe('isConverseConversation', () => {
    it('recognises Converse content blocks', () => {
        expect(isConverseConversation(legacyConversation)).toBe(true);
    });

    it.each([
        undefined,
        null,
        {},
        { messages: [] },
        { messages: [{ role: 'user', content: 'hi' }] },
        { messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] },
    ])('treats %j as a Messages API conversation', (conversation) => {
        expect(isConverseConversation(conversation)).toBe(false);
    });
});

describe('Bedrock Claude on the Messages API', () => {
    it('formats new Claude conversations for the Messages API and keeps legacy ones on Converse', async () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });

        const fresh = await driver.createPrompt(segments, { model: MODEL });
        expect(fresh).toEqual({ messages: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }] });

        const legacy = await driver.createPrompt(segments, {
            model: MODEL,
            conversation: legacyConversation,
        } as ExecutionOptions);
        expect(legacy).toMatchObject({
            modelId: undefined,
            messages: [{ role: 'user', content: [{ text: 'hello' }] }],
        });

        const other = await driver.createPrompt(segments, { model: 'amazon.nova-pro-v1:0' });
        expect(other).toHaveProperty('modelId');
    });

    it('sends Claude through InvokeModel with the service tier header and the Bedrock output limit', async () => {
        const logger = { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
        const { driver, stream } = driverWithClaudeClient(logger);
        const options = {
            model: MODEL,
            model_options: { _option_id: 'bedrock-claude' as const, max_tokens: 10_000_000, service_tier: 'priority' },
        };
        const prompt = await driver.createPrompt(segments, options);

        const completion = await driver.requestTextCompletion(prompt, options);

        const [payload, requestOptions] = stream.mock.calls[0] as [{ max_tokens: number }, { headers?: unknown }];
        expect(payload).toMatchObject({ model: MODEL, stream: true });
        expect(payload.max_tokens).toBeLessThan(10_000_000);
        expect(requestOptions.headers).toEqual({ 'X-Amzn-Bedrock-Service-Tier': 'priority' });
        expect(logger.warn).toHaveBeenCalledWith(
            expect.objectContaining({ option_names: ['max_tokens'], reason: 'claude_output_limit' }),
            expect.any(String),
        );
        expect(completion.result).toEqual([{ type: 'text', value: 'ok' }]);
        expect(completion.conversation).toMatchObject({ messages: [{ role: 'user' }, { role: 'assistant' }] });
    });

    it('streams Claude through the shared Messages helpers', async () => {
        const { driver, stream } = driverWithClaudeClient();
        Reflect.set(driver, '_anthropicService', {
            messages: {
                stream: (...args: [unknown, unknown?]) => {
                    stream(...args);
                    return Object.assign((async function* () {})(), { finalMessage: async () => message() });
                },
            },
        });
        const prompt = await driver.createPrompt(segments, { model: MODEL });

        const completion = await driver.requestTextCompletionStream(prompt, { model: MODEL });
        for await (const _chunk of completion) {
            // drain
        }

        expect(stream).toHaveBeenCalledOnce();
        expect(await completion.finalizeConversation?.()).toMatchObject({
            messages: [{ role: 'user' }, { role: 'assistant', content: [{ type: 'text', text: 'ok' }] }],
        });
    });

    it('signs with explicit credentials and targets the bedrock-runtime endpoint', () => {
        const driver = new BedrockDriver({
            region: 'eu-west-1',
            credentials: { accessKeyId: 'test-key', secretAccessKey: 'test-secret', sessionToken: 'test-token' },
        });
        const service = Reflect.get(driver, 'getAnthropicService').call(driver);
        expect(service).toMatchObject({
            awsAccessKey: 'test-key',
            awsSecretKey: 'test-secret',
            awsSessionToken: 'test-token',
            awsRegion: 'eu-west-1',
            baseURL: 'https://bedrock-runtime.eu-west-1.amazonaws.com',
        });
    });

    it('resolves a credential provider through the SDK provider chain', async () => {
        const credentials = vi.fn(async () => ({ accessKeyId: 'k', secretAccessKey: 's' }));
        const driver = new BedrockDriver({ region: 'us-east-1', credentials });
        const service = Reflect.get(driver, 'getAnthropicService').call(driver);
        expect(await service.providerChainResolver()).toBe(credentials);
    });

    it('formats Anthropic SDK errors', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const error = new BadRequestError(
            400,
            { error: { type: 'invalid_request_error', message: 'bad' } },
            'bad',
            new Headers(),
        );
        const formatted = driver.formatLlumiverseError(error, {
            provider: 'bedrock',
            model: MODEL,
            operation: 'execute',
        });
        expect(formatted.message).toContain('bad');
        expect(formatted.retryable).toBe(false);
    });

    it('marks Bedrock exceptions raised mid-stream with the Bedrock retry rules', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const context = { provider: 'bedrock', model: MODEL, operation: 'stream' as const };
        // The SDK throws a plain APIError (no status) for an SSE error event.
        const streamError = (type: string) =>
            new APIError(undefined, { type: 'error', error: { type, message: 'm' } }, undefined, new Headers());
        expect(driver.formatLlumiverseError(streamError('throttlingException'), context).retryable).toBe(true);
        expect(driver.formatLlumiverseError(streamError('modelStreamErrorException'), context).retryable).toBe(true);
        expect(driver.formatLlumiverseError(streamError('validationException'), context).retryable).toBe(false);
    });

    it('sends a Claude prompt with no messages to the Messages API', async () => {
        const { driver, stream } = driverWithClaudeClient();
        await driver.requestTextCompletion({ messages: [] }, { model: MODEL });
        expect(stream).toHaveBeenCalledOnce();
    });

    it('formats Claude prompts for debugging', async () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const prompt = await driver.createPrompt(segments, { model: MODEL });
        expect(driver.formatDebugPrompt(prompt)).toEqual(prompt);
    });
});
