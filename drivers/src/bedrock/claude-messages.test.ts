import type Anthropic from '@anthropic-ai/sdk';
import { AnthropicError, APIError, BadRequestError } from '@anthropic-ai/sdk/error';
import { type ExecutionOptions, PromptRole, type PromptSegment } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { formatAnthropicLlumiverseError, isAnthropicSdkError } from '../shared/claude-messages.js';
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

/** An InvokeModel response reporting the tier Bedrock applied. */
const tierResponse = () => new Response(null, { headers: { 'x-amzn-bedrock-service-tier': 'priority' } });

function driverWithClaudeClient(logger?: ConstructorParameters<typeof BedrockDriver>[0]['logger']) {
    const driver = new BedrockDriver({ region: 'us-east-1', logger });
    const stream = vi.fn((_payload: unknown, _options?: unknown) => ({
        async *[Symbol.asyncIterator]() {},
        finalMessage: async () => message(),
        withResponse: async () => ({ response: tierResponse() }),
    }));
    Reflect.set(driver, '_anthropicService', { messages: { stream } });
    return { driver, stream };
}

describe('isConverseConversation', () => {
    it('recognises Converse content blocks', () => {
        expect(isConverseConversation(legacyConversation)).toBe(true);
    });

    it('recognises a Converse conversation from its system blocks alone', () => {
        expect(isConverseConversation({ messages: [], system: [{ text: 'You are helpful.' }] })).toBe(true);
    });

    it('skips empty content when deciding the format', () => {
        expect(
            isConverseConversation({ messages: [{ role: 'assistant', content: [] }, ...legacyConversation.messages] }),
        ).toBe(true);
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
        expect(requestOptions.headers).toMatchObject({ 'X-Amzn-Bedrock-Service-Tier': 'priority' });
        expect(logger.warn).toHaveBeenCalledWith(
            expect.objectContaining({ option_names: ['max_tokens'], reason: 'claude_output_limit' }),
            expect.any(String),
        );
        expect(completion.result).toEqual([{ type: 'text', value: 'ok' }]);
        expect(completion.service_tier).toBe('priority');
        expect(completion.conversation).toMatchObject({ messages: [{ role: 'user' }, { role: 'assistant' }] });
    });

    it('streams Claude through the shared Messages helpers', async () => {
        const { driver, stream } = driverWithClaudeClient();
        Reflect.set(driver, '_anthropicService', {
            messages: {
                stream: (...args: [unknown, unknown?]) => {
                    stream(...args);
                    return Object.assign((async function* () {})(), {
                        finalMessage: async () => message(),
                        withResponse: async () => ({ response: tierResponse() }),
                    });
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

    it('resolves credentials through the AWS SDK client provider, which caches them', async () => {
        const credentials = vi.fn(async () => ({ accessKeyId: 'k', secretAccessKey: 's' }));
        const driver = new BedrockDriver({ region: 'us-east-1', credentials });
        const service = Reflect.get(driver, 'getAnthropicService').call(driver);
        const provider = await service.providerChainResolver();
        expect(provider).toBe(driver.getExecutor().config.credentials);
        await expect(provider()).resolves.toMatchObject({ accessKeyId: 'k' });
        expect(credentials).toHaveBeenCalled();
    });

    it('keeps explicit credentials when a Bedrock bearer token is set in the environment', () => {
        vi.stubEnv('AWS_BEARER_TOKEN_BEDROCK', 'env-token');
        try {
            const driver = new BedrockDriver({
                region: 'us-east-1',
                credentials: { accessKeyId: 'test-key', secretAccessKey: 'test-secret' },
            });
            const service = Reflect.get(driver, 'getAnthropicService').call(driver);
            expect(service.authToken).toBeNull();
            expect(service.awsAccessKey).toBe('test-key');
        } finally {
            vi.unstubAllEnvs();
        }
    });

    it('keeps a configured credential provider when a Bedrock bearer token is set in the environment', () => {
        vi.stubEnv('AWS_BEARER_TOKEN_BEDROCK', 'env-token');
        try {
            const credentials = async () => ({ accessKeyId: 'k', secretAccessKey: 's' });
            const driver = new BedrockDriver({ region: 'us-east-1', credentials });
            const service = Reflect.get(driver, 'getAnthropicService').call(driver);
            expect(service.authToken).toBeNull();
        } finally {
            vi.unstubAllEnvs();
        }
    });

    it('applies the Bedrock retry rules to the exception named in x-amzn-errortype', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const error = new BadRequestError(
            400,
            { message: 'quota' },
            'quota',
            new Headers({
                'x-amzn-errortype': 'ServiceQuotaExceededException:http://internal.amazon.com/coral/',
                'x-amzn-requestid': 'aws-req-1',
            }),
        );
        const formatted = driver.formatLlumiverseError(error, {
            provider: 'bedrock',
            model: MODEL,
            operation: 'execute',
        });
        expect(formatted.retryable).toBe(true);
        expect(formatted.name).toBe('ServiceQuotaExceededException');
        expect(formatted.message).toBe('[bedrock] ServiceQuotaExceededException: [400] quota (Request ID: aws-req-1)');
    });

    it('does not treat another SDK error named APIError as an Anthropic error', () => {
        class APIError extends Error {
            status = 429;
        }
        const error = new APIError('other sdk');
        expect(isAnthropicSdkError(error)).toBe(false);
        expect(() =>
            formatAnthropicLlumiverseError(error, { provider: 'bedrock', model: MODEL, operation: 'execute' }),
        ).toThrow(error);
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
        // Not in the Bedrock rules: a client fault, as on the Converse route.
        expect(driver.formatLlumiverseError(streamError('modelTimeoutException'), context).retryable).toBe(false);
        const throttled = driver.formatLlumiverseError(streamError('throttlingException'), context);
        expect(throttled.retryable).toBe(true);
        expect(throttled.message).toBe('[bedrock] ThrottlingException: m');
        // A client fault in the Bedrock API model, non-retryable as on the Converse route.
        expect(driver.formatLlumiverseError(streamError('modelStreamErrorException'), context).retryable).toBe(false);
        expect(driver.formatLlumiverseError(streamError('validationException'), context).retryable).toBe(false);
    });

    it('retries a corrupt stream chunk', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const error = new AnthropicError('Could not parse a Bedrock chunk into JSON: {');
        const context = { provider: 'bedrock', model: MODEL, operation: 'stream' as const };
        expect(driver.formatLlumiverseError(error, context).retryable).toBe(true);
        expect(driver.formatLlumiverseError(new AnthropicError('other'), context).retryable).toBe(false);
    });

    it('drops the Messages API client with the other provider resources', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const first = Reflect.get(driver, 'getAnthropicService').call(driver);
        Reflect.get(driver, 'destroyProviderResources').call(driver);
        expect(Reflect.get(driver, 'getAnthropicService').call(driver)).not.toBe(first);
    });

    it('sends a Claude prompt with no messages to the Messages API', async () => {
        const { driver, stream } = driverWithClaudeClient();
        await driver.requestTextCompletion({ messages: [] }, { model: MODEL });
        expect(stream).toHaveBeenCalledOnce();
    });

    it('recognises errors from another copy of the Anthropic SDK', () => {
        // Platform SDKs can resolve their own Anthropic SDK copy, whose classes fail instanceof checks here.
        class AnthropicError extends Error {}
        class APIError extends AnthropicError {
            status = 429;
            error = { error: { type: 'rate_limit_error', message: 'slow down' } };
            headers = new Headers();
            requestID = 'req-1';
        }
        class RateLimitError extends APIError {}
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const formatted = driver.formatLlumiverseError(new RateLimitError('slow down'), {
            provider: 'bedrock',
            model: MODEL,
            operation: 'execute',
        });
        expect(formatted.retryable).toBe(true);
        expect(formatted.message).toContain('slow down');
    });

    it('formats Claude prompts for debugging', async () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const prompt = await driver.createPrompt(segments, { model: MODEL });
        expect(driver.formatDebugPrompt(prompt)).toEqual(prompt);
    });
});
