import type Anthropic from '@anthropic-ai/sdk';
import {
    AnthropicError,
    APIConnectionError,
    APIError,
    BadRequestError,
    InternalServerError,
} from '@anthropic-ai/sdk/error';
import type { ConverseRequest, ConverseStreamOutput } from '@aws-sdk/client-bedrock-runtime';
import {
    type ExecutionOptions,
    LlumiverseError,
    PromptRole,
    type PromptSegment,
    type ToolDefinition,
} from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { type ClaudePrompt, formatAnthropicLlumiverseError, isAnthropicSdkError } from '../shared/claude-messages.js';
import { MID_CONVERSATION_TOOL_CHANGES_BETA } from '../shared/claude-tool-changes.js';
import { isConverseConversation, isMessagesApiConversation } from './converse.js';
import { type BedrockClaudePrompt, BedrockDriver, type BedrockPrompt } from './index.js';

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

    it('recognises a stored Converse conversation whose images were replaced by text placeholders', () => {
        const stripped = {
            messages: [
                { role: 'user', content: [{ text: 'Describe this.' }, { text: '[Image removed from conversation]' }] },
                { role: 'assistant', content: [{ text: 'A chart.' }] },
            ],
        };
        expect(isConverseConversation(stripped)).toBe(true);
        expect(isMessagesApiConversation(stripped)).toBe(false);
    });

    it('recognises a Messages API conversation that starts with string content', () => {
        const conversation = {
            messages: [
                { role: 'user', content: 'hi' },
                { role: 'assistant', content: [{ type: 'text', text: 'hello' }] },
            ],
        };
        expect(isConverseConversation(conversation)).toBe(false);
        expect(isMessagesApiConversation(conversation)).toBe(true);
        expect(isMessagesApiConversation({ messages: [{ role: 'user', content: 'hi' }] })).toBe(true);
    });

    it.each([
        ['text only', { role: 'assistant', content: [{ type: 'text', text: 'hello' }] }],
        ['tool use', { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'a', input: {} }] }],
        ['thinking', { role: 'assistant', content: [{ type: 'thinking', thinking: '…', signature: 's' }] }],
        [
            'a tool change',
            { role: 'system', content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'a' } }] },
        ],
    ])('recognises a Messages API conversation with %s', (_label, message) => {
        const conversation = { messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }, message] };
        expect(isMessagesApiConversation(conversation)).toBe(true);
    });

    it.each([undefined, null, {}, { messages: [] }])('treats %j as neither format with history', (conversation) => {
        expect(isMessagesApiConversation(conversation)).toBe(false);
    });
});

describe('Bedrock Claude on the Messages API', () => {
    it('formats new Claude conversations for the Messages API and keeps legacy ones on Converse', async () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });

        const fresh = await driver.createPrompt(segments, { model: MODEL });
        expect(fresh).toEqual({
            messages: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }],
            _is_claude_messages: true,
        });

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

    it('keeps the Bedrock request ID when no exception is named', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const error = new InternalServerError(
            503,
            { message: 'unavailable' },
            'unavailable',
            new Headers({ 'x-amzn-requestid': 'aws-req-2' }),
        );
        const formatted = driver.formatLlumiverseError(error, {
            provider: 'bedrock',
            model: MODEL,
            operation: 'execute',
        });
        expect(formatted.message).toContain('(Request ID: aws-req-2)');
        expect(formatted.retryable).toBe(true);
    });

    it('retries a client-side fault raised while reading the response stream', async () => {
        const { driver } = driverWithClaudeClient();
        Reflect.set(driver, '_anthropicService', {
            messages: {
                stream: () => ({
                    async *[Symbol.asyncIterator]() {
                        yield { type: 'message_start', message: message() };
                        throw new AnthropicError('Could not parse a Bedrock chunk into JSON: {');
                    },
                    finalMessage: async () => message(),
                    withResponse: async () => ({ response: tierResponse() }),
                }),
            },
        });
        const prompt = await driver.createPrompt(segments, { model: MODEL });
        const context = { provider: 'bedrock', model: MODEL, operation: 'execute' as const };
        const failure = await driver.requestTextCompletion(prompt, { model: MODEL }).catch((error: unknown) => error);
        const formatted = driver.formatLlumiverseError(failure, context);
        expect(formatted.retryable).toBe(true);
        expect(formatted.message).toContain('Could not parse a Bedrock chunk');
        // Outside the stream, a client-side SDK error is a configuration error.
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
        await driver.requestTextCompletion({ messages: [], _is_claude_messages: true }, { model: MODEL });
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

        const image = 'A'.repeat(5_000);
        const pdf = 'B'.repeat(5_000);
        const binary: BedrockClaudePrompt = {
            _is_claude_messages: true,
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'Compare these.' },
                        { type: 'image', source: { type: 'base64', media_type: 'image/png', data: image } },
                        { type: 'document', source: { type: 'base64', media_type: 'application/pdf', data: pdf } },
                    ],
                },
            ],
        };
        const snapshot = structuredClone(binary);
        const debug = JSON.stringify(driver.formatDebugPrompt(binary));
        expect(debug).toContain('Compare these.');
        expect(debug).not.toContain(image);
        expect(debug).not.toContain(pdf);
        expect(debug).toContain('AAAA');
        expect(debug).toContain('BBBB');
        expect(binary).toEqual(snapshot);
    });

    it('includes the cause of a connection error', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const cause = new Error('AccessDenied: not authorized to perform sts:AssumeRole');
        const formatted = driver.formatLlumiverseError(new APIConnectionError({ cause }), {
            provider: 'bedrock',
            model: MODEL,
            operation: 'execute',
        });
        expect(formatted.message).toContain('Connection error.');
        expect(formatted.message).toContain('AccessDenied: not authorized to perform sts:AssumeRole');
    });

    it('classifies an AWS credential failure behind a connection error with the AWS rules', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const cause = Object.assign(new Error('not authorized to perform sts:AssumeRoleWithWebIdentity'), {
            name: 'AccessDenied',
            $fault: 'client',
            $metadata: { httpStatusCode: 403 },
        });
        const formatted = driver.formatLlumiverseError(new APIConnectionError({ cause }), {
            provider: 'bedrock',
            model: MODEL,
            operation: 'execute',
        });
        expect(formatted.retryable).toBe(false);
        expect(formatted.code).toBe(403);
        expect(formatted.message).toContain('sts:AssumeRoleWithWebIdentity');
    });

    it('builds the streamed conversation in the format of the prompt', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const options = { model: MODEL } as ExecutionOptions;
        const messagesPrompt: BedrockClaudePrompt = {
            messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
            _is_claude_messages: true,
        };
        const messagesConversation = driver.buildStreamingConversation(
            messagesPrompt,
            [{ type: 'text', value: 'ok' }],
            [{ id: 't1', tool_name: 'lookup', tool_input: { q: 'x' } }],
            options,
        ) as ClaudePrompt;
        // The shared builder stores streamed text and tool calls as separate assistant messages.
        expect(messagesConversation.messages).toEqual([
            messagesPrompt.messages[0],
            { role: 'assistant', content: 'ok' },
            { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'lookup', input: { q: 'x' } }] },
        ]);

        const conversePrompt = { modelId: MODEL, messages: [{ role: 'user', content: [{ text: 'hi' }] }] };
        const converseConversation = driver.buildStreamingConversation(
            conversePrompt as BedrockPrompt,
            [{ type: 'text', value: 'ok' }],
            undefined,
            options,
        ) as ConverseRequest;
        expect(converseConversation.messages).toEqual([
            conversePrompt.messages[0],
            { role: 'assistant', content: [{ text: 'ok' }] },
        ]);
    });

    it('sends the tool-change beta and the service tier header together', async () => {
        const { driver, stream } = driverWithClaudeClient();
        const tool = (name: string): ToolDefinition => ({ name, input_schema: { type: 'object', properties: {} } });
        const modelOptions = { _option_id: 'bedrock-claude' as const, service_tier: 'priority' };
        const first = await driver.requestTextCompletion(await driver.createPrompt(segments, { model: MODEL }), {
            model: MODEL,
            model_options: modelOptions,
            tools: [tool('lookup')],
        });
        const options: ExecutionOptions = {
            model: MODEL,
            model_options: modelOptions,
            tools: [tool('lookup'), tool('list_items')],
            conversation: JSON.parse(JSON.stringify(first.conversation)),
        };
        const next = [{ role: PromptRole.user, content: 'more' }];
        await driver.requestTextCompletion(await driver.createPrompt(next, options), options);

        const [payload, requestOptions] = stream.mock.calls[1] as [
            { messages: { role: string }[] },
            { headers: Record<string, string> },
        ];
        expect(payload.messages.at(-1)).toMatchObject({ role: 'system' });
        expect(requestOptions.headers['anthropic-beta'].split(',')).toContain(MID_CONVERSATION_TOOL_CHANGES_BETA);
        expect(requestOptions.headers['X-Amzn-Bedrock-Service-Tier']).toBe('priority');
    });

    it.each([
        ['static credentials', { accessKeyId: 'test-key', secretAccessKey: 'test-secret' }],
        ['a credential provider', async () => ({ accessKeyId: 'k', secretAccessKey: 's' })],
    ])('prefers SigV4 on the AWS SDK clients with %s', async (_case, credentials) => {
        const driver = new BedrockDriver({ region: 'us-east-1', credentials });
        const preference = (config: unknown) =>
            (config as { authSchemePreference?: () => Promise<string[]> }).authSchemePreference?.();
        await expect(preference(driver.getExecutor().config)).resolves.toEqual(['sigv4']);
        await expect(preference(driver.getService().config)).resolves.toEqual(['sigv4']);
    });
});

describe('legacy Converse conversations with Claude on Bedrock', () => {
    const usage: ConverseStreamOutput = {
        metadata: { usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 }, metrics: { latencyMs: 1 } },
    };

    function converseDriver() {
        const converse = vi.fn(async (_request: ConverseRequest) => ({
            output: { message: { role: 'assistant', content: [{ text: 'ok' }] } },
            stopReason: 'end_turn',
            usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
            metrics: { latencyMs: 1 },
        }));
        const converseStream = vi.fn(async (_request: ConverseRequest) => ({
            stream: (async function* () {
                yield { contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'ok' } } };
                yield { messageStop: { stopReason: 'end_turn' } };
                yield usage;
            })(),
        }));
        const messagesStream = vi.fn();
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ converse, converseStream, destroy: vi.fn() }),
        });
        Reflect.set(driver, '_anthropicService', { messages: { stream: messagesStream } });
        return { driver, converse, converseStream, messagesStream };
    }

    it('continues a legacy conversation on Converse when executing', async () => {
        const { driver, converse, messagesStream } = converseDriver();
        const options = { model: MODEL, conversation: structuredClone(legacyConversation) } as ExecutionOptions;
        const completion = await driver.requestTextCompletion(await driver.createPrompt(segments, options), options);

        expect(messagesStream).not.toHaveBeenCalled();
        expect(converse).toHaveBeenCalledOnce();
        expect(converse.mock.calls[0][0].messages?.at(-1)?.content).toContainEqual({ text: 'hello' });
        expect(isConverseConversation(completion.conversation)).toBe(true);
    });

    it('continues a legacy conversation on Converse when streaming', async () => {
        const { driver, converseStream, messagesStream } = converseDriver();
        const options = { model: MODEL, conversation: structuredClone(legacyConversation) } as ExecutionOptions;
        const stream = await driver.requestTextCompletionStream(await driver.createPrompt(segments, options), options);
        for await (const _chunk of stream) {
            // drain
        }

        expect(messagesStream).not.toHaveBeenCalled();
        expect(converseStream).toHaveBeenCalledOnce();
        expect(isConverseConversation(await stream.finalizeConversation?.())).toBe(true);
    });

    it.each(['execute', 'stream'] as const)(
        'rejects a Messages API conversation on a non-Claude model (%s)',
        async (operation) => {
            const { driver, converse, converseStream } = converseDriver();
            const model = 'amazon.nova-pro-v1:0';
            const options = {
                model,
                conversation: {
                    messages: [
                        { role: 'user', content: [{ type: 'text', text: 'earlier' }] },
                        { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'lookup', input: {} }] },
                    ],
                },
            } as ExecutionOptions;
            const prompt = await driver.createPrompt(segments, { model });
            const request =
                operation === 'execute'
                    ? driver.requestTextCompletion(prompt, options)
                    : driver.requestTextCompletionStream(prompt, options);

            await expect(request).rejects.toBeInstanceOf(LlumiverseError);
            await expect(request).rejects.toMatchObject({
                name: 'ConversationFormatError',
                retryable: false,
                context: { provider: 'bedrock', model, operation },
            });
            expect(converse).not.toHaveBeenCalled();
            expect(converseStream).not.toHaveBeenCalled();
        },
    );
});
