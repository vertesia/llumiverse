import type { Message, RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages.js';
import type { ConverseRequest, ConverseStreamOutput } from '@aws-sdk/client-bedrock-runtime';
import { FinishReason } from '@google/genai';
import type { Logger } from '@llumiverse/core';
import type { ServerSentEvent } from '@vertesia/api-fetch-client';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { BedrockDriver } from '../bedrock/index.js';
import { mapResponseStream } from '../openai/index.js';
import { OpenAIDriver } from '../openai/openai.js';
import {
    type OpenAIChatCompletionsPayload,
    OpenAIChatCompletionsProtocol,
    type OpenAIChatCompletionsResponse,
} from '../openai/openai_chat_completions.js';
import type { VertexAIDriver } from '../vertexai/index.js';
import { GeminiModelDefinition } from '../vertexai/models/gemini.js';
import { executeClaudeCompletion, streamClaudeCompletion } from './claude-messages.js';
import { logModelRefusal, MODEL_REFUSAL_LOG_MESSAGE } from './refusal-log.js';

function testLogger() {
    return { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() } satisfies Logger;
}

async function drain(stream: AsyncIterable<unknown>) {
    for await (const _chunk of stream) {
        // drain
    }
}

describe('logModelRefusal', () => {
    it('logs the context at info level with the provider details unchanged', () => {
        const logger = testLogger();
        const details = { category: 'cyber', explanation: 'policy', future_field: { nested: true } };

        logModelRefusal(logger, { provider: 'anthropic', model: 'claude', streaming: false, details });

        expect(logger.info).toHaveBeenCalledWith(
            { provider: 'anthropic', model: 'claude', streaming: false, details },
            MODEL_REFUSAL_LOG_MESSAGE,
        );
        expect(logger.warn).not.toHaveBeenCalled();
    });

    it('does nothing without a logger', () => {
        expect(() => logModelRefusal(undefined, { provider: 'p', model: 'm', streaming: true })).not.toThrow();
    });
});

describe('Claude refusal logging', () => {
    const stopDetails = { type: 'refusal', category: 'cyber', explanation: 'Declined for policy reasons.' };
    const refusalMessage = {
        id: 'msg-refusal',
        type: 'message',
        role: 'assistant',
        model: 'claude-test',
        content: [],
        stop_reason: 'refusal',
        stop_details: stopDetails,
        stop_sequence: null,
        usage: { input_tokens: 2, output_tokens: 0 },
    } as unknown as Message;
    const prompt = { messages: [{ role: 'user' as const, content: 'question' }] };

    function clientFor(message: Message, events: RawMessageStreamEvent[] = []) {
        const stream = () => ({
            async *[Symbol.asyncIterator]() {
                for (const event of events) yield event;
            },
            async finalMessage() {
                return message;
            },
        });
        return { messages: { stream } } as never;
    }

    it('logs stop_details for a non-streaming refusal', async () => {
        const logger = testLogger();
        await executeClaudeCompletion(clientFor(refusalMessage), prompt, { model: 'claude-test' }, logger, 'vertexai');

        expect(logger.info).toHaveBeenCalledWith(
            {
                provider: 'vertexai',
                model: 'claude-test',
                streaming: false,
                finish_reason: 'refusal',
                provider_finish_reason: 'refusal',
                response_id: 'msg-refusal',
                details: stopDetails,
            },
            MODEL_REFUSAL_LOG_MESSAGE,
        );
    });

    it('logs stop_details for a streaming refusal', async () => {
        const logger = testLogger();
        const events = [
            { type: 'message_start', message: { ...refusalMessage, stop_reason: null, stop_details: null } },
            {
                type: 'message_delta',
                delta: { stop_reason: 'refusal', stop_sequence: null, stop_details: stopDetails },
                usage: { output_tokens: 0 },
            },
            { type: 'message_stop' },
        ] as unknown as RawMessageStreamEvent[];

        const stream = await streamClaudeCompletion(
            clientFor(refusalMessage, events),
            prompt,
            { model: 'claude-test' },
            logger,
            'vertexai',
        );
        await drain(stream);

        expect(logger.info).toHaveBeenCalledWith(
            expect.objectContaining({ streaming: true, response_id: 'msg-refusal', details: stopDetails }),
            MODEL_REFUSAL_LOG_MESSAGE,
        );
    });

    it('does not log a normal completion', async () => {
        const logger = testLogger();
        const message = { ...refusalMessage, stop_reason: 'end_turn', stop_details: null } as unknown as Message;
        await executeClaudeCompletion(clientFor(message), prompt, { model: 'claude-test' }, logger);

        expect(logger.info).not.toHaveBeenCalledWith(expect.anything(), MODEL_REFUSAL_LOG_MESSAGE);
    });
});

describe('Bedrock Converse refusal logging', () => {
    const MODEL = 'anthropic.claude-sonnet-4-6-v1:0';

    function driverWith(logger: Logger) {
        const driver = new BedrockDriver({ region: 'us-east-1', logger });
        const converse = vi.fn(async () => ({
            output: { message: { role: 'assistant', content: [] } },
            stopReason: 'guardrail_intervened',
            additionalModelResponseFields: { stop_details: { category: 'bio' } },
            usage: { inputTokens: 1, outputTokens: 0, totalTokens: 1 },
        }));
        const converseStream = vi.fn(async (_request: ConverseRequest) => ({
            $metadata: { requestId: 'req-stream' },
            stream: (async function* () {
                const events: ConverseStreamOutput[] = [
                    { messageStart: { role: 'assistant' } },
                    { messageStop: { stopReason: 'content_filtered' } },
                ];
                for (const event of events) yield event;
            })(),
        }));
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ converse, converseStream, destroy: vi.fn() }),
        });
        return driver;
    }

    it('logs a guardrail stop from a non-streaming response', () => {
        const logger = testLogger();
        driverWith(logger).getExtractedExecution(
            {
                stopReason: 'guardrail_intervened',
                additionalModelResponseFields: { stop_details: { category: 'bio' } },
                $metadata: { requestId: 'req-1' },
            } as never,
            undefined,
            { model: MODEL },
        );

        expect(logger.info).toHaveBeenCalledWith(
            {
                provider: 'bedrock',
                model: MODEL,
                streaming: false,
                finish_reason: 'guardrail_intervened',
                provider_finish_reason: 'guardrail_intervened',
                response_id: 'req-1',
                details: { additionalModelResponseFields: { stop_details: { category: 'bio' } } },
            },
            MODEL_REFUSAL_LOG_MESSAGE,
        );
    });

    it('logs a content-filtered stop from a stream', async () => {
        const logger = testLogger();
        const stream = await driverWith(logger).requestTextCompletionStream(
            { modelId: MODEL, messages: [{ role: 'user', content: [{ text: 'question' }] }] },
            { model: MODEL },
        );
        await drain(stream);

        expect(logger.info).toHaveBeenCalledWith(
            expect.objectContaining({
                streaming: true,
                provider_finish_reason: 'content_filtered',
                response_id: 'req-stream',
            }),
            MODEL_REFUSAL_LOG_MESSAGE,
        );
    });
});

describe('OpenAI Responses refusal logging', () => {
    it('logs refusal parts before rejecting a refusal-only response', () => {
        const logger = testLogger();
        const driver = new OpenAIDriver({ apiKey: 'test-key', logger });
        const refusal = { type: 'refusal', refusal: 'I cannot help with that.' };
        const response = {
            id: 'resp-1',
            status: 'completed',
            output: [{ type: 'message', role: 'assistant', content: [refusal] }],
        } as unknown as OpenAI.Responses.Response;

        expect(() => driver.extractDataFromResponse({ model: 'gpt-test' }, response)).toThrow(
            'Response is not valid: no data',
        );
        expect(logger.info).toHaveBeenCalledWith(
            expect.objectContaining({
                provider: 'openai',
                streaming: false,
                response_id: 'resp-1',
                details: { refusals: [refusal] },
            }),
            MODEL_REFUSAL_LOG_MESSAGE,
        );
    });

    it('logs a streamed refusal before the existing error', async () => {
        const logger = testLogger();
        async function* events() {
            yield { type: 'response.created', response: { id: 'resp-stream' } } as OpenAI.Responses.ResponseStreamEvent;
            yield { type: 'response.refusal.delta', delta: 'No.' } as OpenAI.Responses.ResponseStreamEvent;
            yield { type: 'response.refusal.done', refusal: 'No.' } as OpenAI.Responses.ResponseStreamEvent;
        }
        const stream = mapResponseStream(events(), false, undefined, undefined, {
            logger,
            provider: 'openai',
            model: 'gpt-test',
        });

        await expect(drain(stream)).rejects.toThrow('Model refused');
        expect(logger.info).toHaveBeenCalledWith(
            expect.objectContaining({
                streaming: true,
                response_id: 'resp-stream',
                details: { type: 'refusal', refusal: 'No.' },
            }),
            MODEL_REFUSAL_LOG_MESSAGE,
        );
    });

    it('logs a content-filtered incomplete stream', async () => {
        const logger = testLogger();
        const incomplete_details = { reason: 'content_filter' };
        async function* events() {
            yield {
                type: 'response.incomplete',
                response: { id: 'resp-2', status: 'incomplete', incomplete_details, output: [] },
            } as unknown as OpenAI.Responses.ResponseStreamEvent;
        }
        const stream = mapResponseStream(events(), false, undefined, undefined, {
            logger,
            provider: 'openai',
            model: 'gpt-test',
        });

        await drain(stream).catch(() => undefined);
        expect(logger.info).toHaveBeenCalledWith(
            expect.objectContaining({
                finish_reason: 'content_filter',
                response_id: 'resp-2',
                details: { incomplete_details },
            }),
            MODEL_REFUSAL_LOG_MESSAGE,
        );
    });
});

describe('OpenAI Chat Completions refusal logging', () => {
    type TestDriver = { logger: Logger; provider: string };

    class TestProtocol extends OpenAIChatCompletionsProtocol<TestDriver> {
        constructor(
            private readonly response?: OpenAIChatCompletionsResponse,
            private readonly events: ServerSentEvent[] = [],
        ) {
            super({ modelName: 'test/model' });
        }

        protected async postChatCompletion(
            _driver: TestDriver,
            _payload: OpenAIChatCompletionsPayload,
        ): Promise<OpenAIChatCompletionsResponse> {
            if (!this.response) throw new Error('Missing test response');
            return this.response;
        }

        protected async postChatCompletionStream(): Promise<ReadableStream> {
            const events = this.events;
            return new ReadableStream<ServerSentEvent>({
                start(controller) {
                    for (const event of events) controller.enqueue(event);
                    controller.close();
                },
            });
        }
    }

    const prompt = { _is_openai_chat_completions: true as const, messages: [{ role: 'user', content: 'Hi' }] };
    const options = { model: 'test/model', model_options: { _option_id: 'text-fallback' as const } };

    it('logs message.refusal before rejecting a refusal-only response', async () => {
        const logger = testLogger();
        const model = new TestProtocol({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 1,
            model: 'test/model',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant', content: null, refusal: 'I cannot help.' },
                    finish_reason: 'stop',
                },
            ],
        });

        await expect(
            model.requestTextCompletion({ logger, provider: 'azure_foundry' }, prompt as never, options),
        ).rejects.toThrow('no data');
        expect(logger.info).toHaveBeenCalledWith(
            expect.objectContaining({
                provider: 'azure_foundry',
                streaming: false,
                response_id: 'chatcmpl-1',
                details: { refusal: 'I cannot help.' },
            }),
            MODEL_REFUSAL_LOG_MESSAGE,
        );
    });

    it('logs a content_filter finish from a stream', async () => {
        const logger = testLogger();
        const chunk = (choice: object) => ({
            type: 'event' as const,
            data: JSON.stringify({ id: 'chatcmpl-2', object: 'chat.completion.chunk', choices: [choice] }),
        });
        const model = new TestProtocol(undefined, [
            chunk({ index: 0, delta: { content: 'partial' }, finish_reason: null }),
            chunk({ index: 0, delta: {}, finish_reason: 'content_filter' }),
            // Some providers repeat finish_reason on a trailing usage chunk.
            chunk({ index: 0, delta: {}, finish_reason: 'content_filter' }),
        ]);

        const stream = await model.requestTextCompletionStream(
            { logger, provider: 'openai_compatible' },
            prompt as never,
            options,
        );
        await drain(stream);

        expect(logger.info).toHaveBeenCalledTimes(1);
        expect(logger.info).toHaveBeenCalledWith(
            expect.objectContaining({
                streaming: true,
                provider_finish_reason: 'content_filter',
                details: { refusal: null },
            }),
            MODEL_REFUSAL_LOG_MESSAGE,
        );
    });
});

describe('Gemini refusal logging', () => {
    const modelDef = new GeminiModelDefinition('gemini-2.0-flash');
    const prompt = { contents: [{ role: 'user', parts: [{ text: 'question' }] }] };
    const options = { model: 'publishers/google/models/gemini-2.0-flash' };

    function driverReturning(logger: Logger, response: object) {
        return {
            logger,
            provider: 'vertexai',
            getGoogleGenAIClient: () => ({
                models: {
                    generateContent: async () => response,
                    generateContentStream: async () =>
                        (async function* () {
                            yield response;
                        })(),
                },
            }),
        } as unknown as VertexAIDriver;
    }

    it('logs safety metadata without content before throwing', async () => {
        const logger = testLogger();
        const safetyRatings = [{ category: 'HARM_CATEGORY_DANGEROUS_CONTENT', probability: 'HIGH', blocked: true }];
        const driver = driverReturning(logger, {
            responseId: 'gemini-1',
            candidates: [
                {
                    finishReason: FinishReason.SAFETY,
                    finishMessage: 'blocked',
                    safetyRatings,
                    content: { role: 'model', parts: [{ text: 'secret output' }] },
                },
            ],
        });

        await expect(modelDef.requestTextCompletion(driver, prompt, options)).rejects.toThrow();
        expect(logger.info).toHaveBeenCalledWith(
            {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                streaming: false,
                finish_reason: FinishReason.SAFETY,
                provider_finish_reason: FinishReason.SAFETY,
                response_id: 'gemini-1',
                details: { finishReason: FinishReason.SAFETY, finishMessage: 'blocked', safetyRatings },
            },
            MODEL_REFUSAL_LOG_MESSAGE,
        );
    });

    it('logs prompt feedback when the prompt is blocked', async () => {
        const logger = testLogger();
        const promptFeedback = { blockReason: 'PROHIBITED_CONTENT', blockReasonMessage: 'Prompt blocked.' };
        const driver = driverReturning(logger, { responseId: 'gemini-2', promptFeedback });

        await modelDef.requestTextCompletion(driver, structuredClone(prompt), options);
        await drain(await modelDef.requestTextCompletionStream(driver, structuredClone(prompt), options));

        expect(logger.info).toHaveBeenCalledWith(
            expect.objectContaining({
                streaming: false,
                finish_reason: undefined,
                provider_finish_reason: 'PROHIBITED_CONTENT',
                response_id: 'gemini-2',
                details: promptFeedback,
            }),
            MODEL_REFUSAL_LOG_MESSAGE,
        );
        expect(logger.info).toHaveBeenCalledWith(
            expect.objectContaining({ streaming: true, finish_reason: 'PROHIBITED_CONTENT', details: promptFeedback }),
            MODEL_REFUSAL_LOG_MESSAGE,
        );
    });

    it('logs a refusal on a later candidate of a stream chunk', async () => {
        const logger = testLogger();
        const driver = driverReturning(logger, {
            candidates: [
                { finishReason: FinishReason.STOP, content: { role: 'model', parts: [{ text: 'ok' }] } },
                { index: 1, finishReason: FinishReason.SAFETY },
            ],
        });

        await drain(await modelDef.requestTextCompletionStream(driver, structuredClone(prompt), options)).catch(
            () => undefined,
        );
        expect(logger.info).toHaveBeenCalledWith(
            expect.objectContaining({ streaming: true, provider_finish_reason: FinishReason.SAFETY }),
            MODEL_REFUSAL_LOG_MESSAGE,
        );
    });

    it('does not log a normal stop', async () => {
        const logger = testLogger();
        const driver = driverReturning(logger, {
            candidates: [{ finishReason: FinishReason.STOP, content: { role: 'model', parts: [{ text: 'ok' }] } }],
        });

        await modelDef.requestTextCompletion(driver, prompt, options);
        expect(logger.info).not.toHaveBeenCalledWith(expect.anything(), MODEL_REFUSAL_LOG_MESSAGE);
    });
});
