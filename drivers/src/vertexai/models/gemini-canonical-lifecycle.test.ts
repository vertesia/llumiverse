import {
    type Content,
    FinishReason,
    type GenerateContentParameters,
    type GenerateContentResponse,
    type GoogleGenAI,
} from '@google/genai';
import { parseConversationDocument } from '@llumiverse/conversation';
import { type ExecutionOptions, PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { VertexAIDriver } from '../index.js';
import { GeminiModelDefinition } from './gemini.js';
import {
    exportLegacyGeminiConversation,
    GEMINI_GENERATE_CONTENT_PROTOCOL,
    prepareGeminiCanonicalState,
} from './gemini-conversation-adapter.js';

const MODEL = 'publishers/google/models/gemini-2.5-pro';

type Generate = (request: GenerateContentParameters) => Promise<GenerateContentResponse>;
type GenerateStream = (request: GenerateContentParameters) => Promise<AsyncIterable<GenerateContentResponse>>;

class TestGeminiDriver extends VertexAIDriver {
    constructor(
        private readonly generate: Generate,
        private readonly generateStream: GenerateStream = async () => (async function* () {})(),
    ) {
        super({ project: 'test-project', region: 'global', geminiContextCache: false });
    }

    override getGoogleGenAIClient(): GoogleGenAI {
        return {
            models: {
                generateContent: this.generate,
                generateContentStream: this.generateStream,
            },
        } as unknown as GoogleGenAI;
    }
}

function runtimeOptions(input: {
    flow: string;
    operation: string;
    attempt: string;
    recorded_at: string;
    conversation?: unknown;
}): ExecutionOptions {
    return {
        model: MODEL,
        ...(input.conversation === undefined ? {} : { conversation: input.conversation }),
        conversation_runtime: {
            conversation_id: `conversation:${input.flow}`,
            request_id: `request:${input.flow}:${input.operation}`,
            attempt_id: `attempt:${input.flow}:${input.attempt}`,
            input_operation_id: `input:${input.flow}:${input.operation}`,
            response_operation_id: `response:${input.flow}:${input.operation}`,
            recorded_at: input.recorded_at,
            started_at: input.recorded_at,
            completed_at: input.recorded_at,
        },
    };
}

function response(input: {
    id: string;
    content?: Content;
    finish_reason?: FinishReason;
    usage?: GenerateContentResponse['usageMetadata'];
}): GenerateContentResponse {
    return {
        responseId: input.id,
        modelVersion: 'gemini-2.5-pro-002',
        candidates:
            input.content === undefined
                ? []
                : [
                      {
                          finishReason: input.finish_reason ?? FinishReason.STOP,
                          content: input.content,
                          safetyRatings: [],
                      },
                  ],
        usageMetadata: input.usage ?? {
            promptTokenCount: 100,
            cachedContentTokenCount: 25,
            candidatesTokenCount: 13,
            thoughtsTokenCount: 7,
            totalTokenCount: 120,
            trafficType: 'ON_DEMAND_PRIORITY',
        },
    } as unknown as GenerateContentResponse;
}

function requestContents(request: GenerateContentParameters): Content[] {
    if (!Array.isArray(request.contents)) throw new Error('expected Gemini content array');
    return request.contents as Content[];
}

async function drain(stream: AsyncIterable<unknown>): Promise<void> {
    for await (const _chunk of stream) {
        // Drain provider or recovered output before finalizing its conversation.
    }
}

function latestGeneratedJson(value: unknown): unknown {
    const document = parseConversationDocument(value);
    for (let index = document.turns.length - 1; index >= 0; index -= 1) {
        const turn = document.turns[index];
        if (turn.kind !== 'agent' || turn.provenance.type !== 'generated') continue;
        return turn.blocks.find((block) => block.type === 'json')?.value;
    }
    return undefined;
}

describe('Gemini canonical lifecycle', () => {
    it('validates structured sync output, preserves usage, and durably recovers an accepted response', async () => {
        const raw = '{ "answer" : "Tokyo", "note" : null }';
        const nativeResponse = response({
            id: 'response-structured',
            content: { role: 'model', parts: [{ text: raw }] },
        });
        const generate = vi.fn<Generate>(async () => nativeResponse);
        const driver = new TestGeminiDriver(generate);
        const segments = [{ role: PromptRole.user, content: 'Return JSON.' }];
        const result_schema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' }, note: { type: 'null' } },
            required: ['answer', 'note'],
            additionalProperties: false,
        };
        const first = await driver.execute(segments, {
            ...runtimeOptions({
                flow: 'structured',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:00:00.000Z',
            }),
            result_schema,
        });

        expect(first.result).toEqual([{ type: 'json', value: { answer: 'Tokyo', note: null } }]);
        expect(first.token_usage).toEqual({
            total: 120,
            prompt: 100,
            prompt_cached: 25,
            prompt_new: 75,
            result: 20,
        });
        expect(first.service_tier).toBe('priority');
        expect(latestGeneratedJson(first.conversation)).toEqual({ answer: 'Tokyo', note: null });
        const document = parseConversationDocument(JSON.parse(JSON.stringify(first.conversation)));
        const generation = Object.values(document.generations).find(
            (candidate) => candidate.record_source === 'executed',
        );
        expect(generation).toMatchObject({
            provider: 'vertexai',
            protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
            requested_model: MODEL,
            resolved_model: 'gemini-2.5-pro-002',
            provider_response_id: 'response-structured',
            usage: {
                input_tokens: 100,
                cache_read_tokens: 25,
                input_new_tokens: 75,
                output_tokens: 20,
                reasoning_tokens: 7,
                total_tokens: 120,
            },
        });
        expect(exportLegacyGeminiConversation(document)._arrayConversation.at(-1)).toEqual(
            nativeResponse.candidates?.[0]?.content,
        );

        const retried = await driver.execute(segments, {
            ...runtimeOptions({
                flow: 'structured',
                operation: 'generate',
                attempt: 'retry',
                recorded_at: '2026-09-30T01:05:00.000Z',
                conversation: document,
            }),
            result_schema,
        });
        expect(retried.result).toEqual(first.result);
        expect(retried.token_usage).toEqual(first.token_usage);
        expect(retried.service_tier).toBe(first.service_tier);
        expect(retried.conversation).toEqual(document);
        expect(generate).toHaveBeenCalledTimes(1);
    });

    it('returns direct canonical structured output and retries without another provider request', async () => {
        const nativeResponse = response({
            id: 'response-direct-structured',
            content: { role: 'model', parts: [{ text: '{"answer":"Tokyo"}' }] },
        });
        const generate = vi.fn<Generate>(async () => nativeResponse);
        const driver = new TestGeminiDriver(generate);
        const segments = [{ role: PromptRole.user, content: 'Return JSON.' }];
        const result_schema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const first = await driver.executeCanonical(segments, {
            ...runtimeOptions({
                flow: 'direct-structured',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:06:00.000Z',
            }),
            result_schema,
        });
        expect(first.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'json', value: { answer: 'Tokyo' } }),
        );
        expect(first.accepted_output.generation).toMatchObject({
            status: 'completed',
            usage: { input_tokens: 100, output_tokens: 20, total_tokens: 120 },
        });
        expect(first.service_tier).toBe('priority');
        expect(exportLegacyGeminiConversation(first.conversation)._arrayConversation.at(-1)).toEqual(
            nativeResponse.candidates?.[0]?.content,
        );

        const retried = await driver.executeCanonical(segments, {
            ...runtimeOptions({
                flow: 'direct-structured',
                operation: 'generate',
                attempt: 'retry',
                recorded_at: '2026-09-30T01:07:00.000Z',
                conversation: first.conversation,
            }),
            result_schema,
        });
        expect(retried.conversation).toEqual(first.conversation);
        expect(retried.service_tier).toBe(first.service_tier);
        await expect(
            driver.executeCanonical(segments, {
                ...runtimeOptions({
                    flow: 'direct-structured',
                    operation: 'generate',
                    attempt: 'changed-options',
                    recorded_at: '2026-09-30T01:08:00.000Z',
                    conversation: first.conversation,
                }),
                result_schema,
                model_options: { _option_id: 'vertexai-gemini', temperature: 0.2 },
            }),
        ).rejects.toThrow('incompatible request identity');
        expect(generate).toHaveBeenCalledOnce();
    });

    it('marks invalid required structured output failed in direct sync and stream results', async () => {
        const invalid = response({
            id: 'response-direct-invalid',
            content: { role: 'model', parts: [{ text: '{"wrong":42}' }] },
        });
        const result_schema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const syncDriver = new TestGeminiDriver(async () => invalid);
        const sync = await syncDriver.executeCanonical([{ role: PromptRole.user, content: 'Return JSON.' }], {
            ...runtimeOptions({
                flow: 'direct-invalid-sync',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:08:00.000Z',
            }),
            result_schema,
        });
        expect(sync.accepted_output.generation.status).toBe('failed');
        expect(sync.accepted_output.turn.status).toBe('failed');
        expect(sync.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'text', text: '{"wrong":42}' }),
        );

        const generateStream = vi.fn<GenerateStream>(async () =>
            (async function* () {
                yield {
                    candidates: [{ content: { role: 'model', parts: [{ text: '{"wrong":42}' }] } }],
                } as GenerateContentResponse;
                yield invalid;
                yield {
                    usageMetadata: {
                        promptTokenCount: 9,
                        cachedContentTokenCount: 3,
                        candidatesTokenCount: 4,
                        totalTokenCount: 13,
                        trafficType: 'ON_DEMAND_FLEX',
                    },
                } as GenerateContentResponse;
            })(),
        );
        const streamDriver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const stream = await streamDriver.streamCanonical([{ role: PromptRole.user, content: 'Return JSON.' }], {
            ...runtimeOptions({
                flow: 'direct-invalid-stream',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:09:00.000Z',
            }),
            result_schema,
        });
        for await (const _chunk of stream) {
            // Drain the provider preview so canonical finalization runs.
        }
        expect(stream.completion?.accepted_output.generation).toMatchObject({
            status: 'failed',
            usage: { input_tokens: 9, output_tokens: 4, total_tokens: 13 },
        });
        expect(stream.completion?.accepted_output.turn.status).toBe('failed');
        expect(stream.completion?.service_tier).toBe('flex');
    });

    it('records a Gemini max-token terminal as interrupted and cancelled', async () => {
        const driver = new TestGeminiDriver(async () =>
            response({
                id: 'response-cutoff',
                content: { role: 'model', parts: [{ text: 'partial' }] },
                finish_reason: FinishReason.MAX_TOKENS,
            }),
        );
        const result = await driver.executeCanonical([{ role: PromptRole.user, content: 'Continue.' }], {
            ...runtimeOptions({
                flow: 'direct-cutoff',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:10:00.000Z',
            }),
        });

        expect(result.accepted_output.turn.status).toBe('interrupted');
        expect(result.accepted_output.generation).toMatchObject({ status: 'cancelled', finish_reason: 'length' });
    });

    it('aborts a pending direct canonical Gemini read before iterator cleanup', async () => {
        let providerSignal: AbortSignal | undefined;
        const generateStream = vi.fn<GenerateStream>(async (request) => {
            providerSignal = request.config?.abortSignal;
            return {
                [Symbol.asyncIterator]() {
                    return {
                        next: () =>
                            new Promise<IteratorResult<GenerateContentResponse>>((resolve) => {
                                providerSignal?.addEventListener(
                                    'abort',
                                    () => resolve({ done: true, value: undefined }),
                                    { once: true },
                                );
                            }),
                        return: async () => ({ done: true, value: undefined }),
                    };
                },
            };
        });
        const driver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const stream = await driver.streamCanonical(
            [{ role: PromptRole.user, content: 'Wait.' }],
            runtimeOptions({
                flow: 'direct-cancel',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:11:00.000Z',
            }),
        );
        const iterator = stream[Symbol.asyncIterator]();
        const pending = iterator.next();
        await stream.cancel();
        await expect(pending).resolves.toMatchObject({ done: true });
        expect(providerSignal?.aborted).toBe(true);
        expect(stream.completion).toBeUndefined();
    });

    it.each([
        ['array', '[1,null]', { type: 'array' }, [1, null]],
        ['null', 'null', { type: 'null' }, null],
        ['string', '"Tokyo"', { type: 'string' }, 'Tokyo'],
        ['number', '42', { type: 'number' }, 42],
        ['boolean', 'true', { type: 'boolean' }, true],
    ] as const)('persists a top-level JSON %s as canonical structured output', async (label, raw, schema, expected) => {
        const nativeResponse = response({
            id: `response-${label}`,
            content: { role: 'model', parts: [{ text: raw }] },
        });
        const driver = new TestGeminiDriver(async () => nativeResponse);
        const completion = await driver.execute([{ role: PromptRole.user, content: `Return a JSON ${label}.` }], {
            ...runtimeOptions({
                flow: `structured-${label}`,
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:10:00.000Z',
            }),
            result_schema: schema,
        });

        expect(completion.result).toEqual([{ type: 'json', value: expected }]);
        expect(latestGeneratedJson(completion.conversation)).toEqual(expected);
        expect(
            exportLegacyGeminiConversation(parseConversationDocument(completion.conversation))._arrayConversation.at(
                -1,
            ),
        ).toEqual(nativeResponse.candidates?.[0]?.content);
    });

    it('normalizes split streamed JSON around signed reasoning and recovers exact native parts', async () => {
        const firstFragment = '```json\n{"answer":';
        const secondFragment = '"Tokyo"}\n```';
        const nativeParts = [
            { text: firstFragment },
            { text: 'Check the requested shape.', thought: true, thoughtSignature: 'signed-structured-reasoning' },
            { text: secondFragment },
            { text: '', thoughtSignature: 'signed-empty-answer-terminal' },
            { text: '', thought: true, thoughtSignature: 'signed-empty-reasoning-terminal' },
        ];
        const generateStream = vi.fn<GenerateStream>(async () =>
            (async function* () {
                yield {
                    candidates: [{ content: { role: 'model', parts: [nativeParts[0]] } }],
                } as GenerateContentResponse;
                yield {
                    candidates: [{ content: { role: 'model', parts: [nativeParts[1]] } }],
                } as GenerateContentResponse;
                yield {
                    candidates: [{ content: { role: 'model', parts: [nativeParts[2]] } }],
                } as GenerateContentResponse;
                yield response({
                    id: 'response-structured-stream',
                    content: { role: 'model', parts: nativeParts.slice(3) },
                });
            })(),
        );
        const driver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const segments = [{ role: PromptRole.user, content: 'Return the city as JSON.' }];
        const result_schema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const first = await driver.stream(segments, {
            ...runtimeOptions({
                flow: 'structured-stream',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:20:00.000Z',
            }),
            result_schema,
        });
        await drain(first);

        expect(first.completion?.result).toEqual([
            { type: 'json', value: { answer: 'Tokyo' } },
            { type: 'thoughts', value: 'Check the requested shape.' },
        ]);
        expect(latestGeneratedJson(first.completion?.conversation)).toEqual({ answer: 'Tokyo' });
        const persisted = parseConversationDocument(JSON.parse(JSON.stringify(first.completion?.conversation)));
        expect(exportLegacyGeminiConversation(persisted)._arrayConversation.at(-1)?.parts).toEqual(nativeParts);

        const retried = await driver.stream(segments, {
            ...runtimeOptions({
                flow: 'structured-stream',
                operation: 'generate',
                attempt: 'retry',
                recorded_at: '2026-09-30T01:25:00.000Z',
                conversation: persisted,
            }),
            result_schema,
        });
        await drain(retried);
        expect(retried.completion?.result).toEqual(first.completion?.result);
        expect(retried.completion?.conversation).toEqual(persisted);
        expect(generateStream).toHaveBeenCalledTimes(1);
    });

    it('preserves native call identity and internal error status through a tool continuation', async () => {
        const requests: GenerateContentParameters[] = [];
        const replies = [
            response({
                id: 'response-tool-call',
                content: {
                    role: 'model',
                    parts: [
                        {
                            functionCall: { id: 'native-call-1', name: 'lookup', args: { city: 'Tokyo' } },
                            thoughtSignature: 'signed-call',
                        },
                    ],
                },
            }),
            response({
                id: 'response-tool-result',
                content: { role: 'model', parts: [{ text: 'The lookup failed.' }] },
            }),
        ];
        const generate = vi.fn<Generate>(async (request) => {
            requests.push(request);
            const next = replies.shift();
            if (next === undefined) throw new Error('unexpected provider call');
            return next;
        });
        const driver = new TestGeminiDriver(generate);
        const tools: NonNullable<ExecutionOptions['tools']> = [
            { name: 'lookup', input_schema: { type: 'object', additionalProperties: true } },
        ];
        const first = await driver.execute([{ role: PromptRole.user, content: 'Look it up.' }], {
            ...runtimeOptions({
                flow: 'tools',
                operation: 'ask',
                attempt: 'ask',
                recorded_at: '2026-09-30T02:00:00.000Z',
            }),
            tools,
        });
        expect(first.tool_use).toEqual([
            {
                id: 'native-call-1',
                tool_name: 'lookup',
                tool_input: { city: 'Tokyo' },
                thought_signature: 'signed-call',
            },
        ]);

        const second = await driver.execute(
            [
                {
                    role: PromptRole.tool,
                    tool_use_id: 'native-call-1',
                    tool_result_status: 'error',
                    content: '{"error":"not found"}',
                },
            ],
            {
                ...runtimeOptions({
                    flow: 'tools',
                    operation: 'continue',
                    attempt: 'continue',
                    recorded_at: '2026-09-30T02:01:00.000Z',
                    conversation: first.conversation,
                }),
                tools,
            },
        );
        const projectedResponse = requestContents(requests[1])
            .flatMap((content) => content.parts ?? [])
            .find((part) => part.functionResponse !== undefined)?.functionResponse;
        expect(projectedResponse).toEqual({
            id: 'native-call-1',
            name: 'lookup',
            response: { error: 'not found' },
        });
        expect(JSON.stringify(requests[1])).not.toContain('_llumiverse_tool_result_status');
        const document = parseConversationDocument(second.conversation);
        const toolTurn = document.turns.find((turn) => turn.kind === 'tool');
        expect(toolTurn?.blocks[0]).toMatchObject({ call_id: 'native-call-1', status: 'error' });
        expect(Object.values(document.execution_receipts)).toContainEqual(
            expect.objectContaining({ call_id: 'native-call-1', status: 'error' }),
        );
    });

    it('reuses an accepted input operation exactly once with ordered audio content', async () => {
        const prompt = {
            contents: [
                {
                    role: 'user' as const,
                    parts: [
                        { text: 'Describe this clip.' },
                        { inlineData: { data: 'YXVkaW8=', mimeType: 'audio/mpeg' } },
                    ],
                },
            ],
        };
        const firstOptions = runtimeOptions({
            flow: 'accepted-input',
            operation: 'generate',
            attempt: 'first',
            recorded_at: '2026-09-30T03:00:00.000Z',
        });
        const inputOnly = await prepareGeminiCanonicalState({
            conversation: undefined,
            prompt,
            options: firstOptions,
            provider: 'vertexai',
        });
        const requests: GenerateContentParameters[] = [];
        const generate = vi.fn<Generate>(async (request) => {
            requests.push(request);
            return response({ id: 'response-audio-input', content: { role: 'model', parts: [{ text: 'Audio.' }] } });
        });
        const model = new GeminiModelDefinition('gemini-2.5-pro');
        const driver = new TestGeminiDriver(generate);
        await model.requestTextCompletion(driver, prompt, {
            ...runtimeOptions({
                flow: 'accepted-input',
                operation: 'generate',
                attempt: 'retry',
                recorded_at: '2026-09-30T03:05:00.000Z',
                conversation: JSON.parse(JSON.stringify(inputOnly.document)),
            }),
        });

        expect(requests).toHaveLength(1);
        expect(requestContents(requests[0])).toEqual(prompt.contents);
        const audioParts = requestContents(requests[0])
            .flatMap((content) => content.parts ?? [])
            .filter((part) => part.inlineData?.mimeType === 'audio/mpeg');
        expect(audioParts).toHaveLength(1);
    });

    it('streams signed parts, uses trailing usage, and recovers without a second provider stream', async () => {
        const generateStream = vi.fn<GenerateStream>(async () =>
            (async function* () {
                yield {
                    candidates: [
                        {
                            content: {
                                role: 'model',
                                parts: [{ text: 'plan', thought: true, thoughtSignature: 'reasoning-signature' }],
                            },
                        },
                    ],
                } as unknown as GenerateContentResponse;
                yield {
                    candidates: [
                        {
                            content: {
                                role: 'model',
                                parts: [
                                    {
                                        functionCall: { id: 'native-stream-call', name: 'lookup', args: { q: 'x' } },
                                        thoughtSignature: 'call-signature',
                                    },
                                ],
                            },
                        },
                    ],
                } as unknown as GenerateContentResponse;
                yield {
                    responseId: 'response-stream',
                    modelVersion: 'gemini-2.5-pro-002',
                    candidates: [
                        {
                            finishReason: FinishReason.STOP,
                            content: { role: 'model', parts: [{ text: 'done' }] },
                        },
                    ],
                } as GenerateContentResponse;
                yield {
                    usageMetadata: {
                        promptTokenCount: 10,
                        cachedContentTokenCount: 2,
                        candidatesTokenCount: 3,
                        thoughtsTokenCount: 1,
                        totalTokenCount: 14,
                        trafficType: 'ON_DEMAND_FLEX',
                    },
                } as GenerateContentResponse;
            })(),
        );
        const driver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const model = new GeminiModelDefinition('gemini-2.5-pro');
        const prompt = { contents: [{ role: 'user' as const, parts: [{ text: 'Stream.' }] }] };
        const tools: NonNullable<ExecutionOptions['tools']> = [
            { name: 'lookup', input_schema: { type: 'object', additionalProperties: true } },
        ];
        const firstOptions = {
            ...runtimeOptions({
                flow: 'stream',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T04:00:00.000Z',
            }),
            tools,
        };
        const stream = await model.requestTextCompletionStream(driver, prompt, firstOptions);
        const chunks = [];
        for await (const chunk of stream) chunks.push(chunk);
        const document = parseConversationDocument(await stream.finalizeConversation?.());
        expect(chunks.flatMap((chunk) => chunk.tool_use ?? [])).toContainEqual(
            expect.objectContaining({ id: 'native-stream-call', tool_name: 'lookup' }),
        );
        expect(chunks.at(-1)?.token_usage).toEqual({
            total: 14,
            prompt: 10,
            prompt_cached: 2,
            prompt_new: 8,
            result: 4,
        });
        const generation = Object.values(document.generations).find(
            (candidate) => candidate.record_source === 'executed',
        );
        expect(generation).toMatchObject({
            finish_reason: 'tool_use',
            usage: { input_tokens: 10, output_tokens: 4, reasoning_tokens: 1, total_tokens: 14 },
        });
        expect(exportLegacyGeminiConversation(document)._arrayConversation.at(-1)?.parts).toEqual([
            { text: 'plan', thought: true, thoughtSignature: 'reasoning-signature' },
            {
                functionCall: { id: 'native-stream-call', name: 'lookup', args: { q: 'x' } },
                thoughtSignature: 'call-signature',
            },
            { text: 'done' },
        ]);

        const recovered = await model.requestTextCompletionStream(driver, prompt, {
            ...runtimeOptions({
                flow: 'stream',
                operation: 'generate',
                attempt: 'retry',
                recorded_at: '2026-09-30T04:05:00.000Z',
                conversation: JSON.parse(JSON.stringify(document)),
            }),
            tools,
        });
        await drain(recovered);
        expect(await recovered.finalizeConversation?.()).toEqual(document);
        expect(generateStream).toHaveBeenCalledTimes(1);
    });

    it('fails closed for truncated or ambiguous responses and accepts an explicit prompt block terminal', async () => {
        const model = new GeminiModelDefinition('gemini-2.5-pro');
        const prompt = { contents: [{ role: 'user' as const, parts: [{ text: 'Continue.' }] }] };
        const truncatedDriver = new TestGeminiDriver(
            async () => {
                throw new Error('blocking transport not expected');
            },
            async () =>
                (async function* () {
                    yield {
                        candidates: [{ content: { role: 'model', parts: [{ text: 'partial' }] } }],
                    } as GenerateContentResponse;
                })(),
        );
        const truncated = await model.requestTextCompletionStream(
            truncatedDriver,
            prompt,
            runtimeOptions({
                flow: 'truncated',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T05:00:00.000Z',
            }),
        );
        await drain(truncated);
        await expect(truncated.finalizeConversation?.()).rejects.toThrow(/without a terminal finish reason/);

        const ambiguousDriver = new TestGeminiDriver(
            async () =>
                ({
                    candidates: [
                        { finishReason: FinishReason.STOP, content: { role: 'model', parts: [{ text: 'one' }] } },
                        { finishReason: FinishReason.STOP, content: { role: 'model', parts: [{ text: 'two' }] } },
                    ],
                }) as GenerateContentResponse,
        );
        await expect(
            model.requestTextCompletion(
                ambiguousDriver,
                prompt,
                runtimeOptions({
                    flow: 'ambiguous',
                    operation: 'generate',
                    attempt: 'first',
                    recorded_at: '2026-09-30T05:10:00.000Z',
                }),
            ),
        ).rejects.toThrow(/requires one candidate/);

        const missingDriver = new TestGeminiDriver(
            async () => ({ candidates: [] }) as unknown as GenerateContentResponse,
        );
        await expect(
            model.requestTextCompletion(
                missingDriver,
                prompt,
                runtimeOptions({
                    flow: 'missing',
                    operation: 'generate',
                    attempt: 'first',
                    recorded_at: '2026-09-30T05:20:00.000Z',
                }),
            ),
        ).rejects.toThrow(/no candidate or prompt block reason/);

        const blockedDriver = new TestGeminiDriver(
            async () =>
                ({
                    candidates: [],
                    promptFeedback: { blockReason: 'BLOCKLIST', blockReasonMessage: 'Prompt blocked.' },
                }) as unknown as GenerateContentResponse,
        );
        const blocked = await model.requestTextCompletion(
            blockedDriver,
            prompt,
            runtimeOptions({
                flow: 'blocked',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T05:30:00.000Z',
            }),
        );
        expect(blocked.result).toEqual([{ type: 'text', value: 'Prompt blocked.' }]);
        const blockedTurn = parseConversationDocument(blocked.conversation).turns.at(-1);
        expect(blockedTurn?.kind).toBe('agent');
        expect(blockedTurn?.blocks).toContainEqual(expect.objectContaining({ type: 'text', text: 'Prompt blocked.' }));
    });

    it('keeps generated conversational audio explicitly unsupported until typed audio media exists', async () => {
        const model = new GeminiModelDefinition('gemini-2.5-pro');
        const driver = new TestGeminiDriver(async () =>
            response({
                id: 'response-audio-output',
                content: {
                    role: 'model',
                    parts: [{ inlineData: { data: 'YXVkaW8=', mimeType: 'audio/mpeg' } }],
                },
            }),
        );
        await expect(
            model.requestTextCompletion(
                driver,
                { contents: [{ role: 'user', parts: [{ text: 'Speak.' }] }] },
                runtimeOptions({
                    flow: 'audio-output',
                    operation: 'generate',
                    attempt: 'first',
                    recorded_at: '2026-09-30T06:00:00.000Z',
                }),
            ),
        ).rejects.toThrow(/Audio output requires a file speech model/);
    });
});
