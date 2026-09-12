import { parseConversationDocument } from '@llumiverse/conversation';
import { type ExecutionOptions, PromptRole, Providers } from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIResponsesDriverBase } from './index.js';
import {
    exportLegacyOpenAIResponsesConversation,
    OPENAI_RESPONSES_PROTOCOL,
} from './openai-responses-conversation-adapter.js';

class TestOpenAIResponsesDriver extends OpenAIResponsesDriverBase {
    provider: Providers.openai = Providers.openai;
    service: OpenAI;

    constructor(create: (request: unknown, options?: unknown) => Promise<unknown>) {
        super({});
        this.service = { responses: { create } } as unknown as OpenAI;
    }
}

type ResponseStatus = OpenAI.Responses.Response['status'];

const toolDefinition: NonNullable<ExecutionOptions['tools']>[number] = {
    name: 'lookup_weather',
    description: 'Look up weather',
    input_schema: {
        type: 'object',
        properties: { city: { type: 'string' } },
        required: ['city'],
        additionalProperties: false,
    },
};

function messageItem(id: string, text: string, status: 'completed' | 'incomplete' = 'completed') {
    return {
        type: 'message' as const,
        id,
        role: 'assistant' as const,
        status,
        content: [
            {
                type: 'output_text' as const,
                text,
                annotations: [],
                logprobs: [],
            },
        ],
    };
}

function response(input: {
    id: string;
    output: OpenAI.Responses.ResponseOutputItem[];
    model?: string;
    status?: ResponseStatus;
    inputTokens?: number;
    outputTokens?: number;
    cachedTokens?: number;
    cacheWriteTokens?: number;
    reasoningTokens?: number;
}): OpenAI.Responses.Response {
    const status = input.status ?? 'completed';
    const inputTokens = input.inputTokens ?? 11;
    const outputTokens = input.outputTokens ?? 7;
    return {
        id: input.id,
        object: 'response',
        created_at: 1,
        model: input.model ?? 'gpt-5',
        service_tier: 'default',
        status,
        output: input.output,
        output_text: '',
        parallel_tool_calls: true,
        tool_choice: 'auto',
        tools: [],
        error: status === 'failed' ? { code: 'server_error', message: 'provider failed' } : null,
        incomplete_details: status === 'incomplete' ? { reason: 'max_output_tokens' } : null,
        instructions: null,
        metadata: null,
        temperature: null,
        top_p: null,
        usage: {
            input_tokens: inputTokens,
            output_tokens: outputTokens,
            total_tokens: inputTokens + outputTokens,
            input_tokens_details: {
                cached_tokens: input.cachedTokens ?? 0,
                ...(input.cacheWriteTokens === undefined ? {} : { cache_write_tokens: input.cacheWriteTokens }),
            },
            output_tokens_details: { reasoning_tokens: input.reasoningTokens ?? 0 },
        },
    } as unknown as OpenAI.Responses.Response;
}

function runtimeOptions(input: {
    flow: string;
    operation: string;
    attempt: string;
    recordedAt: string;
    conversation?: unknown;
    model?: string;
}): ExecutionOptions {
    return {
        model: input.model ?? 'gpt-5',
        ...(input.conversation === undefined ? {} : { conversation: input.conversation }),
        conversation_runtime: {
            conversation_id: `conversation:${input.flow}`,
            request_id: `request:${input.flow}:${input.operation}`,
            attempt_id: `attempt:${input.flow}:${input.attempt}`,
            input_operation_id: `input:${input.flow}:${input.operation}`,
            response_operation_id: `response:${input.flow}:${input.operation}`,
            recorded_at: input.recordedAt,
            started_at: input.recordedAt,
        },
    };
}

function latestGeneratedText(value: unknown): string | undefined {
    const document = parseConversationDocument(value);
    for (let index = document.turns.length - 1; index >= 0; index -= 1) {
        const turn = document.turns[index];
        if (turn.kind !== 'agent' || turn.provenance.type !== 'generated') continue;
        return turn.blocks.find((block) => block.type === 'text')?.text;
    }
    return undefined;
}

async function consume(stream: AsyncIterable<string>): Promise<string> {
    let text = '';
    for await (const chunk of stream) text += chunk;
    return text;
}

describe('OpenAI Responses canonical lifecycle', () => {
    it('validates structured sync output, retains exact native data, and recovers a persisted retry', async () => {
        const rawText = '{ "answer" : "Tokyo", "note" : null }';
        const output = [
            {
                ...messageItem('message:structured', rawText),
                content: [
                    {
                        type: 'output_text' as const,
                        text: rawText,
                        annotations: [
                            {
                                type: 'url_citation' as const,
                                start_index: 13,
                                end_index: 20,
                                title: 'Tokyo',
                                url: 'https://example.com/tokyo',
                            },
                        ],
                        logprobs: [],
                    },
                ],
            },
        ] satisfies OpenAI.Responses.ResponseOutputItem[];
        const nativeResponse = response({
            id: 'response:structured',
            output,
            inputTokens: 100,
            outputTokens: 20,
            cachedTokens: 25,
            cacheWriteTokens: 5,
            reasoningTokens: 7,
        });
        const create = vi.fn(async (_request: unknown, _options?: unknown) => nativeResponse);
        const driver = new TestOpenAIResponsesDriver(create);
        const segments = [{ role: PromptRole.user, content: 'Return the city as JSON.' }];
        const resultSchema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' }, note: { type: 'null' } },
            required: ['answer', 'note'],
            additionalProperties: false,
        };
        const firstOptions = {
            ...runtimeOptions({
                flow: 'structured',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T00:00:00.000Z',
            }),
            result_schema: resultSchema,
        };

        const first = await driver.execute(segments, firstOptions);

        expect(first.result).toEqual([{ type: 'json', value: { answer: 'Tokyo', note: null } }]);
        expect(latestGeneratedText(first.conversation)).toBe(rawText);
        const document = parseConversationDocument(JSON.parse(JSON.stringify(first.conversation)));
        const generation = Object.values(document.generations).find(
            (candidate) => candidate.record_source === 'executed',
        );
        expect(generation).toMatchObject({
            provider: 'openai',
            protocol: OPENAI_RESPONSES_PROTOCOL,
            requested_model: 'gpt-5',
            resolved_model: 'gpt-5',
            provider_response_id: 'response:structured',
            usage: {
                input_tokens: 100,
                output_tokens: 20,
                reasoning_tokens: 7,
                cache_read_tokens: 25,
                cache_write_tokens: 5,
                input_new_tokens: 70,
                total_tokens: 120,
            },
        });
        expect(exportLegacyOpenAIResponsesConversation(document).at(-1)).toEqual(output[0]);

        const retried = await driver.execute(segments, {
            ...runtimeOptions({
                flow: 'structured',
                operation: 'generate',
                attempt: 'retry',
                recordedAt: '2026-09-12T00:05:00.000Z',
                conversation: document,
            }),
            result_schema: resultSchema,
        });

        expect(retried.result).toEqual(first.result);
        expect(retried.token_usage).toEqual(first.token_usage);
        expect(retried.service_tier).toBe(first.service_tier);
        expect(retried.conversation).toEqual(document);
        expect(create).toHaveBeenCalledTimes(1);
    });

    it('preserves tool result status internally, projects rich continuation, and recovers its accepted response', async () => {
        const callItem = {
            type: 'function_call' as const,
            id: 'item:lookup',
            call_id: 'call:lookup',
            name: 'lookup_weather',
            arguments: '{ "city" : "Tokyo", "units" : null }',
            status: 'completed' as const,
        };
        const responses = [
            response({ id: 'response:tool-call', output: [callItem] }),
            response({ id: 'response:tool-result', output: [messageItem('message:tool-result', 'Try again later.')] }),
        ];
        const create = vi.fn(async (_request: unknown, _options?: unknown) => {
            const next = responses.shift();
            if (!next) throw new Error('Unexpected provider call');
            return next;
        });
        const driver = new TestOpenAIResponsesDriver(create);
        const tools = [toolDefinition];
        const first = await driver.execute([{ role: PromptRole.user, content: 'Weather in Tokyo?' }], {
            ...runtimeOptions({
                flow: 'tools',
                operation: 'ask',
                attempt: 'ask',
                recordedAt: '2026-09-12T01:00:00.000Z',
            }),
            tools,
        });
        expect(first.tool_use).toEqual([
            { id: 'call:lookup', tool_name: 'lookup_weather', tool_input: { city: 'Tokyo', units: null } },
        ]);

        const resultSegments = [
            {
                role: PromptRole.tool,
                content: '{"error":"weather service unavailable"}',
                tool_use_id: 'call:lookup',
                tool_result_status: 'error' as const,
            },
        ];
        const second = await driver.execute(resultSegments, {
            ...runtimeOptions({
                flow: 'tools',
                operation: 'answer',
                attempt: 'answer',
                recordedAt: '2026-09-12T01:01:00.000Z',
                conversation: first.conversation,
            }),
            tools,
        });
        const secondRequest = create.mock.calls[1]?.[0] as { input?: Array<Record<string, unknown>> };
        const projectedResult = secondRequest.input?.find((item) => item.type === 'function_call_output');
        expect(projectedResult).toEqual({
            type: 'function_call_output',
            call_id: 'call:lookup',
            output: '{"error":"weather service unavailable"}',
        });
        expect(JSON.stringify(secondRequest)).not.toContain('_llumiverse_tool_result_status');

        const persisted = parseConversationDocument(JSON.parse(JSON.stringify(second.conversation)));
        const toolTurn = persisted.turns.find((turn) => turn.kind === 'tool');
        expect(toolTurn?.blocks[0]).toMatchObject({ type: 'tool_result', call_id: 'call:lookup', status: 'error' });
        expect(
            Object.values(persisted.execution_receipts).find(
                (receipt) => receipt.executor === 'application' && receipt.call_id === 'call:lookup',
            ),
        ).toMatchObject({ status: 'error' });

        const retried = await driver.execute(resultSegments, {
            ...runtimeOptions({
                flow: 'tools',
                operation: 'answer',
                attempt: 'answer-retry',
                recordedAt: '2026-09-12T01:05:00.000Z',
                conversation: persisted,
            }),
            tools,
        });
        expect(retried.result).toEqual(second.result);
        expect(retried.conversation).toEqual(persisted);
        expect(create).toHaveBeenCalledTimes(2);
    });

    it('streams reasoning and text through core, finalizes authoritative output, and recovers without transport', async () => {
        const reasoningItem = {
            type: 'reasoning' as const,
            id: 'reasoning:stream',
            summary: [{ type: 'summary_text' as const, text: 'Check the evidence.' }],
            encrypted_content: 'opaque-stream-reasoning',
            status: 'completed' as const,
        };
        const final = response({
            id: 'response:stream',
            output: [reasoningItem, messageItem('message:stream', 'It is clear.')],
            reasoningTokens: 4,
        });
        const create = vi.fn(async (_request: unknown, _options?: unknown) =>
            (async function* () {
                yield {
                    type: 'response.reasoning_summary_text.delta' as const,
                    item_id: 'reasoning:stream',
                    output_index: 0,
                    summary_index: 0,
                    sequence_number: 1,
                    delta: 'Check the evidence.',
                };
                yield {
                    type: 'response.output_text.delta' as const,
                    item_id: 'message:stream',
                    output_index: 1,
                    content_index: 0,
                    sequence_number: 2,
                    delta: 'It is clear.',
                    logprobs: [],
                };
                yield { type: 'response.completed' as const, sequence_number: 3, response: final };
            })(),
        );
        const driver = new TestOpenAIResponsesDriver(create);
        const segments = [{ role: PromptRole.user, content: 'Explain.' }];
        const firstOptions = {
            ...runtimeOptions({
                flow: 'stream',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T02:00:00.000Z',
            }),
            model_options: { _option_id: 'openai-thinking' as const },
        };

        const first = await driver.stream(segments, firstOptions);
        expect(await consume(first)).toBe('Check the evidence.\nIt is clear.');
        expect(first.completion?.result).toEqual([
            { type: 'thoughts', value: 'Check the evidence.' },
            { type: 'text', value: 'It is clear.' },
        ]);
        const persisted = parseConversationDocument(JSON.parse(JSON.stringify(first.completion?.conversation)));
        expect(JSON.stringify(persisted)).toContain('opaque-stream-reasoning');

        const retried = await driver.stream(segments, {
            ...runtimeOptions({
                flow: 'stream',
                operation: 'generate',
                attempt: 'retry',
                recordedAt: '2026-09-12T02:05:00.000Z',
                conversation: persisted,
            }),
            model_options: { _option_id: 'openai-thinking' as const },
        });
        expect(await consume(retried)).toBe('Check the evidence.\nIt is clear.');
        expect(retried.completion?.conversation).toEqual(persisted);
        expect(create).toHaveBeenCalledTimes(1);
    });

    it.each([
        ['truncated', undefined],
        ['failed', 7],
    ] as const)('does not persist a canonical response when a stream is %s', async (failure, billedOutputTokens) => {
        const failed = response({
            id: 'response:failed-stream',
            output: [messageItem('message:failed-stream', 'partial')],
            status: 'failed',
            outputTokens: billedOutputTokens ?? 0,
        });
        const create = vi.fn(async (_request: unknown, _options?: unknown) =>
            (async function* () {
                yield {
                    type: 'response.output_text.delta' as const,
                    item_id: 'message:failed-stream',
                    output_index: 0,
                    content_index: 0,
                    sequence_number: 1,
                    delta: 'partial',
                    logprobs: [],
                };
                if (failure === 'failed') {
                    yield { type: 'response.failed' as const, sequence_number: 2, response: failed };
                }
            })(),
        );
        const driver = new TestOpenAIResponsesDriver(create);
        const stream = await driver.stream([{ role: PromptRole.user, content: 'Continue.' }], {
            ...runtimeOptions({
                flow: `stream-${failure}`,
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T03:00:00.000Z',
            }),
        });

        await expect(consume(stream)).rejects.toThrow(
            failure === 'failed' ? 'provider failed' : 'ended without a final response',
        );
        expect(stream.completion?.conversation).toBeUndefined();
        if (billedOutputTokens !== undefined) {
            expect(stream.completion?.token_usage?.result).toBe(billedOutputTokens);
        }
    });

    it('retains an incomplete native turn and replays it for a later Responses continuation', async () => {
        const partialItem = messageItem('message:partial', 'First,', 'incomplete');
        const partial = response({ id: 'response:partial', output: [partialItem], status: 'incomplete' });
        const completed = response({
            id: 'response:continued',
            output: [messageItem('message:continued', 'the rest follows.')],
        });
        const create = vi.fn(async (request: unknown, _options?: unknown) => {
            if ((request as { stream?: boolean }).stream) {
                return (async function* () {
                    yield {
                        type: 'response.output_text.delta' as const,
                        item_id: 'message:partial',
                        output_index: 0,
                        content_index: 0,
                        sequence_number: 1,
                        delta: 'First,',
                        logprobs: [],
                    };
                    yield { type: 'response.incomplete' as const, sequence_number: 2, response: partial };
                })();
            }
            return completed;
        });
        const driver = new TestOpenAIResponsesDriver(create);
        const first = await driver.stream([{ role: PromptRole.user, content: 'Start an explanation.' }], {
            ...runtimeOptions({
                flow: 'partial',
                operation: 'start',
                attempt: 'start',
                recordedAt: '2026-09-12T04:00:00.000Z',
            }),
        });
        await consume(first);
        expect(first.completion?.finish_reason).toBe('length');
        const partialDocument = parseConversationDocument(first.completion?.conversation);
        expect(partialDocument.turns.at(-1)?.status).toBe('interrupted');

        const continuation = await driver.execute([{ role: PromptRole.user, content: 'Continue.' }], {
            ...runtimeOptions({
                flow: 'partial',
                operation: 'continue',
                attempt: 'continue',
                recordedAt: '2026-09-12T04:01:00.000Z',
                conversation: partialDocument,
            }),
        });
        const continuationRequest = create.mock.calls[1]?.[0] as { input?: unknown[] };
        expect(continuationRequest.input).toEqual(expect.arrayContaining([partialItem]));
        expect(continuation.result).toEqual([{ type: 'text', value: 'the rest follows.' }]);
    });

    it('applies request-only history retention and prompt transforms without rewriting canonical source', async () => {
        const nativeHistory = {
            _arrayConversation: [{ role: 'user' as const, content: '<heartbeat>old status</heartbeat>' }],
            _llumiverse_meta: { turnNumber: 20 },
        };
        const currentPrompt = [
            {
                type: 'message' as const,
                role: 'system' as const,
                content: [
                    { type: 'input_text' as const, text: 'Inspect this image.' },
                    {
                        type: 'input_image' as const,
                        image_url: 'data:image/png;base64,aW1hZ2U=',
                        detail: 'auto' as const,
                    },
                ],
            },
        ];
        const create = vi.fn(async (_request: unknown, _options?: unknown) =>
            response({ id: 'response:projection', output: [messageItem('message:projection', 'Done.')], model: 'o1' }),
        );
        const driver = new TestOpenAIResponsesDriver(create);

        const completion = await driver.requestTextCompletion(currentPrompt, {
            ...runtimeOptions({
                flow: 'projection',
                operation: 'generate',
                attempt: 'first',
                recordedAt: '2026-09-12T05:00:00.000Z',
                conversation: nativeHistory,
                model: 'o1',
            }),
            stripHeartbeatsAfterTurns: 1,
            model_options: { _option_id: 'openai-thinking', image_detail: 'high' },
        });
        const request = create.mock.calls[0]?.[0] as { input?: unknown[] };
        expect(request.input).toEqual([
            { role: 'user', content: '[Heartbeat removed from conversation history]' },
            {
                role: 'developer',
                content: [
                    { type: 'input_text', text: 'Inspect this image.' },
                    { type: 'input_image', image_url: 'data:image/png;base64,aW1hZ2U=', detail: 'high' },
                ],
            },
        ]);

        const projected = exportLegacyOpenAIResponsesConversation(parseConversationDocument(completion.conversation));
        expect(projected.slice(0, 2)).toEqual([
            { role: 'user', content: '<heartbeat>old status</heartbeat>' },
            {
                role: 'system',
                content: [
                    { type: 'input_text', text: 'Inspect this image.' },
                    { type: 'input_image', image_url: 'data:image/png;base64,aW1hZ2U=', detail: 'auto' },
                ],
            },
        ]);
    });
});
