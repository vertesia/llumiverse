import type { ConversationDocument, ConversationStreamEvent } from '@llumiverse/conversation';
import { type CanonicalExecutionEventStream, type ExecutionOptions, PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { WatsonxDriver } from './index.js';

const MODEL = 'ibm/granite-3-2b-instruct';

function options(attempt: string, conversation?: ConversationDocument): ExecutionOptions {
    return {
        model: MODEL,
        ...(conversation === undefined ? {} : { conversation }),
        model_options: {
            _option_id: 'text-fallback',
            max_tokens: 64,
            temperature: 0.2,
            top_k: 20,
            top_p: 0.8,
            stop_sequence: ['END'],
        },
        conversation_runtime: {
            conversation_id: 'conversation:watsonx:canonical',
            request_id: 'request:watsonx:canonical',
            attempt_id: `attempt:watsonx:canonical:${attempt}`,
            input_operation_id: 'input:watsonx:canonical',
            response_operation_id: 'response:watsonx:canonical',
            recorded_at: '2026-09-30T00:00:00.000Z',
            started_at: '2026-09-30T00:00:00.000Z',
            completed_at: '2026-09-30T00:00:01.000Z',
        },
    };
}

function response(stopReason = 'eos_token') {
    return {
        model_id: MODEL,
        created_at: '2026-09-30T00:00:01.000Z',
        results: [
            {
                generated_text: 'Watsonx answer',
                generated_token_count: 3,
                input_token_count: 4,
                stop_reason: stopReason,
            },
        ],
    };
}

function sse(
    ...responses: ReturnType<typeof response>[]
): ReadableStream<{ type: 'event'; event: 'message'; data: string }> {
    return new ReadableStream({
        start(controller) {
            for (const value of responses) {
                controller.enqueue({ type: 'event', event: 'message', data: JSON.stringify(value) });
            }
            controller.enqueue({ type: 'event', event: 'message', data: '[DONE]' });
            controller.close();
        },
    });
}

async function collect(stream: CanonicalExecutionEventStream): Promise<ConversationStreamEvent[]> {
    const events: ConversationStreamEvent[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

describe('Watsonx canonical lifecycle', () => {
    it('binds the exact native request and JSON-recovers without another transport or publication', async () => {
        const driver = new WatsonxDriver({
            apiKey: 'test-key',
            projectId: 'project-1',
            endpointUrl: 'https://watsonx.example.test',
        });
        const post = vi.fn(async () => response());
        Object.defineProperty(driver.fetchClient, 'post', { value: post });
        let prepared = 0;
        const segments = [{ role: PromptRole.user, content: 'Answer exactly.' }];
        const first = await driver.executeCanonical(segments, {
            ...options('first'),
            on_canonical_request_prepared: async () => {
                prepared += 1;
                expect(post).not.toHaveBeenCalled();
            },
        });

        expect(post).toHaveBeenCalledWith('/ml/v1/text/generation?version=2024-03-14', {
            payload: {
                model_id: MODEL,
                input: 'Answer exactly.\n',
                parameters: {
                    max_new_tokens: 64,
                    temperature: 0.2,
                    top_k: 20,
                    top_p: 0.8,
                    stop_sequences: ['END'],
                },
                project_id: 'project-1',
            },
            signal: undefined,
        });
        expect(first.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'text', text: 'Watsonx answer' }),
        ]);
        const generation = first.conversation.generations[first.accepted_output.generation.id];
        expect(generation).toMatchObject({
            provider: 'watsonx',
            protocol: 'watsonx.text-generation',
            requested_model: MODEL,
            resolved_model: MODEL,
            finish_reason: 'stop',
            usage: { input_tokens: 4, output_tokens: 3, total_tokens: 7 },
            request_receipt: {
                target: {
                    provider: 'watsonx',
                    protocol: 'watsonx.text-generation',
                    model: MODEL,
                    options: {
                        endpoint: 'https://watsonx.example.test',
                        parameters: expect.objectContaining({ max_new_tokens: 64 }),
                    },
                },
            },
        });
        expect(JSON.stringify(generation?.request_receipt?.target.options)).not.toContain('Answer exactly.');

        const persisted = JSON.parse(JSON.stringify(first.conversation)) as ConversationDocument;
        const retry = await driver.executeCanonical(segments, {
            ...options('retry', persisted),
            on_canonical_request_prepared: async () => {
                prepared += 1;
            },
        });
        expect(retry.accepted_output).toEqual(first.accepted_output);
        expect(post).toHaveBeenCalledOnce();
        expect(prepared).toBe(1);

        const recoveredStream = await driver.streamCanonicalEvents(
            segments,
            {
                ...options('stream-retry', persisted),
                on_canonical_request_prepared: async () => {
                    prepared += 1;
                },
            },
            undefined,
            { stream_id: 'stream:watsonx:recovered' },
        );
        await expect(collect(recoveredStream)).resolves.toEqual([
            expect.objectContaining({ type: 'response_accepted', origin: 'accepted_recovery' }),
        ]);
        expect(recoveredStream.completion?.accepted_output).toEqual(first.accepted_output);
        expect(post).toHaveBeenCalledOnce();
        expect(prepared).toBe(1);

        const recoveredStringStream = await driver.streamCanonical(segments, options('string-stream-retry', persisted));
        const recoveredChunks: string[] = [];
        for await (const chunk of recoveredStringStream) recoveredChunks.push(chunk);
        expect(recoveredChunks).toEqual([]);
        expect(recoveredStringStream.completion?.accepted_output).toEqual(first.accepted_output);
        expect(post).toHaveBeenCalledOnce();

        await expect(
            driver.executeCanonical(segments, {
                ...options('original-response-retry', persisted),
                include_original_response: true,
            }),
        ).rejects.toThrow(/cannot reconstruct original_response/);

        await expect(
            driver.executeCanonical(segments, {
                ...options('changed', persisted),
                model_options: { ...options('changed').model_options, temperature: 0.9 },
            }),
        ).rejects.toThrow(/fingerprint|request|incompatible/i);
        expect(post).toHaveBeenCalledOnce();

        const rerouted = new WatsonxDriver({
            apiKey: 'test-key',
            projectId: 'project-1',
            endpointUrl: 'https://other-watsonx.example.test',
        });
        const reroutedPost = vi.fn(async () => response());
        Object.defineProperty(rerouted.fetchClient, 'post', { value: reroutedPost });
        await expect(rerouted.executeCanonical(segments, options('rerouted', persisted))).rejects.toThrow(
            /incompatible Watsonx target options/,
        );
        expect(reroutedPost).not.toHaveBeenCalled();
    });

    it('projects native SSE increments through canonical string and typed streams', async () => {
        const driver = new WatsonxDriver({
            apiKey: 'test-key',
            projectId: 'project-1',
            endpointUrl: 'https://watsonx.example.test',
        });
        const nativeEvents = [
            {
                model_id: MODEL,
                created_at: '2025-03-01T02:50:13.751Z',
                results: [
                    {
                        generated_text: '',
                        generated_token_count: 0,
                        input_token_count: 28,
                        stop_reason: 'not_finished',
                    },
                ],
            },
            {
                model_id: MODEL,
                created_at: '2025-03-01T02:50:13.798Z',
                results: [
                    {
                        generated_text: 'Watsonx ',
                        generated_token_count: 1,
                        input_token_count: 0,
                        stop_reason: 'not_finished',
                    },
                ],
            },
            {
                model_id: MODEL,
                created_at: '2025-03-01T02:50:13.821Z',
                results: [
                    {
                        generated_text: 'answer',
                        generated_token_count: 2,
                        input_token_count: 0,
                        stop_reason: 'max_tokens',
                    },
                ],
            },
        ];
        const post = vi.fn(async () => sse(...nativeEvents));
        Object.defineProperty(driver.fetchClient, 'post', { value: post });

        const typedOptions = { ...options('cutoff'), include_original_response: true };
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Keep writing.' }],
            typedOptions,
            undefined,
            { stream_id: 'stream:watsonx:cutoff' },
        );
        const events = await collect(stream);
        expect(events.flatMap((event) => (event.type === 'draft_text_delta' ? [event.text] : []))).toEqual([
            'Watsonx ',
            'answer',
        ]);
        expect(events.at(-1)).toMatchObject({ type: 'response_accepted' });
        expect(stream.completion?.accepted_output.turn.status).toBe('interrupted');
        expect(stream.completion?.accepted_output.generation).toMatchObject({
            status: 'cancelled',
            finish_reason: 'length',
        });
        const typedGenerationId = stream.completion?.accepted_output.generation.id;
        if (typedGenerationId === undefined) throw new Error('Missing Watsonx typed generation');
        expect(stream.completion?.conversation.generations[typedGenerationId]?.usage).toMatchObject({
            input_tokens: 28,
            output_tokens: 2,
            total_tokens: 30,
        });
        expect(stream.completion?.original_response).toEqual({
            model_id: MODEL,
            created_at: '2025-03-01T02:50:13.821Z',
            results: [
                {
                    generated_text: 'Watsonx answer',
                    generated_token_count: 2,
                    input_token_count: 28,
                    stop_reason: 'max_tokens',
                },
            ],
        });
        expect(post).toHaveBeenNthCalledWith(
            1,
            '/ml/v1/text/generation_stream?version=2024-03-14',
            expect.objectContaining({
                payload: expect.objectContaining({ model_id: MODEL, input: 'Keep writing.\n' }),
                reader: 'sse',
                signal: expect.any(AbortSignal),
            }),
        );
        expect(post).toHaveBeenCalledOnce();
        await stream.closed;

        const persisted = JSON.parse(JSON.stringify(stream.completion?.conversation)) as ConversationDocument;
        const recovered = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Keep writing.' }],
            options('cutoff-retry', persisted),
            undefined,
            { stream_id: 'stream:watsonx:cutoff-retry' },
        );
        await expect(collect(recovered)).resolves.toEqual([
            expect.objectContaining({ type: 'response_accepted', origin: 'accepted_recovery' }),
        ]);
        const recoveredGenerationId = recovered.completion?.accepted_output.generation.id;
        if (recoveredGenerationId === undefined) throw new Error('Missing recovered Watsonx generation');
        expect(recovered.completion?.conversation.generations[recoveredGenerationId]?.usage).toMatchObject({
            input_tokens: 28,
            output_tokens: 2,
            total_tokens: 30,
        });
        expect(post).toHaveBeenCalledOnce();
        await recovered.closed;

        const stringOptions = options('cutoff-string');
        if (stringOptions.conversation_runtime === undefined) throw new Error('Missing Watsonx runtime');
        stringOptions.conversation_runtime = {
            ...stringOptions.conversation_runtime,
            conversation_id: 'conversation:watsonx:string-stream',
            request_id: 'request:watsonx:string-stream',
            input_operation_id: 'input:watsonx:string-stream',
            response_operation_id: 'response:watsonx:string-stream',
        };
        stringOptions.include_original_response = true;
        const stringStream = await driver.streamCanonical(
            [{ role: PromptRole.user, content: 'Keep writing.' }],
            stringOptions,
        );
        const chunks: string[] = [];
        for await (const chunk of stringStream) chunks.push(chunk);
        expect(chunks).toEqual(['Watsonx ', 'answer']);
        expect(stringStream.completion?.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'text', text: 'Watsonx answer' }),
        ]);
        const stringGenerationId = stringStream.completion?.accepted_output.generation.id;
        if (stringGenerationId === undefined) throw new Error('Missing Watsonx string generation');
        expect(stringStream.completion?.conversation.generations[stringGenerationId]?.usage).toMatchObject({
            input_tokens: 28,
            output_tokens: 2,
            total_tokens: 30,
        });
        expect(stringStream.completion?.original_response).toEqual({
            model_id: MODEL,
            created_at: '2025-03-01T02:50:13.821Z',
            results: [
                {
                    generated_text: 'Watsonx answer',
                    generated_token_count: 2,
                    input_token_count: 28,
                    stop_reason: 'max_tokens',
                },
            ],
        });
        expect(post).toHaveBeenCalledTimes(2);
    });

    it('classifies every documented terminal reason without completing interrupted or failed output', async () => {
        const driver = new WatsonxDriver({
            apiKey: 'test-key',
            projectId: 'project-1',
            endpointUrl: 'https://watsonx.example.test',
        });
        const terminalCases = [
            { reason: 'stop_sequence', generation: 'completed', turn: 'completed', finish: 'stop_sequence' },
            { reason: 'max_tokens', generation: 'cancelled', turn: 'interrupted', finish: 'length' },
            { reason: 'token_limit', generation: 'cancelled', turn: 'interrupted', finish: 'token_limit' },
            { reason: 'time_limit', generation: 'cancelled', turn: 'interrupted', finish: 'time_limit' },
            { reason: 'canceled', generation: 'cancelled', turn: 'interrupted', finish: 'canceled' },
            { reason: 'cancelled', generation: 'cancelled', turn: 'interrupted', finish: 'cancelled' },
            { reason: 'error', generation: 'failed', turn: 'failed', finish: 'error' },
        ] as const;
        const post = vi.fn(async () => response());
        Object.defineProperty(driver.fetchClient, 'post', { value: post });

        for (const terminalCase of terminalCases) {
            post.mockResolvedValueOnce(response(terminalCase.reason));
            const result = await driver.executeCanonical(
                [{ role: PromptRole.user, content: `Classify ${terminalCase.reason}.` }],
                options(`reason-${terminalCase.reason}`),
            );
            expect(result.accepted_output.turn.status).toBe(terminalCase.turn);
            expect(result.conversation.generations[result.accepted_output.generation.id]).toMatchObject({
                status: terminalCase.generation,
                finish_reason: terminalCase.finish,
            });
        }

        post.mockResolvedValueOnce(response('unknown_reason'));
        await expect(
            driver.executeCanonical(
                [{ role: PromptRole.user, content: 'Reject an unknown terminal.' }],
                options('reason-unknown'),
            ),
        ).rejects.toThrow(/unsupported stop reason unknown_reason/);
    });

    it('fails a named native SSE error event without exposing its provider payload', async () => {
        const driver = new WatsonxDriver({
            apiKey: 'test-key',
            projectId: 'project-1',
            endpointUrl: 'https://watsonx.example.test',
        });
        const post = vi.fn(
            async () =>
                new ReadableStream({
                    start(controller) {
                        controller.enqueue({
                            type: 'event',
                            event: 'error',
                            data: JSON.stringify({ errors: [{ message: 'provider secret payload' }] }),
                        });
                        controller.close();
                    },
                }),
        );
        Object.defineProperty(driver.fetchClient, 'post', { value: post });

        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Fail safely.' }],
            options('native-error'),
            undefined,
            { stream_id: 'stream:watsonx:native-error' },
        );
        const events = await collect(stream);
        const terminal = events.at(-1);
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'failed' });
        expect(JSON.stringify(terminal)).not.toContain('provider secret payload');
        expect(events).not.toContainEqual(expect.objectContaining({ type: 'response_accepted' }));
        expect(stream.completion).toBeUndefined();
        await stream.closed;
    });

    it('rejects unsupported tools and a failed durability barrier before transport', async () => {
        const driver = new WatsonxDriver({
            apiKey: 'test-key',
            projectId: 'project-1',
            endpointUrl: 'https://watsonx.example.test',
        });
        const post = vi.fn(async () => response());
        Object.defineProperty(driver.fetchClient, 'post', { value: post });

        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Use a tool.' }], {
                ...options('tools'),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            }),
        ).rejects.toThrow('does not support tools');
        expect(post).not.toHaveBeenCalled();

        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Do not send.' }], {
                ...options('barrier'),
                on_canonical_request_prepared: async () => {
                    throw new Error('durability barrier failed');
                },
            }),
        ).rejects.toThrow('durability barrier failed');
        expect(post).not.toHaveBeenCalled();

        await expect(
            driver.streamCanonicalEvents(
                [{ role: PromptRole.user, content: 'Do not stream.' }],
                {
                    ...options('stream-barrier'),
                    on_canonical_request_prepared: async () => {
                        throw new Error('stream durability barrier failed');
                    },
                },
                undefined,
                { stream_id: 'stream:watsonx:barrier' },
            ),
        ).rejects.toThrow('stream durability barrier failed');
        expect(post).not.toHaveBeenCalled();
    });

    it('normalizes required structured output and retains invalid evidence as failed', async () => {
        const driver = new WatsonxDriver({
            apiKey: 'test-key',
            projectId: 'project-1',
            endpointUrl: 'https://watsonx.example.test',
        });
        const post = vi
            .fn()
            .mockResolvedValueOnce({
                ...response(),
                results: [{ ...response().results[0], generated_text: '{"answer":"yes"}' }],
            })
            .mockResolvedValueOnce({
                ...response(),
                results: [{ ...response().results[0], generated_text: 'not json' }],
            })
            .mockResolvedValueOnce(
                sse(
                    {
                        ...response('not_finished'),
                        results: [{ ...response('not_finished').results[0], generated_text: '{"answer":' }],
                    },
                    {
                        ...response(),
                        results: [{ ...response().results[0], generated_text: '"streamed"}' }],
                    },
                ),
            );
        Object.defineProperty(driver.fetchClient, 'post', { value: post });
        const resultSchema = {
            type: 'object' as const,
            properties: { answer: { type: 'string' as const } },
            required: ['answer'],
            additionalProperties: false,
        };

        const valid = await driver.executeCanonical([{ role: PromptRole.user, content: 'Return JSON.' }], {
            ...options('structured-valid'),
            result_schema: resultSchema,
        });
        expect(valid.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'json', value: { answer: 'yes' } }),
        ]);
        expect(post.mock.calls[0]?.[1]).toMatchObject({
            payload: { input: expect.stringContaining('IMPORTANT:') },
        });

        const invalidOptions = options('structured-invalid');
        if (invalidOptions.conversation_runtime === undefined) throw new Error('Missing Watsonx runtime');
        const invalid = await driver.executeCanonical([{ role: PromptRole.user, content: 'Return JSON.' }], {
            ...invalidOptions,
            conversation_runtime: {
                ...invalidOptions.conversation_runtime,
                conversation_id: 'conversation:watsonx:canonical-invalid',
                request_id: 'request:watsonx:canonical-invalid',
                input_operation_id: 'input:watsonx:canonical-invalid',
                response_operation_id: 'response:watsonx:canonical-invalid',
            },
            result_schema: resultSchema,
        });
        expect(invalid.accepted_output.turn).toMatchObject({
            status: 'failed',
            blocks: [expect.objectContaining({ type: 'text', text: 'not json' })],
        });
        expect(invalid.conversation.generations[invalid.accepted_output.generation.id]).toMatchObject({
            status: 'failed',
            metadata: { structured_output: { status: 'invalid', code: 'validation_error' } },
        });

        const streamOptions = options('structured-stream');
        if (streamOptions.conversation_runtime === undefined) throw new Error('Missing Watsonx runtime');
        streamOptions.conversation_runtime = {
            ...streamOptions.conversation_runtime,
            conversation_id: 'conversation:watsonx:structured-stream',
            request_id: 'request:watsonx:structured-stream',
            input_operation_id: 'input:watsonx:structured-stream',
            response_operation_id: 'response:watsonx:structured-stream',
        };
        streamOptions.result_schema = resultSchema;
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Return streamed JSON.' }],
            streamOptions,
            undefined,
            { stream_id: 'stream:watsonx:structured' },
        );
        const streamEvents = await collect(stream);
        expect(streamEvents.at(-1)).toMatchObject({
            type: 'response_accepted',
            reconciliations: [{ disposition: 'structured_output' }],
        });
        expect(stream.completion?.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'json', value: { answer: 'streamed' } }),
        ]);
    });

    it('rejects materialized input before transport instead of sending an empty prompt', async () => {
        const driver = new WatsonxDriver({
            apiKey: 'test-key',
            projectId: 'project-1',
            endpointUrl: 'https://watsonx.example.test',
        });
        const post = vi.fn(async () => response());
        Object.defineProperty(driver.fetchClient, 'post', { value: post });
        const materialized = options('materialized');
        if (materialized.conversation_runtime === undefined) throw new Error('Missing Watsonx runtime');
        materialized.conversation_runtime.materialized_input = {
            operation_id: 'input:watsonx:materialized',
            result_revision: 0,
        };

        await expect(driver.executeCanonical([], materialized)).rejects.toThrow(/does not support materialized/);
        expect(post).not.toHaveBeenCalled();
    });

    it('settles typed-stream cancellation and aborts the pending request', async () => {
        const driver = new WatsonxDriver({
            apiKey: 'test-key',
            projectId: 'project-1',
            endpointUrl: 'https://watsonx.example.test',
        });
        let requestSignal: AbortSignal | undefined;
        const post = vi.fn(
            async (_path: string, request: { signal?: AbortSignal }): Promise<ReturnType<typeof response>> => {
                requestSignal = request.signal;
                return await new Promise((_resolve, reject) => {
                    request.signal?.addEventListener(
                        'abort',
                        () => reject(new DOMException('provider request aborted', 'AbortError')),
                        { once: true },
                    );
                });
            },
        );
        Object.defineProperty(driver.fetchClient, 'post', { value: post });

        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Wait.' }],
            options('cancel'),
            undefined,
            { stream_id: 'stream:watsonx:cancel' },
        );
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_started' } });
        const pending = iterator.next();
        await vi.waitFor(() => expect(post).toHaveBeenCalledOnce());
        const terminal = await stream.cancel();
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        await expect(pending).resolves.toMatchObject({ value: terminal });
        expect(requestSignal?.aborted).toBe(true);
        await stream.closed;
        expect(stream.completion).toBeUndefined();
    });

    it('fails a native stream that mutates after its terminal result without accepting the prefix', async () => {
        const driver = new WatsonxDriver({
            apiKey: 'test-key',
            projectId: 'project-1',
            endpointUrl: 'https://watsonx.example.test',
        });
        const post = vi.fn(async () =>
            sse(
                {
                    ...response(),
                    results: [{ ...response().results[0], generated_text: 'terminal' }],
                },
                {
                    ...response(),
                    results: [{ ...response().results[0], generated_text: 'late' }],
                },
            ),
        );
        Object.defineProperty(driver.fetchClient, 'post', { value: post });

        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Reject late bytes.' }],
            options('late-bytes'),
            undefined,
            { stream_id: 'stream:watsonx:late-bytes' },
        );
        const events = await collect(stream);
        expect(events.at(-1)).toMatchObject({ type: 'stream_terminated', outcome: 'failed' });
        expect(events).not.toContainEqual(expect.objectContaining({ type: 'response_accepted' }));
        expect(stream.completion).toBeUndefined();
        await stream.closed;
    });
});
