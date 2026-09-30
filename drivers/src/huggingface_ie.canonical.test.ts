import type { InferenceClient, TextGenerationOutput, TextGenerationStreamOutput } from '@huggingface/inference';
import type { ConversationDocument, ConversationStreamEvent } from '@llumiverse/conversation';
import {
    type CanonicalExecutionEventStream,
    type CompletionChunkObject,
    type ExecutionOptions,
    PromptRole,
} from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { HuggingFaceIEDriver } from './huggingface_ie.js';

const MODEL = 'endpoint-a';
const MANAGEMENT_ENDPOINT = 'https://management.huggingface.test';
const INFERENCE_ENDPOINT = 'https://endpoint-a.us-east-1.aws.endpoints.huggingface.cloud';

function options(attempt: string, conversation?: ConversationDocument): ExecutionOptions {
    return {
        model: MODEL,
        ...(conversation === undefined ? {} : { conversation }),
        model_options: { _option_id: 'text-fallback', max_tokens: 64, temperature: 0.2 },
        conversation_runtime: {
            conversation_id: 'conversation:huggingface:canonical',
            request_id: 'request:huggingface:canonical',
            attempt_id: `attempt:huggingface:canonical:${attempt}`,
            input_operation_id: 'input:huggingface:canonical',
            response_operation_id: 'response:huggingface:canonical',
            recorded_at: '2026-09-30T00:00:00.000Z',
            started_at: '2026-09-30T00:00:00.000Z',
            completed_at: '2026-09-30T00:00:01.000Z',
        },
    };
}

function syncResponse(text = 'Hugging Face answer', finishReason = 'eos_token'): TextGenerationOutput {
    return {
        generated_text: text,
        details: {
            finish_reason: finishReason as 'eos_token',
            generated_tokens: 3,
            prefill: [],
            tokens: [],
        },
    };
}

function token(input: {
    text: string;
    special?: boolean;
    generated_text?: string | null;
    finish_reason?: 'length' | 'eos_token' | 'stop_sequence';
}): TextGenerationStreamOutput {
    return {
        index: 0,
        token: { id: 10, text: input.text, logprob: -0.1, special: input.special ?? false },
        generated_text: input.generated_text ?? null,
        details:
            input.finish_reason === undefined
                ? null
                : {
                      finish_reason: input.finish_reason,
                      generated_tokens: 2,
                      input_length: 2,
                  },
    } as unknown as TextGenerationStreamOutput;
}

async function* streamEvents(
    finishReason: 'length' | 'eos_token' | 'stop_sequence' = 'eos_token',
): AsyncGenerator<TextGenerationStreamOutput> {
    yield token({ text: 'Hugging ' });
    yield token({ text: 'Face' });
    yield token({ text: '</s>', special: true, generated_text: 'Hugging Face', finish_reason: finishReason });
}

function setup(executorOverrides: Partial<InferenceClient> = {}) {
    const driver = new HuggingFaceIEDriver({ apiKey: 'test-key', endpoint_url: MANAGEMENT_ENDPOINT });
    const executor = {
        textGeneration: vi.fn(async () => syncResponse()),
        textGenerationStream: vi.fn(() => streamEvents()),
        ...executorOverrides,
    } as unknown as InferenceClient;
    const target = vi.spyOn(driver, 'getExecutorTarget').mockResolvedValue({ executor, url: INFERENCE_ENDPOINT });
    return { driver, executor, target };
}

async function collect(stream: CanonicalExecutionEventStream): Promise<ConversationStreamEvent[]> {
    const events: ConversationStreamEvent[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

describe('Hugging Face IE canonical lifecycle', () => {
    it('preserves legacy prompt formatting while canonical entrypoints reject unsupported contributions', async () => {
        const { driver, executor } = setup();
        const getStream = vi.fn(async () => new ReadableStream());
        const segments = [
            { role: PromptRole.system, content: 'System context.' },
            {
                role: PromptRole.user,
                content: 'Question.',
                files: [
                    {
                        name: 'ignored.txt',
                        mime_type: 'text/plain',
                        getStream,
                        getURL: async () => 'https://example.test/ignored.txt',
                        getURI: async () => 'artifact://ignored',
                    },
                ],
            },
            { role: PromptRole.assistant, content: 'Earlier answer.' },
            { role: PromptRole.tool, content: 'Legacy ignored tool result.' },
            { role: PromptRole.negative, content: 'Legacy ignored negative prompt.' },
            { role: PromptRole.mask, content: 'Legacy ignored mask.' },
            { role: PromptRole.safety, content: 'Safety rule.' },
        ];

        await driver.execute(segments, options('legacy-formatting'));
        expect(executor.textGeneration).toHaveBeenCalledWith({
            inputs: [
                'CONTEXT: System context.',
                'USER: Question.\nASSISTANT: Earlier answer.',
                'IMPORTANT: Safety rule.',
            ].join('\n'),
            parameters: { max_new_tokens: 64, temperature: 0.2 },
        });
        expect(getStream).not.toHaveBeenCalled();

        await expect(driver.executeCanonical(segments, options('canonical-formatting'))).rejects.toThrow(
            /does not support media input/,
        );
        expect(executor.textGeneration).toHaveBeenCalledOnce();
    });

    it('binds the concrete endpoint and exact request, then JSON-recovers without another inference', async () => {
        const { driver, executor, target } = setup();
        const textGeneration = vi.mocked(executor.textGeneration);
        let prepared = 0;
        const segments = [{ role: PromptRole.user, content: 'Answer exactly.' }];
        const first = await driver.executeCanonical(segments, {
            ...options('first'),
            on_canonical_request_prepared: async () => {
                prepared += 1;
                expect(textGeneration).not.toHaveBeenCalled();
            },
        });

        expect(textGeneration).toHaveBeenCalledWith({
            inputs: 'Answer exactly.',
            parameters: {
                max_new_tokens: 64,
                temperature: 0.2,
                details: true,
                return_full_text: false,
            },
        });
        const generation = first.conversation.generations[first.accepted_output.generation.id];
        expect(generation).toMatchObject({
            provider: 'huggingface_ie',
            protocol: 'huggingface.text-generation',
            requested_model: MODEL,
            resolved_model: MODEL,
            finish_reason: 'stop',
            usage: { output_tokens: 3 },
            request_receipt: {
                target: {
                    model: MODEL,
                    options: {
                        management_endpoint: MANAGEMENT_ENDPOINT,
                        inference_endpoint: INFERENCE_ENDPOINT,
                        parameters: {
                            max_new_tokens: 64,
                            temperature: 0.2,
                            details: true,
                            return_full_text: false,
                        },
                    },
                },
            },
        });
        expect(generation?.usage?.input_tokens).toBeUndefined();
        expect(generation?.usage?.total_tokens).toBeUndefined();
        expect(JSON.stringify(generation?.request_receipt)).not.toContain('test-key');

        const persisted = JSON.parse(JSON.stringify(first.conversation)) as ConversationDocument;
        const recovered = await driver.executeCanonical(segments, options('retry', persisted));
        expect(recovered.accepted_output).toEqual(first.accepted_output);
        expect(textGeneration).toHaveBeenCalledOnce();
        expect(prepared).toBe(1);
        expect(target).toHaveBeenCalledOnce();

        const offline = new HuggingFaceIEDriver({ apiKey: 'test-key', endpoint_url: MANAGEMENT_ENDPOINT });
        const management = vi.fn(async () => {
            throw new Error('management unavailable');
        });
        Object.defineProperty(offline.service, 'get', { value: management });
        const offlineRecovered = await offline.executeCanonical(segments, options('offline-retry', persisted));
        expect(offlineRecovered.accepted_output).toEqual(first.accepted_output);
        const offlineStream = await offline.streamCanonicalEvents(
            segments,
            options('offline-stream-retry', persisted),
            undefined,
            { stream_id: 'stream:huggingface:offline-retry' },
        );
        await expect(collect(offlineStream)).resolves.toEqual([
            expect.objectContaining({ type: 'response_accepted', origin: 'accepted_recovery' }),
        ]);
        await offlineStream.closed;
        expect(management).not.toHaveBeenCalled();

        await expect(
            driver.executeCanonical(segments, {
                ...options('changed', persisted),
                model_options: { _option_id: 'text-fallback', max_tokens: 32, temperature: 0.2 },
            }),
        ).rejects.toThrow(/fingerprint|request|incompatible/i);
        expect(textGeneration).toHaveBeenCalledOnce();

        const rerouted = new HuggingFaceIEDriver({
            apiKey: 'test-key',
            endpoint_url: 'https://other-management.huggingface.test',
        });
        await expect(rerouted.executeCanonical(segments, options('rerouted', persisted))).rejects.toThrow(
            /incompatible Hugging Face target options/,
        );
    });

    it('keeps model endpoint executors isolated instead of reusing the first model target', async () => {
        const driver = new HuggingFaceIEDriver({ apiKey: 'test-key', endpoint_url: MANAGEMENT_ENDPOINT });
        const get = vi.fn(async (path: string) => ({
            status: {
                state: 'running',
                url: path === '/endpoint-a' ? 'https://endpoint-a.test' : 'https://endpoint-b.test',
            },
        }));
        Object.defineProperty(driver.service, 'get', { value: get });

        const first = await driver.getExecutorTarget('endpoint-a');
        const second = await driver.getExecutorTarget('endpoint-b');
        const firstAgain = await driver.getExecutorTarget('endpoint-a');

        expect(first.url).toBe('https://endpoint-a.test');
        expect(second.url).toBe('https://endpoint-b.test');
        expect(first.executor).not.toBe(second.executor);
        expect(firstAgain).toBe(first);
        expect(get).toHaveBeenCalledTimes(2);
    });

    it('projects token increments while retaining special-token finish and usage evidence', async () => {
        const nativeStream = vi.fn((request: { parameters?: Record<string, unknown> }) => {
            if (request.parameters?.decoder_input_details === true) {
                throw new Error('TGI rejects decoder_input_details for streaming');
            }
            return streamEvents('length');
        });
        const { driver, executor } = setup({
            textGenerationStream: nativeStream as InferenceClient['textGenerationStream'],
        });
        const segments = [{ role: PromptRole.user, content: 'Keep writing.' }];
        const typed = await driver.streamCanonicalEvents(segments, options('typed'), undefined, {
            stream_id: 'stream:huggingface:typed',
        });
        const events = await collect(typed);
        expect(events.flatMap((event) => (event.type === 'draft_text_delta' ? [event.text] : []))).toEqual([
            'Hugging ',
            'Face',
        ]);
        expect(events.at(-1)).toMatchObject({ type: 'response_accepted' });
        expect(typed.completion?.accepted_output.turn).toMatchObject({
            status: 'interrupted',
            blocks: [expect.objectContaining({ type: 'text', text: 'Hugging Face' })],
        });
        const generationId = typed.completion?.accepted_output.generation.id;
        if (generationId === undefined) throw new Error('Missing Hugging Face generation');
        expect(typed.completion?.conversation.generations[generationId]).toMatchObject({
            status: 'cancelled',
            finish_reason: 'length',
            usage: { input_tokens: 2, output_tokens: 2, total_tokens: 4 },
        });
        expect(executor.textGenerationStream).toHaveBeenNthCalledWith(
            1,
            {
                inputs: 'Keep writing.',
                parameters: {
                    max_new_tokens: 64,
                    temperature: 0.2,
                    details: true,
                    return_full_text: false,
                },
            },
            { signal: expect.any(AbortSignal) },
        );
        const streamedRequest = vi.mocked(executor.textGenerationStream).mock.calls[0]?.[0];
        expect(streamedRequest?.parameters).not.toHaveProperty('decoder_input_details');
        const typedCompletion = typed.completion;
        if (typedCompletion === undefined) throw new Error('Missing Hugging Face completion');
        await typed.closed;

        const persisted = JSON.parse(JSON.stringify(typedCompletion.conversation)) as ConversationDocument;
        const recovered = await driver.streamCanonicalEvents(segments, options('typed-retry', persisted), undefined, {
            stream_id: 'stream:huggingface:typed-retry',
        });
        await expect(collect(recovered)).resolves.toEqual([
            expect.objectContaining({ type: 'response_accepted', origin: 'accepted_recovery' }),
        ]);
        expect(executor.textGenerationStream).toHaveBeenCalledOnce();
        await recovered.closed;

        const stringOptions = options('string');
        if (stringOptions.conversation_runtime === undefined) throw new Error('Missing Hugging Face runtime');
        stringOptions.conversation_runtime = {
            ...stringOptions.conversation_runtime,
            conversation_id: 'conversation:huggingface:string',
            request_id: 'request:huggingface:string',
            input_operation_id: 'input:huggingface:string',
            response_operation_id: 'response:huggingface:string',
        };
        const strings = await driver.streamCanonical(segments, stringOptions);
        const chunks: string[] = [];
        for await (const chunk of strings) chunks.push(chunk);
        expect(chunks).toEqual(['Hugging ', 'Face']);
        expect(strings.completion?.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'text', text: 'Hugging Face' }),
        ]);
        expect(executor.textGenerationStream).toHaveBeenCalledTimes(2);

        const legacy = await driver.requestTextCompletionStream('Keep writing.', options('legacy'));
        const legacyChunks: CompletionChunkObject[] = [];
        for await (const chunk of legacy) legacyChunks.push(chunk);
        expect(legacyChunks.map((chunk) => chunk.result)).toEqual([
            [{ type: 'text', value: 'Hugging ' }],
            [{ type: 'text', value: 'Face' }],
            [],
        ]);
        expect(legacyChunks.at(-1)).toMatchObject({
            finish_reason: 'length',
            token_usage: { result: 2 },
        });
    });

    it('normalizes valid structured output and retains invalid output as failed', async () => {
        const responses = [syncResponse('{"answer":"yes"}'), syncResponse('not json')];
        const { driver } = setup({ textGeneration: vi.fn(async () => responses.shift() ?? syncResponse()) });
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

        const invalidOptions = options('structured-invalid');
        if (invalidOptions.conversation_runtime === undefined) throw new Error('Missing Hugging Face runtime');
        invalidOptions.conversation_runtime = {
            ...invalidOptions.conversation_runtime,
            conversation_id: 'conversation:huggingface:invalid',
            request_id: 'request:huggingface:invalid',
            input_operation_id: 'input:huggingface:invalid',
            response_operation_id: 'response:huggingface:invalid',
        };
        const invalid = await driver.executeCanonical([{ role: PromptRole.user, content: 'Return JSON.' }], {
            ...invalidOptions,
            result_schema: resultSchema,
        });
        expect(invalid.accepted_output.turn).toMatchObject({
            status: 'failed',
            blocks: [expect.objectContaining({ type: 'text', text: 'not json' })],
        });

        const structuredStream = vi.fn(() =>
            (async function* () {
                yield token({ text: '{"answer":' });
                yield token({
                    text: '"streamed"}',
                    generated_text: '{"answer":"streamed"}',
                    finish_reason: 'stop_sequence',
                });
            })(),
        );
        const streamedSetup = setup({
            textGenerationStream: structuredStream as InferenceClient['textGenerationStream'],
        });
        const streamedOptions = options('structured-stream');
        if (streamedOptions.conversation_runtime === undefined) throw new Error('Missing Hugging Face runtime');
        streamedOptions.conversation_runtime = {
            ...streamedOptions.conversation_runtime,
            conversation_id: 'conversation:huggingface:structured-stream',
            request_id: 'request:huggingface:structured-stream',
            input_operation_id: 'input:huggingface:structured-stream',
            response_operation_id: 'response:huggingface:structured-stream',
        };
        streamedOptions.result_schema = resultSchema;
        const stream = await streamedSetup.driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Return streamed JSON.' }],
            streamedOptions,
            undefined,
            { stream_id: 'stream:huggingface:structured' },
        );
        const events = await collect(stream);
        expect(events.at(-1)).toMatchObject({
            type: 'response_accepted',
            reconciliations: [{ disposition: 'structured_output' }],
        });
        expect(stream.completion?.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'json', value: { answer: 'streamed' } }),
        ]);
        await stream.closed;
    });

    it('rejects tools, media, and continuation before inference', async () => {
        const { driver, executor } = setup();
        const getStream = vi.fn(async () => new ReadableStream());
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Use a tool.' }], {
                ...options('tools'),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            }),
        ).rejects.toThrow(/does not support tools/);
        await expect(
            driver.executeCanonical(
                [
                    {
                        role: PromptRole.user,
                        content: 'Read this.',
                        files: [
                            {
                                name: 'note.txt',
                                mime_type: 'text/plain',
                                getStream,
                                getURL: async () => 'https://example.test/note.txt',
                                getURI: async () => 'artifact://note',
                            },
                        ],
                    },
                ],
                options('media'),
            ),
        ).rejects.toThrow(/does not support media input/);
        expect(getStream).not.toHaveBeenCalled();

        const first = await driver.executeCanonical(
            [{ role: PromptRole.user, content: 'First.' }],
            options('continuation-first'),
        );
        const continuation = options('continuation-second', first.conversation);
        if (continuation.conversation_runtime === undefined) throw new Error('Missing Hugging Face runtime');
        continuation.conversation_runtime = {
            ...continuation.conversation_runtime,
            request_id: 'request:huggingface:continuation',
            attempt_id: 'attempt:huggingface:continuation',
            input_operation_id: 'input:huggingface:continuation',
            response_operation_id: 'response:huggingface:continuation',
        };
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Continue.' }], continuation),
        ).rejects.toThrow(/does not support conversation continuation/);
        expect(executor.textGeneration).toHaveBeenCalledOnce();
    });

    it('blocks sync and typed transport when prepared-request publication fails', async () => {
        const { driver, executor } = setup();
        const fail = async () => {
            throw new Error('durability barrier failed');
        };
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Do not send.' }], {
                ...options('barrier'),
                on_canonical_request_prepared: fail,
            }),
        ).rejects.toThrow('durability barrier failed');
        await expect(
            driver.streamCanonicalEvents(
                [{ role: PromptRole.user, content: 'Do not stream.' }],
                { ...options('typed-barrier'), on_canonical_request_prepared: fail },
                undefined,
                { stream_id: 'stream:huggingface:barrier' },
            ),
        ).rejects.toThrow('durability barrier failed');
        expect(executor.textGeneration).not.toHaveBeenCalled();
        expect(executor.textGenerationStream).not.toHaveBeenCalled();
    });

    it('settles cancellation and aborts the native stream', async () => {
        let requestSignal: AbortSignal | undefined;
        const pendingStream = vi.fn((_request: unknown, requestOptions?: { signal?: AbortSignal }) => {
            requestSignal = requestOptions?.signal;
            return (async function* () {
                await new Promise<void>((_resolve, reject) => {
                    requestOptions?.signal?.addEventListener(
                        'abort',
                        () => reject(new DOMException('aborted', 'AbortError')),
                        { once: true },
                    );
                });
            })();
        });
        const { driver } = setup({ textGenerationStream: pendingStream as InferenceClient['textGenerationStream'] });
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Wait.' }],
            options('cancel'),
            undefined,
            { stream_id: 'stream:huggingface:cancel' },
        );
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_started' } });
        const pending = iterator.next();
        await vi.waitFor(() => expect(pendingStream).toHaveBeenCalledOnce());
        const terminal = await stream.cancel();
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        await expect(pending).resolves.toMatchObject({ value: terminal });
        expect(requestSignal?.aborted).toBe(true);
        await stream.closed;
    });

    it('fails throwing native streams without accepting partial text', async () => {
        const throwing = vi.fn(() =>
            (async function* () {
                yield token({ text: 'partial' });
                throw new Error('provider private payload');
            })(),
        );
        const { driver } = setup({ textGenerationStream: throwing as InferenceClient['textGenerationStream'] });
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Fail safely.' }],
            options('native-error'),
            undefined,
            { stream_id: 'stream:huggingface:error' },
        );
        const events = await collect(stream);
        const terminal = events.at(-1);
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'failed' });
        expect(JSON.stringify(terminal)).not.toContain('provider private payload');
        expect(events).not.toContainEqual(expect.objectContaining({ type: 'response_accepted' }));
        expect(stream.completion).toBeUndefined();
        await stream.closed;

        const incompleteSetup = setup({
            textGenerationStream: vi.fn(() =>
                (async function* () {
                    yield token({ text: 'partial' });
                })(),
            ),
        });
        const incompleteOptions = options('incomplete');
        if (incompleteOptions.conversation_runtime === undefined) throw new Error('Missing Hugging Face runtime');
        incompleteOptions.conversation_runtime = {
            ...incompleteOptions.conversation_runtime,
            conversation_id: 'conversation:huggingface:incomplete',
            request_id: 'request:huggingface:incomplete',
            input_operation_id: 'input:huggingface:incomplete',
            response_operation_id: 'response:huggingface:incomplete',
        };
        const incomplete = await incompleteSetup.driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Require terminal details.' }],
            incompleteOptions,
            undefined,
            { stream_id: 'stream:huggingface:incomplete' },
        );
        const incompleteEvents = await collect(incomplete);
        expect(incompleteEvents.at(-1)).toMatchObject({ type: 'stream_terminated', outcome: 'failed' });
        expect(incomplete.completion).toBeUndefined();
        await incomplete.closed;
    });
});
