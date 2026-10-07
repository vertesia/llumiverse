import { type ConversationDocument, type ConversationStreamEvent, hashContentBytes } from '@llumiverse/conversation';
import {
    type CanonicalExecutionEventStream,
    type CanonicalExecutionInputOptions,
    type ExecutionOptions,
    PromptRole,
} from '@llumiverse/core';
import type { Prediction } from 'replicate';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
    predictions: {
        create: vi.fn(),
        get: vi.fn(),
        cancel: vi.fn(),
    },
    sources: [] as Array<{
        close: ReturnType<typeof vi.fn>;
        emit(type: string, data?: string): void;
    }>,
}));

vi.mock('replicate', () => ({
    default: class {
        predictions = mocks.predictions;
        fetch: typeof fetch;

        constructor(options: { fetch: typeof fetch }) {
            this.fetch = options.fetch;
        }
    },
}));

vi.mock('eventsource', () => ({
    EventSource: class {
        close = vi.fn();
        private readonly listeners = new Map<string, (event: { data: string }) => void>();

        constructor(_url: string, _options: { fetch: typeof fetch }) {
            mocks.sources.push(this);
        }

        addEventListener(type: string, listener: (event: { data: string }) => void) {
            this.listeners.set(type, listener);
        }

        emit(type: string, data = '') {
            this.listeners.get(type)?.({ data });
        }
    },
}));

import { ReplicateDriver } from './replicate.js';

const MODEL = 'owner/model:version-1';
const segments = [
    { role: PromptRole.system, content: 'System context.' },
    { role: PromptRole.user, content: 'Question.' },
    { role: PromptRole.assistant, content: 'Earlier answer.' },
    { role: PromptRole.user, content: 'Continue.' },
    { role: PromptRole.safety, content: 'Safety rule.' },
];

function options(flow: string, conversation?: ConversationDocument): CanonicalExecutionInputOptions {
    return {
        model: MODEL,
        ...(conversation === undefined ? {} : { conversation }),
        model_options: { _option_id: 'text-fallback', max_tokens: 32, temperature: 0.2 },
        conversation_runtime: {
            conversation_id: `conversation:replicate:${flow}`,
            request_id: `request:replicate:${flow}`,
            attempt_id: `attempt:replicate:${flow}:first`,
            input_operation_id: `input:replicate:${flow}`,
            response_operation_id: `response:replicate:${flow}`,
            recorded_at: '2026-09-30T00:00:00.000Z',
            started_at: '2026-09-30T00:00:00.000Z',
            completed_at: '2026-09-30T00:00:01.000Z',
        },
    };
}

function completed(output: unknown = ['Replicate ', 'answer']): Prediction {
    return {
        id: 'prediction-1',
        status: 'succeeded',
        model: 'owner/model',
        version: 'version-1',
        input: {},
        output,
        source: 'api',
        created_at: '2026-09-30T00:00:00.000Z',
        completed_at: '2026-09-30T00:00:01.000Z',
        metrics: { predict_time: 0.7, total_time: 0.9 },
        urls: {
            get: 'https://api.replicate.com/v1/predictions/prediction-1',
            cancel: 'https://api.replicate.com/v1/predictions/prediction-1/cancel',
        },
    } as Prediction;
}

function processing(stream = true): Prediction {
    return {
        ...completed(undefined),
        status: 'processing',
        output: undefined,
        completed_at: undefined,
        urls: {
            get: 'https://api.replicate.com/v1/predictions/prediction-1',
            cancel: 'https://api.replicate.com/v1/predictions/prediction-1/cancel',
            ...(stream ? { stream: 'https://stream.replicate.com/v1/files/prediction-1' } : {}),
        },
    } as Prediction;
}

async function collect(stream: CanonicalExecutionEventStream): Promise<ConversationStreamEvent[]> {
    const events: ConversationStreamEvent[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

describe('Replicate canonical lifecycle', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mocks.sources.length = 0;
        mocks.predictions.cancel.mockResolvedValue({ status: 'canceled' });
    });

    it('publishes the exact prepared request, preserves source authority, and JSON-recovers without transport', async () => {
        mocks.predictions.create.mockResolvedValue(completed());
        const driver = new ReplicateDriver({ apiKey: 'secret' });
        let prepared = 0;
        const firstOptions = options('sync');
        const first = await driver.executeCanonical(segments, {
            ...firstOptions,
            on_canonical_request_prepared: async () => {
                prepared += 1;
                expect(mocks.predictions.create).not.toHaveBeenCalled();
            },
        });

        expect(mocks.predictions.create).toHaveBeenCalledWith({
            version: 'version-1',
            input: {
                prompt: [
                    'CONTEXT: System context.',
                    'USER: Question.\nASSISTANT: Earlier answer.\nUSER: Continue.',
                    'IMPORTANT: Safety rule.',
                ].join('\n'),
                max_new_tokens: 32,
                temperature: 0.2,
            },
        });
        expect(first.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'text', text: 'Replicate answer' }),
        ]);
        const generation = first.conversation.generations[first.accepted_output.generation.id];
        expect(generation).toMatchObject({
            provider: 'replicate',
            protocol: 'replicate.predictions',
            requested_model: MODEL,
            provider_response_id: 'prediction-1',
            usage: {
                reported_usage: [
                    expect.objectContaining({
                        accounting_basis: 'replicate_seconds',
                        payload: { predict_time: 0.7, total_time: 0.9 },
                    }),
                ],
            },
            request_receipt: {
                target: {
                    options: { owner: 'owner', model: 'model', version: 'version-1' },
                },
            },
        });
        expect(generation?.usage).not.toHaveProperty('input_tokens');
        expect(JSON.stringify(generation?.request_receipt)).not.toContain('secret');
        expect(first.conversation.turns.slice(0, 5).map((turn) => [turn.kind, turn.authority])).toEqual([
            ['program', 'system'],
            ['user', 'ordinary'],
            ['agent', 'ordinary'],
            ['user', 'ordinary'],
            ['program', 'system'],
        ]);

        const persisted = JSON.parse(JSON.stringify(first.conversation)) as ConversationDocument;
        const retryOptions = options('sync', persisted);
        if (retryOptions.conversation_runtime === undefined) throw new Error('Missing Replicate runtime');
        retryOptions.conversation_runtime = {
            ...retryOptions.conversation_runtime,
            attempt_id: 'attempt:replicate:sync:retry',
        };
        const recovered = await driver.executeCanonical(segments, retryOptions);
        expect(recovered.accepted_output).toEqual(first.accepted_output);
        expect(mocks.predictions.create).toHaveBeenCalledOnce();
        expect(mocks.predictions.get).not.toHaveBeenCalled();
        expect(prepared).toBe(1);

        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Changed.' }], retryOptions),
        ).rejects.toThrow(/request|context|incompatible|different payload/i);
        expect(mocks.predictions.create).toHaveBeenCalledOnce();
    });

    it('streams native output but waits for the authoritative final Prediction before acceptance', async () => {
        mocks.predictions.create.mockResolvedValue(processing());
        let finishGet!: (prediction: Prediction) => void;
        mocks.predictions.get.mockReturnValue(new Promise((resolve) => (finishGet = resolve)));
        const driver = new ReplicateDriver({ apiKey: 'secret' });
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Stream.' }],
            options('typed'),
            undefined,
            { stream_id: 'stream:replicate:typed' },
        );
        const collected = collect(stream);
        await vi.waitFor(() => expect(mocks.sources).toHaveLength(1));
        mocks.sources[0].emit('output', 'Replicate ');
        mocks.sources[0].emit('output', 'answer');
        mocks.sources[0].emit('done', '{}');
        await vi.waitFor(() =>
            expect(mocks.predictions.get).toHaveBeenCalledWith('prediction-1', { signal: expect.any(AbortSignal) }),
        );
        expect(stream.completion).toBeUndefined();

        finishGet(completed());
        const events = await collected;
        expect(events.flatMap((event) => (event.type === 'draft_text_delta' ? [event.text] : []))).toEqual([
            'Replicate ',
            'answer',
        ]);
        expect(events.at(-1)).toMatchObject({ type: 'response_accepted' });
        expect(stream.completion?.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'text', text: 'Replicate answer' }),
        ]);
        await stream.closed;

        const persisted = JSON.parse(JSON.stringify(stream.completion?.conversation)) as ConversationDocument;
        const retry = options('typed', persisted);
        if (retry.conversation_runtime === undefined || stream.completion === undefined)
            throw new Error('Missing runtime');
        retry.conversation_runtime = {
            ...retry.conversation_runtime,
            conversation_id: persisted.id,
            request_id: stream.completion.accepted_output.generation.request_id,
        };
        const recovered = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Stream.' }],
            retry,
            undefined,
            { stream_id: 'stream:replicate:retry' },
        );
        await expect(collect(recovered)).resolves.toEqual([
            expect.objectContaining({ type: 'response_accepted', origin: 'accepted_recovery' }),
        ]);
        expect(mocks.predictions.create).toHaveBeenCalledOnce();
        expect(mocks.predictions.get).toHaveBeenCalledOnce();
        await recovered.closed;

        mocks.predictions.create.mockResolvedValueOnce(processing());
        mocks.predictions.get.mockResolvedValueOnce(completed(['String ', 'answer']));
        const stringStream = await driver.streamCanonical(
            [{ role: PromptRole.user, content: 'String stream.' }],
            options('string'),
        );
        const stringsPromise = (async () => {
            const chunks: string[] = [];
            for await (const chunk of stringStream) chunks.push(chunk);
            return chunks;
        })();
        await vi.waitFor(() => expect(mocks.sources).toHaveLength(2));
        mocks.sources[1].emit('output', 'String ');
        mocks.sources[1].emit('output', 'answer');
        mocks.sources[1].emit('done', '{}');
        await expect(stringsPromise).resolves.toEqual(['String ', 'answer']);
        expect(stringStream.completion?.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'text', text: 'String answer' }),
        ]);
    });

    it('keeps SSE done non-authoritative when the final Prediction fails', async () => {
        mocks.predictions.create.mockResolvedValue(processing());
        mocks.predictions.get.mockResolvedValue({ ...completed(), status: 'failed', error: 'model crashed' });
        const driver = new ReplicateDriver({ apiKey: 'secret' });
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Stream.' }],
            options('failed'),
            undefined,
            { stream_id: 'stream:replicate:failed' },
        );
        const collected = collect(stream);
        await vi.waitFor(() => expect(mocks.sources).toHaveLength(1));
        mocks.sources[0].emit('output', 'partial');
        mocks.sources[0].emit('done', '{}');
        const events = await collected;
        expect(events.at(-1)).toMatchObject({ type: 'stream_terminated', outcome: 'failed' });
        expect(stream.completion).toBeUndefined();
        await stream.closed;
    });

    it('durably stores trusted Replicate media bytes and never sends credentials to arbitrary output URLs', async () => {
        const bytes = new Uint8Array([0x52, 0x49, 0x46, 0x46, 1, 2, 3, 4, 0x57, 0x41, 0x56, 0x45]);
        const integrity = await hashContentBytes(bytes);
        mocks.predictions.create
            .mockResolvedValueOnce(completed(['https://cdn.replicate.delivery/output.wav']))
            .mockResolvedValueOnce(completed(['https://example.test/not-a-replicate-file.png']));
        const driver = new ReplicateDriver({ apiKey: 'secret' });
        const providerFetch = vi.fn(
            async () =>
                new Response(bytes, {
                    status: 200,
                    headers: { 'content-type': 'audio/wav', 'content-length': String(bytes.byteLength) },
                }),
        );
        vi.spyOn(driver, 'fetchGeneratedAsset').mockImplementation(providerFetch);
        const store = vi.fn<NonNullable<ExecutionOptions['store_generated_asset']>>(async (stream, metadata) => {
            expect(metadata).toEqual({ kind: 'audio', mime_type: 'audio/wav' });
            expect(new Uint8Array(await new Response(stream).arrayBuffer())).toEqual(bytes);
            return {
                storage: { type: 'external', resolver: 'test', locator: { key: 'replicate/output.wav' } },
                byte_length: bytes.byteLength,
                content_hash: integrity.content_hash,
            };
        });
        const media = await driver.executeCanonical([{ role: PromptRole.user, content: 'Audio.' }], {
            ...options('media'),
            store_generated_asset: store,
        });
        expect(media.accepted_output.turn.blocks).toEqual([expect.objectContaining({ type: 'audio' })]);
        expect(Object.values(media.conversation.assets)).toEqual([
            expect.objectContaining({
                kind: 'audio',
                byte_length: bytes.byteLength,
                content_hash: integrity.content_hash,
                storage: { type: 'external', resolver: 'test', locator: { key: 'replicate/output.wav' } },
            }),
        ]);
        expect(providerFetch).toHaveBeenCalledWith(new URL('https://cdn.replicate.delivery/output.wav'), undefined);

        const arbitrary = await driver.executeCanonical(
            [{ role: PromptRole.user, content: 'URL.' }],
            options('arbitrary'),
        );
        expect(arbitrary.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'text', text: 'https://example.test/not-a-replicate-file.png' }),
        ]);
        expect(providerFetch).toHaveBeenCalledOnce();
    });

    it('authenticates only validated asset downloads and refuses credential-bearing redirects', async () => {
        const driver = new ReplicateDriver({ apiKey: 'secret' });
        const managedFetch = vi.fn(async () => new Response(new Uint8Array([1]), { status: 200 }));
        vi.spyOn(driver as unknown as { getDriverFetch(): typeof fetch }, 'getDriverFetch').mockReturnValue(
            managedFetch,
        );

        await driver.fetchGeneratedAsset(new URL('https://cdn.replicate.delivery/output.png'));

        expect(managedFetch).toHaveBeenCalledWith(new URL('https://cdn.replicate.delivery/output.png'), {
            headers: { Authorization: 'Bearer secret' },
            redirect: 'error',
        });
        await expect(driver.fetchGeneratedAsset(new URL('https://example.test/output.png'))).rejects.toThrow(
            /trusted delivery host/,
        );
        expect(managedFetch).toHaveBeenCalledOnce();
    });

    it('normalizes structured text and rejects a successful empty Prediction', async () => {
        mocks.predictions.create
            .mockResolvedValueOnce(completed(['{"answer":"yes"}']))
            .mockResolvedValueOnce(completed({ answer: 'native' }))
            .mockResolvedValueOnce(completed({ other: 42 }))
            .mockResolvedValueOnce(completed([]));
        const driver = new ReplicateDriver({ apiKey: 'secret' });
        const resultSchema = {
            type: 'object' as const,
            properties: { answer: { type: 'string' as const } },
            required: ['answer'],
            additionalProperties: false,
        };
        const result = await driver.executeCanonical([{ role: PromptRole.user, content: 'JSON.' }], {
            ...options('structured'),
            result_schema: resultSchema,
        });
        expect(result.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'json', value: { answer: 'yes' } }),
        ]);

        const native = await driver.executeCanonical([{ role: PromptRole.user, content: 'Native JSON.' }], {
            ...options('structured-native'),
            result_schema: resultSchema,
        });
        expect(native.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'json', value: { answer: 'native' } }),
        ]);

        const invalid = await driver.executeCanonical([{ role: PromptRole.user, content: 'Invalid JSON.' }], {
            ...options('structured-invalid'),
            result_schema: resultSchema,
        });
        expect(invalid.accepted_output.turn).toMatchObject({
            status: 'failed',
            blocks: [expect.objectContaining({ type: 'json', value: { other: 42 } })],
        });

        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Empty.' }], options('empty')),
        ).rejects.toThrow(/no output/);
    });

    it('binds structured schemas independently of a custom formatter that returns the same prompt', async () => {
        mocks.predictions.create.mockResolvedValue(completed(['{"answer":"yes"}']));
        const driver = new ReplicateDriver({ apiKey: 'secret' });
        const format = vi.fn(() => 'constant native prompt');
        const answerSchema = {
            type: 'object' as const,
            properties: { answer: { type: 'string' as const } },
            required: ['answer'],
            additionalProperties: false,
        };
        const first = await driver.executeCanonical([{ role: PromptRole.user, content: 'First schema.' }], {
            ...options('schema-binding'),
            format,
            result_schema: answerSchema,
        });
        const persisted = JSON.parse(JSON.stringify(first.conversation)) as ConversationDocument;

        const exact = await driver.executeCanonical([{ role: PromptRole.user, content: 'First schema.' }], {
            ...options('schema-binding', persisted),
            format,
            result_schema: answerSchema,
        });
        expect(exact.accepted_output).toEqual(first.accepted_output);
        expect(mocks.predictions.create).toHaveBeenCalledOnce();

        const changedSchema = {
            type: 'object' as const,
            properties: { other: { type: 'string' as const } },
            required: ['other'],
            additionalProperties: false,
        };
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'First schema.' }], {
                ...options('schema-binding', persisted),
                format,
                result_schema: changedSchema,
            }),
        ).rejects.toThrow(/incompatible Replicate target options/);
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'First schema.' }], {
                ...options('schema-binding', persisted),
                format,
            }),
        ).rejects.toThrow(/incompatible Replicate target options/);
        expect(mocks.predictions.create).toHaveBeenCalledOnce();
        expect(mocks.predictions.get).not.toHaveBeenCalled();
    });

    it('rejects unsupported contributions and durability failure before provider transport', async () => {
        const driver = new ReplicateDriver({ apiKey: 'secret' });
        const getStream = vi.fn(async () => new ReadableStream());
        await expect(
            driver.executeCanonical(
                [
                    {
                        role: PromptRole.user,
                        content: 'Read.',
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
                options('media-input'),
            ),
        ).rejects.toThrow(/media input/);
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Tool.' }], {
                ...options('tools'),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            }),
        ).rejects.toThrow(/support tools/);
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Barrier.' }], {
                ...options('barrier'),
                on_canonical_request_prepared: async () => {
                    throw new Error('durability failed');
                },
            }),
        ).rejects.toThrow('durability failed');
        expect(getStream).not.toHaveBeenCalled();
        expect(mocks.predictions.create).not.toHaveBeenCalled();
    });

    it('cancels the remote Prediction and settles the typed consumer before transport cleanup', async () => {
        mocks.predictions.create.mockResolvedValue(processing(false));
        let rejectGet!: (reason?: unknown) => void;
        mocks.predictions.get.mockImplementation(
            (_id: string, request?: { signal?: AbortSignal }) =>
                new Promise((_resolve, reject) => {
                    rejectGet = reject;
                    request?.signal?.addEventListener('abort', () => reject(request.signal?.reason), { once: true });
                }),
        );
        const driver = new ReplicateDriver({ apiKey: 'secret' });
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Wait.' }],
            options('cancel'),
            undefined,
            { stream_id: 'stream:replicate:cancel' },
        );
        const collected = collect(stream);
        await vi.waitFor(() => expect(mocks.predictions.get).toHaveBeenCalledOnce());
        const terminal = await stream.cancel();
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        await expect(collected).resolves.toEqual([
            expect.objectContaining({ type: 'draft_started' }),
            expect.objectContaining({ type: 'stream_terminated', outcome: 'cancelled' }),
        ]);
        expect(mocks.predictions.cancel).toHaveBeenCalledWith('prediction-1');
        rejectGet(new DOMException('aborted', 'AbortError'));
        await stream.closed;
    });

    it('does not open SSE when prediction creation resolves after cancellation', async () => {
        let finishCreation!: (prediction: Prediction) => void;
        mocks.predictions.create.mockReturnValue(new Promise((resolve) => (finishCreation = resolve)));
        const driver = new ReplicateDriver({ apiKey: 'secret' });
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Wait.' }],
            options('cancel-create'),
            undefined,
            { stream_id: 'stream:replicate:cancel-create' },
        );
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_started' } });
        const pending = iterator.next();
        await vi.waitFor(() => expect(mocks.predictions.create).toHaveBeenCalledOnce());

        const terminal = await stream.cancel();
        finishCreation(processing());
        await expect(pending).resolves.toMatchObject({ value: terminal });
        await stream.closed;
        expect(mocks.predictions.cancel).toHaveBeenCalledWith('prediction-1');
        expect(mocks.sources).toHaveLength(0);
        expect(mocks.predictions.get).not.toHaveBeenCalled();
    });

    it('cancels the remote Prediction after an SSE error', async () => {
        mocks.predictions.create.mockResolvedValue(processing());
        const driver = new ReplicateDriver({ apiKey: 'secret' });
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Fail.' }],
            options('sse-error'),
            undefined,
            { stream_id: 'stream:replicate:sse-error' },
        );
        const collected = collect(stream);
        await vi.waitFor(() => expect(mocks.sources).toHaveLength(1));
        mocks.sources[0].emit('error', '{"detail":"upstream failed"}');

        const events = await collected;
        expect(events.at(-1)).toMatchObject({ type: 'stream_terminated', outcome: 'failed' });
        expect(mocks.predictions.cancel).toHaveBeenCalledWith('prediction-1');
        await stream.closed;
    });
});
