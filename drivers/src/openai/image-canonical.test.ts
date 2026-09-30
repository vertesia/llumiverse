import {
    appendConversationRecords,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    parseConversationDocument,
} from '@llumiverse/conversation';
import {
    Base64DataSource,
    type ExecutionOptions,
    isCanonicalAcceptedRecovery,
    PromptRole,
    Providers,
} from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIResponsesDriverBase } from './index.js';

class ImageDriver extends OpenAIResponsesDriverBase {
    provider: Providers.openai = Providers.openai;
    service: OpenAI;

    constructor(
        generate: (request: unknown, options?: { signal?: AbortSignal }) => Promise<unknown>,
        private readonly imageFetch: typeof fetch = vi.fn(),
    ) {
        super({});
        this.service = { images: { generate } } as unknown as OpenAI;
    }

    protected override getDriverFetch(): typeof fetch {
        return this.imageFetch;
    }
}

const at = '2026-09-30T03:00:00.000Z';
const pngSignature = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const webpSignature = Uint8Array.from([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50]);

function encodedImage(signature: Uint8Array, label: string): string {
    return Buffer.concat([Buffer.from(signature), Buffer.from(label)]).toString('base64');
}

const pngA = encodedImage(pngSignature, 'first image bytes');
const webpA = encodedImage(webpSignature, 'first webp image bytes');
const webpB = encodedImage(webpSignature, 'second webp image bytes');

function runtime(flow: string, conversation?: unknown): ExecutionOptions {
    return {
        model: 'gpt-image-1',
        ...(conversation === undefined ? {} : { conversation }),
        conversation_runtime: {
            conversation_id: `conversation:${flow}`,
            request_id: `request:${flow}`,
            attempt_id: `attempt:${flow}`,
            input_operation_id: `input:${flow}`,
            response_operation_id: `response:${flow}`,
            recorded_at: at,
        },
    };
}

function imageResponse(
    data: OpenAI.Images.Image[],
    usage: OpenAI.Images.ImagesResponse.Usage | undefined = {
        input_tokens: 12,
        input_tokens_details: { image_tokens: 0, text_tokens: 12 },
        output_tokens: 24,
        total_tokens: 36,
    },
): OpenAI.Images.ImagesResponse {
    return { created: 1, data, ...(usage === undefined ? {} : { usage }) };
}

async function consume(stream: AsyncIterable<string>): Promise<string> {
    let value = '';
    for await (const chunk of stream) value += chunk;
    return value;
}

async function hash(bytes: Uint8Array): Promise<string> {
    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', copy));
    return `sha256:${Array.from(digest, (byte) => byte.toString(16).padStart(2, '0')).join('')}`;
}

describe('OpenAI standalone image canonical lifecycle', () => {
    it('accepts multiple images directly, preserves revised prompts/options/usage, and retries exactly', async () => {
        const generate = vi.fn(async () =>
            imageResponse([
                { b64_json: webpA, revised_prompt: 'First revised prompt' },
                { b64_json: webpB, revised_prompt: 'Second revised prompt' },
            ]),
        );
        const publish = vi.fn(async () => undefined);
        const driver = new ImageDriver(generate);
        const segments = [{ role: PromptRole.user, content: 'Draw two small icons.' }];
        const options: ExecutionOptions = {
            ...runtime('multiple'),
            model_options: {
                _option_id: 'openai-gpt-image',
                size: '1024x1536',
                image_quality: 'high',
                background: 'transparent',
                output_format: 'webp',
            },
            on_canonical_request_prepared: publish,
        };

        const first = await driver.executeCanonical(segments, options);
        expect(isCanonicalAcceptedRecovery(first)).toBe(false);
        expect(first.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'image', caption: 'First revised prompt' }),
            expect.objectContaining({ type: 'image', caption: 'Second revised prompt' }),
        ]);
        expect(first.accepted_output.generation.usage).toMatchObject({
            input_tokens: 12,
            input_new_tokens: 12,
            output_tokens: 24,
            total_tokens: 36,
        });
        const generation = first.conversation.generations[first.accepted_output.generation.id];
        expect(generation?.request_receipt?.target.options).toEqual({
            n: 1,
            size: '1024x1536',
            quality: 'high',
            background: 'transparent',
            output_format: 'webp',
        });
        const assets = Object.values(first.conversation.assets);
        expect(assets).toHaveLength(2);
        expect(assets).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    kind: 'image',
                    mime_type: 'image/webp',
                    storage: { type: 'inline_base64', data: webpA },
                    metadata: { openai_image: { revised_prompt: 'First revised prompt' } },
                }),
            ]),
        );

        const retry = await driver.executeCanonical(segments, {
            ...options,
            conversation: JSON.parse(JSON.stringify(first.conversation)),
        });
        expect(retry.accepted_output).toEqual(first.accepted_output);
        expect(isCanonicalAcceptedRecovery(retry)).toBe(true);
        expect(isCanonicalAcceptedRecovery(JSON.parse(JSON.stringify(retry)))).toBe(false);
        const legacyRetry = await driver.execute(segments, {
            ...options,
            conversation: JSON.parse(JSON.stringify(first.conversation)),
        });
        expect(legacyRetry.result).toEqual(expect.arrayContaining([expect.objectContaining({ type: 'image' })]));
        expect(isCanonicalAcceptedRecovery(legacyRetry)).toBe(true);
        const firstRuntime = options.conversation_runtime;
        if (firstRuntime === undefined) throw new Error('Expected canonical image runtime');
        const typedRetry = await driver.streamCanonicalEvents(
            segments,
            {
                ...options,
                conversation: JSON.parse(JSON.stringify(first.conversation)),
                conversation_runtime: {
                    ...firstRuntime,
                    attempt_id: 'attempt:multiple:typed-retry',
                },
            },
            undefined,
            { stream_id: 'stream:openai-image:typed-retry' },
        );
        const typedEvents = [];
        for await (const event of typedRetry) typedEvents.push(event);
        expect(typedEvents).toEqual([
            expect.objectContaining({ type: 'response_accepted', origin: 'accepted_recovery' }),
        ]);
        expect(isCanonicalAcceptedRecovery(typedRetry.completion)).toBe(true);
        expect(generate).toHaveBeenCalledOnce();
        expect(publish).toHaveBeenCalledOnce();

        const priorAdapter = JSON.parse(JSON.stringify(first.conversation));
        const priorGeneration = priorAdapter.generations[first.accepted_output.generation.id];
        priorGeneration.adapter_version = '2026-09-30.canonical.1';
        priorGeneration.request_receipt.target.adapter_version = '2026-09-30.canonical.1';
        await expect(
            driver.executeCanonical(segments, {
                ...options,
                conversation: priorAdapter,
            }),
        ).rejects.toThrow(/unsupported adapter version 2026-09-30.canonical.1/);
        await expect(
            driver.executeCanonical([{ role: PromptRole.system, content: 'Draw two small icons.' }], {
                ...options,
                conversation: first.conversation,
            }),
        ).rejects.toThrow();
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Draw a changed icon.' }], {
                ...options,
                conversation: first.conversation,
            }),
        ).rejects.toThrow();
        await expect(
            driver.executeCanonical(segments, {
                ...options,
                conversation: first.conversation,
                model_options: {
                    _option_id: 'openai-gpt-image',
                    size: '1024x1536',
                    image_quality: 'low',
                    background: 'transparent',
                    output_format: 'webp',
                },
            }),
        ).rejects.toThrow('incompatible request identity');
        expect(generate).toHaveBeenCalledOnce();
    });

    it('preserves privileged and assistant prompt authority plus text attachments across public execution', async () => {
        const segments = [
            { role: PromptRole.system, content: 'Use a restrained geometric style.' },
            { role: PromptRole.safety, content: 'Do not include words.' },
            { role: PromptRole.assistant, content: 'The earlier result used circles.' },
            {
                role: PromptRole.user,
                content: 'Draw the next version.',
                files: [new Base64DataSource('notes.txt', 'text/plain', 'VXNlIGJsdWUu')],
            },
        ];
        const expectedPrompt = [
            'Use a restrained geometric style.',
            'DO NOT IGNORE - IMPORTANT: Do not include words.',
            'The earlier result used circles.',
            'Use blue.',
            'Draw the next version.',
        ].join('\n');
        const generate = vi.fn(async (_request: unknown) => imageResponse([{ b64_json: pngA }]));
        const driver = new ImageDriver(generate);
        const response = await driver.executeCanonical(segments, runtime('authority'));

        expect(generate.mock.calls[0]?.[0]).toMatchObject({ prompt: expectedPrompt });
        expect(response.conversation.turns).toEqual([
            expect.objectContaining({
                kind: 'program',
                authority: 'system',
                provenance: { type: 'received' },
                blocks: [expect.objectContaining({ type: 'text', text: 'Use a restrained geometric style.' })],
            }),
            expect.objectContaining({
                kind: 'program',
                authority: 'system',
                provenance: { type: 'received' },
                blocks: [
                    expect.objectContaining({
                        type: 'text',
                        text: 'DO NOT IGNORE - IMPORTANT: Do not include words.',
                    }),
                ],
            }),
            expect.objectContaining({
                kind: 'agent',
                authority: 'ordinary',
                provenance: { type: 'received' },
                blocks: [expect.objectContaining({ type: 'text', text: 'The earlier result used circles.' })],
            }),
            expect.objectContaining({
                kind: 'user',
                authority: 'ordinary',
                provenance: { type: 'received' },
                blocks: [
                    expect.objectContaining({ type: 'text', text: 'Use blue.' }),
                    expect.objectContaining({ type: 'text', text: 'Draw the next version.' }),
                ],
            }),
            expect.objectContaining({ kind: 'agent', provenance: { type: 'generated' } }),
        ]);
        const receivedTurnIds = response.conversation.turns
            .filter((turn) => turn.provenance.type === 'received')
            .map((turn) => turn.id);
        expect(receivedTurnIds).toHaveLength(4);
        expect(response.conversation.context.entries).toEqual(
            expect.arrayContaining(receivedTurnIds.map((turnId) => expect.objectContaining({ turn_id: turnId }))),
        );

        const legacyGenerate = vi.fn(async (_request: unknown) => imageResponse([{ b64_json: pngA }]));
        const legacy = new ImageDriver(legacyGenerate);
        const completion = await legacy.execute(segments, runtime('legacy-authority'));
        expect(completion.result).toEqual([{ type: 'image', value: `data:image/png;base64,${pngA}` }]);
        expect(legacyGenerate.mock.calls[0]?.[0]).toMatchObject({ prompt: expectedPrompt });
    });

    it('provides finite canonical streaming and projects legacy image usage from canonical authority', async () => {
        const generate = vi.fn(async () => imageResponse([{ b64_json: pngA }]));
        const driver = new ImageDriver(generate);
        const options = runtime('finite-stream');
        const segments = [{ role: PromptRole.user, content: 'Draw an icon.' }];

        const stream = await driver.streamCanonical(segments, options);
        expect(await consume(stream)).toBe('[Image]');
        expect(stream.completion?.accepted_output.turn.blocks[0]).toMatchObject({ type: 'image' });

        const legacy = await driver.execute(segments, runtime('legacy'));
        expect(legacy.result).toEqual([{ type: 'image', value: `data:image/png;base64,${pngA}` }]);
        expect(legacy.token_usage).toEqual({ prompt: 12, prompt_new: 12, result: 24, result_image: 24, total: 36 });
        expect(parseConversationDocument(legacy.conversation).turns.at(-1)?.kind).toBe('agent');
        expect(generate).toHaveBeenCalledTimes(2);
    });

    it.each(['sync', 'stream'] as const)(
        'blocks %s transport when prepared request publication fails',
        async (mode) => {
            const generate = vi.fn(async () => imageResponse([{ b64_json: pngA }]));
            const driver = new ImageDriver(generate);
            const options: ExecutionOptions = {
                ...runtime(`barrier-${mode}`),
                on_canonical_request_prepared: async () => {
                    throw new Error('image request was not durable');
                },
            };
            const execution =
                mode === 'sync'
                    ? driver.executeCanonical([{ role: PromptRole.user, content: 'Draw.' }], options)
                    : driver
                          .streamCanonical([{ role: PromptRole.user, content: 'Draw.' }], options)
                          .then((stream) => consume(stream));

            await expect(execution).rejects.toThrow('image request was not durable');
            expect(generate).not.toHaveBeenCalled();
        },
    );

    it('hydrates explicit URL output into durable canonical bytes using the actual MIME type', async () => {
        const bytes = Uint8Array.from([0xff, 0xd8, 0xff, ...new TextEncoder().encode('downloaded jpeg bytes')]);
        const generate = vi.fn(async (_request: unknown, _options?: { signal?: AbortSignal }) =>
            imageResponse([{ url: 'https://images.openai.test/generated', revised_prompt: 'A revised scene' }]),
        );
        const imageFetch = vi.fn<typeof fetch>(async (_input, init) => {
            init?.signal?.throwIfAborted();
            return new Response(bytes, {
                status: 200,
                headers: { 'content-type': 'image/jpeg', 'content-length': String(bytes.byteLength) },
            });
        });
        const driver = new ImageDriver(generate, imageFetch);
        const result = await driver.executeCanonical([{ role: PromptRole.user, content: 'Draw a scene.' }], {
            ...runtime('url'),
            model: 'dall-e-3',
            model_options: {
                _option_id: 'openai-dalle',
                size: '1792x1024',
                response_format: 'url',
                image_quality: 'hd',
                style: 'natural',
                n: 1,
            },
        });
        const asset = Object.values(result.conversation.assets)[0];

        expect(generate.mock.calls[0]?.[0]).toEqual(
            expect.objectContaining({
                response_format: 'url',
                quality: 'hd',
                style: 'natural',
                n: 1,
                size: '1792x1024',
            }),
        );
        const generation = result.conversation.generations[result.accepted_output.generation.id];
        expect(generation?.request_receipt?.target.options).toEqual({
            n: 1,
            size: '1792x1024',
            response_format: 'url',
            quality: 'hd',
            style: 'natural',
        });
        expect(asset).toMatchObject({
            mime_type: 'image/jpeg',
            byte_length: bytes.byteLength,
            storage: { type: 'inline_base64', data: Buffer.from(bytes).toString('base64') },
            metadata: {
                openai_image: {
                    revised_prompt: 'A revised scene',
                    source_url: 'https://images.openai.test/generated',
                },
            },
        });
        expect(asset?.content_hash).toBe(await hash(bytes));
        expect(imageFetch).toHaveBeenCalledOnce();
    });

    it('awaits an exact external asset sink before accepting the response', async () => {
        const bytes = new Uint8Array(Buffer.from(pngA, 'base64'));
        const store = vi.fn<NonNullable<ExecutionOptions['store_generated_asset']>>(async (stream, metadata) => {
            const stored = new Uint8Array(await new Response(stream).arrayBuffer());
            expect(stored).toEqual(bytes);
            expect(metadata).toEqual({ kind: 'image', mime_type: 'image/png' });
            return {
                storage: { type: 'external', resolver: 'url', locator: { url: 'gs://bucket/generated.png' } },
                byte_length: stored.byteLength,
                content_hash: await hash(stored),
            };
        });
        const driver = new ImageDriver(vi.fn(async () => imageResponse([{ b64_json: pngA }])));
        const result = await driver.executeCanonical([{ role: PromptRole.user, content: 'Draw.' }], {
            ...runtime('external'),
            store_generated_asset: store,
        });

        expect(Object.values(result.conversation.assets)[0]?.storage).toEqual({
            type: 'external',
            resolver: 'url',
            locator: { url: 'gs://bucket/generated.png' },
        });
        expect(store).toHaveBeenCalledOnce();
    });

    it('decodes an image at the eight megabyte inline boundary without a recursive base64 matcher', async () => {
        const bytes = new Uint8Array(8_000_000);
        bytes.set(pngSignature);
        const encoded = Buffer.from(bytes).toString('base64');
        const driver = new ImageDriver(vi.fn(async () => imageResponse([{ b64_json: encoded }], undefined)));

        const result = await driver.executeCanonical(
            [{ role: PromptRole.user, content: 'Draw a large texture.' }],
            runtime('large-inline'),
        );
        const asset = Object.values(result.conversation.assets)[0];

        expect(asset).toMatchObject({
            mime_type: 'image/png',
            byte_length: 8_000_000,
            storage: { type: 'inline_base64', data: encoded },
        });
    });

    it('counts empty URL response chunks and cancels an abusive body', async () => {
        const cancel = vi.fn();
        let pulls = 0;
        const body = new ReadableStream<Uint8Array>({
            pull(controller) {
                pulls += 1;
                if (pulls <= 8_193) {
                    controller.enqueue(new Uint8Array());
                    return;
                }
                controller.enqueue(pngSignature);
                controller.close();
            },
            cancel,
        });
        const imageFetch = vi.fn<typeof fetch>(async () =>
            Promise.resolve(new Response(body, { status: 200, headers: { 'content-type': 'image/png' } })),
        );
        const driver = new ImageDriver(
            vi.fn(async () => imageResponse([{ url: 'https://images.openai.test/chunked' }], undefined)),
            imageFetch,
        );

        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Draw.' }], runtime('empty-chunks')),
        ).rejects.toThrow('too many chunks');
        expect(cancel).toHaveBeenCalledOnce();
    });

    it('rejects an asset sink that does not return exact external storage', async () => {
        const driver = new ImageDriver(vi.fn(async () => imageResponse([{ b64_json: pngA }], undefined)));
        const store = vi.fn<NonNullable<ExecutionOptions['store_generated_asset']>>(async () => ({
            storage: { type: 'inline_base64', data: pngA },
            byte_length: new Uint8Array(Buffer.from(pngA, 'base64')).byteLength,
            content_hash: await hash(new Uint8Array(Buffer.from(pngA, 'base64'))),
        }));

        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Draw.' }], {
                ...runtime('invalid-sink'),
                store_generated_asset: store,
            }),
        ).rejects.toThrow('must return external canonical storage');
    });

    it('does not publish accepted output when durable asset storage fails', async () => {
        let prepared: Parameters<NonNullable<ExecutionOptions['on_canonical_request_prepared']>>[0] | undefined;
        const driver = new ImageDriver(vi.fn(async () => imageResponse([{ b64_json: pngA }], undefined)));
        const options: ExecutionOptions = {
            ...runtime('sink-failure'),
            on_canonical_request_prepared: async (value) => {
                prepared = value;
            },
            store_generated_asset: async () => {
                throw new Error('durable image write failed');
            },
        };

        await expect(driver.executeCanonical([{ role: PromptRole.user, content: 'Draw.' }], options)).rejects.toThrow(
            'durable image write failed',
        );
        const preparedDocument = prepared?.document;
        expect(preparedDocument).toBeDefined();
        expect(Object.keys(preparedDocument?.generations ?? {})).toHaveLength(0);
        expect(preparedDocument?.turns.every((turn) => turn.kind !== 'agent')).toBe(true);
    });

    it.each([
        ['empty', imageResponse([])],
        ['missing image value', imageResponse([{}])],
        ['malformed base64', imageResponse([{ b64_json: '***' }])],
    ] as const)('rejects %s provider image output without accepting it', async (_name, response) => {
        const driver = new ImageDriver(vi.fn(async () => response));
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Draw.' }], runtime(`malformed-${_name}`)),
        ).rejects.toThrow();
    });

    it('propagates cancellation to the provider request', async () => {
        const generate = vi.fn(
            async (_request: unknown, request?: { signal?: AbortSignal }) =>
                await new Promise<never>((_resolve, reject) => {
                    request?.signal?.addEventListener('abort', () => reject(request.signal?.reason), { once: true });
                }),
        );
        const driver = new ImageDriver(generate);
        const controller = new AbortController();
        const execution = driver.executeCanonical(
            [{ role: PromptRole.user, content: 'Draw.' }],
            runtime('cancel'),
            controller.signal,
        );
        await vi.waitFor(() => expect(generate).toHaveBeenCalledOnce());
        controller.abort(new Error('cancel image'));
        await expect(execution).rejects.toThrow('cancel image');
    });

    it('aborts URL hydration and does not accept output when the caller cancels a finite stream', async () => {
        let downloadSignal: AbortSignal | undefined;
        const imageFetch = vi.fn<typeof fetch>(async (_input, init) => {
            downloadSignal = init?.signal ?? undefined;
            const body = new ReadableStream<Uint8Array>({
                start(controller) {
                    downloadSignal?.addEventListener('abort', () => controller.error(downloadSignal?.reason), {
                        once: true,
                    });
                },
            });
            return new Response(body, { status: 200, headers: { 'content-type': 'image/png' } });
        });
        const driver = new ImageDriver(
            vi.fn(async () => imageResponse([{ url: 'https://images.openai.test/pending' }], undefined)),
            imageFetch,
        );
        const controller = new AbortController();
        const stream = await driver.streamCanonical(
            [{ role: PromptRole.user, content: 'Draw.' }],
            runtime('stream-cancel'),
            controller.signal,
        );
        const consumption = consume(stream);
        await vi.waitFor(() => expect(imageFetch).toHaveBeenCalledOnce());

        controller.abort(new Error('cancel image hydration'));
        await expect(consumption).rejects.toThrow();
        expect(downloadSignal?.aborted).toBe(true);
        expect(stream.completion).toBeUndefined();
    });

    it('rejects unsupported input before reading files or calling the provider', async () => {
        const generate = vi.fn(async () => imageResponse([{ b64_json: pngA }]));
        const getStream = vi.fn();
        const driver = new ImageDriver(generate);
        await expect(
            driver.executeCanonical(
                [
                    {
                        role: PromptRole.user,
                        content: 'Draw.',
                        files: [
                            {
                                name: 'input.png',
                                mime_type: 'image/png',
                                getStream,
                                getURL: vi.fn(),
                                getURI: vi.fn(),
                            },
                        ],
                    },
                ],
                runtime('file'),
            ),
        ).rejects.toThrow('does not support image/png input files');
        await expect(
            driver.executeCanonical([{ role: PromptRole.negative, content: 'Draw.' }], runtime('role')),
        ).rejects.toThrow('does not support negative input');
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Draw.' }], {
                ...runtime('tools'),
                tools: [{ name: 'paint', input_schema: { type: 'object' } }],
            }),
        ).rejects.toThrow('does not support tools');

        const prior = createConversationDocument({ id: 'conversation:continuation', created_at: at });
        const priorTurn = createUserTurn({
            id: 'prior-turn',
            authority: 'ordinary',
            model_visibility: 'include',
            status: 'completed',
            timestamps: { recorded_at: at },
            provenance: { type: 'received' },
            blocks: [createTextBlock({ id: 'prior-text', text: 'Prior input', format: 'plain' })],
        });
        const continued = appendConversationRecords(
            prior,
            {
                turns: [priorTurn],
                context_entries: [{ id: 'prior-context', type: 'source_turn', turn_id: priorTurn.id }],
            },
            {
                expected_revision: 0,
                operation_id: 'prior-operation',
                payload_fingerprint: 'prior-payload',
                recorded_at: at,
            },
        ).document;
        const continuationRuntime = runtime('continuation').conversation_runtime;
        if (continuationRuntime === undefined) throw new Error('Test runtime is missing conversation context');
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Draw.' }], {
                ...runtime('continuation', continued),
                conversation_runtime: {
                    ...continuationRuntime,
                    conversation_id: continued.id,
                },
            }),
        ).rejects.toThrow('does not support conversation continuation');
        expect(getStream).not.toHaveBeenCalled();
        expect(generate).not.toHaveBeenCalled();
    });
});
