import {
    appendConversationRecords,
    createConversationDocument,
    parseConversationDocument,
} from '@llumiverse/conversation';
import {
    Base64DataSource,
    type CanonicalExecutionEventStream,
    type ExecutionOptions,
    PromptRole,
    type XAIGrokImageOptions,
} from '@llumiverse/core';
import type { FetchClient } from '@vertesia/api-fetch-client';
import { describe, expect, it, vi } from 'vitest';
import type { XAIImageResponse } from './image-canonical.js';
import { xAIDriver } from './index.js';

const MODEL = 'grok-imagine-image-2.0';
const PNG_BASE64 = 'iVBORw0KGgo=';
const PNG_HASH = 'sha256:4c4b6a3be1314ab86138bef4314dde022e600960d8689a2c8f8631802d20dab6';
const IMAGE_OUTPUT_MODALITY = 'image' as NonNullable<ExecutionOptions['output_modality']>;

type Post = (path: string, options: unknown) => Promise<XAIImageResponse>;

class XAIImageTestDriver extends xAIDriver {
    imageFetch: typeof fetch | undefined;
    readonly cleanup = vi.fn();

    protected override getDriverFetch(): typeof fetch {
        return this.imageFetch ?? fetch;
    }

    protected override async destroyProviderResources(): Promise<void> {
        this.cleanup();
        await super.destroyProviderResources();
    }
}

function testDriver(
    implementation: Post = async () => ({ data: [{ b64_json: PNG_BASE64, mime_type: 'image/png' }] }),
    endpoint = 'https://xai.test/v1',
) {
    const driver = new XAIImageTestDriver({ apiKey: 'test-key', endpoint });
    const post = vi.fn(implementation);
    driver.xai_service = { post } as unknown as FetchClient;
    return { driver, post };
}

function runtimeOptions(flow: string, conversation?: unknown): ExecutionOptions {
    return {
        model: MODEL,
        ...(conversation === undefined ? {} : { conversation }),
        output_modality: IMAGE_OUTPUT_MODALITY,
        model_options: {
            _option_id: 'xai-grok-image',
            aspect_ratio: '16:9',
            resolution: '2k',
            quality: 'medium',
            response_format: 'b64_json',
            n: 2,
        },
        conversation_runtime: {
            conversation_id: `conversation:xai-image:${flow}`,
            request_id: `request:xai-image:${flow}`,
            attempt_id: `attempt:xai-image:${flow}:first`,
            input_operation_id: `input:xai-image:${flow}`,
            response_operation_id: `response:xai-image:${flow}`,
            recorded_at: '2026-09-30T00:00:00.000Z',
            started_at: '2026-09-30T00:00:00.000Z',
            completed_at: '2026-09-30T00:00:01.000Z',
        },
    };
}

function retryOptions(options: ExecutionOptions, conversation: unknown, attemptId: string): ExecutionOptions {
    if (options.conversation_runtime === undefined) throw new Error('Missing xAI image runtime');
    return {
        ...options,
        conversation,
        conversation_runtime: { ...options.conversation_runtime, attempt_id: attemptId },
    };
}

async function collect(stream: CanonicalExecutionEventStream) {
    const events = [];
    for await (const event of stream) events.push(event);
    return events;
}

describe('xAI canonical image lifecycle', () => {
    it('stages verified images, records reported cost, and exact-retries without transport', async () => {
        const { driver, post } = testDriver(async () => ({
            created: 17,
            data: [
                { b64_json: PNG_BASE64, mime_type: 'image/png', revised_prompt: 'Revised first.' },
                { b64_json: PNG_BASE64, mime_type: 'image/png' },
            ],
            usage: { cost_in_usd_ticks: 400_000_000 },
        }));
        let published = false;
        const publish = vi.fn(async () => {
            expect(post).not.toHaveBeenCalled();
            published = true;
        });
        let stored = 0;
        const store = vi.fn<NonNullable<ExecutionOptions['store_generated_asset']>>(async (stream, metadata) => {
            expect(published).toBe(true);
            expect(new Uint8Array(await new Response(stream).arrayBuffer())).toEqual(
                new Uint8Array(Buffer.from(PNG_BASE64, 'base64')),
            );
            expect(metadata).toEqual({ kind: 'image', mime_type: 'image/png' });
            stored += 1;
            return {
                storage: {
                    type: 'external' as const,
                    resolver: 'url',
                    locator: { url: `s3://generated/xai-${stored}.png` },
                },
                byte_length: 8,
                content_hash: PNG_HASH,
            };
        });
        const options = {
            ...runtimeOptions('accepted'),
            on_canonical_request_prepared: publish,
            store_generated_asset: store,
        };
        const first = await driver.executeCanonical(
            [{ role: PromptRole.user, content: 'A lighthouse in a storm.' }],
            options,
        );

        expect(publish).toHaveBeenCalledOnce();
        expect(post).toHaveBeenCalledOnce();
        expect(post).toHaveBeenCalledWith('/images/generations', {
            payload: {
                model: MODEL,
                prompt: 'A lighthouse in a storm.',
                aspect_ratio: '16:9',
                resolution: '2k',
                quality: 'medium',
                response_format: 'b64_json',
                n: 2,
            },
        });
        expect(first.accepted_output.generation).toMatchObject({
            protocol: 'xai.images',
            requested_model: MODEL,
            resolved_model: MODEL,
            status: 'completed',
            usage: { cost: { amount: '0.04', currency: 'USD', provenance: 'reported' } },
        });
        expect(first.accepted_output.generation).not.toHaveProperty('request_receipt');
        expect(first.accepted_output.turn.blocks).toHaveLength(2);
        expect(first.accepted_output.turn.blocks[0]).toMatchObject({ type: 'image', caption: 'Revised first.' });
        expect(Object.values(first.accepted_output.assets)).toEqual([
            expect.objectContaining({
                content_hash: PNG_HASH,
                storage: { type: 'external', resolver: 'url', locator: { url: 's3://generated/xai-1.png' } },
            }),
            expect.objectContaining({
                content_hash: PNG_HASH,
                storage: { type: 'external', resolver: 'url', locator: { url: 's3://generated/xai-2.png' } },
            }),
        ]);
        const generation = first.conversation.generations[first.accepted_output.generation.id];
        if (generation.request_receipt === undefined) throw new Error('Missing xAI image request receipt');
        expect(generation.request_receipt.target).toMatchObject({
            provider: 'xai',
            protocol: 'xai.images',
            model: MODEL,
            options: {
                endpoint: 'https://xai.test/v1',
                route: '/images/generations',
                parameters: {
                    aspect_ratio: '16:9',
                    resolution: '2k',
                    quality: 'medium',
                    response_format: 'b64_json',
                    n: 2,
                },
                input_image_count: 0,
            },
        });

        const persisted = JSON.parse(JSON.stringify(first.conversation));
        let retryPublishCount = 0;
        const retry = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'A lighthouse in a storm.' }],
            {
                ...retryOptions(options, persisted, 'attempt:xai-image:accepted:retry'),
                on_canonical_request_prepared: async () => {
                    retryPublishCount += 1;
                },
            },
            undefined,
            { stream_id: 'stream:xai-image:accepted:retry' },
        );
        expect(await collect(retry)).toEqual([
            expect.objectContaining({ type: 'response_accepted', sequence: 0, origin: 'accepted_recovery' }),
        ]);
        expect(retry.completion?.accepted_output).toEqual(first.accepted_output);
        expect(retryPublishCount).toBe(0);
        expect(post).toHaveBeenCalledOnce();
        expect(store).toHaveBeenCalledTimes(2);
        await retry.closed;
    });

    it('preserves privileged and assistant prompt authority plus text attachments across public execution', async () => {
        const prompt = [
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
        const canonical = testDriver();
        const response = await canonical.driver.executeCanonical(prompt, runtimeOptions('authority'));

        expect(canonical.post).toHaveBeenCalledWith(
            '/images/generations',
            expect.objectContaining({ payload: expect.objectContaining({ prompt: expectedPrompt }) }),
        );
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

        const legacy = testDriver();
        const completion = await legacy.driver.execute(prompt, {
            model: MODEL,
            output_modality: IMAGE_OUTPUT_MODALITY,
            model_options: { _option_id: 'xai-grok-image', response_format: 'b64_json', n: 1 },
        });
        expect(completion.result).toEqual([{ type: 'image', value: `data:image/png;base64,${PNG_BASE64}` }]);
        expect(legacy.post).toHaveBeenCalledWith(
            '/images/generations',
            expect.objectContaining({ payload: expect.objectContaining({ prompt: expectedPrompt }) }),
        );
    });

    it('hydrates temporary URL output before canonical acceptance', async () => {
        const { driver } = testDriver(async () => ({
            data: [
                {
                    url: 'https://images.xai.test/temporary.jpeg',
                    mime_type: 'image/png',
                    revised_prompt: 'Hydrated prompt.',
                },
            ],
        }));
        const fetchImage = vi.fn<typeof fetch>(
            async () =>
                new Response(Buffer.from(PNG_BASE64, 'base64'), {
                    status: 200,
                    headers: { 'content-type': 'image/png', 'content-length': '8' },
                }),
        );
        driver.imageFetch = fetchImage;
        const options = runtimeOptions('url');
        const modelOptions = options.model_options as XAIGrokImageOptions | undefined;
        if (modelOptions === undefined) throw new Error('Missing xAI image model options');
        modelOptions.response_format = 'url';
        const response = await driver.executeCanonical([{ role: PromptRole.user, content: 'Hydrate this.' }], options);

        expect(fetchImage).toHaveBeenCalledWith(new URL('https://images.xai.test/temporary.jpeg'), {
            signal: undefined,
        });
        const asset = Object.values(response.accepted_output.assets)[0];
        expect(asset).toMatchObject({
            mime_type: 'image/png',
            byte_length: 8,
            content_hash: PNG_HASH,
            storage: { type: 'inline_base64', data: PNG_BASE64 },
        });
        expect(response.conversation.assets[asset.id]).toMatchObject({
            metadata: { xai_image: { source_url: 'https://images.xai.test/temporary.jpeg' } },
        });
    });

    it('preserves up to five verified edit inputs as canonical received assets', async () => {
        const { driver, post } = testDriver();
        const files = [
            new Base64DataSource('first.png', 'image/png', PNG_BASE64),
            new Base64DataSource('second.png', 'image/png', PNG_BASE64),
        ];
        const options = runtimeOptions('edit');
        const response = await driver.executeCanonical(
            [{ role: PromptRole.user, content: 'Combine these.', files }],
            options,
        );

        expect(post).toHaveBeenCalledWith(
            '/images/edits',
            expect.objectContaining({
                payload: expect.objectContaining({
                    images: [
                        { type: 'image_url', url: `data:image/png;base64,${PNG_BASE64}` },
                        { type: 'image_url', url: `data:image/png;base64,${PNG_BASE64}` },
                    ],
                }),
            }),
        );
        const document = parseConversationDocument(response.conversation);
        expect(Object.values(document.assets).filter((asset) => asset.provenance.type === 'received')).toEqual([
            expect.objectContaining({ byte_length: 8, content_hash: PNG_HASH }),
            expect.objectContaining({ byte_length: 8, content_hash: PNG_HASH }),
        ]);
        const generation = document.generations[response.accepted_output.generation.id];
        if (generation.request_receipt === undefined) throw new Error('Missing xAI image request receipt');
        expect(generation.request_receipt.target.options).toMatchObject({ input_image_count: 2 });
        expect(JSON.stringify(generation.request_receipt.target.options)).not.toContain(PNG_BASE64);

        const retry = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Combine these.', files }],
            retryOptions(options, JSON.parse(JSON.stringify(response.conversation)), 'attempt:xai-image:edit:retry'),
            undefined,
            { stream_id: 'stream:xai-image:edit:retry' },
        );
        expect(await collect(retry)).toEqual([
            expect.objectContaining({ type: 'response_accepted', origin: 'accepted_recovery' }),
        ]);
        expect(post).toHaveBeenCalledOnce();
        await retry.closed;
    });

    it('publishes before transport and rejects a failed durability barrier', async () => {
        const { driver, post } = testDriver();
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Do not send.' }], {
                ...runtimeOptions('barrier'),
                on_canonical_request_prepared: async () => {
                    throw new Error('durability barrier failed');
                },
            }),
        ).rejects.toThrow('durability barrier failed');
        expect(post).not.toHaveBeenCalled();
    });

    it.each([
        ['empty output', { data: [] }, /contains no images/],
        ['malformed base64', { data: [{ b64_json: 'not-base64' }] }, /malformed base64/],
        ['unsupported bytes', { data: [{ b64_json: 'AQID' }] }, /unsupported format/],
        ['missing image value', { data: [{}] }, /neither base64 data nor a URL/],
    ])('fails closed for %s', async (label, nativeResponse, expected) => {
        const { driver } = testDriver(async () => nativeResponse);
        await expect(
            driver.executeCanonical(
                [{ role: PromptRole.user, content: 'Invalid.' }],
                runtimeOptions(`invalid:${label}`),
            ),
        ).rejects.toThrow(expected);
    });

    it('rejects unsupported inputs and retained tools before reading files or transport', async () => {
        const { driver, post } = testDriver();
        const unsupported = new Base64DataSource('document.pdf', 'application/pdf', 'AQID');
        const getUnsupportedStream = vi.spyOn(unsupported, 'getStream');
        await expect(
            driver.executeCanonical(
                [{ role: PromptRole.user, content: 'Unsupported.', files: [unsupported] }],
                runtimeOptions('unsupported-mime'),
            ),
        ).rejects.toThrow(/does not support application\/pdf input files/);
        expect(getUnsupportedStream).not.toHaveBeenCalled();
        const tooMany = Array.from(
            { length: 6 },
            (_, index) => new Base64DataSource(`source-${index}.png`, 'image/png', PNG_BASE64),
        );
        const getTooManyStreams = tooMany.map((file) => vi.spyOn(file, 'getStream'));
        await expect(
            driver.executeCanonical(
                [{ role: PromptRole.user, content: 'Too many.', files: tooMany }],
                runtimeOptions('too-many-inputs'),
            ),
        ).rejects.toThrow(/at most five input images/);
        expect(getTooManyStreams.every((getStream) => getStream.mock.calls.length === 0)).toBe(true);
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Tool.' }], {
                ...runtimeOptions('unsupported-tool'),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            }),
        ).rejects.toThrow(/does not support tools/);

        const recordedAt = '2026-09-30T00:00:00.000Z';
        const initial = createConversationDocument({
            id: 'conversation:xai-image:active-tool',
            created_at: recordedAt,
        });
        const document = appendConversationRecords(
            initial,
            {
                tool_definitions: [
                    { id: 'tool-definition:lookup', name: 'lookup', version: '1', input_schema: { type: 'object' } },
                ],
                active_tool_definition_ids: ['tool-definition:lookup'],
            },
            {
                expected_revision: initial.revision,
                operation_id: 'operation:xai-image:active-tool',
                payload_fingerprint: 'sha256:xai-image-active-tool',
                recorded_at: recordedAt,
            },
        ).document;
        const image = new Base64DataSource('source.png', 'image/png', PNG_BASE64);
        const getImageStream = vi.spyOn(image, 'getStream');
        const options = runtimeOptions('active-tool', document);
        if (options.conversation_runtime === undefined) throw new Error('Missing xAI image runtime');
        options.conversation_runtime = { ...options.conversation_runtime, conversation_id: document.id };
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Edit.', files: [image] }], options),
        ).rejects.toThrow(/does not support active canonical tool definitions/);
        expect(getImageStream).not.toHaveBeenCalled();
        expect(post).not.toHaveBeenCalled();
    });

    it('rejects changed retry input, options, endpoint, and continuation before another transport', async () => {
        const firstDriver = testDriver();
        const firstOptions = runtimeOptions('routing');
        const first = await firstDriver.driver.executeCanonical(
            [{ role: PromptRole.user, content: 'Route exactly.' }],
            firstOptions,
        );
        const persisted = JSON.parse(JSON.stringify(first.conversation));
        await expect(
            firstDriver.driver.executeCanonical(
                [{ role: PromptRole.user, content: 'Changed input.' }],
                retryOptions(firstOptions, persisted, 'attempt:xai-image:routing:changed-input'),
            ),
        ).rejects.toThrow();

        const changedOptions = retryOptions(firstOptions, persisted, 'attempt:xai-image:routing:changed-options');
        changedOptions.model_options = {
            ...(firstOptions.model_options as XAIGrokImageOptions),
            quality: 'low',
        };
        await expect(
            firstDriver.driver.executeCanonical([{ role: PromptRole.user, content: 'Route exactly.' }], changedOptions),
        ).rejects.toThrow();

        const routed = testDriver(undefined, 'https://other.xai.test/v1');
        await expect(
            routed.driver.executeCanonical(
                [{ role: PromptRole.user, content: 'Route exactly.' }],
                retryOptions(firstOptions, persisted, 'attempt:xai-image:routing:retry'),
            ),
        ).rejects.toThrow(/incompatible xAI target options/);
        expect(routed.post).not.toHaveBeenCalled();

        const continuation = runtimeOptions('continuation', first.conversation);
        if (continuation.conversation_runtime === undefined) throw new Error('Missing xAI image runtime');
        continuation.conversation_runtime = {
            ...continuation.conversation_runtime,
            conversation_id: first.conversation.id,
        };
        await expect(
            firstDriver.driver.executeCanonical([{ role: PromptRole.user, content: 'Continue.' }], continuation),
        ).rejects.toThrow(/does not support conversation continuation/);
        expect(firstDriver.post).toHaveBeenCalledOnce();
    });

    it('settles cancellation while retaining driver ownership until transport cleanup', async () => {
        let rejectInvocation: ((reason: unknown) => void) | undefined;
        const { driver, post } = testDriver(
            () =>
                new Promise<XAIImageResponse>((_resolve, reject) => {
                    rejectInvocation = reject;
                }),
        );
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Wait.' }],
            runtimeOptions('cancel'),
            undefined,
            { stream_id: 'stream:xai-image:cancel' },
        );
        const pending = stream[Symbol.asyncIterator]().next();
        await vi.waitFor(() => expect(post).toHaveBeenCalledOnce());
        const terminal = await stream.cancel();
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        await expect(pending).resolves.toMatchObject({ value: terminal });
        const request = post.mock.calls[0]?.[1] as { signal?: AbortSignal } | undefined;
        expect(request?.signal?.aborted).toBe(true);
        driver.destroy();
        expect(driver.cleanup).not.toHaveBeenCalled();
        rejectInvocation?.(new DOMException('provider cleanup completed', 'AbortError'));
        await stream.closed;
        await vi.waitFor(() => expect(driver.cleanup).toHaveBeenCalledOnce());
        expect(stream.completion).toBeUndefined();
    });
});
