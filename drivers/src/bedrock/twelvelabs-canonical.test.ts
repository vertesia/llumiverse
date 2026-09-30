import type {
    InvokeModelCommandOutput,
    InvokeModelWithResponseStreamCommandOutput,
} from '@aws-sdk/client-bedrock-runtime';
import {
    appendConversationRecords,
    type ConversationPreparedRequest,
    type ConversationStreamEvent,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    parseConversationDocument,
} from '@llumiverse/conversation';
import {
    Base64DataSource,
    type CanonicalExecutionEventStream,
    type DataSource,
    type ExecutionOptions,
    isCanonicalAcceptedRecovery,
    PromptRole,
    URLDataSource,
} from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { BedrockDriver } from './index.js';
import { TwelvelabsPegasusNativeStreamAccumulator, type TwelvelabsPegasusStreamEvent } from './twelvelabs-canonical.js';

const MODEL = 'twelvelabs.pegasus-1-2-v1:0';
const VIDEO_BASE64 = 'AQID';
const VIDEO_HASH = 'sha256:039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81';
const RESULT_SCHEMA = {
    type: 'object' as const,
    properties: { answer: { type: 'string' as const } },
    required: ['answer'],
    additionalProperties: false,
};

function runtimeOptions(flow: string, conversation?: unknown): ExecutionOptions {
    return {
        model: MODEL,
        ...(conversation === undefined ? {} : { conversation }),
        model_options: {
            _option_id: 'bedrock-twelvelabs-pegasus',
            temperature: 0.2,
            max_tokens: 128,
            service_tier: 'priority',
        },
        conversation_runtime: {
            conversation_id: `conversation:pegasus:${flow}`,
            request_id: `request:pegasus:${flow}`,
            attempt_id: `attempt:pegasus:${flow}:first`,
            input_operation_id: `input:pegasus:${flow}`,
            response_operation_id: `response:pegasus:${flow}`,
            recorded_at: '2026-10-01T00:00:00.000Z',
            started_at: '2026-10-01T00:00:00.000Z',
            completed_at: '2026-10-01T00:00:00.000Z',
        },
    };
}

function retryOptions(options: ExecutionOptions, conversation: unknown): ExecutionOptions {
    if (options.conversation_runtime === undefined) throw new Error('Missing Pegasus runtime');
    return {
        ...options,
        conversation,
        conversation_runtime: {
            ...options.conversation_runtime,
            attempt_id: `${options.conversation_runtime.attempt_id}:retry`,
        },
    };
}

function segments(video: DataSource = new Base64DataSource('clip.mp4', 'video/mp4', VIDEO_BASE64)) {
    return [
        { role: PromptRole.system, content: 'Answer only from the video.' },
        { role: PromptRole.user, content: 'What happens?', files: [video] },
    ];
}

function invokeResponse(
    body: unknown,
    requestId = 'pegasus-response-1',
    serviceTier = 'priority',
): InvokeModelCommandOutput {
    return {
        body: new TextEncoder().encode(JSON.stringify(body)),
        contentType: 'application/json',
        serviceTier,
        $metadata: { requestId },
    } as InvokeModelCommandOutput;
}

function streamEvent(body: unknown): TwelvelabsPegasusStreamEvent {
    return { chunk: { bytes: new TextEncoder().encode(JSON.stringify(body)) } };
}

function streamResponse(
    events: TwelvelabsPegasusStreamEvent[],
    onSignal?: (signal: AbortSignal | undefined) => void,
): InvokeModelWithResponseStreamCommandOutput {
    return {
        body: (async function* () {
            for (const event of events) yield event;
        })(),
        serviceTier: 'priority',
        $metadata: { requestId: 'pegasus-stream-1' },
        _onSignal: onSignal,
    } as unknown as InvokeModelWithResponseStreamCommandOutput;
}

function driverWith(input: {
    invoke?: (...args: unknown[]) => Promise<InvokeModelCommandOutput>;
    stream?: (
        request: unknown,
        options?: { abortSignal?: AbortSignal },
    ) => Promise<InvokeModelWithResponseStreamCommandOutput>;
}) {
    const driver = new BedrockDriver({
        region: 'us-east-1',
        credentials: { accessKeyId: 'test-access-key', secretAccessKey: 'test-secret-key' },
    });
    const invokeModel = vi.fn(
        input.invoke ??
            (async (..._args: unknown[]) => invokeResponse({ message: 'A bird flies.', finishReason: 'stop' })),
    );
    const invokeModelWithResponseStream = vi.fn(
        input.stream ??
            (async () =>
                streamResponse([
                    streamEvent({ delta: 'A bird ' }),
                    streamEvent({ delta: 'flies.' }),
                    streamEvent({ message: 'A bird flies.', finishReason: 'stop' }),
                ])),
    );
    Object.defineProperty(driver, 'getExecutor', {
        value: () => ({ invokeModel, invokeModelWithResponseStream, destroy: vi.fn() }),
    });
    return { driver, invokeModel, invokeModelWithResponseStream };
}

function requestBody(call: unknown): Record<string, unknown> {
    const request = call as { body?: unknown };
    if (typeof request.body !== 'string') throw new Error('Expected serialized Pegasus request body');
    return JSON.parse(request.body) as Record<string, unknown>;
}

async function collectEvents(stream: CanonicalExecutionEventStream): Promise<ConversationStreamEvent[]> {
    const events: ConversationStreamEvent[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

describe('Bedrock TwelveLabs Pegasus canonical lifecycle', () => {
    it('binds the native request, preserves video integrity, and exact-recovers without transport', async () => {
        const { driver, invokeModel, invokeModelWithResponseStream } = driverWith({
            invoke: async () => invokeResponse({ message: '{"answer":"flight"}', finishReason: 'stop' }),
        });
        let prepared = false;
        const publish = vi.fn(async () => {
            expect(invokeModel).not.toHaveBeenCalled();
            prepared = true;
        });
        const options = {
            ...runtimeOptions('sync'),
            result_schema: RESULT_SCHEMA,
            on_canonical_request_prepared: publish,
        };
        const first = await driver.executeCanonical(segments(), options);

        expect(prepared).toBe(true);
        expect(invokeModel).toHaveBeenCalledOnce();
        expect(requestBody(invokeModel.mock.calls[0]?.[0])).toEqual({
            inputPrompt: 'Answer only from the video.\nWhat happens?',
            temperature: 0.2,
            responseFormat: { jsonSchema: RESULT_SCHEMA },
            mediaSource: { base64String: VIDEO_BASE64 },
            maxOutputTokens: 128,
        });
        expect(first.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'json', value: { answer: 'flight' } }),
        ]);
        expect(first.accepted_output.generation).toMatchObject({
            protocol: 'aws.bedrock.invoke_model.twelvelabs_pegasus',
            requested_model: MODEL,
            resolved_model: MODEL,
            provider_response_id: 'pegasus-response-1',
            finish_reason: 'stop',
            status: 'completed',
        });
        expect(first.accepted_output.generation).not.toHaveProperty('usage');
        expect(Object.values(first.conversation.assets)).toEqual([
            expect.objectContaining({
                kind: 'video',
                mime_type: 'video/mp4',
                byte_length: 3,
                content_hash: VIDEO_HASH,
                provenance: expect.objectContaining({ type: 'received' }),
                storage: { type: 'inline_base64', data: VIDEO_BASE64 },
            }),
        ]);
        const fullGeneration = first.conversation.generations[first.accepted_output.generation.id];
        if (fullGeneration.request_receipt === undefined) throw new Error('Missing Pegasus request receipt');
        expect(fullGeneration.request_receipt.target).toMatchObject({
            provider: 'bedrock',
            protocol: 'aws.bedrock.invoke_model.twelvelabs_pegasus',
            model: MODEL,
            options: {
                region: 'us-east-1',
                service_tier: 'priority',
                parameters: { temperature: 0.2, maxOutputTokens: 128, responseFormat: { jsonSchema: RESULT_SCHEMA } },
                video: { segment_index: 1, mime_type: 'video/mp4', source: 'inline_base64' },
            },
        });
        expect(JSON.stringify(first.conversation)).not.toContain('test-secret-key');

        const persisted = JSON.parse(JSON.stringify(first.conversation));
        const retryPublish = vi.fn(async () => undefined);
        const retry = await driver.streamCanonicalEvents(
            segments(),
            {
                ...retryOptions(options, persisted),
                on_canonical_request_prepared: retryPublish,
            },
            undefined,
            { stream_id: 'stream:pegasus:sync:retry' },
        );
        expect(await collectEvents(retry)).toEqual([
            expect.objectContaining({ type: 'response_accepted', origin: 'accepted_recovery', sequence: 0 }),
        ]);
        expect(retry.completion?.accepted_output).toEqual(first.accepted_output);
        expect(isCanonicalAcceptedRecovery(retry.completion)).toBe(true);
        expect(invokeModel).toHaveBeenCalledOnce();
        expect(invokeModelWithResponseStream).not.toHaveBeenCalled();
        expect(retryPublish).not.toHaveBeenCalled();
        await retry.closed;
    });

    it('preserves S3 video provenance without claiming externally verified bytes', async () => {
        const { driver } = driverWith({});
        const first = await driver.executeCanonical(
            segments(new URLDataSource('clip.mp4', 'video/mp4', 's3://video-bucket/source/clip.mp4')),
            runtimeOptions('s3'),
        );
        expect(Object.values(first.conversation.assets)).toEqual([
            expect.objectContaining({
                kind: 'video',
                storage: {
                    type: 'external',
                    resolver: 'aws.s3',
                    locator: { uri: 's3://video-bucket/source/clip.mp4' },
                },
            }),
        ]);
        const asset = Object.values(first.conversation.assets)[0];
        expect(asset).not.toHaveProperty('content_hash');
        expect(asset).not.toHaveProperty('byte_length');
    });

    it('retries the exact prepared request without duplicating canonical input', async () => {
        let prepared: ConversationPreparedRequest | undefined;
        let call = 0;
        const { driver, invokeModel } = driverWith({
            invoke: async () => {
                call += 1;
                if (call === 1) throw new Error('provider failed after durable preparation');
                return invokeResponse({ message: 'Recovered execution.', finishReason: 'stop' });
            },
        });
        const options = {
            ...runtimeOptions('prepared-retry'),
            on_canonical_request_prepared: async (value: ConversationPreparedRequest) => {
                prepared = value;
            },
        };
        await expect(driver.executeCanonical(segments(), options)).rejects.toThrow('provider failed');
        if (prepared === undefined) throw new Error('Missing published Pegasus prepared request');
        const publishedDocument = JSON.parse(JSON.stringify(prepared.document));
        const retry = await driver.executeCanonical(segments(), {
            ...retryOptions(options, publishedDocument),
            on_canonical_request_prepared: async () => undefined,
        });
        expect(invokeModel).toHaveBeenCalledTimes(2);
        expect(retry.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'text', text: 'Recovered execution.' }),
        ]);
        const userTurns = retry.conversation.turns.filter((turn) => turn.kind === 'user');
        expect(userTurns).toHaveLength(1);
        expect(retry.conversation.turns.filter((turn) => turn.kind === 'program')).toHaveLength(1);
    });

    it('rejects a latest matching input receipt when the document has unrelated prefix history', async () => {
        let prepared: ConversationPreparedRequest | undefined;
        const { driver, invokeModel } = driverWith({
            invoke: async () => {
                throw new Error('provider failed after durable preparation');
            },
        });
        const options = {
            ...runtimeOptions('unrelated-prefix'),
            on_canonical_request_prepared: async (value: ConversationPreparedRequest) => {
                prepared = value;
            },
        };
        await expect(driver.executeCanonical(segments(), options)).rejects.toThrow('provider failed');
        if (prepared === undefined) throw new Error('Missing published Pegasus prepared request');
        const runtime = options.conversation_runtime;
        if (runtime === undefined) throw new Error('Missing Pegasus test runtime');
        const unrelatedTurn = createUserTurn({
            id: 'turn:unrelated-prefix',
            authority: 'ordinary',
            blocks: [createTextBlock({ id: 'block:unrelated-prefix', text: 'Unrelated history.', format: 'plain' })],
            status: 'completed',
            timestamps: { recorded_at: runtime.recorded_at },
            model_visibility: 'include',
            provenance: { type: 'received' },
        });
        const unrelatedContext = {
            id: 'context:unrelated-prefix',
            type: 'source_turn' as const,
            turn_id: unrelatedTurn.id,
        };
        const base = createConversationDocument({ id: prepared.document.id, created_at: runtime.recorded_at });
        const prior = appendConversationRecords(
            base,
            { turns: [unrelatedTurn], context_entries: [unrelatedContext] },
            {
                expected_revision: 0,
                operation_id: 'input:unrelated-prefix:prior',
                payload_fingerprint: `sha256:${'1'.repeat(64)}`,
                recorded_at: runtime.recorded_at,
            },
        ).document;
        const originalReceipt = prepared.document.operation_receipts[runtime.input_operation_id];
        if (originalReceipt === undefined) throw new Error('Missing original Pegasus input receipt');
        const withUnrelatedPrefix = appendConversationRecords(
            prior,
            {
                turns: prepared.document.turns,
                assets: Object.values(prepared.document.assets),
                context_entries: prepared.document.context.entries,
                tool_definitions: Object.values(prepared.document.tool_definitions),
                execution_receipts: Object.values(prepared.document.execution_receipts),
                active_tool_definition_ids: [],
            },
            {
                expected_revision: prior.revision,
                operation_id: runtime.input_operation_id,
                payload_fingerprint: originalReceipt.payload_fingerprint,
                recorded_at: runtime.recorded_at,
            },
        ).document;
        expect(withUnrelatedPrefix.operation_receipts[runtime.input_operation_id]?.result_revision).toBe(
            withUnrelatedPrefix.revision,
        );
        await expect(
            driver.executeCanonical(segments(), {
                ...retryOptions(options, withUnrelatedPrefix),
                on_canonical_request_prepared: async () => undefined,
            }),
        ).rejects.toThrow('does not support conversation continuation');
        expect(invokeModel).toHaveBeenCalledOnce();
    });

    it('projects one native stream into exact legacy-string and typed canonical output', async () => {
        const { driver, invokeModelWithResponseStream } = driverWith({});
        const stringStream = await driver.streamCanonical(segments(), runtimeOptions('string'));
        let preview = '';
        for await (const chunk of stringStream) preview += chunk;
        expect(preview).toBe('A bird flies.');
        expect(stringStream.completion?.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'text', text: 'A bird flies.' }),
        ]);
        expect(stringStream.completion?.accepted_output.generation).toMatchObject({
            provider_response_id: 'pegasus-stream-1',
            finish_reason: 'stop',
            status: 'completed',
        });

        const typed = await driver.streamCanonicalEvents(segments(), runtimeOptions('typed'), undefined, {
            stream_id: 'stream:pegasus:typed',
        });
        const events = await collectEvents(typed);
        expect(events.filter((event) => event.type === 'draft_text_delta')).toMatchObject([
            { text: 'A bird ' },
            { text: 'flies.' },
        ]);
        expect(events.at(-1)).toMatchObject({
            type: 'response_accepted',
            origin: 'live_transport',
            reconciliations: [{ disposition: 'direct' }],
        });
        expect(typed.completion?.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'text', text: 'A bird flies.' }),
        ]);
        expect(invokeModelWithResponseStream).toHaveBeenCalledTimes(2);
        await typed.closed;
    });

    it('normalizes streamed structured output and marks length cutoffs interrupted', async () => {
        const { driver } = driverWith({
            stream: async () =>
                streamResponse([
                    streamEvent({ delta: '{"answer":' }),
                    streamEvent({ message: '{"answer":"partial"}', finishReason: 'length' }),
                ]),
        });
        const stream = await driver.streamCanonicalEvents(
            segments(),
            { ...runtimeOptions('structured-length'), result_schema: RESULT_SCHEMA },
            undefined,
            { stream_id: 'stream:pegasus:structured-length' },
        );
        const events = await collectEvents(stream);
        expect(events).toContainEqual(expect.objectContaining({ type: 'draft_finished', outcome: 'interrupted' }));
        expect(events.at(-1)).toMatchObject({
            type: 'response_accepted',
            reconciliations: [{ disposition: 'structured_output' }],
        });
        expect(stream.completion?.accepted_output.turn).toMatchObject({
            status: 'interrupted',
            blocks: [expect.objectContaining({ type: 'json', value: { answer: 'partial' } })],
        });
        expect(stream.completion?.accepted_output.generation).toMatchObject({
            status: 'cancelled',
            finish_reason: 'length',
        });
        await stream.closed;
    });

    it('rejects unsupported input before reading the video or invoking Bedrock', async () => {
        const getURL = vi.fn(async () => 's3://bucket/video.mp4');
        const getStream = vi.fn(async () => new ReadableStream<Uint8Array>());
        const video = {
            name: 'video.mp4',
            mime_type: 'video/mp4',
            getURL,
            getURI: getURL,
            getStream,
        } satisfies DataSource;
        const { driver, invokeModel, invokeModelWithResponseStream } = driverWith({});
        await expect(
            driver.executeCanonical(
                [{ role: PromptRole.assistant, content: 'Earlier answer.', files: [video] }],
                runtimeOptions('unsupported'),
            ),
        ).rejects.toThrow('does not support assistant input');
        await expect(
            driver.executeCanonical(segments(video), {
                ...runtimeOptions('active-tools'),
                tools: [{ name: 'lookup', input_schema: { type: 'object', properties: {} } }],
            }),
        ).rejects.toThrow('does not support tools');
        expect(getURL).not.toHaveBeenCalled();
        expect(getStream).not.toHaveBeenCalled();
        expect(invokeModel).not.toHaveBeenCalled();
        expect(invokeModelWithResponseStream).not.toHaveBeenCalled();
    });

    it('bounds canonical inline video while reading and never retries a failed stream read', async () => {
        const getURL = vi.fn(async () => 'https://files.example/video.mp4');
        const cancel = vi.fn();
        const oversizedStream = new ReadableStream<Uint8Array>({
            start(controller) {
                controller.enqueue(new Uint8Array(25 * 1024 * 1024 + 1));
            },
            cancel,
        });
        const getOversizedStream = vi.fn(async () => oversizedStream);
        const oversized = {
            name: 'large.mp4',
            mime_type: 'video/mp4',
            getURL,
            getURI: getURL,
            getStream: getOversizedStream,
        } satisfies DataSource;
        const { driver, invokeModel } = driverWith({});
        await expect(driver.executeCanonical(segments(oversized), runtimeOptions('oversized'))).rejects.toThrow(
            'exceeds the 25MB limit',
        );
        expect(getOversizedStream).toHaveBeenCalledOnce();
        expect(cancel).toHaveBeenCalledOnce();
        expect(invokeModel).not.toHaveBeenCalled();

        const readFailure = new Error('video read failed');
        const getFailingStream = vi.fn(async () => {
            throw readFailure;
        });
        const failing = {
            name: 'failed.mp4',
            mime_type: 'video/mp4',
            getURL,
            getURI: getURL,
            getStream: getFailingStream,
        } satisfies DataSource;
        await expect(driver.executeCanonical(segments(failing), runtimeOptions('read-failure'))).rejects.toThrow(
            readFailure,
        );
        expect(getFailingStream).toHaveBeenCalledOnce();
        expect(invokeModel).not.toHaveBeenCalled();
    });

    it('fails malformed, provider-error, divergent, and unterminated native streams', () => {
        expect(() => new TwelvelabsPegasusNativeStreamAccumulator().accept({ validationException: {} })).toThrow(
            'validationException',
        );
        expect(() =>
            new TwelvelabsPegasusNativeStreamAccumulator().accept({
                chunk: { bytes: new TextEncoder().encode('{') },
            }),
        ).toThrow();
        const divergent = new TwelvelabsPegasusNativeStreamAccumulator();
        divergent.accept(streamEvent({ delta: 'first' }));
        expect(() => divergent.accept(streamEvent({ message: 'different', finishReason: 'stop' }))).toThrow('diverges');
        const unterminated = new TwelvelabsPegasusNativeStreamAccumulator();
        unterminated.accept(streamEvent({ delta: 'partial' }));
        expect(() => unterminated.response()).toThrow('without a terminal finish reason');
        const terminal = new TwelvelabsPegasusNativeStreamAccumulator();
        terminal.accept(streamEvent({ message: 'done', finishReason: 'stop' }));
        expect(() => terminal.accept(streamEvent({ delta: 'late' }))).toThrow('after its terminal event');
    });

    it('terminates typed delivery on a native exception without accepting the prefix', async () => {
        const { driver } = driverWith({
            stream: async () =>
                streamResponse([
                    streamEvent({ delta: 'unaccepted prefix' }),
                    { modelStreamErrorException: { message: 'upstream failed' } },
                ]),
        });
        const stream = await driver.streamCanonicalEvents(segments(), runtimeOptions('native-error'), undefined, {
            stream_id: 'stream:pegasus:native-error',
        });
        const events = await collectEvents(stream);
        expect(events.at(-1)).toMatchObject({
            type: 'stream_terminated',
            outcome: 'failed',
            diagnostic: { code: 'PROVIDER_STREAM_FAILED' },
        });
        expect(stream.completion).toBeUndefined();
        await stream.closed;
    });

    it('does not open transport when durable prepared publication fails', async () => {
        const { driver, invokeModelWithResponseStream } = driverWith({});
        await expect(
            driver.streamCanonicalEvents(
                segments(),
                {
                    ...runtimeOptions('barrier'),
                    on_canonical_request_prepared: async () => {
                        throw new Error('durability barrier rejected');
                    },
                },
                undefined,
                { stream_id: 'stream:pegasus:barrier' },
            ),
        ).rejects.toThrow('durability barrier rejected');
        expect(invokeModelWithResponseStream).not.toHaveBeenCalled();
    });

    it('publishes before opening transport and cancellation aborts a pending native stream', async () => {
        let transportSignal: AbortSignal | undefined;
        let release: (() => void) | undefined;
        const pending = new Promise<void>((resolve) => {
            release = resolve;
        });
        const { driver, invokeModelWithResponseStream } = driverWith({
            stream: async (_request, options) => {
                transportSignal = options?.abortSignal;
                return {
                    body: (async function* () {
                        try {
                            yield streamEvent({ delta: 'started' });
                            await pending;
                            yield streamEvent({ message: 'started', finishReason: 'stop' });
                        } finally {
                            release?.();
                        }
                    })(),
                    contentType: 'application/json',
                    $metadata: { requestId: 'pegasus-cancel' },
                } as unknown as InvokeModelWithResponseStreamCommandOutput;
            },
        });
        const publish = vi.fn(async () => expect(invokeModelWithResponseStream).not.toHaveBeenCalled());
        const stream = await driver.streamCanonicalEvents(
            segments(),
            { ...runtimeOptions('cancel'), on_canonical_request_prepared: publish },
            undefined,
            { stream_id: 'stream:pegasus:cancel' },
        );
        const iterator = stream[Symbol.asyncIterator]();
        expect((await iterator.next()).value).toMatchObject({ type: 'draft_started' });
        expect((await iterator.next()).value).toMatchObject({ type: 'draft_block_started' });
        expect((await iterator.next()).value).toMatchObject({ type: 'draft_text_delta', text: 'started' });
        const terminal = await stream.cancel();
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        expect(transportSignal?.aborted).toBe(true);
        release?.();
        await stream.closed;
        expect(publish).toHaveBeenCalledOnce();
    });

    it('rejects changed exact-retry target binding before transport', async () => {
        const { driver, invokeModel } = driverWith({});
        const firstOptions = runtimeOptions('binding');
        const first = await driver.executeCanonical(segments(), firstOptions);
        const persisted = parseConversationDocument(JSON.parse(JSON.stringify(first.conversation)));
        await expect(
            driver.executeCanonical(segments(), {
                ...retryOptions(firstOptions, persisted),
                model_options: { ...firstOptions.model_options, temperature: 0.7 },
            }),
        ).rejects.toThrow(/different payload|does not match|incompatible/);
        expect(invokeModel).toHaveBeenCalledOnce();

        const changedRegionDriver = new BedrockDriver({
            region: 'us-west-2',
            credentials: { accessKeyId: 'test-access-key', secretAccessKey: 'test-secret-key' },
        });
        const changedRegionInvoke = vi.fn(async () =>
            invokeResponse({ message: 'Must not execute.', finishReason: 'stop' }),
        );
        Object.defineProperty(changedRegionDriver, 'getExecutor', {
            value: () => ({
                invokeModel: changedRegionInvoke,
                invokeModelWithResponseStream: vi.fn(),
                destroy: vi.fn(),
            }),
        });
        await expect(
            changedRegionDriver.executeCanonical(
                segments(),
                retryOptions(firstOptions, JSON.parse(JSON.stringify(first.conversation))),
            ),
        ).rejects.toThrow('incompatible TwelveLabs Pegasus target options');
        expect(changedRegionInvoke).not.toHaveBeenCalled();
    });
});
