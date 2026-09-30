import {
    type ConversationDocument,
    type ConversationStreamEvent,
    parseConversationDocument,
} from '@llumiverse/conversation';
import {
    type CanonicalExecutionEventStream,
    type DataSource,
    type ExecutionOptions,
    PromptRole,
} from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { VertexAIDriver } from '../index.js';
import { GEMINI_OMNI_1_1_VIDEO_MODEL } from './omni-video.js';

const MODEL = `publishers/google/models/${GEMINI_OMNI_1_1_VIDEO_MODEL}`;
const OUTPUT_PREFIX = 'gs://project-bucket/runs/run-1/media/';

class OmniLifecycleTestDriver extends VertexAIDriver {
    readonly cleanup = vi.fn();

    protected override async destroyProviderResources(): Promise<void> {
        this.cleanup();
        await super.destroyProviderResources();
    }
}

function source(uri = 'gs://project-bucket/input/frame.png', mimeType = 'image/png'): DataSource {
    return {
        name: 'frame.png',
        mime_type: mimeType,
        getURI: vi.fn().mockResolvedValue(uri),
        getURL: vi.fn(),
        getStream: vi.fn(),
    };
}

function completedResponse(
    usage: Record<string, number> = { total_tokens: 9, total_input_tokens: 3, total_output_tokens: 6 },
) {
    return {
        id: 'interaction-1',
        status: 'completed',
        usage,
        steps: [
            {
                type: 'model_output',
                content: [
                    { type: 'text', text: 'Generated video' },
                    { type: 'video', uri: `${OUTPUT_PREFIX}result.mp4`, mime_type: 'video/mp4' },
                ],
            },
        ],
    };
}

function runtime(flow: string, attempt = 'first', conversation?: ConversationDocument): ExecutionOptions {
    return {
        model: MODEL,
        ...(conversation === undefined ? {} : { conversation }),
        output_storage_uri: OUTPUT_PREFIX,
        model_options: {
            _option_id: 'vertexai-gemini-omni-video',
            task: 'image_to_video',
            aspect_ratio: '16:9',
            duration_seconds: 5,
            resolution: '1080p',
        },
        conversation_runtime: {
            conversation_id: `conversation:omni:${flow}`,
            request_id: `request:omni:${flow}`,
            attempt_id: `attempt:omni:${flow}:${attempt}`,
            input_operation_id: `input:omni:${flow}`,
            response_operation_id: `response:omni:${flow}`,
            recorded_at: '2026-09-30T00:00:00.000Z',
            started_at: '2026-09-30T00:00:00.000Z',
            completed_at: '2026-09-30T00:00:01.000Z',
        },
    };
}

function driverWithResponse(response: unknown = completedResponse()) {
    const driver = new OmniLifecycleTestDriver({ project: 'test-project', region: 'us-central1' });
    const post = vi.fn().mockResolvedValue(response);
    const getFetchClientForRegion = vi
        .spyOn(driver, 'getFetchClientForRegion')
        .mockReturnValue({ post } as unknown as ReturnType<VertexAIDriver['getFetchClientForRegion']>);
    vi.spyOn(driver, 'getRequestTimeoutMs').mockReturnValue(900_000);
    return { driver, getFetchClientForRegion, post };
}

async function collect(stream: CanonicalExecutionEventStream): Promise<ConversationStreamEvent[]> {
    const events: ConversationStreamEvent[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

describe('Gemini Omni canonical video lifecycle', () => {
    it('executes through the public canonical seam and exact-retries without another Interaction', async () => {
        const { driver, post, getFetchClientForRegion } = driverWithResponse();
        const frame = source();
        const options = runtime('sync');
        let publishCount = 0;
        const first = await driver.executeCanonical(
            [
                { role: PromptRole.system, content: 'Use a cinematic style.' },
                { role: PromptRole.user, content: 'Animate this frame.', files: [frame] },
            ],
            {
                ...options,
                on_canonical_request_prepared: async () => {
                    publishCount += 1;
                    expect(post).not.toHaveBeenCalled();
                },
            },
        );

        expect(await driver.supportsCanonicalExecution(options)).toBe(true);
        expect(publishCount).toBe(1);
        expect(getFetchClientForRegion).toHaveBeenCalledWith('global', 'v1beta1');
        expect(post).toHaveBeenCalledOnce();
        expect(post).toHaveBeenCalledWith('interactions', {
            payload: {
                model: GEMINI_OMNI_1_1_VIDEO_MODEL,
                input: [
                    { type: 'text', text: 'Use a cinematic style.\nAnimate this frame.' },
                    { type: 'image', uri: 'gs://project-bucket/input/frame.png', mime_type: 'image/png' },
                ],
                response_format: [
                    {
                        type: 'video',
                        delivery: 'uri',
                        gcs_uri: OUTPUT_PREFIX,
                        duration: '5s',
                        aspect_ratio: '16:9',
                        resolution: '1080p',
                    },
                ],
                generation_config: { video_config: { task: 'image_to_video' } },
            },
            signal: undefined,
            timeoutMs: 900_000,
        });
        expect(first.accepted_output.turn.blocks.map((block) => block.type)).toEqual(['text', 'video']);
        expect(first.accepted_output.generation).toMatchObject({
            provider: 'vertexai',
            protocol: 'google.vertex.interactions.video',
            requested_model: MODEL,
            resolved_model: GEMINI_OMNI_1_1_VIDEO_MODEL,
            provider_response_id: 'interaction-1',
            usage: { input_tokens: 3, output_tokens: 6, total_tokens: 9 },
        });
        const document = parseConversationDocument(first.conversation);
        expect(document.turns.slice(0, 2).map((turn) => [turn.kind, turn.authority])).toEqual([
            ['program', 'system'],
            ['user', 'ordinary'],
        ]);
        const assets = Object.values(document.assets);
        expect(assets).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    kind: 'image',
                    mime_type: 'image/png',
                    storage: {
                        type: 'external',
                        resolver: 'google_uri',
                        locator: { uri: 'gs://project-bucket/input/frame.png' },
                    },
                }),
                expect.objectContaining({
                    kind: 'video',
                    mime_type: 'video/mp4',
                    storage: {
                        type: 'external',
                        resolver: 'google_uri',
                        locator: { uri: `${OUTPUT_PREFIX}result.mp4` },
                    },
                    media: { container: 'mp4' },
                }),
            ]),
        );
        for (const asset of assets) {
            expect(asset).not.toHaveProperty('content_hash');
            expect(asset).not.toHaveProperty('byte_length');
        }
        const retainedGeneration = document.generations[first.accepted_output.generation.id];
        const requestReceipt = retainedGeneration.request_receipt;
        expect(requestReceipt).toBeDefined();
        if (!requestReceipt) throw new Error('Expected a retained request receipt');
        expect(requestReceipt.target.options).toMatchObject({
            region: 'global',
            api_version: 'v1beta1',
            task: 'image_to_video',
            response_format: { gcs_uri: OUTPUT_PREFIX, delivery: 'uri' },
            input_media: [{ index: 0, type: 'image', mime_type: 'image/png' }],
        });
        expect(JSON.stringify(requestReceipt.target.options)).not.toContain('Animate this frame');

        const persisted = JSON.parse(JSON.stringify(first.conversation)) as ConversationDocument;
        const retry = await driver.executeCanonical(
            [
                { role: PromptRole.system, content: 'Use a cinematic style.' },
                { role: PromptRole.user, content: 'Animate this frame.', files: [source()] },
            ],
            {
                ...runtime('sync', 'retry', persisted),
                on_canonical_request_prepared: async () => {
                    publishCount += 1;
                },
            },
        );
        expect(retry.accepted_output).toEqual(first.accepted_output);
        expect(post).toHaveBeenCalledOnce();
        expect(publishCount).toBe(1);
    });

    it('preserves legacy text/video projection while canonical output remains authoritative', async () => {
        const legacy = driverWithResponse();
        const canonical = driverWithResponse();
        const segments = [{ role: PromptRole.user, content: 'Animate.', files: [source()] }];

        const legacyResult = await legacy.driver.execute(segments, runtime('legacy'));
        const canonicalResult = await canonical.driver.executeCanonical(segments, runtime('canonical'));

        expect(legacyResult.result).toEqual([
            { type: 'text', value: 'Generated video' },
            { type: 'video', value: `${OUTPUT_PREFIX}result.mp4` },
        ]);
        expect(canonicalResult.accepted_output.turn.blocks.map((block) => block.type)).toEqual(['text', 'video']);
    });

    it('preserves reported cache and thought usage without inventing missing input buckets', async () => {
        const complete = driverWithResponse(
            completedResponse({
                total_input_tokens: 10,
                total_cached_tokens: 4,
                total_output_tokens: 8,
                total_thought_tokens: 3,
                total_tool_use_tokens: 2,
                total_tokens: 18,
            }),
        );
        const completeResult = await complete.driver.executeCanonical(
            [{ role: PromptRole.user, content: 'Animate.', files: [source()] }],
            runtime('usage-complete'),
        );
        expect(completeResult.accepted_output.generation.usage).toMatchObject({
            input_tokens: 10,
            input_new_tokens: 6,
            cache_read_tokens: 4,
            output_tokens: 8,
            reasoning_tokens: 3,
            total_tokens: 18,
            input_partition: { type: 'complete_disjoint', cache_write_bucket: 'inapplicable' },
            accounting_provenance: {
                cache_read_tokens: { method: 'reported' },
                input_new_tokens: { method: 'derived' },
                reasoning_tokens: { method: 'reported' },
            },
        });
        const completeDocument = parseConversationDocument(completeResult.conversation);
        expect(
            completeDocument.generations[completeResult.accepted_output.generation.id].usage?.reported_usage,
        ).toEqual([
            expect.objectContaining({
                payload: expect.objectContaining({ total_tool_use_tokens: 2 }),
            }),
        ]);

        const partial = driverWithResponse(
            completedResponse({ total_cached_tokens: 4, total_output_tokens: 8, total_tool_use_tokens: 2 }),
        );
        const partialResult = await partial.driver.executeCanonical(
            [{ role: PromptRole.user, content: 'Animate.', files: [source()] }],
            runtime('usage-partial'),
        );
        expect(partialResult.accepted_output.generation.usage).toMatchObject({
            cache_read_tokens: 4,
            output_tokens: 8,
            accounting_provenance: {
                cache_read_tokens: { method: 'reported' },
            },
        });
        const partialDocument = parseConversationDocument(partialResult.conversation);
        expect(partialDocument.generations[partialResult.accepted_output.generation.id].usage?.reported_usage).toEqual([
            expect.objectContaining({
                payload: { total_cached_tokens: 4, total_output_tokens: 8, total_tool_use_tokens: 2 },
            }),
        ]);
        expect(partialResult.accepted_output.generation.usage).not.toHaveProperty('input_tokens');
        expect(partialResult.accepted_output.generation.usage).not.toHaveProperty('input_new_tokens');
        expect(partialResult.accepted_output.generation.usage).not.toHaveProperty('input_partition');
        expect(partialResult.accepted_output.generation.usage).not.toHaveProperty('total_tokens');
    });

    it('rejects changed source authority or effective options before accepted recovery', async () => {
        const { driver, post } = driverWithResponse();
        const first = await driver.executeCanonical(
            [{ role: PromptRole.user, content: 'Same native text.', files: [source()] }],
            runtime('binding'),
        );
        const persisted = JSON.parse(JSON.stringify(first.conversation)) as ConversationDocument;

        await expect(
            driver.executeCanonical(
                [{ role: PromptRole.assistant, content: 'Same native text.', files: [source()] }],
                runtime('binding', 'changed-role', persisted),
            ),
        ).rejects.toThrow(/operation|revision|context|different|conflict|incompatible/i);
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Same native text.', files: [source()] }], {
                ...runtime('binding', 'changed-options', persisted),
                model_options: {
                    ...runtime('binding').model_options,
                    duration_seconds: 10,
                },
            }),
        ).rejects.toThrow(/incompatible|fingerprint|request/i);
        expect(post).toHaveBeenCalledOnce();
    });

    it('publishes before transport and rejects unsupported retained tools before media I/O', async () => {
        const barrier = driverWithResponse();
        await expect(
            barrier.driver.executeCanonical([{ role: PromptRole.user, content: 'Do not send.', files: [source()] }], {
                ...runtime('barrier'),
                on_canonical_request_prepared: async () => {
                    throw new Error('durability barrier failed');
                },
            }),
        ).rejects.toThrow('durability barrier failed');
        expect(barrier.post).not.toHaveBeenCalled();

        const activeTools = driverWithResponse();
        const media = source();
        const document = parseConversationDocument(
            (
                await driverWithResponse().driver.executeCanonical(
                    [{ role: PromptRole.user, content: 'Seed.', files: [source()] }],
                    runtime('tool-seed'),
                )
            ).conversation,
        );
        document.context.active_tool_definition_ids = ['tool-definition'];
        document.tool_definitions['tool-definition'] = {
            id: 'tool-definition',
            name: 'lookup',
            version: '1',
            input_schema: { type: 'object' },
        };
        await expect(
            activeTools.driver.executeCanonical([{ role: PromptRole.user, content: 'No.', files: [media] }], {
                ...runtime('active-tools'),
                conversation: document,
            }),
        ).rejects.toThrow('active canonical tools');
        expect(media.getURI).not.toHaveBeenCalled();
        expect(activeTools.post).not.toHaveBeenCalled();
    });

    it('streams finite acceptance and exact recovered acceptance without another Interaction', async () => {
        const { driver, post } = driverWithResponse();
        const segments = [{ role: PromptRole.user, content: 'Animate.', files: [source()] }];
        const first = await driver.streamCanonicalEvents(segments, runtime('typed'), undefined, {
            stream_id: 'stream:omni:typed:first',
        });
        expect(await collect(first)).toEqual([
            expect.objectContaining({ type: 'response_accepted', sequence: 0, origin: 'live_transport' }),
        ]);
        if (first.completion === undefined) throw new Error('Expected accepted Gemini Omni response');
        const persisted = JSON.parse(JSON.stringify(first.completion.conversation)) as ConversationDocument;
        const recovered = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Animate.', files: [source()] }],
            runtime('typed', 'retry', persisted),
            undefined,
            { stream_id: 'stream:omni:typed:retry' },
        );
        expect(await collect(recovered)).toEqual([
            expect.objectContaining({ type: 'response_accepted', sequence: 0, origin: 'accepted_recovery' }),
        ]);
        expect(recovered.completion?.accepted_output).toEqual(first.completion.accepted_output);
        expect(post).toHaveBeenCalledOnce();
    });

    it('settles finite cancellation while retaining the driver lease until transport cleanup', async () => {
        const driver = new OmniLifecycleTestDriver({ project: 'test-project', region: 'us-central1' });
        let rejectTransport: ((reason: unknown) => void) | undefined;
        const post = vi.fn(
            () =>
                new Promise((_resolve, reject) => {
                    rejectTransport = reject;
                }),
        );
        vi.spyOn(driver, 'getFetchClientForRegion').mockReturnValue({ post } as unknown as ReturnType<
            VertexAIDriver['getFetchClientForRegion']
        >);
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Wait.', files: [source()] }],
            runtime('cancel'),
            undefined,
            { stream_id: 'stream:omni:cancel' },
        );
        const pending = stream[Symbol.asyncIterator]().next();
        await vi.waitFor(() => expect(post).toHaveBeenCalledOnce());
        const terminal = await stream.cancel();
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        await expect(pending).resolves.toMatchObject({ value: terminal });
        driver.destroy();
        let closed = false;
        void stream.closed.then(() => {
            closed = true;
        });
        await Promise.resolve();
        expect(closed).toBe(false);
        expect(driver.cleanup).not.toHaveBeenCalled();
        rejectTransport?.(new DOMException('transport cleanup complete', 'AbortError'));
        await stream.closed;
        expect(stream.completion).toBeUndefined();
        await vi.waitFor(() => expect(driver.cleanup).toHaveBeenCalledOnce());
    });

    it.each([
        [{ status: 'completed', id: 'empty', steps: [] }, /without a video/],
        [
            {
                status: 'completed',
                id: 'foreign',
                steps: [
                    {
                        type: 'model_output',
                        content: [{ type: 'video', uri: 'gs://foreign-bucket/result.mp4', mime_type: 'video/mp4' }],
                    },
                ],
            },
            /outside the requested output prefix/,
        ],
    ])('does not accept malformed provider video output %#', async (response, expected) => {
        const { driver } = driverWithResponse(response);
        await expect(
            driver.executeCanonical(
                [{ role: PromptRole.user, content: 'Invalid.', files: [source()] }],
                runtime(`invalid-${String(response.id)}`),
            ),
        ).rejects.toThrow(expected);
    });
});
