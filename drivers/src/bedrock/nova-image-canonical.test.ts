import type { InvokeModelCommandOutput } from '@aws-sdk/client-bedrock-runtime';
import {
    appendConversationRecords,
    type ConversationDocument,
    createConversationDocument,
    parseConversationDocument,
} from '@llumiverse/conversation';
import {
    Base64DataSource,
    type CanonicalExecutionEventStream,
    type CanonicalExecutionInputOptions,
    type ExecutionOptions,
    isCanonicalAcceptedRecovery,
    type NovaCanvasOptions,
    PromptRole,
} from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { BedrockDriver } from './index.js';
import { NovaImageGenerationTaskType } from './nova-image-payload.js';

const MODEL = 'amazon.nova-canvas-v1:0';
const PNG_BASE64 = 'iVBORw0KGgo=';
const PNG_HASH = 'sha256:4c4b6a3be1314ab86138bef4314dde022e600960d8689a2c8f8631802d20dab6';
const IMAGE_OUTPUT_MODALITY = 'image' as NonNullable<ExecutionOptions['output_modality']>;

type Invoke = (...args: unknown[]) => Promise<InvokeModelCommandOutput>;

class NovaLifecycleTestDriver extends BedrockDriver {
    readonly cleanup = vi.fn();

    protected override async destroyProviderResources(): Promise<void> {
        this.cleanup();
    }
}

function response(body: unknown, requestId = 'bedrock-response-1'): InvokeModelCommandOutput {
    return {
        body: new TextEncoder().encode(JSON.stringify(body)),
        $metadata: { requestId },
        contentType: 'application/json',
    } as unknown as InvokeModelCommandOutput;
}

function novaDriver(implementation: Invoke = async () => response({ images: [PNG_BASE64] }), region = 'us-east-1') {
    const driver = new NovaLifecycleTestDriver({
        region,
        credentials: { accessKeyId: 'test-access-key', secretAccessKey: 'test-secret-key' },
    });
    const invokeModel = vi.fn(implementation);
    Object.defineProperty(driver, 'getExecutor', {
        value: () => ({ invokeModel, destroy: vi.fn() }),
    });
    return { driver, invokeModel };
}

function runtimeOptions(
    flow: string,
    modelOptions: Partial<NovaCanvasOptions> = {},
    conversation?: ConversationDocument,
): CanonicalExecutionInputOptions {
    return {
        model: MODEL,
        ...(conversation === undefined ? {} : { conversation }),
        output_modality: IMAGE_OUTPUT_MODALITY,
        model_options: {
            _option_id: 'bedrock-nova-canvas',
            taskType: NovaImageGenerationTaskType.TEXT_IMAGE,
            width: 512,
            height: 512,
            ...modelOptions,
        },
        conversation_runtime: {
            conversation_id: `conversation:nova-canvas:${flow}`,
            request_id: `request:nova-canvas:${flow}`,
            attempt_id: `attempt:nova-canvas:${flow}:first`,
            input_operation_id: `input:nova-canvas:${flow}`,
            response_operation_id: `response:nova-canvas:${flow}`,
            recorded_at: '2026-09-30T00:00:00.000Z',
            started_at: '2026-09-30T00:00:00.000Z',
            completed_at: '2026-09-30T00:00:00.000Z',
        },
    };
}

function retryOptions(
    options: CanonicalExecutionInputOptions,
    conversation: ConversationDocument,
    attempt: string,
): CanonicalExecutionInputOptions {
    if (options.conversation_runtime === undefined) throw new Error('Missing Nova Canvas test runtime');
    return {
        ...options,
        conversation,
        conversation_runtime: { ...options.conversation_runtime, attempt_id: attempt },
    };
}

function requestBody(invokeModel: ReturnType<typeof vi.fn>, call = 0): Record<string, unknown> {
    const request = invokeModel.mock.calls[call]?.[0] as { body?: unknown } | undefined;
    if (typeof request?.body !== 'string') throw new Error('Expected a serialized Nova Canvas request body');
    return JSON.parse(request.body) as Record<string, unknown>;
}

async function collect(stream: CanonicalExecutionEventStream) {
    const events = [];
    for await (const event of stream) events.push(event);
    return events;
}

describe('Bedrock Nova Canvas canonical lifecycle', () => {
    it('stages verified images, persists canonical authority, and exact-retries without another invocation', async () => {
        const { driver, invokeModel } = novaDriver(async () =>
            response({ images: [PNG_BASE64, PNG_BASE64] }, 'bedrock-response-multiple'),
        );
        let published = false;
        const publish = vi.fn(async () => {
            expect(invokeModel).not.toHaveBeenCalled();
            published = true;
        });
        let stored = 0;
        const store = vi.fn<NonNullable<ExecutionOptions['store_generated_asset']>>(async (stream, metadata) => {
            expect(published).toBe(true);
            const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
            expect(bytes).toEqual(new Uint8Array(Buffer.from(PNG_BASE64, 'base64')));
            expect(metadata).toEqual({ kind: 'image', mime_type: 'image/png' });
            stored += 1;
            return {
                storage: {
                    type: 'external' as const,
                    resolver: 'url',
                    locator: { url: `s3://generated/nova-${stored}.png` },
                },
                byte_length: 8,
                content_hash: PNG_HASH,
            };
        });
        const options = {
            ...runtimeOptions('accepted', { numberOfImages: 2, seed: 7 }),
            on_canonical_request_prepared: publish,
            store_generated_asset: store,
        };

        const first = await driver.executeCanonical(
            [{ role: PromptRole.user, content: 'A fox under the moon.' }],
            options,
        );
        expect(isCanonicalAcceptedRecovery(first)).toBe(false);

        expect(publish).toHaveBeenCalledOnce();
        expect(invokeModel).toHaveBeenCalledOnce();
        expect(store).toHaveBeenCalledTimes(2);
        expect(requestBody(invokeModel)).toMatchObject({
            taskType: 'TEXT_IMAGE',
            imageGenerationConfig: { width: 512, height: 512, numberOfImages: 2, seed: 7 },
            textToImageParams: { text: 'A fox under the moon.' },
        });
        expect(first.accepted_output.generation).toMatchObject({
            protocol: 'aws.bedrock.invoke_model.nova_canvas',
            requested_model: MODEL,
            resolved_model: MODEL,
            provider_response_id: 'bedrock-response-multiple',
            status: 'completed',
        });
        expect(first.accepted_output.generation).not.toHaveProperty('request_receipt');
        expect(first.accepted_output.turn.blocks).toHaveLength(2);
        expect(Object.values(first.accepted_output.assets)).toEqual([
            expect.objectContaining({
                kind: 'image',
                mime_type: 'image/png',
                byte_length: 8,
                content_hash: PNG_HASH,
                storage: { type: 'external', resolver: 'url', locator: { url: 's3://generated/nova-1.png' } },
            }),
            expect.objectContaining({
                kind: 'image',
                mime_type: 'image/png',
                byte_length: 8,
                content_hash: PNG_HASH,
                storage: { type: 'external', resolver: 'url', locator: { url: 's3://generated/nova-2.png' } },
            }),
        ]);
        const generation = first.conversation.generations[first.accepted_output.generation.id];
        if (generation.request_receipt === undefined) throw new Error('Missing Nova Canvas request receipt');
        expect(generation.request_receipt.target).toMatchObject({
            provider: 'bedrock',
            protocol: 'aws.bedrock.invoke_model.nova_canvas',
            model: MODEL,
            options: {
                region: 'us-east-1',
                task_type: 'TEXT_IMAGE',
                parameters: { width: 512, height: 512, numberOfImages: 2, seed: 7 },
                input_images: [],
            },
        });
        expect(JSON.stringify(first.conversation)).not.toContain('test-secret-key');
        expect(JSON.stringify(first.conversation)).not.toContain('test-access-key');

        const persisted = JSON.parse(JSON.stringify(first.conversation));
        let retryPublishCount = 0;
        const retry = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'A fox under the moon.' }],
            {
                ...retryOptions(options, persisted, 'attempt:nova-canvas:accepted:retry'),
                on_canonical_request_prepared: async () => {
                    retryPublishCount += 1;
                },
            },
            undefined,
            { stream_id: 'stream:nova-canvas:accepted:retry' },
        );
        expect(await collect(retry)).toEqual([
            expect.objectContaining({ type: 'response_accepted', sequence: 0, origin: 'accepted_recovery' }),
        ]);
        expect(retry.completion?.accepted_output).toEqual(first.accepted_output);
        expect(isCanonicalAcceptedRecovery(retry.completion)).toBe(true);
        expect(isCanonicalAcceptedRecovery(JSON.parse(JSON.stringify(retry.completion)))).toBe(false);
        expect(retryPublishCount).toBe(0);
        expect(invokeModel).toHaveBeenCalledOnce();
        expect(store).toHaveBeenCalledTimes(2);
        await retry.closed;
    });

    it.each([
        {
            task: NovaImageGenerationTaskType.TEXT_IMAGE_WITH_IMAGE_CONDITIONING,
            options: { controlMode: 'CANNY_EDGE' as const },
            files: 1,
            wireTask: 'TEXT_IMAGE',
            parameter: 'textToImageParams',
        },
        {
            task: NovaImageGenerationTaskType.COLOR_GUIDED_GENERATION,
            options: { colors: ['#336699'] },
            files: 0,
            wireTask: 'COLOR_GUIDED_GENERATION',
            parameter: 'colorGuidedGenerationParams',
        },
        {
            task: NovaImageGenerationTaskType.IMAGE_VARIATION,
            options: { similarityStrength: 0.75 },
            files: 1,
            wireTask: 'IMAGE_VARIATION',
            parameter: 'imageVariationParams',
        },
        {
            task: NovaImageGenerationTaskType.INPAINTING,
            options: {},
            files: 2,
            wireTask: 'INPAINTING',
            parameter: 'inPaintingParams',
        },
        {
            task: NovaImageGenerationTaskType.OUTPAINTING,
            options: { outPaintingMode: 'PRECISE' as const },
            files: 2,
            wireTask: 'OUTPAINTING',
            parameter: 'outPaintingParams',
        },
        {
            task: NovaImageGenerationTaskType.BACKGROUND_REMOVAL,
            options: {},
            files: 1,
            content: '',
            wireTask: 'BACKGROUND_REMOVAL',
            parameter: 'backgroundRemovalParams',
        },
    ])('preserves $task request fidelity through the public canonical seam', async (testCase) => {
        const { driver, invokeModel } = novaDriver();
        const files = Array.from(
            { length: testCase.files },
            (_, index) => new Base64DataSource(`source-${index}.png`, 'image/png', PNG_BASE64),
        );
        const flow = testCase.task.toLowerCase();
        const options = runtimeOptions(flow, { taskType: testCase.task, ...testCase.options });

        const result = await driver.executeCanonical(
            [{ role: PromptRole.user, content: testCase.content ?? 'Preserve this prompt.', files }],
            options,
        );

        expect(result.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'image', asset_id: expect.any(String) }),
        );
        const body = requestBody(invokeModel);
        expect(body.taskType).toBe(testCase.wireTask);
        expect(body).toHaveProperty(testCase.parameter);
        if (testCase.task === NovaImageGenerationTaskType.IMAGE_VARIATION) {
            expect(body.imageVariationParams).toMatchObject({ text: 'Preserve this prompt.', images: [PNG_BASE64] });
        }
        const document = parseConversationDocument(result.conversation);
        const receivedAssets = Object.values(document.assets).filter((asset) => asset.provenance.type === 'received');
        expect(receivedAssets).toHaveLength(testCase.files);
        expect(receivedAssets).toEqual(
            expect.arrayContaining(
                files.map(() => expect.objectContaining({ byte_length: 8, content_hash: PNG_HASH })),
            ),
        );
        const generation = document.generations[result.accepted_output.generation.id];
        if (generation.request_receipt === undefined) throw new Error('Missing Nova Canvas request receipt');
        expect(generation.request_receipt.target.options).toMatchObject({
            region: 'us-east-1',
            task_type: testCase.task,
            input_images: files.map((_, index) =>
                expect.objectContaining({
                    path: `messages/0/content/${index}/image`,
                    role: 'user',
                    format: 'png',
                }),
            ),
        });
        expect(JSON.stringify(generation.request_receipt.target.options)).not.toContain(PNG_BASE64);
    });

    it.each([
        NovaImageGenerationTaskType.IMAGE_VARIATION,
        NovaImageGenerationTaskType.INPAINTING,
        NovaImageGenerationTaskType.OUTPAINTING,
    ])('lowers system and safety instructions into the native %s text prompt', async (task) => {
        const { driver, invokeModel } = novaDriver();
        const fileCount = task === NovaImageGenerationTaskType.IMAGE_VARIATION ? 1 : 2;
        const files = Array.from(
            { length: fileCount },
            (_, index) => new Base64DataSource(`source-${index}.png`, 'image/png', PNG_BASE64),
        );
        await driver.executeCanonical(
            [
                { role: PromptRole.system, content: 'Preserve the composition.' },
                { role: PromptRole.safety, content: 'Use family-safe details.' },
                { role: PromptRole.negative, content: 'No text overlays.' },
                { role: PromptRole.user, content: 'A blue fox.', files },
            ],
            runtimeOptions(`instructions:${task}`, { taskType: task }),
        );
        const body = requestBody(invokeModel);
        const parameters =
            body.imageVariationParams ?? body.inPaintingParams ?? body.outPaintingParams ?? Object.create(null);
        expect(parameters.text).toContain('A blue fox.');
        expect(parameters.text).toContain('Preserve the composition.');
        expect(parameters.text).toContain('Use family-safe details.');
        expect(parameters.negativeText).toContain('No text overlays.');
    });

    it('publishes before transport and rejects a failed durability barrier', async () => {
        const { driver, invokeModel } = novaDriver();
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Do not send.' }], {
                ...runtimeOptions('barrier'),
                on_canonical_request_prepared: async () => {
                    throw new Error('durability barrier failed');
                },
            }),
        ).rejects.toThrow('durability barrier failed');
        expect(invokeModel).not.toHaveBeenCalled();
    });

    it('rejects changed target routing on accepted retry before transport', async () => {
        const firstDriver = novaDriver();
        const firstOptions = runtimeOptions('routing');
        const first = await firstDriver.driver.executeCanonical(
            [{ role: PromptRole.user, content: 'Route exactly.' }],
            firstOptions,
        );
        const secondDriver = novaDriver(undefined, 'us-west-2');

        await expect(
            secondDriver.driver.executeCanonical(
                [{ role: PromptRole.user, content: 'Route exactly.' }],
                retryOptions(
                    firstOptions,
                    JSON.parse(JSON.stringify(first.conversation)),
                    'attempt:nova-canvas:routing:retry',
                ),
            ),
        ).rejects.toThrow(/incompatible Nova Canvas target options/);
        expect(secondDriver.invokeModel).not.toHaveBeenCalled();
    });

    it('does not accept output when durable generated-asset staging changes the bytes', async () => {
        const { driver } = novaDriver();
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Stage this.' }], {
                ...runtimeOptions('staging-mismatch'),
                store_generated_asset: async () => ({
                    storage: {
                        type: 'external',
                        resolver: 'url',
                        locator: { url: 's3://generated/wrong.png' },
                    },
                    byte_length: 7,
                    content_hash: 'sha256:wrong',
                }),
            }),
        ).rejects.toThrow(/did not preserve the exact Nova Canvas image bytes/);
    });

    it.each([
        ['provider rejection', response({ error: 'safety policy', images: [] }), /rejected.*safety policy/],
        ['empty output', response({ images: [] }), /contains no images/],
        ['malformed base64', response({ images: ['not-base64'] }), /malformed base64/],
        ['unsupported image bytes', response({ images: ['AQID'] }), /unsupported format/],
    ])('fails closed for %s without accepting an image turn', async (_label, nativeResponse, expected) => {
        const { driver } = novaDriver(async () => nativeResponse);
        await expect(
            driver.executeCanonical(
                [{ role: PromptRole.user, content: 'Invalid result.' }],
                runtimeOptions(`invalid:${_label}`),
            ),
        ).rejects.toThrow(expected);
    });

    it('rejects unsupported capabilities before reading input or invoking Bedrock', async () => {
        const { driver, invokeModel } = novaDriver();
        const unsupported = new Base64DataSource('document.pdf', 'application/pdf', 'AQID');
        const getStream = vi.spyOn(unsupported, 'getStream');
        await expect(
            driver.executeCanonical(
                [{ role: PromptRole.user, content: 'Unsupported.', files: [unsupported] }],
                runtimeOptions('unsupported-mime'),
            ),
        ).rejects.toThrow(/does not support application\/pdf input files/);
        expect(getStream).not.toHaveBeenCalled();
        const unsupportedImage = new Base64DataSource('source.webp', 'image/webp', PNG_BASE64);
        const getUnsupportedImageStream = vi.spyOn(unsupportedImage, 'getStream');
        await expect(
            driver.executeCanonical(
                [{ role: PromptRole.user, content: '', files: [unsupportedImage] }],
                runtimeOptions('unsupported-image-mime', { taskType: NovaImageGenerationTaskType.BACKGROUND_REMOVAL }),
            ),
        ).rejects.toThrow(/does not support image\/webp input files/);
        expect(getUnsupportedImageStream).not.toHaveBeenCalled();
        for (const [role, content] of [
            [PromptRole.user, 'Discarded text.'],
            [PromptRole.system, 'Discarded system instruction.'],
            [PromptRole.safety, 'Discarded safety instruction.'],
            [PromptRole.negative, 'Discarded negative prompt.'],
        ] as const) {
            const background = new Base64DataSource(`${role}.png`, 'image/png', PNG_BASE64);
            const getBackgroundStream = vi.spyOn(background, 'getStream');
            const segments = [
                { role: PromptRole.user, content: '', files: [background] },
                { role, content },
            ];
            await expect(
                driver.executeCanonical(
                    segments,
                    runtimeOptions(`background-${role}`, {
                        taskType: NovaImageGenerationTaskType.BACKGROUND_REMOVAL,
                    }),
                ),
            ).rejects.toThrow(/background removal does not accept text/);
            expect(getBackgroundStream).not.toHaveBeenCalled();
        }
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Tool.' }], {
                ...runtimeOptions('unsupported-tool'),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            }),
        ).rejects.toThrow(/does not support tools/);
        await expect(
            driver.executeCanonical(
                [{ role: PromptRole.assistant, content: 'Prior model output.' }],
                runtimeOptions('unsupported-role'),
            ),
        ).rejects.toThrow(/does not support assistant input/);
        expect(invokeModel).not.toHaveBeenCalled();
    });

    it('rejects retained active tool definitions before reading input or invoking Bedrock', async () => {
        const { driver, invokeModel } = novaDriver();
        const recordedAt = '2026-09-30T00:00:00.000Z';
        const initial = createConversationDocument({
            id: 'conversation:nova-canvas:active-tool',
            created_at: recordedAt,
        });
        const document = appendConversationRecords(
            initial,
            {
                tool_definitions: [
                    {
                        id: 'tool-definition:lookup',
                        name: 'lookup',
                        version: '1',
                        input_schema: { type: 'object' },
                    },
                ],
                active_tool_definition_ids: ['tool-definition:lookup'],
            },
            {
                expected_revision: initial.revision,
                operation_id: 'operation:nova-canvas:active-tool',
                payload_fingerprint: 'sha256:nova-canvas-active-tool',
                recorded_at: recordedAt,
            },
        ).document;
        const image = new Base64DataSource('source.png', 'image/png', PNG_BASE64);
        const getStream = vi.spyOn(image, 'getStream');
        const options = runtimeOptions(
            'active-tool',
            { taskType: NovaImageGenerationTaskType.IMAGE_VARIATION },
            document,
        );
        if (options.conversation_runtime === undefined) throw new Error('Missing Nova Canvas test runtime');
        options.conversation_runtime = { ...options.conversation_runtime, conversation_id: document.id };

        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Variation.', files: [image] }], options),
        ).rejects.toThrow(/does not support active canonical tool definitions/);
        expect(getStream).not.toHaveBeenCalled();
        expect(invokeModel).not.toHaveBeenCalled();
    });

    it('rejects conversation continuation before another invocation', async () => {
        const { driver, invokeModel } = novaDriver();
        const first = await driver.executeCanonical(
            [{ role: PromptRole.user, content: 'First.' }],
            runtimeOptions('continuation:first'),
        );
        const continuation = runtimeOptions('continuation:second', {}, first.conversation);
        if (continuation.conversation_runtime === undefined) throw new Error('Missing continuation runtime');
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Second.' }], {
                ...continuation,
                conversation_runtime: {
                    ...continuation.conversation_runtime,
                    conversation_id: first.conversation.id,
                },
            }),
        ).rejects.toThrow(/does not support conversation continuation/);
        expect(invokeModel).toHaveBeenCalledOnce();
    });

    it('settles cancellation delivery while retaining the Bedrock lease until transport cleanup', async () => {
        let rejectInvocation: ((reason: unknown) => void) | undefined;
        const { driver, invokeModel } = novaDriver(
            () =>
                new Promise<InvokeModelCommandOutput>((_resolve, reject) => {
                    rejectInvocation = reject;
                }),
        );
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Wait.' }],
            runtimeOptions('cancel'),
            undefined,
            { stream_id: 'stream:nova-canvas:cancel' },
        );
        const pending = stream[Symbol.asyncIterator]().next();
        await vi.waitFor(() => expect(invokeModel).toHaveBeenCalledOnce());
        const terminal = await stream.cancel();
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        await expect(pending).resolves.toMatchObject({ value: terminal });
        expect(
            (invokeModel.mock.calls[0]?.[1] as { abortSignal?: AbortSignal } | undefined)?.abortSignal?.aborted,
        ).toBe(true);
        driver.destroy();
        let closed = false;
        void stream.closed.then(() => {
            closed = true;
        });
        await Promise.resolve();
        expect(closed).toBe(false);
        expect(driver.cleanup).not.toHaveBeenCalled();
        rejectInvocation?.(new DOMException('provider cleanup completed', 'AbortError'));
        await stream.closed;
        expect(closed).toBe(true);
        await vi.waitFor(() => expect(driver.cleanup).toHaveBeenCalledOnce());
        expect(stream.completion).toBeUndefined();
    });
});
