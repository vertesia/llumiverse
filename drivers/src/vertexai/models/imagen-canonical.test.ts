import { helpers, type protos } from '@google-cloud/aiplatform';
import { createConversationDocument, parseConversationDocument } from '@llumiverse/conversation';
import {
    Base64DataSource,
    type CanonicalExecutionEventStream,
    type ExecutionOptions,
    PromptRole,
} from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { VertexAIDriver } from '../index.js';
import { ImagenMaskMode, ImagenTaskType } from './imagen.js';

const PNG_BASE64 = 'iVBORw0KGgo=';
const PNG_HASH = 'sha256:4c4b6a3be1314ab86138bef4314dde022e600960d8689a2c8f8631802d20dab6';
const MODEL = 'publishers/google/models/imagen-3.0-generate-002';

class ImagenLifecycleTestDriver extends VertexAIDriver {
    readonly cleanup = vi.fn();

    protected override async destroyProviderResources(): Promise<void> {
        this.cleanup();
    }
}

function runtimeOptions(flow: string, operation = 'first'): ExecutionOptions {
    return {
        model: MODEL,
        model_options: {
            _option_id: 'vertexai-imagen',
            number_of_images: 2,
            image_file_type: 'image/png',
            seed: 7,
        },
        conversation_runtime: {
            conversation_id: `conversation:imagen:${flow}`,
            request_id: `request:imagen:${flow}`,
            attempt_id: `attempt:imagen:${flow}:${operation}`,
            input_operation_id: `input:imagen:${flow}`,
            response_operation_id: `response:imagen:${flow}`,
            recorded_at: '2026-09-30T00:00:00.000Z',
        },
    };
}

function retryRuntime(options: ExecutionOptions, attempt_id: string) {
    if (options.conversation_runtime === undefined) throw new Error('missing test runtime');
    return { ...options.conversation_runtime, attempt_id };
}

function protobufValue(input: Record<string, unknown>): protos.google.protobuf.IValue {
    const value = helpers.toValue(input);
    if (value === undefined || value === null) throw new Error('missing protobuf value');
    return value as protos.google.protobuf.IValue;
}

function imagePrediction(overrides: Record<string, unknown> = {}): protos.google.protobuf.IValue {
    return protobufValue({ bytesBase64Encoded: PNG_BASE64, mimeType: 'image/png', ...overrides });
}

function predictionResponse(predictions: protos.google.protobuf.IValue[]) {
    return { predictions, deployedModelId: 'imagen-deployed', model: 'imagen', modelVersionId: '002' };
}

function predictionPromise(response: protos.google.cloud.aiplatform.v1.IPredictResponse) {
    const promise = Promise.resolve([response, undefined, undefined]) as Promise<
        [protos.google.cloud.aiplatform.v1.IPredictResponse, unknown, unknown]
    > & { cancel(): void };
    promise.cancel = vi.fn();
    return promise;
}

function imagenDriver(response: protos.google.cloud.aiplatform.v1.IPredictResponse) {
    const driver = new VertexAIDriver({ project: 'test-project', region: 'us-central1' });
    const predict = vi.fn((_request: protos.google.cloud.aiplatform.v1.IPredictRequest, _options: unknown) =>
        predictionPromise(response),
    );
    vi.spyOn(driver, 'getImagenClient').mockResolvedValue({ predict } as unknown as Awaited<
        ReturnType<VertexAIDriver['getImagenClient']>
    >);
    return { driver, predict };
}

async function collect(stream: CanonicalExecutionEventStream) {
    const events = [];
    for await (const event of stream) events.push(event);
    return events;
}

describe('Vertex Imagen canonical lifecycle', () => {
    it('accepts verified staged images and exact-retries after JSON reload without another prediction', async () => {
        const { driver, predict } = imagenDriver(
            predictionResponse([
                imagePrediction({ prompt: 'Enhanced first prompt' }),
                imagePrediction({ prompt: 'Enhanced second prompt' }),
            ]),
        );
        const publish = vi.fn(async () => undefined);
        let storedAssetCount = 0;
        const store = vi.fn<NonNullable<ExecutionOptions['store_generated_asset']>>(async (stream, metadata) => {
            const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
            expect(bytes).toEqual(new Uint8Array(Buffer.from(PNG_BASE64, 'base64')));
            expect(metadata).toEqual({ kind: 'image', mime_type: 'image/png' });
            storedAssetCount += 1;
            return {
                storage: {
                    type: 'external' as const,
                    resolver: 'url',
                    locator: { url: `https://assets.test/${storedAssetCount}.png` },
                },
                byte_length: 8,
                content_hash: PNG_HASH,
            };
        });
        const freshOptions = runtimeOptions('sync');
        const options = {
            ...freshOptions,
            conversation: createConversationDocument({
                id: freshOptions.conversation_runtime?.conversation_id ?? 'missing-test-conversation-id',
                created_at: '2026-09-30T00:00:00.000Z',
            }),
            on_canonical_request_prepared: publish,
            store_generated_asset: store,
        };
        const first = await driver.executeCanonical(
            [{ role: PromptRole.user, content: 'A fox under the moon.' }],
            options,
        );

        expect(publish).toHaveBeenCalledOnce();
        expect(predict).toHaveBeenCalledOnce();
        expect(store).toHaveBeenCalledTimes(2);
        expect(first.accepted_output.turn.blocks).toEqual([
            expect.objectContaining({ type: 'image', caption: 'Enhanced first prompt' }),
            expect.objectContaining({ type: 'image', caption: 'Enhanced second prompt' }),
        ]);
        const document = parseConversationDocument(first.conversation);
        const outputAssets = Object.values(first.accepted_output.assets);
        expect(outputAssets).toHaveLength(2);
        expect(outputAssets).toEqual([
            expect.objectContaining({ mime_type: 'image/png', byte_length: 8, content_hash: PNG_HASH }),
            expect.objectContaining({ mime_type: 'image/png', byte_length: 8, content_hash: PNG_HASH }),
        ]);
        const generation = Object.values(document.generations)[0];
        expect(generation.requested_model).toBe(MODEL);
        expect(generation.resolved_model).toBe('imagen-3.0-generate-002');
        expect(generation.request_receipt?.target).toMatchObject({
            provider: 'vertexai',
            protocol: 'google.vertex.imagen.predict',
            model: MODEL,
            options: {
                endpoint:
                    'projects/test-project/locations/us-central1/publishers/google/models/imagen-3.0-generate-002',
                location: 'us-central1',
                task_type: 'TEXT_IMAGE',
                parameters: expect.objectContaining({ sampleCount: 2, seed: 7 }),
            },
        });
        expect(JSON.stringify(generation.request_receipt?.target.options)).not.toContain('A fox');
        expect(predict.mock.calls[0]?.[0]).toMatchObject({
            endpoint: 'projects/test-project/locations/us-central1/publishers/google/models/imagen-3.0-generate-002',
        });

        const recovered = await driver.executeCanonical([{ role: PromptRole.user, content: 'A fox under the moon.' }], {
            ...options,
            conversation: JSON.parse(JSON.stringify(first.conversation)),
            conversation_runtime: retryRuntime(options, 'attempt:imagen:sync:retry'),
        });
        expect(recovered.accepted_output).toEqual(first.accepted_output);
        expect(predict).toHaveBeenCalledOnce();
        expect(publish).toHaveBeenCalledOnce();
        expect(store).toHaveBeenCalledTimes(2);

        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'A changed prompt.' }], {
                ...options,
                conversation: first.conversation,
                conversation_runtime: retryRuntime(options, 'attempt:imagen:sync:changed'),
            }),
        ).rejects.toThrow(/incompatible|fingerprint|retry|different payload/i);
        expect(predict).toHaveBeenCalledOnce();

        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'A fox under the moon.' }], {
                ...options,
                model_options: { ...options.model_options, seed: 8 },
                conversation: first.conversation,
                conversation_runtime: retryRuntime(options, 'attempt:imagen:sync:changed-options'),
            }),
        ).rejects.toThrow(/incompatible|fingerprint|retry|different payload/i);
        expect(predict).toHaveBeenCalledOnce();
    });

    it('retains reference and mask inputs and binds them before transport', async () => {
        const { driver, predict } = imagenDriver(predictionResponse([imagePrediction()]));
        const options: ExecutionOptions = {
            ...runtimeOptions('references'),
            model_options: {
                _option_id: 'vertexai-imagen',
                edit_mode: ImagenTaskType.EDIT_MODE_INPAINT_INSERTION,
                mask_mode: ImagenMaskMode.MASK_MODE_USER_PROVIDED,
            },
        };
        const result = await driver.executeCanonical(
            [
                {
                    role: PromptRole.user,
                    content: 'Replace the sky.',
                    files: [new Base64DataSource('source.png', 'image/png', PNG_BASE64)],
                },
                {
                    role: PromptRole.mask,
                    content: '',
                    files: [new Base64DataSource('mask.png', 'image/png', PNG_BASE64)],
                },
            ],
            options,
        );

        const document = parseConversationDocument(result.conversation);
        const inputTurn = document.turns.find((turn) => turn.kind === 'user');
        expect(inputTurn?.blocks.filter((block) => block.type === 'image')).toHaveLength(2);
        expect(Object.values(document.assets).filter((asset) => asset.provenance.type === 'received')).toEqual([
            expect.objectContaining({ byte_length: 8, content_hash: PNG_HASH }),
            expect.objectContaining({ byte_length: 8, content_hash: PNG_HASH }),
        ]);
        expect(predict.mock.calls[0]?.[0]).toMatchObject({ instances: [expect.any(Object)] });
        const instance = helpers.fromValue(
            predict.mock.calls[0]?.[0].instances?.[0] as Parameters<typeof helpers.fromValue>[0],
        );
        expect(instance).toMatchObject({
            referenceImages: [
                { referenceType: 'REFERENCE_TYPE_RAW', referenceImage: { bytesBase64Encoded: PNG_BASE64 } },
                { referenceType: 'REFERENCE_TYPE_MASK', referenceImage: { bytesBase64Encoded: PNG_BASE64 } },
            ],
        });
        const generation = Object.values(document.generations)[0];
        expect(generation.request_receipt?.target.options).toMatchObject({
            references: [
                { referenceType: 'REFERENCE_TYPE_RAW', referenceId: 1 },
                { referenceType: 'REFERENCE_TYPE_MASK', referenceId: 2 },
            ],
        });
        expect(JSON.stringify(generation.request_receipt?.target.options)).not.toContain(PNG_BASE64);
    });

    it('publishes before transport and rejects a failed durability barrier', async () => {
        const { driver, predict } = imagenDriver(predictionResponse([imagePrediction()]));
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Do not send.' }], {
                ...runtimeOptions('barrier'),
                on_canonical_request_prepared: async () => {
                    throw new Error('durability barrier failed');
                },
            }),
        ).rejects.toThrow('durability barrier failed');
        expect(predict).not.toHaveBeenCalled();
    });

    it('does not accept output when durable generated-asset staging changes the bytes', async () => {
        const { driver } = imagenDriver(predictionResponse([imagePrediction()]));
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Stage this.' }], {
                ...runtimeOptions('staging-mismatch'),
                store_generated_asset: async () => ({
                    storage: {
                        type: 'external',
                        resolver: 'url',
                        locator: { url: 'https://assets.test/wrong.png' },
                    },
                    byte_length: 7,
                    content_hash: 'sha256:wrong',
                }),
            }),
        ).rejects.toThrow(/did not preserve the exact Imagen image bytes/);
    });

    it('streams a finite accepted event and recovers it without another prediction', async () => {
        const { driver, predict } = imagenDriver(predictionResponse([imagePrediction()]));
        const options = runtimeOptions('typed');
        const first = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Finite image.' }],
            options,
            undefined,
            { stream_id: 'stream:imagen:typed:first' },
        );
        const firstEvents = await collect(first);
        expect(firstEvents).toEqual([
            expect.objectContaining({ type: 'response_accepted', sequence: 0, origin: 'live_transport' }),
        ]);
        if (first.completion === undefined) throw new Error('Expected accepted Imagen response');
        const recoveredOptions: ExecutionOptions = {
            ...options,
            conversation: JSON.parse(JSON.stringify(first.completion.conversation)),
            conversation_runtime: retryRuntime(options, 'attempt:imagen:typed:retry'),
        };
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Finite image.' }], recoveredOptions),
        ).resolves.toMatchObject({ accepted_output: first.completion.accepted_output });
        const recovered = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Finite image.' }],
            recoveredOptions,
            undefined,
            { stream_id: 'stream:imagen:typed:retry' },
        );
        expect(await collect(recovered)).toEqual([
            expect.objectContaining({ type: 'response_accepted', sequence: 0, origin: 'accepted_recovery' }),
        ]);
        expect(predict).toHaveBeenCalledOnce();
    });

    it('settles cancellation delivery while retaining ownership of noncooperative prediction cleanup', async () => {
        const driver = new ImagenLifecycleTestDriver({ project: 'test-project', region: 'us-central1' });
        let rejectPrediction: ((reason: unknown) => void) | undefined;
        const cancel = vi.fn(() => {
            throw new Error('GAX cancel failed synchronously');
        });
        const predict = vi.fn(() => {
            const pending = new Promise<[protos.google.cloud.aiplatform.v1.IPredictResponse, unknown, unknown]>(
                (_resolve, reject) => {
                    rejectPrediction = reject;
                },
            ) as Promise<[protos.google.cloud.aiplatform.v1.IPredictResponse, unknown, unknown]> & { cancel(): void };
            pending.cancel = cancel;
            return pending;
        });
        vi.spyOn(driver, 'getImagenClient').mockResolvedValue({ predict } as unknown as Awaited<
            ReturnType<VertexAIDriver['getImagenClient']>
        >);
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Wait.' }],
            runtimeOptions('cancel'),
            undefined,
            { stream_id: 'stream:imagen:cancel' },
        );
        const pending = stream[Symbol.asyncIterator]().next();
        await vi.waitFor(() => expect(predict).toHaveBeenCalledOnce());
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
        rejectPrediction?.(new DOMException('provider cleanup completed', 'AbortError'));
        await stream.closed;
        expect(cancel).toHaveBeenCalledOnce();
        expect(closed).toBe(true);
        await vi.waitFor(() => expect(driver.cleanup).toHaveBeenCalledOnce());
        expect(stream.completion).toBeUndefined();
    });

    it.each([
        ['empty predictions', predictionResponse([]), /no predictions/],
        [
            'filtered prediction',
            predictionResponse([protobufValue({ raiFilteredReason: 'safety policy' })]),
            /filtered: safety policy/,
        ],
        [
            'malformed base64',
            predictionResponse([protobufValue({ bytesBase64Encoded: 'not-base64', mimeType: 'image/png' })]),
            /malformed base64/,
        ],
        [
            'mismatched MIME',
            predictionResponse([imagePrediction({ mimeType: 'image/jpeg' })]),
            /MIME type image\/jpeg does not match/,
        ],
    ])('fails closed for %s without accepting an image turn', async (_label, response, expected) => {
        const { driver } = imagenDriver(response);
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Invalid result.' }], runtimeOptions(_label)),
        ).rejects.toThrow(expected);
    });

    it('rejects unsupported inputs and conversation continuation before prediction', async () => {
        const { driver, predict } = imagenDriver(predictionResponse([imagePrediction()]));
        await expect(
            driver.executeCanonical([{ role: PromptRole.assistant, content: 'Prior answer.' }], runtimeOptions('role')),
        ).rejects.toThrow(/does not support assistant input/);
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Tool request.' }], {
                ...runtimeOptions('tool'),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            }),
        ).rejects.toThrow(/does not support tools/);

        const accepted = await driver.executeCanonical(
            [{ role: PromptRole.user, content: 'First.' }],
            runtimeOptions('continuation'),
        );
        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Second.' }], {
                ...runtimeOptions('continuation-next'),
                conversation: accepted.conversation,
                conversation_runtime: {
                    conversation_id: accepted.conversation.id,
                    request_id: 'request:imagen:continuation-next',
                    attempt_id: 'attempt:imagen:continuation-next:first',
                    input_operation_id: 'input:imagen:continuation-next',
                    response_operation_id: 'response:imagen:continuation-next',
                    recorded_at: '2026-09-30T00:01:00.000Z',
                },
            }),
        ).rejects.toThrow(/does not support conversation continuation/);
        expect(predict).toHaveBeenCalledOnce();
    });
});
