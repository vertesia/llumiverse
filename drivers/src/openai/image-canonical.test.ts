import {
    type Asset,
    appendConversationRecords,
    type ConversationDocument,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    parseConversationDocument,
    type ResolveConversationAsset,
} from '@llumiverse/conversation';
import {
    Base64DataSource,
    type CanonicalExecutionContextInputOptions,
    type CanonicalExecutionInputOptions,
    type ExecutionOptions,
    isCanonicalAcceptedRecovery,
    PromptRole,
    Providers,
} from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { openAIImageEditMaskBinding } from './image.js';
import { OpenAIResponsesDriverBase } from './index.js';

class ImageDriver extends OpenAIResponsesDriverBase {
    provider: OpenAIResponsesDriverBase['provider'] = Providers.openai;
    service: OpenAI;

    constructor(
        generate: (request: unknown, options?: { signal?: AbortSignal }) => Promise<unknown>,
        private readonly imageFetch: typeof fetch = vi.fn(),
        edit: (request: unknown, options?: { signal?: AbortSignal }) => Promise<unknown> = vi.fn(),
    ) {
        super({});
        this.service = { images: { edit, generate } } as unknown as OpenAI;
    }

    protected override getDriverFetch(): typeof fetch {
        return this.imageFetch;
    }
}

class AlternateProviderImageDriver extends ImageDriver {
    override provider: OpenAIResponsesDriverBase['provider'] = Providers.azure_openai;

    protected override supportsCanonicalImageGeneration(_options: ExecutionOptions): boolean {
        return true;
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

function runtime(flow: string, conversation?: ConversationDocument): CanonicalExecutionInputOptions {
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

function contextRuntime(flow: string, conversation: ConversationDocument): CanonicalExecutionContextInputOptions {
    return {
        model: 'gpt-image-1',
        conversation,
        conversation_runtime: {
            conversation_id: conversation.id,
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

async function retainedEditDocument(flow: string, withMask = false): Promise<ConversationDocument> {
    const document = createConversationDocument({ id: `conversation:${flow}`, created_at: at });
    const referenceBytes = Buffer.from(pngA, 'base64');
    const referenceAssetId = `asset:${flow}:reference`;
    const promptTurn = createUserTurn({
        id: `turn:${flow}:prompt`,
        authority: 'ordinary',
        blocks: [
            createTextBlock({ id: `block:${flow}:prompt`, text: 'Edit the retained image.', format: 'plain' }),
            { id: `block:${flow}:reference`, type: 'image', asset_id: referenceAssetId },
        ],
        status: 'completed',
        timestamps: { recorded_at: at },
        provenance: { type: 'received' },
        model_visibility: 'include',
    });
    const turns = [promptTurn];
    const assets: Asset[] = [
        {
            id: referenceAssetId,
            kind: 'image',
            mime_type: 'image/png',
            storage: { type: 'inline_base64', data: pngA },
            provenance: { type: 'received', source_turn_id: promptTurn.id },
            byte_length: referenceBytes.byteLength,
            content_hash: await hash(referenceBytes),
            created_at: at,
        },
    ];
    if (withMask) {
        const maskBytes = Buffer.from(encodedImage(pngSignature, 'mask bytes'), 'base64');
        const maskAssetId = `asset:${flow}:mask`;
        const maskImageBlockId = `block:${flow}:mask:image`;
        const maskTurn = createUserTurn({
            id: `turn:${flow}:mask`,
            authority: 'ordinary',
            blocks: [
                {
                    id: `block:${flow}:mask:binding`,
                    type: 'extension',
                    namespace: 'openai.images.edit_mask',
                    version: '1',
                    payload: { image_block_id: maskImageBlockId },
                    model_projection: 'registered',
                },
                { id: maskImageBlockId, type: 'image', asset_id: maskAssetId },
            ],
            status: 'completed',
            timestamps: { recorded_at: at },
            provenance: { type: 'received' },
            model_visibility: 'include',
        });
        turns.push(maskTurn);
        assets.push({
            id: maskAssetId,
            kind: 'image',
            mime_type: 'image/png',
            storage: { type: 'inline_base64', data: maskBytes.toString('base64') },
            provenance: { type: 'received', source_turn_id: maskTurn.id },
            byte_length: maskBytes.byteLength,
            content_hash: await hash(maskBytes),
            created_at: at,
        });
    }
    return appendConversationRecords(
        document,
        {
            turns,
            assets,
            context_entries: turns.map((turn) => ({
                id: `context:${turn.id}`,
                type: 'source_turn' as const,
                turn_id: turn.id,
            })),
        },
        {
            expected_revision: document.revision,
            operation_id: `materialized:${flow}`,
            payload_fingerprint: `sha256:${'a'.repeat(64)}`,
            recorded_at: at,
        },
    ).document;
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
        const options: CanonicalExecutionInputOptions = {
            ...runtime('multiple'),
            model_options: {
                _option_id: 'openai-gpt-image',
                width: 1024,
                height: 1536,
                image_quality: 'high',
                background: 'transparent',
                output_format: 'webp',
                output_compression: 60,
                moderation: 'low',
                partial_images: 2,
                n: 2,
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
            n: 2,
            size: '1024x1536',
            quality: 'high',
            background: 'transparent',
            output_format: 'webp',
            output_compression: 60,
            moderation: 'low',
            partial_images: 2,
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

        const publishedV2Artifact = JSON.parse(JSON.stringify(first.conversation));
        const publishedV2Generation = publishedV2Artifact.generations[first.accepted_output.generation.id];
        publishedV2Generation.adapter_version = '2026-09-30.canonical.2';
        publishedV2Generation.request_receipt.target.adapter_version = '2026-09-30.canonical.2';
        const v2Retry = await driver.executeCanonical(segments, {
            ...options,
            conversation: JSON.parse(JSON.stringify(publishedV2Artifact)),
        });
        expect(isCanonicalAcceptedRecovery(v2Retry)).toBe(true);
        expect(v2Retry.accepted_output.turn).toEqual(first.accepted_output.turn);
        expect(v2Retry.accepted_output.generation.adapter_version).toBe('2026-09-30.canonical.2');
        expect(v2Retry.accepted_output.generation.usage).toEqual(first.accepted_output.generation.usage);

        await expect(
            driver.executeCanonical([{ role: PromptRole.user, content: 'Draw a changed icon.' }], {
                ...options,
                conversation: JSON.parse(JSON.stringify(publishedV2Artifact)),
            }),
        ).rejects.toThrow('was already used with a different payload');
        await expect(
            driver.executeCanonical(segments, {
                ...options,
                conversation: JSON.parse(JSON.stringify(publishedV2Artifact)),
                model_options: {
                    _option_id: 'openai-gpt-image',
                    width: 1024,
                    height: 1536,
                    image_quality: 'low',
                    background: 'transparent',
                    output_format: 'webp',
                    output_compression: 60,
                    moderation: 'low',
                    partial_images: 2,
                    n: 2,
                },
            }),
        ).rejects.toThrow('incompatible request identity');

        const mismatchedAdapter = JSON.parse(JSON.stringify(publishedV2Artifact));
        mismatchedAdapter.generations[first.accepted_output.generation.id].request_receipt.target.adapter_version =
            '2026-09-30.canonical.3';
        await expect(
            driver.executeCanonical(segments, { ...options, conversation: mismatchedAdapter }),
        ).rejects.toThrow('mismatched adapter versions');

        const priorAdapter = JSON.parse(JSON.stringify(publishedV2Artifact));
        const priorGeneration = priorAdapter.generations[first.accepted_output.generation.id];
        priorGeneration.adapter_version = '2026-09-30.canonical.1';
        priorGeneration.request_receipt.target.adapter_version = '2026-09-30.canonical.1';
        await expect(
            driver.executeCanonical(segments, {
                ...options,
                conversation: priorAdapter,
            }),
        ).rejects.toThrow(/unsupported adapter version 2026-09-30.canonical.1/);
        expect(generate).toHaveBeenCalledOnce();
        expect(publish).toHaveBeenCalledOnce();
    });

    it('executes retained image and mask context, then exact-recovers from JSON without authoring input', async () => {
        const generate = vi.fn();
        const edit = vi.fn(async (_request: unknown, _options?: { signal?: AbortSignal }) =>
            imageResponse([{ b64_json: webpA }]),
        );
        const imageFetch = vi.fn<typeof fetch>(async (input) => {
            const value = String(input);
            const comma = value.indexOf(',');
            const match = /^data:(image\/[a-z0-9.+-]+);base64,/i.exec(value);
            if (match === null || comma < 0) {
                return new Response('unsupported', { status: 404 });
            }
            return new Response(Buffer.from(value.slice(comma + 1), 'base64'), {
                headers: { 'content-type': match[1] },
            });
        });
        const driver = new ImageDriver(generate, imageFetch, edit);
        const document = JSON.parse(JSON.stringify(await retainedEditDocument('retained-context', true)));
        const referenceBlock = document.turns
            .flatMap((turn: ConversationDocument['turns'][number]) => turn.blocks)
            .find(
                (block: ConversationDocument['turns'][number]['blocks'][number]) =>
                    block.id === 'block:retained-context:reference',
            );
        if (referenceBlock?.type !== 'image') throw new Error('Missing retained reference image block');
        referenceBlock.caption = 'Reference image caption.';
        const publish = vi.fn(async () => {
            expect(edit).not.toHaveBeenCalled();
        });
        const options: CanonicalExecutionContextInputOptions = {
            ...contextRuntime('retained-context', document),
            on_canonical_request_prepared: publish,
        };

        expect(await driver.supportsCanonicalContextExecution(options)).toBe(true);
        const first = await driver.executeCanonicalContext(options);
        expect(generate).not.toHaveBeenCalled();
        expect(edit).toHaveBeenCalledOnce();
        const native = edit.mock.calls[0]?.[0] as OpenAI.Images.ImageEditParamsNonStreaming;
        expect(native.prompt).toBe('Edit the retained image.\nReference image caption.');
        expect(Array.isArray(native.image) ? native.image : [native.image]).toHaveLength(1);
        expect(native.mask).toBeInstanceOf(File);
        expect(publish).toHaveBeenCalledOnce();
        expect(imageFetch).not.toHaveBeenCalled();
        expect(first.accepted_output.turn.blocks).toEqual([expect.objectContaining({ type: 'image' })]);

        const persisted = JSON.parse(JSON.stringify(first.conversation));
        const retryOptions = contextRuntime('retained-context', persisted);
        retryOptions.conversation_runtime.attempt_id = 'attempt:retained-context:retry';
        const retry = await driver.executeCanonicalContext(retryOptions);
        expect(retry.accepted_output).toEqual(first.accepted_output);
        expect(isCanonicalAcceptedRecovery(retry)).toBe(true);
        expect(edit).toHaveBeenCalledOnce();
        expect(imageFetch).not.toHaveBeenCalled();

        const typed = await driver.streamCanonicalContextEvents(retryOptions, undefined, {
            stream_id: 'stream:retained-context:retry',
        });
        const events = [];
        for await (const event of typed) events.push(event);
        expect(events).toEqual([expect.objectContaining({ type: 'response_accepted', origin: 'accepted_recovery' })]);
        expect(typed.completion?.accepted_output).toEqual(first.accepted_output);
        expect(edit).toHaveBeenCalledOnce();
        await typed.closed;

        await expect(driver.executeCanonicalContext({ ...retryOptions, model: 'dall-e-3' })).rejects.toThrow(
            /incompatible request identity/,
        );
        expect(edit).toHaveBeenCalledOnce();

        const alternateEdit = vi.fn();
        const alternate = new AlternateProviderImageDriver(vi.fn(), imageFetch, alternateEdit);
        await expect(alternate.executeCanonicalContext(retryOptions)).rejects.toThrow(/incompatible request identity/);
        expect(alternateEdit).not.toHaveBeenCalled();

        const successorOptions = contextRuntime('retained-context-successor', first.conversation);
        successorOptions.model_options = { _option_id: 'openai-gpt-image', width: 1536, height: 1024 };
        const successor = await driver.executeCanonicalContext(successorOptions);
        expect(isCanonicalAcceptedRecovery(successor)).toBe(false);
        expect(edit).toHaveBeenCalledTimes(2);
        const successorGeneration = successor.conversation.generations[successor.accepted_output.generation.id];
        if (successorGeneration?.request_receipt === undefined) throw new Error('Missing successor request receipt');
        expect(successorGeneration.request_receipt.target.options).toMatchObject({
            size: '1536x1024',
        });

        const changed = JSON.parse(JSON.stringify(first.conversation));
        const changedMask = Object.values(changed.assets).find(
            (asset): asset is Asset =>
                typeof asset === 'object' &&
                asset !== null &&
                'id' in asset &&
                asset.id === 'asset:retained-context:mask',
        );
        if (changedMask?.storage.type !== 'inline_base64') throw new Error('Missing retained mask image');
        const changedMaskBytes = Buffer.from(encodedImage(pngSignature, 'changed mask bytes'), 'base64');
        changedMask.storage.data = changedMaskBytes.toString('base64');
        changedMask.byte_length = changedMaskBytes.byteLength;
        changedMask.content_hash = await hash(changedMaskBytes);
        await expect(driver.executeCanonicalContext({ ...retryOptions, conversation: changed })).rejects.toThrow(
            'Conversation document validation failed',
        );
        expect(edit).toHaveBeenCalledTimes(2);

        const changedCaption = JSON.parse(JSON.stringify(first.conversation));
        const changedCaptionBlock = changedCaption.turns
            .flatMap((turn: ConversationDocument['turns'][number]) => turn.blocks)
            .find(
                (block: ConversationDocument['turns'][number]['blocks'][number]) =>
                    block.id === 'block:retained-context:reference',
            );
        if (changedCaptionBlock?.type !== 'image') throw new Error('Missing retained caption block');
        changedCaptionBlock.caption = 'Changed reference caption.';
        await expect(driver.executeCanonicalContext({ ...retryOptions, conversation: changedCaption })).rejects.toThrow(
            /retained request receipt/,
        );
        expect(edit).toHaveBeenCalledTimes(2);
    });

    it('hydrates a generated external image after publication and omits proven revised-prompt provenance', async () => {
        const sourceBytes = new Uint8Array(Buffer.from(pngA, 'base64'));
        const sourceHash = await hash(sourceBytes);
        const generate = vi.fn(async () =>
            imageResponse([{ b64_json: pngA, revised_prompt: 'Provider revised provenance text' }]),
        );
        const edit = vi.fn(async (_request: unknown, _options?: { signal?: AbortSignal }) =>
            imageResponse([{ b64_json: webpA }]),
        );
        const driver = new ImageDriver(generate, vi.fn(), edit);
        const first = await driver.executeCanonical([{ role: PromptRole.user, content: 'Draw a source icon.' }], {
            ...runtime('external-source'),
            store_generated_asset: async () => ({
                storage: {
                    type: 'external',
                    resolver: 'url',
                    locator: { url: 'gs://project-bucket/runs/run-1/media/source.png' },
                },
                byte_length: sourceBytes.byteLength,
                content_hash: sourceHash,
            }),
        });
        const promptTurn = createUserTurn({
            id: 'turn:external-successor:prompt',
            authority: 'ordinary',
            blocks: [
                createTextBlock({
                    id: 'block:external-successor:prompt',
                    text: 'Make it blue.',
                    format: 'plain',
                }),
            ],
            status: 'completed',
            timestamps: { recorded_at: at },
            provenance: { type: 'received' },
            model_visibility: 'include',
        });
        const successorInput = appendConversationRecords(
            first.conversation,
            {
                turns: [promptTurn],
                context_entries: [
                    { id: 'context:external-successor:prompt', type: 'source_turn', turn_id: promptTurn.id },
                ],
            },
            {
                expected_revision: first.conversation.revision,
                operation_id: 'input:external-successor',
                payload_fingerprint: `sha256:${'b'.repeat(64)}`,
                recorded_at: at,
            },
        ).document;
        const publish = vi.fn(async () => {
            expect(edit).not.toHaveBeenCalled();
            expect(resolve).not.toHaveBeenCalled();
        });
        const resolve = vi.fn<ResolveConversationAsset>(async function* (asset) {
            expect(asset.storage).toEqual({
                type: 'external',
                resolver: 'url',
                locator: { url: 'gs://project-bucket/runs/run-1/media/source.png' },
            });
            yield sourceBytes;
        });
        const successorOptions: CanonicalExecutionContextInputOptions = {
            ...contextRuntime('external-successor', successorInput),
            on_canonical_request_prepared: publish,
            resolve_canonical_asset: resolve,
        };

        const successor = await driver.executeCanonicalContext(successorOptions);
        expect(publish).toHaveBeenCalledOnce();
        expect(resolve).toHaveBeenCalledOnce();
        expect(edit).toHaveBeenCalledOnce();
        const request = edit.mock.calls[0]?.[0] as OpenAI.Images.ImageEditParamsNonStreaming;
        expect(request.prompt).toBe('Draw a source icon.\nMake it blue.');
        expect(request.prompt).not.toContain('Provider revised provenance text');
        const inputImage = Array.isArray(request.image) ? request.image[0] : request.image;
        expect(new Uint8Array(await (inputImage as Blob).arrayBuffer())).toEqual(sourceBytes);

        const persisted = JSON.parse(JSON.stringify(successor.conversation));
        const rejectedResolve = vi.fn<ResolveConversationAsset>(() => {
            throw new Error('accepted recovery must not hydrate');
        });
        const retry = await driver.executeCanonicalContext({
            ...contextRuntime('external-successor', persisted),
            resolve_canonical_asset: rejectedResolve,
            on_canonical_request_prepared: async () => {
                throw new Error('accepted recovery must not republish');
            },
        });
        expect(isCanonicalAcceptedRecovery(retry)).toBe(true);
        expect(rejectedResolve).not.toHaveBeenCalled();
        expect(edit).toHaveBeenCalledOnce();

        const unproven = parseConversationDocument(JSON.parse(JSON.stringify(successorInput)));
        const generatedAsset = Object.values(unproven.assets).find(
            (asset): asset is Asset => asset.provenance.type === 'generated',
        );
        if (generatedAsset === undefined) throw new Error('Missing generated image asset');
        generatedAsset.metadata = { openai_image: { revised_prompt: 'Different metadata text' } };
        await expect(
            driver.executeCanonicalContext({
                ...contextRuntime('external-unproven', unproven),
                resolve_canonical_asset: resolve,
                on_canonical_request_prepared: publish,
            }),
        ).rejects.toThrow(/cannot preserve image block .* caption/);
        expect(resolve).toHaveBeenCalledOnce();
        expect(edit).toHaveBeenCalledOnce();
    });

    it('isolates concurrent host asset capabilities and gives each precedence over caller options', async () => {
        const firstBytes = new Uint8Array(Buffer.from(pngA, 'base64'));
        const secondBytes = new Uint8Array(Buffer.from(encodedImage(pngSignature, 'other image'), 'base64'));
        const firstDocument = await retainedEditDocument('host-capability-first');
        const secondDocument = await retainedEditDocument('host-capability-second');
        for (const [document, bytes] of [
            [firstDocument, firstBytes],
            [secondDocument, secondBytes],
        ] as const) {
            const asset = Object.values(document.assets)[0];
            if (asset === undefined) throw new Error('Missing retained image asset');
            asset.storage = {
                type: 'external',
                resolver: 'url',
                locator: {
                    url: `gs://project-bucket/runs/run-1/media/${asset.id}.png`,
                },
            };
            asset.byte_length = bytes.byteLength;
            asset.content_hash = await hash(bytes);
        }
        const edit = vi.fn(async (_request: unknown) => imageResponse([{ b64_json: webpA }]));
        const driver = new ImageDriver(vi.fn(), vi.fn(), edit);
        const callerResolver = vi.fn<ResolveConversationAsset>(() => {
            throw new Error('Caller resolver must not override the host capability');
        });
        const firstResolver = vi.fn<ResolveConversationAsset>(async function* (asset) {
            expect(asset.id).toBe('asset:host-capability-first:reference');
            yield firstBytes;
        });
        const secondResolver = vi.fn<ResolveConversationAsset>(async function* (asset) {
            expect(asset.id).toBe('asset:host-capability-second:reference');
            yield secondBytes;
        });
        const sharedCapability = { resolve_canonical_asset: firstResolver };
        let announcePrepared!: () => void;
        let releasePrepared!: () => void;
        const preparedEntered = new Promise<void>((resolve) => {
            announcePrepared = resolve;
        });
        const preparedProceed = new Promise<void>((resolve) => {
            releasePrepared = resolve;
        });
        const first = driver.executeCanonicalContext(
            {
                ...contextRuntime('host-capability-first', firstDocument),
                resolve_canonical_asset: callerResolver,
                on_canonical_request_prepared: async () => {
                    announcePrepared();
                    await preparedProceed;
                },
            },
            undefined,
            sharedCapability,
        );
        await preparedEntered;
        sharedCapability.resolve_canonical_asset = secondResolver;
        const second = driver.executeCanonicalContext(
            {
                ...contextRuntime('host-capability-second', secondDocument),
                resolve_canonical_asset: callerResolver,
            },
            undefined,
            sharedCapability,
        );
        releasePrepared();
        await Promise.all([first, second]);

        expect(callerResolver).not.toHaveBeenCalled();
        expect(firstResolver).toHaveBeenCalledOnce();
        expect(secondResolver).toHaveBeenCalledOnce();
        expect(edit).toHaveBeenCalledTimes(2);
        const received: string[] = [];
        for (const [request] of edit.mock.calls) {
            const image = (request as OpenAI.Images.ImageEditParamsNonStreaming).image;
            const file = Array.isArray(image) ? image[0] : image;
            received.push(Buffer.from(await (file as Blob).arrayBuffer()).toString('hex'));
        }
        expect(received.sort()).toEqual(
            [firstBytes, secondBytes].map((bytes) => Buffer.from(bytes).toString('hex')).sort(),
        );
    });

    it('rejects unsupported selection and invalid retained bytes before provider transport', async () => {
        const edit = vi.fn(async () => imageResponse([{ b64_json: webpA }]));
        const driver = new ImageDriver(vi.fn(), vi.fn(), edit);
        const selected = JSON.parse(JSON.stringify(await retainedEditDocument('selected-context')));
        const selectedBlock = selected.turns[0]?.blocks.find(
            (block: ConversationDocument['turns'][number]['blocks'][number]) => block.type === 'image',
        );
        if (selectedBlock?.type !== 'image') throw new Error('Missing selected image block');
        selectedBlock.selection = {
            type: 'image_region',
            coordinate_space: 'normalized',
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        };
        const selectedPublish = vi.fn();
        const selectedResolve = vi.fn<ResolveConversationAsset>();
        await expect(
            driver.executeCanonicalContext({
                ...contextRuntime('selected-context', selected),
                on_canonical_request_prepared: selectedPublish,
                resolve_canonical_asset: selectedResolve,
            }),
        ).rejects.toThrow(/cannot preserve image block .* selection/);
        expect(selectedPublish).not.toHaveBeenCalled();
        expect(selectedResolve).not.toHaveBeenCalled();
        expect(edit).not.toHaveBeenCalled();

        const external = JSON.parse(JSON.stringify(await retainedEditDocument('invalid-bytes')));
        const externalAsset = Object.values(external.assets)[0] as Asset;
        externalAsset.storage = {
            type: 'external',
            resolver: 'url',
            locator: { url: 'gs://project-bucket/runs/run-1/media/reference.png' },
        };
        const publish = vi.fn();
        const declaredLength = externalAsset.byte_length;
        if (declaredLength === undefined) throw new Error('Missing retained image byte length');
        const resolve = vi.fn<ResolveConversationAsset>(async function* () {
            yield new Uint8Array(declaredLength).fill(9);
        });
        await expect(
            driver.executeCanonicalContext({
                ...contextRuntime('invalid-bytes', external),
                on_canonical_request_prepared: publish,
                resolve_canonical_asset: resolve,
            }),
        ).rejects.toThrow(/content hash does not match resolved bytes/);
        expect(publish).toHaveBeenCalledOnce();
        expect(resolve).toHaveBeenCalledOnce();
        expect(edit).not.toHaveBeenCalled();

        const aggregateBase = createConversationDocument({ id: 'conversation:aggregate-input', created_at: at });
        const aggregateTurn = createUserTurn({
            id: 'turn:aggregate-input',
            authority: 'ordinary',
            blocks: [
                createTextBlock({ id: 'block:aggregate-text', text: 'Edit both images.', format: 'plain' }),
                { id: 'block:aggregate-image-a', type: 'image', asset_id: 'asset:aggregate-a' },
                { id: 'block:aggregate-image-b', type: 'image', asset_id: 'asset:aggregate-b' },
            ],
            status: 'completed',
            timestamps: { recorded_at: at },
            provenance: { type: 'received' },
            model_visibility: 'include',
        });
        const aggregate = appendConversationRecords(
            aggregateBase,
            {
                turns: [aggregateTurn],
                assets: ['a', 'b'].map(
                    (suffix): Asset => ({
                        id: `asset:aggregate-${suffix}`,
                        kind: 'image',
                        mime_type: 'image/png',
                        storage: {
                            type: 'external',
                            resolver: 'url',
                            locator: { url: `gs://project-bucket/runs/run-1/media/${suffix}.png` },
                        },
                        provenance: { type: 'received', source_turn_id: aggregateTurn.id },
                        byte_length: 30 * 1024 * 1024,
                        content_hash: `sha256:${suffix.repeat(64)}`,
                        created_at: at,
                    }),
                ),
                context_entries: [{ id: 'context:aggregate-input', type: 'source_turn', turn_id: aggregateTurn.id }],
            },
            {
                expected_revision: aggregateBase.revision,
                operation_id: 'materialized:aggregate-input',
                payload_fingerprint: `sha256:${'c'.repeat(64)}`,
                recorded_at: at,
            },
        ).document;
        const aggregatePublish = vi.fn();
        const aggregateResolve = vi.fn<ResolveConversationAsset>();
        await expect(
            driver.executeCanonicalContext({
                ...contextRuntime('aggregate-input', aggregate),
                on_canonical_request_prepared: aggregatePublish,
                resolve_canonical_asset: aggregateResolve,
            }),
        ).rejects.toThrow(/52428800 byte aggregate limit/);
        expect(aggregatePublish).toHaveBeenCalledOnce();
        expect(aggregateResolve).not.toHaveBeenCalled();
        expect(edit).not.toHaveBeenCalled();
    });

    it('cancels an unsuccessful canonical input download before provider transport', async () => {
        const document = await retainedEditDocument('http-failure');
        const asset = Object.values(document.assets)[0];
        asset.storage = {
            type: 'external',
            resolver: 'url',
            locator: { url: 'https://images.openai.test/failed-reference' },
        };
        let cancelled = false;
        const body = new ReadableStream<Uint8Array>({
            cancel() {
                cancelled = true;
            },
        });
        const fetcher = vi.fn<typeof fetch>(async () => new Response(body, { status: 403 }));
        const edit = vi.fn();
        const driver = new ImageDriver(vi.fn(), fetcher, edit);
        const publish = vi.fn();

        await expect(
            driver.executeCanonicalContext({
                ...contextRuntime('http-failure', document),
                on_canonical_request_prepared: publish,
            }),
        ).rejects.toThrow(/HTTP 403/);
        expect(publish).toHaveBeenCalledOnce();
        expect(fetcher).toHaveBeenCalledOnce();
        expect(cancelled).toBe(true);
        expect(body.locked).toBe(false);
        expect(edit).not.toHaveBeenCalled();
    });

    it('aborts a pending canonical input read before provider transport', async () => {
        const document = await retainedEditDocument('input-abort');
        const asset = Object.values(document.assets)[0];
        asset.storage = {
            type: 'external',
            resolver: 'url',
            locator: { url: 'https://images.openai.test/pending-reference' },
        };
        let cancelled = false;
        const body = new ReadableStream<Uint8Array>({
            cancel() {
                cancelled = true;
            },
        });
        const fetcher = vi.fn<typeof fetch>(async () => new Response(body, { status: 200 }));
        const edit = vi.fn();
        const driver = new ImageDriver(vi.fn(), fetcher, edit);
        const publish = vi.fn();
        const controller = new AbortController();
        const execution = driver.executeCanonicalContext(
            { ...contextRuntime('input-abort', document), on_canonical_request_prepared: publish },
            controller.signal,
        );
        await vi.waitFor(() => expect(fetcher).toHaveBeenCalledOnce());
        controller.abort(new Error('cancel input hydration'));

        await expect(execution).rejects.toThrow('cancel input hydration');
        expect(publish).toHaveBeenCalledOnce();
        expect(cancelled).toBe(true);
        expect(body.locked).toBe(false);
        expect(edit).not.toHaveBeenCalled();
    });

    it('edits ordered references with a mask and recovers without re-reading media or calling the provider', async () => {
        const generate = vi.fn();
        const edit = vi.fn(async (_request: unknown, _options?: { signal?: AbortSignal }) =>
            imageResponse([{ b64_json: webpA }]),
        );
        const imageFetch = vi.fn<typeof fetch>(async (input) => {
            const value = String(input);
            const encoded = value.slice(value.indexOf(',') + 1);
            return new Response(Buffer.from(encoded, 'base64'), {
                status: 200,
                headers: { 'content-type': 'image/png' },
            });
        });
        const publish = vi.fn(async () => {
            expect(generate).not.toHaveBeenCalled();
            expect(edit).not.toHaveBeenCalled();
            expect(imageFetch).not.toHaveBeenCalled();
        });
        const driver = new ImageDriver(generate, imageFetch, edit);
        const firstReference = encodedImage(pngSignature, 'first reference');
        const secondReference = encodedImage(pngSignature, 'second reference');
        const mask = encodedImage(pngSignature, 'mask');
        const segments = [
            {
                role: PromptRole.user,
                content: 'Replace the background.',
                files: [
                    new Base64DataSource('first.png', 'image/png', firstReference),
                    new Base64DataSource('second.png', 'image/png', secondReference),
                ],
            },
            {
                role: PromptRole.mask,
                content: '',
                files: [new Base64DataSource('mask.png', 'image/png', mask)],
            },
        ];
        const options: CanonicalExecutionInputOptions = {
            ...runtime('edit'),
            model_options: {
                _option_id: 'openai-gpt-image',
                width: 1536,
                height: 1024,
                image_quality: 'high',
                background: 'transparent',
                output_format: 'webp',
                output_compression: 40,
                partial_images: 2,
                input_fidelity: 'high',
                n: 1,
            },
            on_canonical_request_prepared: publish,
        };

        const first = await driver.executeCanonical(segments, options);
        expect(publish).toHaveBeenCalledOnce();
        expect(generate).not.toHaveBeenCalled();
        expect(edit).toHaveBeenCalledOnce();
        expect(imageFetch).toHaveBeenCalledTimes(3);
        const editRequest = edit.mock.calls[0][0] as OpenAI.Images.ImageEditParamsNonStreaming;
        expect(editRequest).toMatchObject({
            model: 'gpt-image-1',
            prompt: 'Replace the background.',
            size: '1536x1024',
            n: 1,
            quality: 'high',
            background: 'transparent',
            output_format: 'webp',
            output_compression: 40,
            partial_images: 2,
            input_fidelity: 'high',
        });
        const editImages = Array.isArray(editRequest.image) ? editRequest.image : [editRequest.image];
        expect(editImages).toHaveLength(2);
        expect(Buffer.from(await (editImages[0] as Blob).arrayBuffer()).toString('base64')).toBe(firstReference);
        expect(Buffer.from(await (editImages[1] as Blob).arrayBuffer()).toString('base64')).toBe(secondReference);
        expect(Buffer.from(await (editRequest.mask as Blob).arrayBuffer()).toString('base64')).toBe(mask);

        const serialized = JSON.stringify(first.conversation);
        expect(serialized.split(mask)).toHaveLength(2);
        const reloaded = parseConversationDocument(JSON.parse(serialized));
        const maskBinding = openAIImageEditMaskBinding(reloaded);
        expect(maskBinding).toMatchObject({
            asset: {
                kind: 'image',
                mime_type: 'image/png',
                storage: { type: 'inline_base64', data: mask },
                provenance: { type: 'received' },
            },
        });
        if (maskBinding === undefined) throw new Error('Expected a retained OpenAI image edit mask binding');
        const maskTurn = reloaded.turns.find((turn) => turn.id === maskBinding.turn_id);
        expect(maskTurn?.blocks).toEqual([
            expect.objectContaining({
                id: maskBinding.binding_block_id,
                type: 'extension',
                namespace: 'openai.images.edit_mask',
                version: '1',
                payload: { image_block_id: maskBinding.image_block_id },
                model_projection: 'registered',
            }),
            expect.objectContaining({
                id: maskBinding.image_block_id,
                type: 'image',
                asset_id: maskBinding.asset.id,
            }),
        ]);
        expect(reloaded.context.entries).toContainEqual(expect.objectContaining({ turn_id: maskBinding.turn_id }));
        const generation = reloaded.generations[first.accepted_output.generation.id];
        expect(generation?.request_receipt?.asset_versions).toContainEqual({
            asset_id: maskBinding.asset.id,
            content_hash: await hash(Buffer.from(mask, 'base64')),
        });

        const unsupportedV2Mask = JSON.parse(JSON.stringify(first.conversation));
        const unsupportedV2Generation = unsupportedV2Mask.generations[first.accepted_output.generation.id];
        unsupportedV2Generation.adapter_version = '2026-09-30.canonical.2';
        unsupportedV2Generation.request_receipt.target.adapter_version = '2026-09-30.canonical.2';
        await expect(
            driver.executeCanonical(segments, { ...options, conversation: unsupportedV2Mask }),
        ).rejects.toThrow(/unsupported adapter version 2026-09-30.canonical.2/);
        expect(publish).toHaveBeenCalledOnce();
        expect(edit).toHaveBeenCalledOnce();
        expect(imageFetch).toHaveBeenCalledTimes(3);

        const retryPrepared = vi.fn();
        const retry = await driver.executeCanonical(segments, {
            ...options,
            conversation: JSON.parse(JSON.stringify(first.conversation)),
            on_canonical_request_prepared: retryPrepared,
        });
        expect(isCanonicalAcceptedRecovery(retry)).toBe(true);
        expect(retryPrepared).not.toHaveBeenCalled();
        expect(edit).toHaveBeenCalledOnce();
        expect(imageFetch).toHaveBeenCalledTimes(3);

        const changedMask = encodedImage(pngSignature, 'changed mask');
        await expect(
            driver.executeCanonical(
                [
                    segments[0],
                    {
                        role: PromptRole.mask,
                        content: '',
                        files: [new Base64DataSource('mask.png', 'image/png', changedMask)],
                    },
                ],
                {
                    ...options,
                    conversation: JSON.parse(JSON.stringify(first.conversation)),
                },
            ),
        ).rejects.toThrow('was already used with a different payload');
        expect(publish).toHaveBeenCalledOnce();
        expect(edit).toHaveBeenCalledOnce();
        expect(imageFetch).toHaveBeenCalledTimes(3);

        await expect(
            driver.executeCanonical(segments, {
                ...options,
                conversation: JSON.parse(JSON.stringify(first.conversation)),
                model_options: { ...options.model_options, input_fidelity: 'low' },
            }),
        ).rejects.toThrow('incompatible request identity');
        expect(edit).toHaveBeenCalledOnce();
        expect(imageFetch).toHaveBeenCalledTimes(3);
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
            const options: CanonicalExecutionInputOptions = {
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
        const options: CanonicalExecutionInputOptions = {
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
                                name: 'input.mp3',
                                mime_type: 'audio/mpeg',
                                getStream,
                                getURL: vi.fn(),
                                getURI: vi.fn(),
                            },
                        ],
                    },
                ],
                runtime('file'),
            ),
        ).rejects.toThrow('does not support audio/mpeg input files');
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
