import type { ConversationDocument, ConversationStreamEvent } from '@llumiverse/conversation';
import {
    type CanonicalExecutionEventStream,
    type ExecutionOptions,
    PromptRole,
    type PromptSegment,
} from '@llumiverse/core';
import 'dotenv/config';
import { describe, expect, test } from 'vitest';
import { xAIDriver } from '../src/index.js';
import { selectLiveTestDrivers } from './live-model-selection.js';

const TIMEOUT = 180_000;
const MODEL = 'grok-imagine-image-2.0';
const IMAGE_OUTPUT_MODALITY = 'image' as NonNullable<ExecutionOptions['output_modality']>;
const PROMPT: PromptSegment[] = [
    {
        role: PromptRole.user,
        content: 'A small blue circle centered on a plain white background, flat icon.',
    },
];

interface XAIImageLiveDriver {
    name: string;
    models: string[];
    setup(): {
        driver: xAIDriver;
        invokeCalls(): number;
        restore(): void;
    };
}

function countMethodCalls(target: object, key: PropertyKey): { calls(): number; restore(): void } {
    const original = Reflect.get(target, key) as unknown;
    if (typeof original !== 'function') throw new TypeError(`Expected ${String(key)} to be callable`);
    const ownDescriptor = Object.getOwnPropertyDescriptor(target, key);
    let count = 0;
    Object.defineProperty(target, key, {
        configurable: true,
        writable: true,
        value: function countedMethod(this: unknown, ...args: unknown[]) {
            count += 1;
            return Reflect.apply(original, this, args);
        },
    });
    return {
        calls: () => count,
        restore: () => {
            if (ownDescriptor === undefined) Reflect.deleteProperty(target, key);
            else Object.defineProperty(target, key, ownDescriptor);
        },
    };
}

const liveDrivers: XAIImageLiveDriver[] = [
    {
        name: 'xai',
        models: [MODEL],
        setup: () => {
            const apiKey = process.env.XAI_API_KEY;
            if (!apiKey) {
                throw new Error('Canonical xAI image live coverage requires XAI_API_KEY');
            }
            const driver = new xAIDriver({ apiKey });
            const invoke = countMethodCalls(driver.xai_service, 'post');
            return { driver, invokeCalls: invoke.calls, restore: invoke.restore };
        },
    },
];

const selectedDrivers = selectLiveTestDrivers(liveDrivers, {
    providers: process.env.LLUMIVERSE_LIVE_PROVIDERS,
    models: process.env.LLUMIVERSE_LIVE_MODELS,
});

function options(model: string, attempt: string, conversation?: ConversationDocument): ExecutionOptions {
    return {
        model,
        ...(conversation === undefined ? {} : { conversation }),
        output_modality: IMAGE_OUTPUT_MODALITY,
        model_options: {
            _option_id: 'xai-grok-image',
            aspect_ratio: '1:1',
            resolution: '1k',
            quality: 'low',
            response_format: 'b64_json',
            n: 1,
        },
        conversation_runtime: {
            conversation_id: 'live:canonical:xai-image',
            request_id: 'live:canonical:xai-image:request',
            attempt_id: `live:canonical:xai-image:attempt:${attempt}`,
            input_operation_id: 'live:canonical:xai-image:input',
            response_operation_id: 'live:canonical:xai-image:response',
            recorded_at: '2026-09-30T00:00:00.000Z',
            started_at: '2026-09-30T00:00:00.000Z',
            completed_at: '2026-09-30T00:00:00.000Z',
        },
    };
}

async function collect(stream: CanonicalExecutionEventStream): Promise<ConversationStreamEvent[]> {
    const events: ConversationStreamEvent[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

async function contentHash(bytes: Uint8Array): Promise<string> {
    const owned = new Uint8Array(bytes.byteLength);
    owned.set(bytes);
    const digest = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', owned.buffer));
    return `sha256:${Array.from(digest, (byte) => byte.toString(16).padStart(2, '0')).join('')}`;
}

describe.each(selectedDrivers)('Canonical xAI image live execution', (live) => {
    test.each(live.models)(
        'persists verified output and exact-retries through typed streaming without another invocation for %s',
        { timeout: TIMEOUT, retry: 1 },
        async (model) => {
            const transport = live.setup();
            let firstPublishCount = 0;
            let retryPublishCount = 0;
            let retryStream: CanonicalExecutionEventStream | undefined;
            try {
                const first = await transport.driver.executeCanonical(PROMPT, {
                    ...options(model, 'first'),
                    on_canonical_request_prepared: async () => {
                        firstPublishCount += 1;
                    },
                });

                expect(firstPublishCount).toBe(1);
                expect(transport.invokeCalls()).toBe(1);
                expect(first.accepted_output.generation).toMatchObject({
                    protocol: 'xai.images',
                    requested_model: model,
                    resolved_model: model,
                    status: 'completed',
                });
                expect(first.accepted_output.generation).not.toHaveProperty('request_receipt');
                expect(first.conversation.generations[first.accepted_output.generation.id]).toMatchObject({
                    request_receipt: {
                        target: {
                            provider: 'xai',
                            protocol: 'xai.images',
                            model,
                        },
                    },
                });
                const assets = Object.values(first.accepted_output.assets);
                expect(assets).toHaveLength(1);
                const asset = assets[0];
                expect(asset).toMatchObject({
                    kind: 'image',
                    mime_type: expect.stringMatching(/^image\/(jpeg|png|webp)$/),
                    byte_length: expect.any(Number),
                    content_hash: expect.stringMatching(/^sha256:[0-9a-f]{64}$/),
                    storage: { type: 'inline_base64', data: expect.any(String) },
                });
                if (asset.storage.type !== 'inline_base64') throw new Error('Expected inline xAI image asset');
                const imageBytes = new Uint8Array(Buffer.from(asset.storage.data, 'base64'));
                expect(imageBytes.byteLength).toBeGreaterThan(0);
                expect(imageBytes).toHaveLength(asset.byte_length ?? -1);
                expect(asset.content_hash).toBe(await contentHash(imageBytes));

                const persisted = JSON.parse(JSON.stringify(first.conversation)) as ConversationDocument;
                retryStream = await transport.driver.streamCanonicalEvents(
                    PROMPT,
                    {
                        ...options(model, 'retry', persisted),
                        on_canonical_request_prepared: async () => {
                            retryPublishCount += 1;
                        },
                    },
                    undefined,
                    { stream_id: 'live:canonical:xai-image:retry' },
                );
                expect(await collect(retryStream)).toEqual([
                    expect.objectContaining({ type: 'response_accepted', sequence: 0, origin: 'accepted_recovery' }),
                ]);
                expect(retryStream.completion?.accepted_output).toEqual(first.accepted_output);
                expect(retryPublishCount).toBe(0);
                expect(transport.invokeCalls()).toBe(1);
            } finally {
                if (retryStream !== undefined) {
                    try {
                        await retryStream.cancel();
                    } finally {
                        await retryStream.closed;
                    }
                }
                transport.driver.destroy();
                transport.restore();
            }
        },
    );
});
