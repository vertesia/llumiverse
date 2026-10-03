import {
    appendConversationRecords,
    type ConversationPreparedRequest,
    createConversationDocument,
    createTextBlock,
    createToolTurn,
    createUserTurn,
    fingerprintJson,
    hashContentBytes,
    type ResolveConversationAsset,
} from '@llumiverse/conversation';
import { type ExecutionOptions, PromptRole, resolveCanonicalExecutionContextOptions } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { AnthropicDriver } from '../anthropic/index.js';
import {
    bedrockConverseJsonValue,
    prepareBedrockConverseCanonicalContext,
} from '../bedrock/bedrock-converse-conversation-adapter.js';
import { BedrockDriver } from '../bedrock/index.js';
import { BedrockMantleDriver } from '../bedrock-mantle/index.js';
import { prepareClaudeCanonicalContext } from '../shared/claude-messages-conversation-adapter.js';
import { VertexAIDriver } from '../vertexai/index.js';
import { providerJsonValue } from './canonical-runtime.js';

const at = '2026-10-03T01:00:00.000Z';
const png = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg==',
    'base64',
);

vi.mock('@aws/bedrock-token-generator', () => ({
    getTokenProvider: vi.fn(() => async () => 'bedrock-api-key-test'),
}));

async function document(nested = false) {
    const initial = createConversationDocument({ id: 'conversation:concrete-image', created_at: at });
    const asset = {
        id: 'asset:concrete-image',
        kind: 'image' as const,
        mime_type: 'image/png',
        storage: {
            type: 'external' as const,
            resolver: 'vertesia.agent_artifact',
            locator: { storage_id: 'owner', artifact_path: 'image' },
        },
        provenance: { type: 'received' as const },
        created_at: at,
        ...(await hashContentBytes(png)),
    };
    const user = createUserTurn({
        id: 'turn:user',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: at },
        provenance: { type: 'received' },
        model_visibility: 'include',
        blocks: [
            createTextBlock({ id: 'block:text', text: 'Read this image.', format: 'plain' }),
            ...(!nested ? [{ id: 'block:image', type: 'image' as const, asset_id: asset.id }] : []),
        ],
    });
    const call = {
        id: 'turn:call',
        kind: 'agent' as const,
        authority: 'ordinary' as const,
        status: 'completed' as const,
        timestamps: { recorded_at: at },
        provenance: { type: 'received' as const },
        model_visibility: 'include' as const,
        blocks: [
            {
                id: 'block:call',
                type: 'tool_call' as const,
                call_id: 'call:image',
                tool_name: 'inspect',
                executor: 'application' as const,
                arguments: { type: 'json' as const, value: {} },
            },
        ],
    };
    const result = createToolTurn({
        id: 'turn:result',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: at },
        provenance: { type: 'received' },
        model_visibility: 'include',
        blocks: [
            {
                id: 'block:result',
                type: 'tool_result',
                call_id: 'call:image',
                status: 'success',
                content: [{ id: 'block:nested-image', type: 'image', asset_id: asset.id }],
            },
        ],
    });
    return appendConversationRecords(
        initial,
        {
            turns: nested ? [user, call, result] : [user],
            assets: [asset],
            ...(nested
                ? {
                      tool_definitions: [
                          {
                              id: 'tool-definition:inspect:v1',
                              name: 'inspect',
                              version: 'v1',
                              input_schema: { type: 'object' },
                              result_capabilities: ['image' as const],
                          },
                      ],
                      active_tool_definition_ids: ['tool-definition:inspect:v1'],
                  }
                : {}),
            context_entries: [
                { id: 'entry:user', type: 'source_turn', turn_id: user.id },
                ...(nested
                    ? [
                          { id: 'entry:call', type: 'source_turn' as const, turn_id: call.id },
                          { id: 'entry:result', type: 'source_turn' as const, turn_id: result.id },
                      ]
                    : []),
            ],
        },
        {
            expected_revision: initial.revision,
            operation_id: 'append:concrete-image',
            payload_fingerprint: await fingerprintJson({ nested }),
            recorded_at: at,
        },
    ).document;
}

function options(conversation: Awaited<ReturnType<typeof document>>, model: string) {
    return resolveCanonicalExecutionContextOptions({
        model,
        conversation,
        conversation_runtime: {
            conversation_id: conversation.id,
            request_id: 'request:concrete-image',
            attempt_id: 'attempt:concrete-image',
            input_operation_id: 'input:concrete-image',
            response_operation_id: 'response:concrete-image',
            recorded_at: at,
        },
    });
}

describe('concrete driver canonical host image ownership', () => {
    it('forwards owned image resolution through authored executeCanonical for the three native protocols', async () => {
        const source = await document();
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        const prepared = vi.fn<NonNullable<ExecutionOptions['on_canonical_request_prepared']>>(async () => {
            throw new Error('Prepared publication gate');
        });
        const targets = [
            { driver: new AnthropicDriver({ apiKey: 'test-only' }), model: 'claude-sonnet-4-20250514' },
            {
                driver: new VertexAIDriver({ project: 'test-project', region: 'global', geminiContextCache: false }),
                model: 'publishers/google/models/gemini-2.5-pro',
            },
            { driver: new BedrockDriver({ region: 'us-east-1' }), model: 'anthropic.claude-sonnet-4-6-v1:0' },
        ];
        for (const { driver, model } of targets) {
            await expect(
                driver.executeCanonical(
                    [{ role: PromptRole.user, content: 'Describe the retained image.' }],
                    { ...options(source, model), on_canonical_request_prepared: prepared },
                    undefined,
                    { resolve_canonical_asset: resolver },
                ),
            ).rejects.toThrow('Prepared publication gate');
        }
        expect(resolver).toHaveBeenCalledTimes(3);
        expect(prepared).toHaveBeenCalledTimes(3);
        for (const [published] of prepared.mock.calls) {
            expect(published.document.assets['asset:concrete-image']?.storage.type).toBe('external');
            expect(published.record.request_receipt.request_fingerprint).toMatch(/^sha256:/);
        }
    });

    it('forwards the same owned resolver through Vertex Claude and Bedrock Mantle delegation', async () => {
        const source = await document();
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        const prepared = vi.fn<NonNullable<ExecutionOptions['on_canonical_request_prepared']>>(async () => {
            throw new Error('Prepared publication gate');
        });
        const vertex = new VertexAIDriver({ project: 'test-project', region: 'global' });
        Object.defineProperty(vertex, 'getAnthropicClient', {
            value: async () => ({
                messages: {
                    stream: () => {
                        throw new Error('Unexpected transport');
                    },
                },
            }),
        });
        await expect(
            vertex.executeCanonical(
                [{ role: PromptRole.user, content: 'Continue.' }],
                {
                    ...options(source, 'publishers/anthropic/models/claude-sonnet-4-5'),
                    on_canonical_request_prepared: prepared,
                },
                undefined,
                { resolve_canonical_asset: resolver },
            ),
        ).rejects.toThrow('Prepared publication gate');
        const mantle = new BedrockMantleDriver({ region: 'us-east-1' });
        await expect(
            mantle.executeCanonicalContext(
                {
                    ...options(source, 'xai.grok-4.3'),
                    on_canonical_request_prepared: prepared,
                },
                undefined,
                { resolve_canonical_asset: resolver },
            ),
        ).rejects.toThrow('Prepared publication gate');
        expect(resolver).toHaveBeenCalledTimes(2);
        expect(prepared).toHaveBeenCalledTimes(2);
    });

    it('fingerprints the exact hydrated Claude body sent to the concrete transport', async () => {
        const source = await document();
        const driver = new AnthropicDriver({ apiKey: 'test-only' });
        const native = vi.fn((_payload: unknown) => {
            throw new Error('Transport captured');
        });
        Object.defineProperty(driver.client.messages, 'stream', { value: native });
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        let retained: ConversationPreparedRequest | undefined;
        await expect(
            driver.executeCanonicalContext(
                {
                    ...options(source, 'claude-sonnet-4-20250514'),
                    on_canonical_request_prepared: async (prepared) => {
                        retained = prepared;
                    },
                },
                undefined,
                { resolve_canonical_asset: resolver },
            ),
        ).rejects.toThrow('Transport captured');
        const sent = native.mock.calls[0]?.[0];
        if (sent === undefined || retained === undefined)
            throw new Error('Expected concrete prepared request and transport');
        expect(JSON.stringify(sent)).toContain(png.toString('base64'));
        expect(retained.record.request_receipt.request_fingerprint).toBe(
            await fingerprintJson(providerJsonValue(sent)),
        );
        expect(retained.document.assets['asset:concrete-image']?.storage.type).toBe('external');
        expect(resolver).toHaveBeenCalledTimes(1);
    });

    it('fingerprints the hydrated Gemini and Bedrock bodies at their concrete transports', async () => {
        const source = await document();
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        for (const provider of ['gemini', 'bedrock'] as const) {
            const native = vi.fn((_payload: unknown) => {
                throw new Error('Transport captured');
            });
            const driver =
                provider === 'gemini'
                    ? new VertexAIDriver({ project: 'test-project', region: 'global', geminiContextCache: false })
                    : new BedrockDriver({ region: 'us-east-1' });
            if (provider === 'gemini') {
                Object.defineProperty(driver, 'getGoogleGenAIClient', {
                    value: () => ({ models: { generateContent: native } }),
                });
            } else {
                Object.defineProperty(driver, 'getExecutor', {
                    value: () => ({ converse: native, destroy: vi.fn() }),
                });
            }
            let retained: ConversationPreparedRequest | undefined;
            const model =
                provider === 'gemini' ? 'publishers/google/models/gemini-2.5-pro' : 'anthropic.claude-sonnet-4-6-v1:0';
            await expect(
                driver.executeCanonicalContext(
                    {
                        ...options(source, model),
                        on_canonical_request_prepared: async (prepared) => {
                            retained = prepared;
                        },
                    },
                    undefined,
                    { resolve_canonical_asset: resolver },
                ),
            ).rejects.toThrow('Transport captured');
            const sent = native.mock.calls[0]?.[0];
            if (sent === undefined || retained === undefined) throw new Error('Expected prepared native request');
            expect(JSON.stringify(sent)).toContain(provider === 'gemini' ? png.toString('base64') : 'image');
            expect(retained.record.request_receipt.request_fingerprint).toBe(
                await fingerprintJson(provider === 'gemini' ? providerJsonValue(sent) : bedrockConverseJsonValue(sent)),
            );
            expect(retained.document.assets['asset:concrete-image']?.storage.type).toBe('external');
        }
        expect(resolver).toHaveBeenCalledTimes(2);
    });

    it('keeps direct-library accepted retry fail-closed when its exact native image bytes expire', async () => {
        const source = await document();
        const driver = new AnthropicDriver({ apiKey: 'test-only' });
        const transport = vi.fn((_payload: unknown) => ({
            async finalMessage() {
                return {
                    id: 'message:received-image',
                    type: 'message',
                    role: 'assistant',
                    model: 'claude-sonnet-4-20250514',
                    content: [{ type: 'text', text: 'It is a small image.', citations: null }],
                    stop_reason: 'end_turn',
                    stop_sequence: null,
                    usage: { input_tokens: 10, output_tokens: 7 },
                };
            },
            async *[Symbol.asyncIterator]() {},
            abort() {},
        }));
        Object.defineProperty(driver.client.messages, 'stream', { value: transport });
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        const first = await driver.executeCanonicalContext(options(source, 'claude-sonnet-4-20250514'), undefined, {
            resolve_canonical_asset: resolver,
        });
        expect(first.accepted_output.generation.status).toBe('completed');
        expect(transport).toHaveBeenCalledOnce();
        const expired = vi.fn<ResolveConversationAsset>(async () => {
            throw new Error('Image bytes expired');
        });
        await expect(
            driver.executeCanonicalContext(
                {
                    ...options(first.conversation, 'claude-sonnet-4-20250514'),
                    conversation: first.conversation,
                },
                undefined,
                { resolve_canonical_asset: expired },
            ),
        ).rejects.toThrow();
        expect(expired).toHaveBeenCalled();
        expect(transport).toHaveBeenCalledOnce();
    });

    it('forwards only the per-call resolver through Anthropic, Gemini and Bedrock context execution', async () => {
        const source = await document();
        const original = structuredClone(source);
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        const prepared = vi.fn<NonNullable<ExecutionOptions['on_canonical_request_prepared']>>(async () => {
            throw new Error('Prepared publication gate');
        });
        const targets = [
            { driver: new AnthropicDriver({ apiKey: 'test-only' }), model: 'claude-sonnet-4-20250514' },
            {
                driver: new VertexAIDriver({ project: 'test-project', region: 'global', geminiContextCache: false }),
                model: 'publishers/google/models/gemini-2.5-pro',
            },
            { driver: new BedrockDriver({ region: 'us-east-1' }), model: 'anthropic.claude-sonnet-4-6-v1:0' },
        ];
        for (const { driver, model } of targets) {
            await expect(
                driver.executeCanonicalContext(
                    {
                        ...options(source, model),
                        on_canonical_request_prepared: prepared,
                    },
                    undefined,
                    { resolve_canonical_asset: resolver },
                ),
            ).rejects.toThrow('Prepared publication gate');
        }
        expect(resolver).toHaveBeenCalledTimes(3);
        expect(prepared).toHaveBeenCalledTimes(3);
        for (const [published] of prepared.mock.calls) {
            expect(published.document.assets['asset:concrete-image']?.storage.type).toBe('external');
            expect(published.record.request_receipt.request_fingerprint).toMatch(/^sha256:/);
        }
        expect(source).toEqual(original);
    });

    it('forwards the owned resolver through the public Anthropic typed context stream before transport', async () => {
        const source = await document();
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        const driver = new AnthropicDriver({ apiKey: 'test-only' });
        await expect(
            driver.streamCanonicalContextEvents(
                {
                    ...options(source, 'claude-sonnet-4-20250514'),
                    on_canonical_request_prepared: async () => {
                        throw new Error('Prepared publication gate');
                    },
                },
                undefined,
                { stream_id: 'stream:concrete-image' },
                { resolve_canonical_asset: resolver },
            ),
        ).rejects.toThrow('Prepared publication gate');
        expect(resolver).toHaveBeenCalledTimes(1);
    });

    it('hydrates nested tool-result images for Claude and Bedrock and fails before transport without authority', async () => {
        const source = await document(true);
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        const claude = await prepareClaudeCanonicalContext({
            options: options(source, 'claude-sonnet-4-20250514'),
            provider: 'anthropic',
            resolve_asset: resolver,
        });
        const bedrock = await prepareBedrockConverseCanonicalContext({
            options: options(source, 'anthropic.claude-sonnet-4-6-v1:0'),
            provider: 'bedrock',
            resolve_asset: resolver,
        });
        expect(resolver).toHaveBeenCalledTimes(2);
        expect(JSON.stringify(claude.native_conversation)).toContain(png.toString('base64'));
        expect(JSON.stringify(bedrock.native_conversation)).toContain('image');
        expect(claude.document.assets['asset:concrete-image']?.storage.type).toBe('external');
        expect(bedrock.document.assets['asset:concrete-image']?.storage.type).toBe('external');
        await expect(
            new AnthropicDriver({ apiKey: 'test-only' }).executeCanonicalContext(
                options(source, 'claude-sonnet-4-20250514'),
            ),
        ).rejects.toThrow();
    });

    it('rejects cancellation, altered hash, unsupported MIME and excess declared bytes before transport', async () => {
        const source = await document();
        const driver = new AnthropicDriver({ apiKey: 'test-only' });
        const transport = vi.fn((_payload: unknown) => {
            throw new Error('Unexpected transport');
        });
        Object.defineProperty(driver.client.messages, 'stream', { value: transport });
        const prepared = vi.fn<NonNullable<ExecutionOptions['on_canonical_request_prepared']>>(async () => undefined);
        const run = (
            conversation: Awaited<ReturnType<typeof document>>,
            resolver: ResolveConversationAsset,
            signal?: AbortSignal,
        ) =>
            driver.executeCanonicalContext(
                {
                    ...options(conversation, 'claude-sonnet-4-20250514'),
                    on_canonical_request_prepared: prepared,
                },
                signal,
                { resolve_canonical_asset: resolver },
            );
        const corrupt = vi.fn<ResolveConversationAsset>(async function* () {
            yield new Uint8Array(png.byteLength);
        });
        await expect(run(source, corrupt)).rejects.toThrow();
        expect(corrupt).toHaveBeenCalledOnce();
        const invalidMime = structuredClone(source);
        const mimeAsset = invalidMime.assets['asset:concrete-image'];
        if (mimeAsset === undefined) throw new Error('Missing image asset fixture');
        mimeAsset.mime_type = 'image/tiff';
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        await expect(run(invalidMime, resolver)).rejects.toThrow();
        expect(resolver).not.toHaveBeenCalled();
        const excess = structuredClone(source);
        const largeAsset = excess.assets['asset:concrete-image'];
        if (largeAsset === undefined) throw new Error('Missing image asset fixture');
        largeAsset.byte_length = 32 * 1024 * 1024 + 1;
        await expect(run(excess, resolver)).rejects.toThrow();
        expect(resolver).not.toHaveBeenCalled();
        const abort = new AbortController();
        abort.abort(new Error('Cancelled before hydration'));
        await expect(run(source, resolver, abort.signal)).rejects.toThrow();
        expect(resolver).not.toHaveBeenCalled();
        expect(prepared).not.toHaveBeenCalled();
        expect(transport).not.toHaveBeenCalled();
    });
});
