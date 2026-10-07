import {
    appendConversationRecords,
    type ConversationPreparedRequest,
    createConversationDocument,
    createTextBlock,
    createToolTurn,
    createUserTurn,
    fingerprintJson,
    hashContentBytes,
    processingContextFingerprint,
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
import { OpenAIChatCompletionsDriver } from '../openai/openai_chat_completions.js';
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
    it('hydrates selected nested images in the actual OpenAI Chat request and binds its retained fingerprint', async () => {
        const source = await document(true);
        const original = structuredClone(source);
        const driver = new OpenAIChatCompletionsDriver({ apiKey: 'test-only', endpoint: 'http://unused.invalid' });
        const native = vi.fn((_payload: unknown) => ({
            id: 'chatcmpl:accepted-image',
            object: 'chat.completion' as const,
            created: 1,
            model: 'gpt-4.1',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant' as const, content: 'Image received.' },
                    finish_reason: 'stop' as const,
                    logprobs: null,
                },
            ],
            usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
        }));
        Object.defineProperty(driver.service.chat.completions, 'create', { value: native });
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        let retained: ConversationPreparedRequest | undefined;
        const accepted = await driver.executeCanonicalContext(
            {
                ...options(source, 'gpt-4.1'),
                on_canonical_request_prepared: async (prepared) => {
                    retained = prepared;
                },
            },
            undefined,
            { resolve_canonical_asset: resolver },
        );
        expect(accepted.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'text', text: 'Image received.' }),
        );
        expect(accepted.accepted_output.generation.usage?.total_tokens).toBe(6);
        const sent = native.mock.calls[0]?.[0];
        if (sent === undefined || retained === undefined)
            throw new Error('Expected native request and retained record');
        expect(JSON.stringify(sent)).toContain(png.toString('base64'));
        expect(retained.record.request_receipt.request_fingerprint).toBe(
            await fingerprintJson(providerJsonValue(sent)),
        );
        expect(retained.document.assets['asset:concrete-image']?.storage.type).toBe('external');
        expect(source).toEqual(original);
        expect(resolver).toHaveBeenCalledOnce();

        const prepared = vi.fn<NonNullable<ExecutionOptions['on_canonical_request_prepared']>>(async () => {
            throw new Error('Prepared publication gate');
        });
        await expect(
            driver.streamCanonicalContextEvents(
                { ...options(source, 'gpt-4.1'), on_canonical_request_prepared: prepared },
                undefined,
                { stream_id: 'stream:chat:nested-image' },
                { resolve_canonical_asset: resolver },
            ),
        ).rejects.toThrow('Prepared publication gate');
        expect(prepared).toHaveBeenCalledOnce();
        expect(resolver).toHaveBeenCalledTimes(2);
        expect(native).toHaveBeenCalledOnce();

        await expect(
            driver.executeCanonical(
                [{ role: PromptRole.user, content: 'Continue after the image.' }],
                { ...options(source, 'gpt-4.1'), on_canonical_request_prepared: prepared },
                undefined,
                { resolve_canonical_asset: resolver },
            ),
        ).rejects.toThrow('Prepared publication gate');
        expect(prepared).toHaveBeenCalledTimes(2);
        expect(resolver).toHaveBeenCalledTimes(3);
        expect(native).toHaveBeenCalledOnce();
        expect(source).toEqual(original);
    });

    it('replays the exact hydrated native request during host measurement without another image read', async () => {
        const source = await document(true);
        const original = structuredClone(source);
        const driver = new OpenAIChatCompletionsDriver({ apiKey: 'test-only', endpoint: 'http://unused.invalid' });
        const native = vi.fn((_payload: unknown) => ({
            id: 'chatcmpl:measured-image',
            object: 'chat.completion' as const,
            created: 1,
            model: 'gpt-4.1',
            choices: [
                {
                    index: 0,
                    message: { role: 'assistant' as const, content: 'Measured image.' },
                    finish_reason: 'stop' as const,
                    logprobs: null,
                },
            ],
            usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
        }));
        Object.defineProperty(driver.service.chat.completions, 'create', { value: native });
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        const projected = vi.fn<NonNullable<ExecutionOptions['on_canonical_request_projected']>>(
            async (projection, compiler) => {
                expect(await compiler.compileDocument(projection.document)).toEqual(projection.native_request);
                const changed = structuredClone(projection.document);
                const changedAsset = changed.assets['asset:concrete-image'];
                if (!changedAsset) throw new Error('Selected image absent from projected source');
                changedAsset.content_hash = `sha256:${'f'.repeat(64)}`;
                await expect(compiler.compileDocument(changed)).rejects.toThrow('changed during native preparation');

                const added = structuredClone(projection.document);
                const oldAsset = added.assets['asset:concrete-image'];
                const userTurn = added.turns.find((turn) => turn.kind === 'user');
                if (!oldAsset || userTurn?.kind !== 'user') throw new Error('Selected user turn absent');
                const newAsset = { ...oldAsset, id: 'asset:new-external-image' };
                added.assets[newAsset.id] = newAsset;
                userTurn.blocks.push({ id: 'block:new-external-image', type: 'image', asset_id: newAsset.id });
                await expect(compiler.compileDocument(added)).rejects.toThrow(
                    'introduced an external image without prepared host bytes',
                );
                return {
                    counted_request_fingerprint: await fingerprintJson(projection.native_request),
                    measurement: {
                        input_tokens: 7,
                        method: 'estimated' as const,
                        tokenizer: 'named:image-fixture',
                        tokenizer_version: '1',
                        adapter: projection.target.protocol,
                        adapter_version: projection.target.adapter_version,
                        source_fingerprint: await processingContextFingerprint(projection.document),
                        target_model: projection.target.model,
                        measured_at: at,
                    },
                };
            },
        );
        const accepted = await driver.executeCanonicalContext(
            { ...options(source, 'gpt-4.1'), on_canonical_request_projected: projected },
            undefined,
            { resolve_canonical_asset: resolver },
        );
        expect(accepted.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'text', text: 'Measured image.' }),
        );
        expect(projected).toHaveBeenCalledOnce();
        expect(resolver).toHaveBeenCalledOnce();
        expect(native).toHaveBeenCalledOnce();
        expect(source).toEqual(original);
    });

    it('sends hydrated bytes in context and authored typed streams while retaining canonical external refs', async () => {
        const source = await document(true);
        const original = structuredClone(source);
        const driver = new OpenAIChatCompletionsDriver({
            apiKey: 'test-only',
            endpoint: 'http://unused.invalid',
            extraBody: {
                model: 'forged-model',
                messages: [{ role: 'user', content: 'forged-message' }],
                stream_options: { include_usage: false },
                extension: { trace: 'retained' },
            },
        });
        const native = vi.fn(async (_payload: unknown) => ({
            async *[Symbol.asyncIterator]() {
                yield {
                    id: 'chatcmpl:stream-image',
                    object: 'chat.completion.chunk' as const,
                    created: 1,
                    model: 'gpt-4.1',
                    choices: [
                        {
                            index: 0,
                            delta: { role: 'assistant' as const, content: 'Streamed image.' },
                            finish_reason: null,
                        },
                    ],
                };
                yield {
                    id: 'chatcmpl:stream-image',
                    object: 'chat.completion.chunk' as const,
                    created: 1,
                    model: 'gpt-4.1',
                    choices: [{ index: 0, delta: {}, finish_reason: 'stop' as const }],
                    usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
                };
            },
            controller: { abort: vi.fn() },
        }));
        Object.defineProperty(driver.service.chat.completions, 'create', { value: native });
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        const prepared: ConversationPreparedRequest[] = [];
        const publish: NonNullable<ExecutionOptions['on_canonical_request_prepared']> = async (value) => {
            prepared.push(value);
        };
        const context = await driver.streamCanonicalContextEvents(
            { ...options(source, 'gpt-4.1'), on_canonical_request_prepared: publish },
            undefined,
            { stream_id: 'stream:chat:context-native' },
            { resolve_canonical_asset: resolver },
        );
        const contextEvents = [];
        for await (const event of context) contextEvents.push(event);
        expect(contextEvents).toContainEqual(expect.objectContaining({ type: 'response_accepted' }));
        expect(context.completion?.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'text', text: 'Streamed image.' }),
        );
        const authored = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Continue from the retained image.' }],
            { ...options(source, 'gpt-4.1'), on_canonical_request_prepared: publish },
            undefined,
            { stream_id: 'stream:chat:authored-native' },
            { resolve_canonical_asset: resolver },
        );
        const authoredEvents = [];
        for await (const event of authored) authoredEvents.push(event);
        expect(authoredEvents).toContainEqual(expect.objectContaining({ type: 'response_accepted' }));
        expect(prepared).toHaveLength(2);
        expect(native).toHaveBeenCalledTimes(2);
        expect(resolver).toHaveBeenCalledTimes(2);
        for (let index = 0; index < native.mock.calls.length; index++) {
            const payload = native.mock.calls[index]?.[0];
            const record = prepared[index];
            if (payload === undefined || record === undefined) throw new Error('Expected native stream request');
            expect(JSON.stringify(payload)).toContain(png.toString('base64'));
            expect(payload).toMatchObject({
                model: 'gpt-4.1',
                stream: true,
                stream_options: { include_usage: true },
                extension: { trace: 'retained' },
            });
            expect(JSON.stringify(payload)).not.toContain('forged-message');
            expect(record.record.request_receipt.request_fingerprint).toBe(
                await fingerprintJson(providerJsonValue(payload)),
            );
            expect(record.document.assets['asset:concrete-image']?.storage.type).toBe('external');
        }
        expect(source).toEqual(original);
    });

    it('rejects missing, corrupt, mismatched-MIME, and cancelled host images before Chat transport', async () => {
        const source = await document(true);
        const driver = new OpenAIChatCompletionsDriver({ apiKey: 'test-only', endpoint: 'http://unused.invalid' });
        const native = vi.fn(() => {
            throw new Error('Unexpected transport');
        });
        Object.defineProperty(driver.service.chat.completions, 'create', { value: native });
        await expect(driver.executeCanonicalContext(options(source, 'gpt-4.1'))).rejects.toThrow('no host resolver');

        const corrupt = vi.fn<ResolveConversationAsset>(async function* () {
            yield Buffer.from('different bytes');
        });
        await expect(
            driver.executeCanonicalContext(options(source, 'gpt-4.1'), undefined, {
                resolve_canonical_asset: corrupt,
            }),
        ).rejects.toThrow();
        expect(corrupt).toHaveBeenCalledOnce();

        const wrongMime = structuredClone(source);
        const image = wrongMime.assets['asset:concrete-image'];
        if (image === undefined) throw new Error('Missing image fixture');
        image.mime_type = 'image/jpeg';
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        await expect(
            driver.executeCanonicalContext(options(wrongMime, 'gpt-4.1'), undefined, {
                resolve_canonical_asset: resolver,
            }),
        ).rejects.toThrow('invalid media bytes');

        const controller = new AbortController();
        const cancelled = vi.fn<ResolveConversationAsset>(async function* () {
            controller.abort(new DOMException('Cancelled', 'AbortError'));
            yield png;
        });
        await expect(
            driver.executeCanonicalContext(options(source, 'gpt-4.1'), controller.signal, {
                resolve_canonical_asset: cancelled,
            }),
        ).rejects.toMatchObject({ name: 'AbortError' });
        expect(native).not.toHaveBeenCalled();
    });

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
