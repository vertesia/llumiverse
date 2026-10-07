import Anthropic from '@anthropic-ai/sdk';
import type { Message, RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages.js';
import {
    appendConversationRecords,
    appendToolExecutionResult,
    type ConversationStreamEvent,
    createAcceptedOutputFragment,
    createConversationDocument,
    createToolTurn,
    decodedResponseBatchFromAcceptedRecord,
    deriveConversationId,
    fingerprintJson,
    hashContentBytes,
    type IndexedConversationRecordStore,
    loadIndexedAcceptedOutputContinuation,
    loadIndexedSelectedDependencyContext,
    loadIndexedSelectedMediaCompactionContext,
    parseConversationPreparedRequestRecord,
    type ResolveConversationAsset,
    resolveToolExecutionRequest,
    stageIndexedConversationSnapshot,
} from '@llumiverse/conversation';
import { type ExecutionOptions, PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { createRequestReceipt } from '../conversation/canonical-runtime.js';
import {
    getClaudePayload,
    projectClaudeContextResultSchema,
    projectClaudeConversation,
} from '../shared/claude-messages.js';
import { compileClaudeMessagesConversation } from '../shared/claude-messages-conversation-adapter.js';
import { AnthropicDriver } from './index.js';
import { anthropicIndexedCountBody, countAnthropicIndexedRequest } from './indexed-count.js';

const at = '2026-10-05T00:00:00Z';
const model = 'claude-sonnet-4-6';
const tool = {
    id: 'definition:lookup',
    name: 'lookup',
    version: '1',
    input_schema: { type: 'object', properties: {} },
};

async function fixture(externalImage = false) {
    // Seed only ordinary authorable tool definitions. The signed native generation and
    // application call below are produced through the real configured SDK execution path.
    const empty = createConversationDocument({ id: 'conversation:anthropic-indexed', created_at: at });
    const defined = appendConversationRecords(
        empty,
        {
            tool_definitions: [tool],
            active_tool_definition_ids: [tool.id],
        },
        {
            expected_revision: empty.revision,
            operation_id: 'operation:actual-tool-definition',
            payload_fingerprint: await fingerprintJson(tool),
            recorded_at: at,
        },
    ).document;
    const initialDriver = new AnthropicDriver({ apiKey: 'never-upstream' });
    const final: Message = {
        ...response,
        id: 'message:actual-tool-generation',
        content: [
            { type: 'thinking', thinking: 'Original signed plan', signature: 'signed-original' },
            { type: 'redacted_thinking', data: 'redacted-original' },
            { type: 'tool_use', id: 'call:lookup', name: 'lookup', input: {}, caller: { type: 'direct' } },
        ],
        stop_reason: 'tool_use',
    };
    const frames = [
        { type: 'message_start', message: { ...final, content: [], stop_reason: null } },
        { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '', signature: '' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'Original signed plan' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'signature_delta', signature: 'signed-original' } },
        { type: 'content_block_stop', index: 0 },
        {
            type: 'content_block_start',
            index: 1,
            content_block: { type: 'redacted_thinking', data: 'redacted-original' },
        },
        { type: 'content_block_stop', index: 1 },
        {
            type: 'content_block_start',
            index: 2,
            content_block: {
                type: 'tool_use',
                id: 'call:lookup',
                name: 'lookup',
                input: {},
                caller: { type: 'direct' },
            },
        },
        { type: 'content_block_delta', index: 2, delta: { type: 'input_json_delta', partial_json: '{}' } },
        { type: 'content_block_stop', index: 2 },
        { type: 'message_delta', delta: { stop_reason: 'tool_use', stop_sequence: null }, usage: { output_tokens: 8 } },
        { type: 'message_stop' },
    ];
    const initialBodies: unknown[] = [];
    initialDriver.client = new Anthropic({
        apiKey: 'owned-test',
        baseURL: 'https://owned-claude.test',
        fetch: async (url, init) => {
            expect(new URL(String(url)).pathname).toBe('/v1/messages');
            initialBodies.push(JSON.parse(String(init?.body)));
            return new Response(
                frames.map((frame) => `event: ${frame.type}\ndata: ${JSON.stringify(frame)}\n\n`).join(''),
                { headers: { 'content-type': 'text/event-stream' } },
            );
        },
    });
    const generated = await initialDriver.executeCanonical(
        [
            { role: PromptRole.system, content: 'Keep the exact source.' },
            { role: PromptRole.user, content: 'Question' },
        ],
        {
            model,
            model_options: options().model_options,
            conversation: defined,
            conversation_runtime: {
                conversation_id: defined.id,
                request_id: 'request:actual-tool-generation',
                attempt_id: 'attempt:actual-tool-generation',
                input_operation_id: 'input:actual-tool-generation',
                response_operation_id: 'response:actual-tool-generation',
                recorded_at: at,
            },
        },
    );
    expect(initialBodies).toHaveLength(1);
    expect(generated.accepted_output.generation.record_source).toBe('executed');
    const original = generated.conversation;
    const executed = original.generations[generated.accepted_output.generation.id];
    if (executed?.record_source !== 'executed') throw new Error('SDK result lost its actual executed generation');
    expect(executed.request_receipt.request_fingerprint).toBe(await fingerprintJson(initialBodies[0]));
    const callTurn = original.turns.find(
        (turn) => turn.kind === 'agent' && turn.blocks.some((block) => block.type === 'tool_call'),
    );
    const call = callTurn?.blocks.find((block) => block.type === 'tool_call');
    if (!callTurn || call?.type !== 'tool_call')
        throw new Error('Executed SDK generation lost its exact application call');
    const source = {
        conversation: { conversation_id: original.id, revision: original.revision },
        turn_id: callTurn.id,
        block_id: call.id,
        call_id: call.call_id,
        call_fingerprint: await fingerprintJson(call),
    };
    const request = await resolveToolExecutionRequest(original, source, async () => {
        throw new Error('Inline arguments must not resolve');
    });
    expect(request.arguments).toEqual({});
    const resultBlock = {
        id: 'block:actual-result',
        type: 'tool_result' as const,
        call_id: call.call_id,
        status: 'success' as const,
        content: [
            { id: 'text:actual-result', type: 'text' as const, text: 'Result', format: 'plain' as const },
            { id: 'image:actual-result', type: 'image' as const, asset_id: 'asset:actual-image' },
            { id: 'document:actual-result', type: 'document' as const, asset_id: 'asset:actual-document' },
        ],
    };
    const turn = {
        ...createToolTurn({
            id: 'turn:actual-result',
            authority: 'ordinary',
            blocks: [resultBlock],
            status: 'completed',
            timestamps: { recorded_at: at },
            execution_id: 'execution:actual-result',
            provenance: { type: 'received' },
            model_visibility: 'include',
        }),
        execution_id: 'execution:actual-result',
    };
    const imageBytes = Uint8Array.from(
        Buffer.from(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lL0AAAAASUVORK5CYII=',
            'base64',
        ),
    );
    const textBytes = new TextEncoder().encode('Original document');
    const assets = [
        {
            id: 'asset:actual-image',
            kind: 'image' as const,
            mime_type: 'image/png',
            storage: externalImage
                ? {
                      type: 'external' as const,
                      resolver: 'url',
                      locator: { url: 'https://owned-media.test/actual-original.png' },
                  }
                : { type: 'inline_base64' as const, data: Buffer.from(imageBytes).toString('base64') },
            ...(await hashContentBytes(imageBytes)),
            provenance: { type: 'received' as const, source_turn_id: turn.id },
            created_at: at,
        },
        {
            id: 'asset:actual-document',
            kind: 'document' as const,
            mime_type: 'text/plain',
            storage: { type: 'inline_text' as const, text: 'Original document' },
            ...(await hashContentBytes(textBytes)),
            provenance: { type: 'received' as const, source_turn_id: turn.id },
            created_at: at,
        },
    ];
    const accepted = await appendToolExecutionResult(
        original,
        {
            source,
            turn,
            assets,
            execution_receipt: {
                id: 'execution:actual-result',
                call_id: call.call_id,
                executor: 'application',
                status: 'success',
                result_turn_id: turn.id,
                result_fingerprint: await fingerprintJson(resultBlock),
                recorded_at: at,
                call_source: source,
            },
        },
        { operation_id: 'operation:actual-tool-result', expected_revision: original.revision, recorded_at: at },
    );
    const document = accepted.document;
    const pages = new Map<string, Uint8Array>();
    const records = new Map<string, Uint8Array>();
    const store: IndexedConversationRecordStore = {
        async read(ref) {
            const bytes = pages.get(ref.content_hash);
            if (!bytes) throw new Error('Missing actual page');
            return Uint8Array.from(bytes);
        },
        async write(bytes, ref) {
            pages.set(ref.content_hash, Uint8Array.from(bytes));
        },
        async readRecord(ref) {
            const bytes = records.get(ref.content_hash);
            if (!bytes) throw new Error('Missing actual record');
            return Uint8Array.from(bytes);
        },
        async writeRecord(ref, bytes) {
            expect((await hashContentBytes(bytes)).content_hash).toBe(ref.content_hash);
            records.set(ref.content_hash, Uint8Array.from(bytes));
        },
    };
    const staged = await stageIndexedConversationSnapshot(document, undefined, store);
    const selection = await (externalImage
        ? loadIndexedSelectedMediaCompactionContext
        : loadIndexedSelectedDependencyContext)(store, staged.root, staged.locator);
    const runtime = {
        conversation_id: document.id,
        request_id: 'request:indexed',
        attempt_id: 'attempt:indexed',
        input_operation_id: 'input:indexed',
        response_operation_id: 'response:indexed',
        recorded_at: at,
        purpose: 'interaction' as const,
    };
    return { document, selection, runtime, imageBytes, store };
}

function options(): ExecutionOptions {
    const modelOptions = {
        _option_id: 'anthropic-claude',
        max_tokens: 2048,
        thinking_mode: 'adaptive',
        tool_choice: 'required',
        required_tool_name: 'lookup',
        parallel_tool_calls: false,
    } satisfies NonNullable<ExecutionOptions['model_options']> & {
        tool_choice: 'required';
        required_tool_name: string;
        parallel_tool_calls: boolean;
    };
    return {
        model,
        prompt_cache_key: 'agent-stable-prefix',
        model_options: modelOptions,
        result_schema: {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        },
    };
}

async function recordFor(
    prepared: Awaited<ReturnType<AnthropicDriver['prepareIndexedTextRequest']>>,
    runtime: Awaited<ReturnType<typeof fixture>>['runtime'],
) {
    return parseConversationPreparedRequestRecord({
        source: prepared.source,
        runtime,
        request_receipt: prepared.receipt,
        generation_id: await deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
        response_turn_id: await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
    });
}

const response: Message = {
    id: 'message:indexed',
    container: null,
    diagnostics: null,
    stop_details: null,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: '{"answer":"Done"}', citations: null }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: {
        input_tokens: 12,
        output_tokens: 8,
        cache_creation: null,
        cache_creation_input_tokens: null,
        cache_read_input_tokens: null,
        inference_geo: null,
        output_tokens_details: null,
        server_tool_use: null,
        service_tier: null,
    },
};

describe('direct Anthropic indexed native lifecycle', () => {
    it('publishes genuine indexed records and preserves complete materialized tools/cache/reasoning/schema/media payload parity', async () => {
        const f = await fixture();
        const driver = new AnthropicDriver({ apiKey: 'never-network' });
        const prepared = await driver.prepareIndexedTextRequest({
            selection: f.selection,
            runtime: f.runtime,
            options: options(),
            stream: false,
        });
        const compiled = compileClaudeMessagesConversation(f.document, { provider: 'anthropic', model });
        const projected = projectClaudeContextResultSchema(
            projectClaudeConversation(compiled.conversation, options(), 0),
            options(),
            true,
        );
        expect(prepared.payload).toEqual(
            getClaudePayload(options(), projected, 'anthropic', 'execute', undefined, [tool]).payload,
        );
        expect(prepared.receipt).toEqual(
            await createRequestReceipt(
                f.document,
                f.runtime,
                prepared.receipt.target,
                prepared.native_request,
                compiled.mappings,
                [tool],
            ),
        );
        const body = JSON.stringify(prepared.native_request);
        expect(body).toContain('signed-original');
        expect(body).toContain('redacted-original');
        expect(body).toContain('cache_control');
        expect(body).toContain('image/png');
        expect(body).toContain('Original document');
        const counted = vi.spyOn(driver.client.messages, 'countTokens').mockResolvedValue({ input_tokens: 31 });
        const count = await driver.countIndexedNativeRequest(prepared.native_request, prepared.receipt.target);
        expect(count.input_tokens).toBe(31);
        expect(count.request_fingerprint).toBe(await fingerprintJson(prepared.native_request));
        expect(counted.mock.calls[0]?.[0]).toEqual(
            anthropicIndexedCountBody(prepared.native_request, prepared.receipt.target),
        );
        expect(counted.mock.calls[0]?.[0]).toMatchObject({
            tools: expect.any(Array),
            thinking: expect.any(Object),
            tool_choice: expect.any(Object),
        });
    });

    it('fences exact prepared native execution and decodes structured output without inventing a document', async () => {
        const f = await fixture();
        const driver = new AnthropicDriver({ apiKey: 'never-network' });
        const prepared = await driver.prepareIndexedTextRequest({
            selection: f.selection,
            runtime: f.runtime,
            options: options(),
            stream: false,
        });
        const record = await recordFor(prepared, f.runtime);
        const dispatch = vi
            .spyOn(driver.client.messages, 'stream')
            .mockReturnValue({ finalMessage: async () => response } as ReturnType<
                typeof driver.client.messages.stream
            >);
        const revoked = vi.fn(async () => {
            throw new Error('revoked real dispatch fence');
        });
        await expect(
            driver.executeCommittedIndexedTextRequest({
                selection: f.selection,
                record,
                options: options(),
                assert_committed: revoked,
            }),
        ).rejects.toThrow('revoked real dispatch fence');
        expect(dispatch).not.toHaveBeenCalled();
        const committed = vi.fn(async () => undefined);
        const decoded = await driver.executeCommittedIndexedTextRequest({
            selection: f.selection,
            record,
            options: options(),
            assert_committed: committed,
        });
        expect(committed).toHaveBeenCalledOnce();
        expect(dispatch).toHaveBeenCalledOnce();
        expect(decoded.turns[0]?.blocks.some((block) => block.type === 'json')).toBe(true);
        expect(decoded.generation.request_receipt).toEqual(record.request_receipt);
        await expect(
            driver.executeCommittedIndexedTextRequest({
                selection: f.selection,
                record,
                options: { ...options(), model: 'claude-other' },
                assert_committed: committed,
            }),
        ).rejects.toThrow();
        expect(dispatch).toHaveBeenCalledOnce();
    });

    it.each([
        { text: '{"answer":"Done"}', status: 'completed', code: undefined, expectedAnswer: 'Done' },
        { text: '{"answer":17}', status: 'completed', code: undefined, expectedAnswer: '17' },
        { text: '{"answer":{"nested":true}}', status: 'failed', code: 'validation_error', expectedAnswer: undefined },
        { text: 'invalid JSON', status: 'failed', code: 'validation_error', expectedAnswer: undefined },
        { text: '{"answer":"\\uZZZZ"}', status: 'failed', code: 'json_error', expectedAnswer: undefined },
    ])(
        'normalizes required structured output from genuine SDK text: $text',
        async ({ text, status, code, expectedAnswer }) => {
            const f = await fixture();
            const driver = new AnthropicDriver({ apiKey: 'never-upstream' });
            const bodies: unknown[] = [];
            driver.client = new Anthropic({
                apiKey: 'owned-test',
                baseURL: 'https://owned-claude.test',
                fetch: async (url, init) => {
                    expect(new URL(String(url)).pathname).toBe('/v1/messages');
                    bodies.push(JSON.parse(String(init?.body)));
                    const frames = [
                        { type: 'message_start', message: { ...response, content: [], stop_reason: null } },
                        {
                            type: 'content_block_start',
                            index: 0,
                            content_block: { type: 'text', text: '', citations: [] },
                        },
                        { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } },
                        { type: 'content_block_stop', index: 0 },
                        {
                            type: 'message_delta',
                            delta: { stop_reason: 'end_turn', stop_sequence: null },
                            usage: { output_tokens: 8 },
                        },
                        { type: 'message_stop' },
                    ];
                    return new Response(
                        frames.map((frame) => `event: ${frame.type}\ndata: ${JSON.stringify(frame)}\n\n`).join(''),
                        { headers: { 'content-type': 'text/event-stream' } },
                    );
                },
            });
            const executionOptions: ExecutionOptions = {
                model,
                model_options: { max_tokens: 128 },
                result_schema: options().result_schema,
            };
            const prepared = await driver.prepareIndexedTextRequest({
                selection: f.selection,
                runtime: f.runtime,
                options: executionOptions,
                stream: false,
            });
            const record = await recordFor(prepared, f.runtime);
            const fence = vi.fn(async () => undefined);
            const decoded = await driver.executeCommittedIndexedTextRequest({
                selection: f.selection,
                record,
                options: executionOptions,
                assert_committed: fence,
            });
            expect(bodies).toEqual([prepared.native_request]);
            expect(fence).toHaveBeenCalledOnce();
            expect(decoded.generation.status).toBe(status);
            expect(decoded.turns[0]?.status).toBe(status);
            expect(decoded.generation.request_receipt).toEqual(record.request_receipt);
            if (code === undefined) {
                expect(decoded.turns[0]?.blocks).toEqual(
                    expect.arrayContaining([
                        expect.objectContaining({ type: 'json', value: { answer: expectedAnswer } }),
                    ]),
                );
            } else {
                expect(decoded.generation.metadata?.structured_output).toMatchObject({ status: 'invalid', code });
                expect(decoded.turns[0]?.blocks).toEqual(
                    expect.arrayContaining([expect.objectContaining({ type: 'text', text })]),
                );
                expect(decoded.turns[0]?.blocks.some((block) => block.type === 'json')).toBe(false);
            }
        },
    );

    it('hydrates only verified original external media and leaves retained descriptors unchanged', async () => {
        const f = await fixture(true);
        const driver = new AnthropicDriver({ apiKey: 'never-network' });
        const resolve = vi.fn<ResolveConversationAsset>(async function* () {
            yield f.imageBytes;
        });
        const before = structuredClone(f.selection);
        const prepared = await driver.prepareIndexedTextRequest(
            { selection: f.selection, runtime: f.runtime, options: options(), stream: false },
            { resolve_canonical_asset: resolve },
        );
        expect(resolve).toHaveBeenCalledOnce();
        expect(JSON.stringify(prepared.native_request)).toContain(Buffer.from(f.imageBytes).toString('base64'));
        expect(f.selection).toEqual(before);
        const endpoint = vi.spyOn(driver.client.messages, 'countTokens');
        const corrupt = vi.fn<ResolveConversationAsset>(async function* () {
            yield Uint8Array.from([0]);
        });
        await expect(
            driver.prepareIndexedTextRequest(
                { selection: f.selection, runtime: f.runtime, options: options(), stream: false },
                { resolve_canonical_asset: corrupt },
            ),
        ).rejects.toThrow();
        expect(endpoint).not.toHaveBeenCalled();
    });

    it('owns the exact count body across asynchronous endpoint completion and rejects invalid counts', async () => {
        const f = await fixture();
        const driver = new AnthropicDriver({ apiKey: 'never-network' });
        const prepared = await driver.prepareIndexedTextRequest({
            selection: f.selection,
            runtime: f.runtime,
            options: options(),
            stream: false,
        });
        const original = structuredClone(prepared.native_request);
        const endpoint = vi.spyOn(driver.client.messages, 'countTokens').mockResolvedValue({ input_tokens: 17 });
        const counting = driver.countIndexedNativeRequest(prepared.native_request, prepared.receipt.target);
        prepared.native_request.messages = [];
        const counted = await counting;
        expect(counted.request_fingerprint).toBe(await fingerprintJson(original));
        expect(endpoint.mock.calls[0]?.[0]).toEqual(anthropicIndexedCountBody(original, prepared.receipt.target));
        endpoint.mockResolvedValueOnce({ input_tokens: -1 });
        await expect(driver.countIndexedNativeRequest(original, prepared.receipt.target)).rejects.toThrow(
            'invalid input count',
        );
    });

    it('counts and streams the exact owned Claude SDK body with provisional events and a genuine accepted-output CAS', async () => {
        const f = await fixture();
        const bodies: { path: string; body: unknown }[] = [];
        const final: Message = {
            ...response,
            id: 'message:actual-indexed-stream',
            content: [{ type: 'text', text: 'Actual Claude stream', citations: null }],
            stop_reason: 'end_turn',
            usage: { ...response.usage, input_tokens: 11, output_tokens: 7 },
        };
        const frames = [
            {
                type: 'message_start',
                message: { ...final, content: [], stop_reason: null, usage: { ...final.usage, output_tokens: 0 } },
            },
            { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '', citations: [] } },
            { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Actual Claude ' } },
            { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'stream' } },
            { type: 'content_block_stop', index: 0 },
            {
                type: 'message_delta',
                delta: { stop_reason: 'end_turn', stop_sequence: null, container: null, stop_details: null },
                usage: {
                    output_tokens: 7,
                    input_tokens: null,
                    cache_creation_input_tokens: null,
                    cache_read_input_tokens: null,
                    output_tokens_details: null,
                    server_tool_use: null,
                },
            },
            { type: 'message_stop' },
        ] satisfies RawMessageStreamEvent[];
        const driver = new AnthropicDriver({ apiKey: 'never-upstream' });
        driver.client = new Anthropic({
            apiKey: 'owned-test',
            baseURL: 'https://owned-claude.test',
            fetch: async (url, init) => {
                const path = new URL(String(url)).pathname;
                bodies.push({ path, body: JSON.parse(String(init?.body)) });
                if (path === '/v1/messages/count_tokens')
                    return new Response(JSON.stringify({ input_tokens: 11 }), {
                        headers: { 'content-type': 'application/json' },
                    });
                if (path !== '/v1/messages') throw new Error('Unexpected upstream endpoint');
                return new Response(
                    frames.map((frame) => `event: ${frame.type}\ndata: ${JSON.stringify(frame)}\n\n`).join(''),
                    { headers: { 'content-type': 'text/event-stream' } },
                );
            },
        });
        const executionOptions: ExecutionOptions = { model, model_options: { max_tokens: 128 } };
        const prepared = await driver.prepareIndexedTextRequest({
            selection: f.selection,
            runtime: f.runtime,
            options: executionOptions,
            stream: true,
        });
        const counted = await driver.countIndexedNativeRequest(prepared.native_request, prepared.receipt.target);
        expect(counted.request_fingerprint).toBe(await fingerprintJson(prepared.native_request));
        const record = await recordFor(prepared, f.runtime);
        const fence = vi.fn(async () => undefined);
        const accepted = vi.fn(async (decoded: Parameters<typeof decodedResponseBatchFromAcceptedRecord>[1]) => {
            const operation = decodedResponseBatchFromAcceptedRecord(record, decoded, {
                operation_id: f.runtime.response_operation_id,
                recorded_at: decoded.generation.timestamps.completed_at ?? f.runtime.recorded_at,
            });
            const next = appendConversationRecords(f.document, operation.batch, operation.options).document;
            const indexed = await stageIndexedConversationSnapshot(next, undefined, f.store);
            const receipt = createAcceptedOutputFragment(next, f.runtime.response_operation_id).receipt;
            return (await loadIndexedAcceptedOutputContinuation(f.store, indexed.root, receipt)).fragment;
        });
        const stream = await driver.streamCommittedIndexedTextRequest({
            selection: f.selection,
            record,
            options: executionOptions,
            assert_committed: fence,
            accept_output: accepted,
            open: { stream_id: 'stream:actual-claude-indexed' },
        });
        const events: ConversationStreamEvent[] = [];
        for await (const event of stream) events.push(event);
        await stream.closed;
        expect(bodies).toEqual([
            {
                path: '/v1/messages/count_tokens',
                body: anthropicIndexedCountBody(prepared.native_request, prepared.receipt.target),
            },
            { path: '/v1/messages', body: prepared.native_request },
        ]);
        expect(fence).toHaveBeenCalledTimes(2);
        expect(accepted).toHaveBeenCalledOnce();
        expect(events.some((event) => event.type === 'draft_text_delta')).toBe(true);
        expect(events.at(-1)?.type).toBe('response_accepted');
        const revoked = vi.fn(async () => {
            throw new Error('Changed live task/source fence');
        });
        await expect(
            driver.streamCommittedIndexedTextRequest({
                selection: f.selection,
                record,
                options: executionOptions,
                assert_committed: revoked,
                accept_output: accepted,
                open: { stream_id: 'stream:revoked' },
            }),
        ).rejects.toThrow('Changed live task/source fence');
        expect(bodies).toHaveLength(2);
        expect(accepted).toHaveBeenCalledOnce();
    });

    it('rejects unsupported controls and count fields before either provider endpoint', async () => {
        const f = await fixture();
        const driver = new AnthropicDriver({ apiKey: 'never-network' });
        const count = vi.spyOn(driver.client.messages, 'countTokens');
        const dispatch = vi.spyOn(driver.client.messages, 'stream');
        await expect(
            driver.prepareIndexedTextRequest({
                selection: f.selection,
                runtime: f.runtime,
                options: { ...options(), stripImagesAfterTurns: 3 },
                stream: false,
            }),
        ).rejects.toThrow('authenticated lifetime counter');
        const prepared = await driver.prepareIndexedTextRequest({
            selection: f.selection,
            runtime: f.runtime,
            options: options(),
            stream: false,
        });
        await expect(
            countAnthropicIndexedRequest(
                driver.client,
                { ...prepared.native_request, hidden_input: 'uncounted' },
                prepared.receipt.target,
            ),
        ).rejects.toThrow('unsupported target/body');
        expect(count).not.toHaveBeenCalled();
        expect(dispatch).not.toHaveBeenCalled();
    });
});
