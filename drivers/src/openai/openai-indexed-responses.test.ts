import {
    appendConversationRecords,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    decodedResponseBatchFromAcceptedRecord,
    deriveConversationId,
    fingerprintJson,
    hashContentBytes,
    type IndexedConversationRecordStore,
    loadIndexedSelectedDependencyContext,
    loadIndexedSelectedMediaCompactionContext,
    loadIndexedSelectedTextContext,
    parseConversationPreparedRequestRecord,
    type ResolveConversationAsset,
    stageIndexedConversationSnapshot,
} from '@llumiverse/conversation';
import {
    assertCanonicalFailedExecutionMatchesPreparedRecord,
    type CanonicalHostCapabilities,
    canonicalFailedExecution,
    type ExecutionOptions,
} from '@llumiverse/core';
import OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { createRequestReceipt, providerJsonValue } from '../conversation/canonical-runtime.js';
import { OpenAIDriver } from './openai.js';
import { OpenAIChatCompletionsDriver } from './openai_chat_completions.js';
import {
    compileOpenAIResponsesConversation,
    compileOpenAIResponsesIndexedSelectedText,
    OPENAI_RESPONSES_ADAPTER_VERSION,
    OPENAI_RESPONSES_PROTOCOL,
} from './openai-responses-conversation-adapter.js';

async function selectedText(withDependencies = false, externalImage = false) {
    const at = '2026-10-02T00:00:00.000Z';
    const initial = createConversationDocument({ id: 'conversation:indexed-responses', created_at: at });
    const bytes = externalImage
        ? Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10, 1])
        : new TextEncoder().encode('owned inline image');
    const integrity = await hashContentBytes(bytes);
    const asset = {
        id: 'asset:selected-image',
        kind: 'image' as const,
        mime_type: 'image/png',
        storage: externalImage
            ? { type: 'external' as const, resolver: 'url', locator: { url: 'gs://project/runs/run/media/source.png' } }
            : { type: 'inline_base64' as const, data: btoa(new TextDecoder().decode(bytes)) },
        ...integrity,
        provenance: { type: 'received' as const, source_turn_id: 'turn:user' },
        created_at: at,
    };
    const definition = {
        id: 'definition:selected-tool',
        name: 'read_image',
        version: '1',
        input_schema: { type: 'object', properties: {}, additionalProperties: false },
    };
    const turn = createUserTurn({
        id: 'turn:user',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: at },
        provenance: { type: 'received' },
        model_visibility: 'include',
        blocks: [
            createTextBlock({ id: 'block:selected', text: 'Selected text', format: 'plain' }),
            createTextBlock({ id: 'block:cold', text: 'Cold unselected text', format: 'plain' }),
            ...(withDependencies ? [{ id: 'block:selected-image', type: 'image' as const, asset_id: asset.id }] : []),
        ],
    });
    const document = appendConversationRecords(
        initial,
        {
            turns: [turn],
            ...(withDependencies
                ? { assets: [asset], tool_definitions: [definition], active_tool_definition_ids: [definition.id] }
                : {}),
            context_entries: [
                {
                    id: 'entry:user',
                    type: 'source_turn',
                    turn_id: turn.id,
                    block_ids: ['block:selected', ...(withDependencies ? ['block:selected-image'] : [])],
                },
            ],
        },
        {
            expected_revision: initial.revision,
            operation_id: 'operation:user',
            payload_fingerprint: 'sha256:input',
            recorded_at: at,
        },
    ).document;
    const blobs = new Map<string, Uint8Array>();
    const records = new Map<string, Uint8Array>();
    const reads: string[] = [];
    const store: IndexedConversationRecordStore = {
        async read(ref) {
            const bytes = blobs.get(ref.content_hash);
            if (!bytes) throw new Error('missing page');
            return Uint8Array.from(bytes);
        },
        async write(bytes, ref) {
            blobs.set(ref.content_hash, Uint8Array.from(bytes));
        },
        async readRecord(ref) {
            reads.push(`${ref.kind}:${ref.id}`);
            const bytes = records.get(ref.content_hash);
            if (!bytes) throw new Error('missing record');
            return Uint8Array.from(bytes);
        },
        async writeRecord(ref, bytes) {
            records.set(ref.content_hash, Uint8Array.from(bytes));
        },
    };
    const staged = await stageIndexedConversationSnapshot(document, undefined, store);
    reads.length = 0;
    const selection = await (externalImage
        ? loadIndexedSelectedMediaCompactionContext
        : withDependencies
          ? loadIndexedSelectedDependencyContext
          : loadIndexedSelectedTextContext)(store, staged.root, staged.locator);
    const runtime = {
        conversation_id: document.id,
        request_id: 'request:responses',
        attempt_id: 'attempt:responses',
        input_operation_id: 'operation:user',
        response_operation_id: 'response:responses',
        recorded_at: at,
        purpose: 'interaction' as const,
    };
    return { document, selection, runtime, reads, store, bytes };
}

function responseBody(): OpenAI.Responses.Response {
    return {
        id: 'response:indexed',
        access_programs: null,
        object: 'response',
        created_at: 1,
        model: 'gpt-5.4',
        service_tier: 'default',
        status: 'completed',
        output: [
            {
                type: 'message',
                id: 'message:indexed',
                role: 'assistant',
                status: 'completed',
                content: [{ type: 'output_text', text: 'Done', annotations: [], logprobs: [] }],
            },
        ],
        output_text: 'Done',
        parallel_tool_calls: true,
        tool_choice: 'auto',
        tools: [],
        error: null,
        incomplete_details: null,
        instructions: null,
        metadata: null,
        temperature: null,
        top_p: null,
        usage: {
            input_tokens: 4,
            output_tokens: 2,
            total_tokens: 6,
            input_tokens_details: { cached_tokens: 0, cache_write_tokens: 0 },
            output_tokens_details: { reasoning_tokens: 0 },
        },
    };
}

describe('indexed OpenAI Responses selected text', () => {
    it('keeps selected inline media and active tools in the actual Responses body, receipt and decoded call', async () => {
        const { document, selection, runtime, reads } = await selectedText(true);
        const target = { provider: 'openai', model: 'gpt-5.4' };
        const full = compileOpenAIResponsesConversation(document, target);
        expect(compileOpenAIResponsesIndexedSelectedText(selection, target)).toEqual(full);
        expect(reads).not.toContain('blocks:block:cold');
        const bodies: unknown[] = [];
        const driver = new OpenAIDriver({ apiKey: 'test' });
        driver.service = new OpenAI({
            apiKey: 'test',
            baseURL: 'https://indexed.test/v1',
            fetch: async (_url, init) => {
                bodies.push(JSON.parse(String(init?.body)));
                const response = responseBody();
                return Response.json({
                    ...response,
                    output_text: '',
                    output: [
                        {
                            type: 'function_call',
                            id: 'native:read-image',
                            call_id: 'call:read-image',
                            name: 'read_image',
                            arguments: '{}',
                            status: 'completed',
                        },
                    ],
                });
            },
        });
        const prepared = await driver.prepareIndexedTextRequest({
            selection,
            runtime,
            options: { model: target.model },
            stream: false,
        });
        expect(prepared.receipt.tool_definition_ids).toEqual(['definition:selected-tool']);
        expect(prepared.receipt.asset_versions).toEqual([
            expect.objectContaining({
                asset_id: 'asset:selected-image',
                content_hash: document.assets['asset:selected-image'].content_hash,
            }),
        ]);
        expect(prepared.receipt).toEqual(
            await createRequestReceipt(
                document,
                runtime,
                { ...target, protocol: OPENAI_RESPONSES_PROTOCOL, adapter_version: OPENAI_RESPONSES_ADAPTER_VERSION },
                providerJsonValue(prepared.payload),
                full.mappings,
                [document.tool_definitions['definition:selected-tool']],
            ),
        );
        const record = parseConversationPreparedRequestRecord({
            source: selection.source,
            runtime,
            request_receipt: prepared.receipt,
            generation_id: await deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
            response_turn_id: await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
        });
        let committed = false;
        const dispatch = () =>
            driver.executeCommittedIndexedTextRequest({
                selection,
                record,
                options: { model: target.model },
                assert_committed: async () => {
                    if (!committed) throw new Error('Actual indexed target is not committed');
                },
            });
        await expect(dispatch()).rejects.toThrow('not committed');
        expect(bodies).toHaveLength(0);
        committed = true;
        const decoded = await dispatch();
        expect(bodies).toHaveLength(1);
        expect(JSON.stringify(bodies[0])).toContain('data:image/png;base64,');
        expect(JSON.stringify(bodies[0])).toContain('read_image');
        expect(JSON.stringify(bodies[0])).not.toContain('Cold unselected text');
        expect(decoded.generation.request_receipt).toEqual(prepared.receipt);
        expect(decoded.turns[0]?.blocks).toContainEqual(
            expect.objectContaining({
                type: 'tool_call',
                definition_id: 'definition:selected-tool',
                tool_name: 'read_image',
            }),
        );
    });

    it('projects selected pages to the configured full Responses body and dispatches only after the durable fence', async () => {
        const { document, selection, runtime, reads, store } = await selectedText();
        const target = { provider: 'openai', model: 'gpt-5.4' };
        expect(compileOpenAIResponsesIndexedSelectedText(selection, target)).toEqual(
            compileOpenAIResponsesConversation(document, target),
        );
        expect(reads).not.toContain('blocks:block:cold');
        const bodies: unknown[] = [];
        const driver = new OpenAIDriver({ apiKey: 'test' });
        driver.service = new OpenAI({
            apiKey: 'test',
            baseURL: 'https://indexed.test/v1',
            fetch: async (_url, init) => {
                bodies.push(JSON.parse(String(init?.body)));
                return Response.json(responseBody());
            },
        });
        const prepared = await driver.prepareIndexedTextRequest({
            selection,
            runtime,
            options: { model: target.model },
            stream: false,
        });
        const full = compileOpenAIResponsesConversation(document, target);
        expect(prepared.receipt).toEqual(
            await createRequestReceipt(
                document,
                runtime,
                { ...target, protocol: OPENAI_RESPONSES_PROTOCOL, adapter_version: OPENAI_RESPONSES_ADAPTER_VERSION },
                providerJsonValue(prepared.payload),
                full.mappings,
                [],
            ),
        );
        const record = parseConversationPreparedRequestRecord({
            source: selection.source,
            runtime,
            request_receipt: prepared.receipt,
            generation_id: await deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
            response_turn_id: await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
        });
        let committed = false;
        const dispatch = () =>
            driver.executeCommittedIndexedTextRequest({
                selection,
                record,
                options: { model: target.model },
                assert_committed: async () => {
                    if (!committed) throw new Error('indexed handoff not committed');
                },
            });
        await expect(dispatch()).rejects.toThrow('indexed handoff not committed');
        expect(bodies).toHaveLength(0);
        committed = true;
        const decoded = await dispatch();
        expect(bodies).toHaveLength(1);
        expect(bodies[0]).toMatchObject({ model: target.model, input: [{ role: 'user', content: 'Selected text' }] });
        expect(await fingerprintJson(providerJsonValue(bodies[0]))).toBe(prepared.receipt.request_fingerprint);
        expect(JSON.stringify(bodies[0])).not.toContain('Cold unselected text');
        expect(decoded.generation.request_receipt).toEqual(prepared.receipt);
        expect(decoded.turns[0]?.blocks).toContainEqual(expect.objectContaining({ type: 'text', text: 'Done' }));
        const output = decodedResponseBatchFromAcceptedRecord(record, decoded, {
            operation_id: runtime.response_operation_id,
            recorded_at: decoded.generation.timestamps.completed_at ?? new Date().toISOString(),
        });
        const responded = appendConversationRecords(document, output.batch, output.options).document;
        const nextTurn = createUserTurn({
            id: 'turn:next-user',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: new Date().toISOString() },
            provenance: { type: 'received' },
            model_visibility: 'include',
            blocks: [
                createTextBlock({ id: 'block:next-user', text: 'Continue with indexed history.', format: 'plain' }),
            ],
        });
        const continued = appendConversationRecords(
            responded,
            {
                turns: [nextTurn],
                context_entries: [{ id: 'entry:next-user', type: 'source_turn', turn_id: nextTurn.id }],
            },
            {
                expected_revision: responded.revision,
                operation_id: 'operation:next-user',
                payload_fingerprint: 'sha256:next',
                recorded_at: nextTurn.timestamps.recorded_at,
            },
        ).document;
        const nextSnapshot = await stageIndexedConversationSnapshot(continued, undefined, store);
        const nextSelected = await loadIndexedSelectedTextContext(store, nextSnapshot.root, nextSnapshot.locator);
        const nextCompiled = compileOpenAIResponsesIndexedSelectedText(nextSelected, target);
        expect(nextCompiled).toEqual(compileOpenAIResponsesConversation(continued, target));
        expect(JSON.stringify(nextCompiled.conversation)).toContain('Continue with indexed history.');
        const replayed = nextSelected.turns.find((turn) => turn.header.id === record.response_turn_id);
        if (!replayed) throw new Error('Generated Responses turn is missing from selected pages');
        const replayIndex = replayed.selected_blocks.findIndex((block) => block.type === 'native_replay');
        if (replayIndex < 0) throw new Error('Generated Responses turn lost replay evidence');
        const replayId = replayed.selected_blocks[replayIndex]?.id;
        const missingStore: IndexedConversationRecordStore = {
            ...store,
            async readRecord(value) {
                if (value.id === replayId) throw new Error('missing indexed replay record');
                return store.readRecord(value);
            },
        };
        await expect(
            loadIndexedSelectedTextContext(missingStore, nextSnapshot.root, nextSnapshot.locator),
        ).rejects.toThrow('missing indexed replay record');
        const changed = structuredClone(nextSelected);
        const changedTurn = changed.turns.find((turn) => turn.header.id === record.response_turn_id);
        const changedReplay = changedTurn?.selected_blocks[replayIndex];
        if (changedReplay?.type !== 'native_replay') throw new Error('Indexed replay fixture is missing');
        changedReplay.dependencies.block_ids.push('block:missing-dependency');
        expect(() => compileOpenAIResponsesIndexedSelectedText(changed, target)).toThrow();
        const noWitness = structuredClone(nextSelected);
        delete noWitness.generation_witnesses[record.generation_id];
        expect(() => compileOpenAIResponsesIndexedSelectedText(noWitness, target)).toThrow(
            'accepted generation dependencies',
        );
        const protectedMismatch = structuredClone(nextSelected);
        const protectedTurn = protectedMismatch.turns.find((turn) => turn.header.id === record.response_turn_id);
        const protectedReplay = protectedTurn?.selected_blocks[replayIndex];
        if (protectedReplay?.type !== 'native_replay') throw new Error('Indexed protected replay fixture is missing');
        protectedReplay.dependency_policy = undefined;
        protectedReplay.compatibility_scope.model = 'different-model';
        expect(() => compileOpenAIResponsesIndexedSelectedText(protectedMismatch, target)).toThrow();
        await expect(
            driver.executeCommittedIndexedTextRequest({
                selection,
                record,
                options: { model: 'gpt-5.3' },
                assert_committed: async () => undefined,
            }),
        ).rejects.toThrow('differs from its durable prepared receipt');
        // Reasoning models omit temperature. Mutate a supported option that actually changes
        // the native request, then prove the committed receipt blocks it before transport.
        const changedOptions = { model: target.model, model_options: { max_tokens: 17 } };
        const changedPrepared = await driver.prepareIndexedTextRequest({
            selection,
            runtime,
            options: changedOptions,
            stream: false,
        });
        expect(changedPrepared.payload.max_output_tokens).toBe(17);
        expect(changedPrepared.receipt.request_fingerprint).not.toBe(prepared.receipt.request_fingerprint);
        await expect(
            driver.executeCommittedIndexedTextRequest({
                selection,
                record,
                options: changedOptions,
                assert_committed: async () => undefined,
            }),
        ).rejects.toThrow('differs from its durable prepared receipt');
        const cancelled = new AbortController();
        cancelled.abort();
        await expect(
            driver.executeCommittedIndexedTextRequest({
                selection,
                record,
                options: { model: target.model },
                assert_committed: async () => undefined,
                signal: cancelled.signal,
            }),
        ).rejects.toThrow();
        expect(bodies).toHaveLength(1);
    });

    it('blocks hidden provider state and unsupported output or selected content before transport', async () => {
        const { selection, runtime } = await selectedText();
        const driver = new OpenAIDriver({ apiKey: 'test' });
        await expect(
            driver.prepareIndexedTextRequest({
                selection,
                runtime,
                options: { model: 'gpt-5.4', model_options: { extra_body: { previous_response_id: 'unowned' } } },
                stream: false,
            }),
        ).rejects.toThrow('overrides canonical request state');
        await expect(
            driver.prepareIndexedTextRequest({
                selection,
                runtime,
                options: { model: 'gpt-5.4', result_schema: { type: 'object' } },
                stream: false,
            }),
        ).rejects.toThrow('unsupported output or history transformation');
        const changed = structuredClone(selection);
        changed.context.active_tool_definition_ids.push('tool:unselected');
        await expect(
            driver.prepareIndexedTextRequest({
                selection: changed,
                runtime,
                options: { model: 'gpt-5.4' },
                stream: false,
            }),
        ).rejects.toThrow();
    });
});

describe.each([
    ['Chat', () => new OpenAIChatCompletionsDriver({ apiKey: 'test', endpoint: 'https://indexed.test/v1' })],
    ['Responses', () => new OpenAIDriver({ apiKey: 'test' })],
] as const)('indexed %s host capability boundary', (_name, createDriver) => {
    it('captures the actual external resolver across prepare and committed recompile awaits', async () => {
        const { selection, runtime, bytes } = await selectedText(true, true);
        const driver = createDriver();
        let entered = () => {};
        let release = () => {};
        let started = new Promise<void>((resolve) => {
            entered = resolve;
        });
        let gate = new Promise<void>((resolve) => {
            release = resolve;
        });
        const original = vi.fn<ResolveConversationAsset>(async function* (asset) {
            expect(asset).toEqual(selection.assets['asset:selected-image']);
            entered();
            await gate;
            yield bytes;
        });
        const substitute = vi.fn<ResolveConversationAsset>(async function* () {
            yield new Uint8Array();
        });
        const capability = { resolve_canonical_asset: original } satisfies CanonicalHostCapabilities;
        const extraBody = { nested: { token: 'original' } };
        const input = {
            selection,
            runtime,
            options: { model: 'gpt-5.4', model_options: { extra_body: extraBody } },
            stream: false,
        };
        const preparation = driver.prepareIndexedTextRequest(input, capability);
        await started;
        capability.resolve_canonical_asset = substitute;
        extraBody.nested.token = 'late';
        release();
        const prepared = await preparation;
        expect(JSON.stringify(prepared.payload)).toContain('data:image/png;base64,');
        expect(JSON.stringify(prepared.payload)).toContain('original');
        expect(JSON.stringify(prepared.payload)).not.toContain('late');
        extraBody.nested.token = 'original';
        expect(original).toHaveBeenCalledTimes(1);
        expect(substitute).not.toHaveBeenCalled();
        const record = parseConversationPreparedRequestRecord({
            source: selection.source,
            runtime,
            request_receipt: prepared.receipt,
            generation_id: await deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
            response_turn_id: await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
        });
        started = new Promise<void>((resolve) => {
            entered = resolve;
        });
        gate = new Promise<void>((resolve) => {
            release = resolve;
        });
        capability.resolve_canonical_asset = original;
        const committed = vi.fn(async () => {
            throw new Error('Owned current-head fence');
        });
        const execution = driver.executeCommittedIndexedTextRequest(
            {
                selection,
                record,
                options: input.options,
                assert_committed: committed,
            },
            capability,
        );
        const rejected = expect(execution).rejects.toThrow('Owned current-head fence');
        await started;
        capability.resolve_canonical_asset = substitute;
        release();
        await rejected;
        expect(original).toHaveBeenCalledTimes(2);
        expect(substitute).not.toHaveBeenCalled();
        expect(committed).toHaveBeenCalledOnce();
    });

    it('rejects accessor callbacks and JSON option nominations before resolver or transport work', async () => {
        const { selection, runtime, bytes } = await selectedText(true, true);
        const driver = createDriver();
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield bytes;
        });
        const getter = vi.fn(() => resolver);
        const hostile: CanonicalHostCapabilities = {};
        Object.defineProperty(hostile, 'resolve_canonical_asset', { get: getter });
        const input = { selection, runtime, options: { model: 'gpt-5.4' }, stream: false };
        await expect(driver.prepareIndexedTextRequest(input, hostile)).rejects.toThrow('own data properties');
        expect(getter).not.toHaveBeenCalled();
        const nominated: ExecutionOptions = { model: input.options.model };
        Object.defineProperty(nominated, 'resolve_canonical_asset', { value: 'serialized-callback', enumerable: true });
        await expect(
            driver.prepareIndexedTextRequest(
                { ...input, options: nominated },
                {
                    resolve_canonical_asset: resolver,
                },
            ),
        ).rejects.toThrow('per-call host capabilities');
        expect(resolver).not.toHaveBeenCalled();
        const prepared = await driver.prepareIndexedTextRequest(input, { resolve_canonical_asset: resolver });
        const record = parseConversationPreparedRequestRecord({
            source: selection.source,
            runtime,
            request_receipt: prepared.receipt,
            generation_id: await deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
            response_turn_id: await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
        });
        resolver.mockClear();
        const committed = vi.fn(async () => {});
        const executeInput = { selection, record, options: input.options, assert_committed: committed };
        await expect(driver.executeCommittedIndexedTextRequest(executeInput, hostile)).rejects.toThrow(
            'own data properties',
        );
        await expect(
            driver.executeCommittedIndexedTextRequest(
                { ...executeInput, options: nominated },
                {
                    resolve_canonical_asset: resolver,
                },
            ),
        ).rejects.toThrow('per-call host capabilities');
        expect(getter).not.toHaveBeenCalled();
        expect(resolver).not.toHaveBeenCalled();
        expect(committed).not.toHaveBeenCalled();
    });
});

describe('committed indexed Responses received failure custody', () => {
    it.each([false, true])(
        'retains actual failed indexed generation without whole history (output=%s)',
        async (output) => {
            const f = await selectedText();
            const native: OpenAI.Responses.Response = {
                ...responseBody(),
                status: 'failed',
                error: { code: 'server_error', message: 'Actual indexed failure' },
                output: output ? responseBody().output : [],
                output_text: output ? 'Done' : '',
            };
            const bodies: unknown[] = [];
            const driver = new OpenAIDriver({ apiKey: 'offline' });
            driver.service = new OpenAI({
                apiKey: 'offline',
                baseURL: 'https://indexed.invalid/v1',
                fetch: async (_url, init) => {
                    bodies.push(JSON.parse(String(init?.body)));
                    return Response.json(native);
                },
            });
            const prepared = await driver.prepareIndexedTextRequest({
                selection: f.selection,
                runtime: f.runtime,
                options: { model: 'gpt-5.4' },
                stream: false,
            });
            const record = parseConversationPreparedRequestRecord({
                source: f.selection.source,
                runtime: f.runtime,
                request_receipt: prepared.receipt,
                generation_id: await deriveConversationId('generation', f.runtime.request_id, f.runtime.attempt_id),
                response_turn_id: await deriveConversationId('turn', f.runtime.response_operation_id, 'response', '0'),
            });
            let committed = false;
            const dispatch = () =>
                driver.executeCommittedIndexedTextRequest({
                    selection: f.selection,
                    record,
                    options: { model: 'gpt-5.4' },
                    assert_committed: async () => {
                        if (!committed) throw new Error('Actual indexed failure target not committed');
                    },
                });
            await expect(dispatch()).rejects.toThrow('not committed');
            expect(bodies).toHaveLength(0);
            committed = true;
            let error: unknown;
            try {
                await dispatch();
            } catch (failure: unknown) {
                error = failure;
            }
            expect(error).toBeInstanceOf(Error);
            if (!(error instanceof Error)) throw new Error('Indexed terminal did not remain an error');
            expect(error.message).toContain('Actual indexed failure');
            const evidence = canonicalFailedExecution(error);
            if (!evidence) throw new Error('Indexed received failure lost its actual execution evidence');
            await assertCanonicalFailedExecutionMatchesPreparedRecord(evidence, record);
            expect(evidence.prepared_request).toEqual(record);
            expect('document' in evidence).toBe(false);
            expect(evidence.decoded_response.payload_fingerprint).toBe(await fingerprintJson(native));
            expect(evidence.decoded_response.generation).toMatchObject({
                status: 'failed',
                usage: { input_tokens: 4, output_tokens: 2, total_tokens: 6 },
                metadata: { openai_responses_failure: native },
            });
            expect(evidence.decoded_response.turns).toHaveLength(output ? 1 : 0);
            expect('accepted_output' in evidence).toBe(false);
            expect(bodies).toHaveLength(1);
            expect(JSON.stringify(bodies[0])).not.toContain('Cold unselected text');
            expect(await fingerprintJson(providerJsonValue(bodies[0]))).toBe(
                record.request_receipt.request_fingerprint,
            );
        },
    );
});
