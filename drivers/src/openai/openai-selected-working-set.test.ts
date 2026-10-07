import {
    appendConversationRecords,
    applyContextChange,
    applyConversationSliceEdit,
    type ConversationDocument,
    ConversationInsertedTurnSchema,
    type ConversationPreparedRequestRecord,
    type ConversationTurn,
    createConversationDocument,
    createPreparedRequestSourceViewArtifacts,
    deriveConversationId,
    fingerprintJson,
    hashContentBytes,
    planContextChange,
    planConversationSliceEdit,
    type RequestSourceWorkingSet,
    resolveConversationSelection,
    resolvePreparedRequestSourceWorkingSet,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import {
    canonicalConversationTurnNumber,
    createRequestReceipt,
    providerJsonValue,
} from '../conversation/canonical-runtime.js';
import {
    buildOpenAIChatCompletionsPayload,
    OpenAISDKChatCompletionsProtocol,
    prepareOpenAIChatCompletionsConversation,
    projectOpenAIChatCompletionsHistory,
} from './openai_chat_completions.js';
import {
    assertOpenAIChatSelectedPreparedRequestEvidence,
    compileOpenAIChatCompletionsConversation,
    compileOpenAIChatSelectedWorkingSet,
    OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    prepareOpenAIChatCanonicalState,
} from './openai-chat-conversation-adapter.js';

const AT = '2026-09-30T00:00:00.000Z';
const TARGET = { provider: 'openai', model: 'gpt-test' };

async function workingSetFor(
    document: ConversationDocument,
    name: string,
): Promise<{
    record: ConversationPreparedRequestRecord;
    workingSet: RequestSourceWorkingSet;
    nativePayload: ReturnType<typeof providerJsonValue>;
}> {
    const runtime = {
        conversation_id: document.id,
        request_id: `request:${name}`,
        attempt_id: `attempt:${name}`,
        input_operation_id: `input:${name}`,
        response_operation_id: `response:${name}`,
        recorded_at: AT,
        purpose: 'interaction',
    };
    const full = compileOpenAIChatCompletionsConversation(document, TARGET);
    const toolDefinitions = document.context.active_tool_definition_ids.map((id) => {
        const definition = document.tool_definitions[id];
        if (definition === undefined) throw new Error(`Missing selected tool ${id}`);
        return definition;
    });
    const options = { model: TARGET.model };
    const nativeConversation = prepareOpenAIChatCompletionsConversation(
        projectOpenAIChatCompletionsHistory(full.conversation, options, canonicalConversationTurnNumber(document)),
        { model: options.model, tools: toolDefinitions },
    );
    const nativePayload = providerJsonValue(
        buildOpenAIChatCompletionsPayload(
            nativeConversation,
            options,
            {},
            options.model,
            false,
            TARGET.provider,
            toolDefinitions,
        ),
    );
    const receipt = await createRequestReceipt(
        document,
        runtime,
        {
            ...TARGET,
            protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
        },
        nativePayload,
        full.mappings,
        toolDefinitions,
    );
    const prepared = {
        document,
        record: {
            source: { conversation_id: document.id, revision: document.revision },
            runtime,
            request_receipt: receipt,
            generation_id: await deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
            response_turn_id: await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
        },
    };
    const artifacts = await createPreparedRequestSourceViewArtifacts(prepared, `archive/${name}`);
    const record: ConversationPreparedRequestRecord = {
        ...prepared.record,
        request_receipt: {
            ...receipt,
            source_view: {
                version: 1,
                completeness: 'selected_content_unverified',
                source: prepared.record.source,
                context_revision: document.context.revision,
                manifest_storage_key: artifacts.manifest.storage_key,
                manifest_content_hash: artifacts.manifest.content_hash,
                manifest_size_bytes: artifacts.manifest.bytes.byteLength,
                context_fingerprint: receipt.context_fingerprint,
                request_fingerprint: receipt.request_fingerprint,
            },
        },
    };
    const bytes = new Map(
        [artifacts.manifest, ...artifacts.index_pages, ...artifacts.segments].map((item) => [
            item.storage_key,
            item.bytes,
        ]),
    );
    const workingSet = await resolvePreparedRequestSourceWorkingSet(record, {
        read: async (key) => {
            const value = bytes.get(key);
            if (value === undefined) throw new Error(`Missing source view ${key}`);
            return Uint8Array.from(value);
        },
    });
    return { record, workingSet, nativePayload };
}

async function attestSelectedPayload(
    document: ConversationDocument,
    record: ConversationPreparedRequestRecord,
    workingSet: RequestSourceWorkingSet,
): Promise<void> {
    await new OpenAISDKChatCompletionsProtocol({}).attestSelectedPreparedRequest({
        record,
        working_set: workingSet,
        runtime: structuredClone(record.runtime),
        options: { model: TARGET.model },
        provider: TARGET.provider,
        stream: false,
        current_turn: canonicalConversationTurnNumber(document),
    });
}

function receivedUser(id: string, blocks: Extract<ConversationTurn, { kind: 'user' }>['blocks']) {
    return {
        id,
        kind: 'user' as const,
        authority: 'ordinary' as const,
        status: 'completed' as const,
        timestamps: { recorded_at: AT },
        provenance: { type: 'received' as const },
        model_visibility: 'include' as const,
        blocks,
    };
}

describe('OpenAI Chat selected working-set projection', () => {
    it('binds a resolved selected view to the accepted request and owns inputs before hashing', async () => {
        const document = appendConversationRecords(
            createConversationDocument({ id: 'working-set:attestation', created_at: AT }),
            {
                turns: [
                    receivedUser('turn:attestation', [
                        { id: 'text:attestation', type: 'text', text: 'hello', format: 'plain' },
                    ]),
                ],
                context_entries: [{ id: 'entry:attestation', type: 'source_turn', turn_id: 'turn:attestation' }],
            },
            {
                expected_revision: 0,
                operation_id: 'append:attestation',
                payload_fingerprint: 'sha256:input',
                recorded_at: AT,
            },
        ).document;
        const { record, workingSet, nativePayload } = await workingSetFor(document, 'attestation');
        const exact = {
            record,
            working_set: workingSet,
            runtime: structuredClone(record.runtime),
            target: {
                ...TARGET,
                protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
            },
            native_payload: nativePayload,
        };
        await expect(assertOpenAIChatSelectedPreparedRequestEvidence(exact)).resolves.toBeUndefined();
        const protocol = new OpenAISDKChatCompletionsProtocol({});
        await expect(attestSelectedPayload(document, record, workingSet)).resolves.toBeUndefined();
        await expect(
            protocol.attestSelectedPreparedRequest({
                record,
                working_set: workingSet,
                runtime: structuredClone(record.runtime),
                options: { model: TARGET.model, model_options: { max_tokens: 19 } },
                provider: TARGET.provider,
                stream: false,
                current_turn: canonicalConversationTurnNumber(document),
            }),
        ).rejects.toThrow('differs from accepted prepared evidence');
        await expect(
            protocol.attestSelectedPreparedRequest({
                record,
                working_set: workingSet,
                runtime: structuredClone(record.runtime),
                options: { model: TARGET.model },
                provider: TARGET.provider,
                stream: true,
                current_turn: canonicalConversationTurnNumber(document),
            }),
        ).rejects.toThrow('differs from accepted prepared evidence');

        const changedPayload = structuredClone(exact);
        changedPayload.native_payload = { changed: true };
        await expect(assertOpenAIChatSelectedPreparedRequestEvidence(changedPayload)).rejects.toThrow(
            'differs from accepted prepared evidence',
        );
        const changedRuntime = structuredClone(exact);
        changedRuntime.runtime.request_id = 'request:other';
        await expect(assertOpenAIChatSelectedPreparedRequestEvidence(changedRuntime)).rejects.toThrow(
            'differs from accepted prepared evidence',
        );
        for (const key of ['attempt_id', 'input_operation_id', 'response_operation_id'] as const) {
            const changed = structuredClone(exact);
            changed.runtime[key] = `${key}:other`;
            await expect(assertOpenAIChatSelectedPreparedRequestEvidence(changed)).rejects.toThrow(
                'differs from accepted prepared evidence',
            );
        }
        const changedGeneration = structuredClone(exact);
        changedGeneration.record.generation_id = 'generation:other';
        await expect(assertOpenAIChatSelectedPreparedRequestEvidence(changedGeneration)).rejects.toThrow(
            'differs from accepted prepared evidence',
        );
        const changedResponseTurn = structuredClone(exact);
        changedResponseTurn.record.response_turn_id = 'turn:other';
        await expect(assertOpenAIChatSelectedPreparedRequestEvidence(changedResponseTurn)).rejects.toThrow(
            'differs from accepted prepared evidence',
        );
        const changedToolSet = structuredClone(exact);
        changedToolSet.record.request_receipt.tool_set_fingerprint = 'sha256:other';
        await expect(assertOpenAIChatSelectedPreparedRequestEvidence(changedToolSet)).rejects.toThrow(
            'differs from accepted prepared evidence',
        );
        const changedTarget = structuredClone(exact);
        changedTarget.target.model = 'gpt-other';
        await expect(assertOpenAIChatSelectedPreparedRequestEvidence(changedTarget)).rejects.toThrow(
            'differs from accepted prepared evidence',
        );
        const concurrent = structuredClone(exact);
        const pending = assertOpenAIChatSelectedPreparedRequestEvidence(concurrent);
        concurrent.runtime.request_id = 'request:mutated';
        concurrent.native_payload = { mutated: true };
        await expect(pending).resolves.toBeUndefined();
    });

    it('matches a full ordinary context with a partial block selection and inline image', async () => {
        const imageHash = (await hashContentBytes(Uint8Array.from([0, 1, 2, 3]))).content_hash;
        const initial = createConversationDocument({ id: 'working-set:ordinary', created_at: AT });
        const document = appendConversationRecords(
            initial,
            {
                turns: [
                    receivedUser('user:one', [
                        { id: 'text:excluded', type: 'text', text: 'cold', format: 'plain' },
                        { id: 'text:selected', type: 'text', text: 'hello', format: 'plain' },
                        { id: 'image:selected', type: 'image', asset_id: 'asset:image' },
                    ]),
                ],
                assets: [
                    {
                        id: 'asset:image',
                        kind: 'image',
                        mime_type: 'image/png',
                        storage: { type: 'inline_base64', data: 'AAECAw==' },
                        provenance: { type: 'received', source_turn_id: 'user:one' },
                        byte_length: 4,
                        content_hash: imageHash,
                        created_at: AT,
                    },
                ],
                context_entries: [
                    {
                        id: 'entry:one',
                        type: 'source_turn',
                        turn_id: 'user:one',
                        block_ids: ['text:selected', 'image:selected'],
                    },
                ],
            },
            { expected_revision: 0, operation_id: 'append:one', payload_fingerprint: 'sha256:input', recorded_at: AT },
        ).document;
        const { record, workingSet } = await workingSetFor(document, 'ordinary');
        await expect(attestSelectedPayload(document, record, workingSet)).resolves.toBeUndefined();
        expect(workingSet.turns[0]?.selected_block_positions).toEqual([1, 2]);
        expect(workingSet.turns[0]?.source_block_count).toBe(3);
        expect(compileOpenAIChatSelectedWorkingSet(workingSet, TARGET)).toEqual(
            compileOpenAIChatCompletionsConversation(document, TARGET),
        );
        const unsupportedCaption = structuredClone(workingSet);
        const image = unsupportedCaption.turns[0]?.selected_blocks.find((block) => block.type === 'image');
        if (image?.type !== 'image') throw new Error('Expected selected image');
        image.caption = 'unpreservable';
        expect(() => compileOpenAIChatSelectedWorkingSet(unsupportedCaption, TARGET)).toThrow(
            'cannot preserve image block image:selected caption',
        );
    });

    it('matches a retained tool call and result without rebuilding a full document', async () => {
        const state = await prepareOpenAIChatCanonicalState({
            conversation: {
                _is_openai_chat_completions: true,
                messages: [
                    { role: 'user', content: 'weather' },
                    {
                        role: 'assistant',
                        content: 'checking',
                        tool_calls: [
                            {
                                id: 'call:weather',
                                type: 'function',
                                function: { name: 'lookup', arguments: '{"city":"Tokyo"}' },
                            },
                        ],
                    },
                    { role: 'tool', tool_call_id: 'call:weather', content: 'sunny' },
                ],
            },
            prompt: { _is_openai_chat_completions: true, messages: [] },
            options: {
                model: 'gpt-test',
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
                conversation_runtime: {
                    conversation_id: 'working-set:tool',
                    request_id: 'import:request',
                    attempt_id: 'import:attempt',
                    input_operation_id: 'import:input',
                    response_operation_id: 'import:response',
                    recorded_at: AT,
                },
            },
            provider: 'openai',
        });
        const { record, workingSet } = await workingSetFor(state.document, 'tool');
        await expect(attestSelectedPayload(state.document, record, workingSet)).resolves.toBeUndefined();
        expect(compileOpenAIChatSelectedWorkingSet(workingSet, TARGET)).toEqual(
            compileOpenAIChatCompletionsConversation(state.document, TARGET),
        );
        const missingGenerationWitness = structuredClone(workingSet);
        const replay = missingGenerationWitness.turns
            .flatMap((turn) => turn.selected_blocks)
            .find((block) => block.type === 'native_replay' && block.protocol === OPENAI_CHAT_COMPLETIONS_PROTOCOL);
        if (replay?.type !== 'native_replay') throw new Error('Expected retained OpenAI replay');
        delete replay.dependency_policy;
        expect(() => compileOpenAIChatSelectedWorkingSet(missingGenerationWitness, TARGET)).toThrow(
            'needs its retained generation witness',
        );
        const missingReplayDependency = structuredClone(workingSet);
        const dependentReplay = missingReplayDependency.turns
            .flatMap((turn) => turn.selected_blocks)
            .find((block) => block.type === 'native_replay' && block.protocol === OPENAI_CHAT_COMPLETIONS_PROTOCOL);
        if (dependentReplay?.type !== 'native_replay') throw new Error('Expected retained OpenAI replay');
        delete dependentReplay.dependency_policy;
        dependentReplay.compatibility_scope.model = 'gpt-test';
        dependentReplay.dependencies.turn_ids.push('turn:missing');
        expect(() => compileOpenAIChatSelectedWorkingSet(missingReplayDependency, TARGET)).toThrow(
            'needs an unavailable dependency witness',
        );
    });

    it('matches a D2 slice replacement while retaining original selected block positions', async () => {
        const initial = createConversationDocument({ id: 'working-set:slice', created_at: AT });
        const source = appendConversationRecords(
            initial,
            {
                turns: [
                    receivedUser('slice:original', [
                        { id: 'slice:text', type: 'text', text: 'a😀bcdef', format: 'plain' },
                        { id: 'slice:unmatched', type: 'text', text: 'keep', format: 'plain' },
                    ]),
                ],
                context_entries: [{ id: 'slice:entry', type: 'source_turn', turn_id: 'slice:original' }],
            },
            {
                expected_revision: 0,
                operation_id: 'slice:append',
                payload_fingerprint: 'sha256:source',
                recorded_at: AT,
            },
        ).document;
        const block = source.turns[0]?.blocks[0];
        if (block === undefined) throw new Error('Missing slice source block');
        const selected = await resolveConversationSelection(source, {
            conversation: { conversation_id: source.id, revision: source.revision },
            expected_context_revision: source.context.revision,
            selector: {
                source: { kind: 'all' },
                filters: { block_ids: [block.id] },
                subselections: [
                    {
                        kind: 'text_range',
                        entry_id: 'slice:entry',
                        block_id: block.id,
                        expected_block_fingerprint: await fingerprintJson(block),
                        range: { start_code_point: 1, end_code_point: 3 },
                    },
                ],
            },
        });
        if (selected.kind !== 'selected') throw new Error('Expected precise slice selection');
        const request = {
            version: 2 as const,
            operation_id: 'slice:replace',
            conversation: { conversation_id: source.id, revision: source.revision },
            expected_context_revision: source.context.revision,
            recorded_at: AT,
            command: {
                kind: 'replace' as const,
                selection: selected.selection,
                replacement_turn: ConversationInsertedTurnSchema.parse({
                    ...receivedUser('slice:replacement', [
                        { id: 'slice:summary', type: 'text', text: 'summary', format: 'plain' },
                    ]),
                    provenance: { type: 'inserted', operation_id: 'slice:replace' },
                }),
                fidelity: 'semantic' as const,
                placement: { mode: 'first_selected' as const, causal_order: 'contiguous' as const },
            },
        };
        const plan = await planConversationSliceEdit(source, request);
        const edited = await applyConversationSliceEdit(source, {
            ...request,
            expected_source_fingerprint: plan.operation.source_fingerprint,
        });
        const { record, workingSet } = await workingSetFor(edited.document, 'slice');
        await expect(attestSelectedPayload(edited.document, record, workingSet)).resolves.toBeUndefined();
        expect(workingSet.turns.some((turn) => turn.completeness === 'selected_blocks')).toBe(true);
        expect(compileOpenAIChatSelectedWorkingSet(workingSet, TARGET)).toEqual(
            compileOpenAIChatCompletionsConversation(edited.document, TARGET),
        );
    });

    it('matches a retained compaction replacement without loading its cold source turn', async () => {
        const initial = createConversationDocument({ id: 'working-set:compaction', created_at: AT });
        const source = appendConversationRecords(
            initial,
            {
                turns: [
                    receivedUser('cold:turn', [
                        { id: 'cold:text', type: 'text', text: 'cold original', format: 'plain' },
                    ]),
                    receivedUser('warm:turn', [
                        { id: 'warm:text', type: 'text', text: 'warm current', format: 'plain' },
                    ]),
                ],
                context_entries: [
                    { id: 'cold:entry', type: 'source_turn', turn_id: 'cold:turn' },
                    { id: 'warm:entry', type: 'source_turn', turn_id: 'warm:turn' },
                ],
            },
            {
                expected_revision: 0,
                operation_id: 'compact:append',
                payload_fingerprint: 'sha256:source',
                recorded_at: AT,
            },
        ).document;
        const plan = await planContextChange(source, {
            expected_revision: source.revision,
            expected_context_revision: source.context.revision,
            entry_ids: ['cold:entry'],
        });
        const edited = await applyContextChange(source, {
            operation_id: 'compact:replace',
            expected_revision: source.revision,
            expected_context_revision: source.context.revision,
            expected_source_fingerprint: plan.source_fingerprint,
            recorded_at: AT,
            entry_ids: plan.entry_ids,
            proposal: {
                kind: 'replace_with_compaction',
                compaction_id: 'compaction:one',
                strategy: { id: 'test-summary', version: '1', configuration_fingerprint: 'sha256:config' },
                replacement_turns: [
                    {
                        ...receivedUser('summary:turn', [
                            { id: 'summary:text', type: 'text', text: 'condensed', format: 'plain' },
                        ]),
                        kind: 'agent',
                        provenance: {
                            type: 'derived',
                            derivation_id: 'compaction:one',
                            source_turn_ids: plan.source_turn_ids,
                            source_hash: plan.source_fingerprint,
                        },
                    },
                ],
                fidelity: 'semantic',
                retained_asset_ids: [],
                generation_ids: [],
                placement: { mode: 'first_selected', causal_order: 'contiguous' },
            },
        });
        const { record, workingSet } = await workingSetFor(edited.document, 'compaction');
        await expect(attestSelectedPayload(edited.document, record, workingSet)).resolves.toBeUndefined();
        expect(workingSet.turns.some((turn) => turn.header.id === 'cold:turn')).toBe(false);
        expect(workingSet.replacement_turns.map((item) => item.projection.header.id)).toEqual(['summary:turn']);
        expect(compileOpenAIChatSelectedWorkingSet(workingSet, TARGET)).toEqual(
            compileOpenAIChatCompletionsConversation(edited.document, TARGET),
        );
    });
});
