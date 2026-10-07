import {
    appendConversationRecords,
    applyContextChange,
    type ConversationDocument,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    hashUtf8Content,
    planContextChange,
} from '@llumiverse/conversation';
import { ContextChangeRequestSchema } from '@llumiverse/conversation/schemas';
import { describe, expect, it } from 'vitest';
import {
    appendCanonicalPrompt,
    type CanonicalPromptRecords,
    canonicalToolDefinitions,
    createExecutedGeneration,
    createRequestReceipt,
    prepareCanonicalContext,
    type ResolvedConversationRuntimeContext,
} from './canonical-runtime.js';

const RECORDED_AT = '2026-09-30T00:00:00.000Z';
const INPUT_OPERATION_ID = 'operation:materialized-input';

const emptyRecords = (): CanonicalPromptRecords => ({
    turns: [],
    assets: [],
    context_entries: [],
    item_mappings: [],
});

describe('published historical canonical prompt recovery', () => {
    const tools = [
        { name: 'lookup', description: 'Lookup', input_schema: { type: 'object' as const } },
        { name: 'write', description: 'Write', input_schema: { type: 'object' as const } },
    ];

    async function acceptedWithProjectedTools(projectedCount: number): Promise<{
        document: ConversationDocument;
        runtime: ResolvedConversationRuntimeContext;
    }> {
        const initial = createConversationDocument({
            id: 'conversation:historical-selection',
            created_at: RECORDED_AT,
        });
        const proof = freshRuntime(initial, 'historical-selection');
        const input = await appendCanonicalPrompt(initial, emptyRecords(), proof, tools, { prompt: 'same' });
        const requestReceipt = await createRequestReceipt(
            input.document,
            proof,
            { provider: 'test-provider', protocol: 'test.protocol', model: 'model', adapter_version: 'test.v1' },
            { messages: [] },
            [],
            input.tool_definitions.slice(0, projectedCount),
        );
        const generation = await createExecutedGeneration({
            id: 'generation:historical-selection',
            runtime: proof,
            receipt: requestReceipt,
            provider: 'test-provider',
            protocol: 'test.protocol',
            adapter_version: 'test.v1',
            requested_model: 'model',
        });
        const response = appendConversationRecords(
            input.document,
            {
                generations: [generation],
                turns: [
                    {
                        id: 'turn:historical-response',
                        kind: 'agent',
                        authority: 'ordinary',
                        model_visibility: 'include',
                        status: 'completed',
                        timestamps: { recorded_at: RECORDED_AT },
                        provenance: { type: 'generated' },
                        generation_id: generation.id,
                        blocks: [{ id: 'block:historical-response', type: 'text', format: 'plain', text: 'answer' }],
                    },
                ],
            },
            {
                expected_revision: input.document.revision,
                operation_id: proof.response_operation_id,
                payload_fingerprint: 'sha256:historical-response',
                recorded_at: RECORDED_AT,
            },
        ).document;
        const historical = structuredClone(response);
        const inputReceipt = historical.operation_receipts[proof.input_operation_id];
        if (inputReceipt === undefined) throw new Error('missing accepted input receipt');
        delete inputReceipt.accepted_tool_selection;
        return { document: historical, runtime: proof };
    }

    it('reuses only the exact legacy prompt fingerprint and accepted response without changing receipts', async () => {
        const { document, runtime } = await acceptedWithProjectedTools(tools.length);
        const before = structuredClone(document.operation_receipts);
        const recovered = await appendCanonicalPrompt(document, emptyRecords(), runtime, tools, { prompt: 'same' });
        expect(recovered.document).toEqual(document);
        expect(recovered.document.operation_receipts).toEqual(before);
        await expect(
            appendCanonicalPrompt(document, emptyRecords(), runtime, tools, { prompt: 'changed' }),
        ).rejects.toThrow(/exact accepted response and tool selection/);
    });

    it('rejects a request-projected subset as evidence for the prior active tool selection', async () => {
        const { document, runtime } = await acceptedWithProjectedTools(1);
        await expect(
            appendCanonicalPrompt(document, emptyRecords(), runtime, tools, { prompt: 'same' }),
        ).rejects.toThrow(/exact accepted response and tool selection/);
    });

    it('rejects absent or unrelated accepted response evidence even when current tools happen to match', async () => {
        const { document, runtime } = await acceptedWithProjectedTools(tools.length);
        const missing = structuredClone(document);
        delete missing.operation_receipts[runtime.response_operation_id];
        await expect(
            appendCanonicalPrompt(missing, emptyRecords(), runtime, tools, { prompt: 'same' }),
        ).rejects.toThrow(/exact accepted response and tool selection/);
        const wrongChain = structuredClone(document);
        const receipt = wrongChain.operation_receipts[runtime.response_operation_id];
        if (receipt === undefined) throw new Error('missing response receipt');
        receipt.base_revision = 0;
        await expect(
            appendCanonicalPrompt(wrongChain, emptyRecords(), runtime, tools, { prompt: 'same' }),
        ).rejects.toThrow(/exact accepted response and tool selection|validation failed/);
    });
});

function runtime(document: ConversationDocument): ResolvedConversationRuntimeContext {
    return {
        conversation_id: document.id,
        request_id: 'request:next-model-call',
        attempt_id: 'attempt:next-model-call',
        input_operation_id: 'operation:unused-input',
        response_operation_id: 'operation:next-response',
        recorded_at: RECORDED_AT,
        purpose: 'interaction',
        materialized_input: {
            operation_id: INPUT_OPERATION_ID,
            result_revision: document.revision,
        },
    };
}

function materializedDocument(): ConversationDocument {
    const initial = createConversationDocument({ id: 'conversation:materialized', created_at: RECORDED_AT });
    const turn = createUserTurn({
        id: 'turn:materialized-input',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: RECORDED_AT },
        model_visibility: 'include',
        provenance: { type: 'received' },
        blocks: [
            createTextBlock({ id: 'block:materialized-a', text: 'first', format: 'plain' }),
            createTextBlock({ id: 'block:materialized-b', text: 'second', format: 'plain' }),
        ],
    });
    return appendConversationRecords(
        initial,
        {
            turns: [turn],
            context_entries: [{ id: 'context:materialized-input', type: 'source_turn', turn_id: turn.id }],
            active_tool_definition_ids: [],
        },
        {
            expected_revision: initial.revision,
            operation_id: INPUT_OPERATION_ID,
            payload_fingerprint: 'sha256:materialized-input',
            recorded_at: RECORDED_AT,
        },
    ).document;
}

function activeToolDocument(): ConversationDocument {
    const initial = createConversationDocument({ id: 'conversation:active-tools', created_at: RECORDED_AT });
    return appendConversationRecords(
        initial,
        {
            tool_definitions: [
                {
                    id: 'tool-definition:lookup:v1',
                    name: 'lookup',
                    version: 'v1',
                    input_schema: { type: 'object' },
                    result_capabilities: ['json'],
                },
                {
                    id: 'tool-definition:write:v2',
                    name: 'write',
                    version: 'v2',
                    input_schema: { type: 'object' },
                    result_capabilities: ['text', 'document'],
                },
            ],
            active_tool_definition_ids: ['tool-definition:write:v2', 'tool-definition:lookup:v1'],
        },
        {
            expected_revision: initial.revision,
            operation_id: 'operation:active-tools',
            payload_fingerprint: 'sha256:active-tools',
            recorded_at: RECORDED_AT,
        },
    ).document;
}

function freshRuntime(document: ConversationDocument, suffix: string): ResolvedConversationRuntimeContext {
    return {
        conversation_id: document.id,
        request_id: `request:${suffix}`,
        attempt_id: `attempt:${suffix}`,
        input_operation_id: `operation:${suffix}`,
        response_operation_id: `response:${suffix}`,
        recorded_at: RECORDED_AT,
        purpose: 'interaction',
    };
}

describe('canonical active tool precedence', () => {
    it('prepares an exact retained context without an empty append or legacy tool authority', async () => {
        const document = activeToolDocument();
        const prepared = await prepareCanonicalContext({
            options: {
                model: 'test-model',
                conversation: document,
                conversation_runtime: freshRuntime(document, 'direct-context'),
            },
            provider: 'test-provider',
            protocol: 'test.protocol',
            adapter_version: 'test.v1',
        });

        expect(prepared.document).toEqual(document);
        expect(prepared.document).not.toBe(document);
        expect(prepared.document.revision).toBe(document.revision);
        expect(prepared.document.operation_receipts).toEqual(document.operation_receipts);
        expect(prepared.request_document).toEqual(prepared.document);
        expect(prepared.tool_definitions).toEqual([
            document.tool_definitions['tool-definition:write:v2'],
            document.tool_definitions['tool-definition:lookup:v1'],
        ]);
        expect(prepared.tool_definitions[0]).not.toBe(document.tool_definitions['tool-definition:write:v2']);
    });

    it('rejects a missing active definition while preparing a retained context', async () => {
        const document = activeToolDocument();
        delete document.tool_definitions['tool-definition:write:v2'];

        await expect(
            prepareCanonicalContext({
                options: {
                    model: 'test-model',
                    conversation: document,
                    conversation_runtime: freshRuntime(document, 'missing-context-tool'),
                },
                provider: 'test-provider',
                protocol: 'test.protocol',
                adapter_version: 'test.v1',
            }),
        ).rejects.toThrow('Conversation document validation failed');
    });

    it('preserves the exact ordered canonical catalog when legacy tools are omitted', async () => {
        const document = activeToolDocument();
        const appended = await appendCanonicalPrompt(
            document,
            emptyRecords(),
            freshRuntime(document, 'preserve-tools'),
            undefined,
            null,
        );

        expect(appended.tool_definitions).toEqual([
            document.tool_definitions['tool-definition:write:v2'],
            document.tool_definitions['tool-definition:lookup:v1'],
        ]);
        expect(appended.document.context.active_tool_definition_ids).toEqual([
            'tool-definition:write:v2',
            'tool-definition:lookup:v1',
        ]);
        expect(appended.tool_definitions[0]).not.toBe(document.tool_definitions['tool-definition:write:v2']);
    });

    it('keeps explicit legacy arrays as exact replacement and clear operations', async () => {
        const document = activeToolDocument();
        const replacement = [{ name: 'search', description: 'Search', input_schema: { type: 'object' as const } }];
        const replacementDefinitions = await canonicalToolDefinitions(replacement);
        const replaced = await appendCanonicalPrompt(
            document,
            emptyRecords(),
            freshRuntime(document, 'replace-tools'),
            replacement,
            null,
        );
        expect(replaced.tool_definitions).toEqual(replacementDefinitions);
        expect(replaced.document.context.active_tool_definition_ids).toEqual(
            replacementDefinitions.map((definition) => definition.id),
        );

        const cleared = await appendCanonicalPrompt(
            document,
            emptyRecords(),
            freshRuntime(document, 'clear-tools'),
            [],
            null,
        );
        expect(cleared.tool_definitions).toEqual([]);
        expect(cleared.document.context.active_tool_definition_ids).toEqual([]);
    });

    it('rejects an active identity whose canonical definition is unavailable', async () => {
        const document = activeToolDocument();
        delete document.tool_definitions['tool-definition:write:v2'];

        await expect(
            appendCanonicalPrompt(document, emptyRecords(), freshRuntime(document, 'missing-tool'), undefined, null),
        ).rejects.toThrow('Active canonical tool definition tool-definition:write:v2 is missing');
    });
});

async function compactedMaterializedDocument(partial = false) {
    const original = materializedDocument();
    const proof = runtime(original);
    const turn = original.turns[0];
    const selected = partial ? turn.blocks.slice(0, 1) : turn.blocks;
    const assets = await Promise.all(
        selected.map(async (block, index) => {
            if (block.type !== 'text') throw new Error('Expected text input fixture');
            const integrity = await hashUtf8Content(block.text);
            return {
                id: `asset:materialized:${index}`,
                kind: 'text' as const,
                mime_type: 'text/plain',
                storage: { type: 'external' as const, resolver: 'test.archive', locator: { key: block.id } },
                provenance: { type: 'received' as const },
                content_hash: integrity.content_hash,
                byte_length: integrity.byte_length,
                created_at: RECORDED_AT,
            };
        }),
    );
    const definitions = await canonicalToolDefinitions([{ name: 'read_artifact', input_schema: { type: 'object' } }]);
    const definition = definitions[0];
    const staged = appendConversationRecords(
        original,
        {
            assets,
            tool_definitions: definitions,
            active_tool_definition_ids: [definition.id],
        },
        {
            operation_id: 'operation:archive-materialized',
            expected_revision: original.revision,
            payload_fingerprint: 'sha256:archive-materialized',
            recorded_at: RECORDED_AT,
        },
    ).document;
    const selection = {
        expected_revision: staged.revision,
        expected_context_revision: staged.context.revision,
        entry_ids: ['context:materialized-input'],
        ...(partial
            ? {
                  selected_entries: staged.context.entries,
                  selected_block_ids: { 'context:materialized-input': [selected[0].id] },
              }
            : {}),
    };
    const plan = await planContextChange(staged, selection);
    const applied = await applyContextChange(
        staged,
        ContextChangeRequestSchema.parse({
            ...selection,
            operation_id: 'operation:compact-materialized',
            expected_source_fingerprint: plan.source_fingerprint,
            recorded_at: RECORDED_AT,
            proposal: {
                kind: 'replace_with_compaction',
                compaction_id: 'compaction:materialized',
                strategy: { id: 'externalize-text', version: '1', configuration_fingerprint: 'sha256:config' },
                fidelity: 'retrievable',
                retained_asset_ids: assets.map((asset) => asset.id),
                generation_ids: [],
                accepted_asset_operation_id: 'operation:archive-materialized',
                placement: { mode: 'first_selected', causal_order: 'contiguous' },
                replacement_turns: [
                    {
                        ...createUserTurn({
                            id: 'turn:materialized-reference',
                            authority: 'ordinary',
                            status: 'completed',
                            timestamps: { recorded_at: RECORDED_AT },
                            model_visibility: 'include',
                            provenance: { type: 'received' },
                            blocks: [],
                        }),
                        kind: 'agent',
                        provenance: {
                            type: 'derived',
                            derivation_id: 'compaction:materialized',
                            source_turn_ids: plan.source_turn_ids,
                            source_hash: plan.source_fingerprint,
                            source_block_ids: selected.map((block) => block.id),
                        },
                        blocks: assets.map((asset, index) => ({
                            id: `block:materialized-reference:${index}`,
                            type: 'external_reference',
                            original_type: 'text',
                            asset_id: asset.id,
                            content_hash: asset.content_hash,
                            description: 'Exact accepted original',
                            preview: 'accepted original',
                            retrieval: {
                                capability: definition.name,
                                version: 1,
                                tool_definition_id: definition.id,
                                arguments: { asset_id: asset.id },
                            },
                        })),
                    },
                ],
            },
        }),
    );
    return { original, proof, document: applied.document };
}

async function prepareMaterialized(document: ConversationDocument, proof: ResolvedConversationRuntimeContext) {
    return prepareCanonicalContext({
        options: { model: 'test-model', conversation: document, conversation_runtime: proof },
        provider: 'test-provider',
        protocol: 'test.protocol',
        adapter_version: 'test.v1',
    });
}

describe('materialized canonical input proof', () => {
    for (const partial of [false, true]) {
        it(`preserves the original accepted input through ${partial ? 'partial compaction plus remainder' : 'full compaction'}`, async () => {
            const { original, proof, document } = await compactedMaterializedDocument(partial);
            const prepared = await prepareMaterialized(document, proof);
            expect(prepared.document).toEqual(document);
            expect(prepared.document.operation_receipts[INPUT_OPERATION_ID]).toEqual(
                original.operation_receipts[INPUT_OPERATION_ID],
            );
            expect(prepared.document.turns).toEqual(original.turns);
            expect(prepared.document.context.entries.some((entry) => entry.type === 'replacement_turn')).toBe(true);
            expect(prepared.document.revision).toBe(document.revision);
            expect(prepared.document.operation_receipts[proof.input_operation_id]).toBeUndefined();
            expect((await prepareMaterialized(document, proof)).document).toEqual(document);
        });
    }

    it('rejects provenance blocks outside the exact accepted partial compaction selection', async () => {
        const { document, proof } = await compactedMaterializedDocument(true);
        const compaction = document.compactions['compaction:materialized'];
        const replacement = compaction.replacement_turns[0];
        if (replacement.provenance.type !== 'derived') throw new Error('Expected derived fixture');
        expect(compaction.source.block_ids).toEqual(['block:materialized-a']);
        replacement.provenance.source_block_ids = ['block:materialized-b'];
        // Exercise the direct typed guard as well as semantic document validation; no mapping may silently drop it.
        await expect(appendCanonicalPrompt(document, emptyRecords(), proof, undefined, null)).rejects.toThrow(
            'foreign selected source block',
        );
    });

    it('memoizes a repeated unrelated derived DAG without treating its provenance as input authority', async () => {
        const { document: processed } = await compactedMaterializedDocument();
        const document = materializedDocument();
        const unrelated = createUserTurn({
            id: 'turn:unrelated-leaf',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: RECORDED_AT },
            model_visibility: 'include',
            provenance: { type: 'received' },
            blocks: [createTextBlock({ id: 'block:unrelated-leaf', text: 'unrelated', format: 'plain' })],
        });
        document.turns.push(unrelated);
        let previousIds = [unrelated.id];
        let sourceReads = 0;
        for (let layer = 0; layer < 24; layer += 1) {
            const currentIds: string[] = [];
            for (let branch = 0; branch < 2; branch += 1) {
                const id = `compaction:unrelated:${layer}:${branch}`;
                const turnId = `turn:unrelated:${layer}:${branch}`;
                const retained = structuredClone(processed.compactions['compaction:materialized']);
                retained.id = id;
                retained.operation_id = `operation:unrelated:${layer}:${branch}`;
                retained.source.turn_ids = [...previousIds];
                delete retained.source.block_ids;
                const replacement = retained.replacement_turns[0];
                replacement.id = turnId;
                replacement.blocks = [
                    createTextBlock({ id: `block:unrelated:${layer}:${branch}`, text: 'summary', format: 'plain' }),
                ];
                replacement.provenance = {
                    type: 'derived',
                    derivation_id: id,
                    source_turn_ids: [...previousIds],
                    source_hash: retained.source.source_fingerprint,
                };
                // Test-only read instrumentation bounds a regression deterministically, rather than waiting for exponential work.
                const sources = [...previousIds];
                Object.defineProperty(replacement.provenance, 'source_turn_ids', {
                    enumerable: true,
                    get() {
                        sourceReads += 1;
                        if (sourceReads > 4096) throw new Error('Repeated DAG traversal exceeded bounded test reads');
                        return sources;
                    },
                });
                document.compactions[id] = retained;
                currentIds.push(turnId);
            }
            previousIds = currentIds;
        }
        for (const turnId of previousIds)
            document.context.entries.push({
                id: `context:${turnId}`,
                type: 'replacement_turn',
                turn_id: turnId,
                compaction_id: turnId.replace('turn:', 'compaction:'),
            });
        // The genuine original append remains directly selected. This tests only unrelated traversal complexity,
        // not publication/authentication of the adversarial DAG or permission to consume its content.
        const result = await appendCanonicalPrompt(document, emptyRecords(), runtime(document), undefined, null);
        expect(result.document).toBe(document);
        expect(sourceReads).toBeLessThanOrEqual(48);
        expect(document.operation_receipts[INPUT_OPERATION_ID].accepted_context_entry_ids).toEqual([
            'context:materialized-input',
        ]);
    });

    it('rejects missing or foreign compaction receipts, provenance drift and incomplete/duplicate selection', async () => {
        const { proof, document } = await compactedMaterializedDocument();
        const missing = structuredClone(document);
        delete missing.operation_receipts['operation:compact-materialized'];
        await expect(prepareMaterialized(missing, proof)).rejects.toThrow();
        const foreign = structuredClone(document);
        foreign.compactions['compaction:materialized'].operation_id = 'operation:archive-materialized';
        await expect(prepareMaterialized(foreign, proof)).rejects.toThrow();
        const drift = structuredClone(document);
        const derived = drift.compactions['compaction:materialized'].replacement_turns[0];
        if (derived.provenance.type !== 'derived') throw new Error('Expected derived fixture');
        derived.provenance.source_hash = `sha256:${'0'.repeat(64)}`;
        await expect(prepareMaterialized(drift, proof)).rejects.toThrow();
        const incomplete = structuredClone(document);
        incomplete.context.entries[0].block_ids = ['block:materialized-reference:0'];
        await expect(prepareMaterialized(incomplete, proof)).rejects.toThrow();
        const duplicate = structuredClone(document);
        duplicate.context.entries.push({
            id: 'context:duplicate-input',
            type: 'source_turn',
            turn_id: 'turn:materialized-input',
        });
        await expect(prepareMaterialized(duplicate, proof)).rejects.toThrow();
        const laterMarker = structuredClone(document);
        laterMarker.operation_receipts[INPUT_OPERATION_ID].accepted_context_entry_ids = [
            document.context.entries[0].id,
        ];
        await expect(prepareMaterialized(laterMarker, proof)).rejects.toThrow();
    });

    it('validates a retained materialized input without appending or changing its operation receipt', async () => {
        const document = materializedDocument();
        const proof = runtime(document);
        const prepared = await prepareCanonicalContext({
            options: { model: 'test-model', conversation: document, conversation_runtime: proof },
            provider: 'test-provider',
            protocol: 'test.protocol',
            adapter_version: 'test.v1',
        });

        expect(prepared.document.revision).toBe(document.revision);
        expect(prepared.document.operation_receipts[INPUT_OPERATION_ID]).toEqual(
            document.operation_receipts[INPUT_OPERATION_ID],
        );
        expect(prepared.document.operation_receipts[proof.input_operation_id]).toBeUndefined();
    });

    it('reuses an exactly accepted current-head input without appending it again', async () => {
        const document = materializedDocument();
        const options = runtime(document);

        const first = await appendCanonicalPrompt(document, emptyRecords(), options, [], { prompt: [] });
        const retried = await appendCanonicalPrompt(document, emptyRecords(), options, [], { prompt: [] });

        expect(first.document).toBe(document);
        expect(retried.document).toBe(document);
        expect(first.document.revision).toBe(1);
        expect(first.document.turns).toHaveLength(1);
        expect(first.tool_definitions).toEqual([]);
    });

    it('rejects a proof for the wrong operation or revision', async () => {
        const document = materializedDocument();
        const wrongOperation = runtime(document);
        if (wrongOperation.materialized_input === undefined) throw new Error('Expected materialized proof');
        wrongOperation.materialized_input.operation_id = 'operation:unknown';
        await expect(appendCanonicalPrompt(document, emptyRecords(), wrongOperation, [], null)).rejects.toThrow(
            /accepted input-only operation receipt/,
        );

        const wrongRevision = runtime(document);
        if (wrongRevision.materialized_input === undefined) throw new Error('Expected materialized proof');
        wrongRevision.materialized_input.result_revision -= 1;
        await expect(appendCanonicalPrompt(document, emptyRecords(), wrongRevision, [], null)).rejects.toThrow(
            /accepted input-only operation receipt/,
        );
    });

    it('rejects new prompt records alongside an already materialized input', async () => {
        const document = materializedDocument();
        const records = emptyRecords();
        const retainedTurn = document.turns[0];
        if (retainedTurn === undefined) throw new Error('Expected a materialized input turn');
        records.turns.push(retainedTurn);

        await expect(appendCanonicalPrompt(document, records, runtime(document), [], null)).rejects.toThrow(
            /cannot include new prompt records/,
        );
    });

    it('requires the accepted context identity and complete turn selection', async () => {
        const document = materializedDocument();
        const missingAcceptedContext = structuredClone(document);
        const receipt = missingAcceptedContext.operation_receipts[INPUT_OPERATION_ID];
        if (receipt === undefined) throw new Error('Expected input receipt');
        receipt.accepted_context_entry_ids = [];
        await expect(
            appendCanonicalPrompt(missingAcceptedContext, emptyRecords(), runtime(missingAcceptedContext), [], null),
        ).rejects.toThrow(/not selected by an accepted context entry/);

        const partial = structuredClone(document);
        const [entry] = partial.context.entries;
        if (entry?.type !== 'source_turn') throw new Error('Expected source-turn context entry');
        entry.block_ids = ['block:materialized-a'];
        await expect(appendCanonicalPrompt(partial, emptyRecords(), runtime(partial), [], null)).rejects.toThrow(
            /only partially selected/,
        );
    });

    it('records and exactly retries an explicit tool-set change after the materialized input', async () => {
        const definitions = await canonicalToolDefinitions([
            { name: 'lookup', description: 'Lookup', input_schema: { type: 'object' } },
        ]);
        const original = materializedDocument();
        const proof = runtime(original);
        const tools = [{ name: 'lookup', description: 'Lookup', input_schema: { type: 'object' as const } }];

        const first = await appendCanonicalPrompt(original, emptyRecords(), proof, tools, null);
        expect(first.document.revision).toBe(original.revision + 1);
        expect(first.document.context.active_tool_definition_ids).toEqual(
            definitions.map((definition) => definition.id),
        );
        expect(first.document.operation_receipts[proof.input_operation_id]).toMatchObject({
            base_revision: original.revision,
            result_revision: first.document.revision,
            accepted_tool_definition_ids: definitions.map((definition) => definition.id),
            accepted_turn_ids: [],
        });

        const retried = await appendCanonicalPrompt(first.document, emptyRecords(), proof, tools, null);
        expect(retried.document).toEqual(first.document);
        expect(retried.document.revision).toBe(first.document.revision);

        const omittedCompatibilityTools = await appendCanonicalPrompt(
            first.document,
            emptyRecords(),
            proof,
            undefined,
            null,
        );
        expect(omittedCompatibilityTools.document).toEqual(first.document);
        expect(omittedCompatibilityTools.tool_definitions).toEqual(definitions);
    });
});
