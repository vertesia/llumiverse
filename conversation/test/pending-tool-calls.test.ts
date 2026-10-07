import { describe, expect, it } from 'vitest';
import {
    appendConversationRecords,
    createConversationDocument,
    derivePendingApplicationToolCalls,
    externalizeToolCallArguments,
    parseConversationDocument,
    prepareToolArgumentExternalization,
    resolveToolExecutionRequest,
} from '../src/index.js';
import type { Asset, ConversationDocument, ConversationTurn, Generation } from '../src/types.js';

const recordedAt = '2026-09-30T00:00:00.000Z';

function acceptedDocument(): ConversationDocument {
    const document = createConversationDocument({ id: 'conversation-1', created_at: recordedAt });
    const acceptedTurn: ConversationTurn = {
        id: 'turn-latest',
        kind: 'agent',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: recordedAt },
        model_visibility: 'include',
        provenance: { type: 'generated' },
        generation_id: 'generation-latest',
        blocks: [
            {
                id: 'block-latest',
                type: 'tool_call',
                call_id: 'call-latest',
                tool_name: 'latest_write',
                executor: 'application',
                arguments: { type: 'json', value: { text: 'exact' } },
            },
        ],
    };
    const generation = {
        id: 'generation-latest',
        record_source: 'executed',
        request_id: 'request-latest',
        attempt_id: 'attempt-latest',
        purpose: 'interaction',
        requested_model: 'model-1',
        provider: 'provider-1',
        protocol: 'protocol-1',
        adapter_version: 'adapter-1',
        status: 'completed',
        timestamps: { recorded_at: recordedAt },
        source: { conversation_id: document.id, revision: 0 },
        request_receipt: {
            id: 'request-receipt-latest',
            request_id: 'request-latest',
            attempt_id: 'attempt-latest',
            source: { conversation_id: document.id, revision: 0 },
            context_fingerprint: 'context-fingerprint',
            tool_set_fingerprint: 'tool-set-fingerprint',
            request_fingerprint: 'request-fingerprint',
            target: { provider: 'provider-1', protocol: 'protocol-1', model: 'model-1', adapter_version: 'adapter-1' },
            tool_definition_ids: [],
            asset_versions: [],
            item_mappings: [],
            recorded_at: recordedAt,
        },
    } satisfies Generation;
    document.revision = 3;
    document.turns.push(
        {
            id: 'turn-imported-answered',
            kind: 'agent',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: recordedAt },
            model_visibility: 'include',
            provenance: { type: 'imported', source: 'legacy' },
            blocks: [
                {
                    id: 'old-call',
                    type: 'tool_call',
                    call_id: 'old-answered',
                    tool_name: 'old',
                    executor: 'application',
                    arguments: { type: 'json', value: {} },
                },
            ],
        },
        {
            id: 'turn-old-malformed',
            kind: 'agent',
            authority: 'ordinary',
            status: 'interrupted',
            timestamps: { recorded_at: recordedAt },
            model_visibility: 'exclude',
            provenance: { type: 'imported', source: 'legacy' },
            blocks: [
                {
                    id: 'old-invalid',
                    type: 'tool_call',
                    call_id: 'old-invalid',
                    tool_name: 'old',
                    executor: 'application',
                    arguments: { type: 'invalid', raw: '{' },
                },
            ],
        },
        acceptedTurn,
        {
            id: 'turn-old-result',
            kind: 'tool',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: recordedAt },
            model_visibility: 'include',
            provenance: { type: 'imported', source: 'legacy' },
            blocks: [
                {
                    id: 'old-result',
                    type: 'tool_result',
                    call_id: 'old-answered',
                    content: [{ id: 'old-result-text', type: 'text', text: 'done', format: 'plain' }],
                    status: 'success',
                },
            ],
        },
    );
    document.generations[generation.id] = generation;
    document.operation_receipts['response-latest'] = {
        id: 'response-latest',
        conversation_id: document.id,
        payload_fingerprint: 'payload-fingerprint',
        base_revision: 0,
        result_revision: 1,
        recorded_at: recordedAt,
        accepted_turn_ids: [acceptedTurn.id],
        accepted_generation_ids: [generation.id],
    };
    return document;
}

describe('derivePendingApplicationToolCalls', () => {
    it('selects only the exact accepted response across later externalization revisions', async () => {
        const pending = await derivePendingApplicationToolCalls(acceptedDocument(), [
            { id: 'call-latest', tool_name: 'latest_write' },
        ]);

        expect(pending).toHaveLength(1);
        expect(pending[0]).toMatchObject({
            source: { conversation: { conversation_id: 'conversation-1', revision: 3 }, call_id: 'call-latest' },
            call: { call_id: 'call-latest', tool_name: 'latest_write', executor: 'application' },
        });
    });

    it('ignores provider-owned calls when matching the accepted application call order', async () => {
        const document = acceptedDocument();
        const accepted = document.turns.find((turn) => turn.id === 'turn-latest');
        if (accepted?.kind !== 'agent') throw new Error('missing accepted turn');
        accepted.blocks.unshift({
            id: 'provider-call',
            type: 'tool_call',
            call_id: 'provider-call',
            tool_name: 'provider_search',
            executor: 'provider',
            arguments: { type: 'json', value: { query: 'native' } },
        });

        await expect(
            derivePendingApplicationToolCalls(document, [{ id: 'call-latest', tool_name: 'latest_write' }]),
        ).resolves.toHaveLength(1);
    });

    it('rejects a stale mirror instead of re-authorizing an older unresolved response', async () => {
        const document = acceptedDocument();
        const latestReceipt = document.operation_receipts['response-latest'];
        const latestGeneration = document.generations['generation-latest'];
        const latestTurn = document.turns.find((turn) => turn.id === 'turn-latest');
        if (
            !latestReceipt ||
            latestGeneration?.record_source !== 'executed' ||
            !latestTurn ||
            latestTurn.kind !== 'agent' ||
            latestTurn.provenance.type !== 'generated'
        )
            throw new Error('missing latest response');
        latestReceipt.base_revision = 1;
        latestReceipt.result_revision = 2;
        latestGeneration.source.revision = 1;
        latestGeneration.request_receipt.source.revision = 1;
        const oldTurn: ConversationTurn = {
            ...latestTurn,
            id: 'turn-old-executed',
            generation_id: 'generation-old-executed',
            blocks: [
                {
                    id: 'block-old-executed',
                    type: 'tool_call' as const,
                    call_id: 'call-old-executed',
                    tool_name: 'old_write',
                    executor: 'application' as const,
                    arguments: { type: 'json' as const, value: { stale: true } },
                },
            ],
        };
        document.turns.unshift(oldTurn);
        document.generations['generation-old-executed'] = {
            ...latestGeneration,
            id: 'generation-old-executed',
            request_id: 'request-old',
            attempt_id: 'attempt-old',
            source: { conversation_id: document.id, revision: 0 },
            request_receipt: {
                ...latestGeneration.request_receipt,
                id: 'request-receipt-old',
                request_id: 'request-old',
                attempt_id: 'attempt-old',
                source: { conversation_id: document.id, revision: 0 },
            },
        };
        document.operation_receipts['response-old'] = {
            id: 'response-old',
            conversation_id: document.id,
            payload_fingerprint: 'sha256:old',
            base_revision: 0,
            result_revision: 1,
            recorded_at: recordedAt,
            accepted_turn_ids: [oldTurn.id],
            accepted_generation_ids: ['generation-old-executed'],
        };

        await expect(
            derivePendingApplicationToolCalls(document, [{ id: 'call-old-executed', tool_name: 'old_write' }]),
        ).rejects.toThrow('does not match the latest canonical response');
    });

    it.each([
        ['interrupted', 'completed'],
        ['completed', 'cancelled'],
    ] as const)('rejects a matching non-completed accepted response (%s/%s)', async (turnStatus, generationStatus) => {
        const document = acceptedDocument();
        const turn = document.turns.find((candidate) => candidate.id === 'turn-latest');
        const generation = document.generations['generation-latest'];
        if (!turn || !generation) throw new Error('missing accepted response');
        turn.status = turnStatus;
        generation.status = generationStatus;

        await expect(
            derivePendingApplicationToolCalls(document, [{ id: 'call-latest', tool_name: 'latest_write' }]),
        ).rejects.toThrow('inconsistent generation provenance');
    });

    it('ignores a real imported generation receipt preceding the executed response', async () => {
        const initial = createConversationDocument({ id: 'conversation-1', created_at: recordedAt });
        const imported = appendConversationRecords(
            initial,
            {
                turns: [
                    {
                        id: 'turn-imported',
                        kind: 'agent',
                        authority: 'ordinary',
                        status: 'completed',
                        timestamps: { recorded_at: recordedAt },
                        model_visibility: 'include',
                        provenance: { type: 'generated' },
                        generation_id: 'generation-imported',
                        blocks: [
                            {
                                id: 'block-imported',
                                type: 'tool_call',
                                call_id: 'call-imported',
                                tool_name: 'imported_tool',
                                executor: 'application',
                                arguments: { type: 'json', value: {} },
                            },
                        ],
                    },
                ],
                generations: [
                    {
                        id: 'generation-imported',
                        record_source: 'imported',
                        status: 'completed',
                        timestamps: { recorded_at: recordedAt },
                        source: { conversation_id: initial.id, revision: 0 },
                        missing_metadata: ['requested_model', 'provider'],
                    },
                ],
            },
            {
                expected_revision: 0,
                operation_id: 'operation-imported',
                payload_fingerprint: 'sha256:imported',
                recorded_at: recordedAt,
            },
        );
        const candidate = acceptedDocument();
        const turn = candidate.turns.find((entry) => entry.id === 'turn-latest');
        const generation = candidate.generations['generation-latest'];
        if (!turn || !generation || generation.record_source !== 'executed')
            throw new Error('missing executed response');
        generation.source.revision = imported.document.revision;
        generation.request_receipt.source.revision = imported.document.revision;
        const executed = appendConversationRecords(
            imported.document,
            { turns: [turn], generations: [generation] },
            {
                expected_revision: imported.document.revision,
                operation_id: 'operation-executed',
                payload_fingerprint: 'sha256:executed',
                recorded_at: recordedAt,
            },
        );

        await expect(derivePendingApplicationToolCalls(executed.document)).resolves.toMatchObject([
            { call: { call_id: 'call-latest', tool_name: 'latest_write' } },
        ]);
    });

    it('round trips externalized large arguments through the exact durable source revision', async () => {
        const exactText = 'large exact content'.repeat(5_000);
        const document = acceptedDocument();
        const accepted = document.turns.find((turn) => turn.id === 'turn-latest');
        if (!accepted) throw new Error('missing accepted turn');
        document.turns = [accepted];
        document.revision = 1;
        const call = accepted?.blocks.find((block) => block.type === 'tool_call' && block.call_id === 'call-latest');
        if (call?.type !== 'tool_call') throw new Error('missing accepted call');
        call.arguments = { type: 'json', value: { text: exactText } };
        const prepared = await prepareToolArgumentExternalization(document, 'call-latest', ['text']);
        const asset: Asset = {
            id: 'asset-large-input',
            kind: 'text',
            mime_type: 'text/plain',
            storage: {
                type: 'external',
                resolver: 'test.artifact',
                locator: { storage_id: 'run-1', artifact_path: 'tool-inputs/call-latest.txt' },
            },
            provenance: { type: 'imported', source: 'test' },
            byte_length: prepared.byte_length,
            content_hash: prepared.content_hash,
            created_at: recordedAt,
        };
        const externalized = await externalizeToolCallArguments(document, {
            operation_id: 'externalize-large-input',
            expected_revision: document.revision,
            recorded_at: recordedAt,
            call_id: 'call-latest',
            input_path: ['text'],
            model_value: { text: '[externalized]' },
            exact_arguments_hash: prepared.exact_arguments_hash,
            asset,
        });
        const persisted = parseConversationDocument(JSON.parse(JSON.stringify(externalized.document)));
        const [pending] = await derivePendingApplicationToolCalls(persisted, [
            { id: 'call-latest', tool_name: 'latest_write' },
        ]);
        if (!pending) throw new Error('missing pending call');

        const resolved = await resolveToolExecutionRequest(persisted, pending.source, async function* () {
            yield new TextEncoder().encode(exactText);
        });

        expect(pending.source.conversation.revision).toBe(persisted.revision);
        expect(resolved.arguments).toEqual({ text: exactText });
        expect(new TextEncoder().encode(exactText).byteLength).toBeGreaterThan(80 * 1024);
    });

    it('excludes a native answered call even without an execution receipt', async () => {
        const document = acceptedDocument();
        document.turns.push({
            id: 'turn-latest-result',
            kind: 'tool',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: recordedAt },
            model_visibility: 'include',
            provenance: { type: 'imported', source: 'host' },
            blocks: [
                {
                    id: 'latest-result',
                    type: 'tool_result',
                    call_id: 'call-latest',
                    content: [{ id: 'latest-result-text', type: 'text', text: 'done', format: 'plain' }],
                    status: 'success',
                },
            ],
        });

        await expect(
            derivePendingApplicationToolCalls(document, [{ id: 'call-latest', tool_name: 'latest_write' }]),
        ).resolves.toEqual([]);
    });
});
