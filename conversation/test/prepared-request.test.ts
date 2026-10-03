import { describe, expect, it } from 'vitest';
import {
    adoptConversationPreparedRequestRecord,
    appendConversationRecords,
    appendDecodedConversationResponse,
    appendDecodedConversationResponseWithProcessing,
    assertAcceptedResponseMatchesPreparedRequest,
    type ConversationPreparedRequest,
    createConversationDocument,
    type DecodedConversationResponse,
    deriveConversationId,
    INDEXED_TEXT_PREPARED_VALIDATOR_PROFILE,
    parseConversationPreparedRequest,
    parseConversationPreparedRequestRecord,
    type RequestSourceViewReference,
    setProcessingPolicy,
} from '../src/index.js';

const RECORDED_AT = '2026-09-30T00:00:00.000Z';

async function preparedRequest(): Promise<ConversationPreparedRequest> {
    const document = appendConversationRecords(
        createConversationDocument({ id: 'conversation-1', created_at: RECORDED_AT }),
        {
            turns: [
                {
                    id: 'user-turn',
                    kind: 'user',
                    authority: 'ordinary',
                    blocks: [{ id: 'user-text', type: 'text', text: 'hello', format: 'plain' }],
                    status: 'completed',
                    timestamps: { recorded_at: RECORDED_AT },
                    provenance: { type: 'received' },
                    model_visibility: 'include',
                },
            ],
            context_entries: [{ id: 'user-context', type: 'source_turn', turn_id: 'user-turn' }],
        },
        {
            expected_revision: 0,
            operation_id: 'input-operation',
            payload_fingerprint: 'sha256:input',
            recorded_at: RECORDED_AT,
        },
    ).document;
    const runtime = {
        conversation_id: document.id,
        request_id: 'request-1',
        attempt_id: 'attempt-1',
        input_operation_id: 'input-operation',
        response_operation_id: 'response-operation',
        recorded_at: RECORDED_AT,
        purpose: 'conversation',
    };
    return {
        document,
        record: {
            source: { conversation_id: document.id, revision: document.revision },
            runtime,
            request_receipt: {
                id: await deriveConversationId('request_receipt', runtime.request_id, runtime.attempt_id),
                request_id: runtime.request_id,
                attempt_id: runtime.attempt_id,
                source: { conversation_id: document.id, revision: document.revision },
                source_tail_turn_id: 'user-turn',
                context_fingerprint: 'sha256:context',
                tool_set_fingerprint: 'sha256:tools',
                request_fingerprint: 'sha256:request',
                target: {
                    provider: 'test',
                    protocol: 'test.generate',
                    model: 'test-model',
                    adapter_version: '1',
                },
                tool_definition_ids: [],
                asset_versions: [],
                item_mappings: [
                    { canonical_id: 'user-turn', native_id: 'contents/0', kind: 'turn' },
                    { canonical_id: 'user-text', native_id: 'contents/0/parts/0', kind: 'block' },
                ],
                recorded_at: RECORDED_AT,
            },
            generation_id: await deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
            response_turn_id: await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
        },
    };
}

function sourceView(prepared: ConversationPreparedRequest): RequestSourceViewReference {
    return {
        version: 1,
        completeness: 'selected_execution',
        source: prepared.record.source,
        context_revision: prepared.document.context.revision,
        manifest_storage_key: 'opaque-manifest-key',
        manifest_content_hash: `sha256:${'a'.repeat(64)}`,
        manifest_size_bytes: 1024,
        context_fingerprint: prepared.record.request_receipt.context_fingerprint,
        request_fingerprint: prepared.record.request_receipt.request_fingerprint,
    };
}

describe('prepared canonical request evidence', () => {
    it('parses exact pre-provider request evidence without mutating the caller document', async () => {
        const prepared = await preparedRequest();
        const before = structuredClone(prepared);

        await expect(parseConversationPreparedRequest(prepared)).resolves.toEqual(prepared);
        expect(prepared).toEqual(before);
    });

    it('keeps indexed root evidence typed and separate from full-document preparation and adoption', async () => {
        const prepared = await preparedRequest();
        const indexed = structuredClone(prepared.record);
        indexed.indexed_source = {
            version: 1,
            validator_profile: INDEXED_TEXT_PREPARED_VALIDATOR_PROFILE,
            root: { content_hash: `sha256:${'a'.repeat(64)}`, size_bytes: 1024 },
            context_revision: prepared.document.context.revision,
        };
        expect(parseConversationPreparedRequestRecord(indexed)).toEqual(indexed);
        await expect(parseConversationPreparedRequest({ ...prepared, record: indexed })).rejects.toThrow(
            'cannot represent a full materialized document',
        );
        await expect(adoptConversationPreparedRequestRecord(prepared, indexed)).rejects.toThrow(
            'changed finalized prepared-request evidence',
        );
    });

    it.each([
        [
            'source revision',
            (value: ConversationPreparedRequest) => {
                value.record.source.revision = 0;
            },
        ],
        [
            'request identity',
            (value: ConversationPreparedRequest) => {
                value.record.runtime.request_id = 'changed';
            },
        ],
        [
            'selected mapping',
            (value: ConversationPreparedRequest) => {
                value.record.request_receipt.item_mappings[0].canonical_id = 'missing-turn';
            },
        ],
    ])('rejects changed %s before provider transport', async (_label, mutate) => {
        const prepared = await preparedRequest();
        mutate(prepared);
        await expect(parseConversationPreparedRequest(prepared)).rejects.toThrow();
    });

    it('adopts only a bounded immutable host source-view reference while preserving the finalized request', async () => {
        const original = await preparedRequest();
        const retained = structuredClone(original.record);
        retained.request_receipt.source_view = sourceView(original);

        const adopted = await adoptConversationPreparedRequestRecord(original, JSON.parse(JSON.stringify(retained)));
        expect(adopted.record).toEqual(retained);
        expect(original.record.request_receipt.source_view).toBeUndefined();
        await expect(parseConversationPreparedRequest(adopted)).resolves.toEqual(adopted);

        const conflicting = structuredClone(retained);
        if (conflicting.request_receipt.source_view === undefined) throw new Error('Missing source view');
        conflicting.request_receipt.source_view.manifest_content_hash = `sha256:${'b'.repeat(64)}`;
        await expect(adoptConversationPreparedRequestRecord(adopted, conflicting)).rejects.toThrow(
            'changed finalized prepared-request evidence',
        );
    });

    it('owns the returned locator before awaiting validation of the original', async () => {
        const original = await preparedRequest();
        const returned = structuredClone(original.record);
        returned.request_receipt.source_view = sourceView(original);
        const expected = structuredClone(returned);

        const adoption = adoptConversationPreparedRequestRecord(original, returned);
        if (returned.request_receipt.source_view === undefined) throw new Error('Missing source view');
        returned.request_receipt.source_view.manifest_storage_key = 'mutated-after-call';

        await expect(adoption).resolves.toMatchObject({ record: expected });
    });

    it('rejects a host return that changes original evidence or binds a different source view', async () => {
        const original = await preparedRequest();
        const changed = structuredClone(original.record);
        changed.request_receipt.target.model = 'other-model';
        await expect(adoptConversationPreparedRequestRecord(original, changed)).rejects.toThrow(
            'changed finalized prepared-request evidence',
        );

        const wrongSource = structuredClone(original.record);
        wrongSource.request_receipt.source_view = {
            ...sourceView(original),
            context_revision: original.document.context.revision + 1,
        };
        await expect(adoptConversationPreparedRequestRecord(original, wrongSource)).rejects.toThrow(
            'source view does not match',
        );
        expect(original.record.request_receipt.source_view).toBeUndefined();
    });

    it('binds an accepted response to the exact prepared receipt and response identities', async () => {
        const prepared = await preparedRequest();
        const generation = {
            id: prepared.record.generation_id,
            record_source: 'executed' as const,
            request_id: prepared.record.runtime.request_id,
            attempt_id: prepared.record.runtime.attempt_id,
            purpose: prepared.record.runtime.purpose,
            requested_model: 'test-model',
            provider: 'test',
            protocol: 'test.generate',
            adapter_version: '1',
            status: 'completed' as const,
            timestamps: { recorded_at: RECORDED_AT, completed_at: RECORDED_AT },
            source: prepared.record.source,
            request_receipt: prepared.record.request_receipt,
        };
        const turn = {
            id: prepared.record.response_turn_id,
            kind: 'agent' as const,
            authority: 'ordinary' as const,
            blocks: [{ id: 'answer', type: 'text' as const, text: 'done', format: 'plain' as const }],
            status: 'completed' as const,
            timestamps: { recorded_at: RECORDED_AT },
            generation_id: generation.id,
            provenance: { type: 'generated' as const },
            model_visibility: 'include' as const,
        };
        const accepted = appendConversationRecords(
            prepared.document,
            { turns: [turn], generations: [generation] },
            {
                expected_revision: prepared.document.revision,
                operation_id: prepared.record.runtime.response_operation_id,
                payload_fingerprint: 'sha256:response',
                recorded_at: RECORDED_AT,
            },
        ).document;

        await expect(assertAcceptedResponseMatchesPreparedRequest(accepted, prepared)).resolves.toEqual({
            generation,
            turn,
        });
        const changed = structuredClone(accepted);
        const changedGeneration = changed.generations[generation.id];
        if (changedGeneration?.record_source !== 'executed') throw new Error('Expected executed generation fixture');
        changedGeneration.request_receipt.request_fingerprint = 'sha256:changed';
        await expect(assertAcceptedResponseMatchesPreparedRequest(changed, prepared)).rejects.toThrow(
            'durably prepared request',
        );
    });

    it('stages a decoded response and processing jobs together while retaining exact retry and disabled parity', async () => {
        const source = await preparedRequest();
        const options = { operation_id: source.record.runtime.response_operation_id, recorded_at: RECORDED_AT };
        const decoded = (prepared: ConversationPreparedRequest): DecodedConversationResponse => ({
            payload_fingerprint: 'sha256:response',
            diagnostics: [],
            generation: {
                id: prepared.record.generation_id,
                record_source: 'executed',
                request_id: prepared.record.runtime.request_id,
                attempt_id: prepared.record.runtime.attempt_id,
                purpose: prepared.record.runtime.purpose,
                requested_model: prepared.record.request_receipt.target.model,
                provider: prepared.record.request_receipt.target.provider,
                protocol: prepared.record.request_receipt.target.protocol,
                adapter_version: prepared.record.request_receipt.target.adapter_version,
                status: 'completed',
                timestamps: { recorded_at: RECORDED_AT, completed_at: RECORDED_AT },
                source: prepared.record.source,
                request_receipt: prepared.record.request_receipt,
            },
            turns: [
                {
                    id: prepared.record.response_turn_id,
                    kind: 'agent',
                    authority: 'ordinary',
                    blocks: [{ id: 'answer', type: 'text', text: 'done', format: 'plain' }],
                    status: 'completed',
                    timestamps: { recorded_at: RECORDED_AT },
                    generation_id: prepared.record.generation_id,
                    provenance: { type: 'generated' },
                    model_visibility: 'include',
                },
            ],
        });

        const ordinary = appendDecodedConversationResponse(
            {
                ...source,
                receipt: source.record.request_receipt,
                generation_id: source.record.generation_id,
                response_turn_id: source.record.response_turn_id,
                diagnostics: [],
                payload: {},
            },
            decoded(source),
            options,
        );
        const ordinaryAsync = await appendDecodedConversationResponseWithProcessing(
            {
                ...source,
                receipt: source.record.request_receipt,
                generation_id: source.record.generation_id,
                response_turn_id: source.record.response_turn_id,
                diagnostics: [],
                payload: {},
            },
            decoded(source),
            options,
        );
        expect(ordinaryAsync.document).toEqual(ordinary.document);
        expect(ordinaryAsync.acceptance.processing.status).toBe('ready');

        const configured = await setProcessingPolicy(source.document, {
            operation_id: 'policy:response',
            expected_revision: source.document.revision,
            recorded_at: RECORDED_AT,
            enabled: true,
            processors: [
                {
                    id: 'noop',
                    version: 'v1',
                    scope: 'on_append',
                    config: {},
                    required: true,
                    failure_behavior: 'block',
                },
            ],
        });
        const enabled = structuredClone(source);
        enabled.document = configured.document;
        enabled.record.source.revision = configured.document.revision;
        enabled.record.request_receipt.source.revision = configured.document.revision;
        const prepared = {
            ...enabled,
            receipt: enabled.record.request_receipt,
            generation_id: enabled.record.generation_id,
            response_turn_id: enabled.record.response_turn_id,
            diagnostics: [],
            payload: {},
        };
        const response = decoded(enabled);
        expect(() => appendDecodedConversationResponse(prepared, response, options)).toThrow(
            'requires appendConversationRecordsWithProcessing',
        );
        const accepted = await appendDecodedConversationResponseWithProcessing(prepared, response, options);
        expect(accepted.applied).toBe(true);
        expect(accepted.accepted_generation_ids).toEqual([enabled.record.generation_id]);
        expect(accepted.accepted_turn_ids).toEqual([enabled.record.response_turn_id]);
        expect(accepted.acceptance.processing.status).toBe('pending');
        const jobs = Object.values(accepted.document.processing.jobs ?? {});
        expect(jobs).toHaveLength(1);
        expect(jobs[0].source_operation_id).toBe(options.operation_id);
        const retry = await appendDecodedConversationResponseWithProcessing(
            { ...prepared, document: JSON.parse(JSON.stringify(accepted.document)) },
            response,
            options,
        );
        expect(retry.applied).toBe(false);
        expect(retry.document.processing.jobs).toEqual(accepted.document.processing.jobs);
        expect(retry.document.operation_receipts[options.operation_id]).toEqual(
            accepted.document.operation_receipts[options.operation_id],
        );
    });
});
