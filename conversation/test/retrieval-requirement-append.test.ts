import { describe, expect, it } from 'vitest';
import { appendConversationRecords, createConversationDocument, parseConversationDocument } from '../src/index.js';

const at = '2026-10-03T00:00:00.000Z';
const operation = {
    operation_id: 'append:archive',
    expected_revision: 0,
    payload_fingerprint: 'sha256:accepted-archive',
    recorded_at: at,
};
const hash = `sha256:${'a'.repeat(64)}`;
const retrieval = {
    capability: 'read_artifact',
    version: 1,
    tool_definition_id: 'definition:read',
    arguments: { path: 'canonical-tool-results/v1/tool/call/activity-archive.json', start_byte: 0, byte_count: 5000 },
};
const requirement = {
    id: 'requirement:archive',
    asset_id: 'asset:archive',
    retrieval,
    accepted_asset_operation_id: operation.operation_id,
};
const batch = {
    assets: [
        {
            id: 'asset:archive',
            kind: 'text' as const,
            mime_type: 'application/json',
            storage: {
                type: 'external' as const,
                resolver: 'vertesia.agent_artifact',
                locator: { storage_id: 'run:1', artifact_path: retrieval.arguments.path },
            },
            provenance: { type: 'received' as const },
            byte_length: 12,
            content_hash: hash,
            created_at: at,
        },
    ],
    tool_definitions: [
        {
            id: 'definition:read',
            name: 'read_artifact',
            version: 'content-version',
            input_schema: {
                type: 'object',
                properties: { path: { type: 'string' } },
                required: ['path'],
                additionalProperties: false,
            },
        },
    ],
    active_tool_definition_ids: ['definition:read'],
    turns: [
        {
            id: 'turn:archive',
            kind: 'user' as const,
            authority: 'ordinary' as const,
            status: 'completed' as const,
            model_visibility: 'include' as const,
            timestamps: { recorded_at: at },
            provenance: { type: 'received' as const },
            blocks: [
                {
                    id: 'block:archive',
                    type: 'external_reference' as const,
                    asset_id: 'asset:archive',
                    original_type: 'text' as const,
                    description: 'Exact accepted archive',
                    content_hash: hash,
                    preview: 'Archive preview',
                    retrieval,
                },
            ],
        },
    ],
    context_entries: [{ id: 'entry:archive', type: 'source_turn' as const, turn_id: 'turn:archive' }],
    retrieval_requirements: [requirement],
};

describe('append retrieval requirement receipt', () => {
    it('replays the exact accepted requirement after a later active-context edit and rejects changed details', () => {
        const accepted = appendConversationRecords(
            createConversationDocument({ id: 'conversation:archive', created_at: at }),
            batch,
            operation,
        );
        expect(accepted.document.operation_receipts[operation.operation_id]?.accepted_retrieval_requirements).toEqual([
            requirement,
        ]);
        const later = structuredClone(accepted.document);
        later.context.retrieval_requirements = [];
        const validLater = parseConversationDocument(later);
        const replay = appendConversationRecords(validLater, batch, operation);
        expect(replay.applied).toBe(false);
        expect(replay.document).toEqual(validLater);
        expect(() =>
            appendConversationRecords(validLater, { ...batch, retrieval_requirements: [] }, operation),
        ).toThrow('does not identify its accepted retrieval requirements');
        expect(() =>
            appendConversationRecords(
                validLater,
                {
                    ...batch,
                    retrieval_requirements: [requirement, { ...requirement, id: 'requirement:added' }],
                },
                operation,
            ),
        ).toThrow('does not identify its accepted retrieval requirements');
        const missingWitness = structuredClone(validLater);
        delete missingWitness.operation_receipts[operation.operation_id]?.accepted_retrieval_requirements;
        expect(() => appendConversationRecords(parseConversationDocument(missingWitness), batch, operation)).toThrow(
            'Historical append receipt cannot prove its accepted retrieval requirements',
        );
        expect(() =>
            appendConversationRecords(
                validLater,
                {
                    ...batch,
                    retrieval_requirements: [
                        {
                            ...requirement,
                            retrieval: { ...retrieval, arguments: { ...retrieval.arguments, byte_count: 4000 } },
                        },
                    ],
                },
                operation,
            ),
        ).toThrow('changes accepted retrieval requirement');
    });
});
