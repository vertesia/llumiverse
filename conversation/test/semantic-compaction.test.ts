import { describe, expect, it } from 'vitest';
import { type CompactionRecord, type UserTurn, validateConversationDocument } from '../src/index.js';
import { emptyDocument, RECORDED_AT, textBlock, userTurn } from './fixtures.js';

function replacementTurn(id: string, derivationId: string, sourceBlockId = 'source-a'): UserTurn {
    const turn = userTurn(id, `${id}-text`);
    turn.provenance = {
        type: 'derived',
        derivation_id: derivationId,
        source_turn_ids: ['source'],
        source_block_ids: [sourceBlockId],
        source_hash: `sha256:${sourceBlockId}`,
    };
    return turn;
}

function compaction(id: string, replacementId: string, sourceBlockId = 'source-a'): CompactionRecord {
    return {
        id,
        operation_id: `${id}-operation`,
        strategy: { id: 'summarize', version: '1', configuration_fingerprint: 'sha256:config' },
        source: {
            turn_ids: ['source'],
            block_ids: ['source-a'],
            source_fingerprint: 'sha256:source',
        },
        replacement_turns: [replacementTurn(replacementId, id, sourceBlockId)],
        fidelity: 'semantic',
        retained_asset_ids: [],
        generation_ids: [],
        created_at: RECORDED_AT,
    };
}

function partialCompactionDocument() {
    const document = emptyDocument();
    const source = userTurn('source', 'source-a');
    source.blocks.push(textBlock('source-b'));
    document.turns.push(source);
    document.compactions.compaction = compaction('compaction', 'replacement');
    document.context.entries = [
        { id: 'replacement-entry', type: 'replacement_turn', compaction_id: 'compaction', turn_id: 'replacement' },
        { id: 'direct-entry', type: 'source_turn', turn_id: 'source', block_ids: ['source-b'] },
    ];
    return document;
}

describe('partial compaction semantics', () => {
    it('allows a replaced block and a disjoint direct block from the same turn', () => {
        expect(validateConversationDocument(partialCompactionDocument())).toMatchObject({ success: true });
    });

    it('rejects direct selection of a block covered by an active replacement', () => {
        const document = partialCompactionDocument();
        document.context.entries[1] = {
            id: 'direct-entry',
            type: 'source_turn',
            turn_id: 'source',
            block_ids: ['source-a'],
        };
        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'CONTEXT_DIRECT_REPLACEMENT_OVERLAP')).toBe(
            true,
        );
    });

    it('rejects duplicate direct selections', () => {
        const document = partialCompactionDocument();
        document.context.entries.push({
            id: 'duplicate-direct',
            type: 'source_turn',
            turn_id: 'source',
            block_ids: ['source-b'],
        });
        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'CONTEXT_SELECTION_OVERLAP')).toBe(true);
    });

    it('rejects derived source blocks outside the compaction selection', () => {
        const document = partialCompactionDocument();
        document.compactions.compaction.replacement_turns[0] = replacementTurn('replacement', 'compaction', 'source-b');
        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'DERIVED_PROVENANCE_MISMATCH')).toBe(true);
    });

    it('rejects two active replacements whose source selections overlap', () => {
        const document = partialCompactionDocument();
        document.compactions.second = compaction('second', 'second-replacement');
        document.compactions.second.supersedes_compaction_id = 'compaction';
        document.context.entries.push({
            id: 'second-entry',
            type: 'replacement_turn',
            compaction_id: 'second',
            turn_id: 'second-replacement',
        });
        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'CONTEXT_SELECTION_OVERLAP')).toBe(true);
    });

    it('rejects supersession and causal parent cycles', () => {
        const document = partialCompactionDocument();
        document.compactions.second = compaction('second', 'second-replacement');
        document.compactions.compaction.supersedes_compaction_id = 'second';
        document.compactions.second.supersedes_compaction_id = 'compaction';
        const first = userTurn('first');
        const second = userTurn('second-source');
        first.parent_turn_id = second.id;
        second.parent_turn_id = first.id;
        document.turns.push(first, second);

        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'SUPERSEDES_CYCLE')).toBe(true);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'PARENT_TURN_CYCLE')).toBe(true);
    });

    it('rejects authority promotion by a replacement', () => {
        const document = partialCompactionDocument();
        document.compactions.compaction.replacement_turns[0].authority = 'system';
        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'DERIVED_AUTHORITY_PROMOTED')).toBe(true);
    });

    it('rejects cycles between derived replacement provenances', () => {
        const document = emptyDocument();
        const first = compaction('first', 'first-replacement');
        const second = compaction('second', 'second-replacement');
        first.source = {
            turn_ids: ['second-replacement'],
            source_fingerprint: 'sha256:second-replacement',
        };
        second.source = {
            turn_ids: ['first-replacement'],
            source_fingerprint: 'sha256:first-replacement',
        };
        first.replacement_turns[0].provenance = {
            type: 'derived',
            derivation_id: 'first',
            source_turn_ids: ['second-replacement'],
            source_hash: 'sha256:second-replacement',
        };
        second.replacement_turns[0].provenance = {
            type: 'derived',
            derivation_id: 'second',
            source_turn_ids: ['first-replacement'],
            source_hash: 'sha256:first-replacement',
        };
        document.compactions = { first, second };

        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'DERIVED_PROVENANCE_CYCLE')).toBe(true);
    });
});

describe('historical request receipts', () => {
    it('allows retained receipt bindings whose transcript records were deleted', () => {
        const document = emptyDocument();
        document.generations.generation = {
            id: 'generation',
            record_source: 'imported',
            status: 'completed',
            timestamps: { recorded_at: RECORDED_AT },
            source: { conversation_id: document.id, revision: 0 },
            request_receipt: {
                id: 'request-receipt',
                request_id: 'request',
                attempt_id: 'attempt',
                source: { conversation_id: document.id, revision: 0 },
                source_tail_turn_id: 'deleted-turn',
                context_fingerprint: 'sha256:context',
                tool_set_fingerprint: 'sha256:tools',
                request_fingerprint: 'sha256:request',
                target: {
                    provider: 'provider',
                    protocol: 'protocol',
                    model: 'model',
                    adapter_version: 'adapter-v1',
                },
                tool_definition_ids: ['deleted-tool'],
                asset_versions: [{ asset_id: 'deleted-asset', content_hash: 'sha256:asset' }],
                item_mappings: [{ canonical_id: 'deleted-block', native_id: 'native', kind: 'block' }],
                recorded_at: RECORDED_AT,
            },
        };

        expect(validateConversationDocument(document)).toMatchObject({ success: true });
    });
});
