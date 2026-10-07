import { describe, expect, it } from 'vitest';
import { canonicalJsonContentBytes, hashContentBytes } from '../src/content-integrity.js';
import { applyContextMutationWorkingSet } from '../src/context-change-transition.js';
import { materializedContextChangeWorkingSet, planContextChangeWorkingSet } from '../src/context-change-working-set.js';
import { fingerprintJson } from '../src/identity.js';
import { ContextChangeRequestSchema } from '../src/schemas/context-change.js';
import { OperationReceiptSchema } from '../src/schemas/execution.js';
import type { Asset, ContentBlock, ConversationDocument, ToolDefinition } from '../src/types.js';
import { emptyDocument, RECORDED_AT, textBlock, toolCallBlock, toolResultTurn, userTurn } from './fixtures.js';

async function exchangeFixture() {
    const document: ConversationDocument = emptyDocument('conversation:retrievable-exchange');
    const call = toolCallBlock('block:call', 'call:one');
    const callTurn = {
        ...userTurn('turn:call'),
        kind: 'agent' as const,
        provenance: { type: 'received' as const },
        blocks: [call],
    };
    const originalBytes = new TextEncoder().encode('Exact accepted tool archive');
    const originalIntegrity = await hashContentBytes(originalBytes);
    const originalAsset: Asset = {
        id: 'asset:original',
        kind: 'text',
        mime_type: 'text/plain',
        storage: { type: 'external', resolver: 'test.original', locator: { key: 'original' } },
        provenance: { type: 'received', source_turn_id: 'turn:result' },
        created_at: RECORDED_AT,
        ...originalIntegrity,
    };
    const originalRetrieval = {
        capability: 'read_original',
        version: 1 as const,
        tool_definition_id: 'definition:original',
        arguments: { path: 'original' },
    };
    const originalReference = {
        id: 'block:original-reference',
        type: 'external_reference' as const,
        asset_id: originalAsset.id,
        original_type: 'text' as const,
        content_hash: originalAsset.content_hash,
        description: 'Original archive',
        preview: 'Original archive preview',
        retrieval: originalRetrieval,
    };
    const resultTurn = {
        ...toolResultTurn('turn:result', call.call_id),
        blocks: [
            {
                id: 'block:result',
                type: 'tool_result' as const,
                call_id: call.call_id,
                status: 'success' as const,
                content: [originalReference],
            },
        ],
    };
    const originalRequirement = {
        id: 'requirement:original',
        asset_id: originalAsset.id,
        retrieval: originalRetrieval,
        accepted_asset_operation_id: 'operation:original',
    };
    const callIntegrity = await hashContentBytes(canonicalJsonContentBytes(call));
    const callAsset: Asset = {
        id: 'asset:call-copy',
        kind: 'text',
        mime_type: 'application/json',
        storage: { type: 'external', resolver: 'test.copy', locator: { key: 'call-copy' } },
        provenance: { type: 'received', source_turn_id: callTurn.id },
        created_at: RECORDED_AT,
        ...callIntegrity,
    };
    const resultAsset: Asset = {
        id: 'asset:result-copy',
        kind: 'text',
        mime_type: 'text/plain',
        storage: { type: 'external', resolver: 'test.copy', locator: { key: 'result-copy' } },
        provenance: {
            type: 'derived',
            source_asset_id: originalAsset.id,
            transform_id: 'conversation.archive_rehome',
            transform_version: '1',
        },
        created_at: RECORDED_AT,
        ...originalIntegrity,
    };
    const definitions: ToolDefinition[] = [
        { id: 'definition:original', name: 'read_original', version: '1', input_schema: true },
        { id: 'definition:copy', name: 'read_copy', version: '1', input_schema: true },
    ];
    const originalAcceptance = OperationReceiptSchema.parse({
        id: 'operation:original',
        conversation_id: document.id,
        payload_fingerprint: `sha256:${'a'.repeat(64)}`,
        base_revision: 0,
        result_revision: 1,
        recorded_at: RECORDED_AT,
        accepted_turn_ids: [callTurn.id, resultTurn.id],
        accepted_asset_ids: [originalAsset.id],
        accepted_retrieval_requirements: [originalRequirement],
    });
    const copyAcceptance = OperationReceiptSchema.parse({
        id: 'operation:copy',
        conversation_id: document.id,
        payload_fingerprint: `sha256:${'b'.repeat(64)}`,
        base_revision: 1,
        result_revision: 2,
        recorded_at: RECORDED_AT,
        accepted_turn_ids: [],
        accepted_asset_ids: [callAsset.id, resultAsset.id],
    });
    document.revision = 2;
    document.context.revision = 2;
    document.turns.push(callTurn, resultTurn);
    document.context.entries.push(
        { id: 'entry:call', type: 'source_turn', turn_id: callTurn.id },
        { id: 'entry:result', type: 'source_turn', turn_id: resultTurn.id },
    );
    document.context.active_tool_definition_ids = definitions.map((item) => item.id);
    document.context.retrieval_requirements = [originalRequirement];
    for (const asset of [originalAsset, callAsset, resultAsset]) document.assets[asset.id] = asset;
    for (const definition of definitions) document.tool_definitions[definition.id] = definition;
    document.operation_receipts[originalAcceptance.id] = originalAcceptance;
    document.operation_receipts[copyAcceptance.id] = copyAcceptance;
    const frame = materializedContextChangeWorkingSet(document);
    const selection = {
        expected_revision: document.revision,
        expected_context_revision: document.context.revision,
        entry_ids: ['entry:call', 'entry:result'],
        selected_entries: document.context.entries,
        selected_block_ids: { 'entry:call': [call.id], 'entry:result': [resultTurn.blocks[0].id] },
    };
    const plan = await planContextChangeWorkingSet(frame, selection);
    const copiedRetrieval = (asset: Asset) => ({
        capability: 'read_copy',
        version: 1 as const,
        tool_definition_id: 'definition:copy',
        arguments: { asset_id: asset.id },
    });
    const reference = (asset: Asset, id: string): ContentBlock => ({
        id,
        type: 'external_reference',
        original_type: 'text',
        asset_id: asset.id,
        description: 'Exact selected exchange archive',
        preview: 'Read the exact selected archive',
        content_hash: asset.content_hash,
        retrieval: copiedRetrieval(asset),
    });
    const replacement = {
        ...userTurn('turn:replacement'),
        kind: 'agent' as const,
        provenance: {
            type: 'derived' as const,
            derivation_id: 'compaction:exchange',
            source_turn_ids: plan.source_turn_ids,
            source_block_ids: [call.id, resultTurn.blocks[0].id],
            source_hash: plan.source_fingerprint,
        },
        blocks: [reference(callAsset, 'block:call-copy'), reference(resultAsset, 'block:result-copy')],
    };
    const request = ContextChangeRequestSchema.parse({
        ...selection,
        operation_id: 'operation:context-copy',
        expected_source_fingerprint: plan.source_fingerprint,
        recorded_at: RECORDED_AT,
        proposal: {
            kind: 'replace_with_compaction',
            compaction_id: 'compaction:exchange',
            strategy: {
                id: 'retrievable-exchange',
                version: '1',
                configuration_fingerprint: `sha256:${'c'.repeat(64)}`,
            },
            replacement_turns: [replacement],
            fidelity: 'retrievable',
            retained_asset_ids: [callAsset.id, resultAsset.id],
            generation_ids: [],
            accepted_asset_operation_id: copyAcceptance.id,
            placement: { mode: 'first_selected', causal_order: 'contiguous' },
        },
    });
    const evidence = {
        compactions: {},
        tool_definitions: document.tool_definitions,
        operation_receipts: document.operation_receipts,
        original_archive_bytes: new Map([[originalAsset.id, originalBytes]]),
    };
    return { frame, evidence, request, originalBytes, originalAsset, originalReference, originalRequirement };
}

describe('retrievable completed exchange transition', () => {
    it('binds original bytes, both selected blocks, copy assets and the accepted publication', async () => {
        const { frame, evidence, request } = await exchangeFixture();
        const result = await applyContextMutationWorkingSet(frame, evidence, request, await fingerprintJson(request));
        // A whole-entry selection has no partial source.block_ids. The exact two-block mapping
        // remains on the derived replacement's provenance and the accepted request.
        expect(result.compaction?.source.block_ids).toBeUndefined();
        expect(result.compaction?.source.turn_ids).toEqual(['turn:call', 'turn:result']);
        expect(result.compaction?.replacement_turns[0]?.provenance).toMatchObject({
            source_block_ids: ['block:call', 'block:result'],
        });
        expect(result.compaction?.retained_asset_ids).toEqual(['asset:call-copy', 'asset:result-copy']);
        expect(result.receipt.context_change?.removed_entry_ids).toEqual(['entry:call', 'entry:result']);
        expect(result.context.retrieval_requirements).toHaveLength(3);
        expect(frame.context.entries).toHaveLength(2);
    });

    it('rejects changed original bytes, source publication, copied asset, and omitted sibling content', async () => {
        const exact = await exchangeFixture();
        const changedBytes = structuredClone(exact);
        changedBytes.evidence.original_archive_bytes.set(exact.originalAsset.id, new TextEncoder().encode('changed'));
        await expect(
            applyContextMutationWorkingSet(changedBytes.frame, changedBytes.evidence, changedBytes.request, 'changed'),
        ).rejects.toThrow('source archive bytes differ');
        const changedReceipt = structuredClone(exact);
        changedReceipt.evidence.operation_receipts['operation:original'].accepted_retrieval_requirements = [];
        await expect(
            applyContextMutationWorkingSet(
                changedReceipt.frame,
                changedReceipt.evidence,
                changedReceipt.request,
                'changed',
            ),
        ).rejects.toThrow('exact original and publication');
        const changedAsset = structuredClone(exact);
        changedAsset.frame.assets['asset:result-copy'].provenance = { type: 'received', source_turn_id: 'turn:result' };
        await expect(
            applyContextMutationWorkingSet(changedAsset.frame, changedAsset.evidence, changedAsset.request, 'changed'),
        ).rejects.toThrow('exact original and publication');
        const sibling = structuredClone(exact);
        const result = sibling.frame.turns.get('turn:result');
        if (result?.active_blocks[0]?.type !== 'tool_result') throw new Error('Result fixture missing');
        result.active_blocks[0].content.push(textBlock('block:sibling', 'unarchived sibling'));
        const siblingPlan = await planContextChangeWorkingSet(sibling.frame, sibling.request);
        sibling.request.expected_source_fingerprint = siblingPlan.source_fingerprint;
        if (sibling.request.proposal.kind !== 'replace_with_compaction') throw new Error('Compaction fixture missing');
        const replacement = sibling.request.proposal.replacement_turns[0];
        if (replacement?.provenance.type !== 'derived') throw new Error('Replacement fixture missing');
        replacement.provenance.source_hash = siblingPlan.source_fingerprint;
        await expect(
            applyContextMutationWorkingSet(sibling.frame, sibling.evidence, sibling.request, 'changed'),
        ).rejects.toThrow('whole completed call and exact archived result');
    });

    it('requires the exact accepted destination receipt and selected active reader', async () => {
        const missingReceipt = await exchangeFixture();
        delete missingReceipt.evidence.operation_receipts['operation:copy'];
        await expect(
            applyContextMutationWorkingSet(
                missingReceipt.frame,
                missingReceipt.evidence,
                missingReceipt.request,
                'changed',
            ),
        ).rejects.toThrow('exact ordered accepted text assets');
        const inactiveReader = await exchangeFixture();
        inactiveReader.frame.context.active_tool_definition_ids = ['definition:original'];
        await expect(
            applyContextMutationWorkingSet(
                inactiveReader.frame,
                inactiveReader.evidence,
                inactiveReader.request,
                'changed',
            ),
        ).rejects.toThrow('active read tool per block');
    });

    it('rejects a split exchange and a protected dependent replay before archiving', async () => {
        const exact = await exchangeFixture();
        const split = structuredClone(exact);
        split.request.entry_ids = ['entry:result'];
        split.request.selected_entries = [split.frame.context.entries[1]];
        split.request.selected_block_ids = { 'entry:result': ['block:result'] };
        await expect(
            applyContextMutationWorkingSet(split.frame, split.evidence, split.request, 'changed'),
        ).rejects.toThrow('split application tool exchange');
        const protectedReplay = structuredClone(exact);
        protectedReplay.frame.context.protected_entry_ids.push('entry:call');
        await expect(
            applyContextMutationWorkingSet(
                protectedReplay.frame,
                protectedReplay.evidence,
                protectedReplay.request,
                'changed',
            ),
        ).rejects.toThrow('protected entry');
    });
});
