import { describe, expect, it } from 'vitest';
import { hashUtf8Content } from '../src/content-integrity.js';
import { resolveActiveTextExternalReference } from '../src/external-reference-retrieval.js';
import {
    appendConversationRecords,
    applyContextChange,
    type NativeReplayBlock,
    parseConversationDocument,
    planContextChange,
} from '../src/index.js';
import { emptyDocument, textBlock, toolCallBlock, toolResultTurn, userTurn } from './fixtures.js';

const changeTime = '2026-09-11T00:01:00.000Z';

function replayBlock(id: string, dependencies: Partial<NativeReplayBlock['dependencies']>): NativeReplayBlock {
    return {
        id,
        type: 'native_replay',
        adapter: 'adapter',
        protocol: 'protocol',
        compatibility_scope: { provider: 'provider', protocol: 'protocol', adapter_version: 'v1' },
        payload: { opaque: 'signed' },
        dependencies: { turn_ids: [], block_ids: [], call_ids: [], request_ids: [], ...dependencies },
    };
}

function documentWithEntries() {
    const document = emptyDocument();
    document.turns.push(userTurn('first'), userTurn('second'));
    document.context.entries = [
        { id: 'first-entry', type: 'source_turn', turn_id: 'first' },
        { id: 'second-entry', type: 'source_turn', turn_id: 'second' },
    ];
    return parseConversationDocument(document);
}

async function exclude(document = documentWithEntries(), entryIds = ['first-entry'], operationId = 'edit:first') {
    const plan = await planContextChange(document, {
        expected_revision: document.revision,
        expected_context_revision: document.context.revision,
        entry_ids: entryIds,
    });
    const request = {
        operation_id: operationId,
        expected_revision: document.revision,
        expected_context_revision: document.context.revision,
        expected_source_fingerprint: plan.source_fingerprint,
        recorded_at: changeTime,
        entry_ids: entryIds,
        proposal: { kind: 'exclude' as const },
    };
    return { plan, request, result: await applyContextChange(document, request) };
}

describe('pure context changes', () => {
    it('keeps disjoint archived text at its original range positions with exact per-block proof', async () => {
        const source = structuredClone(documentWithEntries());
        source.turns.push(userTurn('third'));
        source.context.entries.push({ id: 'third-entry', type: 'source_turn', turn_id: 'third' });
        const original = parseConversationDocument(source);
        const selected = [original.turns[0], original.turns[2]].map((turn) => turn.blocks[0]);
        if (selected.some((block) => block?.type !== 'text')) throw new Error('Fixture needs text source blocks');
        const textBlocks = selected.filter(
            (block): block is Extract<typeof block, { type: 'text' }> => block?.type === 'text',
        );
        const integrities = await Promise.all(textBlocks.map((block) => hashUtf8Content(block.text)));
        const assets = integrities.map((integrity, index) => ({
            id: `asset:range:${index}`,
            kind: 'text' as const,
            mime_type: 'text/plain',
            storage: { type: 'external' as const, resolver: 'test.blob', locator: { key: `range:${index}` } },
            provenance: { type: 'received' as const },
            content_hash: integrity.content_hash,
            byte_length: integrity.byte_length,
            created_at: changeTime,
        }));
        const tool = { id: 'definition:range-read', name: 'read_blob', version: '1', input_schema: true };
        const staged = appendConversationRecords(
            original,
            { assets, tool_definitions: [tool], active_tool_definition_ids: [tool.id] },
            {
                operation_id: 'archive:ranges',
                expected_revision: original.revision,
                payload_fingerprint: 'sha256:range-archive',
                recorded_at: changeTime,
            },
        ).document;
        const plan = await planContextChange(staged, {
            expected_revision: staged.revision,
            expected_context_revision: staged.context.revision,
            entry_ids: ['first-entry', 'third-entry'],
            selected_entries: [staged.context.entries[0], staged.context.entries[2]],
            selected_block_ids: {
                'first-entry': [textBlocks[0].id],
                'third-entry': [textBlocks[1].id],
            },
        });
        expect(plan.disjoint_ranges).toBe(2);
        const turns = assets.map((asset, index) => ({
            ...userTurn(`reference:range:${index}`, 'placeholder'),
            kind: 'agent' as const,
            blocks: [
                {
                    id: `reference:block:${index}`,
                    type: 'external_reference' as const,
                    asset_id: asset.id,
                    original_type: 'text' as const,
                    description: 'Exact original on demand',
                    preview: textBlocks[index].text,
                    content_hash: asset.content_hash,
                    retrieval: {
                        capability: tool.name,
                        version: 1,
                        arguments: { asset_id: asset.id },
                        tool_definition_id: tool.id,
                    },
                },
            ],
            provenance: {
                type: 'derived' as const,
                derivation_id: 'compaction:ranges',
                source_turn_ids: [index === 0 ? 'first' : 'third'],
                source_block_ids: [textBlocks[index].id],
                source_hash: plan.source_fingerprint,
            },
        }));
        if (!plan.selected_entries || !plan.selected_block_ids) throw new Error('Expected partial selection plan');
        const request = {
            operation_id: 'context:ranges',
            expected_revision: staged.revision,
            expected_context_revision: staged.context.revision,
            expected_source_fingerprint: plan.source_fingerprint,
            recorded_at: changeTime,
            entry_ids: plan.entry_ids,
            selected_entries: plan.selected_entries,
            selected_block_ids: plan.selected_block_ids,
            proposal: {
                kind: 'replace_with_compaction' as const,
                compaction_id: 'compaction:ranges',
                strategy: { id: 'externalize-text', version: '1', configuration_fingerprint: 'sha256:config' },
                replacement_turns: turns,
                fidelity: 'retrievable' as const,
                accepted_asset_operation_id: 'archive:ranges',
                retained_asset_ids: assets.map((asset) => asset.id),
                generation_ids: [],
                placement: { mode: 'per_selected_range' as const, causal_order: 'preserved_disjoint_ranges' as const },
            },
        };
        const applied = await applyContextChange(staged, request);
        expect(applied.document.context.entries.map((entry) => entry.turn_id)).toEqual([
            turns[0].id,
            'second',
            turns[1].id,
        ]);
        expect(applied.document.context.retrieval_requirements).toHaveLength(2);
        for (const [index, asset] of assets.entries()) {
            expect(
                resolveActiveTextExternalReference(applied.document, asset.id, `reference:block:${index}`).asset.id,
            ).toBe(asset.id);
        }
        expect((await applyContextChange(applied.document, request)).applied).toBe(false);
        await expect(
            applyContextChange(staged, {
                ...request,
                proposal: { ...request.proposal, replacement_turns: [turns[1], turns[0]] },
            }),
        ).rejects.toThrow('exact text asset');
        await expect(
            applyContextChange(staged, {
                ...request,
                proposal: { ...request.proposal, retained_asset_ids: [assets[1].id, assets[0].id] },
            }),
        ).rejects.toThrow();
    });

    it('accepts a retrievable replacement only after the exact original asset and read tool are durably accepted', async () => {
        const original = documentWithEntries();
        const sourceText = 'first-text';
        const secondBlock = original.turns[1].blocks[0];
        if (secondBlock?.type !== 'text') throw new Error('Second source fixture must contain text');
        secondBlock.text = sourceText;
        const integrity = await hashUtf8Content(sourceText);
        const path = 'archive/assets/first.txt';
        const asset = {
            id: 'asset:original',
            kind: 'text' as const,
            mime_type: 'text/plain',
            storage: {
                type: 'external' as const,
                resolver: 'vertesia.agent_artifact',
                locator: { storage_id: 'run:owner', artifact_path: path },
            },
            provenance: { type: 'received' as const, source_turn_id: 'first' },
            content_hash: integrity.content_hash,
            byte_length: integrity.byte_length,
            created_at: changeTime,
        };
        const readTool = { id: 'definition:read', name: 'read_artifact', version: '1', input_schema: true };
        const staged = appendConversationRecords(
            original,
            { assets: [asset], tool_definitions: [readTool], active_tool_definition_ids: [readTool.id] },
            {
                operation_id: 'append:archive',
                expected_revision: 0,
                payload_fingerprint: 'sha256:archive',
                recorded_at: changeTime,
            },
        ).document;
        const plan = await planContextChange(staged, {
            expected_revision: staged.revision,
            expected_context_revision: staged.context.revision,
            entry_ids: ['first-entry'],
        });
        const replacement = {
            ...userTurn('reference-turn', 'reference-placeholder'),
            kind: 'agent' as const,
            blocks: [
                {
                    id: 'block:external',
                    type: 'external_reference' as const,
                    asset_id: asset.id,
                    original_type: 'text' as const,
                    description: 'Archived original',
                    preview: 'first-text',
                    content_hash: integrity.content_hash,
                    retrieval: {
                        capability: 'read_artifact',
                        version: 1,
                        arguments: { asset_id: asset.id, path },
                        tool_definition_id: readTool.id,
                    },
                },
            ],
            provenance: {
                type: 'derived' as const,
                derivation_id: 'compaction:external',
                source_turn_ids: plan.source_turn_ids,
                ...(plan.source_block_ids.length ? { source_block_ids: plan.source_block_ids } : {}),
                source_hash: plan.source_fingerprint,
            },
        };
        const request = {
            operation_id: 'context:externalize',
            expected_revision: staged.revision,
            expected_context_revision: staged.context.revision,
            expected_source_fingerprint: plan.source_fingerprint,
            recorded_at: changeTime,
            entry_ids: plan.entry_ids,
            proposal: {
                kind: 'replace_with_compaction' as const,
                compaction_id: 'compaction:external',
                strategy: { id: 'externalize-text', version: '1', configuration_fingerprint: 'sha256:config' },
                replacement_turns: [replacement],
                fidelity: 'retrievable' as const,
                accepted_asset_operation_id: 'append:archive',
                retained_asset_ids: [asset.id],
                generation_ids: [],
                placement: { mode: 'first_selected' as const, causal_order: 'contiguous' as const },
            },
        };
        const applied = await applyContextChange(staged, request);
        expect(applied.applied).toBe(true);
        expect(applied.document.turns).toEqual(staged.turns);
        expect(applied.document.context.entries[0].type).toBe('replacement_turn');
        expect(applied.document.context.retrieval_requirements).toMatchObject([
            {
                asset_id: asset.id,
                accepted_asset_operation_id: 'append:archive',
                retrieval: replacement.blocks[0].retrieval,
            },
        ]);
        expect(applied.document.compactions['compaction:external'].fidelity).toBe('retrievable');
        const retry = await applyContextChange(applied.document, request);
        expect(retry.applied).toBe(false);
        expect(retry.document).toEqual(applied.document);

        const nextPlan = await planContextChange(applied.document, {
            expected_revision: applied.document.revision,
            expected_context_revision: applied.document.context.revision,
            entry_ids: ['second-entry'],
        });
        const nextRequest = {
            ...request,
            operation_id: 'context:externalize-again',
            expected_revision: applied.document.revision,
            expected_context_revision: applied.document.context.revision,
            expected_source_fingerprint: nextPlan.source_fingerprint,
            entry_ids: nextPlan.entry_ids,
            proposal: {
                ...request.proposal,
                compaction_id: 'compaction:external-again',
                replacement_turns: [
                    {
                        ...replacement,
                        id: 'reference-turn-again',
                        provenance: {
                            type: 'derived' as const,
                            derivation_id: 'compaction:external-again',
                            source_turn_ids: nextPlan.source_turn_ids,
                            ...(nextPlan.source_block_ids.length
                                ? { source_block_ids: nextPlan.source_block_ids }
                                : {}),
                            source_hash: nextPlan.source_fingerprint,
                        },
                        blocks: [{ ...replacement.blocks[0], id: 'block:external-again' }],
                    },
                ],
            },
        };
        const repeated = await applyContextChange(applied.document, nextRequest);
        expect(repeated.document.context.retrieval_requirements).toHaveLength(1);
        expect(resolveActiveTextExternalReference(repeated.document, asset.id, 'block:external')).toMatchObject({
            accepted_asset_operation_id: 'append:archive',
        });
        expect(resolveActiveTextExternalReference(repeated.document, asset.id, 'block:external-again')).toMatchObject({
            accepted_asset_operation_id: 'append:archive',
        });

        const otherHost = structuredClone(staged);
        otherHost.assets[asset.id].storage = {
            type: 'external',
            resolver: 'acme.blob_store',
            locator: { opaque_key: 'blob:one' },
        };
        otherHost.tool_definitions[readTool.id].name = 'retrieve_blob';
        const otherRequest = {
            ...request,
            proposal: {
                ...request.proposal,
                replacement_turns: [
                    {
                        ...replacement,
                        blocks: [
                            {
                                ...replacement.blocks[0],
                                retrieval: {
                                    ...replacement.blocks[0].retrieval,
                                    capability: 'retrieve_blob',
                                    arguments: { opaque_key: 'blob:one' },
                                },
                            },
                        ],
                    },
                ],
            },
        };
        const otherApplied = await applyContextChange(otherHost, otherRequest);
        expect(otherApplied.applied).toBe(true);
        expect(otherApplied.document.context.retrieval_requirements[0]?.retrieval).toMatchObject({
            capability: 'retrieve_blob',
            arguments: { opaque_key: 'blob:one' },
        });

        const missingAsset = structuredClone(staged);
        delete missingAsset.assets[asset.id];
        await expect(applyContextChange(missingAsset, request)).rejects.toThrow();
        const wrongAsset = structuredClone(staged);
        wrongAsset.assets[asset.id].content_hash = `sha256:${'0'.repeat(64)}`;
        await expect(applyContextChange(wrongAsset, request)).rejects.toThrow();
        const inactiveTool = structuredClone(staged);
        inactiveTool.context.active_tool_definition_ids = [];
        await expect(applyContextChange(inactiveTool, request)).rejects.toThrow();
        const wrongReceipt = structuredClone(request);
        wrongReceipt.proposal.accepted_asset_operation_id = 'append:missing';
        await expect(applyContextChange(staged, wrongReceipt)).rejects.toThrow();
    });

    it('excludes only active context, retains source and usage, and accepts an exact retry', async () => {
        const source = documentWithEntries();
        const { result, request } = await exclude(source);
        expect(result.applied).toBe(true);
        expect(result.document.turns).toEqual(source.turns);
        expect(result.document.generations).toEqual(source.generations);
        expect(result.document.context.entries.map((entry) => entry.id)).toEqual(['second-entry']);
        expect(result.change).toMatchObject({
            base_revision: 0,
            result_revision: 1,
            operations: [
                {
                    kind: 'exclude',
                    removed_entry_ids: ['first-entry'],
                    inserted_entry_ids: [],
                },
            ],
            diagnostics: [],
        });
        expect(result.document.operation_receipts['edit:first']).toMatchObject({
            operation_kind: 'context_change',
            context_change: { kind: 'exclude', removed_entry_ids: ['first-entry'] },
        });
        const retry = await applyContextChange(result.document, request);
        expect(retry.applied).toBe(false);
        expect(retry.document).toEqual(result.document);
        expect(retry.change).toEqual(result.change);
        const later = appendConversationRecords(
            result.document,
            {},
            {
                operation_id: 'append:later',
                expected_revision: result.document.revision,
                payload_fingerprint: 'sha256:later-payload',
                recorded_at: changeTime,
            },
        );
        const laterRetry = await applyContextChange(later.document, request);
        expect(laterRetry.applied).toBe(false);
        expect(laterRetry.document).toEqual(later.document);
        expect(laterRetry.change).toEqual(result.change);
        await expect(applyContextChange(result.document, { ...request, entry_ids: ['second-entry'] })).rejects.toThrow(
            'conflicts with its accepted payload',
        );
        const forged = structuredClone(result.document);
        forged.operation_receipts['edit:first'].context_change = {
            kind: 'exclude',
            removed_entry_ids: ['second-entry'],
            inserted_entry_ids: [],
            source_fingerprint: 'sha256:unrelated-source',
        };
        await expect(applyContextChange(parseConversationDocument(forged), request)).rejects.toThrow(
            'conflicting retained details',
        );
        const forgedAcceptance = structuredClone(result.document);
        forgedAcceptance.operation_receipts['edit:first'].accepted_asset_ids = ['unrelated-asset'];
        await expect(applyContextChange(parseConversationDocument(forgedAcceptance), request)).rejects.toThrow(
            'conflicting retained details',
        );
        const forgedTime = structuredClone(result.document);
        forgedTime.operation_receipts['edit:first'].recorded_at = '2026-09-11T00:02:00.000Z';
        await expect(applyContextChange(parseConversationDocument(forgedTime), request)).rejects.toThrow(
            'conflicting retained details',
        );
        expect(() =>
            appendConversationRecords(
                result.document,
                {},
                {
                    operation_id: 'edit:first',
                    expected_revision: result.document.revision,
                    payload_fingerprint: result.document.operation_receipts['edit:first'].payload_fingerprint,
                    recorded_at: changeTime,
                },
            ),
        ).toThrow('belongs to a context change');
    });

    it('rejects stale revisions, changed selected bytes, and protected or required-cache edits', async () => {
        const source = documentWithEntries();
        const { request } = await exclude(source);
        await expect(applyContextChange(source, { ...request, expected_revision: 1 })).rejects.toThrow(
            'revision conflict',
        );
        const changed = structuredClone(source);
        changed.turns[0].blocks[0] = { id: 'first-text', type: 'text', text: 'changed', format: 'plain' };
        await expect(applyContextChange(changed, request)).rejects.toThrow('source fingerprint conflict');
        const protectedSource = structuredClone(source);
        protectedSource.context.protected_entry_ids = ['first-entry'];
        await expect(planContextChange(protectedSource, request)).rejects.toThrow('protected entry');
        const requiredCache = structuredClone(source);
        requiredCache.context.cache_intent = { namespace: 'cache', mode: 'required' };
        await expect(applyContextChange(requiredCache, request)).rejects.toThrow('required cache intent');
    });

    it('clears an automatic cache boundary after an accepted selection change', async () => {
        const source = documentWithEntries();
        source.context.cache_intent = { namespace: 'cache', mode: 'auto', stable_through_entry_id: 'first-entry' };
        const { result } = await exclude(source);
        expect(result.document.context.cache_intent).toEqual({ namespace: 'cache', mode: 'auto' });
    });

    it('does not accept an append receipt as a context-edit retry', async () => {
        const source = documentWithEntries();
        const { request } = await exclude(source);
        const appended = appendConversationRecords(
            source,
            {},
            {
                operation_id: request.operation_id,
                expected_revision: source.revision,
                payload_fingerprint: 'sha256:an-append-payload',
                recorded_at: changeTime,
            },
        );
        await expect(applyContextChange(appended.document, request)).rejects.toThrow('different mutation kind');
    });

    it('owns the selected source and request before asynchronous hashing', async () => {
        const baseline = documentWithEntries();
        const source = structuredClone(baseline);
        const selector = ['first-entry'];
        const pendingPlan = planContextChange(source, {
            expected_revision: 0,
            expected_context_revision: 0,
            entry_ids: selector,
        });
        source.turns[0].blocks[0] = { id: 'first-text', type: 'text', text: 'changed', format: 'plain' };
        selector[0] = 'second-entry';
        const plan = await pendingPlan;
        const expected = await planContextChange(baseline, {
            expected_revision: 0,
            expected_context_revision: 0,
            entry_ids: ['first-entry'],
        });
        expect(plan).toEqual(expected);

        const request = {
            operation_id: 'edit:owned',
            expected_revision: 0,
            expected_context_revision: 0,
            expected_source_fingerprint: expected.source_fingerprint,
            recorded_at: changeTime,
            entry_ids: ['first-entry'],
            proposal: { kind: 'exclude' as const },
        };
        const pendingApply = applyContextChange(baseline, request);
        request.entry_ids[0] = 'second-entry';
        baseline.turns[0].blocks[0] = { id: 'first-text', type: 'text', text: 'changed', format: 'plain' };
        const applied = await pendingApply;
        expect(applied.change.operations[0].removed_entry_ids).toEqual(['first-entry']);
        expect(applied.document.turns[0].blocks[0]).toMatchObject({ text: 'first-text' });
    });

    it('rejects an incomplete or pending tool exchange and a retained replay dependency', async () => {
        const source = documentWithEntries();
        const call = {
            ...userTurn('call-turn'),
            kind: 'agent' as const,
            provenance: { type: 'received' as const },
            blocks: [toolCallBlock('call-block', 'call')],
        };
        const result = toolResultTurn('result-turn', 'call');
        source.turns.push(call, result);
        source.context.entries.push(
            { id: 'call-entry', type: 'source_turn', turn_id: call.id },
            { id: 'result-entry', type: 'source_turn', turn_id: result.id },
        );
        await expect(
            planContextChange(source, {
                expected_revision: 0,
                expected_context_revision: 0,
                entry_ids: ['call-entry'],
            }),
        ).rejects.toThrow('split application tool exchange');
        const pending = structuredClone(source);
        pending.context.entries.pop();
        await expect(
            planContextChange(pending, {
                expected_revision: 0,
                expected_context_revision: 0,
                entry_ids: ['call-entry'],
            }),
        ).rejects.toThrow('pending tool call');
        const replay = {
            ...userTurn('replay-turn'),
            kind: 'agent' as const,
            provenance: { type: 'received' as const },
            blocks: [
                {
                    id: 'replay-block',
                    type: 'native_replay' as const,
                    adapter: 'adapter',
                    protocol: 'protocol',
                    compatibility_scope: { provider: 'provider', protocol: 'protocol', adapter_version: 'v1' },
                    payload: { state: 'opaque' },
                    dependencies: { turn_ids: ['first'], block_ids: ['first-text'], call_ids: [], request_ids: [] },
                },
            ],
        };
        const withReplay = documentWithEntries();
        withReplay.turns.push(replay);
        withReplay.context.entries.push({ id: 'replay-entry', type: 'source_turn', turn_id: replay.id });
        await expect(
            planContextChange(withReplay, {
                expected_revision: 0,
                expected_context_revision: 0,
                entry_ids: ['first-entry'],
            }),
        ).rejects.toThrow('orphan native replay dependency');
    });

    it('retains provider call IDs needed by native replay independently of application pairing', async () => {
        const source = emptyDocument();
        const provider = {
            ...userTurn('provider'),
            kind: 'agent' as const,
            provenance: { type: 'received' as const },
            blocks: [toolCallBlock('provider-call', 'pcall', 'provider')],
        };
        const replay = {
            ...userTurn('replay'),
            kind: 'agent' as const,
            provenance: { type: 'received' as const },
            blocks: [replayBlock('provider-replay', { call_ids: ['pcall'] })],
        };
        source.turns.push(provider, replay);
        source.context.entries.push(
            { id: 'provider-entry', type: 'source_turn', turn_id: provider.id },
            { id: 'replay-entry', type: 'source_turn', turn_id: replay.id },
        );
        const valid = parseConversationDocument(source);
        await expect(
            planContextChange(valid, {
                expected_revision: 0,
                expected_context_revision: 0,
                entry_ids: ['provider-entry'],
            }),
        ).rejects.toThrow('orphan native replay dependency');
    });

    it('rejects a partial turn cut when replay depends on the entire turn', async () => {
        const source = emptyDocument();
        const partial = userTurn('partial');
        partial.blocks.push(textBlock('second-block'));
        const replay = {
            ...userTurn('replay'),
            kind: 'agent' as const,
            provenance: { type: 'received' as const },
            blocks: [replayBlock('partial-replay', { turn_ids: ['partial'] })],
        };
        source.turns.push(partial, replay);
        source.context.entries.push(
            { id: 'first-half', type: 'source_turn', turn_id: partial.id, block_ids: ['partial-text'] },
            { id: 'second-half', type: 'source_turn', turn_id: partial.id, block_ids: ['second-block'] },
            { id: 'replay-entry', type: 'source_turn', turn_id: replay.id },
        );
        const valid = parseConversationDocument(source);
        await expect(
            planContextChange(valid, {
                expected_revision: 0,
                expected_context_revision: 0,
                entry_ids: ['first-half'],
            }),
        ).rejects.toThrow('orphan native replay dependency');
    });

    it('traverses replay blocks nested inside retained tool results', async () => {
        const source = emptyDocument();
        const first = userTurn('first');
        const call = {
            ...userTurn('call'),
            kind: 'agent' as const,
            provenance: { type: 'received' as const },
            blocks: [toolCallBlock('call-block', 'call-id')],
        };
        const result = toolResultTurn('result', 'call-id');
        result.blocks[0].content = [replayBlock('nested-replay', { turn_ids: ['first'] })];
        source.turns.push(first, call, result);
        source.context.entries.push(
            { id: 'first-entry', type: 'source_turn', turn_id: first.id },
            { id: 'call-entry', type: 'source_turn', turn_id: call.id },
            { id: 'result-entry', type: 'source_turn', turn_id: result.id },
        );
        const valid = parseConversationDocument(source);
        await expect(
            planContextChange(valid, {
                expected_revision: 0,
                expected_context_revision: 0,
                entry_ids: ['first-entry'],
            }),
        ).rejects.toThrow('orphan native replay dependency');
    });

    it('requires a recorded causal-order policy when combining disjoint ranges', async () => {
        const source = documentWithEntries();
        source.turns.push(userTurn('third'));
        source.context.entries.push({ id: 'third-entry', type: 'source_turn', turn_id: 'third' });
        const plan = await planContextChange(source, {
            expected_revision: 0,
            expected_context_revision: 0,
            entry_ids: ['third-entry', 'first-entry'],
        });
        expect(plan.disjoint_ranges).toBe(2);
        expect(plan.entry_ids).toEqual(['first-entry', 'third-entry']);
        const excluded = await applyContextChange(source, {
            operation_id: 'edit:ordered-selection',
            expected_revision: 0,
            expected_context_revision: 0,
            expected_source_fingerprint: plan.source_fingerprint,
            recorded_at: changeTime,
            entry_ids: plan.entry_ids,
            proposal: { kind: 'exclude' },
        });
        expect(excluded.document.context.entries.map((entry) => entry.id)).toEqual(['second-entry']);
        const replacement = {
            ...userTurn('summary-turn', 'summary-text'),
            kind: 'agent' as const,
            provenance: {
                type: 'derived' as const,
                derivation_id: 'summary-compaction',
                source_turn_ids: plan.source_turn_ids,
                source_hash: plan.source_fingerprint,
            },
        };
        const request = {
            operation_id: 'edit:summary',
            expected_revision: 0,
            expected_context_revision: 0,
            expected_source_fingerprint: plan.source_fingerprint,
            recorded_at: changeTime,
            entry_ids: plan.entry_ids,
            proposal: {
                kind: 'replace_with_compaction' as const,
                compaction_id: 'summary-compaction',
                strategy: { id: 'test-summary', version: '1', configuration_fingerprint: 'sha256:configuration' },
                replacement_turns: [replacement],
                fidelity: 'semantic' as const,
                retained_asset_ids: [],
                generation_ids: [],
                placement: { mode: 'first_selected' as const, causal_order: 'explicit_disjoint_summary' as const },
            },
        };
        await expect(
            applyContextChange(source, {
                ...request,
                proposal: { ...request.proposal, placement: { mode: 'first_selected', causal_order: 'contiguous' } },
            }),
        ).rejects.toThrow('explicit causal-order policy');
        await expect(
            applyContextChange(source, {
                ...request,
                operation_id: 'edit:executable-summary',
                proposal: {
                    ...request.proposal,
                    replacement_turns: [{ ...replacement, blocks: [toolCallBlock('unsafe-call', 'unsafe')] }],
                },
            }),
        ).rejects.toThrow('without executable content');
        const result = await applyContextChange(source, request);
        expect(result.document.context.entries.map((entry) => entry.type)).toEqual(['replacement_turn', 'source_turn']);
        expect(result.change.operations[0].placement?.causal_order).toBe('explicit_disjoint_summary');
        expect(result.document.compactions['summary-compaction'].source.turn_ids).toEqual(['first', 'third']);
    });
});
