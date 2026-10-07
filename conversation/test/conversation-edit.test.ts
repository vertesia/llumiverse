import { describe, expect, it } from 'vitest';
import {
    appendConversationRecords,
    applyContextChange,
    applyConversationEdit,
    type ConversationDocument,
    type ConversationEditPlanInput,
    type ConversationEditRequest,
    ConversationEditRequestSchema,
    type ConversationInsertedTurn,
    ConversationInsertedTurnSchema,
    type ConversationReplacementTurn,
    ConversationReplacementTurnSchema,
    type ConversationSelection,
    fingerprintJson,
    parseConversationDocument,
    planContextChange,
    planConversationEdit,
    resolveConversationSelection,
} from '../src/index.js';
import { emptyDocument, textBlock, toolCallBlock, toolResultTurn, userTurn } from './fixtures.js';

const at = '2026-10-02T00:00:00.000Z';
function source(): ConversationDocument {
    const doc = emptyDocument();
    doc.turns = [{ ...userTurn('first'), blocks: [textBlock('a'), textBlock('b'), textBlock('c')] }, userTurn('last')];
    doc.context.entries = doc.turns.map((turn) => ({ id: `entry:${turn.id}`, type: 'source_turn', turn_id: turn.id }));
    return parseConversationDocument(doc);
}
async function select(doc: ConversationDocument, block_ids?: string[]): Promise<ConversationSelection> {
    const result = await resolveConversationSelection(doc, {
        conversation: { conversation_id: doc.id, revision: doc.revision },
        expected_context_revision: doc.context.revision,
        selector: { source: { kind: 'all' }, ...(block_ids ? { filters: { block_ids } } : {}) },
    });
    if (result.kind !== 'selected') throw new Error(JSON.stringify(result));
    return result.selection;
}
async function request(
    doc: ConversationDocument,
    command: ConversationEditPlanInput['command'],
    operation_id = 'edit',
): Promise<ConversationEditRequest> {
    const input: ConversationEditPlanInput = {
        version: 1,
        operation_id,
        conversation: { conversation_id: doc.id, revision: doc.revision },
        expected_context_revision: doc.context.revision,
        recorded_at: at,
        command,
    };
    const plan = await planConversationEdit(doc, input);
    return ConversationEditRequestSchema.parse({
        ...input,
        expected_source_fingerprint: plan.operation.source_fingerprint,
    });
}
function inserted(id = 'new', op = 'edit'): ConversationInsertedTurn {
    return ConversationInsertedTurnSchema.parse({
        ...userTurn(id),
        timestamps: { recorded_at: at },
        provenance: { type: 'inserted', operation_id: op },
    });
}
async function replacement(
    doc: ConversationDocument,
    selection: ConversationSelection,
    id = 'replacement',
    op = 'edit',
): Promise<ConversationReplacementTurn> {
    const partial: Record<string, string[]> = {};
    let hasPartial = false;
    for (const item of selection.entries) {
        const turn = doc.turns.find((turn) => turn.id === item.entry.turn_id);
        if (!turn) throw new Error('Source fixture required');
        const actual = item.entry.block_ids ?? turn.blocks.map((block) => block.id);
        if (item.blocks.length !== actual.length) {
            partial[item.entry.id] = item.blocks.map((block) => block.block_id);
            hasPartial = true;
        }
    }
    const plan = await planContextChange(doc, {
        expected_revision: doc.revision,
        expected_context_revision: doc.context.revision,
        entry_ids: selection.entries.map((item) => item.entry.id),
        ...(hasPartial
            ? { selected_block_ids: partial, selected_entries: selection.entries.map((item) => item.entry) }
            : {}),
    });
    return ConversationReplacementTurnSchema.parse({
        ...userTurn(id),
        timestamps: { recorded_at: at },
        blocks: [textBlock(`${id}:text`, 'replacement')],
        provenance: {
            type: 'derived',
            derivation_id: op,
            source_hash: selection.source_fingerprint,
            source_turn_ids: plan.source_turn_ids,
            ...(plan.source_block_ids.length ? { source_block_ids: plan.source_block_ids } : {}),
        },
    });
}

describe('whole-block canonical edits', () => {
    it('selectively protects/unprotects blocks with deterministic references, inherited pins and exact later retries', async () => {
        const doc = source();
        doc.context.protected_entry_ids = ['entry:first'];
        const input = await request(doc, { kind: 'protect', selection: await select(doc, ['b']), protected: false });
        const result = await applyConversationEdit(doc, input);
        expect(result.document.turns).toEqual(doc.turns);
        expect(result.document.context.entries.slice(0, 3).map((entry) => entry.block_ids)).toEqual([
            ['a'],
            ['b'],
            ['c'],
        ]);
        expect(result.document.context.protected_entry_ids).toEqual([
            result.document.context.entries[0].id,
            result.document.context.entries[2].id,
        ]);
        expect(result.change).toMatchObject({
            operation_id: 'edit',
            conversation_id: doc.id,
            base_revision: 0,
            result_revision: 1,
            operations: [{ kind: 'protect' }],
        });
        const later = await applyConversationEdit(
            result.document,
            await request(
                result.document,
                { kind: 'protect', selection: await select(result.document, ['b']), protected: true },
                'later',
            ),
        );
        const retry = await applyConversationEdit(JSON.parse(JSON.stringify(later.document)), input);
        expect(retry.applied).toBe(false);
        expect(retry.change).toEqual(result.change);
        expect(retry.document.operation_receipts.edit).toEqual(result.document.operation_receipts.edit);
        const changed = structuredClone(input);
        if (changed.command.kind !== 'protect') throw new Error('fixture');
        changed.command.protected = true;
        await expect(applyConversationEdit(result.document, changed)).rejects.toThrow('conflicts');
        const drift = structuredClone(result.document);
        drift.operation_receipts.edit.conversation_edit!.created_entries[0].fingerprint = 'sha256:drift';
        await expect(applyConversationEdit(drift, input)).rejects.toThrow('details');
    });
    it('removes contextual pins without altering intrinsic instruction/replay protection', async () => {
        const doc = source();
        doc.turns[0].authority = 'system';
        doc.context.protected_entry_ids = ['entry:first'];
        const result = await applyConversationEdit(
            doc,
            await request(doc, { kind: 'protect', selection: await select(doc, ['a', 'b', 'c']), protected: false }),
        );
        expect(result.document.context.protected_entry_ids).toEqual([]);
        expect(result.document.turns[0].authority).toBe('system');
        await expect(
            planContextChange(result.document, {
                expected_revision: 1,
                expected_context_revision: 1,
                entry_ids: ['entry:first'],
            }),
        ).rejects.toThrow('protected system');
        const replay = {
            ...userTurn('replay'),
            kind: 'agent' as const,
            provenance: { type: 'received' as const },
            blocks: [
                {
                    id: 'replay-block',
                    type: 'native_replay' as const,
                    adapter: 'adapter',
                    protocol: 'protocol',
                    compatibility_scope: { provider: 'provider', protocol: 'protocol', adapter_version: '1' },
                    payload: {},
                    dependencies: { turn_ids: [], block_ids: [], call_ids: [], request_ids: [] },
                },
            ],
        };
        const rd = emptyDocument();
        rd.turns.push(replay);
        rd.context.entries.push({ id: 'replay-entry', type: 'source_turn', turn_id: replay.id });
        rd.context.protected_entry_ids = ['replay-entry'];
        const unpinned = await applyConversationEdit(
            rd,
            await request(rd, { kind: 'protect', selection: await select(rd), protected: false }),
        );
        await expect(
            planContextChange(unpinned.document, {
                expected_revision: 1,
                expected_context_revision: 1,
                entry_ids: ['replay-entry'],
            }),
        ).rejects.toThrow('protected native replay');
    });
    it('inserts at exact anchors and preserves arbitrary JSON/null/media with source-bound provenance', async () => {
        const doc = source();
        const turn = inserted();
        turn.blocks = [
            { id: 'new:json', type: 'json', value: { null_value: null, list: [false, 0, ''] } },
            { id: 'new:image', type: 'image', asset_id: 'asset' },
        ];
        const asset = {
            id: 'asset',
            kind: 'image' as const,
            mime_type: 'image/png',
            storage: { type: 'inline_base64' as const, data: 'YWJj' },
            provenance: { type: 'received' as const, source_turn_id: turn.id },
            created_at: at,
        };
        const input = await request(doc, {
            kind: 'insert',
            anchor: { kind: 'after_entry', entry_id: 'entry:first' },
            turns: [turn],
            assets: [asset],
        });
        const result = await applyConversationEdit(doc, input);
        expect(result.document.context.entries.map((entry) => entry.turn_id)).toEqual(['first', 'new', 'last']);
        expect(result.document.turns.slice(0, 2)).toEqual(doc.turns);
        expect(result.document.turns[2]).toEqual(turn);
        expect(result.document.assets.asset).toEqual(asset);
        expect((await applyConversationEdit(JSON.parse(JSON.stringify(result.document)), input)).change).toEqual(
            result.change,
        );
        for (const anchor of [
            { kind: 'head' as const },
            { kind: 'tail' as const },
            { kind: 'before_entry' as const, entry_id: 'entry:first' },
        ]) {
            const next = await applyConversationEdit(
                doc,
                await request(doc, { kind: 'insert', anchor, turns: [inserted()] }),
            );
            expect(next.document.context.entries[anchor.kind === 'tail' ? 2 : 0].turn_id).toBe('new');
        }
        await expect(
            request(doc, { kind: 'insert', anchor: { kind: 'before_entry', entry_id: 'absent' }, turns: [inserted()] }),
        ).rejects.toThrow('anchor');
        await expect(
            request(doc, { kind: 'insert', anchor: { kind: 'tail' }, turns: [inserted('first')] }),
        ).rejects.toThrow();
    });
    it('strictly rejects fabricated execution/generation/replay/elevated received-agent content', () => {
        const agent = { ...inserted(), kind: 'agent' };
        expect(ConversationInsertedTurnSchema.safeParse(agent).success).toBe(true);
        for (const invalid of [
            { ...agent, generation_id: 'fake' },
            { ...agent, execution_id: 'fake' },
            { ...agent, exchange_id: 'fake' },
            { ...agent, authority: 'system' },
            { ...agent, provenance: { type: 'generated' } },
            { ...agent, blocks: [toolCallBlock('call-block', 'call-id')] },
            { ...agent, kind: 'tool', blocks: toolResultTurn('tool', 'call-id').blocks },
        ])
            expect(ConversationInsertedTurnSchema.safeParse(invalid).success).toBe(false);
    });
    it('replaces partial/disjoint whole blocks with exact selected-source provenance and retains every unselected record', async () => {
        const doc = source(),
            selection = await select(doc, ['b', 'last-text']),
            turn = await replacement(doc, selection);
        const input = await request(doc, {
            kind: 'replace',
            selection,
            replacement_turn: turn,
            fidelity: 'semantic',
            placement: { mode: 'first_selected', causal_order: 'explicit_disjoint' },
        });
        const result = await applyConversationEdit(doc, input);
        expect(result.document.turns.slice(0, 2)).toEqual(doc.turns);
        expect(result.document.context.entries.map((entry) => entry.block_ids ?? entry.turn_id)).toEqual([
            ['a'],
            'replacement',
            ['c'],
        ]);
        expect(result.document.turns[2].provenance).toEqual(turn.provenance);
        expect(result.change.operations[0]).toMatchObject({
            kind: 'replace',
            fidelity: 'semantic',
            placement: input.command.kind === 'replace' ? input.command.placement : undefined,
        });
        expect(result.document.operation_receipts.edit.conversation_edit).toEqual(result.change.operations[0]);
        const retained = parseConversationDocument(JSON.parse(JSON.stringify(result.document)));
        const fidelityDrift = structuredClone(retained);
        const detail = fidelityDrift.operation_receipts.edit.conversation_edit;
        if (detail?.kind !== 'replace') throw new Error('fixture');
        detail.fidelity = 'heuristic';
        await expect(applyConversationEdit(fidelityDrift, input)).rejects.toThrow('fidelity or placement');
        const placementDrift = structuredClone(retained);
        const placement = placementDrift.operation_receipts.edit.conversation_edit;
        if (placement?.kind !== 'replace') throw new Error('fixture');
        placement.placement.causal_order = 'contiguous';
        await expect(applyConversationEdit(placementDrift, input)).rejects.toThrow('fidelity or placement');

        expect((await applyConversationEdit(JSON.parse(JSON.stringify(result.document)), input)).applied).toBe(false);
        const changed = structuredClone(input);
        if (changed.command.kind !== 'replace') throw new Error('fixture');
        changed.command.replacement_turn.provenance.source_hash = 'sha256:wrong';
        await expect(applyConversationEdit(doc, changed)).rejects.toThrow('provenance');
        changed.command = { ...changed.command, placement: { mode: 'first_selected', causal_order: 'contiguous' } };
        await expect(applyConversationEdit(doc, changed)).rejects.toThrow('causal');
    });
    it('blocks reintroduced original coverage through one or multiple whole-block replacement ancestors', async () => {
        const doc = source(),
            selection = await select(doc, ['a', 'b', 'c']);
        const first = await applyConversationEdit(
            doc,
            await request(doc, {
                kind: 'replace',
                selection,
                replacement_turn: await replacement(doc, selection, 'B'),
                fidelity: 'semantic',
                placement: { mode: 'first_selected', causal_order: 'contiguous' },
            }),
        );
        const reintroduce = (input: ConversationDocument) =>
            appendConversationRecords(
                input,
                { context_entries: [{ id: 'reintroduced', type: 'source_turn', turn_id: 'first' }] },
                {
                    expected_revision: input.revision,
                    operation_id: 'reintroduce',
                    payload_fingerprint: 'same',
                    recorded_at: at,
                },
            );
        expect(() => reintroduce(first.document)).toThrow('validation');
        const direct = structuredClone(first.document);
        direct.context.entries.push({ id: 'reintroduced', type: 'source_turn', turn_id: 'first' });
        expect(() => parseConversationDocument(direct)).toThrow('validation');
        const nextSelection = await select(first.document, ['B:text']);
        const second = await applyConversationEdit(
            first.document,
            await request(
                first.document,
                {
                    kind: 'replace',
                    selection: nextSelection,
                    replacement_turn: await replacement(first.document, nextSelection, 'C', 'edit:C'),
                    fidelity: 'semantic',
                    placement: { mode: 'first_selected', causal_order: 'contiguous' },
                },
                'edit:C',
            ),
        );
        expect(() => reintroduce(second.document)).toThrow('validation');
        const partial = structuredClone(second.document);
        const turn = partial.turns[partial.turns.length - 1];
        if (turn.kind !== 'user') throw new Error('fixture');
        turn.blocks.push(textBlock('extra'));
        partial.context.entries[0].block_ids = ['C:text'];
        expect(() => parseConversationDocument(partial)).toThrow('validation');
    });
    it('protects complete and open canonical application-call intervals while preserving provider-call behavior', async () => {
        const doc = source();
        doc.turns = [
            {
                ...userTurn('call'),
                kind: 'agent',
                provenance: { type: 'received' },
                blocks: [toolCallBlock('call-block', 'call-id'), toolCallBlock('other-block', 'other-id')],
            },
            userTurn('between'),
            toolResultTurn('result', 'call-id'),
        ];
        doc.context.entries = doc.turns.map((turn) => ({
            id: `entry:${turn.id}`,
            type: 'source_turn',
            turn_id: turn.id,
        }));
        for (const anchor of [
            { kind: 'tail' as const },
            { kind: 'after_entry' as const, entry_id: 'entry:call' },
            { kind: 'before_entry' as const, entry_id: 'entry:result' },
        ])
            await expect(request(doc, { kind: 'insert', anchor, turns: [inserted()] })).rejects.toThrow(
                'causal interval',
            );
        expect(
            (
                await applyConversationEdit(
                    doc,
                    await request(doc, {
                        kind: 'insert',
                        anchor: { kind: 'before_entry', entry_id: 'entry:call' },
                        turns: [inserted()],
                    }),
                )
            ).document.context.entries[0].turn_id,
        ).toBe('new');
        const provider = emptyDocument();
        provider.turns.push({
            ...userTurn('provider'),
            kind: 'agent',
            provenance: { type: 'received' },
            blocks: [toolCallBlock('provider-block', 'provider-call', 'provider')],
        });
        provider.context.entries.push({ id: 'provider-entry', type: 'source_turn', turn_id: 'provider' });
        expect(
            (
                await applyConversationEdit(
                    provider,
                    await request(provider, { kind: 'insert', anchor: { kind: 'tail' }, turns: [inserted()] }),
                )
            ).applied,
        ).toBe(true);
        const single = structuredClone(doc);
        single.turns = [single.turns[0]];
        single.context.entries = [single.context.entries[0]];
        await expect(
            request(single, { kind: 'insert', anchor: { kind: 'tail' }, turns: [inserted()] }),
        ).rejects.toThrow('open application');
    });
    it('protects full replay dependency intervals and refuses replacement/subrange mutation without proven eligibility', async () => {
        const doc = source();
        doc.turns.push({
            ...userTurn('replay'),
            kind: 'agent',
            provenance: { type: 'received' },
            blocks: [
                {
                    id: 'replay-block',
                    type: 'native_replay',
                    adapter: 'adapter',
                    protocol: 'protocol',
                    compatibility_scope: { provider: 'provider', protocol: 'protocol', adapter_version: '1' },
                    payload: {},
                    dependencies: { turn_ids: ['first'], block_ids: [], call_ids: [], request_ids: [] },
                },
            ],
        });
        doc.context.entries.push({ id: 'entry:replay', type: 'source_turn', turn_id: 'replay' });
        await expect(
            request(doc, {
                kind: 'insert',
                anchor: { kind: 'before_entry', entry_id: 'entry:last' },
                turns: [inserted()],
            }),
        ).rejects.toThrow('replay causal');
        const range = await resolveConversationSelection(doc, {
            conversation: { conversation_id: doc.id, revision: 0 },
            expected_context_revision: 0,
            selector: {
                source: { kind: 'all' },
                filters: { block_ids: ['a'] },
                subselections: [
                    {
                        kind: 'text_range',
                        entry_id: 'entry:first',
                        block_id: 'a',
                        expected_block_fingerprint: await fingerprintJson(doc.turns[0].blocks[0]),
                        range: { start_code_point: 0, end_code_point: 1 },
                    },
                ],
            },
        });
        if (range.kind !== 'selected') throw new Error('fixture');
        await expect(request(doc, { kind: 'protect', selection: range.selection, protected: true })).rejects.toThrow(
            'source-slice lineage',
        );
    });
    it('preserves required unchanged cache prefixes; rejects affected required edits and retains auto/off namespace', async () => {
        const doc = source();
        doc.context.cache_intent = { mode: 'required', namespace: 'cache', stable_through_entry_id: 'entry:first' };
        expect(
            (
                await applyConversationEdit(
                    doc,
                    await request(doc, {
                        kind: 'protect',
                        selection: await select(doc, ['a', 'b', 'c']),
                        protected: true,
                    }),
                )
            ).document.context.cache_intent,
        ).toEqual(doc.context.cache_intent);
        expect(
            (
                await applyConversationEdit(
                    doc,
                    await request(doc, { kind: 'insert', anchor: { kind: 'tail' }, turns: [inserted()] }),
                )
            ).document.context.cache_intent,
        ).toEqual(doc.context.cache_intent);
        await expect(request(doc, { kind: 'insert', anchor: { kind: 'head' }, turns: [inserted()] })).rejects.toThrow(
            'required cache',
        );
        await expect(
            request(doc, { kind: 'protect', selection: await select(doc, ['b']), protected: true }),
        ).rejects.toThrow('required cache');
        for (const mode of ['auto', 'off'] as const) {
            const copy = structuredClone(doc);
            copy.context.cache_intent = { ...doc.context.cache_intent, mode };
            const result = await applyConversationEdit(
                copy,
                await request(copy, { kind: 'protect', selection: await select(copy, ['b']), protected: true }),
            );
            expect(result.document.context.cache_intent).toEqual({ mode, namespace: 'cache' });
        }
    });
    it('rejects cross-family operation reuse through actual append/context-edit/edit APIs and keeps historical hashes', async () => {
        const doc = source(),
            input = await request(doc, { kind: 'protect', selection: await select(doc, ['b']), protected: true });
        const edited = await applyConversationEdit(doc, input),
            receipt = edited.document.operation_receipts.edit;
        expect(() =>
            appendConversationRecords(
                edited.document,
                {},
                {
                    expected_revision: 1,
                    operation_id: 'edit',
                    payload_fingerprint: receipt.payload_fingerprint,
                    recorded_at: at,
                },
            ),
        ).toThrow('mutation kind');
        await expect(
            applyContextChange(edited.document, {
                operation_id: 'edit',
                expected_revision: 0,
                expected_context_revision: 0,
                expected_source_fingerprint: 'same',
                entry_ids: [edited.document.context.entries[0].id],
                recorded_at: at,
                proposal: { kind: 'exclude' },
            }),
        ).rejects.toThrow('mutation kind');
        const options = {
            expected_revision: 0,
            operation_id: 'append',
            payload_fingerprint: 'historical-payload',
            recorded_at: at,
        };
        const appended = appendConversationRecords(doc, {}, options);
        expect(appended.change.operations[0].kind).toBe('append');
        expect(appendConversationRecords(appended.document, {}, options).change).toEqual(appended.change);
        expect(appended.document.operation_receipts.append.payload_fingerprint).toBe('historical-payload');
        await expect(
            applyConversationEdit(
                appended.document,
                await request(
                    appended.document,
                    { kind: 'protect', selection: await select(appended.document, ['b']), protected: true },
                    'append',
                ),
            ),
        ).rejects.toThrow('mutation kind');
        expect(edited.document.operation_receipts).toMatchObject(doc.operation_receipts);
    });
    it('retains accepted append evidence across reference splitting and replacement', async () => {
        const original = source(),
            turn = original.turns[0];
        const batch = { turns: [turn], context_entries: [original.context.entries[0]] };
        const options = {
            operation_id: 'append-original',
            expected_revision: 0,
            payload_fingerprint: 'sha256:original',
            recorded_at: at,
        };
        const appended = appendConversationRecords(emptyDocument(), batch, options);
        const split = await applyConversationEdit(
            appended.document,
            await request(appended.document, {
                kind: 'protect',
                selection: await select(appended.document, ['b']),
                protected: true,
            }),
        );
        expect(appendConversationRecords(split.document, batch, options).change).toEqual(appended.change);
        const whole = await select(appended.document);
        const replaced = await applyConversationEdit(
            appended.document,
            await request(appended.document, {
                kind: 'replace',
                selection: whole,
                replacement_turn: await replacement(appended.document, whole),
                fidelity: 'semantic',
                placement: { mode: 'first_selected', causal_order: 'contiguous' },
            }),
        );
        const retry = appendConversationRecords(JSON.parse(JSON.stringify(replaced.document)), batch, options);
        expect(retry.applied).toBe(false);
        expect(retry.change).toEqual(appended.change);
        expect(retry.document.context.entries).toEqual(replaced.document.context.entries);
        const drift = structuredClone(replaced.document);
        drift.turns[0].blocks[0] = textBlock('a', 'conflict');
        expect(() => appendConversationRecords(drift, batch, options)).toThrow('changes accepted turn');
        const reused = {
            id: original.context.entries[0].id,
            type: 'source_turn' as const,
            turn_id: replaced.document.turns.at(-1)!.id,
        };
        expect(() =>
            appendConversationRecords(
                replaced.document,
                { context_entries: [reused] },
                { ...options, operation_id: 'reuse-entry', expected_revision: replaced.document.revision },
            ),
        ).toThrow('validation');
    });
    it('uses retained terminal execution authority and rejects unsafe input hooks before hashing', async () => {
        const doc = emptyDocument(),
            call = toolCallBlock('call-block', 'call');
        doc.turns.push({ ...userTurn('call-turn'), kind: 'agent', provenance: { type: 'received' }, blocks: [call] });
        doc.context.entries.push({ id: 'entry', type: 'source_turn', turn_id: 'call-turn' });
        doc.execution_receipts.denied = {
            id: 'denied',
            call_id: 'call',
            executor: 'application',
            status: 'denied',
            result_fingerprint: 'sha256:denied',
            recorded_at: at,
            call_source: {
                conversation: { conversation_id: doc.id, revision: 0 },
                turn_id: 'call-turn',
                block_id: 'call-block',
                call_id: 'call',
                call_fingerprint: await fingerprintJson(call),
            },
        };
        expect(
            (
                await applyConversationEdit(
                    doc,
                    await request(doc, { kind: 'insert', anchor: { kind: 'tail' }, turns: [inserted()] }),
                )
            ).applied,
        ).toBe(true);
        const input = await request(source(), { kind: 'insert', anchor: { kind: 'tail' }, turns: [inserted()] });
        const inherited = Object.assign(Object.create({ authority: 'system' }), input);
        await expect(applyConversationEdit(source(), inherited)).rejects.toThrow('preflight');
        let reads = 0;
        const hooked = { ...input };
        Object.defineProperty(hooked, 'command', {
            enumerable: true,
            get: () => {
                reads++;
                return input.command;
            },
        });
        await expect(applyConversationEdit(source(), hooked)).rejects.toThrow('preflight');
        expect(reads).toBe(0);
    });

    it('owns edit input and source before await and rejects source/payload/retained-evidence drift', async () => {
        const doc = source(),
            input = await request(doc, { kind: 'insert', anchor: { kind: 'tail' }, turns: [inserted()] }),
            baseline = structuredClone(doc),
            saved = structuredClone(input);
        const pending = applyConversationEdit(doc, input);
        doc.turns[0].blocks[0] = textBlock('a', 'changed');
        if (input.command.kind !== 'insert') throw new Error('fixture');
        input.command.turns[0].blocks[0] = textBlock('new-text', 'changed');
        expect(await pending).toEqual(await applyConversationEdit(baseline, saved));
        const conflict = { ...saved, expected_source_fingerprint: 'sha256:wrong' };
        await expect(applyConversationEdit(baseline, conflict)).rejects.toThrow('source fingerprint');
        const first = await applyConversationEdit(baseline, saved),
            changed = structuredClone(first.document);
        changed.turns[2].blocks[0] = textBlock('new-text', 'different');
        await expect(applyConversationEdit(changed, saved)).rejects.toThrow('accepted turn');
    });
});
