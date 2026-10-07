import { Ajv2020 } from 'ajv/dist/2020.js';
import formats from 'ajv-formats';
import { describe, expect, expectTypeOf, it } from 'vitest';
import type { z } from 'zod';
import {
    applyContextChange,
    type ContextChangePlan,
    type ContextChangeRequest,
    ContextChangeRequestSchema,
    type ContextSelectionRequest,
    ContextSelectionRequestSchema,
    type ContextSelectionResult,
    ContextSelectionResultSchema,
    type ContextSelector,
    ContextSelectorSchema,
    type ConversationDocument,
    deriveConversationId,
    fingerprintJson,
    parseConversationDocument,
    planContextChange,
    resolveContextSelection,
} from '../src/index.js';
import {
    CONVERSATION_JSON_SCHEMAS,
    ContextChangeRequestJsonSchema,
    ContextSelectionRequestJsonSchema,
    ContextSelectionResultJsonSchema,
    ContextSelectorJsonSchema,
} from '../src/json-schema.js';
import { emptyDocument, textBlock, toolCallBlock, toolResultTurn, userTurn } from './fixtures.js';

function source() {
    const document = emptyDocument();
    const first = userTurn('first');
    first.actor_id = 'alice';
    first.metadata = { topic: 'finance', score: 3, nested: { ready: true } };
    first.blocks = [textBlock('a'), textBlock('b'), textBlock('c')];
    const second = userTurn('second');
    second.actor_id = 'bob';
    second.metadata = { topic: 'engineering' };
    document.turns.push(first, second, userTurn('third'));
    document.context.entries = document.turns.map((turn) => ({
        id: `${turn.id}-entry`,
        type: 'source_turn',
        turn_id: turn.id,
    }));
    return parseConversationDocument(document);
}
function query(document: ConversationDocument, selector: ContextSelector): ContextSelectionRequest {
    return {
        conversation: { conversation_id: document.id, revision: document.revision },
        expected_context_revision: document.context.revision,
        selector,
    };
}
async function select(document: ConversationDocument, selector: ContextSelector): Promise<ContextChangePlan> {
    const result = await resolveContextSelection(document, query(document, selector));
    if (result.kind !== 'selected') throw new Error(JSON.stringify(result));
    return result.plan;
}
function request(document: ConversationDocument, plan: ContextChangePlan, operation = 'edit'): ContextChangeRequest {
    return ContextChangeRequestSchema.parse({
        operation_id: operation,
        expected_revision: document.revision,
        expected_context_revision: document.context.revision,
        expected_source_fingerprint: plan.source_fingerprint,
        entry_ids: plan.entry_ids,
        ...(plan.selected_block_ids
            ? { selected_block_ids: plan.selected_block_ids, selected_entries: plan.selected_entries }
            : {}),
        recorded_at: '2026-10-02T00:00:00.000Z',
        proposal: { kind: 'exclude' },
    });
}
function compactionRequest(document: ConversationDocument, plan: ContextChangePlan, id: string): ContextChangeRequest {
    return ContextChangeRequestSchema.parse({
        ...request(document, plan, `edit:${id}`),
        proposal: {
            kind: 'replace_with_compaction',
            compaction_id: id,
            strategy: { id: 'summary', version: '1', configuration_fingerprint: 'sha256:config' },
            replacement_turns: [
                {
                    ...userTurn(`${id}-turn`),
                    kind: 'agent',
                    provenance: {
                        type: 'derived',
                        derivation_id: id,
                        source_hash: plan.source_fingerprint,
                        source_turn_ids: plan.source_turn_ids,
                        ...(plan.source_block_ids.length ? { source_block_ids: plan.source_block_ids } : {}),
                    },
                    blocks: [textBlock(`${id}-block`, 'summary')],
                },
            ],
            fidelity: 'semantic',
            retained_asset_ids: [],
            generation_ids: [],
            placement: {
                mode: 'first_selected',
                causal_order: plan.disjoint_ranges > 1 ? 'explicit_disjoint_summary' : 'contiguous',
            },
        },
    });
}
const all: ContextSelector = { source: { kind: 'all' } };
const entry = (id: string) => ({ kind: 'entry' as const, id });
const turn = (id: string) => ({ kind: 'turn' as const, id });

describe('materialized context selector', () => {
    it('resolves inclusive turn/entry ranges and disjoint ranges in document order', async () => {
        const document = source();
        const plan = await select(document, {
            source: {
                kind: 'ranges',
                ranges: [
                    { from: entry('third-entry'), through: turn('third') },
                    { from: turn('first'), through: entry('first-entry') },
                ],
            },
        });
        expect(plan.entry_ids).toEqual(['first-entry', 'third-entry']);
        expect(plan.disjoint_ranges).toBe(2);
        const ranged = await select(document, {
            source: { kind: 'range', range: { from: turn('first'), through: turn('second') } },
        });
        expect(ranged.entry_ids).toEqual(['first-entry', 'second-entry']);
        const direct = await select(document, { source: { kind: 'turn_ids', turn_ids: ['third', 'first'] } });
        expect(direct.entry_ids).toEqual(plan.entry_ids);
    });
    it('rejects unknown/reversed/overlapping and ambiguous split-turn anchors rather than repairing them', async () => {
        const document = source();
        for (const selector of [
            { source: { kind: 'turn_ids', turn_ids: ['missing'] } },
            { source: { kind: 'range', range: { from: entry('second-entry'), through: entry('first-entry') } } },
            {
                source: {
                    kind: 'ranges',
                    ranges: [
                        { from: turn('first'), through: turn('second') },
                        { from: turn('second'), through: turn('third') },
                    ],
                },
            },
        ] satisfies ContextSelector[])
            expect((await resolveContextSelection(document, query(document, selector))).kind).toBe('rejected');
        document.context.entries.splice(
            0,
            1,
            { id: 'first-a', type: 'source_turn', turn_id: 'first', block_ids: ['a'] },
            { id: 'first-bc', type: 'source_turn', turn_id: 'first', block_ids: ['b', 'c'] },
        );
        expect(
            (
                await resolveContextSelection(
                    document,
                    query(document, {
                        source: { kind: 'range', range: { from: turn('first'), through: turn('second') } },
                    }),
                )
            ).kind,
        ).toBe('rejected');
        expect((await select(document, { source: { kind: 'turn_ids', turn_ids: ['first'] } })).entry_ids).toEqual([
            'first-a',
            'first-bc',
        ]);
        expect(
            (
                await select(document, {
                    source: { kind: 'range', range: { from: entry('first-bc'), through: entry('first-bc') } },
                })
            ).source_block_ids,
        ).toEqual(['b', 'c']);
    });
    it('combines actor, whole-block and declarative own-metadata predicates', async () => {
        const document = source();
        const plan = await select(document, {
            ...all,
            filters: {
                actor_kinds: ['user'],
                actor_ids: ['alice'],
                block_types: ['text'],
                block_ids: ['b'],
                metadata: [
                    { op: 'equals', path: ['topic'], value: 'finance' },
                    { op: 'in', path: ['score'], values: [2, 3] },
                    { op: 'exists', path: ['nested', 'ready'], exists: true },
                    { op: 'exists', path: ['absent'], exists: false },
                ],
            },
        });
        expect(plan.selected_block_ids).toEqual({ 'first-entry': ['b'] });
        expect(plan.source_block_ids).toEqual(['b']);
        const result = await resolveContextSelection(
            document,
            query(document, { ...all, filters: { metadata: [{ op: 'equals', path: ['topic'], value: 'missing' }] } }),
        );
        expect(result.kind).toBe('no_match');
        expect(
            (await resolveContextSelection(document, query(document, { ...all, filters: { block_ids: ['missing'] } })))
                .kind,
        ).toBe('rejected');
        const invalid = structuredClone(document);
        invalid.turns[0].metadata = Object.create({ topic: 'finance' });
        expect((await resolveContextSelection(invalid, query(document, all))).kind).toBe('rejected');
        const malicious = query(document, all);
        Object.setPrototypeOf(malicious.selector, { filters: {} });
        expect((await resolveContextSelection(document, malicious)).kind).toBe('rejected');
    });
    it('matches tool names through actual call identity and reports dependency cuts', async () => {
        const document = source();
        const call = {
            ...userTurn('call'),
            kind: 'agent' as const,
            provenance: { type: 'received' as const },
            blocks: [toolCallBlock('call-block', 'call-id')],
        };
        const result = toolResultTurn('result', 'call-id');
        document.turns.push(call, result);
        document.context.entries.push(
            { id: 'call-entry', type: 'source_turn', turn_id: 'call' },
            { id: 'result-entry', type: 'source_turn', turn_id: 'result' },
        );
        const plan = await select(document, { ...all, filters: { tool_names: ['read'] } });
        expect(plan.entry_ids).toEqual(['call-entry', 'result-entry']);
        const cut = await resolveContextSelection(
            document,
            query(document, { ...all, filters: { block_types: ['tool_call'] } }),
        );
        expect(cut).toMatchObject({
            kind: 'rejected',
            diagnostics: [{ stage: 'semantic', code: 'SELECTION_RANGE_INVALID' }],
        });
        document.context.entries.pop();
        expect(
            (await resolveContextSelection(document, query(document, { ...all, filters: { tool_names: ['read'] } })))
                .kind,
        ).toBe('rejected');
    });
    it('rejects a same-turn block cut required by a retained protected replay dependency', async () => {
        const document = source();
        document.turns.push({
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
                    payload: { signed: true },
                    dependencies: { turn_ids: ['first'], block_ids: [], call_ids: [], request_ids: [] },
                },
            ],
        });
        document.context.entries.push({ id: 'replay-entry', type: 'source_turn', turn_id: 'replay' });
        const result = await resolveContextSelection(
            document,
            query(document, { ...all, filters: { block_ids: ['b'] } }),
        );
        expect(result).toMatchObject({
            kind: 'rejected',
            diagnostics: [{ message: 'Context change would orphan native replay dependency replay-block' }],
        });
    });
    it('owns both source and selector before asynchronous hashing and enforces exact revision pins', async () => {
        const document = source(),
            baseline = structuredClone(document),
            input = query(document, { ...all, filters: { block_ids: ['b'] } });
        const promise = resolveContextSelection(document, input);
        document.turns[0].blocks[1] = textBlock('b', 'changed');
        input.selector.filters = { block_ids: ['a'] };
        expect(await promise).toEqual(
            await resolveContextSelection(baseline, query(baseline, { ...all, filters: { block_ids: ['b'] } })),
        );
        expect(
            (await resolveContextSelection(baseline, { ...query(baseline, all), expected_context_revision: 1 })).kind,
        ).toBe('rejected');
    });
});

describe('partial context edits', () => {
    it('rejects unpaired source references in direct planner input rather than upgrading them to a whole-entry edit', async () => {
        const document = source();
        const malformed = {
            expected_revision: 0,
            expected_context_revision: 0,
            entry_ids: ['first-entry'],
            selected_entries: [document.context.entries[0]],
        };
        await expect(planContextChange(document, malformed)).rejects.toThrow();
        const partial = await select(document, { ...all, filters: { block_ids: ['b'] } });
        const input = request(document, partial);
        const pending = applyContextChange(document, input);
        if (input.selected_block_ids) input.selected_block_ids['first-entry'][0] = 'a';
        if (input.selected_entries) input.selected_entries[0].id = 'changed';
        const result = await pending;
        expect(result.change.operations[0].selected_block_ids).toEqual({ 'first-entry': ['b'] });
        expect(result.document.context.entries.slice(0, 2).map((item) => item.block_ids)).toEqual([['a'], ['c']]);
    });
    it('uses own-key maps for a partial entry mixed with whole constructor/toString entries', async () => {
        const document = source();
        document.context.entries[1].id = 'constructor';
        document.context.entries[2].id = 'toString';
        const plan = await select(document, { ...all, filters: { block_ids: ['b', 'second-text', 'third-text'] } });
        expect(plan.entry_ids).toEqual(['first-entry', 'constructor', 'toString']);
        expect(plan.selected_block_ids).toEqual({ 'first-entry': ['b'] });
        const input = request(document, plan);
        const result = await applyContextChange(document, input);
        expect(result.document.context.entries.map((item) => item.block_ids)).toEqual([['a'], ['c']]);
        expect((await applyContextChange(result.document, input)).applied).toBe(false);
    });
    it('rejects an own __proto__ partial-map key with the existing experimental diagnostic, never as a whole edit', async () => {
        const document = source();
        document.context.entries[0].id = '__proto__';
        const result = await resolveContextSelection(
            document,
            query(document, { ...all, filters: { block_ids: ['b'] } }),
        );
        expect(result).toMatchObject({
            kind: 'rejected',
            diagnostics: [{ stage: 'preflight', code: 'JSON_RESERVED_PROPERTY_KEY' }],
        });
        const direct = JSON.parse(
            JSON.stringify({
                expected_revision: 0,
                expected_context_revision: 0,
                entry_ids: ['__proto__'],
                selected_entries: [document.context.entries[0]],
            }),
        );
        direct.selected_block_ids = JSON.parse('{"__proto__":["b"]}');
        await expect(planContextChange(document, direct)).rejects.toMatchObject({
            diagnostics: [{ code: 'JSON_RESERVED_PROPERTY_KEY' }],
        });
        // The ID is legal as a string; only using it as a record key is unsupported in this revision.
        const whole = await select(document, { source: { kind: 'turn_ids', turn_ids: ['first'] } });
        expect(whole).not.toHaveProperty('selected_block_ids');
        expect(
            (await applyContextChange(document, request(document, whole))).document.context.entries.map(
                (item) => item.id,
            ),
        ).toEqual(['second-entry', 'third-entry']);
    });
    it('retains unmatched blocks in order with deterministic receipt-bound remainder IDs and exact retries', async () => {
        const document = source(),
            plan = await select(document, { ...all, filters: { block_ids: ['b'] } }),
            input = request(document, plan);
        const result = await applyContextChange(document, input);
        expect(result.document.turns).toEqual(document.turns);
        expect(result.document.context.entries.slice(0, 2).map((item) => item.block_ids)).toEqual([['a'], ['c']]);
        expect(result.change.operations[0].remainder_entry_ids).toEqual(
            result.document.context.entries.slice(0, 2).map((item) => item.id),
        );
        expect((await applyContextChange(JSON.parse(JSON.stringify(result.document)), input)).applied).toBe(false);
        const nextPlan = await select(result.document, { ...all, filters: { block_ids: ['a'] } });
        const later = await applyContextChange(result.document, request(result.document, nextPlan, 'later'));
        expect((await applyContextChange(later.document, input)).applied).toBe(false);
        const drift = structuredClone(result.document);
        drift.operation_receipts.edit.context_change!.remainder_entry_ids!.reverse();
        await expect(applyContextChange(drift, input)).rejects.toThrow('conflicting retained details');
        const changedRefs = structuredClone(input);
        if (changedRefs.selected_entries) changedRefs.selected_entries[0].block_ids = ['b'];
        await expect(applyContextChange(document, changedRefs)).rejects.toThrow('source entries conflict');
    });
    it('rejects deterministic remainder ID collisions without modifying the caller source', async () => {
        const document = source();
        document.context.entries[1].id = await deriveConversationId('context_entry', 'edit', 'remainder', '0');
        const owned = structuredClone(document);
        const plan = await select(document, { ...all, filters: { block_ids: ['b'] } });
        await expect(applyContextChange(document, request(document, plan))).rejects.toMatchObject({
            diagnostics: expect.arrayContaining([expect.objectContaining({ code: 'DUPLICATE_ID' })]),
        });
        expect(document).toEqual(owned);
    });
    it('places summary at first selected block while preserving surrounding and disjoint unmatched blocks', async () => {
        const document = source(),
            plan = await select(document, { ...all, filters: { block_ids: ['b', 'third-text'] } });
        const input = compactionRequest(document, plan, 'summary');
        const result = await applyContextChange(document, input);
        expect(
            result.document.context.entries.map((item) =>
                item.type === 'replacement_turn' ? 'summary' : (item.block_ids ?? item.turn_id),
            ),
        ).toEqual([['a'], 'summary', ['c'], 'second']);
        expect(result.document.compactions.summary.source).toMatchObject({
            turn_ids: ['first', 'third'],
            block_ids: ['b', 'third-text'],
        });
        expect(result.document.operation_receipts).toMatchObject(document.operation_receipts);
        expect((await applyContextChange(result.document, input)).applied).toBe(false);
    });
    it('retries a partial summary after its replacement is consumed by a later compaction', async () => {
        const document = source();
        const plan = await select(document, { ...all, filters: { block_ids: ['b'] } });
        const input = compactionRequest(document, plan, 'partial-summary');
        const first = await applyContextChange(document, input);
        const next = await select(first.document, { source: { kind: 'turn_ids', turn_ids: ['partial-summary-turn'] } });
        const later = await applyContextChange(
            first.document,
            compactionRequest(first.document, next, 'later-summary'),
        );
        const retry = await applyContextChange(JSON.parse(JSON.stringify(later.document)), input);
        expect(retry.applied).toBe(false);
        expect(retry.change).toEqual(first.change);
        expect(retry.document.compactions['partial-summary']).toEqual(first.document.compactions['partial-summary']);
        expect(retry.document.turns).toEqual(document.turns);
    });
    it('keeps historical whole-entry payload identity and denies protected/cache-required cuts', async () => {
        const document = source(),
            plan = await select(document, { source: { kind: 'turn_ids', turn_ids: ['second'] } }),
            input = request(document, plan);
        const result = await applyContextChange(document, input);
        expect(result.document.operation_receipts.edit.payload_fingerprint).toBe(await fingerprintJson(input));
        expect(result.change.operations[0]).not.toHaveProperty('selected_block_ids');
        expect(result.change.operations[0]).not.toHaveProperty('remainder_entry_ids');
        const protectedDoc = source();
        protectedDoc.context.protected_entry_ids = ['first-entry'];
        expect(
            (
                await resolveContextSelection(
                    protectedDoc,
                    query(protectedDoc, { ...all, filters: { block_ids: ['b'] } }),
                )
            ).kind,
        ).toBe('rejected');
        for (const mode of ['required', 'auto', 'off'] as const) {
            const cached = source();
            cached.context.cache_intent = { namespace: 'cache', mode, stable_through_entry_id: 'first-entry' };
            const partial = await select(cached, { ...all, filters: { block_ids: ['b'] } });
            if (mode === 'required')
                await expect(applyContextChange(cached, request(cached, partial))).rejects.toThrow('required cache');
            else
                expect(
                    (await applyContextChange(cached, request(cached, partial))).document.context.cache_intent,
                ).toEqual({ namespace: 'cache', mode });
        }
    });
    it('rejects partial replacement cuts with indivisible provenance; retains every ancestor of a multi-summary compaction', async () => {
        const document = source();
        let first = await applyContextChange(
            document,
            compactionRequest(
                document,
                await select(document, { source: { kind: 'turn_ids', turn_ids: ['first'] } }),
                'summary1',
            ),
        );
        const secondPlan = await select(first.document, { source: { kind: 'turn_ids', turn_ids: ['third'] } });
        first = await applyContextChange(first.document, compactionRequest(first.document, secondPlan, 'summary2'));
        const combinedPlan = await select(first.document, {
            source: { kind: 'turn_ids', turn_ids: ['summary1-turn', 'summary2-turn'] },
        });
        const combined = await applyContextChange(
            first.document,
            compactionRequest(first.document, combinedPlan, 'combined'),
        );
        expect(combined.document.compactions.combined).not.toHaveProperty('supersedes_compaction_id');
        expect(combined.document.compactions.combined.source.turn_ids).toEqual(['summary1-turn', 'summary2-turn']);
        expect(combined.document.compactions.summary1).toEqual(first.document.compactions.summary1);
        expect(combined.document.compactions.summary2).toEqual(first.document.compactions.summary2);
        const singlePlan = await select(combined.document, {
            source: { kind: 'turn_ids', turn_ids: ['combined-turn'] },
        });
        const single = await applyContextChange(
            combined.document,
            compactionRequest(combined.document, singlePlan, 'successor'),
        );
        expect(single.document.compactions.successor.supersedes_compaction_id).toBe('combined');
        const multi = source();
        const mp = await select(multi, { source: { kind: 'turn_ids', turn_ids: ['first'] } });
        const mr = compactionRequest(multi, mp, 'multi');
        if (mr.proposal.kind === 'replace_with_compaction') {
            const replacement = mr.proposal.replacement_turns[0];
            if (replacement.kind !== 'agent') throw new Error('Fixture must be an agent summary');
            replacement.blocks = [...replacement.blocks, textBlock('extra')];
        }
        const compacted = await applyContextChange(multi, mr);
        const partial = await resolveContextSelection(
            compacted.document,
            query(compacted.document, { ...all, filters: { block_ids: ['extra'] } }),
        );
        expect(partial).toMatchObject({
            kind: 'rejected',
            diagnostics: [{ message: 'Compaction replacement is an indivisible provenance unit' }],
        });
    });
});

it('exports inferred selector types and schema-derived JSON with identical strict shape rules', () => {
    expectTypeOf<ContextSelector>().toEqualTypeOf<z.infer<typeof ContextSelectorSchema>>();
    expectTypeOf<ContextSelectionRequest>().toEqualTypeOf<z.infer<typeof ContextSelectionRequestSchema>>();
    expectTypeOf<ContextSelectionResult>().toEqualTypeOf<z.infer<typeof ContextSelectionResultSchema>>();
    const ajv = new Ajv2020({ strict: false });
    formats.default(ajv);
    const q = query(source(), all);
    const partial = {
        operation_id: 'edit',
        expected_revision: 0,
        expected_context_revision: 0,
        expected_source_fingerprint: 'sha256:x',
        entry_ids: ['first-entry'],
        recorded_at: '2026-10-02T00:00:00.000Z',
        proposal: { kind: 'exclude' },
        selected_block_ids: { 'first-entry': ['b'] },
        selected_entries: [{ id: 'first-entry', type: 'source_turn', turn_id: 'first' }],
    };
    const { selected_block_ids: _blocks, ...unpairedEntries } = partial;
    const { selected_entries: _entries, ...unpairedBlocks } = partial;
    for (const [schema, json, values] of [
        [
            ContextSelectorSchema,
            ContextSelectorJsonSchema,
            [
                all,
                { ...all, extra: true },
                { source: { kind: 'ranges', ranges: [] } },
                { ...all, filters: { metadata: [{ op: 'callback', path: ['x'] }] } },
            ],
        ],
        [
            ContextSelectionRequestSchema,
            ContextSelectionRequestJsonSchema,
            [q, { ...q, conversation: { ...q.conversation, revision: -1 } }],
        ],
        [
            ContextSelectionResultSchema,
            ContextSelectionResultJsonSchema,
            [
                {
                    kind: 'no_match',
                    conversation: q.conversation,
                    context_revision: 0,
                    source_fingerprint: 'sha256:test',
                    diagnostics: [],
                },
                { kind: 'rejected', diagnostics: [] },
            ],
        ],
        [
            ContextChangeRequestSchema,
            ContextChangeRequestJsonSchema,
            [
                partial,
                unpairedEntries,
                unpairedBlocks,
                { ...partial, selected_entries: [{ ...partial.selected_entries[0], metadata: { arbitrary: true } }] },
            ],
        ],
    ] as const) {
        const validate = ajv.compile(json);
        if (schema === ContextChangeRequestSchema) {
            expect(validate(partial)).toBe(true);
            expect(validate(unpairedEntries)).toBe(false);
            expect(validate(unpairedBlocks)).toBe(false);
        }
        for (const value of values) expect(validate(value)).toBe(schema.safeParse(value).success);
    }
    expect(CONVERSATION_JSON_SCHEMAS.context_selector).toBe(ContextSelectorJsonSchema);
    expect(CONVERSATION_JSON_SCHEMAS.context_selection_request).toBe(ContextSelectionRequestJsonSchema);
    expect(CONVERSATION_JSON_SCHEMAS.context_selection_result).toBe(ContextSelectionResultJsonSchema);
});
