import { describe, expect, it } from 'vitest';
import {
    applyContextChange,
    type ContextChangeRequest,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    parseConversationDocument,
    planContextChange,
} from '../src/index.js';

const at = '2026-10-01T00:00:00.000Z';
function source() {
    const document = createConversationDocument({ id: 'conversation:fidelity', created_at: at });
    for (const name of ['old', 'recent']) {
        document.turns.push(
            createUserTurn({
                id: `turn:${name}`,
                authority: 'ordinary',
                status: 'completed',
                timestamps: { recorded_at: at },
                provenance: { type: 'received' },
                model_visibility: 'include',
                blocks: [createTextBlock({ id: `block:${name}`, text: name, format: 'plain' })],
            }),
        );
        document.context.entries.push({ id: `entry:${name}`, type: 'source_turn', turn_id: `turn:${name}` });
    }
    return document;
}

async function replacement(fidelity: 'heuristic' | 'semantic') {
    const document = source();
    const plan = await planContextChange(document, {
        expected_revision: 0,
        expected_context_revision: 0,
        entry_ids: ['entry:old'],
    });
    const request: ContextChangeRequest = {
        operation_id: `operation:${fidelity}`,
        expected_revision: 0,
        expected_context_revision: 0,
        expected_source_fingerprint: plan.source_fingerprint,
        recorded_at: at,
        entry_ids: plan.entry_ids,
        proposal: {
            kind: 'replace_with_compaction',
            compaction_id: `compaction:${fidelity}`,
            strategy: { id: 'caller_summary', version: '1', configuration_fingerprint: 'sha256:config' },
            replacement_turns: [
                {
                    id: `turn:${fidelity}`,
                    kind: 'agent',
                    authority: 'ordinary',
                    status: 'completed',
                    timestamps: { recorded_at: at },
                    model_visibility: 'include',
                    provenance: {
                        type: 'derived',
                        derivation_id: `compaction:${fidelity}`,
                        source_turn_ids: plan.source_turn_ids,
                        source_hash: plan.source_fingerprint,
                    },
                    blocks: [createTextBlock({ id: `block:${fidelity}`, text: 'Caller summary', format: 'plain' })],
                },
            ],
            fidelity,
            retained_asset_ids: [],
            generation_ids: [],
            placement: { mode: 'first_selected', causal_order: 'contiguous' },
        },
    };
    return { document, request };
}

describe('bounded text context-change fidelity and disabled cache', () => {
    it.each(['heuristic', 'semantic'] as const)(
        'accepts a new %s summary without rewriting retained source',
        async (fidelity) => {
            const { document, request } = await replacement(fidelity);
            const applied = await applyContextChange(document, request);
            expect(applied.document.compactions[`compaction:${fidelity}`].fidelity).toBe(fidelity);
            expect(applied.document.turns).toEqual(document.turns);
            expect(parseConversationDocument(JSON.parse(JSON.stringify(applied.document)))).toEqual(applied.document);
        },
    );

    it.each(['value_preserving', 'reversible_representation', 'retrievable'] as const)(
        'rejects new unverified %s proposals while retaining readable historical classification',
        async (fidelity) => {
            const { document, request } = await replacement('semantic');
            Reflect.set(request.proposal, 'fidelity', fidelity);
            if (fidelity === 'retrievable') {
                await expect(applyContextChange(document, request)).rejects.toThrow(
                    'Retrievable compaction replacement must be completed ordinary external content',
                );
            } else {
                await expect(applyContextChange(document, request)).rejects.toMatchObject({
                    issues: expect.arrayContaining([
                        expect.objectContaining({
                            code: 'invalid_value',
                            path: ['proposal', 'fidelity'],
                            values: ['heuristic', 'semantic', 'retrievable'],
                        }),
                    ]),
                });
            }
            const historical = await replacement('semantic');
            const accepted = await applyContextChange(historical.document, historical.request);
            accepted.document.compactions['compaction:semantic'].fidelity = fidelity;
            expect(
                parseConversationDocument(JSON.parse(JSON.stringify(accepted.document))).compactions[
                    'compaction:semantic'
                ].fidelity,
            ).toBe(fidelity);
        },
    );

    it.each(['entry:old', 'entry:recent'] as const)(
        'preserves disabled cache and only drops its boundary if selected (%s)',
        async (boundary) => {
            const document = source();
            document.context.cache_intent = {
                namespace: 'disabled-cache',
                mode: 'off',
                stable_through_entry_id: boundary,
                ttl_seconds: 300,
            };
            const parsed = parseConversationDocument(document);
            const plan = await planContextChange(parsed, {
                expected_revision: 0,
                expected_context_revision: 0,
                entry_ids: ['entry:old'],
            });
            const applied = await applyContextChange(parsed, {
                operation_id: `operation:off:${boundary}`,
                expected_revision: 0,
                expected_context_revision: 0,
                expected_source_fingerprint: plan.source_fingerprint,
                recorded_at: at,
                entry_ids: plan.entry_ids,
                proposal: { kind: 'exclude' },
            });
            expect(applied.document.context.cache_intent).toEqual({
                namespace: 'disabled-cache',
                mode: 'off',
                ttl_seconds: 300,
                ...(boundary === 'entry:recent' ? { stable_through_entry_id: boundary } : {}),
            });
            expect(applied.document.context.entries.map((entry) => entry.id)).toEqual(['entry:recent']);
            expect(parseConversationDocument(JSON.parse(JSON.stringify(applied.document)))).toEqual(applied.document);
            expect(applied.document.turns).toEqual(parsed.turns);
        },
    );
});
