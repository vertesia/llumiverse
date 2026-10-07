import { Ajv2020 } from 'ajv/dist/2020.js';
import formatsPlugin from 'ajv-formats';
import { describe, expect, expectTypeOf, it } from 'vitest';
import type { z } from 'zod';
import {
    CONVERSATION_EXPERIMENTAL_REVISION,
    type ContextChangeOperation,
    ContextChangeOperationSchema,
    type ContextChangePlacement,
    ContextChangePlacementSchema,
    type ContextChangeProposal,
    ContextChangeProposalSchema,
    type ContextChangeRequest,
    ContextChangeRequestSchema,
    type ConversationChange,
    ConversationChangeSchema,
} from '../src/index.js';
import {
    CONVERSATION_JSON_SCHEMAS,
    ContextChangeOperationJsonSchema,
    ContextChangePlacementJsonSchema,
    ContextChangeProposalJsonSchema,
    ContextChangeRequestJsonSchema,
    ConversationChangeJsonSchema,
    ConversationDocumentJsonSchema,
} from '../src/json-schema.js';

const placement: ContextChangePlacement = { mode: 'first_selected', causal_order: 'contiguous' };
const operation: ContextChangeOperation = {
    kind: 'exclude',
    removed_entry_ids: ['entry:first'],
    inserted_entry_ids: [],
    source_fingerprint: 'sha256:source',
};
const proposal: ContextChangeProposal = {
    kind: 'replace_with_compaction',
    compaction_id: 'compaction:first',
    strategy: { id: 'summary', version: '1', configuration_fingerprint: 'sha256:config' },
    replacement_turns: [
        {
            id: 'turn:summary',
            kind: 'agent',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: '2026-10-01T00:00:00.000Z' },
            model_visibility: 'include',
            provenance: {
                type: 'derived',
                derivation_id: 'compaction:first',
                source_turn_ids: ['turn:first'],
                source_hash: 'sha256:source',
            },
            blocks: [{ id: 'block:summary', type: 'text', text: 'Summary', format: 'plain' }],
        },
    ],
    fidelity: 'semantic',
    retained_asset_ids: [],
    generation_ids: [],
    placement,
};
const request: ContextChangeRequest = {
    operation_id: 'change:first',
    expected_revision: 0,
    expected_context_revision: 0,
    expected_source_fingerprint: 'sha256:source',
    recorded_at: '2026-10-01T00:00:00.000Z',
    entry_ids: ['entry:first'],
    proposal: { kind: 'exclude' },
};
const change: ConversationChange = {
    operation_id: 'change:first',
    conversation_id: 'conversation:first',
    base_revision: 0,
    result_revision: 1,
    operations: [operation],
    diagnostics: [],
};

function expectParity(
    schema: z.ZodType,
    jsonSchema: Readonly<Record<string, unknown>>,
    fixtures: [unknown, boolean][],
) {
    const ajv = new Ajv2020({ allErrors: true, strict: true });
    formatsPlugin.default(ajv);
    const validate = ajv.compile(jsonSchema);
    for (const [fixture, accepted] of fixtures) {
        expect(schema.safeParse(fixture).success).toBe(accepted);
        expect(validate(fixture), JSON.stringify(validate.errors)).toBe(accepted);
    }
}

function assertFrozen(value: unknown): void {
    if (value === null || typeof value !== 'object') return;
    expect(Object.isFrozen(value)).toBe(true);
    for (const item of Object.values(value)) assertFrozen(item);
}

describe('context-change JSON Schema contract', () => {
    it('exports revision-scoped, deeply frozen schemas and named receipt dependencies', () => {
        for (const [schema, suffix] of [
            [ContextChangeRequestJsonSchema, 'context-change-request'],
            [ContextChangeProposalJsonSchema, 'context-change-proposal'],
            [ContextChangeOperationJsonSchema, 'context-change-operation'],
            [ContextChangePlacementJsonSchema, 'context-change-placement'],
            [ConversationChangeJsonSchema, 'change'],
        ] as const) {
            expect(schema.$schema).toBe('https://json-schema.org/draft/2020-12/schema');
            expect(schema.$id).toBe(`urn:llumiverse:conversation:${CONVERSATION_EXPERIMENTAL_REVISION}:${suffix}`);
            assertFrozen(schema);
        }
        expect(Object.isFrozen(CONVERSATION_JSON_SCHEMAS)).toBe(true);
        for (const [name, schema] of [
            ['context_change_request', ContextChangeRequestJsonSchema],
            ['context_change_proposal', ContextChangeProposalJsonSchema],
            ['context_change_operation', ContextChangeOperationJsonSchema],
            ['context_change_placement', ContextChangePlacementJsonSchema],
            ['change', ConversationChangeJsonSchema],
        ] as const) {
            expect(CONVERSATION_JSON_SCHEMAS[name]).toBe(schema);
        }
        expect(ConversationDocumentJsonSchema.$defs).toHaveProperty('ConversationContextChangeOperation');
        expect(ConversationDocumentJsonSchema.$defs).toHaveProperty('ConversationContextChangePlacement');
    });

    it('keeps request branches, required source identity, timestamp and revision bounds in parity', () => {
        const { expected_source_fingerprint: _source, ...missingSource } = request;
        expectParity(ContextChangeRequestSchema, ContextChangeRequestJsonSchema, [
            [request, true],
            [{ ...request, proposal }, true],
            [missingSource, false],
            [{ ...request, entry_ids: [] }, false],
            [{ ...request, expected_revision: -1 }, false],
            [{ ...request, expected_context_revision: Number.MAX_SAFE_INTEGER + 1 }, false],
            [{ ...request, recorded_at: 'yesterday' }, false],
            [{ ...request, unknown_field: true }, false],
            [{ ...request, proposal: { kind: 'delete' } }, false],
        ]);
    });

    it('keeps nested replacement and explicit disjoint placement shape in parity', () => {
        expectParity(ContextChangeProposalSchema, ContextChangeProposalJsonSchema, [
            [{ kind: 'exclude' }, true],
            [proposal, true],
            [{ ...proposal, fidelity: 'heuristic' }, true],
            [{ ...proposal, fidelity: 'value_preserving' }, false],
            [{ ...proposal, fidelity: 'reversible_representation' }, false],
            [{ ...proposal, fidelity: 'retrievable' }, true],
            [{ ...proposal, placement: { ...placement, causal_order: 'explicit_disjoint_summary' } }, true],
            [
                { ...proposal, placement: { mode: 'per_selected_range', causal_order: 'preserved_disjoint_ranges' } },
                true,
            ],
            [{ kind: 'exclude', replacement_turns: [] }, false],
            [{ ...proposal, replacement_turns: [] }, false],
            // Range-to-turn identity and uniqueness are checked by the accepted context edit, not this shape schema.
            [{ ...proposal, replacement_turns: [proposal.replacement_turns[0], proposal.replacement_turns[0]] }, true],
            [{ ...proposal, placement: { mode: 'tail', causal_order: 'contiguous' } }, false],
            [
                {
                    ...proposal,
                    replacement_turns: [
                        { ...proposal.replacement_turns[0], blocks: [{ id: 'block:json', type: 'json' }] },
                    ],
                },
                false,
            ],
        ]);
        expectParity(ContextChangePlacementSchema, ContextChangePlacementJsonSchema, [
            [placement, true],
            [{ ...placement, unknown_field: true }, false],
            [{ ...placement, causal_order: 'implicit' }, false],
        ]);
    });

    it('keeps change and receipt detail shape in parity without claiming graph validation', () => {
        expectParity(ContextChangeOperationSchema, ContextChangeOperationJsonSchema, [
            [operation, true],
            [{ ...operation, removed_entry_ids: [] }, false],
            [{ ...operation, inserted_entry_ids: [null] }, false],
            [{ ...operation, unknown_field: true }, false],
        ]);
        expectParity(ConversationChangeSchema, ConversationChangeJsonSchema, [
            [change, true],
            [{ ...change, operations: [] }, false],
            [{ ...change, operations: [operation, operation] }, false],
            [{ ...change, result_revision: -1 }, false],
            [{ ...change, diagnostics: [{}] }, false],
        ]);
        expectTypeOf<ContextChangeRequest>().toEqualTypeOf<z.infer<typeof ContextChangeRequestSchema>>();
        expectTypeOf<ContextChangeProposal>().toEqualTypeOf<z.infer<typeof ContextChangeProposalSchema>>();
        expectTypeOf<ContextChangeOperation>().toEqualTypeOf<z.infer<typeof ContextChangeOperationSchema>>();
        expectTypeOf<ContextChangePlacement>().toEqualTypeOf<z.infer<typeof ContextChangePlacementSchema>>();
        expectTypeOf<ConversationChange>().toEqualTypeOf<z.infer<typeof ConversationChangeSchema>>();
    });
});
