import { Ajv2020 } from 'ajv/dist/2020.js';
import formatsPlugin from 'ajv-formats';
import { describe, expect, expectTypeOf, it } from 'vitest';
import type { z } from 'zod';
import * as core from '../src/index.js';
import * as json from '../src/json-schema.js';
import { emptyDocument, userTurn } from './fixtures.js';

function parity(schema: z.ZodType, generated: Readonly<Record<string, unknown>>, fixtures: [unknown, boolean][]) {
    const ajv = new Ajv2020({ allErrors: true, strict: true });
    formatsPlugin.default(ajv);
    const validate = ajv.compile(generated);
    for (const [fixture, success] of fixtures) {
        expect(schema.safeParse(fixture).success).toBe(success);
        expect(validate(fixture), JSON.stringify(validate.errors)).toBe(success);
    }
}

describe('canonical edit schema and export parity', () => {
    it('exports each authoritative schema, inferred alias and discoverable JSON schema', () => {
        const pairs = [
            ['accepted_tool_selection', 'AcceptedToolSelection'],
            ['conversation_edit_record_ref', 'ConversationEditRecordRef'],
            ['conversation_edit_anchor', 'ConversationEditAnchor'],
            ['conversation_edit_operation', 'ConversationEditOperation'],
            ['conversation_edit_placement', 'ConversationEditPlacement'],
            ['conversation_editable_block', 'ConversationEditableBlock'],
            ['conversation_inserted_turn', 'ConversationInsertedTurn'],
            ['conversation_replacement_turn', 'ConversationReplacementTurn'],
            ['conversation_edit_command', 'ConversationEditCommand'],
            ['conversation_edit_plan_input', 'ConversationEditPlanInput'],
            ['conversation_edit_request', 'ConversationEditRequest'],
            ['conversation_edit_plan', 'ConversationEditPlan'],
            ['conversation_edit_result', 'ConversationEditResult'],
            ['context_change', 'ContextChange'],
            ['conversation_append_operation', 'ConversationAppendOperation'],
            ['conversation_append_change', 'ConversationAppendChange'],
            ['conversation_edit_change', 'ConversationEditChange'],
        ] as const;
        for (const [key, name] of pairs) {
            expect(core[`${name}Schema`].safeParse).toBeTypeOf('function');
            const generated = json[`${name}JsonSchema`];
            expect(generated.$schema).toBe('https://json-schema.org/draft/2020-12/schema');
            expect(Object.isFrozen(generated)).toBe(true);
            expect(json.CONVERSATION_JSON_SCHEMAS[key]).toBe(generated);
        }
        expectTypeOf<core.ConversationEditRequest>().toEqualTypeOf<
            z.infer<typeof core.ConversationEditRequestSchema>
        >();
        expectTypeOf<core.ConversationEditResult>().toEqualTypeOf<z.infer<typeof core.ConversationEditResultSchema>>();
        expectTypeOf<core.ContextChange>().toEqualTypeOf<z.infer<typeof core.ContextChangeSchema>>();
        expectTypeOf<core.ConversationAppendChange>().toEqualTypeOf<
            z.infer<typeof core.ConversationAppendChangeSchema>
        >();
        expectTypeOf<core.ConversationChange>().toEqualTypeOf<z.infer<typeof core.ConversationChangeSchema>>();
    });

    it('enforces strict received content, bounded revisions, exact request structure and named result families', async () => {
        const document = emptyDocument();
        document.turns.push(userTurn('source'));
        document.context.entries.push({ id: 'source-entry', type: 'source_turn', turn_id: 'source' });
        const at = '2026-10-02T00:00:00.000Z';
        const turn = core.ConversationInsertedTurnSchema.parse({
            ...userTurn('new'),
            timestamps: { recorded_at: at },
            provenance: { type: 'inserted', operation_id: 'insert' },
        });
        const input = {
            version: 1 as const,
            operation_id: 'insert',
            conversation: { conversation_id: document.id, revision: 0 },
            expected_context_revision: 0,
            recorded_at: at,
            command: { kind: 'insert' as const, anchor: { kind: 'tail' as const }, turns: [turn] },
        };
        const plan = await core.planConversationEdit(document, input);
        const request = { ...input, expected_source_fingerprint: plan.operation.source_fingerprint };
        const result = await core.applyConversationEdit(document, request);
        parity(core.ConversationEditRequestSchema, json.ConversationEditRequestJsonSchema, [
            [request, true],
            [{ ...request, version: 2 }, false],
            [{ ...request, expected_source_fingerprint: undefined }, false],
            [{ ...request, expected_context_revision: Number.MAX_SAFE_INTEGER + 1 }, false],
            [{ ...request, command: { ...input.command, proof: {} } }, false],
            [{ ...request, command: { ...input.command, turns: [] } }, false],
        ]);
        parity(core.AcceptedToolSelectionSchema, json.AcceptedToolSelectionJsonSchema, [
            [{ kind: 'unchanged' }, true],
            [{ kind: 'replace', definition_ids: [] }, true],
            [{ kind: 'unchanged', definition_ids: [] }, false],
            [{ kind: 'replace' }, false],
        ]);
        parity(core.ConversationInsertedTurnSchema, json.ConversationInsertedTurnJsonSchema, [
            [turn, true],
            [{ ...turn, kind: 'agent' }, true],
            [{ ...turn, authority: 'system' }, false],
            [{ ...turn, generation_id: 'forged' }, false],
            [{ ...turn, blocks: [{ id: 'json', type: 'json', value: null }] }, true],
            [{ ...turn, blocks: [{ id: 'json', type: 'json' }] }, false],
            [
                {
                    ...turn,
                    blocks: [
                        {
                            id: 'call',
                            type: 'tool_call',
                            call_id: 'call',
                            tool_name: 'read',
                            executor: 'application',
                            arguments: {},
                        },
                    ],
                },
                false,
            ],
        ]);
        parity(core.ConversationEditAnchorSchema, json.ConversationEditAnchorJsonSchema, [
            [{ kind: 'head' }, true],
            [{ kind: 'tail' }, true],
            [{ kind: 'before_entry', entry_id: 'source-entry' }, true],
            [{ kind: 'after_entry' }, false],
            [{ kind: 'tail', entry_id: 'ignored' }, false],
        ]);
        parity(core.ConversationEditOperationSchema, json.ConversationEditOperationJsonSchema, [
            [plan.operation, true],
            [{ ...plan.operation, source_fingerprint: undefined }, false],
            [{ ...plan.operation, created_entries: [{ id: 'new', text: 'invented' }] }, false],
        ]);
        const { anchor: _anchor, ...operation } = core.ConversationEditOperationV1Schema.options[1].parse(
            plan.operation,
        );
        const replacementOperation = {
            ...operation,
            kind: 'replace',
            fidelity: 'semantic',
            placement: { mode: 'first_selected', causal_order: 'contiguous' },
        };
        const { fidelity: _fidelity, ...missingFidelity } = replacementOperation;
        const { placement: _placement, ...missingPlacement } = replacementOperation;
        parity(core.ConversationEditOperationSchema, json.ConversationEditOperationJsonSchema, [
            [replacementOperation, true],
            [{ ...replacementOperation, fidelity: 'heuristic' }, true],
            [missingFidelity, false],
            [missingPlacement, false],
            [{ ...replacementOperation, fidelity: 'retrievable' }, false],
            [{ ...replacementOperation, placement: { mode: 'tail', causal_order: 'contiguous' } }, false],
            [
                { ...replacementOperation, placement: { mode: 'first_selected', causal_order: 'explicit_disjoint' } },
                true,
            ],
            [{ ...plan.operation, fidelity: 'semantic' }, false],
        ]);
        parity(core.ConversationEditPlacementSchema, json.ConversationEditPlacementJsonSchema, [
            [replacementOperation.placement, true],
            [{ mode: 'first_selected', causal_order: 'explicit_disjoint' }, true],
            [{ mode: 'first_selected' }, false],
            [{ ...replacementOperation.placement, unclaimed: true }, false],
        ]);
        parity(core.ConversationEditResultSchema, json.ConversationEditResultJsonSchema, [
            [result, true],
            [{ ...result, change: undefined }, false],
        ]);
        parity(core.ConversationEditChangeSchema, json.ConversationEditChangeJsonSchema, [
            [result.change, true],
            [{ ...result.change, operations: [] }, false],
        ]);
        parity(core.ConversationChangeSchema, json.ConversationChangeJsonSchema, [[result.change, true]]);
        const append = core.appendConversationRecords(
            document,
            {},
            { expected_revision: 0, operation_id: 'append', payload_fingerprint: 'sha256:append', recorded_at: at },
        );
        parity(core.ConversationAppendChangeSchema, json.ConversationAppendChangeJsonSchema, [
            [append.change, true],
            [result.change, false],
        ]);
        parity(core.ContextChangeSchema, json.ContextChangeJsonSchema, [
            [result.change, false],
            [
                {
                    operation_id: 'old',
                    conversation_id: document.id,
                    base_revision: 0,
                    result_revision: 1,
                    operations: [
                        {
                            kind: 'exclude',
                            removed_entry_ids: ['source-entry'],
                            inserted_entry_ids: [],
                            source_fingerprint: 'sha256:old',
                        },
                    ],
                    diagnostics: [],
                },
                true,
            ],
        ]);
        expect(json.ConversationDocumentJsonSchema.$defs).toHaveProperty('ConversationEditOperation');
        expect(json.ConversationDocumentJsonSchema.$defs).toHaveProperty('ConversationContextEntry');
    });
});
