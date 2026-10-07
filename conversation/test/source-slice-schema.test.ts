import { Ajv2020 } from 'ajv/dist/2020.js';
import formatsPlugin from 'ajv-formats';
import { describe, expect, expectTypeOf, it } from 'vitest';
import type { z } from 'zod';
import * as core from '../src/index.js';
import * as json from '../src/json-schema.js';

function parity(schema: z.ZodType, generated: Readonly<Record<string, unknown>>, fixtures: [unknown, boolean][]) {
    const ajv = new Ajv2020({ strict: true, allErrors: true });
    formatsPlugin.default(ajv);
    const validate = ajv.compile(generated);
    for (const [fixture, success] of fixtures) {
        expect(schema.safeParse(fixture).success).toBe(success);
        expect(validate(fixture), JSON.stringify(validate.errors)).toBe(success);
    }
}
const source = {
    source: { conversation_id: 'doc', revision: 1 },
    turn_id: 'original',
    block_id: 'block',
    block_fingerprint: 'hash:original',
};
const textSlice = { ...source, selection: { kind: 'text_range', range: { start_code_point: 1, end_code_point: 3 } } };
const operation = {
    version: 2,
    kind: 'protect',
    source: source.source,
    source_context_revision: 0,
    source_fingerprint: 'hash:source',
    selected_entries: [{ id: 'entry', fingerprint: 'entry' }],
    removed_entry_ids: [],
    created_entries: [],
    created_turns: [],
    created_assets: [],
    protected_entry_ids: [],
    unprotected_entry_ids: [],
    source_slices: [textSlice],
    remainder_turn_ids: [],
    source_protected_entry_ids: [],
    source_entry_positions: [0],
    source_topology_fingerprint: 'topology',
};

describe('precise source-slice published schema parity', () => {
    it('keeps inferred types/entry exports and JSON inventory discoverable without handwritten copies', () => {
        expectTypeOf<core.SourceBlockSlice>().toEqualTypeOf<z.infer<typeof core.SourceBlockSliceSchema>>();
        expectTypeOf<core.DerivedBlockLineage>().toEqualTypeOf<z.infer<typeof core.DerivedBlockLineageSchema>>();
        expectTypeOf<core.ConversationSliceEditRequest>().toEqualTypeOf<
            z.infer<typeof core.ConversationSliceEditRequestSchema>
        >();
        expectTypeOf<core.ConversationSliceEditOperation>().toEqualTypeOf<
            z.infer<typeof core.ConversationSliceEditOperationSchema>
        >();
        expectTypeOf<core.ConversationEditOperationV1>().toEqualTypeOf<
            z.infer<typeof core.ConversationEditOperationV1Schema>
        >();
        expectTypeOf<core.DerivedLineageVerificationScope>().toEqualTypeOf<
            z.infer<typeof core.DerivedLineageVerificationScopeSchema>
        >();
        const pairs = [
            ['conversation_edit_operation_v1', 'ConversationEditOperationV1'],
            ['conversation_slice_edit_operation', 'ConversationSliceEditOperation'],
            ['conversation_slice_edit_command', 'ConversationSliceEditCommand'],
            ['conversation_slice_edit_plan_input', 'ConversationSliceEditPlanInput'],
            ['conversation_slice_edit_request', 'ConversationSliceEditRequest'],
            ['source_block_slice', 'SourceBlockSlice'],
            ['derived_block_lineage', 'DerivedBlockLineage'],
            ['derived_block_lineage_group', 'DerivedBlockLineageGroup'],
            ['json_source_region', 'JsonSourceRegion'],
            ['json_inverse_mapping', 'JsonInverseMapping'],
            ['derived_lineage_verification_scope', 'DerivedLineageVerificationScope'],
        ] as const;
        for (const [key, name] of pairs) {
            expect(core[`${name}Schema`].safeParse).toBeTypeOf('function');
            expect(json.CONVERSATION_JSON_SCHEMAS[key]).toBe(json[`${name}JsonSchema`]);
            expect(Object.isFrozen(json[`${name}JsonSchema`])).toBe(true);
        }
    });
    it('validates exact nested ranges, JSON inverse runs and strict media evidence in Zod and AJV', () => {
        parity(core.SourceBlockSliceSchema, json.SourceBlockSliceJsonSchema, [
            [textSlice, true],
            [
                { ...textSlice, selection: { kind: 'text_range', range: { start_code_point: -1, end_code_point: 3 } } },
                false,
            ],
            [
                {
                    ...source,
                    selection: { kind: 'json_region', region: { pointer: '/a~1b', excluded_pointers: ['/a~1b/0'] } },
                },
                true,
            ],
            [
                { ...source, selection: { kind: 'json_region', region: { pointer: '/a~2b', excluded_pointers: [] } } },
                false,
            ],
            [{ ...source, selection: { kind: 'whole', range: {} } }, false],
            [
                {
                    ...source,
                    selection: {
                        kind: 'media_range',
                        range: { type: 'time_range', start_seconds: 0, end_seconds: 1 },
                        asset_id: 'asset',
                        asset_metadata_fingerprint: 'metadata',
                        verified_content_hash: 'bytes',
                    },
                },
                true,
            ],
            [
                {
                    ...source,
                    selection: {
                        kind: 'media_range',
                        range: { type: 'time_range', start_seconds: 0, end_seconds: 1 },
                        asset_id: 'asset',
                        verified_content_hash: 'bytes',
                    },
                },
                false,
            ],
        ]);
        const inverse = {
            root_source_pointer: '',
            arrays: [
                { target_pointer: '/a', source_pointer: '/a', runs: [{ target_start: 0, source_start: 2, length: 1 }] },
            ],
        };
        parity(core.JsonInverseMappingSchema, json.JsonInverseMappingJsonSchema, [
            [inverse, true],
            [
                {
                    ...inverse,
                    arrays: [{ ...inverse.arrays[0], runs: [{ target_start: 0, source_start: 2, length: 0 }] }],
                },
                false,
            ],
        ]);
        const group = {
            transform: 'text_slice',
            fidelity: 'value_preserving',
            target_block_ids: ['target'],
            source_slices: [textSlice],
        };
        parity(core.DerivedBlockLineageSchema, json.DerivedBlockLineageJsonSchema, [
            [{ version: 1, groups: [group] }, true],
            [{ version: 2, groups: [group] }, false],
            [{ version: 1, groups: [{ ...group, fidelity: 'lossless' }] }, false],
            [{ version: 1, groups: [{ ...group, target_block_ids: [] }] }, false],
            [{ version: 1, groups: [{ ...group, transform: 'json_projection' }] }, false],
        ]);
    });
    it('retains v1 operation values and distinguishes required v2 retained source proofs', () => {
        parity(core.ConversationSliceEditOperationSchema, json.ConversationSliceEditOperationJsonSchema, [
            [operation, true],
            [{ ...operation, version: 1 }, false],
            [{ ...operation, source_slices: [] }, false],
            [{ ...operation, source_protected_entry_ids: undefined }, false],
        ]);
        parity(core.ConversationEditOperationSchema, json.ConversationEditOperationJsonSchema, [[operation, true]]);
        const {
            source_slices: _slices,
            remainder_turn_ids: _turns,
            source_protected_entry_ids: _pins,
            source_entry_positions: _positions,
            source_topology_fingerprint: _topology,
            ...v1
        } = operation;
        parity(core.ConversationEditOperationV1Schema, json.ConversationEditOperationV1JsonSchema, [
            [{ ...v1, version: 1 }, true],
            [operation, false],
        ]);
        parity(core.ConversationEditOperationSchema, json.ConversationEditOperationJsonSchema, [
            [{ ...v1, version: 1 }, true],
            [{ ...v1, version: 2 }, false],
        ]);
        parity(core.DerivedLineageVerificationScopeSchema, json.DerivedLineageVerificationScopeJsonSchema, [
            [{ turn_ids: ['target'] }, true],
            [{ turn_ids: [] }, false],
            [{ turn_ids: ['target'], trust_hashes: true }, false],
        ]);
    });
    it('keeps the new request family strict and refuses caller-authored lineage or executable replacement content', () => {
        const selection = {
            conversation: source.source,
            context_revision: 0,
            source_fingerprint: 'selection',
            access: 'read_only',
            entries: [
                {
                    entry: { id: 'entry', type: 'source_turn', turn_id: 'original' },
                    blocks: [
                        {
                            kind: 'text_range',
                            block_type: 'text',
                            block_id: 'block',
                            block_fingerprint: 'hash:original',
                            range: { start_code_point: 1, end_code_point: 3 },
                        },
                    ],
                },
            ],
        };
        const command = { kind: 'protect', selection, protected: true };
        const input = {
            version: 2,
            operation_id: 'slice',
            conversation: source.source,
            expected_context_revision: 0,
            recorded_at: '2026-10-02T00:00:00.000Z',
            command,
        };
        const request = { ...input, expected_source_fingerprint: 'source' };
        parity(core.ConversationSliceEditCommandSchema, json.ConversationSliceEditCommandJsonSchema, [
            [command, true],
            [{ ...command, selection: { ...selection, access: 'write' } }, false],
            [{ ...command, operation_id: 'injected' }, false],
        ]);
        parity(core.ConversationSliceEditPlanInputSchema, json.ConversationSliceEditPlanInputJsonSchema, [
            [input, true],
            [{ ...input, version: 1 }, false],
            [request, false],
        ]);
        parity(core.ConversationSliceEditRequestSchema, json.ConversationSliceEditRequestJsonSchema, [
            [request, true],
            [input, false],
            [{ ...request, expected_revision: 999 }, false],
        ]);
        const replacement = {
            id: 'replacement',
            kind: 'user',
            authority: 'ordinary',
            model_visibility: 'include',
            status: 'completed',
            timestamps: { recorded_at: input.recorded_at },
            blocks: [{ id: 'new-block', type: 'text', text: 'summary', format: 'plain' }],
            provenance: { type: 'inserted', operation_id: input.operation_id },
        };
        const replace = {
            kind: 'replace',
            selection,
            replacement_turn: replacement,
            fidelity: 'semantic',
            placement: { mode: 'first_selected', causal_order: 'contiguous' },
        };
        parity(core.ConversationSliceEditCommandSchema, json.ConversationSliceEditCommandJsonSchema, [
            [replace, true],
            [{ ...replace, fidelity: 'lossless' }, false],
            [
                {
                    ...replace,
                    replacement_turn: {
                        ...replacement,
                        provenance: {
                            type: 'derived',
                            derivation_id: 'caller',
                            source_hash: 'invented',
                            source_turn_ids: ['original'],
                        },
                    },
                },
                false,
            ],
            [{ ...replace, replacement_turn: { ...replacement, authority: 'system' } }, false],
            [
                {
                    ...replace,
                    replacement_turn: {
                        ...replacement,
                        blocks: [
                            {
                                id: 'call',
                                type: 'tool_call',
                                call_id: 'call',
                                tool_name: 'injected',
                                executor: 'application',
                                arguments: { type: 'json', value: {} },
                            },
                        ],
                    },
                },
                false,
            ],
        ]);
    });
});
