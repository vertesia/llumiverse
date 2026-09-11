import { Ajv2020 } from 'ajv/dist/2020.js';
import formatsPlugin from 'ajv-formats';
import { describe, expect, it } from 'vitest';
import {
    AppendConversationRecordsOptionsJsonSchema,
    AppendConversationRecordsResultJsonSchema,
    ConversationRecordBatchJsonSchema,
    DecodedConversationResponseJsonSchema,
} from '../src/json-schema.js';
import {
    AppendConversationRecordsOptionsSchema,
    AppendConversationRecordsResultSchema,
    ConversationRecordBatchSchema,
    DecodedConversationResponseSchema,
} from '../src/schemas/ingestion.js';
import { emptyDocument, importedGeneration, RECORDED_AT, userTurn } from './fixtures.js';

describe('ingestion schema contracts', () => {
    const options = {
        expected_revision: 0,
        operation_id: 'operation',
        payload_fingerprint: 'fingerprint',
        recorded_at: RECORDED_AT,
    };

    it('keeps Zod and emitted JSON Schema aligned on append inputs and results', () => {
        const pairs = [
            {
                zod: AppendConversationRecordsOptionsSchema,
                json: AppendConversationRecordsOptionsJsonSchema,
                valid: options,
                invalid: [
                    { ...options, expected_revision: -1 },
                    { ...options, expected_revision: 1.5 },
                    { ...options, expected_revision: Number.MAX_SAFE_INTEGER + 1 },
                    { ...options, operation_id: '' },
                    { ...options, recorded_at: 'tomorrow' },
                    { ...options, extra: true },
                ],
            },
            {
                zod: ConversationRecordBatchSchema,
                json: ConversationRecordBatchJsonSchema,
                valid: { turns: [userTurn('user')], active_tool_definition_ids: [] },
                invalid: [{ turns: [{}] }, { active_tool_definition_ids: [''] }, { extra: true }],
            },
            {
                zod: AppendConversationRecordsResultSchema,
                json: AppendConversationRecordsResultJsonSchema,
                valid: { document: emptyDocument(), applied: true, accepted_turn_ids: [], accepted_generation_ids: [] },
                invalid: [
                    { document: emptyDocument(), applied: 'yes', accepted_turn_ids: [], accepted_generation_ids: [] },
                ],
            },
        ];
        for (const pair of pairs) {
            const ajv = new Ajv2020({ strict: true });
            formatsPlugin.default(ajv);
            const validate = ajv.compile(pair.json);
            expect(pair.zod.safeParse(pair.valid).success).toBe(true);
            expect(validate(pair.valid), JSON.stringify(validate.errors)).toBe(true);
            for (const invalid of pair.invalid) {
                expect(pair.zod.safeParse(invalid).success).toBe(false);
                expect(validate(invalid)).toBe(false);
            }
        }
    });

    it('does not accept imported historical metadata as a decoded live response', () => {
        const value = {
            turns: [],
            generation: importedGeneration('imported'),
            diagnostics: [],
            payload_fingerprint: 'fp',
        };
        const ajv = new Ajv2020({ strict: true });
        formatsPlugin.default(ajv);
        expect(DecodedConversationResponseSchema.safeParse(value).success).toBe(false);
        expect(ajv.compile(DecodedConversationResponseJsonSchema)(value)).toBe(false);
    });
});
