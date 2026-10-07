import { Ajv2020 } from 'ajv/dist/2020.js';
import formatsPlugin from 'ajv-formats';
import { describe, expect, it } from 'vitest';
import { ExecutionReceiptSchema } from '../src/schemas/execution.js';

function receipt() {
    return {
        id: 'execution:read',
        call_id: 'call:read',
        executor: 'application',
        status: 'success',
        result_turn_id: 'turn:read',
        result_fingerprint: `sha256:${'1'.repeat(64)}`,
        recorded_at: '2026-10-04T00:00:00.000Z',
        metadata: {
            retrieval_excerpt: {
                version: 1,
                source: { conversation_id: 'conversation:read', revision: 2 },
                asset_id: 'asset:text',
                accepted_asset_operation_id: 'archive:text',
                external_reference_block_id: 'block:reference',
                reference_fingerprint: `sha256:${'2'.repeat(64)}`,
                content_hash: `sha256:${'3'.repeat(64)}`,
                byte_start: 0,
                byte_end_exclusive: 10,
                tool_definition_id: 'definition:read',
                projection: { kind: 'json_byte_excerpt' },
                returned_block_id: 'block:read',
                returned_block_fingerprint: `sha256:${'4'.repeat(64)}`,
            },
        },
    };
}

function validator() {
    const ajv = new Ajv2020({ allErrors: true, strict: true });
    formatsPlugin.default(ajv);
    return ajv.compile(ExecutionReceiptSchema.toJSONSchema({ target: 'draft-2020-12', io: 'input' }));
}

describe('bounded retrieval execution metadata schema parity', () => {
    it('compiles strict AJV and accepts exact bounded receipt metadata without granting read authority', () => {
        const validate = validator();
        const value = receipt();
        expect(ExecutionReceiptSchema.safeParse(value).success).toBe(true);
        expect(validate(value), JSON.stringify(validate.errors)).toBe(true);
    });

    it.each([
        ['asset_id', 512],
        ['accepted_asset_operation_id', 512],
        ['external_reference_block_id', 512],
        ['tool_definition_id', 512],
        ['returned_block_id', 512],
        ['reference_fingerprint', 128],
        ['content_hash', 128],
        ['returned_block_fingerprint', 128],
    ] as const)('preserves the exact %s bound in both runtime validators', (field, bound) => {
        const validate = validator();
        const value = receipt();
        value.metadata.retrieval_excerpt[field] = 'x'.repeat(bound);
        expect(ExecutionReceiptSchema.safeParse(value).success).toBe(true);
        expect(validate(value), JSON.stringify(validate.errors)).toBe(true);
        value.metadata.retrieval_excerpt[field] += 'x';
        expect(ExecutionReceiptSchema.safeParse(value).success).toBe(false);
        expect(validate(value)).toBe(false);
        value.metadata.retrieval_excerpt[field] = '';
        expect(ExecutionReceiptSchema.safeParse(value).success).toBe(false);
        expect(validate(value)).toBe(false);
    });

    it('bounds the source conversation identity and rejects unknown metadata fields in both validators', () => {
        const validate = validator();
        const value = receipt();
        value.metadata.retrieval_excerpt.source.conversation_id = 'x'.repeat(513);
        expect(ExecutionReceiptSchema.safeParse(value).success).toBe(false);
        expect(validate(value)).toBe(false);
        const unknown = { ...receipt(), metadata: { ...receipt().metadata, model_grant: true } };
        expect(ExecutionReceiptSchema.safeParse(unknown).success).toBe(false);
        expect(validate(unknown)).toBe(false);
    });
});
