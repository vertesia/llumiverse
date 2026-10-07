import { Ajv2020 } from 'ajv/dist/2020.js';
import formatsPlugin from 'ajv-formats';
import { describe, expect, it } from 'vitest';
import { JsonMinificationConfigurationSchema, JsonMinificationMeasuredProjectionSchema } from '../src/index.js';
import {
    JsonMinificationConfigurationJsonSchema,
    JsonMinificationMeasuredProjectionJsonSchema,
} from '../src/json-schema.js';

const ajv = new Ajv2020({ strict: false });
formatsPlugin.default(ajv);
const configuration = {
    format: 'raw_json_text',
    minimum_token_reduction: 2,
    max_code_units: 1048576,
    max_depth: 128,
    max_lexical_tokens: 262144,
};
const measured = {
    input_tokens: 5,
    method: 'exact',
    tokenizer: 'tokenizer',
    tokenizer_version: '1',
    adapter: 'adapter',
    adapter_version: '1',
    source_fingerprint: 'sha256:source',
    target_model: 'target',
    measured_at: '2026-10-02T00:00:00.000Z',
};
describe('JSON minification emitted schema parity', () => {
    it.each([
        ['max_code_units', 1048577],
        ['max_depth', 129],
        ['max_lexical_tokens', 262145],
        ['max_code_units', 0],
        ['max_depth', 0],
        ['max_lexical_tokens', 0],
        ['minimum_token_reduction', 0],
    ])('rejects %s=%s in Zod and exported JSON Schema', (field, value) => {
        const validate = ajv.compile(JsonMinificationConfigurationJsonSchema);
        expect(validate(configuration)).toBe(true);
        const invalid = { ...configuration, [field]: value };
        expect(JsonMinificationConfigurationSchema.safeParse(invalid).success).toBe(false);
        expect(validate(invalid)).toBe(false);
    });
    it('requires the pinned tokenizer version without changing general ContextMeasurement', () => {
        const validate = ajv.compile(JsonMinificationMeasuredProjectionJsonSchema);
        expect(validate(measured)).toBe(true);
        const { tokenizer_version: _version, ...without } = measured;
        expect(JsonMinificationMeasuredProjectionSchema.safeParse(without).success).toBe(false);
        expect(validate(without)).toBe(false);
    });
});
