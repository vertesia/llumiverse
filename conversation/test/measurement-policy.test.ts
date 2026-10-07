import { Ajv2020 } from 'ajv/dist/2020.js';
import formatsPlugin from 'ajv-formats';
import { describe, expect, it } from 'vitest';
import {
    acceptsProcessingMeasurement,
    type ContextMeasurement,
    createConversationDocument,
    ProcessingBudgetSchema,
} from '../src/index.js';
import { ConversationDocumentJsonSchema } from '../src/json-schema.js';

const measurement: ContextMeasurement = {
    input_tokens: 7,
    method: 'estimated',
    tokenizer: 'full-native-json-bpe:o200k_base',
    tokenizer_version: 'tiktoken:1.0.22:profile:1',
    adapter: 'openai-chat-completions',
    adapter_version: '1',
    source_fingerprint: 'sha256:source',
    target_model: 'gpt-4o-2024-08-06',
    measured_at: '2026-10-02T00:00:00.000Z',
};
const budget = { max_input_tokens: 100, output_reserve_tokens: 10 };

describe('processing measurement policy', () => {
    it('defaults to exact and requires complete provider counts', () => {
        expect(acceptsProcessingMeasurement(budget, measurement)).toBe(false);
        expect(acceptsProcessingMeasurement(budget, { ...measurement, method: 'exact' })).toBe(true);
        const counted = { ...measurement, method: 'provider_counted' as const };
        expect(acceptsProcessingMeasurement(budget, counted)).toBe(false);
        expect(acceptsProcessingMeasurement(budget, counted, true)).toBe(true);
        expect(acceptsProcessingMeasurement({ ...budget, measurement_policy: 'identified_estimate' }, counted)).toBe(
            false,
        );
    });

    it('requires opt-in and a named estimate version', () => {
        const optIn = { ...budget, measurement_policy: 'identified_estimate' as const };
        expect(acceptsProcessingMeasurement(optIn, measurement)).toBe(true);
        expect(acceptsProcessingMeasurement(optIn, { ...measurement, tokenizer_version: undefined })).toBe(false);
        expect(acceptsProcessingMeasurement({ ...budget, measurement_policy: 'exact_only' }, measurement)).toBe(false);
    });

    it('emits identical supported and rejected public budget shapes through document JSON schema', () => {
        const ajv = new Ajv2020({ allErrors: true, strict: true });
        formatsPlugin.default(ajv);
        const validate = ajv.compile(ConversationDocumentJsonSchema);
        for (const policy of [undefined, 'exact_only', 'identified_estimate', null, 'future', 0]) {
            const input = { ...budget, ...(policy === undefined ? {} : { measurement_policy: policy }) };
            const document = createConversationDocument({ id: 'policy', created_at: measurement.measured_at });
            const json = { ...document, processing: { ...document.processing, budget: input } };
            const valid = policy === undefined || policy === 'exact_only' || policy === 'identified_estimate';
            expect(ProcessingBudgetSchema.safeParse(input).success).toBe(valid);
            expect(validate(json), JSON.stringify(validate.errors)).toBe(valid);
        }
    });
});
