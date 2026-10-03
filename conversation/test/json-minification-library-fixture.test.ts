import { Ajv2020 } from 'ajv/dist/2020.js';
import formatsPlugin from 'ajv-formats';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import {
    JsonMinificationCandidateSchema,
    JsonMinificationConfigurationSchema,
    JsonMinificationMeasurementIdentitySchema,
} from '../src/index.js';
import {
    JsonMinificationCandidateJsonSchema,
    JsonMinificationConfigurationJsonSchema,
    JsonMinificationMeasurementIdentityJsonSchema,
} from '../src/json-schema.js';
import data from '../test-fixtures/json-minification-library.js';

const fixture = z
    .strictObject({
        purpose: z.string(),
        entries: z.array(
            z.strictObject({
                name: z.string(),
                component: z.enum([
                    'ConversationJsonMinificationConfiguration',
                    'ConversationJsonMinificationCandidate',
                    'ConversationJsonMinificationMeasurementIdentity',
                ]),
                valid: z.boolean(),
                value: z.unknown(),
            }),
        ),
    })
    .parse(data);
const zodSchemas = {
    ConversationJsonMinificationConfiguration: JsonMinificationConfigurationSchema,
    ConversationJsonMinificationCandidate: JsonMinificationCandidateSchema,
    ConversationJsonMinificationMeasurementIdentity: JsonMinificationMeasurementIdentitySchema,
};
const emittedSchemas = {
    ConversationJsonMinificationConfiguration: JsonMinificationConfigurationJsonSchema,
    ConversationJsonMinificationCandidate: JsonMinificationCandidateJsonSchema,
    ConversationJsonMinificationMeasurementIdentity: JsonMinificationMeasurementIdentityJsonSchema,
};
const ajv = new Ajv2020({ strict: false, allErrors: true });
formatsPlugin.default(ajv);
describe('retained library-only JSON minification fixture coverage', () => {
    for (const entry of fixture.entries) {
        it(entry.name, () => {
            expect(zodSchemas[entry.component].safeParse(entry.value).success).toBe(entry.valid);
            const validate = ajv.compile(emittedSchemas[entry.component]);
            expect(validate(entry.value), JSON.stringify(validate.errors)).toBe(entry.valid);
        });
    }
});
