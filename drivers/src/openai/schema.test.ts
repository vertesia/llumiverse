import { describe, expect, it } from 'vitest';
import { formatOpenAISchema } from './schema.js';

describe('formatOpenAISchema', () => {
    it('adds additionalProperties false to nullable object properties in strict mode', () => {
        const result = formatOpenAISchema({
            type: 'object',
            properties: {
                suggested_domain: {
                    type: ['object', 'null'],
                    properties: {
                        name: { type: 'string' },
                    },
                },
            },
        });

        expect(result).toEqual({
            strict: true,
            schema: {
                type: 'object',
                properties: {
                    suggested_domain: {
                        type: ['object', 'null'],
                        properties: {
                            name: { type: 'string' },
                        },
                        required: ['name'],
                        additionalProperties: false,
                    },
                },
                required: ['suggested_domain'],
                additionalProperties: false,
            },
        });
    });

    it('keeps nullable scalar properties in strict mode', () => {
        const result = formatOpenAISchema({
            type: 'object',
            properties: {
                unit: { type: ['string', 'null'] },
            },
        });

        expect(result.strict).toBe(true);
        expect(result.schema.properties?.unit).toEqual({ type: ['string', 'null'] });
    });
});
