import type { JSONSchema } from '@llumiverse/core';
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
                    required: ['name'],
                    additionalProperties: false,
                },
            },
            required: ['suggested_domain'],
            additionalProperties: false,
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
            required: ['unit'],
            additionalProperties: false,
        });

        expect(result.strict).toBe(true);
        expect(result.schema.properties?.unit).toEqual({ type: ['string', 'null'] });
    });

    it('uses non-strict mode instead of narrowing optional fields or extra keys', () => {
        const schema = {
            type: 'object',
            properties: {
                optional: { type: 'string' },
            },
        } satisfies JSONSchema;

        const result = formatOpenAISchema(schema);

        expect(result.strict).toBe(false);
        expect(result.schema).toEqual(schema);
    });

    it.each<[string, JSONSchema]>([
        ['unsupported string format', { type: 'string', format: 'uri' }],
        ['unsupported array keyword', { type: 'array', items: { type: 'string' }, uniqueItems: true }],
        ['unsupported composition keyword', { type: 'object', allOf: [{ type: 'object' }] }],
        ['unsupported union keyword', { oneOf: [{ type: 'string' }, { type: 'number' }] }],
    ])('uses non-strict mode for %s', (_name, schema) => {
        const rootSchema = {
            type: 'object',
            properties: { value: schema },
            required: ['value'],
            additionalProperties: false,
        } satisfies JSONSchema;
        const result = formatOpenAISchema(rootSchema);

        expect(result.strict).toBe(false);
        expect(result.schema).toEqual(rootSchema);
    });

    it('keeps supported constraints in strict mode', () => {
        const result = formatOpenAISchema({
            type: 'object',
            properties: {
                email: { type: 'string', format: 'email', pattern: '.*@.*' },
                values: { type: 'array', items: { type: 'integer' }, minItems: 1, maxItems: 3 },
            },
            required: ['email', 'values'],
            additionalProperties: false,
        });

        expect(result.strict).toBe(true);
        expect(result.schema.properties?.email).toEqual({
            type: 'string',
            format: 'email',
            pattern: '.*@.*',
        });
    });

    it('normalizes object variants inside anyOf and definitions', () => {
        const result = formatOpenAISchema({
            type: 'object',
            properties: {
                value: {
                    anyOf: [
                        {
                            type: 'object',
                            properties: { name: { type: 'string' } },
                            required: ['name'],
                            additionalProperties: false,
                        },
                        { type: 'null' },
                    ],
                },
            },
            required: ['value'],
            additionalProperties: false,
            $defs: {
                item: {
                    type: 'object',
                    properties: { id: { type: 'string' } },
                    required: ['id'],
                    additionalProperties: false,
                },
            },
        });

        expect(result.strict).toBe(true);
        expect(result.schema.properties?.value).toEqual({
            anyOf: [
                {
                    type: 'object',
                    properties: { name: { type: 'string' } },
                    required: ['name'],
                    additionalProperties: false,
                },
                { type: 'null' },
            ],
        });
        expect(result.schema.$defs).toEqual({
            item: {
                type: 'object',
                properties: { id: { type: 'string' } },
                required: ['id'],
                additionalProperties: false,
            },
        });
    });
});
