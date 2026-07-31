import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import { JSONSchemaPropertiesSchema, JSONSchemaSchema } from './json-schema.js';

/**
 * The emission these two components must keep producing.
 *
 * Vertesia publishes `JSONSchema` and `JSONSchemaProperties` as OpenAPI components. Unlike the
 * option types in this package, they are NOT short-circuited by its OpenAPI scanner — they keep a
 * named TypeScript type, because a recursive schema cannot hand TypeScript a usable inferred one —
 * so the scanner still derives them wherever an unconverted type reaches them, and twenty-seven
 * published components do. A schema-backed component and the one derived from the TypeScript type
 * have to agree, or the build fails several packages downstream.
 *
 * That agreement check is shape-based rather than literal: key order carries no meaning in JSON
 * Schema, and `ts-json-schema-generator` is not self-consistent about it. The assertion below is
 * stricter than the constraint on purpose. It pins the exact emission, so any change to the schema
 * — including one that only moves a key — shows up here, in llumiverse's own test run, with the
 * before and after side by side.
 */
const EXPECTED_JSON_SCHEMA = {
    type: 'object',
    properties: {
        type: {},
        description: { type: 'string' },
        properties: { $ref: '#/$defs/JSONSchemaProperties' },
        items: { $ref: '#' },
        format: { type: 'string' },
        editor: {},
        default: {},
        additionalProperties: { anyOf: [{ type: 'boolean' }, { $ref: '#' }] },
        required: { type: 'array', items: { type: 'string' } },
    },
};

describe('JSONSchemaSchema', () => {
    it('emits the published component shape, property order included', () => {
        const emitted = z.toJSONSchema(JSONSchemaSchema, { target: 'draft-2020-12', io: 'input' }) as Record<
            string,
            unknown
        >;
        const { $schema: _schema, $defs: _defs, additionalProperties, ...rest } = emitted;
        // Compared as text, so a reordered property fails here too. Deliberately stricter than the
        // agreement the generator enforces: this is the place a change to the emission should be
        // read and approved, not discovered downstream.
        expect(JSON.stringify(rest)).toBe(JSON.stringify(EXPECTED_JSON_SCHEMA));
        // `looseObject` emits this; Vertesia's adapter drops it because `{}` and an absent
        // `additionalProperties` mean the same thing, which is how the component stays byte-identical
        // to the one derived from the interface. Anything OTHER than `{}` would not be dropped.
        expect(additionalProperties).toEqual({});
    });

    it('emits a property map with no propertyNames', () => {
        const emitted = z.toJSONSchema(JSONSchemaSchema, { target: 'draft-2020-12', io: 'input' }) as {
            $defs: Record<string, unknown>;
        };
        // `z.record` would add `propertyNames: {type: 'string'}` here, which the derived component
        // does not have — the reason `JSONSchemaPropertiesSchema` is a catchall object instead.
        expect(emitted.$defs.JSONSchemaProperties).toEqual({
            type: 'object',
            properties: {},
            additionalProperties: { $ref: '#' },
        });
    });

    it('stays open, so keywords the interface never enumerated survive', () => {
        // `enum`, `minimum`, `$ref` and friends are all valid JSON Schema and callers do send them.
        const parsed = JSONSchemaSchema.parse({ type: 'string', enum: ['a', 'b'], minLength: 1 });
        expect(parsed).toEqual({ type: 'string', enum: ['a', 'b'], minLength: 1 });
    });

    it('accepts the recursive shapes it exists to describe', () => {
        const nested = {
            type: 'object',
            properties: {
                tags: { type: 'array', items: { type: 'string' } },
                nested: { type: 'object', additionalProperties: { type: 'number' } },
            },
            required: ['tags'],
        };
        expect(JSONSchemaSchema.parse(nested)).toEqual(nested);
        expect(JSONSchemaPropertiesSchema.parse(nested.properties)).toEqual(nested.properties);
    });

    it('declares exactly the known fields, no more and no less', () => {
        // The compile-time guard is `satisfies KnownFieldSchemas` in the source, which is what a
        // `typecheck` run enforces. This is its runtime mirror, so the coverage property survives
        // even if someone loosens the mapped type: the emitted `properties` map IS the known-field
        // list, and a field added to the interface without a schema (or the reverse) moves it.
        const emitted = z.toJSONSchema(JSONSchemaSchema, { target: 'draft-2020-12', io: 'input' }) as {
            properties: Record<string, unknown>;
        };
        expect(Object.keys(emitted.properties)).toEqual([
            'type',
            'description',
            'properties',
            'items',
            'format',
            'editor',
            'default',
            'additionalProperties',
            'required',
        ]);
    });

    it('rejects a value that is not a schema at all', () => {
        expect(JSONSchemaSchema.safeParse('string').success).toBe(false);
        expect(JSONSchemaSchema.safeParse({ description: 42 }).success).toBe(false);
        expect(JSONSchemaSchema.safeParse({ required: 'name' }).success).toBe(false);
    });
});
