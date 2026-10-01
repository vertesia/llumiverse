import type { CompletionResult, JSONValue } from '@llumiverse/common';
import { describe, expect, it } from 'vitest';
import { normalizeCanonicalStructuredOutput, normalizeCompletionResult, validateResult } from './validation.js';

describe('normalizeCanonicalStructuredOutput', () => {
    it('matches the legacy wrapper while owning multipart parsing and schema normalization', () => {
        const sourceTexts = ['```json\n{"count":"2",', '"optional_date":null}\n```'];
        const schema = {
            type: 'object',
            properties: {
                count: { type: 'integer' },
                enabled: { type: 'boolean', default: false },
                optional_date: { type: 'string', format: 'date-time' },
            },
            required: ['count', 'enabled'],
            additionalProperties: false,
        };

        const canonical = normalizeCanonicalStructuredOutput({ type: 'text', source_texts: sourceTexts }, schema);
        const legacy = normalizeCompletionResult(
            sourceTexts.map((value) => ({ type: 'text' as const, value })),
            schema,
        );

        expect(canonical).toEqual({
            status: 'valid',
            structured_output: { value: { count: 2, enabled: false }, source_texts: sourceTexts },
        });
        if (canonical.status !== 'valid') throw canonical.error;
        expect(legacy.status).toBe('valid');
        if (legacy.status === 'valid') expect(legacy.structured_output).toEqual(canonical.structured_output);
    });

    it('preserves exact top-level falsy and aggregate JSON values', () => {
        const cases: Array<{ value: JSONValue; schema: object }> = [
            { value: false, schema: { type: 'boolean' } },
            { value: 0, schema: { type: 'number' } },
            { value: '', schema: { type: 'string' } },
            { value: null, schema: { type: 'null' } },
            { value: [0, false, null], schema: { type: 'array' } },
        ];

        for (const { value, schema } of cases) {
            expect(normalizeCanonicalStructuredOutput({ type: 'json', value }, schema)).toEqual({
                status: 'valid',
                structured_output: { value, source_texts: [] },
            });
        }
    });

    it('normalizes an owned JSON clone without mutating provider evidence', () => {
        const value = { count: '3', optional_date: '' };
        const normalized = normalizeCanonicalStructuredOutput(
            { type: 'json', value },
            {
                type: 'object',
                properties: {
                    count: { type: 'integer' },
                    enabled: { type: 'boolean', default: false },
                    optional_date: { type: 'string', format: 'date-time' },
                },
                required: ['count', 'enabled'],
                additionalProperties: false,
            },
        );

        expect(normalized).toEqual({
            status: 'valid',
            structured_output: { value: { count: 3, enabled: false }, source_texts: [] },
        });
        expect(value).toEqual({ count: '3', optional_date: '' });
    });

    it('keeps parse failures distinct from schema-validation failures', () => {
        expect(normalizeCanonicalStructuredOutput({ type: 'text', source_texts: [] }, {})).toMatchObject({
            status: 'invalid',
            error: { code: 'json_error' },
        });
        expect(
            normalizeCanonicalStructuredOutput(
                { type: 'json', value: { answer: { nested: true } } },
                {
                    type: 'object',
                    properties: { answer: { type: 'string' } },
                    required: ['answer'],
                    additionalProperties: false,
                },
            ),
        ).toMatchObject({ status: 'invalid', error: { code: 'validation_error' } });
    });
});

describe('validateResult', () => {
    it('preserves thoughts while replacing the response content with validated JSON', () => {
        const result: CompletionResult[] = [
            { type: 'thoughts', value: 'first thought' },
            { type: 'text', value: '{"answer":"ok"}' },
            { type: 'thoughts', value: 'second thought' },
        ];

        expect(validateResult(result, { type: 'object' })).toEqual([
            { type: 'thoughts', value: 'first thought' },
            { type: 'json', value: { answer: 'ok' } },
            { type: 'thoughts', value: 'second thought' },
        ]);
    });

    // A stored result schema is deserialized into a new object on every execution, so an `$id` that
    // is already in the shared Ajv registry used to throw `schema with key or id "..." already
    // exists` from the second execution onward.
    it('validates repeatedly against an equal schema carrying the same $id', () => {
        const schema = () => ({
            $id: 'https://example.test/schemas/photo-observation.json',
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
        });
        const result: CompletionResult[] = [{ type: 'text', value: '{"answer":"ok"}' }];

        for (let attempt = 0; attempt < 3; attempt++) {
            expect(validateResult(result, schema())).toEqual([{ type: 'json', value: { answer: 'ok' } }]);
        }
    });

    it('applies the newest definition when a schema is edited but keeps its $id', () => {
        const id = 'https://example.test/schemas/edited.json';
        const result: CompletionResult[] = [{ type: 'text', value: '{"answer":"ok"}' }];

        expect(validateResult(result, { $id: id, type: 'object' })).toEqual([
            { type: 'json', value: { answer: 'ok' } },
        ]);

        // Same $id, stricter definition: the new requirement must be enforced, not the cached one.
        expect(() => validateResult(result, { $id: id, type: 'object', required: ['missing'] })).toThrow(
            /must have required property/,
        );
    });

    it('combines split answer text while preserving interleaved reasoning and media', () => {
        const result: CompletionResult[] = [
            { type: 'thoughts', value: 'before' },
            { type: 'text', value: '```json\n{"answer":' },
            { type: 'thoughts', value: 'between' },
            { type: 'image', value: 'data:image/png;base64,AAAA' },
            { type: 'text', value: '"ok"}\n```' },
        ];

        expect(validateResult(result, { type: 'object' })).toEqual([
            { type: 'thoughts', value: 'before' },
            { type: 'json', value: { answer: 'ok' } },
            { type: 'thoughts', value: 'between' },
            { type: 'image', value: 'data:image/png;base64,AAAA' },
        ]);
    });

    it.each([
        ['json language tag', '```json\n{"answer":"ok"}\n```'],
        ['case-insensitive language tag', '```JSON   {"answer":"ok"}   ```'],
        ['untagged fence', '```\n{"answer":"ok"}\n```'],
    ])('parses a fenced structured output with a %s', (_label, value) => {
        expect(validateResult([{ type: 'text', value }], { type: 'object' })).toEqual([
            { type: 'json', value: { answer: 'ok' } },
        ]);
    });

    it('parses a fenced structured output surrounded by a large whitespace run in linear time', () => {
        const whitespace = ' '.repeat(500_000);
        const value = `\`\`\`json${whitespace}{"answer":"ok"}${whitespace}\`\`\``;

        expect(validateResult([{ type: 'text', value }], { type: 'object' })).toEqual([
            { type: 'json', value: { answer: 'ok' } },
        ]);
    });

    it('rejects an unclosed fence followed by a large whitespace run without regexp backtracking', () => {
        const value = `\`\`\`json${' '.repeat(500_000)}!`;

        expect(() => validateResult([{ type: 'text', value }], { type: 'object' })).toThrow();
    });

    it.each([
        ['array', '[1,null,true]', { type: 'array' }, [1, null, true]],
        ['null', 'null', { type: 'null' }, null],
        ['string', '"value"', { type: 'string' }, 'value'],
        ['number', '42', { type: 'number' }, 42],
        ['boolean', 'true', { type: 'boolean' }, true],
    ] as const)('normalizes a top-level JSON %s', (_label, text, schema, expected) => {
        expect(validateResult([{ type: 'text', value: text }], schema)).toEqual([{ type: 'json', value: expected }]);
    });

    it.each([
        ['number', '"42"', { type: 'number' }],
        ['array', '1', { type: 'array', items: { type: 'number' } }],
    ] as const)(
        'rejects root %s coercion when the stored value would retain the wrong type',
        (_label, text, schema) => {
            const original: CompletionResult[] = [{ type: 'text', value: text }];
            const normalized = normalizeCompletionResult(original, schema);

            expect(normalized).toMatchObject({ status: 'invalid', error: { code: 'validation_error' } });
            expect(original).toEqual([{ type: 'text', value: text }]);
        },
    );

    it('retains every raw partition when the historical independent-part fallback succeeds', () => {
        const normalized = normalizeCompletionResult(
            [
                { type: 'text', value: '{"answer":"ok"}' },
                { type: 'thoughts', value: 'provider reasoning between answer partitions' },
                { type: 'text', value: 'provider trailer' },
            ],
            { type: 'object' },
        );

        expect(normalized).toMatchObject({
            status: 'valid',
            result: [
                { type: 'json', value: { answer: 'ok' } },
                { type: 'thoughts', value: 'provider reasoning between answer partitions' },
            ],
            structured_output: {
                source_texts: ['{"answer":"ok"}', 'provider trailer'],
            },
        });
    });

    it('does not mutate raw JSON evidence when schema validation fails', () => {
        const value = { discard: 'raw' };
        const result: CompletionResult[] = [{ type: 'json', value }];
        const normalized = normalizeCompletionResult(result, {
            type: 'object',
            properties: { required: { type: 'string', default: 'inserted' } },
            required: ['missing'],
            additionalProperties: false,
        });

        expect(normalized.status).toBe('invalid');
        expect(value).toEqual({ discard: 'raw' });
        expect(result).toEqual([{ type: 'json', value: { discard: 'raw' } }]);
    });

    it.each([[''], [null]] as const)(
        'removes an empty optional date before declaring canonical JSON valid (%s)',
        (emptyDate) => {
            const value = { answer: 'ok', optional_date: emptyDate };
            const schema = {
                type: 'object',
                properties: {
                    answer: { type: 'string' },
                    optional_date: { type: 'string', format: 'date-time' },
                },
                required: ['answer'],
                additionalProperties: false,
            };

            expect(validateResult([{ type: 'json', value }], schema)).toEqual([
                { type: 'json', value: { answer: 'ok' } },
            ]);
            expect(value).toEqual({ answer: 'ok', optional_date: emptyDate });
        },
    );

    it('rejects an empty date when the field is required', () => {
        const normalized = normalizeCompletionResult([{ type: 'json', value: { required_date: null } }], {
            type: 'object',
            properties: { required_date: { type: 'string', format: 'date-time' } },
            required: ['required_date'],
            additionalProperties: false,
        });

        expect(normalized).toMatchObject({ status: 'invalid', error: { code: 'validation_error' } });
    });

    it('rejects an optional invalid date default instead of repeatedly deleting and recreating it', () => {
        const normalized = normalizeCompletionResult([{ type: 'json', value: {} }], {
            type: 'object',
            properties: { optional_date: { type: 'string', format: 'date', default: '' } },
            additionalProperties: false,
        });

        expect(normalized).toMatchObject({ status: 'invalid', error: { code: 'validation_error' } });
    });
});
