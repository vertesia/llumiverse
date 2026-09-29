import type { CompletionResult } from '@llumiverse/common';
import { describe, expect, it } from 'vitest';
import { normalizeCompletionResult, validateResult } from './validation.js';

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
