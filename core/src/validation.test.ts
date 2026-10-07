import type { CompletionResult } from '@llumiverse/common';
import { describe, expect, it } from 'vitest';
import { validateResult } from './validation.js';

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

    it('parses JSON split across multiple text parts as one response', () => {
        const result: CompletionResult[] = [
            { type: 'text', value: '{"answer":' },
            { type: 'text', value: '"ok"}' },
        ];

        expect(validateResult(result, { type: 'object', required: ['answer'] })).toEqual([
            { type: 'json', value: { answer: 'ok' } },
        ]);
    });

    it.each<CompletionResult[]>([
        [
            { type: 'json', value: { a: 1 } },
            { type: 'json', value: { b: 2 } },
        ],
        [
            { type: 'json', value: { a: 1 } },
            { type: 'text', value: '{"b":2}' },
        ],
        [
            { type: 'text', value: '{"a":1}' },
            { type: 'image', value: 'image' },
        ],
        [{ type: 'thoughts', value: 'no answer' }],
    ])('rejects multiple values or non-text response content: %j', (...parts) => {
        expect(() => validateResult(parts, { type: 'object' })).toThrow(/Expected one JSON value/);
    });

    it('preserves the original parser failure as the validation error cause', () => {
        expect(() => validateResult([{ type: 'text', value: '{"a":}' }], { type: 'object' })).toThrow(
            expect.objectContaining({ code: 'json_error', cause: expect.any(SyntaxError) }),
        );
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
});
