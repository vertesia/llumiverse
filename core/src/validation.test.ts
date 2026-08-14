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
