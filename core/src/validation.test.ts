import type { CompletionResult } from '@llumiverse/common';
import { describe, expect, it, vi } from 'vitest';
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

    // Caller result schemas often carry formats nobody registered; they must validate without console noise.
    it('validates a schema with an unregistered format without logging', () => {
        const warn = vi.spyOn(console, 'warn');
        const log = vi.spyOn(console, 'log');
        const error = vi.spyOn(console, 'error');
        try {
            const schema = {
                type: 'object',
                properties: { answer: { type: 'string', format: 'not-a-registered-format' } },
                required: ['answer'],
            };
            const result: CompletionResult[] = [{ type: 'text', value: '{"answer":"ok"}' }];

            expect(validateResult(result, schema)).toEqual([{ type: 'json', value: { answer: 'ok' } }]);
            expect(warn).not.toHaveBeenCalled();
            expect(log).not.toHaveBeenCalled();
            expect(error).not.toHaveBeenCalled();
        } finally {
            vi.restoreAllMocks();
        }
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

    describe('with a cached validator', () => {
        // A fresh object per call, as a stored schema deserialized for every execution would be.
        const invoiceSchema = () => ({
            type: 'object',
            properties: {
                total: { type: 'number' },
                currency: { type: 'string', default: 'EUR' },
                lines: {
                    type: 'array',
                    items: {
                        type: 'object',
                        properties: { sku: { type: 'string' } },
                        required: ['sku'],
                        additionalProperties: false,
                    },
                },
            },
            required: ['total'],
        });

        function validateText(value: string) {
            try {
                return { result: validateResult([{ type: 'text', value }], invoiceSchema()) };
            } catch (error: unknown) {
                return { error: error instanceof Error ? error.message : String(error) };
            }
        }

        it('returns the same coerced, defaulted and pruned data on every call', () => {
            const input = '{"total":"12.5","lines":[{"sku":"A","extra":true}]}';
            const first = validateText(input);

            expect(first).toEqual({
                result: [{ type: 'json', value: { total: 12.5, currency: 'EUR', lines: [{ sku: 'A' }] } }],
            });
            for (let attempt = 0; attempt < 3; attempt++) {
                expect(validateText(input)).toEqual(first);
            }
        });

        it('reports the same errors on every call', () => {
            const input = '{"total":1,"lines":[{"sku":"A"},{"name":"no sku"}]}';
            const first = validateText(input);

            expect(first.error).toBe('/lines/1: must have required property \'sku\'\n{"missingProperty":"sku"}');
            for (let attempt = 0; attempt < 3; attempt++) {
                expect(validateText(input)).toEqual(first);
            }
        });

        it('applies an edited schema immediately', () => {
            const result: CompletionResult[] = [{ type: 'text', value: '{"total":1}' }];
            expect(validateResult(result, invoiceSchema())).toEqual([
                { type: 'json', value: { total: 1, currency: 'EUR' } },
            ]);
            expect(() => validateResult(result, { ...invoiceSchema(), required: ['total', 'lines'] })).toThrow(
                /must have required property 'lines'/,
            );
        });

        it('keeps results and errors separate for interleaved concurrent calls', async () => {
            const inputs = Array.from({ length: 40 }, (_, i) =>
                i % 2 === 0 ? `{"total":"${i}"}` : `{"lines":[{"sku":${i}}],"total":"not a number ${i}"}`,
            );

            const outcomes = await Promise.all(
                inputs.map(async (input, i) => {
                    await new Promise<void>((resolve) => setTimeout(resolve, (i * 7) % 5));
                    return validateText(input);
                }),
            );

            outcomes.forEach((outcome, i) => {
                if (i % 2 === 0) {
                    expect(outcome).toEqual({ result: [{ type: 'json', value: { total: i, currency: 'EUR' } }] });
                } else {
                    expect(outcome.error).toBe('/total: must be number\n{"type":"number"}');
                }
            });
        });
    });
});
