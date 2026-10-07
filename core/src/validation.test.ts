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

    it('parses JSON split across multiple text parts as one response', () => {
        const result: CompletionResult[] = [
            { type: 'text', value: '{"answer":' },
            { type: 'text', value: '"ok"}' },
        ];

        expect(validateResult(result, { type: 'object', required: ['answer'] })).toEqual([
            { type: 'json', value: { answer: 'ok' } },
        ]);
    });

    it('joins numeric fragments rather than treating the first digit as a complete answer', () => {
        expect(
            validateResult(
                [
                    { type: 'text', value: '4' },
                    { type: 'text', value: '2' },
                ],
                { type: 'number' },
            ),
        ).toEqual([{ type: 'json', value: 42 }]);
    });

    it.each([
        ['{"items":[', '{"id":"2","extra":true}', ']}'],
        ['[', '{"id":"2","extra":true}', ']'],
        ['{"item":', '{"id":"2","extra":true}', '}'],
    ])('prefers the complete joined answer to its inner container: %j', (...texts) => {
        const parts: CompletionResult[] = texts.map((value) => ({ type: 'text', value }));
        const itemSchema = {
            type: 'object',
            properties: { id: { type: 'number' }, label: { type: 'string', default: 'default' } },
            required: ['id'],
            additionalProperties: false,
        };
        const schema =
            texts[0] === '['
                ? { type: 'array', items: itemSchema }
                : {
                      type: 'object',
                      properties: texts[0].includes('items')
                          ? { items: { type: 'array', items: itemSchema } }
                          : { item: itemSchema },
                  };
        const item = { id: 2, label: 'default' };
        const expected = texts[0] === '[' ? [item] : texts[0].includes('items') ? { items: [item] } : { item };
        for (const allowRepair of [true, false]) {
            expect(validateResult(parts, {}, allowRepair)).toEqual([
                { type: 'json', value: JSON.parse(texts.join('')) },
            ]);
            expect(validateResult(parts, schema, allowRepair)).toEqual([{ type: 'json', value: expected }]);
        }
    });

    it.each([
        ['```json\n[', '{"id":"2","extra":true}', ']\n```', [{ id: 2, label: 'default' }]],
        ['```json\n{"item":', '{"id":"2","extra":true}', '}\n```', { item: { id: 2, label: 'default' } }],
    ])('preserves a complete fenced multipart container: %s', (start, inner, end, expected) => {
        const itemSchema = {
            type: 'object',
            properties: { id: { type: 'number' }, label: { type: 'string', default: 'default' } },
            required: ['id'],
            additionalProperties: false,
        };
        const schema = Array.isArray(expected)
            ? { type: 'array', items: itemSchema }
            : { type: 'object', properties: { item: itemSchema }, required: ['item'] };
        const text = [start, inner, end].join('');
        const parts: CompletionResult[] = [start, inner, end].map((value) => ({ type: 'text', value }));
        for (const allowRepair of [true, false]) {
            expect(validateResult(parts, {}, allowRepair)).toEqual([
                { type: 'json', value: JSON.parse(text.slice(text.indexOf('\n') + 1, text.lastIndexOf('\n'))) },
            ]);
            const onDiagnostic = vi.fn();
            expect(validateResult(parts, schema, { allowRepair, onDiagnostic })).toEqual([
                { type: 'json', value: expected },
            ]);
            expect(onDiagnostic).toHaveBeenCalledWith({ extracted: true, repaired: false, original_text: text });
        }
    });

    it('retains a complete first text answer before interpreting later independent text', () => {
        expect(
            validateResult(
                [
                    { type: 'text', value: '{"a":1}' },
                    { type: 'text', value: '{"b":2}' },
                ],
                { type: 'object' },
            ),
        ).toEqual([{ type: 'json', value: { a: 1 } }]);
    });

    it.each([
        ['```json\n{"a":"2"}\n``` [done]'],
        ['Answer [note {"a":"2"}]'],
        ['[note {"a":"2"}]'],
        ['[[note {"a":"2"}]]'],
        ['[42 notes {"a":"2"}]'],
        ['[true explanation {"a":"2"}]'],
        ['"Answer" {"a":"2"}\nextra [brackets]'],
        ['"Answer"\n{"a":"2"}'],
        ['\'Answer\'\n{"a":"2"}'],
        ['```json\n{"a":"1","extra":true}\n```', '{"a":"2"}'],
        ['Answer: {"a":"1","extra":true}', '{"a":"2"}'],
    ])('preserves the complete answer through extraction and schema validation: %j', (...texts) => {
        const parts: CompletionResult[] = texts.map((value) => ({ type: 'text', value }));
        const a = texts.length === 1 ? '2' : '1';
        for (const allowRepair of [true, false]) {
            expect(validateResult(parts, {}, allowRepair)).toEqual([
                { type: 'json', value: { a, ...(texts.length > 1 ? { extra: true } : {}) } },
            ]);
            expect(
                validateResult(
                    parts,
                    {
                        type: 'object',
                        properties: { a: { type: 'number' }, d: { type: 'string', default: 'default' } },
                        required: ['a'],
                        additionalProperties: false,
                    },
                    allowRepair,
                ),
            ).toEqual([{ type: 'json', value: { a: Number(a), d: 'default' } }]);
        }
    });

    it.each(['```json\n[1]\n```', 'Answer: [1]'])(
        'preserves an earlier wrapped array before a later object: %s',
        (first) => {
            for (const allowRepair of [true, false]) {
                expect(
                    validateResult(
                        [
                            { type: 'text', value: first },
                            { type: 'text', value: '{"a":2}' },
                        ],
                        {},
                        allowRepair,
                    ),
                ).toEqual([{ type: 'json', value: [1] }]);
            }
        },
    );

    it.each<CompletionResult[]>([
        [
            { type: 'json', value: { a: 1 } },
            { type: 'json', value: { b: 2 } },
        ],
        [
            { type: 'json', value: { a: 1 } },
            { type: 'text', value: 'Explanation' },
        ],
        [
            { type: 'text', value: '{"a":1}' },
            { type: 'image', value: 'image' },
        ],
    ])('preserves typed JSON precedence and validates text with auxiliary parts: %j', (...parts) => {
        expect(validateResult(parts, { type: 'object' })).toEqual([{ type: 'json', value: { a: 1 } }]);
    });

    it('rejects responses with no JSON or text content', () => {
        expect(() => validateResult([{ type: 'thoughts', value: 'no answer' }], { type: 'object' })).toThrow(
            /No JSON compatible response/,
        );
    });

    it('rejects repaired output that fails its result schema', () => {
        expect(() =>
            validateResult([{ type: 'text', value: '{"a":}' }], {
                type: 'object',
                properties: { a: { type: 'number', minimum: 1 } },
            }),
        ).toThrow(expect.objectContaining({ code: 'validation_error' }));
    });

    it('preserves the original parser failure as the validation error cause', () => {
        expect(() => validateResult([{ type: 'text', value: '{"a":1} {"b":2}' }], { type: 'object' })).toThrow(
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
