import { JSONRepairError } from 'jsonrepair';
import { describe, expect, it, vi } from 'vitest';
import { parseJSON, parseJSONOutput } from './json.js';

describe('parseJSONOutput', () => {
    it.each([
        '"keep [this] and {this}"',
        '"escaped \\" quote and ]"',
        '42',
        '-2.5e3',
        'true',
        'false',
        'null',
        '[1, {"text":"[}]"}]',
        '{"a":1,"a":2}',
    ])('preserves valid JSON without recovery: %s', (text) => {
        const onDiagnostic = vi.fn();
        expect(parseJSONOutput(text, { onDiagnostic })).toEqual(JSON.parse(text));
        expect(onDiagnostic).not.toHaveBeenCalled();
    });

    it.each([
        ["{a:'x'}", { a: 'x' }],
        ['{"a":1,}', { a: 1 }],
        ['[1 2]', [1, 2]],
        ['{"a" 1}', { a: 1 }],
        ['{"a":1 /* don\'t treat } ] as syntax */,"b":2}', { a: 1, b: 2 }],
        ['{"a":1 // a comment containing " [ }\n}', { a: 1 }],
        ["'complete string'", 'complete string'],
        ['{"items":[{"id":"a"}', { items: [{ id: 'a' }] }],
        ['{"a":}', { a: null }],
        ['{"a":"raw\nnewline"}', { a: 'raw\nnewline' }],
        ['{"name":Alice Smith}', { name: 'Alice Smith' }],
        ['{"a":"unfinished', { a: 'unfinished' }],
        ['{"a":"keep } the whole string"', { a: 'keep } the whole string' }],
        ['[1,2,...]', [1, 2]],
    ])('uses normal jsonrepair recovery and reports diagnostics: %s', (text, expected) => {
        const onDiagnostic = vi.fn();
        expect(parseJSONOutput(text, { onDiagnostic })).toEqual(expected);
        expect(onDiagnostic).toHaveBeenCalledWith({
            repaired: true,
            extracted: false,
            original_text: text,
            parse_error: expect.any(String),
        });
    });

    it.each([
        '{"a":1} {"b":2}',
        '{"a":1} {"b":',
        '{"a":1} 42',
        'Answer: {"a":1} [2]',
        'Answer: {"a":1} "second"',
        'Answer: {"a":1} true',
        '```json\n{"a":1}\n```\n```json\n{"b":2}\n```',
        '{"a":1 + 2}',
        '// {"a":1}',
        '/* {"a":1} */',
    ])('rejects ambiguous extraction or unrecoverable output: %s', (text) => {
        const onDiagnostic = vi.fn();
        expect(() => parseJSONOutput(text, { onDiagnostic })).toThrow();
        expect(onDiagnostic).not.toHaveBeenCalled();
    });

    it.each([
        ['```json\n[1,2]\n```', [1, 2]],
        ['```\n"keep {this}"\n```', 'keep {this}'],
        ['```json\n{"text":"keep ```"}\n```', { text: 'keep ```' }],
        ['Answer: {"a":1} done.', { a: 1 }],
        ['{"a":1} — confidence: 0.9', { a: 1 }],
        ['callback({"a":1});', { a: 1 }],
        ['```json\n{"a":1}', { a: 1 }],
    ])('reports extraction independently of repair: %s', (text, expected) => {
        const onDiagnostic = vi.fn();
        expect(parseJSONOutput(text, { allowRepair: false, onDiagnostic })).toEqual(expected);
        expect(onDiagnostic).toHaveBeenCalledWith({ extracted: true, repaired: false, original_text: text });
    });

    it('uses jsonrepair for malformed fenced output and honors the opt-out', () => {
        const text = '```json\n{"a":1,}\n```';
        const onDiagnostic = vi.fn();
        expect(parseJSONOutput(text, { onDiagnostic })).toEqual({ a: 1 });
        expect(onDiagnostic).toHaveBeenCalledWith(expect.objectContaining({ repaired: true }));
        expect(() => parseJSONOutput(text, { allowRepair: false })).toThrow();
    });

    it('retains the historical fallback for malformed JSON in prose', () => {
        const text = 'Answer: {"name":Alice Smith} done.';
        const onDiagnostic = vi.fn();
        expect(parseJSONOutput(text, { onDiagnostic })).toEqual({ name: 'Alice Smith' });
        expect(onDiagnostic).toHaveBeenCalledWith(expect.objectContaining({ extracted: true, repaired: true }));
    });

    it('does not swallow a diagnostic consumer failure or retry successful extraction', () => {
        const failure = new Error('diagnostic consumer failed');
        const onDiagnostic = vi.fn(() => {
            throw failure;
        });
        expect(() => parseJSONOutput('```json\n{"a":1}\n```', { onDiagnostic })).toThrow(failure);
        expect(onDiagnostic).toHaveBeenCalledTimes(1);
    });

    it('keeps strict parsing and repair failures in the error cause', () => {
        expect(() => parseJSONOutput('{"a":1} {"b":2}')).toThrow(
            expect.objectContaining({
                cause: expect.objectContaining({
                    errors: [expect.any(SyntaxError), expect.any(JSONRepairError)],
                }),
            }),
        );
    });
});

describe('parseJSON', () => {
    it('supports repair and its opt-out in both entry points', () => {
        expect(parseJSON('{"a":1')).toEqual({ a: 1 });
        expect(() => parseJSON('{"a":1', false)).toThrow();
        expect(parseJSONOutput('{"a":1')).toEqual({ a: 1 });
        expect(() => parseJSONOutput('{"a":1', { allowRepair: false })).toThrow();
    });
});
