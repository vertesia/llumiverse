import { describe, expect, it } from 'vitest';
import { assertJsonMinification, JsonMinificationError, minifyJsonLexically } from '../src/json-minification.js';

describe('RFC 8259 lexical JSON minification', () => {
    it.each([
        [' \t\r\n null \n', 'null'],
        [' true ', 'true'],
        [' false ', 'false'],
        [' { } ', '{}'],
        [' [ ] ', '[]'],
        [
            ' [ 0, -0, 9999999999999999999999999999999999999999, 1E+999999, -9.100e-999 ] ',
            '[0,-0,9999999999999999999999999999999999999999,1E+999999,-9.100e-999]',
        ],
        [' { "z" : 1, "a" : 2, "z" : 3 } ', '{"z":1,"a":2,"z":3}'],
        [' [ { "nested" : [ null, true, false ] } ] ', '[{"nested":[null,true,false]}]'],
        [' " space   escape\\t remains " ', '" space   escape\\t remains "'],
        [
            ' { "literal" : "quoted , : [ ] { } \\" and \\\\ end" } ',
            '{"literal":"quoted , : [ ] { } \\" and \\\\ end"}',
        ],
        [
            ' [ "\\uD800", "\\u0041", "\\u00e9", "é", "é", "日本語😀", "\\/" ] ',
            '["\\uD800","\\u0041","\\u00e9","é","é","日本語😀","\\/"]',
        ],
        [' "\u2028\u2029\u00a0" ', '"\u2028\u2029\u00a0"'],
        [' [ -123.000, 0e00, 1e+000, 1E-000 ] ', '[-123.000,0e00,1e+000,1E-000]'],
    ])('retains the exact token lexemes in %s', (input, output) => {
        expect(minifyJsonLexically(input)).toBe(output);
        expect(minifyJsonLexically(output)).toBe(output);
    });

    it.each([
        '',
        ' ',
        'undefined',
        'NaN',
        'Infinity',
        '-Infinity',
        '+1',
        '.1',
        '1.',
        '01',
        '-01',
        '--1',
        '- 1',
        '1e',
        '1E+',
        '1e-',
        '1e+ 1',
        '0x10',
        'truefalse',
        'null 0',
        '{}[]',
        '[1 2]',
        '[,1]',
        '[1,]',
        '{,}',
        '{"x",1}',
        '{"x" 1}',
        '{x:1}',
        '{"x":}',
        '{"x":1,}',
        '{"x":1 "y":2}',
        '[}',
        '{]',
        '[[1]',
        '{"x":[]',
        '"unfinished',
        '"\\"',
        '"\\q"',
        '"\\u123"',
        '"\\u12xz"',
        '"literal\nnewline"',
        '"literal\ttab"',
        '"\u0000"',
        '//comment\n1',
        '/*comment*/1',
        '[1/*comment*/]',
        '"ok" trailing prose',
        '\u00a0null',
        '\u000b0',
        '\u000c0',
        '\ufeff{}',
        '```json\n{}\n```',
        'remove prose , : punctuation',
    ])('rejects invalid JSON %s', (input) => {
        expect(() => minifyJsonLexically(input)).toThrow(JsonMinificationError);
    });

    it('matches ordinary JSON values while retaining every original and minified byte', () => {
        const inputs = [null, true, 17, 'escaped " punctuation : [,] \\', { first: [1, null], second: 'é' }];
        for (const value of inputs) {
            const text = JSON.stringify(value, null, 4);
            expect(JSON.parse(minifyJsonLexically(text))).toEqual(value);
            expect(minifyJsonLexically(text)).toBe(JSON.stringify(value));
            expect(text).toBe(JSON.stringify(value, null, 4));
        }
    });

    it('bounds source size, nesting and token work before allocating an unbounded output', () => {
        expect(() => minifyJsonLexically('1234', { limits: { max_code_units: 3 } })).toThrow('JSON_MINIFICATION_LIMIT');
        expect(minifyJsonLexically('[[0]]', { limits: { max_depth: 2 } })).toBe('[[0]]');
        expect(() => minifyJsonLexically('[[[0]]]', { limits: { max_depth: 2 } })).toThrow('JSON_MINIFICATION_LIMIT');
        expect(minifyJsonLexically('[0]', { limits: { max_lexical_tokens: 3 } })).toBe('[0]');
        expect(() => minifyJsonLexically('[0,1]', { limits: { max_lexical_tokens: 3 } })).toThrow(
            'JSON_MINIFICATION_LIMIT',
        );
        expect(() => minifyJsonLexically('0', { limits: { max_depth: 129 } })).toThrow(RangeError);
        expect(() => minifyJsonLexically('0', { limits: { max_lexical_tokens: Number.NaN } })).toThrow(RangeError);
    });

    it('honors cancellation without returning a partial replacement', () => {
        const controller = new AbortController();
        const reason = new Error('cancel minification');
        controller.abort(reason);
        expect(() => minifyJsonLexically(' [ 1 ] ', { signal: controller.signal })).toThrow(reason);
    });
});

describe('independent lexical proposal verification', () => {
    it('validates exact lossless output rather than JSON value equality', () => {
        const source =
            ' { "duplicate": -0, "duplicate": 1E+01, "escape": "\\u0041", "large": 9999999999999999999999 } ';
        const exact = '{"duplicate":-0,"duplicate":1E+01,"escape":"\\u0041","large":9999999999999999999999}';
        expect(() => assertJsonMinification(source, exact)).not.toThrow();
        for (const forged of [
            exact.replace('-0', '0'),
            exact.replace('1E+01', '10'),
            exact.replace('\\u0041', 'A'),
            exact.replace('9999999999999999999999', '1e22'),
            '{"large":9999999999999999999999,"duplicate":-0,"duplicate":1E+01,"escape":"\\u0041"}',
            '{"duplicate":1E+01,"escape":"\\u0041","large":9999999999999999999999}',
            `${exact} `,
        ])
            expect(() => assertJsonMinification(source, forged)).toThrow('JSON_MINIFICATION_MISMATCH');
        expect(() => assertJsonMinification('true false', 'truefalse')).toThrow('INVALID_JSON');
    });
});

it('rejects non-string JS inputs explicitly before scanning', () => {
    for (const input of [undefined, null, 0, false, {}, []]) {
        expect(() => Reflect.apply(minifyJsonLexically, undefined, [input])).toThrow('source must be a string');
        expect(() => Reflect.apply(assertJsonMinification, undefined, ['0', input])).toThrow(
            'replacement must be a string',
        );
    }
});
