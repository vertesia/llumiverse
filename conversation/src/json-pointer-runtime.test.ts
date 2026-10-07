import { describe, expect, test } from 'vitest';
import { pointerTokens, resolveJsonPointer } from './json-pointer.js';
import { inverseJsonPointer } from './json-pointer-inverse.js';
import {
    JSON_POINTER_PATTERN_SOURCE,
    MAX_PROCESSING_OUTPUT_BYTES,
    MAX_PROCESSOR_CONFIGURATION_BYTES,
} from './runtime-constants.js';
import { JsonPointerSchema } from './schemas/content-ranges.js';
import * as ProcessingSchemas from './schemas/processing.js';

describe('schema-free canonical pointer grammar', () => {
    test.each(['', '/', '/a/b', '/a~0b', '/a~1b', '/~01', '/space and 😀', '/line\nfeed'])(
        'matches public schema for valid %s',
        (pointer) => {
            expect(JsonPointerSchema.safeParse(pointer).success).toBe(true);
            expect(new RegExp(JSON_POINTER_PATTERN_SOURCE).test(pointer)).toBe(true);
            expect(() => pointerTokens(pointer)).not.toThrow();
        },
    );
    test.each(['not-root', 'a/b', '/bad~', '/~2', '/~10~x'])('rejects invalid %s at both boundaries', (pointer) => {
        expect(JsonPointerSchema.safeParse(pointer).success).toBe(false);
        expect(() => pointerTokens(pointer)).toThrow('RFC 6901');
    });
    test('rejects non-string JS ingress and preserves escapes and own-member/array rules', () => {
        expect(() => Reflect.apply(pointerTokens, undefined, [null])).toThrow('RFC 6901');
        expect(pointerTokens('/~01/~1/~0')).toEqual(['~1', '/', '~']);
        expect(resolveJsonPointer({ '~1': { '/': 7 } }, '/~01/~1').value).toBe(7);
        expect(() => resolveJsonPointer([1], '/01')).toThrow();
        expect(() => resolveJsonPointer({}, '/constructor')).toThrow();
    });
    test('retains inverse array mapping and the shared processing bounds', () => {
        expect(
            inverseJsonPointer(
                {
                    root_source_pointer: '/items',
                    arrays: [
                        {
                            target_pointer: '',
                            source_pointer: '/items',
                            runs: [{ target_start: 0, source_start: 2, length: 2 }],
                        },
                    ],
                },
                '/1/name',
            ),
        ).toBe('/items/3/name');
        expect(() => inverseJsonPointer({ root_source_pointer: '', arrays: [] }, '/bad~2')).toThrow();
        expect(ProcessingSchemas.MAX_PROCESSING_OUTPUT_BYTES).toBe(MAX_PROCESSING_OUTPUT_BYTES);
        expect(ProcessingSchemas.MAX_PROCESSOR_CONFIGURATION_BYTES).toBe(MAX_PROCESSOR_CONFIGURATION_BYTES);
    });
});
