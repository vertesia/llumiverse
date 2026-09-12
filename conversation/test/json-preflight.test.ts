import { afterEach, describe, expect, it, vi } from 'vitest';
import { preflightJsonInput } from '../src/index.js';

afterEach(() => {
    vi.restoreAllMocks();
});

describe('preflightJsonInput', () => {
    it('accepts shared acyclic values and counts each serialized occurrence', () => {
        const shared = { text: 'same' };
        const input = { first: shared, second: shared };
        const result = preflightJsonInput(input);

        expect(result).toMatchObject({ success: true, nodes: 5 });
        expect(result.bytes).toBe(new TextEncoder().encode(JSON.stringify(input)).byteLength);
    });

    it('rejects cycles while reporting the earlier active path', () => {
        const input: { self?: unknown } = {};
        input.self = input;

        const result = preflightJsonInput(input);
        expect(result.success).toBe(false);
        expect(result.diagnostics[0]).toMatchObject({
            code: 'JSON_CYCLE',
            path: '/self',
            related_paths: [''],
        });
    });

    it('rejects accessors without invoking their getters', () => {
        let getterCalls = 0;
        const input = Object.defineProperty({}, 'secret', {
            enumerable: true,
            get() {
                getterCalls += 1;
                return 'should-not-run';
            },
        });

        const result = preflightJsonInput(input);
        expect(result.success).toBe(false);
        expect(result.diagnostics[0]?.code).toBe('JSON_ACCESSOR_PROPERTY');
        expect(getterCalls).toBe(0);
    });

    it.each([
        { value: new Date(0), code: 'JSON_NON_PLAIN_OBJECT' },
        { value: [undefined], code: 'JSON_UNSUPPORTED_TYPE' },
        { value: [Number.NaN], code: 'JSON_NON_FINITE_NUMBER' },
        { value: [-0], code: 'JSON_NEGATIVE_ZERO' },
        { value: Array(1), code: 'JSON_SPARSE_ARRAY' },
    ])('rejects non-JSON input with $code', ({ value, code }) => {
        const result = preflightJsonInput(value);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === code)).toBe(true);
    });

    it('rejects symbols and named array properties', () => {
        const symbolObject = { value: 1, [Symbol('hidden')]: 2 };
        const symbolResult = preflightJsonInput(symbolObject);
        expect(symbolResult.success).toBe(false);
        expect(symbolResult.diagnostics[0]?.code).toBe('JSON_SYMBOL_KEY');

        const array = [1] as number[] & { label?: string };
        array.label = 'named';
        const arrayResult = preflightJsonInput(array);
        expect(arrayResult.success).toBe(false);
        expect(arrayResult.diagnostics[0]?.code).toBe('JSON_ARRAY_PROPERTY');
    });

    it('checks oversized array length before scanning keys or element descriptors', () => {
        let ownKeyReads = 0;
        let descriptorReads = 0;
        const input = new Proxy(Array(10_000), {
            ownKeys(target) {
                ownKeyReads += 1;
                return Reflect.ownKeys(target);
            },
            getOwnPropertyDescriptor(target, key) {
                descriptorReads += 1;
                return Reflect.getOwnPropertyDescriptor(target, key);
            },
        });

        const result = preflightJsonInput(input, { max_array_length: 1 });
        expect(result.success).toBe(false);
        expect(result.diagnostics[0]?.code).toBe('JSON_MAX_ARRAY_LENGTH');
        expect(ownKeyReads).toBe(0);
        expect(descriptorReads).toBe(0);
    });

    it('enforces depth and object-property limits', () => {
        const depth = preflightJsonInput({ nested: { value: 1 } }, { max_depth: 1 });
        expect(depth.success).toBe(false);
        expect(depth.diagnostics.some((diagnostic) => diagnostic.code === 'JSON_MAX_DEPTH')).toBe(true);

        const properties = preflightJsonInput({ first: 1, second: 2 }, { max_object_properties: 1 });
        expect(properties.success).toBe(false);
        expect(properties.diagnostics[0]?.code).toBe('JSON_MAX_OBJECT_PROPERTIES');
    });

    it('caps diagnostics exactly and does not append a synthetic extra diagnostic', () => {
        const result = preflightJsonInput([undefined, undefined], { max_diagnostics: 1 });
        expect(result.success).toBe(false);
        expect(result.diagnostics).toHaveLength(1);
    });

    it('bounds oversized diagnostic paths and marks truncation', () => {
        const key = 'x'.repeat(100_000);
        const result = preflightJsonInput({ [key]: 'value' }, { max_string_bytes: 32, max_bytes: 128 });
        expect(result.success).toBe(false);
        expect(result.diagnostics[0]?.path).toHaveLength(512);
        expect(result.diagnostics[0]?.path.endsWith('…')).toBe(true);
        expect(JSON.stringify(result.diagnostics).length).toBeLessThan(5_000);
    });

    it('stops byte observation at a bounded over-limit value', () => {
        const result = preflightJsonInput('x'.repeat(100_000), { max_string_bytes: 32, max_bytes: 128 });
        expect(result.success).toBe(false);
        expect(result.diagnostics[0]).toMatchObject({ code: 'JSON_MAX_STRING_BYTES', observed: 33 });
    });

    it.each([
        { limits: { max_bytes: 128 }, code: 'JSON_MAX_BYTES' },
        { limits: { max_string_bytes: 128 }, code: 'JSON_MAX_STRING_BYTES' },
    ])('stops scanning at the first applicable $code budget', ({ limits, code }) => {
        const charCodeAt = vi.spyOn(String.prototype, 'charCodeAt');
        const result = preflightJsonInput('x'.repeat(1_000_000), limits);
        expect(result.success).toBe(false);
        expect(result.diagnostics[0]?.code).toBe(code);
        expect(charCodeAt).not.toHaveBeenCalled();
    });

    it('does not schedule or inspect pending siblings beyond the node budget', () => {
        let siblingKeyReads = 0;
        const untouchedSibling = new Proxy([1, 2, 3], {
            ownKeys(target) {
                siblingKeyReads += 1;
                return Reflect.ownKeys(target);
            },
        });
        const result = preflightJsonInput([[0], untouchedSibling], { max_nodes: 2 });
        expect(result.success).toBe(false);
        expect(result.diagnostics[0]?.code).toBe('JSON_MAX_NODES');
        expect(siblingKeyReads).toBe(0);
    });

    it('rejects the reserved record key without rejecting ordinary built-in-looking keys', () => {
        const reserved = preflightJsonInput(JSON.parse('{"__proto__":{"value":1}}'));
        expect(reserved.success).toBe(false);
        expect(reserved.diagnostics[0]?.code).toBe('JSON_RESERVED_PROPERTY_KEY');

        expect(preflightJsonInput({ constructor: 1, toString: 2, hasOwnProperty: 3 }).success).toBe(true);
    });
});
