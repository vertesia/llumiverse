import { Ajv2020 } from 'ajv/dist/2020.js';
import formatsPlugin from 'ajv-formats';
import { describe, expect, it } from 'vitest';
import {
    copyNativeConversationByteView,
    NativeConversationImportDiagnosticSchema,
    NativeConversationImportOptionsSchema,
    NativeConversationImportReportSchema,
    NativeConversationImportResultSchema,
    parseNativeConversationImportOptions,
    parseNativeConversationImportReport,
    parseNativeConversationImportResult,
    preflightJsonInput,
    preflightNativeConversationImportInput,
} from '../src/index.js';
import {
    CONVERSATION_JSON_SCHEMAS,
    NativeConversationImportDiagnosticJsonSchema,
    NativeConversationImportOptionsJsonSchema,
    NativeConversationImportReportJsonSchema,
    NativeConversationImportResultJsonSchema,
} from '../src/json-schema.js';
import { emptyDocument } from './fixtures.js';

const origin = { conversation_id: 'source', recorded_at: '2026-10-01T00:00:00.000Z', provider: 'recorded-host' };
const diagnostic = { code: 'IMPORT_CONTINUATION_NOT_VALIDATED', message: 'Continuation requires target validation.' };
const report = {
    protocol: 'explicit-protocol',
    adapter_version: 'adapter-1',
    completeness: 'unknown',
    readiness: 'not_validated',
    diagnostics: [diagnostic],
};

describe('authoritative native import contracts', () => {
    it.each([
        {
            schema: NativeConversationImportOptionsSchema,
            json: NativeConversationImportOptionsJsonSchema,
            valid: origin,
            invalid: [
                { ...origin, provider: '' },
                { ...origin, recorded_at: 'unknown' },
                { ...origin, tool_definitions: [{}] },
            ],
        },
        {
            schema: NativeConversationImportDiagnosticSchema,
            json: NativeConversationImportDiagnosticJsonSchema,
            valid: diagnostic,
            invalid: [
                { ...diagnostic, code: 'unsupported' },
                { ...diagnostic, entity_ids: [''] },
            ],
        },
        {
            schema: NativeConversationImportReportSchema,
            json: NativeConversationImportReportJsonSchema,
            valid: report,
            invalid: [
                { ...report, readiness: 'ready' },
                { ...report, completeness: 'maybe' },
                { ...report, diagnostics: [{}] },
            ],
        },
        {
            schema: NativeConversationImportResultSchema,
            json: NativeConversationImportResultJsonSchema,
            valid: { document: emptyDocument(), report },
            invalid: [{ document: { ...emptyDocument(), schema_version: 'future' }, report }],
        },
    ])('keeps Zod and generated AJV contract $json.$id in parity', ({ schema, json, valid, invalid }) => {
        const ajv = new Ajv2020({ strict: true, allErrors: true });
        formatsPlugin.default(ajv);
        const validate = ajv.compile(json);
        for (const input of [valid, { ...valid, extra: true }, ...invalid]) {
            expect(validate(input), JSON.stringify(validate.errors)).toBe(schema.safeParse(input).success);
        }
        expect(validate(valid)).toBe(true);
        expect(Object.isFrozen(json)).toBe(true);
        expect(Object.values(CONVERSATION_JSON_SCHEMAS)).toContain(json);
    });

    it('validates options before trusting origin metadata and report before publishing readiness', () => {
        expect(parseNativeConversationImportOptions(origin)).toEqual(origin);
        expect(parseNativeConversationImportReport(report)).toEqual(report);
        expect(() => parseNativeConversationImportOptions({ ...origin, model: undefined })).toThrow(/preflight/);
        expect(() => parseNativeConversationImportOptions({ ...origin, tool_definitions: [{}] })).toThrow(/schema/);
        expect(() => parseNativeConversationImportReport({ ...report, readiness: 'ready' })).toThrow(/schema/);
    });

    it('checks result document-wide reference semantics beyond its shape schema', () => {
        const document = emptyDocument();
        document.context.entries.push({ id: 'dangling', type: 'source_turn', turn_id: 'missing' });
        expect(NativeConversationImportResultSchema.safeParse({ document, report }).success).toBe(true);
        expect(() => parseNativeConversationImportResult({ document, report })).toThrow(/document validation/);
        const valid = { document: emptyDocument(), report };
        expect(parseNativeConversationImportResult(valid)).toEqual(valid);
    });
});

describe('shared binary-aware native input preflight', () => {
    it('preserves ordinary JSON walker results and does not widen the JSON boundary', () => {
        const input = { messages: [{ role: 'user', content: ['text', 0, null] }] };
        expect(preflightNativeConversationImportInput(input)).toEqual(preflightJsonInput(input));
        const bytes = new Uint8Array([97, 98, 99]);
        expect(preflightNativeConversationImportInput({ bytes }).success).toBe(true);
        expect(preflightJsonInput({ bytes }).success).toBe(false);
        expect(preflightNativeConversationImportInput({ bytes }, { max_bytes: 20 }).success).toBe(false);
        expect(preflightNativeConversationImportInput({ bytes }, { max_string_bytes: 4 }).success).toBe(false);
    });

    it('reports cycles, symbols, accessors, unsupported prototypes and sparse arrays without invoking getters', () => {
        const cycle: { self?: unknown } = {};
        cycle.self = cycle;
        const symbols = { [Symbol('hidden')]: 1 };
        let accessed = false;
        const accessor = Object.defineProperty({}, 'unsafe', {
            enumerable: true,
            get() {
                accessed = true;
                return 1;
            },
        });
        const reserved = Object.fromEntries([['__proto__', 1]]);
        for (const [input, code] of [
            [cycle, 'JSON_CYCLE'],
            [symbols, 'JSON_SYMBOL_KEY'],
            [accessor, 'JSON_ACCESSOR_PROPERTY'],
            [new Date(), 'JSON_NON_PLAIN_OBJECT'],
            [new Array(1), 'JSON_SPARSE_ARRAY'],
            [reserved, 'JSON_RESERVED_PROPERTY_KEY'],
            [-0, 'JSON_NEGATIVE_ZERO'],
        ] as const) {
            const native = preflightNativeConversationImportInput(input);
            expect(native).toEqual(preflightJsonInput(input));
            expect(native.diagnostics.map((entry) => entry.code)).toContain(code);
        }
        expect(accessed).toBe(false);
    });

    it('stops at bounded object/array/node/depth limits rather than pushing the whole source', () => {
        expect(
            preflightNativeConversationImportInput({ a: 1, b: 2 }, { max_object_properties: 1 }).diagnostics[0]?.code,
        ).toBe('JSON_MAX_OBJECT_PROPERTIES');
        expect(preflightNativeConversationImportInput([1, 2], { max_array_length: 1 }).diagnostics[0]?.code).toBe(
            'JSON_MAX_ARRAY_LENGTH',
        );
        expect(preflightNativeConversationImportInput([1, 2], { max_nodes: 1 }).diagnostics[0]?.code).toBe(
            'JSON_MAX_NODES',
        );
        expect(preflightNativeConversationImportInput({ a: { b: 1 } }, { max_depth: 1 }).diagnostics[0]?.code).toBe(
            'JSON_MAX_DEPTH',
        );
    });

    it('treats registered byte views as atomic byte content without interpreting annotations', () => {
        class CustomBytes extends Uint8Array {}
        const symbols = new Uint8Array([1]);
        Object.defineProperty(symbols, Symbol('annotation'), { value: 1 });
        expect(preflightNativeConversationImportInput(new CustomBytes([97, 98, 99])).success).toBe(false);
        expect(
            preflightNativeConversationImportInput(new CustomBytes([97, 98, 99]), {}, [CustomBytes.prototype]).success,
        ).toBe(true);
        const shadow = new Uint8Array([1]);
        Object.defineProperty(shadow, 'byteLength', {
            get() {
                throw new Error('must not invoke');
            },
        });
        expect(preflightNativeConversationImportInput(symbols).success).toBe(true);
        const decorated = new Uint8Array([1]);
        Object.defineProperty(decorated, 'annotation', { value: 'not byte data' });
        expect(preflightNativeConversationImportInput(decorated).success).toBe(true);
        expect([...copyNativeConversationByteView(decorated)]).toEqual([1]);
        for (const input of [new CustomBytes([1]), shadow]) {
            expect(preflightNativeConversationImportInput(input).diagnostics[0]?.code).toBe('JSON_NON_PLAIN_OBJECT');
        }
    });
});

describe('byte views reject conversion-visible shadow fields', () => {
    it.each(['length', 'buffer', 'byteOffset', 'byteLength'])('does not invoke an own %s accessor', (key) => {
        const bytes = new Uint8Array([1]);
        let accessed = false;
        Object.defineProperty(bytes, key, {
            get() {
                accessed = true;
                return 0;
            },
        });
        expect(preflightNativeConversationImportInput(bytes).success).toBe(false);
        expect(accessed).toBe(false);
    });
});
