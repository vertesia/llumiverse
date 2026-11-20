import type { Completion, JSONSchema, JSONValue, PromptSegment } from '@llumiverse/common';

/** The text of a fake answer to a prompt without a result schema. */
export const FAKE_ANSWER_TEXT = 'Fake answer of the test driver: no model was called.';

// A schema nested deeper than this is cut with `null`: a recursive `$ref` never ends otherwise.
const MAX_DEPTH = 32;

/**
 * Builds the completion of a fake model: the smallest value that matches the result schema, or a fixed text
 * without a schema. Its token usage is estimated from the size of the prompt and of the answer, at 4 characters
 * per token, so that the metering of the caller records a call of a realistic size.
 */
export function createFakeAnswer(prompt: PromptSegment[], schema: JSONSchema | undefined): Completion {
    const value = schema ? fakeValue(schema) : FAKE_ANSWER_TEXT;
    const promptTokens = estimateTokens(JSON.stringify(prompt.map((segment) => segment.content)));
    const resultTokens = estimateTokens(typeof value === 'string' ? value : JSON.stringify(value));
    return {
        result: schema ? [{ type: 'json', value }] : [{ type: 'text', value: FAKE_ANSWER_TEXT }],
        token_usage: { prompt: promptTokens, result: resultTokens, total: promptTokens + resultTokens },
        finish_reason: 'stop',
    };
}

function estimateTokens(text: string): number {
    return Math.max(1, Math.ceil(text.length / 4));
}

/**
 * Returns the smallest value that matches a JSON schema: the required properties of an object, `minItems` items
 * of an array, the first value of an `enum`, the lower bound of a number, `false`, etc. It doesn't resolve `$ref`
 * and doesn't follow `pattern`: the caller's validation reports such a mismatch as for a real model.
 */
export function fakeValue(schema: unknown, depth = 0): JSONValue {
    if (!isSchema(schema) || depth > MAX_DEPTH) {
        return null;
    }
    if ('const' in schema) {
        return schema.const as JSONValue;
    }
    if (schema.default !== undefined) {
        return schema.default as JSONValue;
    }
    if (Array.isArray(schema.enum) && schema.enum.length > 0) {
        return schema.enum[0] as JSONValue;
    }
    for (const key of ['anyOf', 'oneOf'] as const) {
        const branches = schema[key];
        if (Array.isArray(branches) && branches.length > 0) {
            return fakeValue(branches[0], depth + 1);
        }
    }
    if (Array.isArray(schema.allOf) && schema.allOf.length > 0) {
        return fakeAllOf(schema.allOf, depth);
    }
    switch (schemaType(schema)) {
        case 'object':
            return fakeObject(schema, depth);
        case 'array':
            return fakeArray(schema, depth);
        case 'string':
            return fakeString(schema);
        case 'number':
            return fakeNumber(schema, false);
        case 'integer':
            return fakeNumber(schema, true);
        case 'boolean':
            return false;
        default:
            return null;
    }
}

function isSchema(value: unknown): value is JSONSchema {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

// The first type other than `null` of a list of types. Without a type, `properties` means an object and `items`
// an array.
function schemaType(schema: JSONSchema): string | undefined {
    const type: unknown = schema.type;
    if (Array.isArray(type)) {
        return type.find((t) => t !== 'null') ?? type[0];
    }
    if (typeof type === 'string') {
        return type;
    }
    if (schema.properties) {
        return 'object';
    }
    if (schema.items || schema.prefixItems) {
        return 'array';
    }
    return undefined;
}

function fakeObject(schema: JSONSchema, depth: number): JSONValue {
    const value: { [key: string]: JSONValue } = {};
    for (const name of schema.required ?? []) {
        value[name] = fakeValue(schema.properties?.[name] ?? {}, depth + 1);
    }
    return value;
}

function fakeArray(schema: JSONSchema, depth: number): JSONValue {
    const prefixItems = Array.isArray(schema.prefixItems) ? schema.prefixItems : [];
    const minItems = typeof schema.minItems === 'number' ? schema.minItems : 0;
    const items: JSONValue[] = prefixItems.map((item) => fakeValue(item, depth + 1));
    while (items.length < minItems) {
        items.push(fakeValue(schema.items ?? {}, depth + 1));
    }
    return items;
}

// A value of the formats that the validation of llumiverse checks (ajv-formats).
const FAKE_FORMATS: Record<string, string> = {
    'date-time': '1970-01-01T00:00:00.000Z',
    date: '1970-01-01',
    time: '00:00:00Z',
    email: 'fake@example.com',
    uri: 'https://example.com/',
    url: 'https://example.com/',
    uuid: '00000000-0000-0000-0000-000000000000',
};

function fakeString(schema: JSONSchema): string {
    const formatted = schema.format ? FAKE_FORMATS[schema.format] : undefined;
    if (formatted !== undefined) {
        return formatted;
    }
    const minLength = typeof schema.minLength === 'number' ? schema.minLength : 0;
    return 'x'.repeat(minLength);
}

function fakeNumber(schema: JSONSchema, integer: boolean): number {
    let value = 0;
    if (typeof schema.minimum === 'number') {
        value = schema.minimum;
    } else if (typeof schema.exclusiveMinimum === 'number') {
        value = schema.exclusiveMinimum + 1;
    }
    if (typeof schema.maximum === 'number' && value > schema.maximum) {
        value = schema.maximum;
    }
    return integer ? Math.ceil(value) : value;
}

// The objects of each branch merged into one, since every branch must match. A branch that isn't an object
// gives its own value.
function fakeAllOf(branches: unknown[], depth: number): JSONValue {
    const values = branches.map((branch) => fakeValue(branch, depth + 1));
    if (values.every((v) => typeof v === 'object' && v !== null && !Array.isArray(v))) {
        return Object.assign({}, ...values) as JSONValue;
    }
    return values[0];
}
