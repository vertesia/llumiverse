import { boundConversationDiagnostic, diagnosticPointer } from './diagnostics.js';
import type { JsonPreflightDiagnostic } from './types.js';

export interface JsonInputLimits {
    max_depth: number;
    max_nodes: number;
    max_bytes: number;
    max_string_bytes: number;
    max_array_length: number;
    max_object_properties: number;
    max_diagnostics: number;
}

export const DEFAULT_JSON_INPUT_LIMITS: Readonly<JsonInputLimits> = Object.freeze({
    max_depth: 128,
    max_nodes: 250_000,
    max_bytes: 32 * 1024 * 1024,
    max_string_bytes: 32 * 1024 * 1024,
    max_array_length: 100_000,
    max_object_properties: 100_000,
    max_diagnostics: 32,
});

export interface JsonPreflightSuccess {
    success: true;
    diagnostics: [];
    bytes: number;
    nodes: number;
}

export interface JsonPreflightFailure {
    success: false;
    diagnostics: JsonPreflightDiagnostic[];
    bytes: number;
    nodes: number;
}

export type JsonPreflightResult = JsonPreflightSuccess | JsonPreflightFailure;

type PathSegment = string | number;

interface EnterFrame {
    type: 'enter';
    value: unknown;
    path: PathSegment[];
    depth: number;
}

interface ExitFrame {
    type: 'exit';
    value: object;
}

interface ArrayChildFrame {
    type: 'array_child';
    value: unknown[];
    index: number;
    path: PathSegment[];
    depth: number;
}

interface ObjectChildFrame {
    type: 'object_child';
    value: Record<string, unknown>;
    keys: string[];
    index: number;
    path: PathSegment[];
    depth: number;
}

type TraversalFrame = EnterFrame | ExitFrame | ArrayChildFrame | ObjectChildFrame;

function toPointer(path: PathSegment[]): string {
    return diagnosticPointer(path);
}

function serializedStringBytes(value: string, stopAfter: number): number {
    if (value.length + 2 > stopAfter) {
        return stopAfter + 1;
    }
    let bytes = 2;
    for (let index = 0; index < value.length; index += 1) {
        const codeUnit = value.charCodeAt(index);
        if (codeUnit === 0x22 || codeUnit === 0x5c || codeUnit === 0x08 || codeUnit === 0x09) {
            bytes += 2;
        } else if (codeUnit === 0x0a || codeUnit === 0x0c || codeUnit === 0x0d) {
            bytes += 2;
        } else if (codeUnit <= 0x1f) {
            bytes += 6;
        } else if (codeUnit <= 0x7f) {
            bytes += 1;
        } else if (codeUnit <= 0x7ff) {
            bytes += 2;
        } else if (codeUnit >= 0xd800 && codeUnit <= 0xdbff) {
            const next = value.charCodeAt(index + 1);
            if (next >= 0xdc00 && next <= 0xdfff) {
                bytes += 4;
                index += 1;
            } else {
                bytes += 6;
            }
        } else if (codeUnit >= 0xdc00 && codeUnit <= 0xdfff) {
            bytes += 6;
        } else {
            bytes += 3;
        }
        if (bytes > stopAfter) {
            return stopAfter + 1;
        }
    }
    return bytes;
}

function scalarBytes(value: number | boolean | null): number {
    if (value === null) {
        return 4;
    }
    if (typeof value === 'boolean') {
        return value ? 4 : 5;
    }
    return String(value).length;
}

function isArrayIndex(key: string, length: number): boolean {
    if (key === '') {
        return false;
    }
    const index = Number(key);
    return Number.isSafeInteger(index) && index >= 0 && index < length && String(index) === key;
}

function resolveLimits(overrides: Partial<JsonInputLimits>): JsonInputLimits {
    const limits = { ...DEFAULT_JSON_INPUT_LIMITS, ...overrides };
    for (const [name, value] of Object.entries(limits)) {
        const allowsZero = name === 'max_depth';
        if (!Number.isSafeInteger(value) || value < (allowsZero ? 0 : 1)) {
            throw new RangeError(`${name} must be a ${allowsZero ? 'nonnegative' : 'positive'} safe integer`);
        }
    }
    return limits;
}

export function preflightJsonInput(input: unknown, limitOverrides: Partial<JsonInputLimits> = {}): JsonPreflightResult {
    const limits = resolveLimits(limitOverrides);
    const diagnostics: JsonPreflightDiagnostic[] = [];
    const activeObjects = new WeakMap<object, string>();
    const stack: TraversalFrame[] = [{ type: 'enter', value: input, path: [], depth: 0 }];
    let bytes = 0;
    let nodes = 0;
    let stopped = false;

    const addDiagnostic = (diagnostic: JsonPreflightDiagnostic): void => {
        if (diagnostics.length >= limits.max_diagnostics) {
            stopped = true;
            return;
        }
        diagnostics.push(boundConversationDiagnostic(diagnostic));
        if (diagnostics.length >= limits.max_diagnostics) {
            stopped = true;
        }
    };

    const addBytes = (added: number, path: PathSegment[]): boolean => {
        const exceedsLimit = added > limits.max_bytes - bytes;
        bytes = Math.min(Number.MAX_SAFE_INTEGER, bytes + added);
        if (!exceedsLimit) {
            return true;
        }
        addDiagnostic({
            code: 'JSON_MAX_BYTES',
            stage: 'preflight',
            path: toPointer(path),
            message: 'Serialized JSON size exceeds the configured limit',
            limit: limits.max_bytes,
            observed: bytes,
        });
        stopped = true;
        return false;
    };

    while (stack.length > 0 && !stopped) {
        const frame = stack.pop();
        if (frame === undefined) {
            break;
        }
        if (frame.type === 'exit') {
            activeObjects.delete(frame.value);
            continue;
        }
        if (frame.type === 'array_child') {
            if (frame.index >= frame.value.length) {
                continue;
            }
            stack.push({ ...frame, index: frame.index + 1 });
            const descriptor = Object.getOwnPropertyDescriptor(frame.value, String(frame.index));
            const childPath = [...frame.path, frame.index];
            if (descriptor === undefined) {
                addDiagnostic({
                    code: 'JSON_SPARSE_ARRAY',
                    stage: 'preflight',
                    path: toPointer(childPath),
                    message: 'JSON arrays cannot contain holes',
                });
            } else if ('get' in descriptor || 'set' in descriptor) {
                addDiagnostic({
                    code: 'JSON_ACCESSOR_PROPERTY',
                    stage: 'preflight',
                    path: toPointer(childPath),
                    message: 'JSON input cannot contain accessor properties',
                });
            } else if (!descriptor.enumerable) {
                addDiagnostic({
                    code: 'JSON_NON_ENUMERABLE_PROPERTY',
                    stage: 'preflight',
                    path: toPointer(childPath),
                    message: 'JSON input cannot contain non-enumerable data properties',
                });
            } else {
                stack.push({ type: 'enter', value: descriptor.value, path: childPath, depth: frame.depth + 1 });
            }
            continue;
        }
        if (frame.type === 'object_child') {
            if (frame.index >= frame.keys.length) {
                continue;
            }
            stack.push({ ...frame, index: frame.index + 1 });
            const key = frame.keys[frame.index];
            const childPath = [...frame.path, key];
            const remainingBytes = Math.max(0, limits.max_bytes - bytes);
            const keyBytes = serializedStringBytes(key, Math.min(limits.max_string_bytes, remainingBytes));
            if (keyBytes > limits.max_string_bytes) {
                addDiagnostic({
                    code: 'JSON_MAX_STRING_BYTES',
                    stage: 'preflight',
                    path: toPointer(childPath),
                    message: 'Serialized JSON property name exceeds the configured per-string limit',
                    limit: limits.max_string_bytes,
                    observed: keyBytes,
                });
            }
            if (stopped || !addBytes(keyBytes, childPath)) {
                continue;
            }
            const descriptor = Object.getOwnPropertyDescriptor(frame.value, key);
            if (descriptor === undefined) {
                continue;
            }
            if ('get' in descriptor || 'set' in descriptor) {
                addDiagnostic({
                    code: 'JSON_ACCESSOR_PROPERTY',
                    stage: 'preflight',
                    path: toPointer(childPath),
                    message: 'JSON input cannot contain accessor properties',
                });
            } else if (!descriptor.enumerable) {
                addDiagnostic({
                    code: 'JSON_NON_ENUMERABLE_PROPERTY',
                    stage: 'preflight',
                    path: toPointer(childPath),
                    message: 'JSON input cannot contain non-enumerable data properties',
                });
            } else {
                stack.push({ type: 'enter', value: descriptor.value, path: childPath, depth: frame.depth + 1 });
            }
            continue;
        }

        const { value, path, depth } = frame;
        const pointer = toPointer(path);
        nodes += 1;
        if (nodes > limits.max_nodes) {
            addDiagnostic({
                code: 'JSON_MAX_NODES',
                stage: 'preflight',
                path: pointer,
                message: 'JSON node count exceeds the configured limit',
                limit: limits.max_nodes,
                observed: nodes,
            });
            stopped = true;
            continue;
        }
        if (depth > limits.max_depth) {
            addDiagnostic({
                code: 'JSON_MAX_DEPTH',
                stage: 'preflight',
                path: pointer,
                message: 'JSON nesting depth exceeds the configured limit',
                limit: limits.max_depth,
                observed: depth,
            });
            continue;
        }

        if (value === null || typeof value === 'boolean') {
            addBytes(scalarBytes(value), path);
            continue;
        }
        if (typeof value === 'string') {
            const remainingBytes = Math.max(0, limits.max_bytes - bytes);
            const valueBytes = serializedStringBytes(value, Math.min(limits.max_string_bytes, remainingBytes));
            if (valueBytes > limits.max_string_bytes) {
                addDiagnostic({
                    code: 'JSON_MAX_STRING_BYTES',
                    stage: 'preflight',
                    path: pointer,
                    message: 'Serialized JSON string exceeds the configured per-string limit',
                    limit: limits.max_string_bytes,
                    observed: valueBytes,
                });
            }
            if (!stopped) {
                addBytes(valueBytes, path);
            }
            continue;
        }
        if (typeof value === 'number') {
            if (!Number.isFinite(value)) {
                addDiagnostic({
                    code: 'JSON_NON_FINITE_NUMBER',
                    stage: 'preflight',
                    path: pointer,
                    message: 'JSON numbers must be finite',
                });
            } else if (Object.is(value, -0)) {
                addDiagnostic({
                    code: 'JSON_NEGATIVE_ZERO',
                    stage: 'preflight',
                    path: pointer,
                    message: 'Negative zero is rejected because JSON serialization normalizes it to zero',
                });
            } else {
                addBytes(scalarBytes(value), path);
            }
            continue;
        }
        if (typeof value !== 'object') {
            addDiagnostic({
                code: 'JSON_UNSUPPORTED_TYPE',
                stage: 'preflight',
                path: pointer,
                message: `JSON does not support values of type ${typeof value}`,
            });
            continue;
        }

        const priorPath = activeObjects.get(value);
        if (priorPath !== undefined) {
            addDiagnostic({
                code: 'JSON_CYCLE',
                stage: 'preflight',
                path: pointer,
                related_paths: [priorPath],
                message: 'JSON input contains a cycle',
            });
            continue;
        }

        const isArray = Array.isArray(value);
        const prototype = Object.getPrototypeOf(value);
        if (
            (isArray && prototype !== Array.prototype) ||
            (!isArray && prototype !== Object.prototype && prototype !== null)
        ) {
            addDiagnostic({
                code: 'JSON_NON_PLAIN_OBJECT',
                stage: 'preflight',
                path: pointer,
                message: `JSON ${isArray ? 'arrays' : 'objects'} must have a plain prototype`,
            });
            continue;
        }

        if (isArray && value.length > limits.max_array_length) {
            addDiagnostic({
                code: 'JSON_MAX_ARRAY_LENGTH',
                stage: 'preflight',
                path: pointer,
                message: 'JSON array length exceeds the configured limit',
                limit: limits.max_array_length,
                observed: value.length,
            });
            continue;
        }

        const ownKeys = Reflect.ownKeys(value);
        const maximumKeys = isArray ? limits.max_array_length + 1 : limits.max_object_properties;
        if (ownKeys.length > maximumKeys) {
            addDiagnostic({
                code: isArray ? 'JSON_MAX_ARRAY_LENGTH' : 'JSON_MAX_OBJECT_PROPERTIES',
                stage: 'preflight',
                path: pointer,
                message: `JSON ${isArray ? 'array' : 'object'} own-key count exceeds the configured limit`,
                limit: isArray ? limits.max_array_length : limits.max_object_properties,
                observed: ownKeys.length - (isArray ? 1 : 0),
            });
            continue;
        }

        const dataKeys: string[] = [];
        for (const key of ownKeys) {
            if (stopped || (isArray && key === 'length')) {
                continue;
            }
            if (typeof key === 'symbol') {
                addDiagnostic({
                    code: 'JSON_SYMBOL_KEY',
                    stage: 'preflight',
                    path: pointer,
                    message: 'JSON objects and arrays cannot contain symbol keys',
                });
            } else if (key === '__proto__') {
                addDiagnostic({
                    code: 'JSON_RESERVED_PROPERTY_KEY',
                    stage: 'preflight',
                    path: toPointer([...path, key]),
                    message:
                        'The experimental revision rejects __proto__ because Zod 4 does not validate that record key',
                });
            } else {
                dataKeys.push(key);
            }
        }
        if (stopped) {
            continue;
        }

        if (isArray) {
            if (dataKeys.some((key) => !isArrayIndex(key, value.length))) {
                addDiagnostic({
                    code: 'JSON_ARRAY_PROPERTY',
                    stage: 'preflight',
                    path: pointer,
                    message: 'JSON arrays cannot contain named own properties',
                });
                continue;
            }
            if (dataKeys.length !== value.length) {
                addDiagnostic({
                    code: 'JSON_SPARSE_ARRAY',
                    stage: 'preflight',
                    path: pointer,
                    message: 'JSON arrays cannot contain holes',
                });
                continue;
            }
            if (!addBytes(2 + Math.max(0, value.length - 1), path)) {
                continue;
            }
            activeObjects.set(value, pointer);
            stack.push({ type: 'exit', value });
            if (value.length > 0) {
                stack.push({ type: 'array_child', value, index: 0, path, depth });
            }
            continue;
        }

        if (!addBytes(2 + Math.max(0, dataKeys.length - 1) + dataKeys.length, path)) {
            continue;
        }
        activeObjects.set(value, pointer);
        stack.push({ type: 'exit', value });
        if (dataKeys.length > 0) {
            stack.push({
                type: 'object_child',
                value: value as Record<string, unknown>,
                keys: dataKeys,
                index: 0,
                path,
                depth,
            });
        }
    }

    if (diagnostics.length === 0) {
        return { success: true, diagnostics: [], bytes, nodes };
    }
    return { success: false, diagnostics, bytes, nodes };
}
