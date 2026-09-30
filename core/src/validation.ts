import type { CompletionResult, JSONValue, ResultValidationError } from '@llumiverse/common';
import { Ajv, type ValidateFunction } from 'ajv';
import addFormats from 'ajv-formats';
import { extractAndParseJSON, parseJSON } from './json.js';
import { resolveField } from './resolver.js';

const ajv = new Ajv({
    coerceTypes: 'array',
    allowDate: true,
    strict: false,
    useDefaults: true,
    removeAdditional: 'failing',
});
const exactAjv = new Ajv({
    coerceTypes: false,
    allowDate: true,
    strict: false,
});

// biome-ignore lint/suspicious/noTsIgnore: ajv-formats' runtime module.exports is the callable plugin, but its shipped .d.ts declares an ESM default that resolves to a non-callable namespace under module:nodenext; no cast-free import form works
// @ts-ignore - ajv-formats default export is not callable under module:nodenext ESM resolution
addFormats(ajv);
// biome-ignore lint/suspicious/noTsIgnore: see the matching import-interop explanation above
// @ts-ignore - ajv-formats default export is not callable under module:nodenext ESM resolution
addFormats(exactAjv);

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

/**
 * Compile `schema` against the shared Ajv instance without tripping over its `$id` registry.
 *
 * `ajv.compile()` registers a schema under its `$id` and caches compiled validators by object
 * identity. A stored result schema is deserialized into a *fresh* object on every execution, so the
 * second execution of any interaction whose schema declares an `$id` reached Ajv as an unknown
 * object claiming an already-taken id and threw `schema with key or id "..." already exists`. That
 * turned every such run into a `validation_error` on output the model had produced correctly.
 *
 * Dropping the previous registration first — rather than reusing the cached validator — keeps the
 * newest definition authoritative, because a schema can be edited between executions while keeping
 * its `$id`. `removeSchema` is a no-op when nothing is registered, and it also bounds the registry
 * at one entry per id instead of leaking one per execution.
 */
function compileSchema(schema: object, compiler = ajv): ValidateFunction {
    const id = (schema as { $id?: unknown }).$id;
    if (typeof id === 'string' && id) {
        compiler.removeSchema(id);
    }
    return compiler.compile(schema);
}

function validationErrorsMessage(validate: ValidateFunction): string {
    return (validate.errors ?? [])
        .map((error) => `${error.instancePath}: ${error.message}\n${JSON.stringify(error.params)}`)
        .join(',\n\n');
}

function getRequiredFields(schemaField: unknown): string[] {
    if (!schemaField || typeof schemaField !== 'object') {
        return [];
    }
    const required = (schemaField as Record<string, unknown>).required;
    return Array.isArray(required) ? required.filter((field): field is string => typeof field === 'string') : [];
}

function decodeJsonPointer(path: string): string[] {
    return path
        .split('/')
        .slice(1)
        .map((part) => part.replaceAll('~1', '/').replaceAll('~0', '~'));
}

function deleteField(object: JSONValue, path: string[]): boolean {
    if (path.length === 0) return false;

    let parent: unknown = object;
    for (const part of path.slice(0, -1)) {
        if (Array.isArray(parent)) {
            const index = Number(part);
            if (!Number.isSafeInteger(index) || index < 0 || index >= parent.length) return false;
            parent = parent[index];
        } else if (parent !== null && typeof parent === 'object' && Object.hasOwn(parent, part)) {
            parent = (parent as Record<string, unknown>)[part];
        } else {
            return false;
        }
    }

    const field = path[path.length - 1];
    if (parent === null || Array.isArray(parent) || typeof parent !== 'object' || !Object.hasOwn(parent, field)) {
        return false;
    }
    return delete (parent as Record<string, unknown>)[field];
}

export class ValidationError extends Error implements ResultValidationError {
    constructor(
        public code: 'validation_error' | 'json_error',
        message: string,
    ) {
        super(message);
        this.name = 'ValidationError';
    }
}

export interface CanonicalStructuredOutput {
    value: JSONValue;
    /** Exact answer-text partitions which produced `value`, in provider order. */
    source_texts: string[];
}

export type CompletionResultNormalization =
    | {
          status: 'valid';
          result: CompletionResult[];
          structured_output: CanonicalStructuredOutput;
      }
    | {
          status: 'invalid';
          result: CompletionResult[];
          error: ValidationError;
      };

function parseCompletionAsJson(data: CompletionResult[]): CanonicalStructuredOutput {
    const sourceTexts = data.flatMap((part) => (part.type === 'text' ? [part.value] : []));
    if (sourceTexts.length === 0) {
        throw new ValidationError('json_error', 'No JSON compatible response found in completion result');
    }
    const source = sourceTexts.join('').trim();
    let lastError: ValidationError | undefined;
    try {
        return { value: parseStructuredOutputText(source), source_texts: sourceTexts };
    } catch (error: unknown) {
        lastError = new ValidationError('json_error', errorMessage(error));
    }

    // Preserve the historical fallback for providers which return multiple independent
    // alternatives as text results. Streaming and multipart responses are handled by the
    // joined parse above, while this still accepts the first independently valid value.
    for (const text of sourceTexts) {
        try {
            return { value: parseStructuredOutputText(text.trim()), source_texts: sourceTexts };
        } catch (error: unknown) {
            lastError = new ValidationError('json_error', errorMessage(error));
        }
    }
    throw lastError;
}

function fencedStructuredOutputBody(text: string): string | undefined {
    if (!text.startsWith('```') || !text.endsWith('```') || text.length < 6) return undefined;
    let start = 3;
    let end = text.length - 3;
    if (text.slice(start, start + 4).toLowerCase() === 'json') start += 4;
    while (start < end && text[start]?.trim() === '') start += 1;
    while (end > start && text[end - 1]?.trim() === '') end -= 1;
    return text.slice(start, end);
}

function parseStructuredOutputText(text: string): JSONValue {
    try {
        return JSON.parse(text) as JSONValue;
    } catch (exactError: unknown) {
        const fenced = fencedStructuredOutputBody(text);
        if (fenced !== undefined) return parseJSON(fenced);
        try {
            return extractAndParseJSON(text);
        } catch {
            try {
                return parseJSON(text);
            } catch {
                throw exactError;
            }
        }
    }
}

function normalizeValidatedResult(data: CompletionResult[], json: JSONValue): CompletionResult[] {
    const validatedResult: CompletionResult = { type: 'json', value: json };
    const firstContentIndex = data.findIndex((part) => part.type === 'text' || part.type === 'json');
    if (firstContentIndex === -1) return [...data, validatedResult];

    return data.reduce<CompletionResult[]>((results, part, index) => {
        if (part.type !== 'text' && part.type !== 'json') {
            results.push(part);
        } else if (index === firstContentIndex) {
            results.push(validatedResult);
        }
        return results;
    }, []);
}

export function normalizeCompletionResult(data: CompletionResult[], schema: object): CompletionResultNormalization {
    let json: JSONValue;
    let sourceTexts: string[] = [];
    if (Array.isArray(data)) {
        const jsonResults = data.filter((r) => r.type === 'json');
        if (jsonResults.length > 0) {
            json = structuredClone(jsonResults[0].value);
        } else {
            try {
                const parsed = parseCompletionAsJson(data);
                json = parsed.value;
                sourceTexts = parsed.source_texts;
            } catch (error: unknown) {
                return {
                    status: 'invalid',
                    result: data,
                    error:
                        error instanceof ValidationError
                            ? error
                            : new ValidationError('json_error', errorMessage(error)),
                };
            }
        }
    } else {
        return {
            status: 'invalid',
            result: data,
            error: new ValidationError('validation_error', 'Data to validate must be an array'),
        };
    }

    let validate: ValidateFunction;
    try {
        validate = compileSchema(schema);
    } catch (error: unknown) {
        return {
            status: 'invalid',
            result: data,
            error: new ValidationError('validation_error', errorMessage(error)),
        };
    }
    let valid = validate(json);
    const removedEmptyDatePointers = new Set<string>();

    while (!valid && validate.errors) {
        const errors = [];
        let removedEmptyOptionalDate = false;

        for (const e of validate.errors) {
            const path = decodeJsonPointer(e.instancePath);
            const value = resolveField(json, path);
            const schemaPath = e.schemaPath.split('/').slice(1);
            const schemaFieldFormat = resolveField(schema, schemaPath);
            const schemaField = resolveField(schema, schemaPath.slice(0, -3));

            // Preserve the historical acceptance of an omitted optional date represented as an
            // empty value, but remove that value from the normalized clone. A result declared
            // valid must itself satisfy the schema; retaining the invalid date would violate that
            // invariant even though the legacy validator ignored the diagnostic.
            if (
                (value === '' || value === null) &&
                typeof schemaFieldFormat === 'string' &&
                ['date', 'date-time'].includes(schemaFieldFormat) &&
                !getRequiredFields(schemaField).includes(path[path.length - 1])
            ) {
                if (removedEmptyDatePointers.has(e.instancePath) || !deleteField(json, path)) {
                    errors.push(e);
                } else {
                    removedEmptyDatePointers.add(e.instancePath);
                    removedEmptyOptionalDate = true;
                }
            } else {
                errors.push(e);
            }
        }

        //console.log("Errors", errors)
        if (errors.length > 0) {
            const errorsMessage = errors
                .map((e) => `${e.instancePath}: ${e.message}\n${JSON.stringify(e.params)}`)
                .join(',\n\n');
            return {
                status: 'invalid',
                result: data,
                error: new ValidationError('validation_error', errorsMessage),
            };
        }

        if (!removedEmptyOptionalDate) break;
        valid = validate(json);
    }

    // Ajv cannot replace a caller's root primitive when coercion is enabled. It can therefore
    // return true for (for example) the string "42" against a number schema while `json` remains a
    // string. Revalidate the post-normalization value without coercion so the canonical value we
    // persist always satisfies the declared result schema. Nested coercions/defaults already
    // applied by the first pass remain visible on the cloned value and continue to validate here.
    try {
        const exactValidate = compileSchema(schema, exactAjv);
        if (!exactValidate(json)) {
            return {
                status: 'invalid',
                result: data,
                error: new ValidationError('validation_error', validationErrorsMessage(exactValidate)),
            };
        }
    } catch (error: unknown) {
        return {
            status: 'invalid',
            result: data,
            error: new ValidationError('validation_error', errorMessage(error)),
        };
    }

    return {
        status: 'valid',
        result: normalizeValidatedResult(data, json),
        structured_output: { value: json, source_texts: sourceTexts },
    };
}

export function validateResult(data: CompletionResult[], schema: object): CompletionResult[] {
    const normalized = normalizeCompletionResult(data, schema);
    if (normalized.status === 'invalid') throw normalized.error;
    return normalized.result;
}
