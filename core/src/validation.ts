import type { CompletionResult, JSONValue, ResultValidationError } from '@llumiverse/common';
import { Ajv, type ValidateFunction } from 'ajv';
import addFormats from 'ajv-formats';
import { type JSONOutputParseOptions, parseJSONOutput } from './json.js';
import { resolveField } from './resolver.js';

const ajv = new Ajv({
    coerceTypes: 'array',
    allowDate: true,
    strict: false,
    useDefaults: true,
    removeAdditional: 'failing',
});

// biome-ignore lint/suspicious/noTsIgnore: ajv-formats' runtime module.exports is the callable plugin, but its shipped .d.ts declares an ESM default that resolves to a non-callable namespace under module:nodenext; no cast-free import form works
// @ts-ignore - ajv-formats default export is not callable under module:nodenext ESM resolution
addFormats(ajv);

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
function compileSchema(schema: object): ValidateFunction {
    const id = (schema as { $id?: unknown }).$id;
    if (typeof id === 'string' && id) {
        ajv.removeSchema(id);
    }
    return ajv.compile(schema);
}

function getRequiredFields(schemaField: unknown): string[] {
    if (!schemaField || typeof schemaField !== 'object') {
        return [];
    }
    const required = (schemaField as Record<string, unknown>).required;
    return Array.isArray(required) ? required.filter((field): field is string => typeof field === 'string') : [];
}

export class ValidationError extends Error implements ResultValidationError {
    constructor(
        public code: 'validation_error' | 'json_error',
        message: string,
        options?: ErrorOptions,
    ) {
        super(message, options);
        this.name = 'ValidationError';
    }
}

export function validateResult(
    data: CompletionResult[],
    schema: object,
    options: boolean | JSONOutputParseOptions = true,
): CompletionResult[] {
    const parseOptions = typeof options === 'boolean' ? { allowRepair: options } : options;
    const jsonResults = data.filter((part) => part.type === 'json');
    let json: JSONValue | undefined;
    if (jsonResults.length > 0) {
        json = jsonResults[0].value;
    } else {
        const textParts = data.filter(
            (part): part is Extract<CompletionResult, { type: 'text' }> => part.type === 'text',
        );
        const text = textParts.map((part) => part.value).join('');
        if (!text.trim()) {
            throw new ValidationError('json_error', 'No JSON compatible response found in completion result');
        }
        // A complete joined answer outranks containers that are merely nested fragments of it.
        let joined: JSONValue | undefined;
        const fence = text.trim().match(/^```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```$/i);
        try {
            joined = JSON.parse(fence ? fence[1] : text);
        } catch {
            // Preserve independent complete answers before attempting repair of joined text.
        }
        // Preserve complete independent containers. Scalars can still be fragments (for example, 1 followed by 2).
        if (joined === undefined && textParts.length > 1) {
            for (const part of textParts) {
                try {
                    const complete = parseJSONOutput(part.value, { allowRepair: false });
                    if (complete !== null && typeof complete === 'object') {
                        json = complete;
                        break;
                    }
                } catch {
                    // This part is not a complete JSON value.
                }
            }
        }
        if (joined !== undefined) {
            json = joined;
            if (fence) parseOptions.onDiagnostic?.({ extracted: true, repaired: false, original_text: text });
        } else if (json !== undefined) {
            parseOptions.onDiagnostic?.({ extracted: true, repaired: false, original_text: text });
        } else {
            try {
                json = parseJSONOutput(text, parseOptions);
            } catch (error: unknown) {
                throw new ValidationError('json_error', errorMessage(error), { cause: error });
            }
        }
    }

    const validate = compileSchema(schema);
    const valid = validate(json);

    if (!valid && validate.errors) {
        const errors = [];

        for (const e of validate.errors) {
            const path = e.instancePath.split('/').slice(1);
            const value = resolveField(json, path);
            const schemaPath = e.schemaPath.split('/').slice(1);
            const schemaFieldFormat = resolveField(schema, schemaPath);
            const schemaField = resolveField(schema, schemaPath.slice(0, -3));

            //ignore date if empty or null
            if (
                !value &&
                typeof schemaFieldFormat === 'string' &&
                ['date', 'date-time'].includes(schemaFieldFormat) &&
                !getRequiredFields(schemaField).includes(path[path.length - 1])
            ) {
            } else {
                errors.push(e);
            }
        }

        //console.log("Errors", errors)
        if (errors.length > 0) {
            const errorsMessage = errors
                .map((e) => `${e.instancePath}: ${e.message}\n${JSON.stringify(e.params)}`)
                .join(',\n\n');
            throw new ValidationError('validation_error', errorsMessage);
        }
    }

    // Structured-output validation applies only to response content. Thoughts are
    // separate first-class results and must remain available to callers.
    const validatedResult: CompletionResult = { type: 'json', value: json };
    const firstContentIndex = data.findIndex((part) => part.type !== 'thoughts');
    if (firstContentIndex === -1) {
        return [validatedResult];
    }

    return data.reduce<CompletionResult[]>((results, part, index) => {
        if (part.type === 'thoughts') {
            results.push(part);
        } else if (index === firstContentIndex) {
            results.push(validatedResult);
        }
        return results;
    }, []);
}
