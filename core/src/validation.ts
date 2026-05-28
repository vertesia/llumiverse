import type { CompletionResult, JSONValue, ResultValidationError } from '@llumiverse/common';
import { Ajv } from 'ajv';
import addFormats from 'ajv-formats';
import { extractAndParseJSON } from './json.js';
import { resolveField } from './resolver.js';

const ajv = new Ajv({
    coerceTypes: 'array',
    allowDate: true,
    strict: false,
    useDefaults: true,
    removeAdditional: 'failing',
});

// biome-ignore lint/suspicious/noTsIgnore: ajv-formats default export inconsistently callable under ESM/CJS interop
// @ts-ignore - ajv-formats default export is not callable under one of the TS configs in this monorepo
addFormats(ajv);

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
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
    ) {
        super(message);
        this.name = 'ValidationError';
    }
}

function parseCompletionAsJson(data: CompletionResult[]) {
    let lastError: ValidationError | undefined;
    for (const part of data) {
        if (part.type === 'text') {
            const text = part.value.trim();
            try {
                return extractAndParseJSON(text);
            } catch (error: unknown) {
                lastError = new ValidationError('json_error', errorMessage(error));
            }
        }
    }
    if (!lastError) {
        lastError = new ValidationError('json_error', 'No JSON compatible response found in completion result');
    }
    throw lastError;
}

export function validateResult(data: CompletionResult[], schema: object): CompletionResult[] {
    let json: JSONValue;
    if (Array.isArray(data)) {
        const jsonResults = data.filter((r) => r.type === 'json');
        if (jsonResults.length > 0) {
            json = jsonResults[0].value;
        } else {
            try {
                json = parseCompletionAsJson(data);
            } catch (error: unknown) {
                throw new ValidationError('json_error', errorMessage(error));
            }
        }
    } else {
        throw new Error('Data to validate must be an array');
    }

    const validate = ajv.compile(schema);
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

    return [{ type: 'json', value: json }];
}
