import type { JSONSchema, JSONSchemaTypeName } from '@llumiverse/core';

export interface OpenAISchemaFormatResult {
    schema: JSONSchema;
    strict: boolean;
}

const OPENAI_SUPPORTED_FORMATS = new Set([
    'date-time',
    'time',
    'date',
    'duration',
    'email',
    'hostname',
    'ipv4',
    'ipv6',
    'uuid',
]);

const OPENAI_SUPPORTED_TYPES = new Set<JSONSchemaTypeName>([
    'string',
    'number',
    'integer',
    'boolean',
    'object',
    'array',
    'null',
]);

function hasSchemaType(schema: JSONSchema, type: JSONSchemaTypeName): boolean {
    return Array.isArray(schema.type) ? schema.type.includes(type) : schema.type === type;
}

export function formatOpenAISchema(schema: JSONSchema): OpenAISchemaFormatResult {
    try {
        return { schema: openAISchemaFormat(schema), strict: true };
    } catch {
        return { schema: limitedSchemaFormat(schema), strict: false };
    }
}

// For strict mode false.
export function limitedSchemaFormat(schema: JSONSchema): JSONSchema {
    const formattedSchema: JSONSchema = { ...schema };

    // Defaults not supported.
    delete formattedSchema.default;

    // OpenAI requires type field even in non-strict mode.
    // If no type is specified, default to 'object' for properties with format/editor hints,
    // otherwise 'string' as a safe fallback.
    if (!formattedSchema.type && formattedSchema.description) {
        // Properties with format: "document" or editor hints are typically objects.
        if (formattedSchema.format === 'document' || formattedSchema.editor) {
            formattedSchema.type = 'object';
        } else {
            formattedSchema.type = 'string';
        }
    }

    if (formattedSchema.properties) {
        formattedSchema.properties = Object.fromEntries(
            Object.entries(formattedSchema.properties).map(([name, property]) => [name, limitedSchemaFormat(property)]),
        );
    }
    if (Array.isArray(formattedSchema.anyOf)) {
        formattedSchema.anyOf = formattedSchema.anyOf.map((variant) => limitedSchemaFormat(variant as JSONSchema));
    }
    if (formattedSchema.items && typeof formattedSchema.items === 'object') {
        formattedSchema.items = limitedSchemaFormat(formattedSchema.items);
    }
    if (formattedSchema.$defs && typeof formattedSchema.$defs === 'object') {
        formattedSchema.$defs = Object.fromEntries(
            Object.entries(formattedSchema.$defs as Record<string, JSONSchema>).map(([name, definition]) => [
                name,
                limitedSchemaFormat(definition),
            ]),
        );
    }

    return formattedSchema;
}

// For strict mode true.
export function openAISchemaFormat(schema: JSONSchema, nesting: number = 0): JSONSchema {
    if (nesting === 0) validateOpenAIStrictSchema(schema);

    const formattedSchema: JSONSchema = { ...schema };

    // Defaults not supported.
    delete formattedSchema.default;

    // Additional properties not supported, required to be set.
    if (hasSchemaType(formattedSchema, 'object')) {
        formattedSchema.additionalProperties = false;
    }

    if (formattedSchema.properties) {
        // Set all properties as required.
        formattedSchema.required = Object.keys(formattedSchema.properties);

        for (const propName of Object.keys(formattedSchema.properties)) {
            const property = formattedSchema.properties[propName];
            formattedSchema.properties[propName] = openAISchemaFormat(property, nesting + 1);
        }
    }
    if (formattedSchema.items && typeof formattedSchema.items === 'object') {
        formattedSchema.items = openAISchemaFormat(formattedSchema.items, nesting + 1);
    }
    if (Array.isArray(formattedSchema.anyOf)) {
        formattedSchema.anyOf = formattedSchema.anyOf.map((variant) =>
            openAISchemaFormat(variant as JSONSchema, nesting + 1),
        );
    }
    if (formattedSchema.$defs && typeof formattedSchema.$defs === 'object') {
        formattedSchema.$defs = Object.fromEntries(
            Object.entries(formattedSchema.$defs as Record<string, JSONSchema>).map(([name, definition]) => [
                name,
                openAISchemaFormat(definition, nesting + 1),
            ]),
        );
    }
    return formattedSchema;
}

function validateOpenAIStrictSchema(schema: JSONSchema, nesting: number = 0, root: boolean = true): void {
    if (nesting > 10) {
        throw new Error('OpenAI schema nesting too deep');
    }

    const allowedKeys = new Set([
        'type',
        'description',
        'properties',
        'items',
        'required',
        'additionalProperties',
        'enum',
        'anyOf',
        '$defs',
        '$ref',
        'pattern',
        'format',
        'multipleOf',
        'maximum',
        'exclusiveMaximum',
        'minimum',
        'exclusiveMinimum',
        'minItems',
        'maxItems',
    ]);

    for (const key of Object.keys(schema)) {
        if (key !== 'default' && !allowedKeys.has(key)) {
            throw new Error(`OpenAI strict mode does not support schema keyword '${key}'`);
        }
    }

    if (root && (!hasSchemaType(schema, 'object') || schema.anyOf)) {
        throw new Error('OpenAI strict mode requires the root schema to be an object without anyOf');
    }

    if (schema.$ref) {
        return;
    }

    const types = schema.type === undefined ? [] : Array.isArray(schema.type) ? schema.type : [schema.type];
    if (types.length === 0 && !schema.anyOf) {
        throw new Error('OpenAI strict mode requires a type for every schema');
    }
    if (types.some((type) => !OPENAI_SUPPORTED_TYPES.has(type))) {
        throw new Error('OpenAI strict mode does not support one or more schema types');
    }

    if (
        schema.format !== undefined &&
        (!hasSchemaType(schema, 'string') || !OPENAI_SUPPORTED_FORMATS.has(schema.format))
    ) {
        throw new Error(`OpenAI strict mode does not support format '${schema.format}'`);
    }

    if (schema.enum && Array.isArray(schema.enum)) {
        const values = schema.enum.map((value) => JSON.stringify(value));
        if (new Set(values).size !== values.length) {
            throw new Error('OpenAI strict mode does not support duplicate enum values');
        }
    }

    if (hasSchemaType(schema, 'object')) {
        const properties = schema.properties ?? {};
        if (Object.keys(properties).length === 0 || schema.additionalProperties !== false) {
            throw new Error('OpenAI strict mode requires non-empty objects with additionalProperties set to false');
        }
        const propertyNames = Object.keys(properties);
        const required = schema.required ?? [];
        if (required.length !== propertyNames.length || required.some((name) => !propertyNames.includes(name))) {
            throw new Error('OpenAI strict mode requires every object property to be required');
        }
        for (const property of Object.values(properties)) {
            validateOpenAIStrictSchema(property, nesting + 1, false);
        }
    }

    if (hasSchemaType(schema, 'array')) {
        if (!schema.items || typeof schema.items !== 'object') {
            throw new Error('OpenAI strict mode requires array items');
        }
        validateOpenAIStrictSchema(schema.items, nesting + 1, false);
    }

    if (Array.isArray(schema.anyOf)) {
        for (const variant of schema.anyOf) {
            validateOpenAIStrictSchema(variant as JSONSchema, nesting + 1, false);
        }
    }

    if (schema.$defs && typeof schema.$defs === 'object') {
        for (const definition of Object.values(schema.$defs as Record<string, JSONSchema>)) {
            validateOpenAIStrictSchema(definition, nesting + 1, false);
        }
    }
}
