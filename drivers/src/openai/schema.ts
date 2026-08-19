import type { JSONSchema, JSONSchemaTypeName } from '@llumiverse/core';
import type { JSONSchema as OpenAIJSONSchema } from 'openai/lib/jsonschema.js';
import { forEachJSONSchemaChild, toStrictJsonSchema } from 'openai/lib/transform.js';

export interface OpenAISchemaFormatResult {
    schema: JSONSchema;
    strict: boolean;
}

function hasSchemaType(schema: JSONSchema, type: JSONSchemaTypeName): boolean {
    return Array.isArray(schema.type) ? schema.type.includes(type) : schema.type === type;
}

export function formatOpenAISchema(schema: JSONSchema): OpenAISchemaFormatResult {
    try {
        assertOpenAIStrictSchemaContract(schema);
        const strictSchema = toStrictJsonSchema(schema as OpenAIJSONSchema) as JSONSchema;
        removeDefaults(strictSchema);
        return { schema: strictSchema, strict: true };
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

function assertOpenAIStrictSchemaContract(schema: JSONSchema): void {
    const visited = new Set<object>();
    const visit = (candidate: unknown): void => {
        if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate) || visited.has(candidate)) return;
        visited.add(candidate);

        const candidateSchema = candidate as JSONSchema;
        if (hasSchemaType(candidateSchema, 'object') || candidateSchema.properties) {
            const properties = candidateSchema.properties ?? {};
            const propertyNames = Object.keys(properties);
            const required = candidateSchema.required ?? [];
            if (
                candidateSchema.additionalProperties !== false ||
                propertyNames.length === 0 ||
                required.length !== propertyNames.length ||
                required.some((name) => !propertyNames.includes(name))
            ) {
                throw new Error('Schema changes object optionality or additional properties in OpenAI strict mode');
            }
        }

        forEachJSONSchemaChild(candidate as OpenAIJSONSchema, [], (child) => visit(child));
    };

    visit(schema);
}

function removeDefaults(schema: JSONSchema): void {
    const visited = new Set<object>();
    const visit = (candidate: unknown): void => {
        if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate) || visited.has(candidate)) return;
        visited.add(candidate);
        const candidateSchema = candidate as JSONSchema;
        delete candidateSchema.default;
        forEachJSONSchemaChild(candidate as OpenAIJSONSchema, [], (child) => visit(child));
    };

    visit(schema);
}
