import type { JSONSchema } from '@llumiverse/core';

export interface OpenAISchemaFormatResult {
    schema: JSONSchema;
    strict: boolean;
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

    if (formattedSchema?.properties) {
        // Process each property recursively.
        for (const propName of Object.keys(formattedSchema.properties)) {
            const property = formattedSchema.properties[propName];

            formattedSchema.properties[propName] = limitedSchemaFormat(property);

            if (property?.type === 'array' && property.items && property.items?.type === 'object') {
                formattedSchema.properties[propName] = {
                    ...property,
                    items: limitedSchemaFormat(property.items),
                };
            }
        }
    }

    return formattedSchema;
}

// For strict mode true.
export function openAISchemaFormat(schema: JSONSchema, nesting: number = 0): JSONSchema {
    if (nesting > 5) {
        throw new Error('OpenAI schema nesting too deep');
    }

    const formattedSchema: JSONSchema = { ...schema };

    // Defaults not supported.
    delete formattedSchema.default;

    // Additional properties not supported, required to be set.
    if (formattedSchema?.type === 'object') {
        formattedSchema.additionalProperties = false;
    }

    if (formattedSchema?.properties) {
        // Set all properties as required.
        formattedSchema.required = Object.keys(formattedSchema.properties);

        for (const propName of Object.keys(formattedSchema.properties)) {
            const property = formattedSchema.properties[propName];

            // OpenAI strict mode requires all properties to have a type.
            if (!property?.type) {
                throw new Error(`Property '${propName}' is missing required 'type' field for OpenAI strict mode`);
            }

            formattedSchema.properties[propName] = openAISchemaFormat(property, nesting + 1);

            if (property?.type === 'array' && property.items && property.items?.type === 'object') {
                formattedSchema.properties[propName] = {
                    ...property,
                    items: openAISchemaFormat(property.items, nesting + 1),
                };
            }
        }
    }
    if (
        formattedSchema?.type === 'object' &&
        (!formattedSchema?.properties || Object.keys(formattedSchema?.properties ?? {}).length === 0)
    ) {
        // If no properties are defined, then additionalProperties: true was set or the object would be empty.
        // OpenAI does not support this on structured output / strict mode.
        throw new Error('OpenAI does not support empty objects or objects with additionalProperties set to true');
    }
    return formattedSchema;
}
