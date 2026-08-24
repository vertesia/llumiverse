import { JSON_SCHEMA_INSTRUCTION_PREFIX, type JSONSchema } from '@llumiverse/common';

export function getJSONSafetyNotice(schema: JSONSchema) {
    return `${JSON_SCHEMA_INSTRUCTION_PREFIX}\n${JSON.stringify(schema, undefined, 2)}`;
}
