import { JSONSchema } from "../types.js";

export function getJSONSafetyNotice(schema: JSONSchema) {
    return "The answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(schema, undefined, 2);
}
