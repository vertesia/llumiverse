import { JSONSchema } from "@llumiverse/common";

export function getJSONSafetyNotice(schema: JSONSchema) {
    return "The answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(schema, undefined, 2);
}
