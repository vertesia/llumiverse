import { Ajv } from 'ajv';
import addFormats from 'ajv-formats';
import { extractAndParseJSON } from "./json.js";
import { resolveField } from './resolver.js';
import { ResultValidationError } from "@llumiverse/common";


const ajv = new Ajv({
    coerceTypes: 'array',
    allowDate: true,
    strict: false,
    useDefaults: true,
    removeAdditional: "failing"
});

//use ts ignore to avoid error with ESM and ajv-formats
// @ts-ignore This expression is not callable
addFormats(ajv)


export class ValidationError extends Error implements ResultValidationError {
    constructor(
        public code: 'validation_error' | 'json_error',
        message: string
    ) {
        super(message)
        this.name = 'ValidationError'
    }
}

export function validateResult(data: any, schema: Object) {
    let json;

    if (typeof data === "string") {
        try {
            json = extractAndParseJSON(data);
        } catch (error: any) {
            throw new ValidationError("json_error", error.message)
        }
    } else {
        json = data;
    }

    const validate = ajv.compile(schema);
    const valid = validate(json);

    if (!valid && validate.errors) {
        let errors = [];

        for (const e of validate.errors) {
            const path = e.instancePath.split("/").slice(1);
            const value = resolveField(json, path);
            const schemaPath = e.schemaPath.split("/").slice(1);
            const schemaFieldFormat = resolveField(schema, schemaPath);
            const schemaField = resolveField(schema, schemaPath.slice(0, -3));

            //ignore date if empty or null
            if (!value
                && ["date", "date-time"].includes(schemaFieldFormat)
                && !schemaField?.required?.includes(path[path.length - 1])) {
                continue;
            } else {
                errors.push(e);
            }
        }

        //console.log("Errors", errors)
        if (errors.length > 0) {
            const errorsMessage = errors.map(e => `${e.instancePath}: ${e.message}\n${JSON.stringify(e.params)}`).join(",\n\n");
            throw new ValidationError("validation_error", errorsMessage)
        }
    }

    return json;
}
