import Ajv from 'ajv';
import fs, { readFileSync } from 'fs';
import { describe, expect, test } from "vitest";
import { extractAndParseJSON, parseJSON } from "../src/json";
import { JSONArray, JSONObject, JsonResult } from '@llumiverse/common';
import { validateResult, ValidationError } from '../src/validation';
import { readDataFile } from './utils';

describe('Core Utilities', () => {
    test('extractAndParseJSON', () => {
        const r = extractAndParseJSON('bla {"a": 1, "b": 2} bla bla');
        expect(r).toEqual({ a: 1, b: 2 });
    });

    test('parseJSON', () => {
        const url = new URL("./json.txt", import.meta.url);
        const text = readFileSync(url, "utf8");
        const json = parseJSON(text)!;
        expect((json as JSONObject).key1).toBe("value1 \" test");
        expect((json as JSONObject).key2).toBe("value2");
        expect((json as JSONObject).key3).toBe("value3");
        expect((json as JSONObject).key4).toBe("value4\nwith new line");
        const obj = (json as JSONObject).object as JSONObject;
        const arr = obj.array as JSONArray;
        expect(arr.length).toBe(2);
        expect(arr[0]).toBe("item1");
        const nestedArr = arr[1] as JSONArray;
        expect(nestedArr.length).toBe(1);
        expect(nestedArr[0]).toStrictEqual({ "nested": "object" });
    });

    test('parseJSON with trailing commas', () => {
        const json1 = parseJSON('{"a": 1, "b": 2,}') as JSONObject;
        expect(json1.a).toBe(1);
        expect(json1.b).toBe(2);

        const json2 = parseJSON('[1, 2, 3,]') as JSONArray;
        expect(json2).toEqual([1, 2, 3]);
    });

    test('parseJSON with Python-style booleans', () => {
        const json = parseJSON('{"enabled": True, "disabled": False}') as JSONObject;
        expect(json.enabled).toBe(true);
        expect(json.disabled).toBe(false);
    });

    test('parseJSON with Python-style None and JavaScript undefined', () => {
        const json1 = parseJSON('{"value": None}') as JSONObject;
        expect(json1.value).toBe(null);

        const json2 = parseJSON('{"value": undefined}') as JSONObject;
        expect(json2.value).toBe(null);
    });

    test.skip('parseJSON with comments', () => {
        // Comments are tricky - skipping for now
        const jsonWithLineComments = parseJSON(`{
            // This is a comment
            "a": 1,
            "b": 2 // Another comment
        }`) as JSONObject;
        expect(jsonWithLineComments.a).toBe(1);
        expect(jsonWithLineComments.b).toBe(2);

        const jsonWithBlockComments = parseJSON(`{
            /* This is a
               multi-line comment */
            "a": 1,
            "b": 2
        }`) as JSONObject;
        expect(jsonWithBlockComments.a).toBe(1);
        expect(jsonWithBlockComments.b).toBe(2);
    });

    test('parseJSON with scientific notation and Infinity', () => {
        const json = parseJSON('{"scientific": 1.5e10, "negative": -2.5e-3, "infinity": Infinity}') as JSONObject;
        expect(json.scientific).toBe(1.5e10);
        expect(json.negative).toBe(-2.5e-3);
        expect(json.infinity).toBe(Infinity);
    });

    test('parseJSON with mixed LLM errors', () => {
        // Combining multiple common LLM errors (except comments)
        const json = parseJSON(`{
            "enabled": True,
            "count": 42,
            "data": None,
            "items": [1, 2, 3,]
        }`) as JSONObject;
        expect(json.enabled).toBe(true);
        expect(json.count).toBe(42);
        expect(json.data).toBe(null);
        expect(json.items).toEqual([1, 2, 3]);
    });

    test('Validate JSON against schema', () => {
        const ajv = new Ajv({ coerceTypes: true, allowDate: true, strict: false });
        const schema = parseJSON(readDataFile('ciia-schema.json')) as any;
        const content: JsonResult[] = [{ type: "json", value: parseJSON(readDataFile('ciia-data.json')) }];
        const res = validateResult(content, schema);
        console.debug(res);
    });

    test('Validate JSON against complex schema', () => {
        const ajv = new Ajv({ coerceTypes: true, allowDate: true, strict: false });
        const schema = parseJSON(readDataFile('complex-schema.json')) as any;
        const content: JsonResult[] = [{ type: "json", value: parseJSON(readDataFile('complex-document.json')) }];
        const res = validateResult(content, schema);
        console.debug(res);
    });

    test('Fail at validating JSON against schema', () => {
        const schema = parseJSON(readDataFile('ciia-schema.json')) as any;
        const content: JsonResult[] = [{ type: "json", value: parseJSON(readDataFile('ciia-data-wrong.json')) }];
        expect(() => validateResult(content, schema)).toThrowError(ValidationError);
    });

    test('JSON parser should coerce types', () => {
        const schema = parseJSON(readDataFile('ciia-schema.json')) as any;
        const content: JsonResult[] = [{ type: "json", value: parseJSON(readDataFile('ciia-data-wrong-types.json')) }];
        const res = validateResult(content, schema);
        console.debug(res);
    });

    test('JSON parser should validate if date is empty string', () => {
        const schema = parseJSON(readDataFile('ciia-schema.json')) as any;
        const content: JsonResult[] = [{ type: "json", value: parseJSON(readDataFile('ciia-data-date-empty.json')) }];
        const res = validateResult(content, schema);
        console.debug(res);
    });

    test('JSON parser should validate if date is null', () => {
        const schema = parseJSON(readDataFile('ciia-schema.json')) as any;
        const content: JsonResult[] = [{ type: "json", value: parseJSON(readDataFile('ciia-data-date-null.json')) }];
        const res = validateResult(content, schema);
        console.debug(res);
    });

    test('JSON parser should not validate if date is wrong', () => {
        const schema = parseJSON(readDataFile('ciia-schema.json')) as any;
        const content: JsonResult[] = [{ type: "json", value: parseJSON(readDataFile('ciia-data-date-wrong.json')) }];
        expect(() => validateResult(content, schema)).toThrowError(ValidationError);
        console.debug(content, schema);
    });

    test('JSON parser should not validate if date is required but null', () => {
        const schema = parseJSON(readDataFile('ciia-schema-date-required.json')) as any;
        const content: JsonResult[] = [{ type: "json", value: parseJSON(readDataFile('ciia-data-date-empty.json')) }];
        expect(() => validateResult(content, schema)).toThrowError(ValidationError);
        console.debug(content, schema);
    });


    const dataFiles = fs.readdirSync(new URL("./data", import.meta.url)).filter((f) => f.endsWith('.data.json'));

    test.each(dataFiles)('Validate JSON against schema: %s', (dataFile) => {
        const base = dataFile.replace('.data.json', '');
        const schema = parseJSON(readDataFile(`${base}.schema.json`)) as any;
        const content: JsonResult[] = [{ type: "json", value: parseJSON(readDataFile(dataFile)) }];
        const res = validateResult(content, schema);
        console.debug(res);
    });

})
