import { JSONValue } from "@llumiverse/common";
import { jsonrepair } from 'jsonrepair';

function extractJsonFromText(text: string): string {
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    text = text.substring(start, end + 1);
    return text;
}

export function extractAndParseJSON(text: string): JSONValue {
    return parseJSON(extractJsonFromText(text));
}

export function parseJSON(text: string): JSONValue {
    text = text.trim();
    try {
        return JSON.parse(text);
    } catch (err: any) {
        // use a relaxed parser
        try {
            return JSON.parse(jsonrepair(text));
        } catch (err2: any) { // throw the original error
            throw err;
        }
    }
}
