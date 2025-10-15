import { dirname, join } from 'path';
import { readFileSync } from 'fs';
import { CompletionResult } from '@llumiverse/common';

const dataDir = join(dirname(new URL(import.meta.url).pathname), 'data');
const dataFile = (file: string) => join(dataDir, file);
const readDataFile = (file: string, enc: BufferEncoding = 'utf-8') => {
    return readFileSync(dataFile(file), enc);
}


export { dataDir, dataFile, readDataFile };

/**
 * Output as string
 */
export function completionResultToString(result: CompletionResult): string {
    switch (result.type) {
        case "text":
            return result.value;
        case "json":
            return JSON.stringify(result.value, null, 2);
        case "image":
            return result.value;
    }
}

/**
 * Output as JSON, only handles the first JSON result or tries to parse text as JSON
 * Expects the text to be pure JSON if no JSON result is found
 * Throws if no JSON result is found or if parsing fails
 */
export function parseCompletionResultsToJson(results: CompletionResult[]): any {
    const jsonResults = results.filter(r => r.type === "json");
    if (jsonResults.length >= 1) {
        return jsonResults[0].value;
        //TODO: Handle multiple json type results
    }

    const textResults = results.filter(r => r.type === "text").join();
    if (textResults.length === 0) {
        throw new Error("No JSON result found or failed to parse text");
    }
    try {
        return JSON.parse(textResults);
    }
    catch {
        throw new Error("No JSON result found or failed to parse text");
    }
}

/**
 * Output as JSON if possible, otherwise as concatenated text
 * Joins text results with the specified separator, default is empty string
 * If multiple JSON results are found only the first one is returned
 */
export function parseCompletionResults(result: CompletionResult[], separator: string = ""): any {
    try {
        return parseCompletionResultsToJson(result);
    } catch {
        return result.map(completionResultToString).join(separator);
    }
}

